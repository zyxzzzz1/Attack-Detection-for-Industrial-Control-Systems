"""滑窗 Dataset：窗口按需切片，不预先展开；窗口只在段内滑动，不跨越任何段边界（约束 8）。

段边界 = 文件边界、train/val 切分点、val 内部切分点（由 load_hai.split_segments 产生）。
每个窗口的分数赋给窗口的最后一秒；每段的前 W−1 秒没有分数，记为 NaN（scatter）。
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, segments: list[np.ndarray], window: int, stride: int) -> None:
        if window < 1 or stride < 1:
            raise ValueError("window 与 stride 必须为正整数")
        self.segments = [np.ascontiguousarray(s, dtype=np.float32) for s in segments]
        self.window, self.stride = window, stride
        # 每段的窗口数；长度不足一个窗口的段没有窗口
        self.counts = np.array([max(0, (len(s) - window) // stride + 1) for s in self.segments], dtype=np.int64)
        self.offsets = np.concatenate(([0], np.cumsum(self.counts)))
        # 每段的滑窗视图（不复制数据），形状 (n−W+1, D, W)
        self._views = [np.lib.stride_tricks.sliding_window_view(s, window, axis=0) if len(s) >= window else None
                       for s in self.segments]

    @property
    def n_features(self) -> int:
        return self.segments[0].shape[1]

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def locate(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """窗口编号 → （所在段编号, 窗口在段内的起始行）。"""
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError("窗口编号越界")
        seg = np.searchsorted(self.offsets, idx, side="right") - 1
        return seg, (idx - self.offsets[seg]) * self.stride

    def __getitem__(self, i: int) -> torch.Tensor:
        seg, start = self.locate(np.array([i]))
        return torch.from_numpy(self.segments[seg[0]][start[0]:start[0] + self.window])

    def get_batch(self, idx: np.ndarray) -> torch.Tensor:
        """一次取一批窗口，形状 (B, W, D)。只复制这一批。"""
        seg, start = self.locate(idx)
        out = np.empty((len(seg), self.window, self.n_features), dtype=np.float32)
        for s in np.unique(seg):
            m = seg == s
            out[m] = self._views[s][start[m]].transpose(0, 2, 1)
        return torch.from_numpy(out)

    def iter_batches(self, batch_size: int, shuffle: bool = False, generator: np.random.Generator | None = None):
        """按批产生 (窗口编号, 批张量)。shuffle 只打乱窗口的顺序，窗口内部的时间顺序不变。"""
        order = np.arange(len(self))
        if shuffle:
            (generator or np.random.default_rng()).shuffle(order)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            yield idx, self.get_batch(idx)

    def scatter(self, values: np.ndarray) -> list[np.ndarray]:
        """把逐窗口的值放回逐秒数组：赋给窗口最后一秒，其余位置（含每段前 W−1 秒）为 NaN。

        values 形状 (窗口数,) 或 (窗口数, D)；返回每段一个数组，长度等于该段的秒数。
        """
        values = np.asarray(values)
        if len(values) != len(self):
            raise ValueError(f"values 长度 {len(values)} 与窗口数 {len(self)} 不一致")
        seg, start = self.locate(np.arange(len(self)))
        out = []
        for s, x in enumerate(self.segments):
            arr = np.full((len(x), *values.shape[1:]), np.nan, dtype=np.float32)
            m = seg == s
            arr[start[m] + self.window - 1] = values[m]
            out.append(arr)
        return out
