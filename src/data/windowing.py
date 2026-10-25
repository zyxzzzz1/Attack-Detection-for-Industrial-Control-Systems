"""Sliding-window Dataset: windows are sliced on demand, not materialized in advance; windows slide only within a segment and never cross a segment boundary (constraint 8).

Segment boundaries = file boundaries, the train/val split point and the split point inside val (produced by load_hai.split_segments).
Each window's score is assigned to its last second; the first W−1 seconds of each segment have no score and are NaN (scatter).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, segments: list[np.ndarray], window: int, stride: int) -> None:
        if window < 1 or stride < 1:
            raise ValueError("window and stride must be positive integers")
        self.segments = [np.ascontiguousarray(s, dtype=np.float32) for s in segments]
        self.window, self.stride = window, stride
        # Number of windows per segment; segments shorter than one window have none
        self.counts = np.array([max(0, (len(s) - window) // stride + 1) for s in self.segments], dtype=np.int64)
        self.offsets = np.concatenate(([0], np.cumsum(self.counts)))
        # Sliding-window view of each segment (no data copied), shape (n−W+1, D, W)
        self._views = [np.lib.stride_tricks.sliding_window_view(s, window, axis=0) if len(s) >= window else None
                       for s in self.segments]

    @property
    def n_features(self) -> int:
        return self.segments[0].shape[1]

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def locate(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Window index -> (segment index, start row of the window within the segment)."""
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError("window index out of range")
        seg = np.searchsorted(self.offsets, idx, side="right") - 1
        return seg, (idx - self.offsets[seg]) * self.stride

    def __getitem__(self, i: int) -> torch.Tensor:
        seg, start = self.locate(np.array([i]))
        return torch.from_numpy(self.segments[seg[0]][start[0]:start[0] + self.window])

    def get_batch(self, idx: np.ndarray) -> torch.Tensor:
        """Fetch one batch of windows, shape (B, W, D). Only this batch is copied."""
        seg, start = self.locate(idx)
        out = np.empty((len(seg), self.window, self.n_features), dtype=np.float32)
        for s in np.unique(seg):
            m = seg == s
            out[m] = self._views[s][start[m]].transpose(0, 2, 1)
        return torch.from_numpy(out)

    def iter_batches(self, batch_size: int, shuffle: bool = False, generator: np.random.Generator | None = None):
        """Yield (window indices, batch tensor) per batch. shuffle only permutes the order of windows; time order within a window is unchanged."""
        order = np.arange(len(self))
        if shuffle:
            (generator or np.random.default_rng()).shuffle(order)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            yield idx, self.get_batch(idx)

    def scatter(self, values: np.ndarray) -> list[np.ndarray]:
        """Put per-window values back into per-second arrays: assigned to each window's last second, NaN elsewhere (including the first W−1 seconds of each segment).

        values has shape (n_windows,) or (n_windows, D); returns one array per segment, as long as the segment in seconds.
        """
        values = np.asarray(values)
        if len(values) != len(self):
            raise ValueError(f"values length {len(values)} does not match the number of windows {len(self)}")
        seg, start = self.locate(np.arange(len(self)))
        out = []
        for s, x in enumerate(self.segments):
            arr = np.full((len(x), *values.shape[1:]), np.nan, dtype=np.float32)
            m = seg == s
            arr[start[m] + self.window - 1] = values[m]
            out.append(arr)
        return out
