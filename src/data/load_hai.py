"""读取 HAI CSV：只取传感器列，按文件顺序组织成连续段（Segment），并切分 train | val_es | val_thr。

- 时间列只用于核对逐秒连续并记录每段的起始时间，不作为特征；标签列不在 usecols 中，从不读入。
- 段（Segment）是一块逐秒连续的数据。文件边界、train/val 切分点、val 内部切分点都是段边界，滑窗只在段内进行。
- 约束 1：验证集 = 按时间拼接后的最后 val_fraction，不打乱。
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import hai_data_dir

SPLITS = ("train", "val_es", "val_thr")


@dataclass
class Segment:
    name: str          # 来源文件名
    split: str         # train / val_es / val_thr / test
    start_row: int     # 第一行在来源文件中的行号（0 起）
    start_time: str    # 第一行的时间（HAI 原始时间字符串）
    columns: list[str]
    x: np.ndarray      # (n, D) float32

    def __len__(self) -> int:
        return len(self.x)

    def slice(self, lo: int, hi: int, split: str | None = None) -> "Segment":
        t = pd.Timestamp(self.start_time) + pd.Timedelta(seconds=lo)
        return replace(self, split=split or self.split, start_row=self.start_row + lo,
                       start_time=str(t), x=self.x[lo:hi])


def read_segment(path: Path, time_col: str, exclude_cols: list[str], split: str) -> Segment:
    """读取一个文件的传感器列（float32）。exclude_cols 中的列不读入。文件内部必须逐秒连续。"""
    header = [c.strip() for c in pd.read_csv(path, nrows=0).columns]
    sensors = [c for c in header if c != time_col and c not in exclude_cols]
    df = pd.read_csv(path, skipinitialspace=True, header=0, names=header, usecols=[time_col, *sensors],
                     dtype={c: np.float32 for c in sensors})
    t = pd.to_datetime(df[time_col])
    step = t.diff().dt.total_seconds().to_numpy()[1:]
    if not (step == 1).all():
        raise ValueError(f"{path.name} 不是逐秒连续（有 {(step != 1).sum()} 处间隔不等于 1 秒），不能作为一个连续段")
    x = df[sensors].to_numpy(dtype=np.float32)
    return Segment(name=path.name, split=split, start_row=0, start_time=str(t.iloc[0]), columns=sensors, x=x)


def _take_fraction(seg: Segment, fraction: float) -> Segment:
    """冒烟测试用：只保留段的前 fraction（仍然连续）。"""
    return seg if fraction >= 1 else seg.slice(0, max(1, int(len(seg) * fraction)))


def load_train_segments(cfg: dict, fraction: float = 1.0) -> list[Segment]:
    """按配置顺序读取 train 文件，每个文件一段；开关打开时先删除与 test 内容相同的行（文件首尾）。"""
    ml = cfg["ml"]
    hai, data_cfg = ml["hai"], ml["data"]
    drops = {d["file"]: d for d in data_cfg["train_overlap_drops"]} if data_cfg["drop_train_overlap"] else {}
    segments = []
    for name in hai["train_files"]:
        seg = read_segment(hai_data_dir(cfg) / name, hai["time_col"], hai["label_cols"], "train")
        if name in drops:
            head, tail = drops[name].get("head_rows", 0), drops[name].get("tail_rows", 0)
            seg = seg.slice(head, len(seg) - tail)
        segments.append(_take_fraction(seg, fraction))
    starts = [pd.Timestamp(s.start_time) for s in segments]
    if starts != sorted(starts):
        raise ValueError("train 文件未按时间顺序排列，请检查 ml.hai.train_files")
    return segments


def load_test_segments(cfg: dict, fraction: float = 1.0) -> list[Segment]:
    """读取 test 文件的传感器列，每个文件一段。只应由打分 / 最终评估流程调用（约束 2）。

    冒烟测试（fraction < 1）时每个文件取前 fraction，再加上 ml.smoke.extra_test_clips 中按行号描述的片段
    （各自成为一段，段内前 W−1 秒同样不打分）。片段的位置写在 config 里，这里不读取标签。
    """
    hai = cfg["ml"]["hai"]
    clips = cfg["ml"]["smoke"].get("extra_test_clips", []) if fraction < 1 else []
    segments = []
    for name in hai["test_files"]:
        seg = read_segment(hai_data_dir(cfg) / name, hai["time_col"], hai["label_cols"], "test")
        head = _take_fraction(seg, fraction)
        segments.append(head)
        for clip in clips:
            lo, hi = max(clip["start_row"], len(head)), min(clip["start_row"] + clip["rows"], len(seg))
            if clip["file"] == name and hi > lo:
                segments.append(seg.slice(lo, hi))
    return segments


def split_segments(segments: list[Segment], val_fraction: float, val_es_fraction: float) -> dict[str, list[Segment]]:
    """把按时间排好的段切成 train | val_es | val_thr。切分点落在某段内部时把该段切开，切分点成为新的段边界。"""
    total = sum(len(s) for s in segments)
    n_val = int(round(total * val_fraction))
    n_val_es = int(round(n_val * val_es_fraction))
    cuts = [0, total - n_val, total - n_val + n_val_es, total]  # 全局行号上的切分点
    out = {k: [] for k in SPLITS}
    offset = 0
    for seg in segments:
        for split, lo, hi in zip(SPLITS, cuts[:-1], cuts[1:]):
            a, b = max(lo, offset) - offset, min(hi, offset + len(seg)) - offset
            if b > a:
                out[split].append(seg.slice(a, b, split))
        offset += len(seg)
    return out
