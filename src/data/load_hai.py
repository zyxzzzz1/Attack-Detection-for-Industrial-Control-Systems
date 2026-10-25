"""Read HAI CSVs: keep only sensor columns, organize them per file into contiguous Segments, and split into train | val_es | val_thr.

- The time column is only used to verify per-second continuity and to record each segment's start time, never as a feature; label columns are not in usecols and are never read.
- A Segment is a block of data contiguous per second. File boundaries, the train/val split point and the split point inside val are all segment boundaries; sliding windows stay within a segment.
- Constraint 1: validation set = the last val_fraction after concatenating in time order, not shuffled.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import hai_data_dir

SPLITS = ("train", "val_es", "val_thr")


@dataclass
class Segment:
    name: str          # Source file name
    split: str         # train / val_es / val_thr / test
    start_row: int     # Row number of the first row in the source file (0-based)
    start_time: str    # Time of the first row (HAI raw time string)
    columns: list[str]
    x: np.ndarray      # (n, D) float32

    def __len__(self) -> int:
        return len(self.x)

    def slice(self, lo: int, hi: int, split: str | None = None) -> "Segment":
        t = pd.Timestamp(self.start_time) + pd.Timedelta(seconds=lo)
        return replace(self, split=split or self.split, start_row=self.start_row + lo,
                       start_time=str(t), x=self.x[lo:hi])


def read_segment(path: Path, time_col: str, exclude_cols: list[str], split: str) -> Segment:
    """Read the sensor columns of one file (float32). Columns in exclude_cols are not read. The file must be contiguous per second."""
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
    """For smoke tests: keep only the first fraction of a segment (still contiguous)."""
    return seg if fraction >= 1 else seg.slice(0, max(1, int(len(seg) * fraction)))


def load_train_segments(cfg: dict, fraction: float = 1.0) -> list[Segment]:
    """Read the train files in config order, one segment per file; when the switch is on, first drop the rows identical to test content (at the file head/tail)."""
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
    """Read the sensor columns of the test files, one segment per file. Must only be called by the scoring / final evaluation pipeline (constraint 2).

    In smoke tests (fraction < 1) each file contributes its first fraction, plus the clips described by row numbers in ml.smoke.extra_test_clips
    (each becomes its own segment; the first W−1 seconds of each are likewise unscored). Clip positions are in the config; no labels are read here.
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
    """Split the time-ordered segments into train | val_es | val_thr. A split point inside a segment cuts that segment, making the split point a new segment boundary."""
    total = sum(len(s) for s in segments)
    n_val = int(round(total * val_fraction))
    n_val_es = int(round(n_val * val_es_fraction))
    cuts = [0, total - n_val, total - n_val + n_val_es, total]  # Split points in global row numbers
    out = {k: [] for k in SPLITS}
    offset = 0
    for seg in segments:
        for split, lo, hi in zip(SPLITS, cuts[:-1], cuts[1:]):
            a, b = max(lo, offset) - offset, min(hi, offset + len(seg)) - offset
            if b > a:
                out[split].append(seg.slice(a, b, split))
        offset += len(seg)
    return out
