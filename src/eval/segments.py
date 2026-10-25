"""Merge a per-second 0/1 sequence into intervals. Alarm segments and events share this one function (main-line decision 6).

Both the event-level statistics of evaluate.py and scripts/export_alerts.py must call to_segments, so both use the same definition of "an alarm segment".
"""

import numpy as np


def to_segments(binary, merge_gap_s: int = 0) -> list[tuple[int, int]]:
    """Return [(start, end), ...] with closed-interval indices (start and end are both seconds with value 1), in time order.

    merge_gap_s = 0 (default) merges only consecutive 1s; when greater than 0, two runs separated by at most merge_gap_s zeros are also merged into one.
    The input must be a single file (or part) contiguous per second; do not concatenate several files before passing them in.
    """
    if merge_gap_s < 0:
        raise ValueError("merge_gap_s 不能为负")
    b = np.asarray(binary).astype(bool)
    if b.ndim != 1:
        raise ValueError("binary 必须是一维序列")
    if not b.any():
        return []
    edges = np.diff(np.concatenate(([0], b.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    merged = [(int(starts[0]), int(ends[0]))]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged[-1][1] - 1 <= merge_gap_s:
            merged[-1] = (merged[-1][0], int(e))
        else:
            merged.append((int(s), int(e)))
    return merged


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Whether two closed intervals share a second."""
    return a[0] <= b[1] and b[0] <= a[1]
