"""把逐秒的 0/1 序列合并成区间。报警段与事件共用这一个函数（主线补充决策 6）。

evaluate.py 的事件级统计和 scripts/export_alerts.py 都必须调用 to_segments，保证两处对"一段报警"的定义相同。
"""

import numpy as np


def to_segments(binary, merge_gap_s: int = 0) -> list[tuple[int, int]]:
    """返回 [(start, end), ...]，下标为闭区间（start、end 都是取值为 1 的秒），按时间排序。

    merge_gap_s = 0（默认）只合并连续的 1；大于 0 时，两段之间为 0 的秒数不超过 merge_gap_s 也合并成一段。
    输入必须是逐秒连续的一个文件（或一段），不要把多个文件拼起来再传入。
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
    """两个闭区间是否有公共的秒。"""
    return a[0] <= b[1] and b[0] <= a[1]
