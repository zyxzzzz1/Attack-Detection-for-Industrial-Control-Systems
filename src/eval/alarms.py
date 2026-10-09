"""报警段及其传感器归因的公共计算（不涉及标签）。attribution.py 和 scripts/export_alerts.py 共用，
保证归因评估与导出的报警对"一段报警"和"Top-k 传感器"的定义完全相同（主线补充决策 6、7）。

报警段由 to_segments 得到，不跨 test_index 中的任何一段（文件边界，或冒烟数据里的片段边界）。
"""

import numpy as np
import pandas as pd

from src.eval.metrics import alarms_from_scores
from src.eval.segments import to_segments


def alarm_segments(scores: np.ndarray, index: dict, threshold: float, merge_gap_s: int = 0) -> list[dict]:
    """全部报警段，按时间顺序。start / end 是段内行号（闭区间），global_* 是拼接数组中的下标。"""
    out = []
    for k, part in enumerate(index["files"]):
        s = scores[part["offset"]:part["offset"] + part["rows"]]
        t0 = pd.Timestamp(part["start_time"])
        for a, b in to_segments(alarms_from_scores(s, threshold), merge_gap_s):
            out.append({"part": k, "file": part["file"], "start": a, "end": b,
                        "global_start": part["offset"] + a, "global_end": part["offset"] + b,
                        "start_time": str(t0 + pd.Timedelta(seconds=a)), "end_time": str(t0 + pd.Timedelta(seconds=b)),
                        "duration_s": b - a + 1, "max_score": float(np.nanmax(s[a:b + 1]))})
    return out


def sensor_shares(per_sensor_err: np.ndarray, lo: int, hi: int, baseline: np.ndarray | None = None) -> np.ndarray:
    """报警段 [lo, hi]（拼接数组下标，闭区间）内各传感器误差的均值，归一化为占比（和为 1）。未打分的秒不参与。

    baseline（可选，只用于事后的探索性分析）：各传感器在 val_thr 上的平均误差。给出时先把段内均值逐传感器除以它，
    再归一化为占比，即比较的是"相对平时误差的倍数"；基线不为正的传感器不参与（占比记 0）。
    """
    block = np.asarray(per_sensor_err[lo:hi + 1], dtype=np.float64)
    block = block[~np.isnan(block).all(axis=1)]
    if len(block) == 0:
        return np.full(per_sensor_err.shape[1], np.nan)
    return normalize_shares(block.mean(axis=0), baseline)


def normalize_shares(mean: np.ndarray, baseline: np.ndarray | None = None) -> np.ndarray:
    if baseline is not None:
        ok = np.asarray(baseline) > 0
        mean = np.where(ok, mean / np.where(ok, baseline, 1.0), 0.0)
    total = mean.sum()
    return mean / total if total > 0 else np.full_like(mean, np.nan)


def top_k(shares: np.ndarray, features: list[str], k: int) -> list[dict]:
    """占比最大的 k 个传感器：[{sensor, share}, ...]，占比是相对全部传感器的（不在 Top-k 内重新归一化）。"""
    if np.isnan(shares).all():
        return []
    order = np.argsort(-shares, kind="stable")[:k]
    return [{"sensor": features[i], "share": float(shares[i])} for i in order]


def subsystem_of(sensor: str) -> str:
    """传感器名的前缀（P1 ~ P4）就是它所属的子系统。"""
    return sensor.split("_", 1)[0]


def subsystem_shares(shares: np.ndarray, features: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for f, s in zip(features, shares):
        out[subsystem_of(f)] = out.get(subsystem_of(f), 0.0) + float(s)
    return out
