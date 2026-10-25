"""Shared computation of alarm segments and their sensor attribution (label-free). Shared by attribution.py and scripts/export_alerts.py,
so the attribution evaluation and the exported alerts use exactly the same definition of "an alarm segment" and "Top-k sensors" (main-line decisions 6, 7).

Alarm segments come from to_segments and never cross any part in test_index (file boundaries, or clip boundaries in the smoke data).
"""

import numpy as np
import pandas as pd

from src.eval.metrics import alarms_from_scores
from src.eval.segments import to_segments


def alarm_segments(scores: np.ndarray, index: dict, threshold: float, merge_gap_s: int = 0) -> list[dict]:
    """All alarm segments in time order. start / end are row numbers within the part (closed interval); global_* are indices into the concatenated array."""
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
    """Mean error of each sensor within the alarm segment [lo, hi] (indices into the concatenated array, closed interval), normalized to shares (summing to 1). Unscored seconds are ignored.

    baseline (optional, only for the post-hoc exploratory analysis): each sensor's mean error on val_thr. When given, the in-segment means are first divided by it per sensor
    and then normalized, i.e. what is compared is the "multiple of the usual error"; sensors with a non-positive baseline are ignored (share 0).
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
    """The k sensors with the largest shares: [{sensor, share}, ...]; shares are relative to all sensors (not renormalized within the Top-k)."""
    if np.isnan(shares).all():
        return []
    order = np.argsort(-shares, kind="stable")[:k]
    return [{"sensor": features[i], "share": float(shares[i])} for i in order]


def subsystem_of(sensor: str) -> str:
    """The prefix of a sensor name (P1 ~ P4) is its subsystem."""
    return sensor.split("_", 1)[0]


def subsystem_shares(shares: np.ndarray, features: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for f, s in zip(features, shares):
        out[subsystem_of(f)] = out.get(subsystem_of(f), 0.0) + float(s)
    return out
