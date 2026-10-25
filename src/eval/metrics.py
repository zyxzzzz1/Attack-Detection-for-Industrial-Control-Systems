"""Evaluation metrics: point-wise and event-level. Shared by all models, with uniform input (scores, labels, threshold) (constraint 6).

- No point-adjust (constraint 4): point-wise metrics use each second's own prediction.
- Seconds with NaN scores (the first W−1 seconds of each test file) are excluded from every metric; the number excluded is reported (n_unscored).
- Alarm rule: score > threshold.
- This module reads no files; labels are passed in by the caller (evaluate.py).
"""

import math

import numpy as np
from sklearn.metrics import roc_auc_score

from src.eval.segments import overlaps, to_segments

NAN = float("nan")


def alarms_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Per-second alarms (bool). Unscored seconds (NaN) never alarm."""
    scores = np.asarray(scores, dtype=np.float64)
    return np.greater(scores, threshold, where=~np.isnan(scores), out=np.zeros(scores.shape, dtype=bool))


def point_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Point-wise Precision / Recall / F1 / AUROC.

    precision is NaN when there are no alarms; recall, F1 and AUROC are NaN when there are no positives (undefined, not recorded as 0).
    """
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} differ in length")
    scored = ~np.isnan(scores)
    s, y = scores[scored], labels[scored]
    pred = s > threshold
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    precision = tp / (tp + fp) if tp + fp else NAN
    recall = tp / (tp + fn) if tp + fn else NAN
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn else NAN  # With positives but no alarms, F1 = 0 and precision is NaN
    auroc = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else NAN
    return {"precision": precision, "recall": recall, "f1": f1, "auroc": auroc,
            "tp": tp, "fp": fp, "fn": fn, "tn": int((~pred & ~y).sum()),
            "n_points": int(scored.sum()), "n_positive": int(y.sum()), "n_unscored": int((~scored).sum()),
            "n_positive_unscored": int(labels[~scored].sum())}


def event_metrics(files: list[tuple[np.ndarray, np.ndarray]], threshold: float, merge_gap_s: int = 0):
    """Event-level metrics. files holds (scores, labels) of each test file; neither events nor alarm segments cross files.

    - Event = contiguous interval where the label is 1; detected = at least one alarm second within the interval.
    - Detection delay = time of the first alarm within the event − event start, only over detected events.
    - False-alarm rate = alarm segments overlapping no event ÷ normal duration (hours); normal duration counts only scored seconds with label 0.

    Returns (summary dict, per-event list, per-alarm-segment list); start / end in the details are row numbers within the file (closed interval).
    """
    events, alarms, normal_s = [], [], 0
    for k, (scores, labels) in enumerate(files):
        scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
        if scores.shape != labels.shape:
            raise ValueError(f"file {k}: scores {scores.shape} and labels {labels.shape} differ in length")
        alarm = alarms_from_scores(scores, threshold)
        normal_s += int((~labels & ~np.isnan(scores)).sum())
        file_events = to_segments(labels, 0)
        for start, end in file_events:
            hit = np.flatnonzero(alarm[start:end + 1])
            events.append({"file_index": k, "start": start, "end": end, "duration_s": end - start + 1,
                           "detected": bool(len(hit)), "delay_s": int(hit[0]) if len(hit) else None,
                           "n_unscored_s": int(np.isnan(scores[start:end + 1]).sum())})
        for start, end in to_segments(alarm, merge_gap_s):
            alarms.append({"file_index": k, "start": start, "end": end, "duration_s": end - start + 1,
                           "overlaps_event": any(overlaps((start, end), e) for e in file_events)})
    delays = [e["delay_s"] for e in events if e["detected"]]
    n_false = sum(not a["overlaps_event"] for a in alarms)
    hours = normal_s / 3600
    summary = {
        "n_events": len(events), "n_detected": len(delays), "n_missed": len(events) - len(delays),
        "event_recall": len(delays) / len(events) if events else NAN,
        "delay_median_s": float(np.median(delays)) if delays else NAN,
        "delay_mean_s": float(np.mean(delays)) if delays else NAN,
        "n_alarm_segments": len(alarms), "n_false_alarm_segments": n_false,
        "normal_hours": hours, "false_alarms_per_hour": n_false / hours if hours else NAN,
        "merge_gap_s": merge_gap_s,
    }
    return summary, events, alarms


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson confidence interval of the binomial proportion k / n (95% by default). Returns (NaN, NaN) when n = 0."""
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def precision_at_k(relevant_in_rank_order, k: int) -> float:
    """Fraction of true items among the top k after sorting; with fewer than k items the actual count is used, NaN when there are none."""
    top = list(relevant_in_rank_order)[:k]
    return sum(bool(x) for x in top) / len(top) if top else float("nan")
