"""评估指标：逐点指标与事件级指标。所有模型共用，输入统一为 (scores, labels, threshold)（约束 6）。

- 不使用 point-adjust（约束 4）：逐点指标按每一秒各自的预测计算。
- 分数为 NaN 的秒（每个测试文件的前 W−1 秒）不参与任何指标，被排除的秒数在结果中报告（n_unscored）。
- 报警规则：分数 > 阈值。
- 本模块不读取任何文件，标签由调用方（evaluate.py）传入。
"""

import math

import numpy as np
from sklearn.metrics import roc_auc_score

from src.eval.segments import overlaps, to_segments

NAN = float("nan")


def alarms_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    """逐秒报警（bool）。未打分的秒（NaN）不报警。"""
    scores = np.asarray(scores, dtype=np.float64)
    return np.greater(scores, threshold, where=~np.isnan(scores), out=np.zeros(scores.shape, dtype=bool))


def point_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """逐点 Precision / Recall / F1 / AUROC。

    没有报警时 precision 为 NaN；没有正样本时 recall、F1、AUROC 为 NaN（无法定义，不记为 0）。
    """
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} 与 labels {labels.shape} 长度不一致")
    scored = ~np.isnan(scores)
    s, y = scores[scored], labels[scored]
    pred = s > threshold
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    precision = tp / (tp + fp) if tp + fp else NAN
    recall = tp / (tp + fn) if tp + fn else NAN
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn else NAN  # 有正样本但没有报警时 F1 = 0，precision 为 NaN
    auroc = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else NAN
    return {"precision": precision, "recall": recall, "f1": f1, "auroc": auroc,
            "tp": tp, "fp": fp, "fn": fn, "tn": int((~pred & ~y).sum()),
            "n_points": int(scored.sum()), "n_positive": int(y.sum()), "n_unscored": int((~scored).sum()),
            "n_positive_unscored": int(labels[~scored].sum())}


def event_metrics(files: list[tuple[np.ndarray, np.ndarray]], threshold: float, merge_gap_s: int = 0):
    """事件级指标。files 为每个测试文件的 (scores, labels)；事件与报警段都不跨文件。

    - 事件 = 标签为 1 的连续区间；检出 = 区间内至少有一秒报警。
    - 检测延迟 = 事件内首次报警的时间 − 事件开始时间，只统计检出的事件。
    - 误报率 = 不与任何事件重叠的报警段数 ÷ 正常时长（小时）；正常时长只计已打分且标签为 0 的秒。

    返回 (汇总 dict, 事件明细 list, 报警段明细 list)，明细中的 start / end 是文件内的行号（闭区间）。
    """
    events, alarms, normal_s = [], [], 0
    for k, (scores, labels) in enumerate(files):
        scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
        if scores.shape != labels.shape:
            raise ValueError(f"第 {k} 个文件：scores {scores.shape} 与 labels {labels.shape} 长度不一致")
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
    """二项比例 k / n 的 Wilson 置信区间（默认 95%）。n = 0 时返回 (NaN, NaN)。"""
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def precision_at_k(relevant_in_rank_order, k: int) -> float:
    """排序后前 k 条中为真的比例；不足 k 条时按实际条数计算，没有任何条目时为 NaN。"""
    top = list(relevant_in_rank_order)[:k]
    return sum(bool(x) for x in top) / len(top) if top else float("nan")
