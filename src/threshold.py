"""阈值阶段：阈值 = val_thr 分数的 ml.threshold.quantile 分位数（约束 3，默认 99%）。只读取 scores_val_thr.npy。

同时记录 val_thr 分数的分布（均值、方差等，M2c 用来检查 Deep SVDD 是否坍塌）和其他分位数
（sensitivity_quantiles，只用于敏感性分析，不据此选阈值）。

用法：
    python -m src.threshold --model pca [--smoke]
"""

import json
import sys

import numpy as np

from src.utils.context import stage_cli

THRESHOLD_FILE = "threshold.json"


def compute_threshold(val_scores: np.ndarray, quantile: float, sensitivity_quantiles: list[float]) -> dict:
    scores = np.asarray(val_scores, dtype=np.float64)
    valid = scores[~np.isnan(scores)]
    if len(valid) == 0:
        raise ValueError("val_thr 没有任何已打分的秒，无法计算阈值")
    mean, std = float(valid.mean()), float(valid.std())
    return {
        "score_cv": std / abs(mean) if mean != 0 else (0.0 if std == 0 else float("inf")),
        "threshold": float(np.quantile(valid, quantile)), "quantile": quantile, "source": "val_thr",
        "n_scored": int(len(valid)), "n_unscored": int(len(scores) - len(valid)),
        "score_stats": {"mean": float(valid.mean()), "std": float(valid.std()), "var": float(valid.var()),
                        "min": float(valid.min()), "median": float(np.median(valid)), "max": float(valid.max())},
        "sensitivity": {str(q): float(np.quantile(valid, q)) for q in sensitivity_quantiles},
    }


def run(ctx) -> dict:
    cfg = ctx.cfg["ml"]["threshold"]
    result = compute_threshold(np.load(ctx.directory / "scores_val_thr.npy"), cfg["quantile"],
                               cfg["sensitivity_quantiles"])
    # 坍塌检查：所有窗口的分数几乎相同（方差接近 0）说明模型把所有输入映射到了同一个点
    result["collapsed"] = bool(result["score_cv"] < ctx.cfg["ml"]["collapse"]["min_score_cv"])
    if result["collapsed"]:
        ctx.log.warning("val_thr 分数几乎没有变化（方差 %.3g，变异系数 %.3g）：模型可能已坍塌，需要在汇报中指出",
                        result["score_stats"]["var"], result["score_cv"])
    (ctx.directory / THRESHOLD_FILE).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    ctx.log.info("阈值 %.6g（val_thr %s 分位，%d 秒参与，%d 秒未打分）", result["threshold"], cfg["quantile"],
                 result["n_scored"], result["n_unscored"])
    return {k: result[k] for k in ("threshold", "quantile", "n_scored", "n_unscored", "score_cv", "collapsed")}


if __name__ == "__main__":
    sys.exit(stage_cli("threshold", "按 val_thr 分数的分位数计算阈值"))
