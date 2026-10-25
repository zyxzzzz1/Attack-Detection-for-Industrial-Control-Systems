"""Threshold stage: threshold = the ml.threshold.quantile quantile of the val_thr scores (constraint 3, 99% by default). Reads only scores_val_thr.npy.

Also records the distribution of the val_thr scores (mean, variance etc., used in M2c to check whether Deep SVDD collapsed) and other quantiles
(sensitivity_quantiles, only for sensitivity analysis, never for choosing the threshold).

Usage:
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
        raise ValueError("val_thr has no scored seconds, cannot compute a threshold")
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
    # Collapse check: nearly identical scores for all windows (variance close to 0) mean the model maps every input to the same point
    result["collapsed"] = bool(result["score_cv"] < ctx.cfg["ml"]["collapse"]["min_score_cv"])
    if result["collapsed"]:
        ctx.log.warning("val_thr scores barely vary (variance %.3g, coefficient of variation %.3g): the model may have collapsed, must be reported",
                        result["score_stats"]["var"], result["score_cv"])
    (ctx.directory / THRESHOLD_FILE).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    ctx.log.info("threshold %.6g (val_thr quantile %s, %d s used, %d s unscored)", result["threshold"], cfg["quantile"],
                 result["n_scored"], result["n_unscored"])
    return {k: result[k] for k in ("threshold", "quantile", "n_scored", "n_unscored", "score_cv", "collapsed")}


if __name__ == "__main__":
    sys.exit(stage_cli("threshold", "Compute the threshold as a quantile of the val_thr scores"))
