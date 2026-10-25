"""Scoring stage: output per-second anomaly scores (each window's score is assigned to its last second; the first W−1 seconds of each segment are NaN).

Artifacts (in the run directory):
    scores_val_thr.npy        (T_val,)   per-second val_thr scores, used only to compute the threshold
    scores_test.npy           (T,)       all test files concatenated in config order; the first W−1 seconds of each file are NaN
    per_sensor_err_test.npy   (T, D)     reconstruction models only
    test_index.json           position of each test file in the concatenated array, feature names, window length

Only the sensor columns of the test files are read here; labels are read only in the evaluation stage.

Usage:
    python -m src.score --model pca [--smoke]
"""

import json
import sys

import numpy as np

from src.data.preprocess import prepare_test
from src.data.windowing import WindowDataset
from src.utils.context import stage_cli

TEST_INDEX = "test_index.json"


def per_second(ds: WindowDataset, values: np.ndarray | None) -> np.ndarray | None:
    """Per-window values -> per-second array over all segments concatenated (NaN where unscored)."""
    return None if values is None else np.concatenate(ds.scatter(values))


def score_val_thr(ctx) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-second scores and per-sensor errors of val_thr. Shared by the diagnose and score stages, computed only once per run."""
    if "val_thr" not in ctx.state:
        ds = ctx.dataset("val_thr")
        scores, per_sensor = ctx.ensure_model().score(ds)
        ctx.state["val_thr"] = (per_second(ds, scores), per_second(ds, per_sensor))
    return ctx.state["val_thr"]


def run(ctx) -> dict:
    model = ctx.ensure_model()
    data = ctx.ensure_data()
    window = ctx.cfg["ml"]["window"]

    val_scores, _ = score_val_thr(ctx)
    np.save(ctx.directory / "scores_val_thr.npy", val_scores)

    segments = prepare_test(ctx.cfg, data.pre, ctx.fraction)
    ds = WindowDataset([s.x for s in segments], window["length"], window["eval_stride"])
    scores, per_sensor = model.score(ds)
    test_scores, test_err = per_second(ds, scores), per_second(ds, per_sensor)
    np.save(ctx.directory / "scores_test.npy", test_scores)
    if test_err is not None:
        np.save(ctx.directory / "per_sensor_err_test.npy", test_err)

    files, offset = [], 0
    for s in segments:
        files.append({"file": s.name, "offset": offset, "rows": len(s), "start_row": s.start_row,
                      "start_time": s.start_time})
        offset += len(s)
    index = {"files": files, "features": data.pre.features, "window": window["length"],
             "has_per_sensor_err": test_err is not None}
    (ctx.directory / TEST_INDEX).write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = {"val_thr_seconds": len(val_scores), "val_thr_unscored": int(np.isnan(val_scores).sum()),
              "test_seconds": len(test_scores), "test_unscored": int(np.isnan(test_scores).sum()),
              "has_per_sensor_err": test_err is not None}
    ctx.log.info("打分完成：%s", result)
    return result


if __name__ == "__main__":
    sys.exit(stage_cli("score", "用训练好的模型给 val_thr 和 test 打分"))
