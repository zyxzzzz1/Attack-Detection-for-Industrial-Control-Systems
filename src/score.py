"""打分阶段：输出逐秒异常分数（窗口分数赋给窗口最后一秒，每段前 W−1 秒为 NaN）。

产物（运行目录下）：
    scores_val_thr.npy        (T_val,)   val_thr 的逐秒分数，只用于计算阈值
    scores_test.npy           (T,)       全部 test 文件按配置顺序拼接；每个文件的前 W−1 秒为 NaN
    per_sensor_err_test.npy   (T, D)     仅重构类模型
    test_index.json           各 test 文件在拼接数组中的位置、特征名、窗口长度

这里只读取 test 文件的传感器列；标签只在评估阶段读取。

用法：
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
    """逐窗口的值 → 各段拼接后的逐秒数组（未打分处为 NaN）。"""
    return None if values is None else np.concatenate(ds.scatter(values))


def score_val_thr(ctx) -> tuple[np.ndarray, np.ndarray | None]:
    """val_thr 的逐秒分数与逐传感器误差。诊断阶段和打分阶段共用，同一次运行内只算一次。"""
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
