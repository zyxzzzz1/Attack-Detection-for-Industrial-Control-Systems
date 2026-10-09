"""重尾特征诊断（只用 val_thr，不读取 test）：各传感器对重构 MSE 的平均贡献占比。

对有逐传感器误差的模型，取 val_thr 上已打分各秒的平方误差均值，归一化为占比，写到运行目录的
val_sensor_contrib.csv。单个传感器的占比超过 ml.diagnosis.max_sensor_share 时抛出 DiagnosisStop：
运行在读取 test 之前停下，由人决定是否调整缩放方式。决定之后用 ml.diagnosis.allow_over_limit=true 放行。

用法：
    python -m src.diagnose --model pca [--smoke]
"""

import sys

import numpy as np
import pandas as pd

from src.score import score_val_thr
from src.utils.context import stage_cli

CONTRIB_FILE = "val_sensor_contrib.csv"


class DiagnosisStop(RuntimeError):
    """诊断结果需要人工决定，运行在读取 test 之前停下。"""


def sensor_contributions(per_sensor_err: np.ndarray, features: list[str]) -> pd.DataFrame:
    """逐秒逐传感器误差 (T, D) → 每个传感器的平均平方误差与占比，按占比从大到小排序。全为 NaN 的行不参与。"""
    scored = ~np.isnan(per_sensor_err).all(axis=1)
    mean_err = per_sensor_err[scored].astype(np.float64).mean(axis=0)
    total = mean_err.sum()
    df = pd.DataFrame({"sensor": features, "subsystem": [f.split("_")[0] for f in features],
                       "mean_sq_err": mean_err, "share": mean_err / total if total > 0 else np.nan})
    return df.sort_values("share", ascending=False, kind="stable").reset_index(drop=True)


def scale_profile(data) -> pd.DataFrame:
    """每个传感器在 train 部分的标准差（scaler 使用的值），以及标准化后 val_thr 中 |z| 的分布。"""
    z = np.abs(np.concatenate([s.x for s in data.segments["val_thr"]]))
    q50, q99, q999 = np.quantile(z, [0.5, 0.99, 0.999], axis=0)
    return pd.DataFrame({"sensor": data.pre.features, "train_std": data.pre.scaler.scale_,
                         "val_thr_absz_p50": q50, "val_thr_absz_p99": q99, "val_thr_absz_p999": q999,
                         "val_thr_absz_max": z.max(axis=0), "val_thr_frac_absz_gt5": (z > 5).mean(axis=0)})


def run(ctx) -> dict:
    limits = ctx.cfg["ml"]["diagnosis"]
    _, per_sensor = score_val_thr(ctx)
    if per_sensor is None:
        ctx.log.info("模型 %s 不输出逐传感器误差，跳过诊断", ctx.model)
        return {"skipped": True}
    data = ctx.ensure_data()
    df = sensor_contributions(per_sensor, data.pre.features).merge(scale_profile(data), on="sensor")
    df.to_csv(ctx.directory / CONTRIB_FILE, index=False)
    top = df.iloc[0]
    gated = ctx.model == limits["model"]  # 是否停下只依据指定的基线（PCA）；其他模型只输出占比表
    over = bool(top["share"] > limits["max_sensor_share"])
    result = {"skipped": False, "top_sensor": top["sensor"], "top_share": float(top["share"]),
              "top5": {r.sensor: round(float(r.share), 4) for r in df.head(5).itertuples()},
              "max_sensor_share": limits["max_sensor_share"], "over_limit": over, "gated": gated}
    ctx.log.info("val_thr 上占比最大的传感器：%s（%.1f%%），前 5：%s", top["sensor"], 100 * top["share"], result["top5"])
    if over and ctx.smoke:
        ctx.log.warning("冒烟测试的数字只用于检查流程，超限不停下")
    elif over and gated and not limits["allow_over_limit"]:
        ctx.manifest.stage("diagnose", **result)
        raise DiagnosisStop(f"{top['sensor']} 占 val_thr 重构 MSE 的 {100 * top['share']:.1f}%，超过 "
                            f"{100 * limits['max_sensor_share']:.0f}%：在读取 test 之前停下，等待决定是否调整缩放方式")
    return result


if __name__ == "__main__":
    sys.exit(stage_cli("diagnose", "只用 val_thr 的重尾特征诊断"))
