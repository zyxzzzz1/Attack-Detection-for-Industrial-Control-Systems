"""Heavy-tail feature diagnosis (val_thr only, test is not read): average share of each sensor in the reconstruction MSE.

For models with per-sensor errors, take the mean squared error over the scored seconds of val_thr, normalize to shares, and write
val_sensor_contrib.csv into the run directory. If a single sensor's share exceeds ml.diagnosis.max_sensor_share, raise DiagnosisStop:
the run stops before reading test and a human decides whether to change the scaling. After the decision, ml.diagnosis.allow_over_limit=true lets it through.

Usage:
    python -m src.diagnose --model pca [--smoke]
"""

import sys

import numpy as np
import pandas as pd

from src.score import score_val_thr
from src.utils.context import stage_cli

CONTRIB_FILE = "val_sensor_contrib.csv"


class DiagnosisStop(RuntimeError):
    """The diagnosis result needs a human decision; the run stops before reading test."""


def sensor_contributions(per_sensor_err: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Per-second per-sensor errors (T, D) -> mean squared error and share per sensor, sorted by share descending. All-NaN rows are ignored."""
    scored = ~np.isnan(per_sensor_err).all(axis=1)
    mean_err = per_sensor_err[scored].astype(np.float64).mean(axis=0)
    total = mean_err.sum()
    df = pd.DataFrame({"sensor": features, "subsystem": [f.split("_")[0] for f in features],
                       "mean_sq_err": mean_err, "share": mean_err / total if total > 0 else np.nan})
    return df.sort_values("share", ascending=False, kind="stable").reset_index(drop=True)


def scale_profile(data) -> pd.DataFrame:
    """Each sensor's standard deviation on the train part (the value used by the scaler), and the distribution of |z| in val_thr after standardization."""
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
    gated = ctx.model == limits["model"]  # Whether to stop depends only on the designated baseline (PCA); other models only write the share table
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
