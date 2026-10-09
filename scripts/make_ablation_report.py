"""M4：汇总 LSTM-AE 消融（W × hidden）的结果。只做敏感性分析，不得据此修改主模型（约束 9）。

读取 产物目录/ablation/lstm_ae_w<W>_h<H>/ 下各运行的 eval_metrics.json、threshold.json、run_manifest.json。

输出（正式：reports/main/；--smoke：产物目录/smoke_reports/）：
    ablation.csv              每个组合一行：主表的各列 + 训练 epoch 数、每个 epoch 耗时、参数量
    ablation_quantiles.csv    可选的阈值分位敏感性（99 / 99.5 / 99.9）：每个组合 × 分位一行
    ablation.md               表格与说明
    ablation_heatmap.png      W × hidden 网格上的 F1、AUROC、事件检出率

缺少的组合（尚未运行或不合格）在表中留空并列出，不影响其余组合。

用法：
    python -m scripts.make_ablation_report [--smoke]
"""

import argparse
import sys

import numpy as np
import pandas as pd

from src.eval.run_io import read_json
from src.utils.paths import load_config
from src.utils.plotstyle import INK, SEQUENTIAL, SURFACE, apply_style, plt
from src.utils.i18n import t
from src.utils.reporting import ReportSource, add_common_args
from src.utils.run import MANIFEST_NAME, ablation_grid

METRICS = ["precision", "recall", "f1", "auroc", "event_recall", "delay_median_s", "delay_mean_s",
           "false_alarms_per_hour", "threshold", "n_unscored"]
HEATMAP_METRICS = [("f1", "逐点 F1"), ("auroc", "AUROC"), ("event_recall", "事件检出率")]  # 标题为中文源文，画图时经 t()


def collect(src: ReportSource, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    main = cfg["ml"]
    rows, quantile_rows, missing = [], [], []
    grid = ablation_grid(cfg)
    usable = set(src.usable_runs([g["run_name"] for g in grid], required=("eval_metrics.json", "threshold.json")))
    for g in grid:
        if g["run_name"] not in usable:
            missing.append(g["run_name"])
            continue
        d = src.root / g["run_name"]
        metrics, manifest = read_json(d / "eval_metrics.json"), read_json(d / MANIFEST_NAME)
        hp = manifest["hyperparameters"]
        if hp["window"]["length"] != g["window"] or hp["models"][main["ablation"]["model"]]["hidden"] != g["hidden"]:
            raise ValueError(f"{g['run_name']} 的 manifest 超参数与目录名不符")
        train = manifest.get("stages", {}).get("train", {})
        is_main = g["window"] == main["window"]["length"] and g["hidden"] == main["models"]["lstm_ae"]["hidden"]
        rows.append({"index": g["index"], "window": g["window"], "hidden": g["hidden"], "is_main_config": is_main,
                     **{k: metrics["main_row"][k] for k in METRICS},
                     "final_epoch": manifest.get("final_epoch"), "best_epoch": train.get("best_epoch"),
                     "early_stop_epoch": manifest.get("early_stop_epoch"),
                     "epoch_seconds_mean": train.get("epoch_seconds_mean"), "n_parameters": train.get("n_parameters")})
        for q, s in metrics.get("threshold_sensitivity", {}).items():
            quantile_rows.append({"window": g["window"], "hidden": g["hidden"], "quantile": float(q), **s})
    return pd.DataFrame(rows), pd.DataFrame(quantile_rows), missing


def plot_heatmaps(df: pd.DataFrame, windows: list[int], hiddens: list[int], path, smoke: bool) -> None:
    """每个指标一个 W × hidden 色块图（各自的色标，单一色相由浅到深），格子里写数值，主模型配置加框。"""
    apply_style()
    fig, axes = plt.subplots(1, len(HEATMAP_METRICS), figsize=(3.4 * len(HEATMAP_METRICS), 3.4), squeeze=False)
    for ax, (col, title) in zip(axes[0], HEATMAP_METRICS):
        grid = df.pivot(index="window", columns="hidden", values=col).reindex(index=windows, columns=hiddens)
        values = grid.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        lo, hi = (finite.min(), finite.max()) if len(finite) else (0, 1)
        vmin, vmax = (lo, hi) if hi > lo else (lo - 0.5, lo + 0.5)  # 全部相同时给色标一个非零的范围
        im = ax.imshow(values, cmap=SEQUENTIAL, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(len(windows)):
            for j in range(len(hiddens)):
                v = values[i, j]
                dark = np.isfinite(v) and hi > lo and (v - lo) / (hi - lo) > 0.55
                ax.text(j, i, "—" if not np.isfinite(v) else f"{v:.3f}", ha="center", va="center", fontsize=9,
                        color=SURFACE if dark else INK)
        main = df[df["is_main_config"]]
        if len(main):
            i, j = windows.index(int(main["window"].iloc[0])), hiddens.index(int(main["hidden"].iloc[0]))
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor=INK, linewidth=1.6))
        ax.set_xticks(range(len(hiddens)), hiddens)
        ax.set_yticks(range(len(windows)), windows)
        ax.set_xticks(np.arange(len(hiddens) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(windows) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("hidden")
        ax.set_title(t(title), loc="left", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0, 0].set_ylabel(t("窗口长度 W（秒）"))
    fig.suptitle(t("LSTM-AE 消融（黑框 = 事先固定的主模型配置）") + f"{t('（冒烟数据，仅检查流程）') if smoke else ''}",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def to_md(df: pd.DataFrame, qdf: pd.DataFrame, missing: list[str], smoke: bool) -> str:
    def cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    def table(frame):
        cols = list(frame.columns)
        return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + \
               ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]

    lines = ["# LSTM-AE 消融" + ("（冒烟数据，仅检查流程，不得引用）" if smoke else ""), "",
             "窗口长度 W × 隐层大小 hidden。主模型配置（is_main_config）事先固定，本表只用于敏感性分析，"
             "不据此修改主模型，也不据此选阈值。指标口径与主结果表相同（不使用 point-adjust）。", ""]
    lines += table(df.drop(columns=["index"])) if len(df) else ["（没有可用的组合）"]
    if missing:
        lines += ["", f"缺少的组合（{len(missing)} 个）：" + "、".join(missing)]
    if len(qdf):
        lines += ["", "## 阈值分位的敏感性", "",
                  "用 val_thr 分数的其他分位数作阈值重算同一套指标（AUROC 与阈值无关，不重复列出）。", ""]
        lines += table(qdf)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M4：LSTM-AE 消融汇总")
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    src = ReportSource(cfg, args.smoke)
    src.banner()
    df, qdf, missing = collect(src, cfg)
    if df.empty:
        print("[提示] 没有可用的消融运行，未生成结果")
        return 0
    ab = cfg["ml"]["ablation"]
    df.to_csv(src.out / "ablation.csv", index=False)
    if len(qdf):
        qdf.to_csv(src.out / "ablation_quantiles.csv", index=False)
    (src.out / "ablation.md").write_text(to_md(df, qdf, missing, args.smoke), encoding="utf-8")
    plot_heatmaps(df, ab["windows"], ab["hiddens"], src.out / "ablation_heatmap.png", args.smoke)
    print(f"[生成] {src.out}：ablation.csv / .md、ablation_heatmap.png（{len(df)} 个组合"
          f"{'，缺 ' + str(len(missing)) + ' 个' if missing else ''}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
