"""M4: summarize the LSTM-AE ablation (W x hidden). Sensitivity analysis only; the main model must not be changed based on it (constraint 9).

Reads eval_metrics.json, threshold.json and run_manifest.json of each run under <artifact dir>/ablation/lstm_ae_w<W>_h<H>/.

Output (official: reports/main/; --smoke: <artifact dir>/smoke_reports/):
    ablation.csv              one row per combination: the main-table columns + epochs trained, time per epoch, parameter count
    ablation_quantiles.csv    optional threshold-quantile sensitivity (99 / 99.5 / 99.9): one row per combination x quantile
    ablation.md               table and notes
    ablation_heatmap.png      F1, AUROC and event recall on the W x hidden grid

Missing combinations (not yet run or not acceptable) are left blank in the table and listed, without affecting the others.

Usage:
    python -m scripts.make_ablation_report [--smoke]
"""

import argparse
import sys

import numpy as np
import pandas as pd

from src.eval.run_io import read_json
from src.utils.paths import load_config
from src.utils.plotstyle import INK, SEQUENTIAL, SURFACE, apply_style, plt
from src.utils.reporting import ReportSource, add_common_args
from src.utils.run import MANIFEST_NAME, ablation_grid

METRICS = ["precision", "recall", "f1", "auroc", "event_recall", "delay_median_s", "delay_mean_s",
           "false_alarms_per_hour", "threshold", "n_unscored"]
HEATMAP_METRICS = [("f1", "Point-wise F1"), ("auroc", "AUROC"), ("event_recall", "Event detection rate")]


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
            raise ValueError(f"the manifest hyperparameters of {g['run_name']} do not match the directory name")
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
    """One W x hidden heatmap per metric (each with its own color scale, a single hue from light to dark), values written in the cells, main-model configuration outlined."""
    apply_style()
    fig, axes = plt.subplots(1, len(HEATMAP_METRICS), figsize=(3.4 * len(HEATMAP_METRICS), 3.4), squeeze=False)
    for ax, (col, title) in zip(axes[0], HEATMAP_METRICS):
        grid = df.pivot(index="window", columns="hidden", values=col).reindex(index=windows, columns=hiddens)
        values = grid.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        lo, hi = (finite.min(), finite.max()) if len(finite) else (0, 1)
        vmin, vmax = (lo, hi) if hi > lo else (lo - 0.5, lo + 0.5)  # If all values are equal, give the color scale a nonzero range
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
        ax.set_title(title, loc="left", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0, 0].set_ylabel("Window length W (s)")
    fig.suptitle("LSTM-AE ablation (black frame = main model configuration fixed in advance)" + f"{' (smoke-test data, pipeline check only)' if smoke else ''}",
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

    lines = ["# LSTM-AE ablation" + (" (smoke-test data, pipeline check only, do not cite)" if smoke else ""), "",
             "Window length W × hidden size. The main-model configuration (is_main_config) was fixed in advance; this table is only for sensitivity analysis, "
             "and is used neither to change the main model nor to choose the threshold. Metric definitions are the same as in the main results table (no point-adjust).", ""]
    lines += table(df.drop(columns=["index"])) if len(df) else ["(no combinations available)"]
    if missing:
        lines += ["", f"Missing combinations ({len(missing)}): " + ", ".join(missing)]
    if len(qdf):
        lines += ["", "## Threshold-quantile sensitivity", "",
                  "The same metrics recomputed with other quantiles of the val_thr scores as the threshold (AUROC does not depend on the threshold and is not repeated).", ""]
        lines += table(qdf)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M4: LSTM-AE ablation summary")
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    src = ReportSource(cfg, args.smoke)
    src.banner()
    df, qdf, missing = collect(src, cfg)
    if df.empty:
        print("[INFO] no ablation runs available, nothing generated")
        return 0
    ab = cfg["ml"]["ablation"]
    df.to_csv(src.out / "ablation.csv", index=False)
    if len(qdf):
        qdf.to_csv(src.out / "ablation_quantiles.csv", index=False)
    (src.out / "ablation.md").write_text(to_md(df, qdf, missing, args.smoke), encoding="utf-8")
    plot_heatmaps(df, ab["windows"], ab["hiddens"], src.out / "ablation_heatmap.png", args.smoke)
    print(f"[GENERATED] {src.out}: ablation.csv / .md, ablation_heatmap.png ({len(df)} combinations"
          f"{', ' + str(len(missing)) + ' missing' if missing else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
