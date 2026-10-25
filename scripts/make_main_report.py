"""Generate the main-line report material into reports/main/ (tracked) from cluster artifacts synced back.

Output:
- main_results.csv / .md       main results table, one row per evaluated model (in M2a: PCA and IsolationForest)
- main_results_seeds.csv       multiple random seeds (ml.seed_study, only LSTM-AE in the main-model configuration): per-seed results with mean and std;
                               when all seed runs are available, F1, AUROC, event recall and false-alarm rate of that model in the .md table are written as "mean ± std"
- val_sensor_contrib.csv / .png  average share of each sensor in the PCA reconstruction MSE on val_thr (based on val_thr only)

Only full runs on the cluster are accepted: smoke=false, git_dirty=false, status=completed in run_manifest.json
(the diagnosis table also accepts stopped_for_review). Other runs are rejected with the reason; --allow-unofficial is only for tests
and must be combined with --output-dir pointing somewhere other than reports/main.

Usage:
    python -m scripts.make_main_report [--models pca iforest] [--output-dir DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.paths import DEFAULT_CONFIG, artifact_dir, load_config, resolve  # noqa: E402
from src.utils.reporting import report_problem as official_problem  # noqa: E402

MAIN_COLUMNS = ["model", "precision", "recall", "f1", "auroc", "event_recall", "delay_median_s", "delay_mean_s",
                "false_alarms_per_hour", "threshold", "n_unscored"]
SEED_COLUMNS = ["seed", "run_name", *MAIN_COLUMNS[1:]]
SEED_STAT_COLUMNS = ("f1", "auroc", "event_recall", "false_alarms_per_hour")  # Columns written as "mean ± std" in the main table
CONTRIB_FILE = "val_sensor_contrib.csv"
DIAGNOSIS_MODEL = "pca"
LOCAL_DIAGNOSTIC_TOLERANCE = 0.005  # Maximum allowed difference between local and cluster diagnosis shares (floating-point differences between BLAS libraries are far smaller)

# Colors match scripts/make_extension_report.py (dataviz reference palette, light): a single series in blue, bars over the limit hatched
SURFACE, INK, INK2, GRID, BLUE, HATCH_EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#0b0b0b"
plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "savefig.dpi": 200,
})


def main_table(root: Path, models: list[str], allow_unofficial: bool) -> tuple[pd.DataFrame, list[str]]:
    rows, skipped = [], []
    for name in models:
        run_dir = root / name
        metrics = run_dir / "eval_metrics.json"
        if not metrics.exists():
            skipped.append(f"{name}: no eval_metrics.json (not evaluated or not synced yet)")
            continue
        problem = official_problem(run_dir)
        if problem and not allow_unofficial:
            skipped.append(f"{name}: {problem}")
            continue
        rows.append(json.loads(metrics.read_text(encoding="utf-8"))["main_row"])
    return pd.DataFrame(rows, columns=MAIN_COLUMNS), skipped


def seed_table(root: Path, cfg: dict, allow_unofficial: bool) -> tuple[pd.DataFrame | None, list[str]]:
    """Per-seed results of the multi-seed study. Returns None if any seed's run is unavailable (no mean over an incomplete seed set)."""
    study = cfg["ml"]["seed_study"]
    model, base_seed = study["model"], cfg["ml"]["seed"]
    rows, problems = [], []
    for seed in study["seeds"]:
        run_name = model if seed == base_seed else f"{study['run_subdir']}/{model}_s{seed}"
        run_dir = root / run_name
        metrics = run_dir / "eval_metrics.json"
        if not metrics.exists():
            problems.append(f"{run_name}: no eval_metrics.json (not run or not synced yet)")
            continue
        problem = official_problem(run_dir)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")) if not problem else {}
        if not problem:
            # Apart from the seed, the configuration must be identical to the main model: the main-model run has no overrides, the others only ml.seed
            expected = [] if seed == base_seed else [f"{study['seed_key']}={seed}"]
            if manifest.get("seed") != seed or manifest.get("overrides") != expected:
                problem = f"seed or overrides do not match (seed={manifest.get('seed')}, overrides={manifest.get('overrides')}, expected {expected})"
        if problem and not allow_unofficial:
            problems.append(f"{run_name}: {problem}")
            continue
        row = json.loads(metrics.read_text(encoding="utf-8"))["main_row"]
        rows.append({"seed": seed, "run_name": run_name, **{c: row[c] for c in MAIN_COLUMNS[1:]}})
    if problems:
        return None, problems
    return pd.DataFrame(rows, columns=SEED_COLUMNS), []


def seed_summary(seeds: pd.DataFrame) -> pd.DataFrame:
    """Append mean and std rows after the per-seed rows (sample std, ddof=1)."""
    numeric = seeds[MAIN_COLUMNS[1:]].astype(float)
    stats = pd.DataFrame([{"seed": "mean", "run_name": "", **numeric.mean().to_dict()},
                          {"seed": "std", "run_name": "", **numeric.std(ddof=1).to_dict()}])
    return pd.concat([seeds.astype({"seed": object}), stats], ignore_index=True)


def fmt(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def main_table_md(df: pd.DataFrame, seeds: pd.DataFrame | None = None, seed_model: str | None = None,
                  base_seed: int | None = None) -> str:
    lines = ["# Main results", "",
             "All numbers come from full runs on the cluster. Threshold = 99% quantile of the val_thr scores; point-wise metrics **do not use point-adjust**, "
             "so a low point-wise F1 is expected.", "",
             "The first W−1 seconds of each test file have no score and are excluded from every metric; their number is in the n_unscored column. "
             "Detection delay counts only detected events; false-alarm rate = alarm segments overlapping no attack event ÷ normal duration (hours, counting only scored seconds without attack).", "",
             "Metric conventions: alarm rule is score > threshold; with no alarms precision is shown as — (NaN), and if there are attacks within the evaluation range "
             "recall and F1 are 0; AUROC depends only on the scores, not on the threshold, and is undefined only when all labels within the evaluation range are of one class.", "",
             "| " + " | ".join(MAIN_COLUMNS) + " |", "|" + "---|" * len(MAIN_COLUMNS)]
    for row in df.itertuples(index=False):
        cells = [fmt(v, 6 if c == "threshold" else 1 if c.startswith("delay") else 4)
                 for c, v in zip(MAIN_COLUMNS, row)]
        if seeds is not None and row.model == seed_model:
            for c in SEED_STAT_COLUMNS:
                cells[MAIN_COLUMNS.index(c)] = f"{seeds[c].mean():.4f} ± {seeds[c].std(ddof=1):.4f}"
        lines.append("| " + " | ".join(cells) + " |")
    if seeds is not None and seed_model in set(df["model"]):
        n = len(seeds)
        main = df[df["model"] == seed_model].iloc[0]
        lines += ["", "## Random seeds", "",
                  f"In the {seed_model} row, {', '.join(SEED_STAT_COLUMNS)} are the mean ± sample std (ddof=1) over {n} random seeds"
                  f" ({', '.join(str(s) for s in seeds['seed'])}); "
                  f"the other columns of that row come from the main-model run with seed {base_seed} (that run's "
                  + ", ".join(f"{c} {main[c]:.4f}" for c in SEED_STAT_COLUMNS)
                  + "); alert export, attribution and figures also use only that run. "
                  "The seed runs have identical configurations apart from the seed, and each threshold is the 99% quantile of its own val_thr. "
                  "**All other models have results for a single seed only**; their random variation was not assessed.", "",
                  "| " + " | ".join(SEED_COLUMNS) + " |", "|" + "---|" * len(SEED_COLUMNS)]
        for r in seed_summary(seeds).itertuples(index=False):
            cells = [str(r.seed), r.run_name] + [
                fmt(float(v), 6 if c == "threshold" else 1 if c.startswith("delay") else 0 if c == "n_unscored" else 4)
                for c, v in zip(MAIN_COLUMNS[1:], r[2:])]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_contrib(df: pd.DataFrame, limit: float, path: Path, top_n: int = 20) -> None:
    """Horizontal bar chart of the top_n sensors by share; bars over limit are hatched (not distinguished by color alone)."""
    top = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(top) + 1.6))
    for y, row in enumerate(top.itertuples()):
        over = row.share > limit
        ax.barh(y, 100 * row.share, height=0.62, color=BLUE, hatch="//" if over else None,
                edgecolor=HATCH_EDGE if over else SURFACE, linewidth=0.6 if over else 0)
        ax.text(100 * row.share, y, f" {100 * row.share:.1f}%", va="center", ha="left", fontsize=8, color=INK)
    ax.set_yticks(range(len(top)), top["sensor"], fontsize=8)
    ax.axvline(100 * limit, color=INK2, linestyle="--", linewidth=1)
    ax.text(100 * limit, len(top) - 0.3, " " + "limit {pct}%".format(pct=f"{100 * limit:.0f}"), color=INK2, fontsize=9, va="bottom")
    ax.set_xlim(0, max(100 * limit, 100 * df["share"].max()) * 1.15)
    ax.set_ylim(-0.6, len(top) + 0.4)
    ax.set_xlabel("Share of the reconstruction MSE on val_thr (%)")
    ax.set_title("PCA: error share of each sensor on val_thr (top {top} of {n})".format(top=len(top), n=len(df)),
                 loc="left", fontsize=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate the main-line report material (main results table, val_thr sensor shares)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--models", nargs="+", help="models written to the main table, defaults to all models in ml.models of the config")
    p.add_argument("--output-dir", type=Path, help="defaults to ml.report_dir in the config")
    p.add_argument("--allow-unofficial", action="store_true", help="accept smoke or dirty runs (tests only; requires --output-dir)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    report_dir = resolve(cfg["ml"]["report_dir"])
    out = (args.output_dir or report_dir).resolve()
    if args.allow_unofficial and out == report_dir.resolve():
        p.error("--allow-unofficial cannot write to the official report directory; point --output-dir elsewhere")
    out.mkdir(parents=True, exist_ok=True)
    root = artifact_dir(cfg)

    df, skipped = main_table(root, args.models or list(cfg["ml"]["models"]), args.allow_unofficial)
    for s in skipped:
        print(f"[SKIPPED] {s}")
    study = cfg["ml"]["seed_study"]
    seeds, seed_problems = seed_table(root, cfg, args.allow_unofficial)
    if seeds is None:
        print(f"[INFO] multi-seed study ({study['model']}) is incomplete, the main table stays single-seed:")
        for s in seed_problems:
            print(f"  [SKIPPED] {s}")
    if len(df):
        df.to_csv(out / "main_results.csv", index=False)
        (out / "main_results.md").write_text(main_table_md(df, seeds, study["model"], cfg["ml"]["seed"]), encoding="utf-8")
        print(f"[GENERATED] main_results.csv / .md ({len(df)} rows: {', '.join(df['model'])})")
        if seeds is not None and study["model"] in set(df["model"]):
            seed_summary(seeds).to_csv(out / "main_results_seeds.csv", index=False)
            print(f"[GENERATED] main_results_seeds.csv ({len(seeds)} seeds); {study['model']}: "
                  + ", ".join(f"{c} {seeds[c].mean():.4f} ± {seeds[c].std(ddof=1):.4f}" for c in SEED_STAT_COLUMNS))
    else:
        print("[INFO] no runs available for the main table, main_results not written")

    contrib = root / DIAGNOSIS_MODEL / CONTRIB_FILE
    problem = official_problem(root / DIAGNOSIS_MODEL, ("completed", "stopped_for_review"))
    if not contrib.exists():
        print(f"[SKIPPED] diagnosis: the run directory of {DIAGNOSIS_MODEL} has no {CONTRIB_FILE}")
    elif problem and not args.allow_unofficial:
        print(f"[SKIPPED] diagnosis: {problem}")
    else:
        cdf = pd.read_csv(contrib)
        limit = cfg["ml"]["diagnosis"]["max_sensor_share"]
        cdf.to_csv(out / CONTRIB_FILE, index=False)
        plot_contrib(cdf, limit, out / "val_sensor_contrib.png")
        top = cdf.iloc[0]
        local = root / "local_diagnostic" / DIAGNOSIS_MODEL / CONTRIB_FILE
        if local.exists():  # The local diagnostic run and the full cluster run should agree
            both = cdf.merge(pd.read_csv(local), on="sensor", suffixes=("", "_local"))
            diff = (both["share"] - both["share_local"]).abs().max()
            same = len(both) == len(cdf) and diff < LOCAL_DIAGNOSTIC_TOLERANCE
            print(f"[{'MATCH' if same else 'MISMATCH'}] compared with the local diagnostic run: largest share difference {100 * diff:.3f} percentage points"
                  f"{'' if same else ' (must be reported)'}")
        print(f"[GENERATED] {CONTRIB_FILE} / .png; largest share: {top['sensor']} {100 * top['share']:.1f}%"
              f" (limit {100 * limit:.0f}%, {'over the limit' if top['share'] > limit else 'within the limit'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
