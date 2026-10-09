"""用同步回来的集群产物生成主线报告素材到 reports/main/（入库）。

输出：
- main_results.csv / .md       主结果表，每个已评估的模型一行（M2a 为 PCA、IsolationForest 两行）
- main_results_seeds.csv       多随机种子（ml.seed_study，只对主模型配置的 LSTM-AE）逐种子的结果与均值、标准差；
                               全部种子的运行都可用时，主表 .md 中该模型的 F1、AUROC、事件检出率、误报率写成"均值 ± 标准差"
- val_sensor_contrib.csv / .png  PCA 在 val_thr 上各传感器对重构 MSE 的平均贡献占比（只依据 val_thr）

只接受集群上的完整运行：run_manifest.json 中 smoke=false、git_dirty=false、status=completed
（诊断表还接受 stopped_for_review）。不满足的运行被拒绝并说明原因；--allow-unofficial 只用于测试，
此时必须同时用 --output-dir 指向 reports/main 以外的目录。

用法：
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
from src.utils.i18n import t  # noqa: E402
from src.utils.reporting import report_problem as official_problem  # noqa: E402

MAIN_COLUMNS = ["model", "precision", "recall", "f1", "auroc", "event_recall", "delay_median_s", "delay_mean_s",
                "false_alarms_per_hour", "threshold", "n_unscored"]
SEED_COLUMNS = ["seed", "run_name", *MAIN_COLUMNS[1:]]
SEED_STAT_COLUMNS = ("f1", "auroc", "event_recall", "false_alarms_per_hour")  # 主表中写成"均值 ± 标准差"的列
CONTRIB_FILE = "val_sensor_contrib.csv"
DIAGNOSIS_MODEL = "pca"
LOCAL_DIAGNOSTIC_TOLERANCE = 0.005  # 本地与集群的诊断占比允许的最大差（不同 BLAS 的浮点误差远小于此）

# 配色与 scripts/make_extension_report.py 一致（dataviz 参考调色板，浅色）：单一系列用蓝色，超限的条加斜线纹理
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
            skipped.append(f"{name}: 没有 eval_metrics.json（尚未评估或尚未同步）")
            continue
        problem = official_problem(run_dir)
        if problem and not allow_unofficial:
            skipped.append(f"{name}: {problem}")
            continue
        rows.append(json.loads(metrics.read_text(encoding="utf-8"))["main_row"])
    return pd.DataFrame(rows, columns=MAIN_COLUMNS), skipped


def seed_table(root: Path, cfg: dict, allow_unofficial: bool) -> tuple[pd.DataFrame | None, list[str]]:
    """多随机种子的逐种子结果。任何一个种子的运行不可用时返回 None（不用不完整的种子集合算均值）。"""
    study = cfg["ml"]["seed_study"]
    model, base_seed = study["model"], cfg["ml"]["seed"]
    rows, problems = [], []
    for seed in study["seeds"]:
        run_name = model if seed == base_seed else f"{study['run_subdir']}/{model}_s{seed}"
        run_dir = root / run_name
        metrics = run_dir / "eval_metrics.json"
        if not metrics.exists():
            problems.append(f"{run_name}: 没有 eval_metrics.json（尚未运行或尚未同步）")
            continue
        problem = official_problem(run_dir)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")) if not problem else {}
        if not problem:
            # 除种子外必须与主模型配置完全相同：主模型运行没有覆盖项，其余只允许 ml.seed 一项
            expected = [] if seed == base_seed else [f"{study['seed_key']}={seed}"]
            if manifest.get("seed") != seed or manifest.get("overrides") != expected:
                problem = f"种子或覆盖项不符（seed={manifest.get('seed')}，overrides={manifest.get('overrides')}，应为 {expected}）"
        if problem and not allow_unofficial:
            problems.append(f"{run_name}: {problem}")
            continue
        row = json.loads(metrics.read_text(encoding="utf-8"))["main_row"]
        rows.append({"seed": seed, "run_name": run_name, **{c: row[c] for c in MAIN_COLUMNS[1:]}})
    if problems:
        return None, problems
    return pd.DataFrame(rows, columns=SEED_COLUMNS), []


def seed_summary(seeds: pd.DataFrame) -> pd.DataFrame:
    """逐种子的行后面加均值、标准差两行（样本标准差，ddof=1）。"""
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
    lines = ["# 主结果表", "",
             "数字全部来自集群上的完整运行。阈值 = val_thr 分数的 99% 分位数；逐点指标**不使用 point-adjust**，"
             "因此逐点 F1 偏低是预期结果。", "",
             "每个测试文件的前 W−1 秒没有分数，不参与任何指标，数量见 n_unscored 列。"
             "检测延迟只统计检出的事件；误报率 = 不与任何攻击事件重叠的报警段数 ÷ 正常时长（小时，只计已打分且无攻击的秒）。", "",
             "指标约定：报警规则为分数 > 阈值；没有报警时 precision 记为 —（NaN），此时若评估范围内有攻击，"
             "recall 与 F1 记 0；AUROC 只依赖分数、与阈值无关，只有评估范围内全是同一类标签时才无法定义。", "",
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
        lines += ["", "## 随机种子", "",
                  f"{seed_model} 一行的 {'、'.join(SEED_STAT_COLUMNS)} 是 {n} 个随机种子"
                  f"（{', '.join(str(s) for s in seeds['seed'])}）的均值 ± 样本标准差（ddof=1）；"
                  f"该行其余各列来自种子 {base_seed} 的主模型运行（这次运行的 "
                  + "、".join(f"{c} {main[c]:.4f}" for c in SEED_STAT_COLUMNS)
                  + "），报警导出、归因和图也只用这一次运行。"
                  "各种子的运行除种子外配置完全相同，阈值各自按自己的 val_thr 99% 分位数计算。"
                  "**其余模型都只有单个种子的结果**，它们的随机波动没有评估。", "",
                  "| " + " | ".join(SEED_COLUMNS) + " |", "|" + "---|" * len(SEED_COLUMNS)]
        for r in seed_summary(seeds).itertuples(index=False):
            cells = [str(r.seed), r.run_name] + [
                fmt(float(v), 6 if c == "threshold" else 1 if c.startswith("delay") else 0 if c == "n_unscored" else 4)
                for c, v in zip(MAIN_COLUMNS[1:], r[2:])]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_contrib(df: pd.DataFrame, limit: float, path: Path, top_n: int = 20) -> None:
    """横向条形图：占比最大的 top_n 个传感器；超过 limit 的条加斜线纹理（不只靠颜色区分）。"""
    top = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(top) + 1.6))
    for y, row in enumerate(top.itertuples()):
        over = row.share > limit
        ax.barh(y, 100 * row.share, height=0.62, color=BLUE, hatch="//" if over else None,
                edgecolor=HATCH_EDGE if over else SURFACE, linewidth=0.6 if over else 0)
        ax.text(100 * row.share, y, f" {100 * row.share:.1f}%", va="center", ha="left", fontsize=8, color=INK)
    ax.set_yticks(range(len(top)), top["sensor"], fontsize=8)
    ax.axvline(100 * limit, color=INK2, linestyle="--", linewidth=1)
    ax.text(100 * limit, len(top) - 0.3, " " + t("上限 {pct}%").format(pct=f"{100 * limit:.0f}"), color=INK2, fontsize=9, va="bottom")
    ax.set_xlim(0, max(100 * limit, 100 * df["share"].max()) * 1.15)
    ax.set_ylim(-0.6, len(top) + 0.4)
    ax.set_xlabel(t("占 val_thr 重构 MSE 的比例（%）"))
    ax.set_title(t("PCA 在 val_thr 上各传感器的误差占比（前 {top} 个，共 {n} 个）").format(top=len(top), n=len(df)),
                 loc="left", fontsize=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="生成主线报告素材（主结果表、val_thr 传感器占比）")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--models", nargs="+", help="写入主表的模型，默认为 config 的 ml.models 中全部模型")
    p.add_argument("--output-dir", type=Path, help="默认为 config 的 ml.report_dir")
    p.add_argument("--allow-unofficial", action="store_true", help="接受冒烟或 dirty 的运行（只用于测试，需同时给 --output-dir）")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    report_dir = resolve(cfg["ml"]["report_dir"])
    out = (args.output_dir or report_dir).resolve()
    if args.allow_unofficial and out == report_dir.resolve():
        p.error("--allow-unofficial 不能写进正式的报告目录，请用 --output-dir 指向别处")
    out.mkdir(parents=True, exist_ok=True)
    root = artifact_dir(cfg)

    df, skipped = main_table(root, args.models or list(cfg["ml"]["models"]), args.allow_unofficial)
    for s in skipped:
        print(f"[跳过] {s}")
    study = cfg["ml"]["seed_study"]
    seeds, seed_problems = seed_table(root, cfg, args.allow_unofficial)
    if seeds is None:
        print(f"[提示] 多随机种子（{study['model']}）还不完整，主表维持单种子：")
        for s in seed_problems:
            print(f"  [跳过] {s}")
    if len(df):
        df.to_csv(out / "main_results.csv", index=False)
        (out / "main_results.md").write_text(main_table_md(df, seeds, study["model"], cfg["ml"]["seed"]), encoding="utf-8")
        print(f"[生成] main_results.csv / .md（{len(df)} 行：{', '.join(df['model'])}）")
        if seeds is not None and study["model"] in set(df["model"]):
            seed_summary(seeds).to_csv(out / "main_results_seeds.csv", index=False)
            print(f"[生成] main_results_seeds.csv（{len(seeds)} 个种子）；{study['model']}："
                  + "，".join(f"{c} {seeds[c].mean():.4f} ± {seeds[c].std(ddof=1):.4f}" for c in SEED_STAT_COLUMNS))
    else:
        print("[提示] 没有可用于主表的运行，未写 main_results")

    contrib = root / DIAGNOSIS_MODEL / CONTRIB_FILE
    problem = official_problem(root / DIAGNOSIS_MODEL, ("completed", "stopped_for_review"))
    if not contrib.exists():
        print(f"[跳过] 诊断：{DIAGNOSIS_MODEL} 的运行目录里没有 {CONTRIB_FILE}")
    elif problem and not args.allow_unofficial:
        print(f"[跳过] 诊断：{problem}")
    else:
        cdf = pd.read_csv(contrib)
        limit = cfg["ml"]["diagnosis"]["max_sensor_share"]
        cdf.to_csv(out / CONTRIB_FILE, index=False)
        plot_contrib(cdf, limit, out / "val_sensor_contrib.png")
        top = cdf.iloc[0]
        local = root / "local_diagnostic" / DIAGNOSIS_MODEL / CONTRIB_FILE
        if local.exists():  # 本地诊断运行与集群完整运行应当一致
            both = cdf.merge(pd.read_csv(local), on="sensor", suffixes=("", "_local"))
            diff = (both["share"] - both["share_local"]).abs().max()
            same = len(both) == len(cdf) and diff < LOCAL_DIAGNOSTIC_TOLERANCE
            print(f"[{'一致' if same else '不一致'}] 与本地诊断运行比较：占比最大相差 {100 * diff:.3f} 个百分点"
                  f"{'' if same else '（需要汇报）'}")
        print(f"[生成] {CONTRIB_FILE} / .png；占比最大：{top['sensor']} {100 * top['share']:.1f}%"
              f"（上限 {100 * limit:.0f}%，{'超限' if top['share'] > limit else '未超限'}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
