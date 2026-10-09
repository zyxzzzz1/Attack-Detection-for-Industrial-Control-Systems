"""M3 的图和分子系统检出表（不读取标签：攻击区间取自 evaluate 阶段写出的 eval_events.csv）。

输出（正式：reports/main/；--smoke：产物目录/smoke_reports/）：
    scores_<文件>[_rowN].png        每个测试文件一张：各模型的分数曲线（对数纵轴）、阈值线、攻击区间阴影
    score_distribution.png          各模型的分数分布：val_thr、test 正常、test 攻击
    training_curves.png             LSTM 类模型的训练 / 验证损失与最好的 epoch
    per_subsystem_detection.csv     各模型按受攻击子系统统计的事件检出率与延迟

主结果表由 scripts.make_main_report 生成。

用法：
    python -m scripts.make_figures [--models pca iforest lstm_ae lstm_vae deep_svdd] [--smoke]
"""

import argparse
import sys

import numpy as np
import pandas as pd

from src.eval.run_io import load_run
from src.utils.paths import load_config
from src.utils.i18n import t  # noqa: E402
from src.utils.plotstyle import EVENT_EDGE, EVENT_FILL, INK2, SERIES, apply_style, clean_axes, plt
from src.utils.reporting import ReportSource, add_common_args

REQUIRED = ("scores_test.npy", "scores_val_thr.npy", "threshold.json", "test_index.json", "eval_events.csv")
MAX_POINTS = 6000  # 每条曲线最多画这么多点：按块取最大值降采样，尖峰不会被抹掉


def read_events(run_dir) -> pd.DataFrame:
    try:
        return pd.read_csv(run_dir / "eval_events.csv")
    except pd.errors.EmptyDataError:  # 评估范围内没有事件
        return pd.DataFrame(columns=["part", "start", "end", "detected", "delay_s", "subsystems"])


def downsample_max(y: np.ndarray, max_points: int = MAX_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """(x, y)：点数超过 max_points 时按等长的块取最大值（全为 NaN 的块保持 NaN）。"""
    n = len(y)
    step = int(np.ceil(n / max_points))
    if step <= 1:
        return np.arange(n), y
    pad = (-n) % step
    blocks = np.concatenate([y, np.full(pad, np.nan)]).reshape(-1, step)
    out = np.full(len(blocks), np.nan)
    valid = ~np.isnan(blocks).all(axis=1)
    out[valid] = np.nanmax(blocks[valid], axis=1)
    return np.arange(len(blocks)) * step, out


def part_label(part: dict) -> str:
    stem = part["file"].split(".")[0]
    return stem if part["start_row"] == 0 else f"{stem}_row{part['start_row']}"


def plot_scores(runs: dict, part_idx: int, path, smoke: bool) -> None:
    """一个测试文件（或片段）一张图，每个模型一个子图，共用时间轴。分数尺度各不相同，所以不共用纵轴。"""
    names = list(runs)
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 1.7 * len(names) + 1), sharex=True, squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        run, events = runs[name]
        part = run.index["files"][part_idx]
        s = run.scores[part["offset"]:part["offset"] + part["rows"]]
        t0 = pd.Timestamp(part["start_time"])
        x, y = downsample_max(s)
        ax.plot(t0 + pd.to_timedelta(x, unit="s"), y, color=SERIES[0], linewidth=0.8, label=t("异常分数"))
        ax.axhline(run.threshold["threshold"], color=INK2, linestyle="--", linewidth=0.9, label=t("阈值（val_thr 99% 分位）"))
        for i, e in enumerate(events[events["part"] == part_idx].itertuples()):
            ax.axvspan(t0 + pd.Timedelta(seconds=e.start), t0 + pd.Timedelta(seconds=e.end + 1), facecolor=EVENT_FILL,
                       edgecolor=EVENT_EDGE, linewidth=0.5, hatch="///", label=t("攻击区间") if i == 0 else None, zorder=0)
        ax.set_yscale("log")
        ax.set_ylabel(name)
        clean_axes(ax)
    axes[0, 0].legend(loc="lower right", bbox_to_anchor=(1, 1.02), ncol=3, fontsize=8)
    part = next(iter(runs.values()))[0].index["files"][part_idx]
    axes[0, 0].set_title(t("{part}：各模型的逐秒异常分数").format(part=part_label(part)) + f"{t('（冒烟数据，仅检查流程）') if smoke else ''}",
                         loc="left", fontsize=11)
    axes[-1, 0].set_xlabel(t("时间"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def event_mask(run, events: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(run.scores), dtype=bool)
    for e in events.itertuples():
        offset = run.index["files"][e.part]["offset"]
        mask[offset + e.start:offset + e.end + 1] = True
    return mask


def plot_distributions(runs: dict, root, path, smoke: bool) -> None:
    """每个模型一个子图：val_thr、test 正常、test 攻击三组分数的分布（对数横轴，密度）。"""
    names = list(runs)
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        run, events = runs[name]
        mask = event_mask(run, events)
        val = np.load(root / name / "scores_val_thr.npy")
        groups = [("val_thr", val), (t("test 正常"), run.scores[~mask]), (t("test 攻击"), run.scores[mask])]
        pooled = np.concatenate([g[~np.isnan(g)] for _, g in groups])
        pooled = pooled[pooled > 0]
        if len(pooled) == 0:
            continue
        bins = np.logspace(np.log10(pooled.min()), np.log10(pooled.max()) + 1e-9, 60)
        for (label, g), color in zip(groups, SERIES):
            g = g[~np.isnan(g)]
            if len(g):
                ax.hist(g, bins=bins, density=True, histtype="step", linewidth=1.4, color=color, label=label)
        ax.axvline(run.threshold["threshold"], color=INK2, linestyle="--", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xlabel(t("异常分数"))
        clean_axes(ax)
    axes[0, 0].set_ylabel(t("密度"))
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(t("分数分布（虚线为阈值）") + f"{t('（冒烟数据，仅检查流程）') if smoke else ''}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_training(root, names: list[str], path, smoke: bool) -> bool:
    logs = {n: pd.read_csv(root / n / "train_log.csv") for n in names if (root / n / "train_log.csv").exists()}
    if not logs:
        return False
    fig, axes = plt.subplots(1, len(logs), figsize=(3.6 * len(logs), 3.2), squeeze=False)
    for ax, (name, log) in zip(axes[0], logs.items()):
        ax.plot(log["epoch"], log["train_loss"], color=SERIES[0], linewidth=1.6, marker="o", ms=3, label="train")
        ax.plot(log["epoch"], log["val_es_loss"], color=SERIES[1], linewidth=1.6, marker="o", ms=3, label="val_es")
        best = log.loc[log["val_es_loss"].idxmin()]
        ax.axvline(best["epoch"], color=INK2, linestyle=":", linewidth=0.9)
        ax.annotate(t("最好：第 {n} 个 epoch").format(n=int(best["epoch"])), (best["epoch"], best["val_es_loss"]), fontsize=8,
                    xytext=(4, 8), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xlabel("epoch")
        clean_axes(ax)
    axes[0, 0].set_ylabel(t("损失"))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(t("训练曲线") + f"{t('（冒烟数据，仅检查流程）') if smoke else ''}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def per_subsystem_detection(runs: dict) -> pd.DataFrame:
    """按受攻击子系统统计事件检出率。一个事件涉及多个子系统时，在每个子系统下各计一次；ALL 为全部事件。"""
    rows = []
    for name, (_, events) in runs.items():
        groups = {"ALL": events}
        subs = sorted({s for v in events["subsystems"].fillna("") for s in v.split("+") if s})
        for s in subs:
            groups[s] = events[[s in str(v).split("+") for v in events["subsystems"].fillna("")]]
        groups["无子系统标签"] = events[events["subsystems"].fillna("") == ""]
        for sub, ev in groups.items():
            if sub == "无子系统标签" and ev.empty:
                continue
            detected = ev[ev["detected"].astype(bool)] if len(ev) else ev
            rows.append({"model": name, "subsystem": sub, "n_events": len(ev), "n_detected": len(detected),
                         "event_recall": len(detected) / len(ev) if len(ev) else np.nan,
                         "delay_median_s": detected["delay_s"].median() if len(detected) else np.nan,
                         "delay_mean_s": detected["delay_s"].mean() if len(detected) else np.nan})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M3：分数曲线、分数分布、训练曲线、分子系统检出表")
    add_common_args(p)
    p.add_argument("--models", nargs="+", help="默认为 config 的 ml.models 中全部模型")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    src = ReportSource(cfg, args.smoke)
    src.banner()
    names = src.usable_runs(args.models or list(cfg["ml"]["models"]), required=REQUIRED)
    if not names:
        print("[提示] 没有可用的运行，未生成图")
        return 0
    apply_style()
    runs = {n: (load_run(src.root / n), read_events(src.root / n)) for n in names}
    parts = [[(p["file"], p["start_row"], p["rows"]) for p in run.index["files"]] for run, _ in runs.values()]
    if any(x != parts[0] for x in parts):
        print("[错误] 各模型的 test_index.json 不一致（不是同一份测试数据），不能画在一起", file=sys.stderr)
        return 1
    first = next(iter(runs.values()))[0]
    for k, part in enumerate(first.index["files"]):
        plot_scores(runs, k, src.out / f"scores_{part_label(part)}.png", args.smoke)
    plot_distributions(runs, src.root, src.out / "score_distribution.png", args.smoke)
    has_training = plot_training(src.root, names, src.out / "training_curves.png", args.smoke)
    per_subsystem_detection(runs).to_csv(src.out / "per_subsystem_detection.csv", index=False)
    print(f"[生成] {src.out}：scores_*.png × {len(first.index['files'])}、score_distribution.png、"
          f"{'training_curves.png、' if has_training else ''}per_subsystem_detection.csv（模型：{', '.join(names)}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
