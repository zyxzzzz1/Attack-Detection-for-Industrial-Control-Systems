"""传感器归因及其评估（M5，主线补充决策 7）。除 evaluate.py 外唯一读取 test 标签的模块。

每个报警段内，各传感器 per_sensor_err 的均值归一化为占比，取 Top-k（k 来自 ml.attribution.top_k）。
报警段、Top-k 的计算在 src/eval/alarms.py（不涉及标签），与 scripts/export_alerts.py 共用；这里只负责用标签评估：

- 只对与攻击事件重叠的报警段计算子系统命中率：Top-1 传感器所属子系统 ∈ 事件的子系统集合即为命中。
  报警段与多个事件重叠时，按重叠时间最长的那个事件判断（主口径；重叠时间相同取较早的事件）；
  另输出并集口径作为补充（Top-1 所属子系统 ∈ 这些事件子系统集合的并集），并报告与多个事件重叠的报警段数量。
- 事件的子系统集合 = 事件内所有秒的 attack_P* 并集。并集为空的事件不计入命中率分母，单列为"无子系统标签事件"。
- 单独统计 Top-1 落在 P4 的比例（P4 是 HIL 仿真，没有对应的子系统标签，Top-1 落在 P4 必然不命中）。
- 不与攻击重叠的报警段单列，不计入命中率。
- 传感器级 Hit@k 暂不做：HAI 21.03 的 CSV 没有逐传感器的攻击目标；拿到官方的攻击目标表后再补。
- 按事件的命中率（2026-10-09 定）：主口径取每个事件的第一个与之重叠的报警段（运维人员最先看到的那一条），
  看它的 Top-1 所属子系统是否在事件的子系统集合内；补充口径把事件区间内全部报警秒的误差合并后取 Top-1。
- 所有命中率附 95% Wilson 置信区间；另按报警段长度分组（ml.attribution.duration_bins_s）给出命中率。
- 多随机种子（ml.seed_study）：对每个种子的运行各算一遍，报告均值 ± 样本标准差。
- 探索性分析（事后分析，不替换主结果）：段内误差先逐传感器除以它在 val_thr 上的平均误差再排 Top-k。
  基线只来自 val_thr（运行目录的 val_sensor_contrib.csv），这一步没有用到任何测试标签；标签只用于事后评估命中率。

输出（正式：reports/main/；--smoke：产物目录/smoke_reports/）：
    attribution.csv            每个报警段一行（全部参与比较的模型）
    attribution_summary.csv/.md  每个模型一行：命中率、P4 比例、各类报警段数量
    attribution_heatmap_<模型>.png  每个被检出的事件在各子系统上的误差占比
    attribution_by_duration.csv  按报警段长度分组的命中率（主结果与探索性分析各一组）
    attribution_seeds.csv        多随机种子的逐种子命中率与均值、标准差
    attribution_exploratory_valnorm.csv  探索性分析：按 val_thr 平均误差归一化前后的命中率

用法：
    python -m src.attribution [--models lstm_ae lstm_vae pca] [--smoke]
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.diagnose import CONTRIB_FILE
from src.eval.alarms import alarm_segments, normalize_shares, sensor_shares, subsystem_of, subsystem_shares, top_k
from src.eval.labels import load_test_labels
from src.eval.run_io import load_run
from src.eval.metrics import wilson_interval
from src.eval.segments import overlaps, to_segments
from src.utils.i18n import t
from src.utils.paths import load_config
from src.utils.reporting import ReportSource, add_common_args

SUMMARY_COLUMNS = ["model", "n_alarm_segments", "n_overlap_attack", "n_evaluable", "n_hit", "subsystem_hit_rate",
                   "subsystem_hit_rate_ci_low", "subsystem_hit_rate_ci_high",
                   "n_multi_event_segments", "n_evaluable_union", "n_hit_union", "subsystem_hit_rate_union",
                   "n_top1_p4", "top1_p4_ratio", "n_overlap_no_subsystem_label", "n_non_overlap",
                   "n_events", "n_events_without_subsystem", "n_events_attributed",
                   "n_events_first_evaluable", "n_events_first_hit", "event_first_hit_rate",
                   "event_first_hit_rate_ci_low", "event_first_hit_rate_ci_high",
                   "n_events_merged_hit", "event_hit_rate", "event_hit_rate_ci_low", "event_hit_rate_ci_high"]
RATE_COLUMNS = ("subsystem_hit_rate", "event_first_hit_rate", "event_hit_rate")  # 多种子汇总时取均值 ± 标准差的列


def label_events(index: dict, labels: dict[str, pd.DataFrame], label_cols: list[str]) -> list[dict]:
    """各段内的攻击事件及其子系统集合。part 为 test_index.json 中的第几段，start / end 为段内行号。"""
    main_col, sub_cols = label_cols[0], label_cols[1:]
    events = []
    for k, part in enumerate(index["files"]):
        df = labels[part["file"]].iloc[part["start_row"]:part["start_row"] + part["rows"]]
        if len(df) != part["rows"]:
            raise ValueError(f"{part['file']} 的标签行数与分数对不上")
        for a, b in to_segments(df[main_col].to_numpy(), 0):
            block = df.iloc[a:b + 1]
            subs = sorted(c.split("_", 1)[1] for c in sub_cols if block[c].any())
            events.append({"event_id": len(events) + 1, "part": k, "file": part["file"], "start": a, "end": b,
                           "subsystems": subs})
    return events


def attribute_run(scores: np.ndarray, per_sensor_err: np.ndarray, index: dict, threshold: float, merge_gap_s: int,
                  events: list[dict], ks: list[int], baseline: np.ndarray | None = None) -> tuple[list[dict], dict, list[dict]]:
    """返回 (每个报警段的归因与评估, 汇总, 每个事件在各子系统上的误差占比)。

    baseline 只用于探索性分析：各传感器在 val_thr 上的平均误差，给出时误差先除以它再算占比。
    """
    features = index["features"]
    rows, event_err = [], {}
    for seg in alarm_segments(scores, index, threshold, merge_gap_s):
        shares = sensor_shares(per_sensor_err, seg["global_start"], seg["global_end"], baseline)
        tops = {k: top_k(shares, features, k) for k in ks}
        top1 = tops[ks[0]][0] if tops[ks[0]] else None
        hit_events = [e for e in events if e["part"] == seg["part"] and overlaps((seg["start"], seg["end"]), (e["start"], e["end"]))]
        # 主口径：重叠时间最长的事件（max 取第一个最大值，事件按时间排列，所以相同时取较早的）
        primary = max(hit_events, key=lambda e: min(seg["end"], e["end"]) - max(seg["start"], e["start"]), default=None)
        target = primary["subsystems"] if primary else []
        union = sorted({s for e in hit_events for s in e["subsystems"]})  # 补充口径
        top1_sub = subsystem_of(top1["sensor"]) if top1 else None
        if not hit_events:
            category, hit = "non_overlap", None
        elif not target:
            category, hit = "overlap_no_subsystem_label", None
        else:
            category, hit = "evaluable", top1_sub in target
        rows.append({**{k: seg[k] for k in ("file", "start_time", "end_time", "duration_s", "max_score")},
                     "category": category, "event_ids": "+".join(str(e["event_id"]) for e in hit_events),
                     "n_events_overlap": len(hit_events), "primary_event_id": primary["event_id"] if primary else None,
                     "event_subsystems": "+".join(target), "union_subsystems": "+".join(union),
                     "top1_sensor": top1["sensor"] if top1 else None,
                     "top1_subsystem": top1_sub, "top1_share": top1["share"] if top1 else None, "hit": hit,
                     "hit_union": (top1_sub in union) if union else None,
                     **{f"top{k}": json.dumps(tops[k], ensure_ascii=False) for k in ks}})
        # 事件级：把与事件重叠的报警段内（事件区间之内的报警秒）的误差累加起来
        for e in hit_events:
            lo = index["files"][e["part"]]["offset"] + max(seg["start"], e["start"])
            hi = index["files"][e["part"]]["offset"] + min(seg["end"], e["end"])
            block = np.asarray(per_sensor_err[lo:hi + 1], dtype=np.float64)
            acc = event_err.setdefault(e["event_id"], [np.zeros(len(features)), 0])
            acc[0] += block.sum(axis=0)
            acc[1] += len(block)

    event_rows = []
    for e in events:
        if e["event_id"] not in event_err:
            continue
        total, n = event_err[e["event_id"]]
        shares = normalize_shares(total / max(n, 1), baseline)
        by_sub = subsystem_shares(shares, features)
        top = top_k(shares, features, 1)
        event_rows.append({"event_id": e["event_id"], "file": e["file"], "subsystems": "+".join(e["subsystems"]),
                           "alarm_seconds": n, "top1_sensor": top[0]["sensor"] if top else None,
                           "top1_subsystem": subsystem_of(top[0]["sensor"]) if top else None,
                           "hit": (subsystem_of(top[0]["sensor"]) in e["subsystems"]) if top and e["subsystems"] else None,
                           **{f"share_{s}": v for s, v in sorted(by_sub.items())}})

    overlap = [r for r in rows if r["category"] != "non_overlap"]
    evaluable = [r for r in rows if r["category"] == "evaluable"]
    n_hit = sum(bool(r["hit"]) for r in evaluable)
    union_eval = [r for r in rows if r["hit_union"] is not None]
    n_hit_union = sum(bool(r["hit_union"]) for r in union_eval)
    n_p4 = sum(r["top1_subsystem"] == "P4" for r in overlap)
    ev_eval = [r for r in event_rows if r["hit"] is not None]
    summary = {
        "n_alarm_segments": len(rows), "n_overlap_attack": len(overlap), "n_evaluable": len(evaluable), "n_hit": n_hit,
        "subsystem_hit_rate": n_hit / len(evaluable) if evaluable else float("nan"),
        "n_multi_event_segments": sum(r["n_events_overlap"] > 1 for r in rows),
        "n_evaluable_union": len(union_eval), "n_hit_union": n_hit_union,
        "subsystem_hit_rate_union": n_hit_union / len(union_eval) if union_eval else float("nan"),
        "n_top1_p4": n_p4, "top1_p4_ratio": n_p4 / len(overlap) if overlap else float("nan"),
        "n_overlap_no_subsystem_label": sum(r["category"] == "overlap_no_subsystem_label" for r in rows),
        "n_non_overlap": len(rows) - len(overlap),
        "n_events": len(events), "n_events_without_subsystem": sum(not e["subsystems"] for e in events),
        "n_events_attributed": len(event_rows),
        "event_hit_rate": sum(bool(r["hit"]) for r in ev_eval) / len(ev_eval) if ev_eval else float("nan"),
    }
    return rows, summary, event_rows


def add_first_segment(event_rows: list[dict], rows: list[dict]) -> list[dict]:
    """按事件的主口径：每个事件的第一个与之重叠的报警段（rows 按时间排列）及其 Top-1 是否命中。"""
    first: dict[int, dict] = {}
    for r in rows:
        for eid in (int(x) for x in r["event_ids"].split("+") if x):
            first.setdefault(eid, r)
    out = []
    for e in event_rows:
        r = first[e["event_id"]]
        subs = [x for x in e["subsystems"].split("+") if x]
        out.append({**e, "first_start_time": r["start_time"], "first_duration_s": r["duration_s"],
                    "first_top1_sensor": r["top1_sensor"], "first_top1_subsystem": r["top1_subsystem"],
                    "first_hit": (r["top1_subsystem"] in subs) if subs and r["top1_subsystem"] else None})
    return out


def rate_with_ci(hits: list) -> dict:
    """hits 中的 None 不计入分母。返回 n、n_hit、rate 与 95% Wilson 区间。"""
    valid = [bool(h) for h in hits if h is not None]
    n, k = len(valid), sum(valid)
    lo, hi = wilson_interval(k, n)
    return {"n": n, "n_hit": k, "rate": k / n if n else float("nan"), "ci_low": lo, "ci_high": hi}


def full_summary(summary: dict, rows: list[dict], event_rows: list[dict]) -> dict:
    """attribute_run 的汇总加上置信区间和按事件的两种口径。"""
    seg = rate_with_ci([r["hit"] for r in rows if r["category"] == "evaluable"])
    first = rate_with_ci([e["first_hit"] for e in event_rows])
    merged = rate_with_ci([e["hit"] for e in event_rows])
    return {**summary,
            "subsystem_hit_rate_ci_low": seg["ci_low"], "subsystem_hit_rate_ci_high": seg["ci_high"],
            "n_events_first_evaluable": first["n"], "n_events_first_hit": first["n_hit"], "event_first_hit_rate": first["rate"],
            "event_first_hit_rate_ci_low": first["ci_low"], "event_first_hit_rate_ci_high": first["ci_high"],
            "n_events_merged_hit": merged["n_hit"],
            "event_hit_rate_ci_low": merged["ci_low"], "event_hit_rate_ci_high": merged["ci_high"]}


def duration_group(duration_s: int, bins: list[int]) -> str:
    """bins = [10, 60] → "<10 秒"、"10–60 秒"（含两端）、">60 秒"。"""
    lo, hi = bins
    return f"<{lo} 秒" if duration_s < lo else f"{lo}–{hi} 秒" if duration_s <= hi else f">{hi} 秒"


def by_duration(rows: list[dict], bins: list[int]) -> list[dict]:
    """与攻击重叠且可评估的报警段，按长度分组的命中率。"""
    groups = [f"<{bins[0]} 秒", f"{bins[0]}–{bins[1]} 秒", f">{bins[1]} 秒"]
    evaluable = [r for r in rows if r["category"] == "evaluable"]
    return [{"duration_group": g,
             **rate_with_ci([r["hit"] for r in evaluable if duration_group(r["duration_s"], bins) == g])} for g in groups]


def val_thr_baseline(run_dir, features: list[str]) -> np.ndarray:
    """各传感器在 val_thr 上的平均平方误差（diagnose 阶段写出，只依据 val_thr），按 features 的顺序。"""
    df = pd.read_csv(run_dir / CONTRIB_FILE).set_index("sensor")
    return df.loc[features, "mean_sq_err"].to_numpy(dtype=np.float64)


def plot_heatmap(event_rows: list[dict], model: str, path, smoke: bool) -> None:
    """行 = 被检出的事件，列 = 子系统，颜色 = 该事件报警秒内误差落在各子系统的占比；圆点标出标签中的受攻击子系统。"""
    from src.utils.plotstyle import INK, SEQUENTIAL, SURFACE, apply_style, plt
    apply_style()
    df = pd.DataFrame(event_rows)
    cols = sorted(c for c in df.columns if c.startswith("share_"))
    values = df[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 3.2, 0.24 * len(df) + 1.6))
    im = ax.imshow(values, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(df.itertuples()):
        for j, col in enumerate(cols):
            if col.removeprefix("share_") in row.subsystems.split("+"):
                ax.plot(j, i, "o", ms=5, mfc=SURFACE, mec=INK, mew=1)
    ax.set_xticks(range(len(cols)), [c.removeprefix("share_") for c in cols])
    ax.set_yticks(range(len(df)), [f"{r.file.split('.')[0]} #{r.event_id}" for r in df.itertuples()], fontsize=7)
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(df) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)  # 相邻色块之间留出底色间隙
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).set_label(t("误差占比"))
    ax.set_title(t("{model}：被检出事件的误差在各子系统的占比").format(model=model) + f"{t('（冒烟数据，仅检查流程）') if smoke else ''}\n"
                 + t("○ = 标签中的受攻击子系统"), loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summary_md(df: pd.DataFrame, smoke: bool, duration: pd.DataFrame | None = None, seeds: pd.DataFrame | None = None,
               explore: pd.DataFrame | None = None) -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else f"{v:.4f}"
        return str(v)

    def table(frame: pd.DataFrame) -> list[str]:
        cols = list(frame.columns)
        return (["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                + ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)])

    lines = ["# 归因评估汇总" + ("（冒烟数据，仅检查流程，不得引用）" if smoke else ""), "",
             "每个报警段取段内各传感器重构误差均值的占比，Top-1 传感器所属子系统落在事件的子系统集合内即为命中。"
             "报警段与多个事件重叠时（n_multi_event_segments 个），主口径按重叠时间最长的那个事件判断"
             "（n_evaluable、n_hit、subsystem_hit_rate）；并集口径（Top-1 所属子系统落在这些事件子系统集合的并集内，"
             "n_evaluable_union、n_hit_union、subsystem_hit_rate_union）作为补充，只会不低于主口径的命中数。"
             "只有与攻击事件重叠、且事件有子系统标签的报警段计入命中率；"
             "不与攻击重叠的报警段（n_non_overlap）和只与无子系统标签事件重叠的报警段单列。"
             "top1_p4_ratio 是与攻击重叠的报警段中 Top-1 落在 P4 的比例（P4 没有子系统标签，落在 P4 必然不命中）。", "",
             "按事件的命中率有两种口径，分母都是被检出且有子系统标签的事件：", "",
             "- **主口径 event_first_hit_rate**：取每个事件的第一个与之重叠的报警段（按报警段开始时间），"
             "看这个报警段的 Top-1 所属子系统是否在事件的子系统集合内。它对应运维人员最先看到的那一条报警；"
             "报警段的误差按整段计算（报警可能在事件开始之前就已经开始）。",
             "- **补充口径 event_hit_rate**：把事件区间内全部报警秒的误差合并后取 Top-1。它用到了事件结束之前的全部报警，"
             "在报警刚出现时还得不到，所以只作为补充。", "",
             "各命中率后的 ci_low、ci_high 是 95% Wilson 置信区间。分母都很小，区间很宽，模型之间的差别要结合区间来读。", "",
             "传感器级 Hit@k 未做：HAI 21.03 的数据文件没有逐传感器的攻击目标标注。", "",
             *table(df[SUMMARY_COLUMNS])]
    if duration is not None and len(duration):
        lines += ["", "## 按报警段长度分组", "",
                  "只含与攻击重叠且可评估的报警段。variant = main 为主结果；valnorm 为下面的探索性分析。", "",
                  *table(duration)]
    if seeds is not None and len(seeds):
        lines += ["", "## 多随机种子", "",
                  "对 ml.seed_study 中每个种子的运行各算一遍归因（配置除种子外完全相同）；最后两行是均值和样本标准差（ddof=1）。"
                  "主结果仍以主模型运行（表中第一行的种子）为准。", "", *table(seeds)]
    if explore is not None and len(explore):
        lines += ["", "## 探索性分析：按 val_thr 平均误差归一化（事后分析，不替换主结果）", "",
                  "**这是看到主结果之后才做的分析。** 做法：报警段内各传感器的平均误差先除以该传感器在 val_thr 上的平均误差，"
                  "再归一化、排 Top-k，即比较的是「相对平时误差的倍数」。基线只来自 val_thr，没有用到测试标签；"
                  "但「要不要做这一步」是在看过测试集上的命中率之后才决定的，所以它的命中率是乐观的，"
                  "不能当作事先确定的方法的结果来引用。主结果、alerts.json 和告警卡片都不使用这种归一化。", "",
                  *table(explore)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="传感器归因与子系统命中率（M5）")
    add_common_args(p)
    p.add_argument("--models", nargs="+", help="默认取 config 的 ml.attribution.models")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    ml = cfg["ml"]
    src = ReportSource(cfg, args.smoke)
    src.banner()
    models = src.usable_runs(args.models or ml["attribution"]["models"],
                             required=("scores_test.npy", "per_sensor_err_test.npy", "threshold.json", "test_index.json"))
    if not models:
        print("[提示] 没有可用的运行，未生成归因结果")
        return 0
    labels = load_test_labels(cfg)
    attr, bins = ml["attribution"], ml["attribution"]["duration_bins_s"]

    def analyse(run, baseline=None):
        events = label_events(run.index, labels, ml["hai"]["label_cols"])
        rows, summary, event_rows = attribute_run(run.scores, run.per_sensor_err, run.index, run.threshold["threshold"],
                                                  ml["eval"]["merge_gap_s"], events, attr["top_k"], baseline)
        event_rows = add_first_segment(event_rows, rows)
        return rows, full_summary(summary, rows, event_rows), event_rows

    all_rows, summaries, duration_rows, explore_rows = [], [], [], []
    for name in models:
        run = load_run(src.root / name, per_sensor=True)
        rows, summary, event_rows = analyse(run)
        all_rows += [{"model": name, **r} for r in rows]
        summaries.append({"model": name, **summary})
        duration_rows += [{"model": name, "variant": "main", **d} for d in by_duration(rows, bins)]
        if event_rows:
            plot_heatmap(event_rows, name, src.out / f"attribution_heatmap_{name}.png", args.smoke)
            pd.DataFrame(event_rows).to_csv(src.out / f"attribution_events_{name}.csv", index=False)
        print(f"[归因] {name}: 报警段 {summary['n_alarm_segments']}，与攻击重叠 {summary['n_overlap_attack']}，"
              f"可评估 {summary['n_evaluable']}，命中 {summary['n_hit']}（并集口径 {summary['n_hit_union']}/{summary['n_evaluable_union']}），"
              f"与多个事件重叠 {summary['n_multi_event_segments']}，Top-1 在 P4 {summary['n_top1_p4']}，"
              f"无子系统标签事件 {summary['n_events_without_subsystem']}；"
              f"按事件：第一个报警段 {summary['n_events_first_hit']}/{summary['n_events_first_evaluable']}，"
              f"合并 {summary['n_events_merged_hit']}/{summary['n_events_first_evaluable']}")

        # 探索性分析（事后）：按 val_thr 平均误差归一化
        if attr["exploratory_val_norm"] and (run.directory / CONTRIB_FILE).exists():
            n_rows, n_summary, n_events = analyse(run, val_thr_baseline(run.directory, run.features))
            duration_rows += [{"model": name, "variant": "valnorm", **d} for d in by_duration(n_rows, bins)]
            changed = sum(a["top1_sensor"] != b["top1_sensor"] for a, b in zip(rows, n_rows) if a["category"] == "evaluable")
            for level, n_key, hit_key, rate in (("报警段", "n_evaluable", "n_hit", "subsystem_hit_rate"),
                                                ("事件（第一个报警段）", "n_events_first_evaluable", "n_events_first_hit", "event_first_hit_rate"),
                                                ("事件（合并）", "n_events_first_evaluable", "n_events_merged_hit", "event_hit_rate")):
                explore_rows.append({
                    "model": name, "level": level, "n": summary[n_key],
                    "n_hit_main": summary[hit_key], "rate_main": summary[rate],
                    "n_hit_valnorm": n_summary[hit_key], "rate_valnorm": n_summary[rate],
                    "rate_valnorm_ci_low": n_summary[f"{rate}_ci_low"], "rate_valnorm_ci_high": n_summary[f"{rate}_ci_high"],
                    "n_top1_p4_main": summary["n_top1_p4"], "n_top1_p4_valnorm": n_summary["n_top1_p4"],
                    "n_segments_top1_changed": changed})
            print(f"[探索性分析] {name}: 按 val_thr 归一化后报警段命中 {n_summary['n_hit']}/{n_summary['n_evaluable']}"
                  f"（主结果 {summary['n_hit']}/{summary['n_evaluable']}）；事后分析，不替换主结果")

    # 多随机种子：对每个种子的运行各算一遍（正式模式；种子不齐时跳过）
    seed_df = None
    study = ml["seed_study"]
    if not args.smoke and study["model"] in models:
        names = {seed: study["model"] if seed == ml["seed"] else f"{study['run_subdir']}/{study['model']}_s{seed}"
                 for seed in study["seeds"]}
        usable = src.usable_runs(list(names.values()), required=("scores_test.npy", "per_sensor_err_test.npy",
                                                                  "threshold.json", "test_index.json"))
        if len(usable) == len(names):
            seed_rows = []
            for seed, name in names.items():
                _, summary, _ = analyse(load_run(src.root / name, per_sensor=True))
                seed_rows.append({"seed": seed, "run_name": name,
                                  **{k: summary[k] for k in ("n_alarm_segments", "n_evaluable", "n_hit", "subsystem_hit_rate",
                                                             "n_events_first_evaluable", "n_events_first_hit",
                                                             "event_first_hit_rate", "n_events_merged_hit", "event_hit_rate",
                                                             "n_top1_p4", "top1_p4_ratio")}})
            seed_df = pd.DataFrame(seed_rows)
            numeric = seed_df.drop(columns=["seed", "run_name"]).astype(float)
            seed_df = pd.concat([seed_df.astype({"seed": object}),
                                 pd.DataFrame([{"seed": "mean", "run_name": "", **numeric.mean().to_dict()},
                                               {"seed": "std", "run_name": "", **numeric.std(ddof=1).to_dict()}])],
                                ignore_index=True)
            seed_df.to_csv(src.out / "attribution_seeds.csv", index=False)
            print(f"[多种子] {study['model']}（{len(names)} 个种子）："
                  + "，".join(f"{c} {numeric[c].mean():.4f} ± {numeric[c].std(ddof=1):.4f}" for c in RATE_COLUMNS))
        else:
            print(f"[提示] 多随机种子的运行不齐（{len(usable)}/{len(names)}），未生成 attribution_seeds.csv")

    pd.DataFrame(all_rows).to_csv(src.out / "attribution.csv", index=False)
    sdf = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    sdf.to_csv(src.out / "attribution_summary.csv", index=False)
    ddf = pd.DataFrame(duration_rows)
    ddf.to_csv(src.out / "attribution_by_duration.csv", index=False)
    edf = pd.DataFrame(explore_rows)
    if len(edf):
        edf.to_csv(src.out / "attribution_exploratory_valnorm.csv", index=False)
    (src.out / "attribution_summary.md").write_text(summary_md(sdf, args.smoke, ddf, seed_df, edf), encoding="utf-8")
    print(f"[生成] {src.out}/attribution.csv、attribution_summary.csv/.md、attribution_by_duration.csv、attribution_heatmap_*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
