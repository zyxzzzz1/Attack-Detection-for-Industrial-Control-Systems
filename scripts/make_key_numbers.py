"""R1：把写报告要引用的主线关键数字汇总到 reports/main/key_numbers.md（英文）和 key_numbers.zh.md（中文），
按"检测 / 归因 / 融合 / 数据问题"四组，每个数字标明口径和来源文件。数字全部从来源文件读出，不手工抄写；
PROGRESS.md 顶部的"关键数字"是中文版的副本。

两种语言由同一次读数生成：每一行的数字只格式化一次，再填进中文或英文的句子模板（L(中文, 英文)），
所以两个文件的数字不会各写各的；测试逐行核对两种语言的数字相同，并核对英文版与来源表一致。

来源全部在 reports/main/ 下（入库）：各张结果表，以及 runs/ 中各次集群完整运行的 eval_metrics.json、run_manifest.json、
threshold.json、preprocess.json、train_log.csv 的入库副本。本脚本先调用 scripts.snapshot_runs 把这些小文件从产物目录
脱敏后复制过去（只接受非冒烟、git_dirty=false、已完成的运行；主机名等替换为占位符，原件不动），再从入库副本读数。
本脚本不读取标签：与标签有关的数字都来自 evaluate / attribution 已经写出的结果。

用法：
    python -m scripts.make_key_numbers
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.fusion.prioritize import severity
from scripts.snapshot_runs import SNAPSHOT_SUBDIR, snapshot
from src.utils.paths import DEFAULT_CONFIG, load_config, resolve
from src.utils.reporting import report_problem

MODELS = ["pca", "iforest", "lstm_ae", "lstm_vae", "deep_svdd"]
OUTPUTS = {"en": "key_numbers.md", "zh": "key_numbers.zh.md"}
# 来源表里的中文取值（归因脚本写出）在英文版中的写法
LEVELS_EN = {"报警段": "alarm segments", "事件（第一个报警段）": "events (first segment)", "事件（合并）": "events (merged)"}


def pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


class Table:
    def __init__(self, title: str, header: str) -> None:
        self.lines = [f"### {title}", "", header, "|---|---|---|"]

    def add(self, value: str, scope: str, source: str) -> None:
        self.lines.append(f"| {value} | {scope} | {source} |")

    def text(self) -> list[str]:
        return [*self.lines, ""]


def run_json(root: Path, run: str, name: str) -> dict:
    problem = report_problem(root / run)
    if problem:
        raise ValueError(f"{run} 不能用于报告：{problem}")
    return json.loads((root / run / name).read_text(encoding="utf-8"))


def build(cfg: dict, lang: str = "en") -> str:
    if lang not in OUTPUTS:
        raise ValueError(f"lang 只能是 {' / '.join(OUTPUTS)}，现在是 {lang!r}")
    zh = lang == "zh"

    def L(zh_text: str, en_text: str) -> str:
        return zh_text if zh else en_text

    C, S = L("，", ", "), L("；", "; ")          # 并列项、分句的分隔符
    header = L("| 数字 | 口径 | 来源 |", "| Number | Definition and scope | Source |")
    ml = cfg["ml"]
    rep = resolve(ml["report_dir"])
    root = rep / SNAPSHOT_SUBDIR          # 各次运行的小文件的入库副本
    art = f"{ml['report_dir']}/{SNAPSHOT_SUBDIR}"
    main = pd.read_csv(rep / "main_results.csv").set_index("model")
    seeds = pd.read_csv(rep / "main_results_seeds.csv").set_index("seed")
    n_seeds = len(seeds) - 2
    evals = {m: run_json(root, m, "eval_metrics.json") for m in MODELS}
    n_events = evals["lstm_ae"]["event"]["n_events"]

    # ---------- 检测 ----------
    det = Table(L("检测", "Detection"), header)
    det.add(L("**不使用 point-adjust**；阈值 = val_thr 分数的 99% 分位数；报警规则为分数 > 阈值",
              "**No point-adjust**; threshold = 99th percentile of the val_thr scores; an alarm is raised when score > threshold"),
            L("全部检测指标的共同口径；逐点 F1 偏低是预期结果",
              "Common to all detection metrics; a low point-wise F1 is the expected result"),
            L("CLAUDE.md 约束 3、4，决策 6b", "Fixed in advance: project constraints 3, 4 and decision 6b"))
    for m in MODELS:
        r, ev = main.loc[m], evals[m]["event"]
        det.add(L("{m}：precision {p}，recall {r}，F1 {f1}，AUROC {auc}；事件 {d}/{n}；延迟中位数 {dl} 秒；"
                  "误报 {fa} 次/小时（{seg} 段）",
                  "{m}: precision {p}, recall {r}, F1 {f1}, AUROC {auc}; events {d}/{n}; median delay {dl} s; "
                  "false alarms {fa} per hour ({seg} segments)").format(
                      m=m, p=f"{r.precision:.4f}", r=f"{r.recall:.4f}", f1=f"{r.f1:.4f}", auc=f"{r.auroc:.4f}",
                      d=ev["n_detected"], n=ev["n_events"], dl=f"{r.delay_median_s:.0f}",
                      fa=f"{r.false_alarms_per_hour:.2f}", seg=ev["n_false_alarm_segments"]),
                L("单个随机种子（{seed}）；逐点指标排除未打分的 {u} 秒；集群完整运行",
                  "Single random seed ({seed}); point-wise metrics exclude the {u} unscored seconds; full cluster run").format(
                      seed=ml["seed"], u=int(r.n_unscored)),
                "`reports/main/main_results.csv`")
    mean, std = seeds.loc["mean"], seeds.loc["std"]
    det.add(L("**LSTM-AE {n} 个种子**：F1 **{f1}**，AUROC **{auc}**，事件检出率 **{er}**（平均 {avg}/{ne}），"
              "误报 **{fa}** 次/小时；precision {p}，recall {r}",
              "**LSTM-AE, {n} seeds**: F1 **{f1}**, AUROC **{auc}**, event detection rate **{er}** ({avg}/{ne} on average), "
              "false alarms **{fa}** per hour; precision {p}, recall {r}").format(
                  n=n_seeds, f1=pm(mean.f1, std.f1), auc=pm(mean.auroc, std.auroc),
                  er=pm(mean.event_recall, std.event_recall), avg=f"{mean.event_recall * n_events:.1f}", ne=n_events,
                  fa=pm(mean.false_alarms_per_hour, std.false_alarms_per_hour, 2),
                  p=pm(mean.precision, std.precision), r=pm(mean.recall, std.recall)),
            L("主配置 W=60、hidden=64；种子 {s}；均值 ± 样本标准差（ddof=1）。报告主表中 LSTM-AE 一行用这组数；其余模型只有单种子",
              "Main configuration W=60, hidden=64; seeds {s}; mean ± sample standard deviation (ddof=1). "
              "The LSTM-AE row of the main table uses these numbers; the other models have a single seed").format(
                  s=", ".join(str(s) for s in seeds.index[:n_seeds])),
            L("`reports/main/main_results_seeds.csv`、`main_results.md`", "`reports/main/main_results_seeds.csv`, `main_results.md`"))
    f1_if = main.loc["iforest", "f1"]
    n_above = int((seeds.iloc[:n_seeds]["f1"].astype(float) > f1_if).sum())
    det.add(L("LSTM-AE 与 IsolationForest 的逐点 F1：{a} 对 {b}，相差 {d} = {sd} 个标准差；{n} 个种子中 {k} 个高于 IsolationForest",
              "LSTM-AE vs IsolationForest, point-wise F1: {a} vs {b}, a difference of {d} = {sd} standard deviations; "
              "{k} of the {n} seeds are above IsolationForest").format(
                  a=f"{mean.f1:.4f}", b=f"{f1_if:.4f}", d=f"{f1_if - mean.f1:.4f}", sd=f"{(f1_if - mean.f1) / std.f1:.2f}",
                  n=n_seeds, k=n_above),
            L("差距在随机波动范围内；IsolationForest 自己的种子波动没有评估",
              "The difference is within random variation; the seed variation of IsolationForest itself was not evaluated"),
            L("同上", "Same as above"))
    rows = [f"{m} {100 * (e['point']['tp'] + e['point']['fp']) / e['point']['n_points']:.1f}%" for m, e in evals.items()]
    det.add(L("test 上报警的秒占比：", "Share of test seconds raising an alarm: ") + C.join(rows),
            L("已打分的秒中分数 > 阈值的比例；val_thr 上按阈值的定义为 1%。所有模型都偏高（阈值迁移问题）",
              "Share of scored seconds with score > threshold; 1% on val_thr by the definition of the threshold. "
              "Too high for every model (threshold transfer problem)"),
            f"`{art}/<{L('模型', 'model')}>/eval_metrics.json`")
    sub = pd.read_csv(rep / "per_subsystem_detection.csv")
    for m in ("iforest", "lstm_ae"):
        g = sub[sub.model == m].set_index("subsystem")
        det.add(m + L(" 分子系统事件检出：", " event detection by subsystem: ")
                + C.join(f"{s} {int(g.loc[s, 'n_detected'])}/{int(g.loc[s, 'n_events'])}" for s in ("P1", "P2", "P3")),
                L("事件涉及多个子系统时在每个子系统下各计一次", "An event involving several subsystems is counted once under each"),
                "`reports/main/per_subsystem_detection.csv`")
    abl = pd.read_csv(rep / "ablation.csv")
    order = abl.sort_values("f1", ascending=False).reset_index(drop=True)
    rank = int(order.index[order["is_main_config"]][0]) + 1
    best, worst = order.iloc[0], order.iloc[-1]
    det.add(L("消融 9 个组合：F1 {f_lo}~{f_hi}，AUROC {a_lo}~{a_hi}，误报 {fa_lo}~{fa_hi} 次/小时；"
              "最高 W={bw}、hidden={bh}（F1 {bf}），最低 W={ww}、hidden={wh}（F1 {wf}）；主模型配置按 F1 排第 {rank}",
              "Ablation, 9 combinations: F1 {f_lo}–{f_hi}, AUROC {a_lo}–{a_hi}, false alarms {fa_lo}–{fa_hi} per hour; "
              "best W={bw}, hidden={bh} (F1 {bf}), worst W={ww}, hidden={wh} (F1 {wf}); "
              "the main configuration has rank {rank} by F1").format(
                  f_lo=f"{abl.f1.min():.3f}", f_hi=f"{abl.f1.max():.3f}", a_lo=f"{abl.auroc.min():.3f}",
                  a_hi=f"{abl.auroc.max():.3f}", fa_lo=f"{abl.false_alarms_per_hour.min():.2f}",
                  fa_hi=f"{abl.false_alarms_per_hour.max():.2f}", bw=int(best.window), bh=int(best.hidden),
                  bf=f"{best.f1:.3f}", ww=int(worst.window), wh=int(worst.hidden), wf=f"{worst.f1:.3f}", rank=rank),
            L("LSTM-AE，W ∈ {30, 60, 120} × hidden ∈ {32, 64, 128}，单种子；只作敏感性分析，不据此修改主模型（约束 9）",
              "LSTM-AE, W ∈ {30, 60, 120} × hidden ∈ {32, 64, 128}, single seed; sensitivity analysis only, "
              "the main model is not changed because of it (constraint 9)"),
            L("`reports/main/ablation.csv`、`ablation.md`、`ablation_heatmap.png`",
              "`reports/main/ablation.csv`, `ablation.md`, `ablation_heatmap.png`"))
    q = pd.read_csv(rep / "ablation_quantiles.csv")
    w, h = ml["window"]["length"], ml["models"]["lstm_ae"]["hidden"]
    mq = q[(q.window == w) & (q.hidden == h)].set_index("quantile")
    top = q[q["quantile"] == q["quantile"].max()]
    item = L("{k}% 分位 F1 {f}、事件检出率 {e}", "{k}th percentile F1 {f}, event detection rate {e}")
    det.add(L("阈值分位敏感性（主模型配置）：", "Threshold percentile sensitivity (main configuration): ")
            + L("，", "; ").join(item.format(k=f"{100 * k:g}", f=f"{v.f1:.3f}", e=f"{v.event_recall:.2f}") for k, v in mq.iterrows())
            + L("；{k}% 分位下 9 个组合的 F1 为 {lo}~{hi}",
                "; at the {k}th percentile the F1 of the 9 combinations is {lo}–{hi}").format(
                    k=f"{100 * q['quantile'].max():g}", lo=f"{top.f1.min():.2f}", hi=f"{top.f1.max():.2f}"),
            L("只作敏感性分析，不据此选阈值；主表仍用 99% 分位",
              "Sensitivity analysis only, not used to choose the threshold; the main table still uses the 99th percentile"),
            "`reports/main/ablation_quantiles.csv`")
    for m in ("lstm_ae", "lstm_vae", "deep_svdd"):
        tr = run_json(root, m, "run_manifest.json")["stages"]["train"]
        early = (L("第 {e} 个 epoch 早停", "early stopping at epoch {e}").format(e=tr["early_stop_epoch"])
                 if tr["early_stop_epoch"] else L("未触发早停", "early stopping not triggered"))
        det.add(L("{m} 训练：{fe} 个 epoch，最好第 {be} 个，{early}；每个 epoch 平均 {sec} 秒；参数 {npar} 个",
                  "{m} training: {fe} epochs, best epoch {be}, {early}; {sec} s per epoch on average; {npar} parameters").format(
                      m=m, fe=tr["final_epoch"], be=tr["best_epoch"], early=early,
                      sec=f"{tr['epoch_seconds_mean']:.2f}", npar=f"{tr['n_parameters']:,}"),
                L("集群 NVIDIA L4，种子 42；max_epochs=50、patience=5 事先固定",
                  "Cluster NVIDIA L4, seed 42; max_epochs=50, patience=5 fixed in advance"),
                f"`{art}/{m}/run_manifest.json`" + L("、`train_log.csv`；图 ", ", `train_log.csv`; figure ")
                + "`reports/main/training_curves.png`")
    c = run_json(root, "deep_svdd", "run_manifest.json")["deep_svdd_center"]
    thr = run_json(root, "deep_svdd", "threshold.json")
    det.add(L("Deep SVDD 中心：替换前 \\|c\\| 最小 {lo}、中位数 {med}、最大 {hi}；{nr}/{dim} 维被替换为 ±{eps}。"
              "坍塌判据 collapsed={col}（score_cv {cv}，被 1 个极端窗口拉高）；val_thr 分数中位数 {smed}，阈值 {thr}",
              "Deep SVDD centre: before replacement \\|c\\| min {lo}, median {med}, max {hi}; {nr}/{dim} dimensions "
              "replaced by ±{eps}. Collapse check collapsed={col} (score_cv {cv}, inflated by 1 extreme window); "
              "val_thr score median {smed}, threshold {thr}").format(
                  lo=f"{c['abs_c_before_min']:.5f}", med=f"{c['abs_c_before_median']:.4f}", hi=f"{c['abs_c_before_max']:.4f}",
                  nr=c["n_replaced"], dim=c["rep_dim"], eps=c["eps"], col=str(thr["collapsed"]).lower(),
                  cv=f"{thr['score_cv']:.0f}", smed=f"{thr['score_stats']['median']:.2g}", thr=f"{thr['threshold']:.2g}"),
            L("中心 = 初始网络在全部训练窗口上输出的均值，接近 0 的分量设为 ±eps（决策 5、6b）",
              "Centre = mean output of the initial network over all training windows, with components close to 0 "
              "set to ±eps (decisions 5, 6b)"),
            f"`{art}/deep_svdd/run_manifest.json`" + L("、", ", ") + "`threshold.json`")
    contrib = pd.read_csv(rep / "val_sensor_contrib.csv").iloc[0]
    det.add(L("PCA 在 val_thr 上占比最大的传感器：{s} {x}%（上限 {lim}%，未超限）",
              "Sensor with the largest share for PCA on val_thr: {s} {x}% (limit {lim}%, not exceeded)").format(
                  s=contrib.sensor, x=f"{100 * contrib.share:.1f}", lim=f"{100 * ml['diagnosis']['max_sensor_share']:.0f}"),
            L("只用 val_thr 的重尾诊断，在第一次评估 test 之前确定缩放方式（StandardScaler，不设 std 下限）",
              "Heavy-tail diagnostic on val_thr only; the scaling (StandardScaler, no lower bound on std) was fixed "
              "before the first evaluation on test"),
            L("`reports/main/val_sensor_contrib.csv`、`.png`", "`reports/main/val_sensor_contrib.csv`, `.png`"))

    # ---------- 归因 ----------
    att = Table(L("归因", "Attribution"), header)
    summ = pd.read_csv(rep / "attribution_summary.csv").set_index("model")
    seg = pd.read_csv(rep / "attribution.csv")
    for m, s in summ.iterrows():
        ov = seg[(seg.model == m) & (seg.category == "evaluable")]
        guess = ov["event_subsystems"].str.split("+").apply(lambda x: "P1" in x).mean()
        att.add(L("{m}：报警段 {seg}；事件·第一个报警段 {first}；事件·合并 {merged}；"
                  "同一批报警段上总是猜 P1 的命中率 {guess}；Top-1 在 P4 {p4}",
                  "{m}: alarm segments {seg}; events, first segment {first}; events, merged {merged}; "
                  "always guessing P1 on the same segments {guess}; top-1 in P4 {p4}").format(
                      m=m,
                      seg=f"{int(s.n_hit)}/{int(s.n_evaluable)} = {s.subsystem_hit_rate:.3f} "
                          f"{ci(s.subsystem_hit_rate_ci_low, s.subsystem_hit_rate_ci_high)}",
                      first=f"{int(s.n_events_first_hit)}/{int(s.n_events_first_evaluable)} = {s.event_first_hit_rate:.3f} "
                            f"{ci(s.event_first_hit_rate_ci_low, s.event_first_hit_rate_ci_high)}",
                      merged=f"{int(s.n_events_merged_hit)}/{int(s.n_events_first_evaluable)} = {s.event_hit_rate:.3f} "
                             f"{ci(s.event_hit_rate_ci_low, s.event_hit_rate_ci_high)}",
                      guess=f"{guess:.3f}", p4=f"{int(s.n_top1_p4)}/{int(s.n_overlap_attack)}"),
                L("子系统命中 = Top-1 传感器所属子系统 ∈ 事件的子系统集合；只计与攻击重叠的报警段；方括号为 95% Wilson 区间。"
                  "按事件：主口径取事件的第一个重叠报警段，合并口径为补充。单种子",
                  "Subsystem hit = the subsystem of the top-1 sensor ∈ the event's subsystem set; only alarm segments "
                  "overlapping an attack are counted; brackets are 95% Wilson intervals. Per event: the main definition "
                  "takes the first alarm segment overlapping the event, the merged definition is supplementary. Single seed"),
                L("`reports/main/attribution_summary.csv`、`.md`、`attribution.csv`",
                  "`reports/main/attribution_summary.csv`, `.md`, `attribution.csv`"))
    first = summ.iloc[0]
    att.add(L("与多个事件重叠的报警段 {multi} 个；无子系统标签事件 {nosub} 个；不与攻击重叠的报警段：",
              "Alarm segments overlapping several events: {multi}; events without subsystem labels: {nosub}; "
              "alarm segments not overlapping an attack: ").format(
                  multi=int(summ.n_multi_event_segments.max()), nosub=int(first.n_events_without_subsystem))
            + C.join(f"{m} {int(s.n_non_overlap)}" for m, s in summ.iterrows()),
            L("主口径与并集口径因此完全相同；不与攻击重叠的报警段不计入命中率",
              "The main and union definitions are therefore identical; alarm segments not overlapping an attack "
              "do not count towards the hit rate"),
            "`reports/main/attribution_summary.csv`")
    aseed = pd.read_csv(rep / "attribution_seeds.csv").set_index("seed")
    am, asd = aseed.loc["mean"], aseed.loc["std"]
    att.add(L("**LSTM-AE {n} 个种子**：报警段 **{seg}**，事件·第一个报警段 **{first}**，事件·合并 {merged}；Top-1 在 P4 的比例 {p4}",
              "**LSTM-AE, {n} seeds**: alarm segments **{seg}**, events, first segment **{first}**, events, merged {merged}; "
              "share of top-1 in P4 {p4}").format(
                  n=len(aseed) - 2, seg=pm(am.subsystem_hit_rate, asd.subsystem_hit_rate, 3),
                  first=pm(am.event_first_hit_rate, asd.event_first_hit_rate, 3),
                  merged=pm(am.event_hit_rate, asd.event_hit_rate, 3), p4=pm(am.top1_p4_ratio, asd.top1_p4_ratio, 3)),
            L("每个种子的运行各算一遍归因；均值 ± 样本标准差。LSTM-VAE、PCA 只有单种子",
              "Attribution computed separately for each seed's run; mean ± sample standard deviation. "
              "LSTM-VAE and PCA have a single seed"),
            "`reports/main/attribution_seeds.csv`")
    dur = pd.read_csv(rep / "attribution_by_duration.csv")
    for m in summ.index:
        g = dur[(dur.model == m) & (dur.variant == "main")]
        att.add(m + L(" 按报警段长度：", " by alarm segment length: ") + S.join(
            f"{r.duration_group if zh else r.duration_group.replace(' 秒', ' s')} {int(r.n_hit)}/{int(r.n)}"
            + (f" = {r.rate:.3f} {ci(r.ci_low, r.ci_high)}" if r.n else "") for r in g.itertuples()),
                L("与攻击重叠的报警段按长度分组；10–60 秒含两端。短报警段归因明显更差（LSTM-AE）",
                  "Alarm segments overlapping an attack, grouped by length; 10–60 s inclusive at both ends. "
                  "Short alarm segments are attributed clearly worse (LSTM-AE)"),
                "`reports/main/attribution_by_duration.csv`")
    exp = pd.read_csv(rep / "attribution_exploratory_valnorm.csv")
    for m in summ.index:
        g = exp[exp.model == m].set_index("level")
        att.add(L("（探索性）{m} 按 val_thr 归一化后：", "(Exploratory) {m} after normalising by val_thr: ").format(m=m) + S.join(
            f"{lv if zh else LEVELS_EN[lv]} {int(r.n_hit_main)}/{int(r.n)} → {int(r.n_hit_valnorm)}/{int(r.n)}"
            for lv, r in g.iterrows()),
                L("**事后分析，不替换主结果**：段内误差先除以各传感器在 val_thr 上的平均误差再排 Top-k；"
                  "基线只来自 val_thr，但这一步是看过测试结果后才做的，数字偏乐观",
                  "**Post-hoc analysis, does not replace the main results**: within-segment errors are divided by each "
                  "sensor's mean error on val_thr before ranking the top-k; the baseline comes from val_thr only, but "
                  "this step was done after seeing the test results, so the numbers are optimistic"),
                "`reports/main/attribution_exploratory_valnorm.csv`")
    att.add(L("传感器级 Hit@k：未做", "Sensor-level Hit@k: not computed"),
            L("HAI 21.03 的数据文件没有逐传感器的攻击目标，官方攻击目标表未取得；写进局限性",
              "The HAI 21.03 data files have no per-sensor attack targets and the official attack target table was "
              "not obtained; listed under limitations"),
            "`reports/main/attribution_summary.md`")

    # ---------- 融合 ----------
    fus = Table(L("融合（原型演示；不声称提升检测指标，约束 7）",
                  "Fusion (prototype demonstration; no claim of improved detection metrics, constraint 7)"), header)
    alerts = json.loads(resolve(ml["alerts"]["output"]).read_text(encoding="utf-8"))
    fr = pd.read_csv(rep / "fusion_ranking_eval.csv")
    n_true = int(fr.n_true_total.iloc[0])
    fus.add(L("真实报警 {n} 条，其中 {k} 条与攻击事件重叠（基准比例 {base}）",
              "Real alarms: {n}, of which {k} overlap an attack event (base rate {base})").format(
                  n=len(alerts), k=n_true, base=f"{n_true / len(alerts):.3f}"),
            L("LSTM-AE 主模型运行（种子 42）在 test 上的全部报警段，不按标签筛选；is_mock=false",
              "All alarm segments of the main LSTM-AE run (seed 42) on test, not filtered by label; is_mock=false"),
            L("`data/processed/alerts.json`、`reports/main/fusion_ranking_eval.csv`",
              "`data/processed/alerts.json`, `reports/main/fusion_ranking_eval.csv`"))
    for name, g in fr.groupby("ranking", sort=False):
        k_max = g.iloc[-1]
        shared = (L("；前 {k} 条中 {n} 条与原始分数排序相同",
                    "; {n} of the top {k} are the same as in the raw-score ranking").format(
                        k=int(k_max.k), n=int(k_max.n_shared_with_raw_score)) if name != "raw_score" else "")
        fus.add(L("排序 {name}：", "Ranking {name}: ").format(name=name)
                + C.join(f"precision@{int(r.k)} = {int(r.n_true)}/{int(r.k)}" for r in g.itertuples())
                + L("；precision@{k} 的 95% 区间 {ci}", "; 95% interval of precision@{k} {ci}").format(
                    k=int(k_max.k), ci=ci(k_max.ci_low, k_max.ci_high))
                + shared,
                L("前 k 条报警中与攻击事件重叠的比例；raw_score 按最大异常分数，alpha=0 / alpha=1 为融合优先级。"
                  "三种排序没有可测量的差别：排序的作用来自异常分数，不来自漏洞上下文",
                  "Share of the top k alarms overlapping an attack event; raw_score ranks by maximum anomaly score, "
                  "alpha=0 / alpha=1 are fused priorities. No measurable difference between the three rankings: "
                  "the ranking effect comes from the anomaly score, not from the vulnerability context"),
                L("`reports/main/fusion_ranking_eval.csv`、`.md`", "`reports/main/fusion_ranking_eval.csv`, `.md`"))
    n_sat = sum(severity(a["max_score"], a["threshold"]) >= 1.0 for a in alerts)
    fus.add(L("S 封顶的报警 {n} 条", "Alarms with S capped: {n}").format(n=n_sat),
            L("S = min(1, 最大分数 ÷ 阈值 − 1)，分数达到阈值 2 倍即为 1；这些报警之间不再按分数区分。"
              "只写成局限和未来工作，不改公式",
              "S = min(1, max score ÷ threshold − 1), which equals 1 once the score reaches 2 times the threshold; "
              "these alarms are no longer separated by score. Recorded as a limitation and future work only, "
              "the formula is unchanged"),
            L("`data/processed/alerts.json`、`reports/main/cases.md`", "`data/processed/alerts.json`, `reports/main/cases.md`"))
    fus.add(L("卡片与演示只渲染优先级最高的前 {n} 条", "Cards and the demo render only the top {n} alarms by priority").format(
                n=cfg["fusion"]["max_rendered_alerts"]),
            L("融合打分仍对全部报警进行；条数为 `fusion.max_rendered_alerts`",
              "Fusion scoring still covers all alarms; the count is `fusion.max_rendered_alerts`"),
            "`config/config.yaml`")
    fus.add(L("3 个案例：归因正确的真实攻击、归因错误的真实攻击、高优先级的误报",
              "3 cases: a correctly attributed real attack, a wrongly attributed real attack, a high-priority false alarm"),
            L("按融合优先级从前往后各取第一条，不人工挑选；对照标签的说明来自事后评估",
              "For each type, the first alarm in descending fused priority, with no manual selection; "
              "the notes against the labels come from post-hoc evaluation"),
            L("`reports/main/cases.md`、`cases/`", "`reports/main/cases.md`, `cases/`"))
    fus.add(L("漏洞与资产侧的数字（540 条漏洞、KEV 命中 9 条、min_qod=70 后保留 66 条、E_sub 只有 0 / 1 两个取值）",
              "Vulnerability and asset numbers (540 vulnerabilities, 9 KEV hits, 66 kept after min_qod=70, "
              "E_sub takes only the values 0 / 1)"),
            L("真实扫描只有一台靶机 ews_01；资产映射为人为构造",
              "The real scan covers a single target, ews_01; the asset mapping is constructed by hand"),
            L("`reports/extension/key_numbers.md`、`kev_table.md`", "`reports/extension/key_numbers.md`, `kev_table.md`"))

    # ---------- 数据问题 ----------
    dat = Table(L("数据问题", "Data issues"), header)
    pre = run_json(root, "lstm_ae", "preprocess.json")
    drops = ml["data"]["train_overlap_drops"][0]
    dat.add(L("官方 train1 末尾 {n} 秒与 test3 开头 {n} 秒是同一份数据；另有 train1 第 {h} 行 = test2 最后 1 行。"
              "两处都已从训练集删除，test 保持原样",
              "The last {n} seconds of the official train1 and the first {n} seconds of test3 are the same data; "
              "in addition, row {h} of train1 = the last 1 row of test2. Both were removed from the training set, "
              "test is left as released").format(n=f"{drops['tail_rows']:,}", h=drops["head_rows"]),
            L("按 79 个传感器的逐行内容哈希比较全部文件对；这段数据没有攻击。不处理会低估 test3 上的误报",
              "All file pairs compared by a row-wise content hash over the 79 sensors; this data contains no attack. "
              "Left untreated, it would understate false alarms on test3"),
            L("`reports/main/data_summary.md` 第 3 节；`config/config.yaml` 的 `ml.data.train_overlap_drops`",
              "`reports/main/data_overlap.md`; `reports/main/data_summary.md` section 3; "
              "`ml.data.train_overlap_drops` in `config/config.yaml`"))
    r = pre["rows"]
    dat.add(L("划分：train {a} / val_es {b} / val_thr {c} 行；特征 {f} 个（删除 {k} 个常量列）",
              "Split: train {a} / val_es {b} / val_thr {c} rows; {f} features ({k} constant columns removed)").format(
                  a=f"{r['train']:,}", b=f"{r['val_es']:,}", c=f"{r['val_thr']:,}", f=pre["n_features"],
                  k=len(pre["constant_cols"])),
            L("删除重叠行之后按时间切分，最后 20% 对半分为 val_es（早停）和 val_thr（定阈值）；scaler 只在 train 部分 fit",
              "Split in time order after removing the overlapping rows; the last 20% is halved into val_es "
              "(early stopping) and val_thr (threshold); the scaler is fitted on the train part only"),
            f"`{art}/lstm_ae/preprocess.json`" + L("；`PROGRESS.md` 的 M1 一节", "; section M1 of `PROGRESS.md`"))
    p = evals["lstm_ae"]["point"]
    events = pd.read_csv(rep / "attack_events.csv")
    dat.add(L("test：{n} 秒，攻击 {a} 秒（{pct}%），{e} 个攻击事件（长度 {lo}~{hi} 秒）",
              "test: {n} seconds, {a} attack seconds ({pct}%), {e} attack events (length {lo}–{hi} s)").format(
                  n=f"{p['n_points'] + p['n_unscored']:,}", a=f"{p['n_positive']:,}",
                  pct=f"{100 * p['n_positive'] / (p['n_points'] + p['n_unscored']):.2f}", e=len(events),
                  lo=int(events.duration_s.min()), hi=int(events.duration_s.max())),
            L("事件 = attack=1 的连续区间", "Event = a maximal run of consecutive seconds with attack=1"),
            L("`reports/main/data_summary.md`、`attack_events.csv`", "`reports/main/data_summary.md`, `attack_events.csv`"))
    dat.add(L("未打分 {u} 秒（5 个文件 × {w} 秒），其中攻击秒 {k} 个",
              "Unscored: {u} seconds (5 files × {w} seconds), {k} of them attack seconds").format(
                  u=p["n_unscored"], w=ml["window"]["length"] - 1, k=p["n_positive_unscored"]),
            L("每个测试文件的前 W−1 秒没有分数（W=60），不参与任何指标；W=30 / 120 的消融组合为 145 / 595 秒",
              "The first W−1 seconds of each test file have no score (W=60) and enter no metric; "
              "for the W=30 / 120 ablation combinations it is 145 / 595 seconds"),
            L("`reports/main/main_results.csv` 的 n_unscored 列", "column n_unscored of `reports/main/main_results.csv`"))
    dat.add(L("attack=1 但三个子系统标签全为 0 的有 {n} 秒，分布在 {k} 个事件内部；按并集规则 50 个事件的子系统集合全部非空",
              "attack=1 with all three subsystem labels 0: {n} seconds, in the interior of {k} events; "
              "under the union rule the subsystem sets of all 50 events are non-empty").format(
                  n=int(events.no_sub_s.sum()), k=int((events.no_sub_s > 0).sum())),
            L("事件的子系统集合 = 事件内所有秒的 attack_P* 并集；检测指标只用 attack 列，不受影响",
              "Subsystem set of an event = union of attack_P* over all its seconds; detection metrics use only "
              "the attack column and are unaffected"),
            L("`reports/main/attack_events.csv`、`data_summary.md`", "`reports/main/attack_events.csv`, `data_summary.md`"))
    files = [f"{x['file'].split('.')[0]} {100 * (x['tp'] + x['fp']) / x['n_points']:.1f}%" for x in evals["lstm_ae"]["per_file"]]
    dat.add(L("LSTM-AE 各测试文件报警的秒占比：", "Share of seconds raising an alarm per test file, LSTM-AE: ") + C.join(files),
            L("val_thr 上为 1%。各文件差别很大，推测 test 的正常工况相对训练数据有漂移（未单独验证是哪些传感器）",
              "1% on val_thr. The files differ widely; the presumed cause is drift of normal operation in test "
              "relative to the training data (which sensors drift was not verified separately)"),
            f"`{art}/lstm_ae/eval_metrics.json`" + L("；图 `reports/main/scores_test*.png`、`score_distribution.png`",
                                                    "; figures `reports/main/scores_test*.png`, `score_distribution.png`"))
    dat.add(L("P4 没有子系统标签", "P4 has no subsystem label"),
            L("P4 是 HIL 仿真部分；Top-1 落在 P4 的归因必然计为未命中",
              "P4 is the HIL simulation part; an attribution whose top-1 falls in P4 necessarily counts as a miss"),
            L("`data/samples/hai_columns.txt`、决策 7", "`data/samples/hai_columns.txt`, decision 7"))

    head = L(
        ["## 关键数字（主线）", "",
         "写报告时直接引用。数字由 `python -m scripts.make_key_numbers` 从来源文件读出（本节是 `reports/main/key_numbers.zh.md` 的副本）；"
         "检测、消融、归因的数字全部来自集群上的完整运行（git_dirty=false），冒烟测试的数字没有进入这里。",
         "来源文件全部入库：`reports/main/runs/` 是各次集群运行的小文件（指标、manifest、阈值、预处理摘要、训练记录）的副本，"
         "由本脚本从产物目录复制，主机名等集群信息已替换为占位符。", ""],
        ["## Key numbers (main line)", "",
         "Quote these when writing the report. The numbers are read from the source files by "
         "`python -m scripts.make_key_numbers` (the same run writes the Chinese version, `key_numbers.zh.md`). "
         "All detection, ablation and attribution numbers come from full cluster runs (git_dirty=false); "
         "no smoke-test number appears here.",
         "All source files are tracked: `reports/main/runs/` holds copies of the small files of each cluster run "
         "(metrics, manifest, threshold, preprocessing summary, training log), copied from the artifact directory "
         "by this script with host names and other cluster details replaced by placeholders.", ""])
    return "\n".join(head + det.text() + att.text() + fus.text() + dat.text()).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="汇总主线关键数字到 reports/main/key_numbers.md（英文）和 key_numbers.zh.md（中文）")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    # 先把产物目录里各次运行的小文件脱敏后复制到入库的位置；产物目录里没有的运行沿用已入库的副本
    copied, skipped = snapshot(cfg)
    print(f"[复制] {len(copied)} 次运行的小文件（已脱敏）→ {resolve(cfg['ml']['report_dir']) / SNAPSHOT_SUBDIR}"
          + (f"；{len(skipped)} 次运行在产物目录中不可用，沿用已入库的副本" if skipped else ""))
    try:
        texts = {lang: build(cfg, lang) for lang in OUTPUTS}
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1
    for lang, text in texts.items():
        out = resolve(cfg["ml"]["report_dir"]) / OUTPUTS[lang]
        out.write_text(text, encoding="utf-8")
        print(f"[生成] {out}（{text.count(chr(10))} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
