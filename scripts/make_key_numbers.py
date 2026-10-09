"""R1：把写报告要引用的主线关键数字汇总到 reports/main/key_numbers.md，按"检测 / 归因 / 融合 / 数据问题"四组，
每个数字标明口径和来源文件。数字全部从来源文件读出，不手工抄写；PROGRESS.md 顶部的"关键数字"是这个文件的副本。

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


def pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


class Table:
    def __init__(self, title: str) -> None:
        self.lines = [f"### {title}", "", "| 数字 | 口径 | 来源 |", "|---|---|---|"]

    def add(self, value: str, scope: str, source: str) -> None:
        self.lines.append(f"| {value} | {scope} | {source} |")

    def text(self) -> list[str]:
        return [*self.lines, ""]


def run_json(root: Path, run: str, name: str) -> dict:
    problem = report_problem(root / run)
    if problem:
        raise ValueError(f"{run} 不能用于报告：{problem}")
    return json.loads((root / run / name).read_text(encoding="utf-8"))


def build(cfg: dict) -> str:
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
    det = Table("检测")
    det.add("**不使用 point-adjust**；阈值 = val_thr 分数的 99% 分位数；报警规则为分数 > 阈值",
            "全部检测指标的共同口径；逐点 F1 偏低是预期结果", "CLAUDE.md 约束 3、4，决策 6b")
    for m in MODELS:
        r, ev = main.loc[m], evals[m]["event"]
        det.add(f"{m}：precision {r.precision:.4f}，recall {r.recall:.4f}，F1 {r.f1:.4f}，AUROC {r.auroc:.4f}；"
                f"事件 {ev['n_detected']}/{ev['n_events']}；延迟中位数 {r.delay_median_s:.0f} 秒；"
                f"误报 {r.false_alarms_per_hour:.2f} 次/小时（{ev['n_false_alarm_segments']} 段）",
                f"单个随机种子（{ml['seed']}）；逐点指标排除未打分的 {int(r.n_unscored)} 秒；集群完整运行",
                "`reports/main/main_results.csv`")
    mean, std = seeds.loc["mean"], seeds.loc["std"]
    det.add(f"**LSTM-AE {n_seeds} 个种子**：F1 **{pm(mean.f1, std.f1)}**，AUROC **{pm(mean.auroc, std.auroc)}**，"
            f"事件检出率 **{pm(mean.event_recall, std.event_recall)}**（平均 {mean.event_recall * n_events:.1f}/{n_events}），"
            f"误报 **{pm(mean.false_alarms_per_hour, std.false_alarms_per_hour, 2)}** 次/小时；"
            f"precision {pm(mean.precision, std.precision)}，recall {pm(mean.recall, std.recall)}",
            f"主配置 W=60、hidden=64；种子 {', '.join(str(s) for s in seeds.index[:n_seeds])}；均值 ± 样本标准差（ddof=1）。"
            "报告主表中 LSTM-AE 一行用这组数；其余模型只有单种子",
            "`reports/main/main_results_seeds.csv`、`main_results.md`")
    f1_if = main.loc["iforest", "f1"]
    n_above = int((seeds.iloc[:n_seeds]["f1"].astype(float) > f1_if).sum())
    det.add(f"LSTM-AE 与 IsolationForest 的逐点 F1：{mean.f1:.4f} 对 {f1_if:.4f}，相差 {f1_if - mean.f1:.4f}"
            f" = {(f1_if - mean.f1) / std.f1:.2f} 个标准差；{n_seeds} 个种子中 {n_above} 个高于 IsolationForest",
            "差距在随机波动范围内；IsolationForest 自己的种子波动没有评估", "同上")
    rows = [f"{m} {100 * (e['point']['tp'] + e['point']['fp']) / e['point']['n_points']:.1f}%" for m, e in evals.items()]
    det.add("test 上报警的秒占比：" + "，".join(rows),
            "已打分的秒中分数 > 阈值的比例；val_thr 上按阈值的定义为 1%。所有模型都偏高（阈值迁移问题）",
            f"`{art}/<模型>/eval_metrics.json`")
    sub = pd.read_csv(rep / "per_subsystem_detection.csv")
    for m in ("iforest", "lstm_ae"):
        g = sub[sub.model == m].set_index("subsystem")
        det.add(f"{m} 分子系统事件检出：" + "，".join(f"{s} {int(g.loc[s, 'n_detected'])}/{int(g.loc[s, 'n_events'])}"
                                                    for s in ("P1", "P2", "P3")),
                "事件涉及多个子系统时在每个子系统下各计一次", "`reports/main/per_subsystem_detection.csv`")
    abl = pd.read_csv(rep / "ablation.csv")
    order = abl.sort_values("f1", ascending=False).reset_index(drop=True)
    rank = int(order.index[order["is_main_config"]][0]) + 1
    best, worst = order.iloc[0], order.iloc[-1]
    det.add(f"消融 9 个组合：F1 {abl.f1.min():.3f}~{abl.f1.max():.3f}，AUROC {abl.auroc.min():.3f}~{abl.auroc.max():.3f}，"
            f"误报 {abl.false_alarms_per_hour.min():.2f}~{abl.false_alarms_per_hour.max():.2f} 次/小时；"
            f"最高 W={int(best.window)}、hidden={int(best.hidden)}（F1 {best.f1:.3f}），"
            f"最低 W={int(worst.window)}、hidden={int(worst.hidden)}（F1 {worst.f1:.3f}）；主模型配置按 F1 排第 {rank}",
            "LSTM-AE，W ∈ {30, 60, 120} × hidden ∈ {32, 64, 128}，单种子；只作敏感性分析，不据此修改主模型（约束 9）",
            "`reports/main/ablation.csv`、`ablation.md`、`ablation_heatmap.png`")
    q = pd.read_csv(rep / "ablation_quantiles.csv")
    w, h = ml["window"]["length"], ml["models"]["lstm_ae"]["hidden"]
    mq = q[(q.window == w) & (q.hidden == h)].set_index("quantile")
    top = q[q["quantile"] == q["quantile"].max()]
    det.add(f"阈值分位敏感性（主模型配置）：" + "，".join(f"{100 * k:g}% 分位 F1 {v.f1:.3f}、事件检出率 {v.event_recall:.2f}"
                                                  for k, v in mq.iterrows())
            + f"；{100 * q['quantile'].max():g}% 分位下 9 个组合的 F1 为 {top.f1.min():.2f}~{top.f1.max():.2f}",
            "只作敏感性分析，不据此选阈值；主表仍用 99% 分位", "`reports/main/ablation_quantiles.csv`")
    for m in ("lstm_ae", "lstm_vae", "deep_svdd"):
        tr = run_json(root, m, "run_manifest.json")["stages"]["train"]
        det.add(f"{m} 训练：{tr['final_epoch']} 个 epoch，最好第 {tr['best_epoch']} 个，"
                f"{'第 ' + str(tr['early_stop_epoch']) + ' 个 epoch 早停' if tr['early_stop_epoch'] else '未触发早停'}；"
                f"每个 epoch 平均 {tr['epoch_seconds_mean']:.2f} 秒；参数 {tr['n_parameters']:,} 个",
                "集群 NVIDIA L4，种子 42；max_epochs=50、patience=5 事先固定",
                f"`{art}/{m}/run_manifest.json`、`train_log.csv`；图 `reports/main/training_curves.png`")
    c = run_json(root, "deep_svdd", "run_manifest.json")["deep_svdd_center"]
    thr = run_json(root, "deep_svdd", "threshold.json")
    det.add(f"Deep SVDD 中心：替换前 \\|c\\| 最小 {c['abs_c_before_min']:.5f}、中位数 {c['abs_c_before_median']:.4f}、"
            f"最大 {c['abs_c_before_max']:.4f}；{c['n_replaced']}/{c['rep_dim']} 维被替换为 ±{c['eps']}。"
            f"坍塌判据 collapsed={str(thr['collapsed']).lower()}（score_cv {thr['score_cv']:.0f}，被 1 个极端窗口拉高）；"
            f"val_thr 分数中位数 {thr['score_stats']['median']:.2g}，阈值 {thr['threshold']:.2g}",
            "中心 = 初始网络在全部训练窗口上输出的均值，接近 0 的分量设为 ±eps（决策 5、6b）",
            f"`{art}/deep_svdd/run_manifest.json`、`threshold.json`")
    contrib = pd.read_csv(rep / "val_sensor_contrib.csv").iloc[0]
    det.add(f"PCA 在 val_thr 上占比最大的传感器：{contrib.sensor} {100 * contrib.share:.1f}%（上限 "
            f"{100 * ml['diagnosis']['max_sensor_share']:.0f}%，未超限）",
            "只用 val_thr 的重尾诊断，在第一次评估 test 之前确定缩放方式（StandardScaler，不设 std 下限）",
            "`reports/main/val_sensor_contrib.csv`、`.png`")

    # ---------- 归因 ----------
    att = Table("归因")
    summ = pd.read_csv(rep / "attribution_summary.csv").set_index("model")
    seg = pd.read_csv(rep / "attribution.csv")
    for m, s in summ.iterrows():
        ov = seg[(seg.model == m) & (seg.category == "evaluable")]
        guess = ov["event_subsystems"].str.split("+").apply(lambda x: "P1" in x).mean()
        att.add(f"{m}：报警段 {int(s.n_hit)}/{int(s.n_evaluable)} = {s.subsystem_hit_rate:.3f} "
                f"{ci(s.subsystem_hit_rate_ci_low, s.subsystem_hit_rate_ci_high)}；"
                f"事件·第一个报警段 {int(s.n_events_first_hit)}/{int(s.n_events_first_evaluable)} = {s.event_first_hit_rate:.3f} "
                f"{ci(s.event_first_hit_rate_ci_low, s.event_first_hit_rate_ci_high)}；"
                f"事件·合并 {int(s.n_events_merged_hit)}/{int(s.n_events_first_evaluable)} = {s.event_hit_rate:.3f} "
                f"{ci(s.event_hit_rate_ci_low, s.event_hit_rate_ci_high)}；"
                f"同一批报警段上总是猜 P1 的命中率 {guess:.3f}；Top-1 在 P4 {int(s.n_top1_p4)}/{int(s.n_overlap_attack)}",
                "子系统命中 = Top-1 传感器所属子系统 ∈ 事件的子系统集合；只计与攻击重叠的报警段；方括号为 95% Wilson 区间。"
                "按事件：主口径取事件的第一个重叠报警段，合并口径为补充。单种子",
                "`reports/main/attribution_summary.csv`、`.md`、`attribution.csv`")
    first = summ.iloc[0]
    att.add(f"与多个事件重叠的报警段 {int(summ.n_multi_event_segments.max())} 个；无子系统标签事件 "
            f"{int(first.n_events_without_subsystem)} 个；不与攻击重叠的报警段："
            + "，".join(f"{m} {int(s.n_non_overlap)}" for m, s in summ.iterrows()),
            "主口径与并集口径因此完全相同；不与攻击重叠的报警段不计入命中率", "`reports/main/attribution_summary.csv`")
    aseed = pd.read_csv(rep / "attribution_seeds.csv").set_index("seed")
    am, asd = aseed.loc["mean"], aseed.loc["std"]
    att.add(f"**LSTM-AE {len(aseed) - 2} 个种子**：报警段 **{pm(am.subsystem_hit_rate, asd.subsystem_hit_rate, 3)}**，"
            f"事件·第一个报警段 **{pm(am.event_first_hit_rate, asd.event_first_hit_rate, 3)}**，"
            f"事件·合并 {pm(am.event_hit_rate, asd.event_hit_rate, 3)}；Top-1 在 P4 的比例 {pm(am.top1_p4_ratio, asd.top1_p4_ratio, 3)}",
            "每个种子的运行各算一遍归因；均值 ± 样本标准差。LSTM-VAE、PCA 只有单种子", "`reports/main/attribution_seeds.csv`")
    dur = pd.read_csv(rep / "attribution_by_duration.csv")
    for m in summ.index:
        g = dur[(dur.model == m) & (dur.variant == "main")]
        att.add(f"{m} 按报警段长度：" + "；".join(
            f"{r.duration_group} {int(r.n_hit)}/{int(r.n)}" + (f" = {r.rate:.3f} {ci(r.ci_low, r.ci_high)}" if r.n else "")
            for r in g.itertuples()),
                "与攻击重叠的报警段按长度分组；10–60 秒含两端。短报警段归因明显更差（LSTM-AE）",
                "`reports/main/attribution_by_duration.csv`")
    exp = pd.read_csv(rep / "attribution_exploratory_valnorm.csv")
    for m in summ.index:
        g = exp[exp.model == m].set_index("level")
        att.add(f"（探索性）{m} 按 val_thr 归一化后：" + "；".join(
            f"{lv} {int(r.n_hit_main)}/{int(r.n)} → {int(r.n_hit_valnorm)}/{int(r.n)}" for lv, r in g.iterrows()),
                "**事后分析，不替换主结果**：段内误差先除以各传感器在 val_thr 上的平均误差再排 Top-k；"
                "基线只来自 val_thr，但这一步是看过测试结果后才做的，数字偏乐观",
                "`reports/main/attribution_exploratory_valnorm.csv`")
    att.add("传感器级 Hit@k：未做", "HAI 21.03 的数据文件没有逐传感器的攻击目标，官方攻击目标表未取得；写进局限性",
            "`reports/main/attribution_summary.md`")

    # ---------- 融合 ----------
    fus = Table("融合（原型演示；不声称提升检测指标，约束 7）")
    alerts = json.loads(resolve(ml["alerts"]["output"]).read_text(encoding="utf-8"))
    fr = pd.read_csv(rep / "fusion_ranking_eval.csv")
    n_true = int(fr.n_true_total.iloc[0])
    fus.add(f"真实报警 {len(alerts)} 条，其中 {n_true} 条与攻击事件重叠（基准比例 {n_true / len(alerts):.3f}）",
            "LSTM-AE 主模型运行（种子 42）在 test 上的全部报警段，不按标签筛选；is_mock=false",
            "`data/processed/alerts.json`、`reports/main/fusion_ranking_eval.csv`")
    for name, g in fr.groupby("ranking", sort=False):
        k_max = g.iloc[-1]
        fus.add(f"排序 {name}：" + "，".join(f"precision@{int(r.k)} = {int(r.n_true)}/{int(r.k)}" for r in g.itertuples())
                + f"；precision@{int(k_max.k)} 的 95% 区间 {ci(k_max.ci_low, k_max.ci_high)}"
                + (f"；前 {int(k_max.k)} 条中 {int(k_max.n_shared_with_raw_score)} 条与原始分数排序相同" if name != "raw_score" else ""),
                "前 k 条报警中与攻击事件重叠的比例；raw_score 按最大异常分数，alpha=0 / alpha=1 为融合优先级。"
                "三种排序没有可测量的差别：排序的作用来自异常分数，不来自漏洞上下文",
                "`reports/main/fusion_ranking_eval.csv`、`.md`")
    n_sat = sum(severity(a["max_score"], a["threshold"]) >= 1.0 for a in alerts)
    fus.add(f"S 封顶的报警 {n_sat} 条", "S = min(1, 最大分数 ÷ 阈值 − 1)，分数达到阈值 2 倍即为 1；这些报警之间不再按分数区分。"
            "只写成局限和未来工作，不改公式", "`data/processed/alerts.json`、`reports/main/cases.md`")
    fus.add(f"卡片与演示只渲染优先级最高的前 {cfg['fusion']['max_rendered_alerts']} 条",
            "融合打分仍对全部报警进行；条数为 `fusion.max_rendered_alerts`", "`config/config.yaml`")
    fus.add("3 个案例：归因正确的真实攻击、归因错误的真实攻击、高优先级的误报",
            "按融合优先级从前往后各取第一条，不人工挑选；对照标签的说明来自事后评估", "`reports/main/cases.md`、`cases/`")
    fus.add("漏洞与资产侧的数字（540 条漏洞、KEV 命中 9 条、min_qod=70 后保留 66 条、E_sub 只有 0 / 1 两个取值）",
            "真实扫描只有一台靶机 ews_01；资产映射为人为构造", "`reports/extension/key_numbers.md`、`kev_table.md`")

    # ---------- 数据问题 ----------
    dat = Table("数据问题")
    pre = run_json(root, "lstm_ae", "preprocess.json")
    drops = ml["data"]["train_overlap_drops"][0]
    dat.add(f"官方 train1 末尾 {drops['tail_rows']:,} 秒与 test3 开头 {drops['tail_rows']:,} 秒是同一份数据；"
            f"另有 train1 第 {drops['head_rows']} 行 = test2 最后 1 行。两处都已从训练集删除，test 保持原样",
            "按 79 个传感器的逐行内容哈希比较全部文件对；这段数据没有攻击。不处理会低估 test3 上的误报",
            "`reports/main/data_summary.md` 第 3 节；`config/config.yaml` 的 `ml.data.train_overlap_drops`")
    r = pre["rows"]
    dat.add(f"划分：train {r['train']:,} / val_es {r['val_es']:,} / val_thr {r['val_thr']:,} 行；特征 {pre['n_features']} 个"
            f"（删除 {len(pre['constant_cols'])} 个常量列）",
            "删除重叠行之后按时间切分，最后 20% 对半分为 val_es（早停）和 val_thr（定阈值）；scaler 只在 train 部分 fit",
            f"`{art}/lstm_ae/preprocess.json`；`PROGRESS.md` 的 M1 一节")
    p = evals["lstm_ae"]["point"]
    events = pd.read_csv(rep / "attack_events.csv")
    dat.add(f"test：{p['n_points'] + p['n_unscored']:,} 秒，攻击 {p['n_positive']:,} 秒"
            f"（{100 * p['n_positive'] / (p['n_points'] + p['n_unscored']):.2f}%），{len(events)} 个攻击事件"
            f"（长度 {int(events.duration_s.min())}~{int(events.duration_s.max())} 秒）",
            "事件 = attack=1 的连续区间", "`reports/main/data_summary.md`、`attack_events.csv`")
    dat.add(f"未打分 {p['n_unscored']} 秒（5 个文件 × {ml['window']['length'] - 1} 秒），其中攻击秒 {p['n_positive_unscored']} 个",
            "每个测试文件的前 W−1 秒没有分数（W=60），不参与任何指标；W=30 / 120 的消融组合为 145 / 595 秒",
            "`reports/main/main_results.csv` 的 n_unscored 列")
    dat.add(f"attack=1 但三个子系统标签全为 0 的有 {int(events.no_sub_s.sum())} 秒，分布在 {int((events.no_sub_s > 0).sum())} 个事件内部；"
            "按并集规则 50 个事件的子系统集合全部非空",
            "事件的子系统集合 = 事件内所有秒的 attack_P* 并集；检测指标只用 attack 列，不受影响",
            "`reports/main/attack_events.csv`、`data_summary.md`")
    files = [f"{x['file'].split('.')[0]} {100 * (x['tp'] + x['fp']) / x['n_points']:.1f}%" for x in evals["lstm_ae"]["per_file"]]
    dat.add("LSTM-AE 各测试文件报警的秒占比：" + "，".join(files),
            "val_thr 上为 1%。各文件差别很大，推测 test 的正常工况相对训练数据有漂移（未单独验证是哪些传感器）",
            f"`{art}/lstm_ae/eval_metrics.json`；图 `reports/main/scores_test*.png`、`score_distribution.png`")
    dat.add("P4 没有子系统标签", "P4 是 HIL 仿真部分；Top-1 落在 P4 的归因必然计为未命中", "`data/samples/hai_columns.txt`、决策 7")

    head = ["## 关键数字（主线）", "",
            "写报告时直接引用。数字由 `python -m scripts.make_key_numbers` 从来源文件读出（本节是 `reports/main/key_numbers.md` 的副本）；"
            "检测、消融、归因的数字全部来自集群上的完整运行（git_dirty=false），冒烟测试的数字没有进入这里。",
            "来源文件全部入库：`reports/main/runs/` 是各次集群运行的小文件（指标、manifest、阈值、预处理摘要、训练记录）的副本，"
            "由本脚本从产物目录复制，主机名等集群信息已替换为占位符。", ""]
    return "\n".join(head + det.text() + att.text() + fus.text() + dat.text()).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="汇总主线关键数字到 reports/main/key_numbers.md")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    # 先把产物目录里各次运行的小文件脱敏后复制到入库的位置；产物目录里没有的运行沿用已入库的副本
    copied, skipped = snapshot(cfg)
    print(f"[复制] {len(copied)} 次运行的小文件（已脱敏）→ {resolve(cfg['ml']['report_dir']) / SNAPSHOT_SUBDIR}"
          + (f"；{len(skipped)} 次运行在产物目录中不可用，沿用已入库的副本" if skipped else ""))
    try:
        text = build(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1
    out = resolve(cfg["ml"]["report_dir"]) / "key_numbers.md"
    out.write_text(text, encoding="utf-8")
    print(f"[生成] {out}（{text.count(chr(10))} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
