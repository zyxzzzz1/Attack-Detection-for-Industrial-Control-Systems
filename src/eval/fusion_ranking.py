"""融合排序的评估（I1）：前 k 条报警里有多少条是真的，比较三种排序。

- raw_score：按报警段的最大异常分数从大到小。
- alpha=0：融合优先级 priority，但 alpha = 0（不使用漏洞暴露风险 E_sub，只有异常严重度 S 与子系统关键度）。
- alpha=1：融合优先级 priority，alpha = 1（config/fusion.yaml 的默认值）。

precision@k = 排在前 k 条的报警中与攻击事件重叠的比例。"是否与攻击事件重叠"取自 evaluate 阶段写出的
eval_alarms.csv（标签只在 evaluate.py 中读取）；本模块不读取标签，scripts/export_alerts.py 和 alerts.json 里也没有这一列。
报警与 eval_alarms.csv 的行按顺序一一对应（同一个 to_segments），对不上时报错。

注意（约束 7）：资产映射是人为构造的，漏洞数据来自实验靶机，与 HAI 中的攻击没有因果关系。
这里的比较只说明"按这套公式排序后，真报警排得靠前还是靠后"，不能说明漏洞上下文提升了检测能力。

输出（正式：reports/main/；--smoke：产物目录/smoke_reports/）：fusion_ranking_eval.csv / .md

用法：
    python -m src.eval.fusion_ranking [--smoke]
"""

import argparse
import json
import sys

import pandas as pd

from src.eval.metrics import precision_at_k, wilson_interval
from src.fusion.asset_graph import AssetGraph
from src.fusion.prioritize import prioritize, severity
from src.fusion.risk import load_params
from src.utils.paths import load_config, resolve
from src.utils.reporting import ReportSource, add_common_args

COLUMNS = ["ranking", "k", "n_true", "precision", "ci_low", "ci_high", "n_alerts", "n_true_total", "base_rate",
           "n_tied_with_kth", "n_shared_with_raw_score"]


def truth_by_alert(alerts: list[dict], alarms: pd.DataFrame) -> dict[str, bool]:
    """alert_id → 是否与攻击事件重叠。两边必须是同一批报警段。"""
    if len(alerts) != len(alarms):
        raise ValueError(f"alerts.json 有 {len(alerts)} 条，eval_alarms.csv 有 {len(alarms)} 行：不是同一次运行的产物")
    for a, row in zip(alerts, alarms.itertuples()):
        if (a["start"], a["end"]) != (row.start_time, row.end_time):
            raise ValueError(f"{a['alert_id']} 的起止时间与 eval_alarms.csv 对不上：不是同一次运行的产物")
    return {a["alert_id"]: bool(t) for a, t in zip(alerts, alarms["overlaps_event"])}


def rankings(alerts: list[dict], vulns: list[dict], graph: AssetGraph, params: dict) -> dict[str, list[tuple[str, float]]]:
    """三种排序，各为 [(alert_id, 排序用的值), ...]。并列时都按 alert_id（即时间顺序）排。"""
    out = {"raw_score": sorted(((a["alert_id"], a["max_score"]) for a in alerts), key=lambda x: (-x[1], x[0]))}
    for alpha in (0.0, 1.0):
        ranked = prioritize(alerts, vulns, graph, {**params, "alpha": alpha})["alerts"]
        out[f"alpha={alpha:g}"] = [(a["alert_id"], a["priority"]) for a in ranked]
    return out


def evaluate_rankings(ranked: dict[str, list[tuple[str, float]]], truth: dict[str, bool], ks: list[int]) -> pd.DataFrame:
    n, n_true = len(truth), sum(truth.values())
    rows = []
    reference = [i for i, _ in ranked["raw_score"]] if "raw_score" in ranked else []
    for name, order in ranked.items():
        hits = [truth[i] for i, _ in order]
        for k in ks:
            kk = min(k, n)
            shared = len({i for i, _ in order[:kk]} & set(reference[:kk]))  # 前 k 条中与原始分数排序相同的报警条数
            n_hit = sum(hits[:kk])
            lo, hi = wilson_interval(n_hit, kk)
            # 与第 k 条的排序值相同的报警条数：大于 1 说明第 k 条附近是并列的，谁排进前 k 取决于并列时的次序
            tied = sum(v == order[kk - 1][1] for _, v in order) if kk else 0
            rows.append({"ranking": name, "k": k, "n_true": n_hit, "precision": precision_at_k(hits, k),
                         "ci_low": lo, "ci_high": hi, "n_alerts": n, "n_true_total": n_true,
                         "base_rate": n_true / n if n else float("nan"), "n_tied_with_kth": tied,
                         "n_shared_with_raw_score": shared})
    return pd.DataFrame(rows, columns=COLUMNS)


def to_md(df: pd.DataFrame, model: str, n_saturated: int, smoke: bool) -> str:
    n, n_true = int(df["n_alerts"].iloc[0]), int(df["n_true_total"].iloc[0])
    lines = ["# 融合排序评估" + ("（冒烟数据，仅检查流程，不得引用）" if smoke else ""), "",
             f"{model} 的 {n} 条报警中有 {n_true} 条与攻击事件重叠（基准比例 {n_true / n:.4f}，即随机排序时 precision@k 的期望）。"
             "precision@k 是排在前 k 条的报警中与攻击事件重叠的比例，ci_low / ci_high 为 95% Wilson 置信区间。", "",
             "- raw_score：按报警段内的最大异常分数排序。",
             "- alpha=0：融合优先级，不使用漏洞暴露风险（只有异常严重度 S 与子系统关键度）。",
             "- alpha=1：融合优先级，使用漏洞暴露风险 E_sub（默认参数）。", "",
             f"异常严重度 S = min(1, 最大分数 ÷ 阈值 − 1) 在分数达到阈值的 2 倍时封顶：{n_saturated} 条报警的 S = 1。"
             "这些报警在融合排序里只靠子系统的权重、关键度和 E_sub 区分；n_tied_with_kth 是与第 k 条优先级完全相同的报警条数，"
             "并列时按报警编号（时间顺序）排。n_shared_with_raw_score 是前 k 条中同时也在原始分数排序前 k 条里的报警条数"
             "（precision 相同不代表选出的是同一批报警）。", "",
             "**局限（约束 7）**：资产映射是人为构造的，漏洞数据来自实验靶机，与 HAI 中的攻击没有因果关系。"
             "本表只描述这套排序公式在这份数据上的表现，不说明漏洞上下文能提升检测能力；融合部分不声称提升检测指标。", "",
             "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="融合排序评估：precision@k（原始分数、alpha=0、alpha=1）")
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    ml, fcfg = cfg["ml"], cfg["fusion"]
    model = ml["alerts"]["model"]
    src = ReportSource(cfg, args.smoke)
    src.banner()
    alerts_path = src.out / "alerts.json" if args.smoke else resolve(ml["alerts"]["output"])
    if not src.usable_runs([model], required=("eval_alarms.csv",)) or not alerts_path.exists():
        print(f"[错误] 需要 {model} 的 eval_alarms.csv 和 {alerts_path}（先运行 scripts.export_alerts）", file=sys.stderr)
        return 1
    alerts = json.loads(alerts_path.read_text(encoding="utf-8"))
    truth = truth_by_alert(alerts, pd.read_csv(src.root / model / "eval_alarms.csv"))
    graph = AssetGraph.from_yaml(resolve(fcfg["asset_map"]))
    params = load_params(resolve(fcfg["params"]))
    vulns = json.loads(resolve(cfg["vuln"]["vulns_enriched_json"]).read_text(encoding="utf-8"))["vulns"]

    df = evaluate_rankings(rankings(alerts, vulns, graph, params), truth, ml["fusion_eval"]["ks"])
    n_saturated = sum(severity(a["max_score"], a["threshold"]) >= 1.0 for a in alerts)
    df.to_csv(src.out / "fusion_ranking_eval.csv", index=False)
    (src.out / "fusion_ranking_eval.md").write_text(to_md(df, model, n_saturated, args.smoke), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"[生成] {src.out}/fusion_ranking_eval.csv / .md（S 封顶的报警 {n_saturated} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
