"""I1：从真实报警中按固定规则挑 3 个案例，写 reports/main/cases.md，并把对应的告警卡片存到 reports/main/cases/。

挑选规则（按融合优先级的排名从前往后找，各取第一条；不人工挑选）：
1. 归因正确的真实攻击：与攻击事件重叠、Top-1 传感器所属子系统在事件的子系统集合内。
2. 归因错误的真实攻击：与攻击事件重叠、Top-1 传感器所属子系统不在事件的子系统集合内。
3. 高优先级的误报：不与任何攻击事件重叠。

输入：fusion.enriched_alerts_json（先运行 python -m src.pipeline）、reports/main/attribution.csv（python -m src.attribution）、
主模型运行目录的 eval_events.csv。"是否真实攻击、归因是否命中"取自归因评估和 evaluate 的输出，本脚本不读取标签；
告警卡片本身不含这些信息。

用法：
    python -m scripts.make_cases
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.eval.alarms import subsystem_of
from src.fusion.render_card import card_context, make_env
from src.utils import i18n
from src.utils.i18n import t
from src.utils.paths import DEFAULT_CONFIG, artifact_dir, load_config, resolve
from src.utils.reporting import report_problem

CASES = [("归因正确的真实攻击", lambda r: r.category == "evaluable" and r.hit is True),
         ("归因错误的真实攻击", lambda r: r.category == "evaluable" and r.hit is False),
         ("高优先级的误报", lambda r: r.category == "non_overlap")]


def load_table(cfg: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """每条报警一行：融合排名 + 归因评估结果。两边按报警段顺序一一对应，起止时间对不上就报错。"""
    ml, fcfg = cfg["ml"], cfg["fusion"]
    model = ml["alerts"]["model"]
    run_dir = artifact_dir(cfg) / model
    problem = report_problem(run_dir)
    if problem:
        raise ValueError(f"{model} 的运行不能用于报告：{problem}")
    enriched = json.loads(resolve(fcfg["enriched_alerts_json"]).read_text(encoding="utf-8"))
    if enriched["is_mock"]:
        raise ValueError("enriched_alerts.json 是 mock 报警的输出：先运行 python -m src.pipeline（不带 --mock）")
    alerts = json.loads(resolve(ml["alerts"]["output"]).read_text(encoding="utf-8"))
    attr = pd.read_csv(resolve(ml["report_dir"]) / "attribution.csv")
    attr = attr[attr["model"] == model].reset_index(drop=True)
    if len(attr) != len(alerts) or any((a["start"], a["end"]) != (r.start_time, r.end_time)
                                       for a, r in zip(alerts, attr.itertuples())):
        raise ValueError("alerts.json 与 attribution.csv 不是同一批报警段：重新运行 export_alerts 和 attribution")
    attr["alert_id"] = [a["alert_id"] for a in alerts]
    attr["hit"] = attr["hit"].map({True: True, False: False, "True": True, "False": False}).astype(object)
    attr["hit"] = attr["hit"].where(attr["hit"].notna(), None)
    rank = {a["alert_id"]: a["rank"] for a in enriched["alerts"]}
    attr["rank"] = attr["alert_id"].map(rank)
    return attr.sort_values("rank").reset_index(drop=True), enriched, pd.read_csv(run_dir / "eval_events.csv")


def describe(title: str, row, alert: dict, events: pd.DataFrame, n_total: int, card_file: str) -> list[str]:
    ratio = alert["max_score"] / alert["threshold"]
    top = alert["topk"]
    sep = t("、")
    if alert["vulns"]:
        vuln_line = t("- 路径上未评估的资产：{assets}；列出的漏洞 {n} 条（最高的是 {asset} 上的 {name}），需人工核实 {m} 条").format(
            assets=sep.join(alert["unassessed_assets"]) or t("无"), n=len(alert["vulns"]),
            asset=alert["vulns"][0]["asset"], name=alert["vulns"][0]["name"], m=len(alert["needs_review"]))
    else:
        vuln_line = t("- 路径上没有参与计算的漏洞")
    lines = [f"## {title}{t('：')}{alert['alert_id']}", "",
             t("- 融合优先级排名：第 {rank} / {n}（priority {p}，S {s}，E_sub {e}）").format(
                 rank=alert["rank"], n=n_total, p=f"{alert['priority']:.3f}", s=f"{alert['S']:.3f}", e=f"{alert['E_sub']:.3f}"),
             t("- 报警时间：{start} ~ {end}（{d} 秒，{file}）").format(
                 start=alert["start"], end=alert["end"], d=alert["duration_s"], file=row.file.split(".")[0]),
             t("- 最大异常分数 {score}，是阈值（{thr}）的 {ratio} 倍").format(
                 score=f"{alert['max_score']:.4g}", thr=f"{alert['threshold']:.4f}", ratio=f"{ratio:.1f}"),
             t("- Top-5 传感器（占全部传感器误差的比例）：") + sep.join(f"{x['sensor']} {x['share']:.1%}" for x in top),
             t("- 卡片推断的子系统：") + sep.join(
                 t("{id} {w}（关键度 {c}，E_sub {e}）").format(id=x["id"], w=f"{x['weight']:.0%}", c=x["criticality"], e=f"{x['E_sub']:.0f}")
                 for x in alert["subsystems"]),
             t("- 卡片给出的入侵路径：") + t("；").join(" → ".join(x["path"]) for x in alert["path"]),
             vuln_line,
             t("- 告警卡片：") + f"[`cases/{card_file}`](cases/{card_file})", ""]
    top1_sub = subsystem_of(top[0]["sensor"])
    if row.category == "evaluable":
        ev = events[events["event_id"] == int(row.primary_event_id)].iloc[0]
        verdict = (t("**在**事件的子系统集合内，归因命中。") if row.hit
                   else t("**不在**事件的子系统集合内，按决策 7 的规则归因未命中。"))
        lines += [t("**对照标签**：这条报警与攻击事件 #{eid} 重叠（{start} ~ {end}，{d} 秒，标签中的受攻击子系统 {subs}；"
                    "该事件的检测延迟 {delay} 秒）。Top-1 传感器 {sensor} 属于 {sub}，").format(
                      eid=int(ev.event_id), start=ev.start_time, end=ev.end_time, d=int(ev.duration_s), subs=ev.subsystems,
                      delay=int(ev.delay_s), sensor=top[0]["sensor"], sub=top1_sub) + verdict, ""]
    else:
        same = events[events["file"] == row.file]
        before = same[pd.to_datetime(same["end_time"]) < pd.Timestamp(alert["start"])]
        after = same[pd.to_datetime(same["start_time"]) > pd.Timestamp(alert["end"])]
        gaps = []
        if len(before):
            b = before.iloc[-1]
            minutes = (pd.Timestamp(alert["start"]) - pd.Timestamp(b.end_time)).total_seconds() / 60
            gaps.append(t("前一个攻击事件 #{eid} 结束于 {time}（{m} 分钟之前）").format(eid=int(b.event_id), time=b.end_time, m=f"{minutes:.0f}"))
        if len(after):
            n = after.iloc[0]
            minutes = (pd.Timestamp(n.start_time) - pd.Timestamp(alert["end"])).total_seconds() / 60
            gaps.append(t("后一个攻击事件 #{eid} 开始于 {time}（{m} 分钟之后）").format(eid=int(n.event_id), time=n.start_time, m=f"{minutes:.0f}"))
        lines += [t("**对照标签**：这条报警不与任何攻击事件重叠，是误报。")
                  + (t("同一文件中{gaps}。").format(gaps=t("，").join(gaps)) if gaps else ""), ""]
    return lines


def observations(picked: dict) -> list[str]:
    """三个案例放在一起能看到的现象；每一条只在数据确实如此时才写。"""
    out = []
    if 2 in picked:
        row, alert = picked[2]
        weights = {s["id"]: s["weight"] for s in alert["subsystems"]}
        true_subs = [x for x in str(row.event_subsystems).split("+") if x]
        share = sum(weights.get(x, 0.0) for x in true_subs)
        if share >= 0.5:
            out.append(t("- 案例 2 的 Top-1 传感器不在受攻击的子系统内，但 Top-5 里其余传感器大多属于受攻击的子系统："
                         "卡片按 Top-5 的误差占比加权后，仍把 {share} 的权重放在 {subs} 上，给出的入侵路径也是它的。"
                         "也就是说\"Top-1 是否命中\"比卡片实际展示的信息更严格；这只是这一条报警的情况，不能推广。").format(
                             share=f"{share:.0%}", subs="+".join(true_subs)))
        if "P4" in weights:
            out.append(t("- 案例 2 的 Top-1 落在 P4（HIL 仿真部分，没有子系统标签，按规则必然算未命中）。P4 与 P1、P2、P3 都有物理耦合，"
                         "指向 P4 不一定是错的，现有标签无法判断；卡片对 P4 附有说明，建议同时排查 P1 / P2 / P3。"))
    if 2 in picked and 3 in picked:
        (_, real), (_, false) = picked[2], picked[3]
        r_real, r_false = real["max_score"] / real["threshold"], false["max_score"] / false["threshold"]
        if false["rank"] < real["rank"] and r_false < r_real:
            out.append(t("- 案例 3 的误报（分数是阈值的 {f} 倍）排在案例 2 的真实攻击（{r} 倍）之前。"
                         "原因是异常严重度 S = min(1, 分数 ÷ 阈值 − 1) 在分数达到阈值的 2 倍时就封顶为 1，"
                         "两条报警的 S 相同，排序只由子系统的权重、关键度和 E_sub 决定。S 的封顶点太低，是这套打分公式的一个缺陷。").format(
                             f=f"{r_false:.1f}", r=f"{r_real:.1f}"))
    tops = {a["vulns"][0]["nvt_oid"] for _, a in picked.values() if a["vulns"]}
    assets = {v["asset"] for _, a in picked.values() for v in a["vulns"]}
    if len(picked) > 1 and len(tops) == 1 and len(assets) == 1:
        out.append(t("- 三张卡片列出的漏洞都来自同一台已扫描的资产（{asset}），排在最前的也是同一条："
                     "只有一台靶机被评估，凡是涉及 P1 或 P2 的报警都会得到几乎相同的漏洞上下文，真实攻击和误报之间没有区别。"
                     "这是单靶机和风险饱和造成的局限（见 PROGRESS.md 的 M9 一节）。").format(asset=next(iter(assets))))
    if 3 in picked:
        out.append(t("- 误报的卡片与真实攻击的卡片在形式上完全一样（同样的入侵路径、漏洞和修复建议）。"
                     "卡片不能帮助判断一条报警是不是误报，它只是在报警为真的前提下给出排查的起点。"))
    return ([f"## {t('放在一起看')}", "", *out, ""] if out else [])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="挑 3 个案例写 reports/main/cases.md")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    try:
        table, enriched, events = load_table(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1
    out = resolve(cfg["ml"]["report_dir"])
    cards = out / "cases"
    cards.mkdir(parents=True, exist_ok=True)
    for old in cards.glob("*.md"):
        old.unlink()
    env = make_env(resolve(cfg["fusion"]["templates_dir"]))
    by_id = {a["alert_id"]: a for a in enriched["alerts"]}
    n_total = len(enriched["alerts"])

    i18n.configure(cfg)   # 显示语言（ui.lang）
    lines = [f"# {t('案例')}", "",
             "> " + t("资产映射与漏洞数据来自实验环境，仅用于演示") + t("。")
             + t("资产映射是人为构造的，漏洞数据来自自建靶机的扫描，与 HAI 中的攻击没有因果关系；"
                 "卡片中的入侵路径和漏洞只说明原型如何把信息组织到一起，不是对这些攻击的溯源结论。"), "",
             t("报警来自 LSTM-AE 主模型运行（{model}，种子 {seed}）在 HAI 21.03 测试集上的全部 {n} 条报警段。"
               "案例按固定规则挑选，不人工挑选：按融合优先级的排名从前往后，分别取第一条"
               "归因正确的真实攻击、第一条归因错误的真实攻击、第一条误报。"
               "\"是否真实攻击、归因是否命中\"来自事后用标签做的评估，卡片上没有这些信息。").format(
                   model=cfg["ml"]["alerts"]["model"], seed=cfg["ml"]["seed"], n=n_total), ""]
    chosen, picked = [], {}
    for k, (title, rule) in enumerate(CASES, 1):
        row = next((r for r in table.itertuples() if rule(r)), None)
        if row is None:
            lines += [f"## {t(title)}", "", t("没有符合条件的报警。"), ""]
            continue
        alert = by_id[row.alert_id]
        card_file = f"case{k}_{alert['alert_id']}.md"
        (cards / card_file).write_text(env.get_template("alert_card.md.j2").render(**card_context(alert, enriched)),
                                       encoding="utf-8")
        lines += describe(t("案例 {k}　").format(k=k) + t(title), row, alert, events, n_total, card_file)
        chosen.append((title, alert["alert_id"], alert["rank"]))
        picked[k] = (row, alert)
    lines += observations(picked)
    (out / "cases.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    for title, aid, rank in chosen:
        print(f"[案例] {title}：{aid}（排名 {rank}）")
    print(f"[生成] {out / 'cases.md'}、{cards}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
