"""对比两次融合打分结果（E3：加入第二台靶机前后），输出 R_asset、E_sub 与告警排序的变化。

输入为 prioritize 输出的两份 enriched_alerts JSON；输出 Markdown（打印并写文件）。
报警来自 mock 时在输出中标注；该对比基于 mock 报警，按提交策略不入库。

用法：
    python -m scripts.compare_fusion --before data/processed/enriched_alerts_before_e3.json \
        [--after data/processed/enriched_alerts.json] [--output reports/extension/e3_compare.md]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "资产映射与漏洞数据来自实验环境，仅用于演示"


def _fmt(x) -> str:
    return "未评估" if x is None else f"{x:.3f}"


def _delta(a, b) -> str:
    if a is None or b is None:
        return "-" if a == b else "新评估" if a is None else "不再评估"
    d = b - a
    return "0" if abs(d) < 5e-4 else f"{d:+.3f}"


def compare(before: dict, after: dict) -> str:
    lines = ["# 融合打分对比（加入第二台靶机前后）", "",
             f"> 免责声明：{DISCLAIMER}"]
    if before.get("is_mock") or after.get("is_mock"):
        lines.append("> [MOCK] 告警排序基于 mock 报警；R_asset / E_sub 基于真实扫描数据。")
    lines += ["", "## 资产风险 R_asset", "", "| 资产 | 之前 | 之后 | 变化 | 状态（之前 → 之后） |", "|---|---|---|---|---|"]
    ab, aa = before["asset_risk"], after["asset_risk"]
    for aid in sorted(set(ab) | set(aa)):
        b, a = ab.get(aid, {}), aa.get(aid, {})
        rb, ra = b.get("R_asset"), a.get("R_asset")
        lines.append(f"| {aid} | {_fmt(rb)} | {_fmt(ra)} | {_delta(rb, ra)} | "
                     f"{b.get('status', '-')} → {a.get('status', '-')} |")

    lines += ["", "## 子系统暴露风险 E_sub", "", "| 子系统 | 之前 | 之后 | 变化 | 来源资产（之后） |", "|---|---|---|---|---|"]
    sb, sa = before["subsystem_risk"], after["subsystem_risk"]
    for sid in sorted(set(sb) | set(sa)):
        eb, ea = sb.get(sid, {}).get("E_sub", 0.0), sa.get(sid, {}).get("E_sub", 0.0)
        lines.append(f"| {sid} | {eb:.3f} | {ea:.3f} | {_delta(eb, ea)} | {sa.get(sid, {}).get('max_asset') or '-'} |")

    lines += ["", "## 告警排序", "", "| 告警 | 排名 之前 → 之后 | priority 之前 | priority 之后 | 变化 |", "|---|---|---|---|---|"]
    xb = {a["alert_id"]: a for a in before["alerts"]}
    xa = {a["alert_id"]: a for a in after["alerts"]}
    for aid in sorted(xa, key=lambda k: xa[k]["rank"]):
        b, a = xb.get(aid), xa[aid]
        rank_b = b["rank"] if b else "-"
        pb = b["priority"] if b else 0.0
        lines.append(f"| {aid} | {rank_b} → {a['rank']} | {pb:.3f} | {a['priority']:.3f} | {_delta(pb, a['priority'])} |")
    order_b = [a["alert_id"] for a in sorted(before["alerts"], key=lambda a: a["rank"])]
    order_a = [a["alert_id"] for a in sorted(after["alerts"], key=lambda a: a["rank"])]
    lines += ["", f"排序是否变化：{'是' if order_b != order_a else '否'}", "", "---", DISCLAIMER, ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="对比两次融合打分结果")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, help="默认取 config.yaml 的 fusion.enriched_alerts_json")
    p.add_argument("--output", type=Path, default=ROOT / "reports" / "extension" / "e3_compare.md")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["fusion"]
    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads((args.after or ROOT / cfg["enriched_alerts_json"]).read_text(encoding="utf-8"))
    text = compare(before, after)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"[保存] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
