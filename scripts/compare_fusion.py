"""Compare two fusion scoring results (E3: before and after adding the second target) and report changes in R_asset, E_sub and alert ranking.

Inputs are two enriched_alerts JSON files produced by prioritize; output is Markdown (printed and written to a file).
If the alerts come from mock data this is noted in the output; per the commit policy, this mock-based comparison is not tracked.

Usage:
    python -m scripts.compare_fusion --before data/processed/enriched_alerts_before_e3.json \
        [--after data/processed/enriched_alerts.json] [--output reports/extension/e3_compare.md]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.utils.disclaimer import DISCLAIMER

ROOT = Path(__file__).resolve().parents[1]


def _fmt(x) -> str:
    return "Not assessed" if x is None else f"{x:.3f}"


def _delta(a, b) -> str:
    if a is None or b is None:
        return "-" if a == b else "newly assessed" if a is None else "no longer assessed"
    d = b - a
    return "0" if abs(d) < 5e-4 else f"{d:+.3f}"


def compare(before: dict, after: dict) -> str:
    lines = ["# Fusion scoring comparison (before and after adding the second target)", "",
             f"> Disclaimer: {DISCLAIMER}"]
    if before.get("is_mock") or after.get("is_mock"):
        lines.append("> [MOCK] The alert ranking is based on mock alerts; R_asset / E_sub are based on real scan data.")
    lines += ["", "## Asset risk R_asset", "", "| Asset | Before | After | Change | Status (before → after) |", "|---|---|---|---|---|"]
    ab, aa = before["asset_risk"], after["asset_risk"]
    for aid in sorted(set(ab) | set(aa)):
        b, a = ab.get(aid, {}), aa.get(aid, {})
        rb, ra = b.get("R_asset"), a.get("R_asset")
        lines.append(f"| {aid} | {_fmt(rb)} | {_fmt(ra)} | {_delta(rb, ra)} | "
                     f"{b.get('status', '-')} → {a.get('status', '-')} |")

    lines += ["", "## Subsystem exposure risk E_sub", "", "| Subsystem | Before | After | Change | Source asset (after) |", "|---|---|---|---|---|"]
    sb, sa = before["subsystem_risk"], after["subsystem_risk"]
    for sid in sorted(set(sb) | set(sa)):
        eb, ea = sb.get(sid, {}).get("E_sub", 0.0), sa.get(sid, {}).get("E_sub", 0.0)
        lines.append(f"| {sid} | {eb:.3f} | {ea:.3f} | {_delta(eb, ea)} | {sa.get(sid, {}).get('max_asset') or '-'} |")

    lines += ["", "## Alert ranking", "", "| Alert | Rank before → after | Priority before | Priority after | Change |", "|---|---|---|---|---|"]
    xb = {a["alert_id"]: a for a in before["alerts"]}
    xa = {a["alert_id"]: a for a in after["alerts"]}
    for aid in sorted(xa, key=lambda k: xa[k]["rank"]):
        b, a = xb.get(aid), xa[aid]
        rank_b = b["rank"] if b else "-"
        pb = b["priority"] if b else 0.0
        lines.append(f"| {aid} | {rank_b} → {a['rank']} | {pb:.3f} | {a['priority']:.3f} | {_delta(pb, a['priority'])} |")
    order_b = [a["alert_id"] for a in sorted(before["alerts"], key=lambda a: a["rank"])]
    order_a = [a["alert_id"] for a in sorted(after["alerts"], key=lambda a: a["rank"])]
    lines += ["", f"Ranking changed: {'yes' if order_b != order_a else 'no'}", "", "---", DISCLAIMER, ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Compare two fusion scoring results")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, help="defaults to fusion.enriched_alerts_json in config.yaml")
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
    print(f"[SAVED] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
