"""漏洞风险 → 资产风险 → 子系统暴露风险（claude.md M9 第 1~3 条）。

- r_v     = min(1, (cvss/10) × (1 + beta·epss) × (gamma if in_kev else 1))
- R_asset = 该资产所有参与计算的漏洞 r_v 的最大值；已扫描但无漏洞记 0
- E_sub   = 子系统全部上游资产（见 asset_graph.upstream_assets）的 R_asset 最大值；无上游资产记 0

未扫描资产 = "未评估"：输出中 R_asset 为 None、assessed 为 False，不显示成 0 风险；
计算 E_sub 时按 0 处理。子系统记录 unassessed_assets（未评估的上游资产）与 E_sub_assessed
（是否至少有一个上游资产被评估）。

QoD 规则：qod < min_qod（或缺失）的漏洞不参与计算；其中命中 KEV 的列入 needs_review，
供告警卡片显示"需人工核实"。

用法：
    python -m src.fusion.risk [--vulns JSON]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.fusion.asset_graph import AssetGraph

ROOT = Path(__file__).resolve().parents[2]


def vuln_risk(v: dict, beta: float, gamma: float) -> float:
    r = (v["cvss"] / 10) * (1 + beta * v["epss"]) * (gamma if v["in_kev"] else 1)
    return min(1.0, r)


def compute_risk(vulns: list[dict], graph: AssetGraph, params: dict) -> dict:
    """返回 {vulns, needs_review, unmapped_ips, assets, subsystems}。

    vulns / needs_review 中每条记录额外带 asset（资产 id）和 r_v 字段。
    """
    min_qod = params["min_qod"]
    scored, needs_review, unmapped = [], [], set()
    for v in vulns:
        asset = graph.asset_by_ip(v["asset_ip"])
        if asset is None:
            unmapped.add(v["asset_ip"])
            continue
        rec = {**v, "asset": asset, "r_v": vuln_risk(v, params["beta"], params["gamma"])}
        if v.get("qod") is not None and v["qod"] >= min_qod:
            scored.append(rec)
        elif v["in_kev"]:
            needs_review.append(rec)
    # r_v 相同（常见于截断到 1）时依次按 KEV、CVSS、EPSS 排序
    order = lambda r: (-r["r_v"], not r["in_kev"], -r["cvss"], -r["epss"])  # noqa: E731
    scored.sort(key=order)
    needs_review.sort(key=order)

    assets = {}
    for aid, a in graph.assets.items():
        mine = [r for r in scored if r["asset"] == aid]
        if not a.scanned:
            status = "not_scanned"
        elif not mine:
            status = "no_vulns"
        else:
            status = "scanned"
        assets[aid] = {
            "R_asset": max((r["r_v"] for r in mine), default=0.0) if a.scanned else None,
            "assessed": a.scanned,
            "status": status,
            "n_vulns": len(mine),
            "n_needs_review": sum(r["asset"] == aid for r in needs_review),
        }

    def risk_or_zero(aid: str) -> float:
        # 未评估资产在计算中按 0 处理
        return assets[aid]["R_asset"] or 0.0

    subsystems = {}
    for sid in graph.subsystems:
        up = graph.upstream_assets(sid)
        best = max(up, key=risk_or_zero, default=None)
        subsystems[sid] = {
            "E_sub": risk_or_zero(best) if best else 0.0,
            "max_asset": best if best and risk_or_zero(best) > 0 else None,
            "upstream_assets": up,
            "unassessed_assets": [a for a in up if not assets[a]["assessed"]],
            "E_sub_assessed": any(assets[a]["assessed"] for a in up),
        }

    return {"vulns": scored, "needs_review": needs_review, "unmapped_ips": sorted(unmapped),
            "assets": assets, "subsystems": subsystems}


def load_params(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    p = argparse.ArgumentParser(description="计算资产风险 R_asset 与子系统暴露风险 E_sub")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--vulns", type=Path, help="默认取 config.yaml 的 vuln.vulns_enriched_json")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    graph = AssetGraph.from_yaml(ROOT / cfg["fusion"]["asset_map"])
    params = load_params(ROOT / cfg["fusion"]["params"])
    src = args.vulns or ROOT / cfg["vuln"]["vulns_enriched_json"]
    risk = compute_risk(json.loads(src.read_text(encoding="utf-8"))["vulns"], graph, params)

    print(f"参与计算的漏洞 {len(risk['vulns'])} 条（min_qod={params['min_qod']}），"
          f"needs_review {len(risk['needs_review'])} 条，未映射 IP {risk['unmapped_ips']}")
    for aid, a in risk["assets"].items():
        r = "未评估" if a["R_asset"] is None else f"{a['R_asset']:.3f}"
        print(f"  {aid:<10} R_asset={r:<6} {a['status']:<11} 漏洞 {a['n_vulns']}，"
              f"需核实 {a['n_needs_review']}")
    for sid, s in risk["subsystems"].items():
        print(f"  {sid}  E_sub={s['E_sub']:.3f}  来自 {s['max_asset']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
