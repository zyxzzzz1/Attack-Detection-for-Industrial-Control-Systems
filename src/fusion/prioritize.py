"""告警融合打分与排序（claude.md M9 第 4~6 条），输出 enriched_alerts.json。

- S        = clip(max_score / threshold − 1, 0, 1)
- w_sub    = 该子系统 Top-k 传感器误差占比之和 / 可识别传感器占比之和（和为 1）；
             前缀无法识别的传感器不参与加权，列入 unmapped_sensors
- priority = S × Σ_sub w_sub·(1 + alpha·E_sub)·criticality_sub
             Top-k 全部无法识别时 priority = 0
- E_sub（告警级）= Σ_sub w_sub·E_sub，逐子系统的值见 subsystems[*].E_sub
- 未扫描资产标注为未评估（R_asset = null），计算中按 0；路径上有未评估资产时 risk_underestimated = true

用法：
    python -m src.fusion.prioritize [--alerts JSON] [--vulns JSON] [--output JSON]
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.fusion.asset_graph import AssetGraph
from src.fusion.risk import compute_risk, load_params
from src.utils.i18n import DISCLAIMER_ZH, disclaimer

ROOT = Path(__file__).resolve().parents[2]
DISCLAIMER = DISCLAIMER_ZH  # 中文原文；输出中的免责声明按 ui.lang 取 disclaimer()
VULN_FIELDS = ["asset", "asset_ip", "port", "name", "cvss", "epss", "in_kev", "kev_cves", "qod",
               "r_v", "solution", "solution_type", "nvt_oid"]


class TopK(BaseModel):
    sensor: str
    share: float = Field(ge=0, le=1)


class Alert(BaseModel):
    """alerts.json 单条记录（第 8 节数据契约）；额外字段原样保留。"""
    model_config = ConfigDict(extra="allow")
    alert_id: str
    start: str
    end: str
    max_score: float = Field(ge=0)
    threshold: float = Field(gt=0)
    topk: list[TopK] = Field(min_length=1)


def severity(max_score: float, threshold: float) -> float:
    return min(1.0, max(0.0, max_score / threshold - 1))


def subsystem_weights(topk: list[TopK], graph: AssetGraph) -> tuple[dict[str, float], list[str]]:
    """按误差占比聚合到子系统并归一化；返回 (weights, unmapped_sensors)。"""
    raw: dict[str, float] = defaultdict(float)
    unmapped = []
    for t in topk:
        sid = graph.sensor_to_subsystem(t.sensor)
        if sid is None:
            unmapped.append(t.sensor)
        else:
            raw[sid] += t.share
    total = sum(raw.values())
    if total <= 0:
        return {}, unmapped
    return {sid: s / total for sid, s in raw.items()}, unmapped


def _vuln_view(r: dict) -> dict:
    return {k: r.get(k) for k in VULN_FIELDS}


def enrich_alert(alert: Alert, graph: AssetGraph, risk: dict, params: dict) -> dict:
    s = severity(alert.max_score, alert.threshold)
    weights, unmapped = subsystem_weights(alert.topk, graph)

    subs, factor = [], 0.0
    for sid, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        sub = graph.subsystems[sid]
        rs = risk["subsystems"][sid]
        e = rs["E_sub"]
        factor += w * (1 + params["alpha"] * e) * sub.criticality
        subs.append({"id": sid, "name": sub.name, "criticality": sub.criticality, "weight": w,
                     "E_sub": e, "E_sub_assessed": rs["E_sub_assessed"],
                     "has_assets": bool(rs["upstream_assets"]),
                     "unassessed_assets": rs["unassessed_assets"], "note": sub.note})

    path_assets = sorted({a for sid in weights for a in graph.upstream_assets(sid)})
    vulns = [r for r in risk["vulns"] if r["asset"] in path_assets][: params["max_vulns_per_alert"]]
    review = [r for r in risk["needs_review"] if r["asset"] in path_assets]
    unassessed = [a for a in path_assets if not risk["assets"][a]["assessed"]]

    # 第 8 节：时间字段保留 HAI 原始字符串，内部转为 pandas Timestamp
    start, end = pd.Timestamp(alert.start), pd.Timestamp(alert.end)
    if end < start:
        raise ValueError(f"{alert.alert_id} 的 end 早于 start")
    base = alert.model_dump()
    base["topk"] = [{**t, "subsystem": graph.sensor_to_subsystem(t["sensor"])} for t in base["topk"]]
    return {
        **base,
        "duration_s": int((end - start).total_seconds()) + 1,
        "S": s,
        "subsystems": subs,
        "unmapped_sensors": unmapped,
        "E_sub": sum(x["weight"] * x["E_sub"] for x in subs),
        "priority": s * factor,
        "path": [{"subsystem": sid, "path": p} for sid in weights for p in graph.intrusion_paths(sid)],
        "path_assets": [{"asset": a, **risk["assets"][a]} for a in path_assets],
        # 路径上有未评估（未扫描）资产时，E_sub 按 0 计入，实际风险可能被低估
        "unassessed_assets": unassessed,
        "risk_underestimated": bool(unassessed),
        "vulns": [_vuln_view(r) for r in vulns],
        "needs_review": [_vuln_view(r) for r in review],
        "disclaimer": disclaimer(),
    }


def prioritize(alerts: list[dict], vulns: list[dict], graph: AssetGraph, params: dict) -> dict:
    risk = compute_risk(vulns, graph, params)
    parsed = [Alert.model_validate(a) for a in alerts]
    out = [enrich_alert(a, graph, risk, params) for a in parsed]
    out.sort(key=lambda a: (-a["priority"], -a["S"], a["alert_id"]))
    for i, a in enumerate(out, 1):
        a["rank"] = i
    return {
        "disclaimer": disclaimer(),
        "is_mock": any(a.get("is_mock", False) for a in alerts),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {k: params[k] for k in ("alpha", "beta", "gamma", "min_qod")},
        "asset_risk": risk["assets"],
        "subsystem_risk": {sid: {k: s[k] for k in ("E_sub", "max_asset", "E_sub_assessed", "unassessed_assets")}
                           for sid, s in risk["subsystems"].items()},
        "unmapped_ips": risk["unmapped_ips"],
        "alerts": out,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="alerts.json + vulns_enriched.json → enriched_alerts.json")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--alerts", type=Path, help="默认取 config.yaml 的 fusion.alerts_json")
    p.add_argument("--vulns", type=Path, help="默认取 config.yaml 的 vuln.vulns_enriched_json")
    p.add_argument("--output", type=Path, help="默认取 config.yaml 的 fusion.enriched_alerts_json")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    fcfg = cfg["fusion"]
    graph = AssetGraph.from_yaml(ROOT / fcfg["asset_map"])
    params = load_params(ROOT / fcfg["params"])
    alerts = json.loads((args.alerts or ROOT / fcfg["alerts_json"]).read_text(encoding="utf-8"))
    vulns_src = args.vulns or ROOT / cfg["vuln"]["vulns_enriched_json"]
    vulns = json.loads(vulns_src.read_text(encoding="utf-8"))["vulns"]

    out = prioritize(alerts, vulns, graph, params)
    dst = args.output or ROOT / fcfg["enriched_alerts_json"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[保存] {dst}（is_mock={out['is_mock']}）")
    for a in out["alerts"]:
        subs = " + ".join(f"{s['id']}×{s['weight']:.2f}" for s in a["subsystems"]) or "-"
        print(f"  #{a['rank']} {a['alert_id']:<10} priority={a['priority']:.3f}  S={a['S']:.3f}  "
              f"E_sub={a['E_sub']:.3f}  子系统 {subs}  需核实 {len(a['needs_review'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
