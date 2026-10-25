"""Tests of src/fusion: asset_graph / risk / prioritize, with a small hand-built asset graph and hand-built vulnerabilities."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.make_mock_alerts import build_alerts, load_sensor_names
from src.fusion.asset_graph import AssetGraph, AssetMap
from src.fusion.prioritize import DISCLAIMER, prioritize, severity
from src.fusion.risk import compute_risk, vuln_risk

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"alpha": 1.0, "beta": 1.0, "gamma": 1.5, "min_qod": 70, "max_vulns_per_alert": 10}


def amap_dict() -> dict:
    return {
        "subsystems": {
            "P1": {"name": "Boiler", "criticality": 0.9, "sensor_prefix": "P1_", "assets": ["plc_p1", "hmi", "ews_a"]},
            "P3": {"name": "Water treatment", "criticality": 0.5, "sensor_prefix": "P3_", "assets": ["plc_p3", "hmi", "ews_b"]},
            "P4": {"name": "HIL", "criticality": 0.6, "sensor_prefix": "P4_", "assets": [], "note": "HIL note"},
        },
        "assets": {
            "ews_a": {"role": "Engineering workstation", "purdue_level": 2, "ip": "192.168.56.11", "scanned": True},
            "ews_b": {"role": "Engineering workstation", "purdue_level": 2, "ip": "192.168.56.12", "scanned": True},
            "hmi": {"role": "HMI", "purdue_level": 2, "ip": None, "scanned": False},
            "hist": {"role": "Historian", "purdue_level": 3, "ip": "192.168.56.30", "scanned": True},
            "plc_p1": {"role": "PLC", "purdue_level": 1, "ip": None, "scanned": False},
            "plc_p3": {"role": "PLC", "purdue_level": 1, "ip": None, "scanned": False},
        },
        "edges": [["hist", "hmi"], ["ews_a", "plc_p1"], ["hmi", "plc_p1"], ["ews_b", "plc_p3"], ["hmi", "plc_p3"]],
    }


@pytest.fixture
def graph() -> AssetGraph:
    return AssetGraph(AssetMap.model_validate(amap_dict()))


def vuln(ip, cvss, epss=0.0, kev=False, qod=80, name="v"):
    return {"asset_ip": ip, "port": "80/tcp", "nvt_oid": "1.3.6.1.4.1.25623.1", "name": name,
            "cvss": cvss, "cves": [], "solution": "fix", "qod": qod, "threat": "High",
            "solution_type": "VendorFix", "cvss_version": "v3",
            "in_kev": kev, "kev_cves": ["CVE-2000-0001"] if kev else [], "epss": epss}


def alert(aid, ratio, topk, thr=0.1):
    return {"alert_id": aid, "start": "2020-07-15 10:00:00", "end": "2020-07-15 10:00:59",
            "max_score": thr * ratio, "threshold": thr,
            "topk": [{"sensor": s, "share": sh} for s, sh in topk], "is_mock": True}


# ------------------------------------------------------------ asset_graph

def test_real_asset_map_matches_hai_subsystems():
    g = AssetGraph.from_yaml(ROOT / "config" / "asset_map.yaml")
    sensors = load_sensor_names(ROOT / "data" / "samples" / "hai_columns.txt")
    hai_subs = {s.split("_")[0] for s in sensors}
    assert hai_subs == {"P1", "P2", "P3", "P4"} == set(g.subsystems)
    assert all(g.sensor_to_subsystem(s) == s.split("_")[0] for s in sensors)
    assert g.subsystems["P4"].assets == [] and g.subsystems["P4"].note


def test_validation_rejects_bad_refs():
    d = amap_dict()
    d["subsystems"]["P1"]["assets"].append("ghost")
    d["edges"].append(["hmi", "nowhere"])
    with pytest.raises(ValidationError) as e:
        AssetMap.model_validate(d)
    assert "ghost" in str(e.value) and "nowhere" in str(e.value)


def test_validation_scanned_needs_ip():
    d = amap_dict()
    d["assets"]["hmi"]["scanned"] = True
    with pytest.raises(ValidationError, match="must have an ip"):
        AssetMap.model_validate(d)


def test_sensor_lookup_unknown_prefix(graph):
    assert graph.sensor_to_subsystem("P1_FT03") == "P1"
    assert graph.sensor_to_subsystem("P2_SIT01") is None
    assert graph.sensor_to_subsystem("attack") is None


def test_upstream_and_paths(graph):
    assert graph.upstream_assets("P1") == ["ews_a", "hist", "hmi", "plc_p1"]
    assert graph.intrusion_paths("P1") == [["ews_a", "plc_p1"], ["hist", "hmi", "plc_p1"]]
    assert graph.upstream_assets("P4") == [] and graph.intrusion_paths("P4") == []


def test_paths_terminate_on_cycle():
    d = amap_dict()
    d["edges"].append(["hmi", "hist"])  # Build a cycle hist <-> hmi
    g = AssetGraph(AssetMap.model_validate(d))
    paths = g.intrusion_paths("P1")
    assert ["ews_a", "plc_p1"] in paths and ["hist", "hmi", "plc_p1"] in paths
    assert all(len(p) == len(set(p)) for p in paths)


# ------------------------------------------------------------ risk

def test_vuln_risk_formula():
    assert vuln_risk(vuln("x", 5.0, epss=0.2), 1.0, 1.5) == pytest.approx(0.6)
    assert vuln_risk(vuln("x", 5.0, epss=0.2, kev=True), 1.0, 1.5) == pytest.approx(0.9)
    assert vuln_risk(vuln("x", 9.8, epss=0.9, kev=True), 1.0, 1.5) == 1.0  # Clipped to 1


def test_asset_without_vulns_and_unscanned_plc(graph):
    r = compute_risk([vuln("192.168.56.11", 5.0, epss=0.2)], graph, PARAMS)
    a = r["assets"]
    assert a["ews_a"] == {"R_asset": pytest.approx(0.6), "assessed": True, "status": "scanned",
                          "n_vulns": 1, "n_needs_review": 0}
    # Scanned with no vulnerabilities: the risk really is 0
    assert a["ews_b"]["R_asset"] == 0.0 and a["ews_b"]["assessed"] and a["ews_b"]["status"] == "no_vulns"
    # Not scanned (PLC): not assessed, R_asset is None rather than 0
    assert a["plc_p1"]["R_asset"] is None and a["plc_p1"]["assessed"] is False
    assert a["plc_p1"]["status"] == "not_scanned"
    assert r["subsystems"]["P1"]["E_sub"] == pytest.approx(0.6)
    assert r["subsystems"]["P1"]["max_asset"] == "ews_a"
    # None of P3's upstream assets have vulnerabilities
    assert r["subsystems"]["P3"]["E_sub"] == 0.0 and r["subsystems"]["P3"]["max_asset"] is None
    assert r["subsystems"]["P4"]["E_sub"] == 0.0


def test_unassessed_counted_as_zero_but_flagged(graph):
    r = compute_risk([], graph, PARAMS)
    p1 = r["subsystems"]["P1"]
    assert p1["unassessed_assets"] == ["hmi", "plc_p1"]
    assert p1["E_sub_assessed"] is True  # ews_a and hist are scanned
    assert p1["E_sub"] == 0.0
    # All upstream assets unscanned -> E_sub computed as 0 but marked not assessed
    d = amap_dict()
    for aid in ("ews_a", "ews_b", "hist"):
        d["assets"][aid].update(scanned=False, ip=None)
    g = AssetGraph(AssetMap.model_validate(d))
    r = compute_risk([], g, PARAMS)
    assert r["subsystems"]["P1"]["E_sub"] == 0.0 and r["subsystems"]["P1"]["E_sub_assessed"] is False
    # No upstream assets (P4): no unassessed assets
    assert r["subsystems"]["P4"]["unassessed_assets"] == [] and r["subsystems"]["P4"]["E_sub_assessed"] is False


def test_risk_underestimated_flag(graph):
    out = prioritize([alert("A", 1.5, [("P1_X", 1.0)])], [], graph, PARAMS)
    a = out["alerts"][0]
    assert a["risk_underestimated"] is True and a["unassessed_assets"] == ["hmi", "plc_p1"]
    hmi = next(x for x in a["path_assets"] if x["asset"] == "hmi")
    assert hmi["R_asset"] is None and hmi["assessed"] is False
    assert out["asset_risk"]["hmi"]["R_asset"] is None
    assert out["subsystem_risk"]["P1"]["unassessed_assets"] == ["hmi", "plc_p1"]
    # No warning when all assets are scanned
    d = amap_dict()
    d["assets"]["hmi"].update(scanned=True, ip="192.168.56.20")
    d["assets"]["plc_p1"].update(scanned=True, ip="192.168.56.101")
    g = AssetGraph(AssetMap.model_validate(d))
    a = prioritize([alert("A", 1.5, [("P1_X", 1.0)])], [], g, PARAMS)["alerts"][0]
    assert a["risk_underestimated"] is False and a["unassessed_assets"] == []
    # P4 has no associated assets: not counted as not assessed
    a = prioritize([alert("B", 1.5, [("P4_X", 1.0)])], [], graph, PARAMS)["alerts"][0]
    assert a["risk_underestimated"] is False


def test_upstream_reachability_in_E_sub(graph):
    # historian is not in P1's assets, but is reachable backwards via hmi
    r = compute_risk([vuln("192.168.56.30", 7.0)], graph, PARAMS)
    assert r["subsystems"]["P1"]["E_sub"] == pytest.approx(0.7)
    assert r["subsystems"]["P3"]["E_sub"] == pytest.approx(0.7)


def test_low_qod_kev_goes_to_needs_review(graph):
    vs = [vuln("192.168.56.11", 9.8, epss=0.9, kev=True, qod=30, name="kev-low"),
          vuln("192.168.56.11", 9.0, epss=0.0, kev=False, qod=30, name="plain-low"),
          vuln("192.168.56.11", 4.0, epss=0.0, kev=False, qod=70, name="ok")]
    r = compute_risk(vs, graph, PARAMS)
    assert [v["name"] for v in r["vulns"]] == ["ok"]  # qod == min_qod is kept
    assert [v["name"] for v in r["needs_review"]] == ["kev-low"]
    assert r["needs_review"][0]["asset"] == "ews_a"
    assert r["assets"]["ews_a"]["R_asset"] == pytest.approx(0.4)  # Low QoD is excluded


def test_unmapped_ip(graph):
    r = compute_risk([vuln("10.0.0.1", 9.0)], graph, PARAMS)
    assert r["unmapped_ips"] == ["10.0.0.1"] and r["vulns"] == []


# ------------------------------------------------------------ prioritize

@pytest.mark.parametrize("ratio,expected", [(0.5, 0.0), (1.0, 0.0), (1.25, 0.25), (2.0, 1.0), (5.0, 1.0)])
def test_severity_clipped(ratio, expected):
    assert severity(0.1 * ratio, 0.1) == pytest.approx(expected)


def test_cross_subsystem_weighting(graph):
    vs = [vuln("192.168.56.11", 5.0)]  # E_P1 = 0.5, E_P3 = 0
    a = alert("A", 1.5, [("P1_X", 0.3), ("P3_Y", 0.5), ("P3_Z", 0.2)])
    out = prioritize([a], vs, graph, PARAMS)["alerts"][0]
    w = {s["id"]: s["weight"] for s in out["subsystems"]}
    assert w == pytest.approx({"P1": 0.3, "P3": 0.7})
    expect = 0.5 * (0.3 * (1 + 0.5) * 0.9 + 0.7 * (1 + 0.0) * 0.5)
    assert out["priority"] == pytest.approx(expect)
    assert out["E_sub"] == pytest.approx(0.3 * 0.5)
    assert [s["id"] for s in out["subsystems"]] == ["P3", "P1"]  # Sorted by weight, descending


def test_unknown_prefix_dropped_and_renormalized(graph):
    a = alert("A", 1.5, [("P1_X", 0.3), ("P2_UNKNOWN", 0.5), ("P3_Y", 0.2)])
    out = prioritize([a], [], graph, PARAMS)["alerts"][0]
    assert out["unmapped_sensors"] == ["P2_UNKNOWN"]
    w = {s["id"]: s["weight"] for s in out["subsystems"]}
    assert w == pytest.approx({"P1": 0.6, "P3": 0.4})


def test_all_unknown_sensors_gives_zero_priority(graph):
    out = prioritize([alert("A", 3.0, [("P9_X", 1.0)])], [], graph, PARAMS)["alerts"][0]
    assert out["subsystems"] == [] and out["priority"] == 0.0 and out["path"] == []


def test_p4_note_and_no_assets(graph):
    out = prioritize([alert("A", 1.5, [("P4_ST_PT01", 1.0)])], [vuln("192.168.56.11", 9.0)], graph, PARAMS)
    a = out["alerts"][0]
    assert a["subsystems"][0]["note"] == "HIL note"
    assert a["E_sub"] == 0.0 and a["vulns"] == [] and a["path"] == []
    assert a["priority"] == pytest.approx(0.5 * 1 * 0.6)


def test_needs_review_attached_to_alert(graph):
    vs = [vuln("192.168.56.11", 9.8, epss=0.9, kev=True, qod=30)]
    out = prioritize([alert("A", 1.5, [("P1_X", 1.0)]), alert("B", 1.5, [("P3_Y", 1.0)])], vs, graph, PARAMS)
    by_id = {a["alert_id"]: a for a in out["alerts"]}
    assert len(by_id["A"]["needs_review"]) == 1 and by_id["A"]["vulns"] == []
    assert by_id["B"]["needs_review"] == []  # ews_a is not upstream of P3


def test_ranking_disclaimer_and_mock_flag(graph):
    alerts = [alert("LOW", 1.1, [("P3_Y", 1.0)]), alert("HIGH", 3.0, [("P1_X", 1.0)])]
    out = prioritize(alerts, [], graph, PARAMS)
    assert [a["alert_id"] for a in out["alerts"]] == ["HIGH", "LOW"]
    assert [a["rank"] for a in out["alerts"]] == [1, 2]
    assert out["disclaimer"] == DISCLAIMER and all(a["disclaimer"] == DISCLAIMER for a in out["alerts"])
    assert out["is_mock"] is True and all(a["is_mock"] for a in out["alerts"])
    assert out["alerts"][0]["duration_s"] == 60


def test_mock_alerts_use_real_sensors():
    sensors = load_sensor_names(ROOT / "data" / "samples" / "hai_columns.txt")
    assert len(sensors) == 79
    alerts = build_alerts(sensors)
    assert len(alerts) == 5 and all(a["is_mock"] is True for a in alerts)
    with pytest.raises(ValueError, match="do not exist in HAI"):
        build_alerts(sensors - {"P1_FT03"})
    # The saved mock file matches the generated output
    saved = json.loads((ROOT / "data" / "mock" / "alerts.json").read_text(encoding="utf-8"))
    assert saved == alerts


def test_end_before_start_rejected(graph):
    a = alert("A", 1.5, [("P1_X", 1.0)])
    a["end"] = "2020-07-15 09:59:59"
    with pytest.raises(ValueError, match="earlier than start"):
        prioritize([a], [], graph, PARAMS)
