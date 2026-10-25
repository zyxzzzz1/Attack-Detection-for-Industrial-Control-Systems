"""Tests of src/fusion/render_card.py: render a hand-built asset graph + hand-built vulnerabilities + mock alerts into a temporary directory."""

import re
from pathlib import Path

import pytest

from src.fusion.asset_graph import AssetGraph, AssetMap
from src.fusion.prioritize import DISCLAIMER, prioritize
from src.fusion.render_card import build_actions, exposure, fmt_duration, make_env, md_cell, render_all, shown_subsystems
from src.fusion.risk import compute_risk
from tests.test_fusion import PARAMS, alert, amap_dict, vuln

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def meta():
    g = AssetGraph(AssetMap.model_validate(amap_dict()))
    vs = [vuln("192.168.56.11", 9.8, epss=0.99, kev=True, qod=99, name="Ghostcat <b>AJP</b> | pipe"),
          vuln("192.168.56.11", 9.8, epss=0.97, kev=True, qod=30, name="LowQoD KEV"),
          vuln("192.168.56.11", 6.0, epss=0.1, qod=80, name="Medium")]
    alerts = [alert("A-P1", 2.0, [("P1_FT03", 0.7), ("P1_LIT01", 0.3)]),
              alert("A-P4", 1.5, [("P4_ST_PT01", 0.8), ("P9_UNK", 0.2)]),
              alert("A-P3", 1.01, [("P3_LIT01", 1.0)])]
    alerts[0]["mock_case"] = "single subsystem"
    return prioritize(alerts, vs, g, PARAMS)


@pytest.fixture
def rendered(meta, tmp_path):
    render_all(meta, make_env(ROOT / "templates"), tmp_path, ["md", "html"])
    return tmp_path


def read(d: Path, name: str) -> str:
    return (d / name).read_text(encoding="utf-8")


def test_files_written(rendered):
    names = sorted(p.name for p in rendered.iterdir())
    assert names == ["01_A-P1.html", "01_A-P1.md", "02_A-P4.html", "02_A-P4.md",
                     "03_A-P3.html", "03_A-P3.md", "index.html", "index.md"]


@pytest.mark.parametrize("ext", ["md", "html"])
def test_card_contains_all_m10_fields(rendered, ext):
    t = read(rendered, f"01_A-P1.{ext}")
    for s in ["A-P1", "2020-07-15 10:00:00", "2020-07-15 10:00:59", "1 min",   # ID, start/end, duration
              "0.2000", "0.1000",                                          # Score / threshold
              "P1_FT03", "70.0%",                                          # Top-k and shares
              "P1 Boiler", "0.9",                                            # Subsystems and criticality
              "hist → hmi → plc_p1", "ews_a → plc_p1",                     # Intrusion paths
              "KEV", "CVE-2000-0001",                                      # KEV mark on vulnerabilities
              "Investigate these assets first", "OpenVAS remediation advice", "fix",   # Recommended actions and remediation
              "rank 1", "[MOCK]", "single subsystem", DISCLAIMER]:
        assert s in t, f"{ext} card is missing {s!r}"
    assert t.count(DISCLAIMER) >= 2  # Header and footer


@pytest.mark.parametrize("ext", ["md", "html"])
def test_needs_review_block(rendered, ext):
    assert "Needs manual verification" in read(rendered, f"01_A-P1.{ext}")
    assert "LowQoD KEV" in read(rendered, f"01_A-P1.{ext}")
    # P3's upstream does not include ews_a, so nothing needs verification
    p3 = read(rendered, f"03_A-P3.{ext}")
    assert "LowQoD KEV" not in p3 and "have not been assessed (not scanned)" in p3


@pytest.mark.parametrize("ext", ["md", "html"])
def test_p4_note_and_unmapped(rendered, ext):
    t = read(rendered, f"02_A-P4.{ext}")
    assert "HIL note" in t and "P9_UNK" in t and "Unknown prefix" in t
    assert "no control chain attached" in t


def test_html_escapes_and_md_pipes(rendered):
    html = read(rendered, "01_A-P1.html")
    assert "<b>AJP</b>" not in html and "&lt;b&gt;AJP&lt;/b&gt;" in html
    md = read(rendered, "01_A-P1.md")
    row = next(line for line in md.splitlines() if "Ghostcat" in line and line.startswith("| 1 |"))
    assert "\\| pipe" in row and len(re.findall(r"(?<!\\)\|", row)) == 10  # 9 columns -> 10 unescaped pipes


@pytest.mark.parametrize("ext", ["md", "html"])
def test_index_sorted_with_links(rendered, ext):
    t = read(rendered, f"index.{ext}")
    pos = [t.index(a) for a in ("01_A-P1", "02_A-P4", "03_A-P3")]
    assert pos == sorted(pos)
    assert "[MOCK]" in t and DISCLAIMER in t


def test_no_mock_banner_for_real_alerts(tmp_path):
    g = AssetGraph(AssetMap.model_validate(amap_dict()))
    a = alert("REAL", 1.5, [("P1_X", 1.0)])
    del a["is_mock"]
    meta = prioritize([a], [], g, PARAMS)
    render_all(meta, make_env(ROOT / "templates"), tmp_path, ["md", "html"])
    for name in ("01_REAL.md", "01_REAL.html", "index.md", "index.html"):
        t = read(tmp_path, name)
        assert "[MOCK]" not in t and DISCLAIMER in t


def test_actions_dedup_fixes(meta):
    a = meta["alerts"][0]
    acts = build_actions(a)
    assert len(acts["fixes"]) == 1  # The remediation text of both vulnerabilities is "fix"
    assert acts["steps"][0].startswith("Investigate these assets first: ews_a")


def test_helpers():
    assert fmt_duration(1) == "1 s"
    assert fmt_duration(180) == "3 min"
    assert fmt_duration(3725) == "1 h 2 min 5 s"
    assert md_cell("a|b\nc") == "a\\|b c"


def test_tie_break_kev_then_cvss():
    g = AssetGraph(AssetMap.model_validate(amap_dict()))
    vs = [vuln("192.168.56.11", 10.0, epss=0.0, name="cvss10"),
          vuln("192.168.56.11", 7.5, epss=1.0, kev=True, name="kev"),
          vuln("192.168.56.11", 9.8, epss=0.9, name="cvss9.8")]
    r = compute_risk(vs, g, PARAMS)
    assert all(v["r_v"] == 1.0 for v in r["vulns"])
    assert [v["name"] for v in r["vulns"]] == ["kev", "cvss10", "cvss9.8"]


@pytest.mark.parametrize("ext", ["md", "html"])
def test_unassessed_shown_not_zero(rendered, ext):
    t = read(rendered, f"01_A-P1.{ext}")
    assert "This path contains assets that have not been assessed; the actual risk may be underestimated" in t
    assert "may be underestimated" in t
    # hmi is unscanned: shows "Not assessed" (not assessed), not 0.000
    row = next(line for line in t.splitlines() if re.search(r"(\| |<td>)hmi( \||</td>)", line))
    assert "Not assessed" in row and "0.000" not in row
    assert "2 more upstream assets not assessed" in t
    idx = read(rendered, f"index.{ext}")
    assert "Possibly underestimated" in idx


@pytest.mark.parametrize("ext", ["md", "html"])
def test_subsystem_all_unassessed_and_no_assets(tmp_path, ext):
    d = amap_dict()
    for aid in ("ews_a", "ews_b", "hist"):
        d["assets"][aid].update(scanned=False, ip=None)
    g = AssetGraph(AssetMap.model_validate(d))
    meta = prioritize([alert("A", 1.5, [("P1_X", 0.5), ("P4_Y", 0.5)])], [], g, PARAMS)
    render_all(meta, make_env(ROOT / "templates"), tmp_path, [ext])
    t = read(tmp_path, f"01_A.{ext}")
    assert "E_sub not assessed (no upstream asset has been scanned; counted as 0)" in t
    assert "E_sub = 0.000 (no control chain: HIL simulation)" in t and "no associated assets" not in t
    # All path assets unscanned: E_sub shows "Not assessed" (not assessed), not "Possibly underestimated" (may be underestimated) (the warning sentence is still there)
    assert exposure(meta["alerts"][0]) == "not_assessed"
    assert ("0.000 (not assessed: no upstream asset scanned)" if ext == "md" else "Not assessed: no upstream asset scanned</span>") in t
    assert " (possibly underestimated)" not in t and "Possibly underestimated</span>" not in t
    idx = read(tmp_path, f"index.{ext}")
    assert ("0.000 (not assessed: no upstream asset scanned)" if ext == "md" else "Not assessed: no upstream asset scanned</span>") in idx
    assert " (possibly underestimated)" not in idx and "Possibly underestimated</span>" not in idx


def test_no_underestimate_banner_when_all_assessed(tmp_path):
    d = amap_dict()
    d["assets"]["hmi"].update(scanned=True, ip="192.168.56.20")
    d["assets"]["plc_p1"].update(scanned=True, ip="192.168.56.101")
    g = AssetGraph(AssetMap.model_validate(d))
    meta = prioritize([alert("A", 1.5, [("P1_X", 1.0)])], [], g, PARAMS)
    render_all(meta, make_env(ROOT / "templates"), tmp_path, ["md", "html"])
    for name in ("01_A.md", "01_A.html"):
        assert "Possibly underestimated" not in read(tmp_path, name)


@pytest.mark.parametrize("ext", ["md", "html"])
def test_index_columns_tie_rule_and_exposure_text(rendered, meta, ext):
    """Index page: a "score / threshold" (multiple) column before S; the tie-breaking rule is stated; partially unscanned assets show "Possibly underestimated" (may be underestimated),
    and those without upstream assets (P4) get no mark."""
    idx = read(rendered, f"index.{ext}")
    head = next(line for line in idx.splitlines() if "Priority" in line and "E_sub" in line)
    assert head.index("Priority") < head.index("Score / threshold") < head.index("S<" if ext == "html" else "| S |")
    assert "KEV vulns needing review" in head and "priority" not in head
    row = next(line for line in idx.splitlines() if "A-P1" in line) if ext == "md" else idx
    assert "2.00" in row and "1.50" in idx and "1.01" in idx
    assert "Alerts with equal Priority are ordered by S, descending, and then by alert ID, ascending" in idx
    by_id = {a["alert_id"]: a for a in meta["alerts"]}
    assert [exposure(by_id[k]) for k in ("A-P1", "A-P4", "A-P3")] == ["partial", "assessed", "partial"]
    assert idx.count(" (possibly underestimated)" if ext == "md" else "Possibly underestimated</span>") == 2


def test_tie_order_is_s_then_alert_id():
    """Tie rule: equal priority sorted by S descending, then by alert ID ascending; cards, index page and cases all use the rank given by prioritize."""
    g = AssetGraph(AssetMap.model_validate(amap_dict()))
    alerts = [alert("B", 3.0, [("P1_X", 1.0)]), alert("A", 9.0, [("P1_X", 1.0)]), alert("C", 2.5, [("P1_X", 1.0)])]
    out = prioritize(alerts, [], g, PARAMS)["alerts"]
    assert len({a["priority"] for a in out}) == 1 and [a["alert_id"] for a in out] == ["A", "B", "C"]
    assert [a["rank"] for a in out] == [1, 2, 3]


@pytest.mark.parametrize("ext", ["md", "html"])
def test_zero_weight_subsystems_hidden_in_list(tmp_path, ext):
    g = AssetGraph(AssetMap.model_validate(amap_dict()))
    meta = prioritize([alert("A", 1.5, [("P1_X", 0.9997), ("P4_Y", 0.0003)])], [], g, PARAMS)
    a = meta["alerts"][0]
    assert [s["id"] for s in a["subsystems"]] == ["P1", "P4"]             # The computed result is unchanged; it is just not shown in the list
    assert [s["id"] for s in shown_subsystems(a, 1)] == ["P1"] == [s["id"] for s in shown_subsystems(a)]
    render_all(meta, make_env(ROOT / "templates"), tmp_path, [ext])
    idx = read(tmp_path, f"index.{ext}")
    assert "P1 100.0%" in idx and "P4" not in idx
