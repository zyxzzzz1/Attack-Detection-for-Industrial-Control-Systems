"""src/fusion/render_card.py 的测试：用构造资产图 + 构造漏洞 + mock 报警渲染到临时目录。"""

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
    alerts[0]["mock_case"] = "单子系统"
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
    for s in ["A-P1", "2020-07-15 10:00:00", "2020-07-15 10:00:59", "1 分",   # ID、起止、时长
              "0.2000", "0.1000",                                          # 分数 / 阈值
              "P1_FT03", "70.0%",                                          # Top-k 与占比
              "P1 锅炉", "0.9",                                            # 子系统与关键度
              "hist → hmi → plc_p1", "ews_a → plc_p1",                     # 入侵路径
              "KEV", "CVE-2000-0001",                                      # 漏洞 KEV 标注
              "优先排查资产", "修复建议", "fix",                           # 建议动作与修复建议
              "第 1", "[MOCK]", "单子系统", DISCLAIMER]:
        assert s in t, f"{ext} 卡片缺少 {s!r}"
    assert t.count(DISCLAIMER) >= 2  # 顶部与页脚


@pytest.mark.parametrize("ext", ["md", "html"])
def test_needs_review_block(rendered, ext):
    assert "需人工核实" in read(rendered, f"01_A-P1.{ext}")
    assert "LowQoD KEV" in read(rendered, f"01_A-P1.{ext}")
    # P3 的上游不含 ews_a，没有需核实的漏洞
    p3 = read(rendered, f"03_A-P3.{ext}")
    assert "LowQoD KEV" not in p3 and "未评估（未扫描）的资产" in p3


@pytest.mark.parametrize("ext", ["md", "html"])
def test_p4_note_and_unmapped(rendered, ext):
    t = read(rendered, f"02_A-P4.{ext}")
    assert "HIL 说明" in t and "P9_UNK" in t and "未知前缀" in t
    assert "没有挂接控制链" in t


def test_html_escapes_and_md_pipes(rendered):
    html = read(rendered, "01_A-P1.html")
    assert "<b>AJP</b>" not in html and "&lt;b&gt;AJP&lt;/b&gt;" in html
    md = read(rendered, "01_A-P1.md")
    row = next(line for line in md.splitlines() if "Ghostcat" in line and line.startswith("| 1 |"))
    assert "\\| pipe" in row and len(re.findall(r"(?<!\\)\|", row)) == 10  # 9 列 → 10 个未转义竖线


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
    assert len(acts["fixes"]) == 1  # 两条漏洞的修复文本都是 "fix"
    assert acts["steps"][0].startswith("优先排查资产：ews_a")


def test_helpers():
    assert fmt_duration(1) == "1 秒"
    assert fmt_duration(180) == "3 分"
    assert fmt_duration(3725) == "1 小时 2 分 5 秒"
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
    assert "该路径存在未评估资产，实际风险可能被低估" in t
    assert "可能被低估" in t
    # hmi 未扫描：显示"未评估"，不显示 0.000
    row = next(line for line in t.splitlines() if re.search(r"(\| |<td>)hmi( \||</td>)", line))
    assert "未评估" in row and "0.000" not in row
    assert "另有 2 个上游资产未评估" in t
    idx = read(rendered, f"index.{ext}")
    assert "可能低估" in idx


@pytest.mark.parametrize("ext", ["md", "html"])
def test_subsystem_all_unassessed_and_no_assets(tmp_path, ext):
    d = amap_dict()
    for aid in ("ews_a", "ews_b", "hist"):
        d["assets"][aid].update(scanned=False, ip=None)
    g = AssetGraph(AssetMap.model_validate(d))
    meta = prioritize([alert("A", 1.5, [("P1_X", 0.5), ("P4_Y", 0.5)])], [], g, PARAMS)
    render_all(meta, make_env(ROOT / "templates"), tmp_path, [ext])
    t = read(tmp_path, f"01_A.{ext}")
    assert "E_sub 未评估（上游资产均未扫描，按 0 计算）" in t
    assert "E_sub = 0.000（无控制链：HIL 仿真）" in t and "无关联资产" not in t
    # 路径上的资产全部未扫描：E_sub 写"未评估"，不写"可能被低估"（提示语那一句仍在）
    assert exposure(meta["alerts"][0]) == "not_assessed"
    assert ("0.000（未评估：上游资产均未扫描）" if ext == "md" else "未评估：上游资产均未扫描</span>") in t
    assert "（可能被低估）" not in t and "可能被低估</span>" not in t
    idx = read(tmp_path, f"index.{ext}")
    assert ("0.000（未评估：上游资产均未扫描）" if ext == "md" else "未评估：上游资产均未扫描</span>") in idx
    assert "（可能低估）" not in idx and "可能低估</span>" not in idx


def test_no_underestimate_banner_when_all_assessed(tmp_path):
    d = amap_dict()
    d["assets"]["hmi"].update(scanned=True, ip="192.168.56.20")
    d["assets"]["plc_p1"].update(scanned=True, ip="192.168.56.101")
    g = AssetGraph(AssetMap.model_validate(d))
    meta = prioritize([alert("A", 1.5, [("P1_X", 1.0)])], [], g, PARAMS)
    render_all(meta, make_env(ROOT / "templates"), tmp_path, ["md", "html"])
    for name in ("01_A.md", "01_A.html"):
        assert "可能被低估" not in read(tmp_path, name)


@pytest.mark.parametrize("ext", ["md", "html"])
def test_index_columns_tie_rule_and_exposure_text(rendered, meta, ext):
    """索引页：S 前面有"分数 / 阈值"（倍数）一列；写明并列时的排序规则；部分资产未扫描写"可能低估"，
    没有上游资产的（P4）不加任何标注。"""
    idx = read(rendered, f"index.{ext}")
    head = next(line for line in idx.splitlines() if "Priority" in line and "E_sub" in line)
    assert head.index("Priority") < head.index("分数 / 阈值") < head.index("S<" if ext == "html" else "| S |")
    assert "需核实的 KEV 漏洞" in head and "priority" not in head
    row = next(line for line in idx.splitlines() if "A-P1" in line) if ext == "md" else idx
    assert "2.00" in row and "1.50" in idx and "1.01" in idx
    assert "Priority 相同时按 S 降序，仍相同时按告警编号升序" in idx
    by_id = {a["alert_id"]: a for a in meta["alerts"]}
    assert [exposure(by_id[k]) for k in ("A-P1", "A-P4", "A-P3")] == ["partial", "assessed", "partial"]
    assert idx.count("（可能低估）" if ext == "md" else "可能低估</span>") == 2


def test_tie_order_is_s_then_alert_id():
    """并列规则：priority 相同按 S 降序，再相同按告警编号升序；卡片、索引页、案例都用 prioritize 给出的 rank。"""
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
    assert [s["id"] for s in a["subsystems"]] == ["P1", "P4"]             # 计算结果不变，只是列表里不显示
    assert [s["id"] for s in shown_subsystems(a, 1)] == ["P1"] == [s["id"] for s in shown_subsystems(a)]
    render_all(meta, make_env(ROOT / "templates"), tmp_path, [ext])
    idx = read(tmp_path, f"index.{ext}")
    assert "P1 100.0%" in idx and "P4" not in idx
