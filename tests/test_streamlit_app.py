"""Smoke test of app/streamlit_app.py: runs headless with streamlit's built-in AppTest."""

import json
from pathlib import Path

import pytest
import yaml

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
MOCK_ORDER = ["MOCK-004", "MOCK-002", "MOCK-001", "MOCK-005", "MOCK-003"]


def use_mock(at):
    return at.sidebar.radio[0].set_value("mock 报警").run()


def test_app_defaults_to_real_alerts():
    """Real alerts by default: no MOCK notice, only the top N by priority are listed, and the cards are those of the real alerts."""
    fcfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))["fusion"]
    n_real = len(json.loads((ROOT / fcfg["alerts_json"]).read_text(encoding="utf-8")))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "真实报警（LSTM-AE）" and not at.warning
    assert any(str(n_real) in i.value and "包含误报" in i.value for i in at.info)
    assert any("仅用于演示" in c.value for c in at.caption)
    df = at.dataframe[0].value
    assert len(df) == fcfg["max_rendered_alerts"] < n_real and df["告警"].str.startswith("LSTMAE-").all()
    assert list(df["排名"]) == list(range(1, len(df) + 1)) and set(df["相对默认参数"]) == {"—"}
    card = "\n".join(m.value for m in at.markdown)
    assert f"告警 {df['告警'].iloc[0]}" in card and "免责声明" in card and "MOCK" not in card


def test_app_can_switch_back_to_mock():
    at = use_mock(AppTest.from_file(APP, default_timeout=120).run())
    assert not at.exception
    assert any("MOCK" in w.value for w in at.warning)
    df = at.dataframe[0].value
    assert list(df["告警"]) == MOCK_ORDER
    assert set(df["相对默认参数"]) == {"—"}
    card = "\n".join(m.value for m in at.markdown)
    assert "告警 MOCK-004" in card and "免责声明" in card


def test_param_change_updates_ranking():
    at = use_mock(AppTest.from_file(APP, default_timeout=120).run())
    at.sidebar.slider[0].set_value(0.0).run()  # α = 0: exposure risk no longer affects the ranking
    assert not at.exception
    df = at.dataframe[0].value
    assert df.loc[df["告警"] == "MOCK-004", "Priority"].item() == pytest.approx(1.0)


def test_list_columns_tie_rule_and_hidden_zero_weight_subsystems():
    """Alert list: a "score / threshold" (multiple) column before S; the tie-breaking rule stated below the table; subsystems with 0% weight are hidden."""
    fcfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))["fusion"]
    alerts = {a["alert_id"]: a for a in json.loads((ROOT / fcfg["alerts_json"]).read_text(encoding="utf-8"))}
    at = AppTest.from_file(APP, default_timeout=120).run()
    df = at.dataframe[0].value
    cols = list(df.columns)
    assert cols.index("分数 / 阈值") == cols.index("S") - 1 == cols.index("Priority") + 1
    assert "需核实的 KEV 漏洞" in cols and "priority" not in cols and "需核实" not in cols
    for aid, ratio in zip(df["告警"], df["分数 / 阈值"]):
        assert ratio == pytest.approx(alerts[aid]["max_score"] / alerts[aid]["threshold"], abs=0.005)
    assert any("Priority 相同时按 S 降序，仍相同时按告警编号升序" in c.value for c in at.caption)
    assert not df["子系统"].str.contains(r"\b0%").any() and df["子系统"].str.contains("%").all()


def test_path_risk_text_when_no_upstream_asset_is_scanned():
    """All upstream assets unscanned (the mock alert involving only P3): shows "未评估" (not assessed), not "可能低估" (may be underestimated); partially unscanned still shows "可能低估"."""
    at = use_mock(AppTest.from_file(APP, default_timeout=120).run())
    df = at.dataframe[0].value.set_index("告警")
    assert df.loc["MOCK-003", "路径风险"] == "未评估：上游资产均未扫描" and df.loc["MOCK-003", "E_sub"] == 0
    assert df.loc["MOCK-004", "路径风险"] == "可能低估"
    assert len(at.dataframe) == 2 and len(at.table) == 1     # The subsystem table uses st.table: long cells wrap instead of being truncated
    sub = at.table[0].value
    assert sub.index.name == "子系统" and sub.loc["P3", "未评估上游"] == "ews_02、historian、hmi_main、plc_p3"
    assert sub.loc["P3", "E_sub"] == "0.000（未评估：上游资产均未扫描）"
    assert sub.loc["P4", "E_sub"] == "0.000（无控制链：HIL 仿真）" and sub.loc["P4", "未评估上游"] == "-"   # P4 has no assets; it is not "未评估" (not assessed)
    assert sub.loc["P4", "名称"] == "HIL 仿真"


def test_streamlit_config_hides_deploy_button():
    toml = pytest.importorskip("toml")
    assert toml.load(ROOT / ".streamlit" / "config.toml")["client"]["toolbarMode"] == "minimal"
