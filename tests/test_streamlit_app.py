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
    return at.sidebar.radio[0].set_value("Mock alerts").run()


def test_app_defaults_to_real_alerts():
    """Real alerts by default: no MOCK notice, only the top N by priority are listed, and the cards are those of the real alerts."""
    fcfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))["fusion"]
    n_real = len(json.loads((ROOT / fcfg["alerts_json"]).read_text(encoding="utf-8")))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "Real alerts (LSTM-AE)" and not at.warning
    assert any(str(n_real) in i.value and "include false alarms" in i.value for i in at.info)
    assert any("for demonstration only" in c.value for c in at.caption)
    df = at.dataframe[0].value
    assert len(df) == fcfg["max_rendered_alerts"] < n_real and df["Alert"].str.startswith("LSTMAE-").all()
    assert list(df["Rank"]) == list(range(1, len(df) + 1)) and set(df["vs. default parameters"]) == {"—"}
    card = "\n".join(m.value for m in at.markdown)
    assert f"Alert {df['Alert'].iloc[0]}" in card and "Disclaimer" in card and "MOCK" not in card


def test_app_can_switch_back_to_mock():
    at = use_mock(AppTest.from_file(APP, default_timeout=120).run())
    assert not at.exception
    assert any("MOCK" in w.value for w in at.warning)
    df = at.dataframe[0].value
    assert list(df["Alert"]) == MOCK_ORDER
    assert set(df["vs. default parameters"]) == {"—"}
    card = "\n".join(m.value for m in at.markdown)
    assert "Alert MOCK-004" in card and "Disclaimer" in card


def test_param_change_updates_ranking():
    at = use_mock(AppTest.from_file(APP, default_timeout=120).run())
    at.sidebar.slider[0].set_value(0.0).run()  # α = 0: exposure risk no longer affects the ranking
    assert not at.exception
    df = at.dataframe[0].value
    assert df.loc[df["Alert"] == "MOCK-004", "Priority"].item() == pytest.approx(1.0)


def test_list_columns_tie_rule_and_hidden_zero_weight_subsystems():
    """Alert list: a "score / threshold" (multiple) column before S; the tie-breaking rule stated below the table; subsystems with 0% weight are hidden."""
    fcfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))["fusion"]
    alerts = {a["alert_id"]: a for a in json.loads((ROOT / fcfg["alerts_json"]).read_text(encoding="utf-8"))}
    at = AppTest.from_file(APP, default_timeout=120).run()
    df = at.dataframe[0].value
    cols = list(df.columns)
    assert cols.index("Score / threshold") == cols.index("S") - 1 == cols.index("Priority") + 1
    assert "KEV vulns needing review" in cols and "priority" not in cols and "Needs review" not in cols
    for aid, ratio in zip(df["Alert"], df["Score / threshold"]):
        assert ratio == pytest.approx(alerts[aid]["max_score"] / alerts[aid]["threshold"], abs=0.005)
    assert any("Alerts with equal Priority are ordered by S, descending, and then by alert ID, ascending" in c.value for c in at.caption)
    assert not df["Subsystem"].str.contains(r"\b0%").any() and df["Subsystem"].str.contains("%").all()


def test_path_risk_text_when_no_upstream_asset_is_scanned():
    """All upstream assets unscanned (the mock alert involving only P3): shows "Not assessed" (not assessed), not "Possibly underestimated" (may be underestimated); partially unscanned still shows "Possibly underestimated"."""
    at = use_mock(AppTest.from_file(APP, default_timeout=120).run())
    df = at.dataframe[0].value.set_index("Alert")
    assert df.loc["MOCK-003", "Path risk"] == "Not assessed: no upstream asset scanned" and df.loc["MOCK-003", "E_sub"] == 0
    assert df.loc["MOCK-004", "Path risk"] == "Possibly underestimated"
    assert len(at.dataframe) == 2 and len(at.table) == 1     # The subsystem table uses st.table: long cells wrap instead of being truncated
    sub = at.table[0].value
    assert sub.index.name == "Subsystem" and sub.loc["P3", "Unassessed upstream"] == "ews_02, historian, hmi_main, plc_p3"
    assert sub.loc["P3", "E_sub"] == "0.000 (not assessed: no upstream asset scanned)"
    assert sub.loc["P4", "E_sub"] == "0.000 (no control chain: HIL simulation)" and sub.loc["P4", "Unassessed upstream"] == "-"   # P4 has no assets; it does not count as not assessed
    assert sub.loc["P4", "Name"] == "HIL simulation"


def test_streamlit_config_hides_deploy_button():
    toml = pytest.importorskip("toml")
    assert toml.load(ROOT / ".streamlit" / "config.toml")["client"]["toolbarMode"] == "minimal"
