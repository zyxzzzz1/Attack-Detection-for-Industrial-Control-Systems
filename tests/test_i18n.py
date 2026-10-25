"""English localization of the UI and report material (ui.lang): only display text changes, never any computation, field name or number."""

import json
import re
import string
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest
import yaml

from scripts import i18n_keys, make_cases
from src.fusion.asset_graph import AssetGraph
from src.fusion.prioritize import prioritize
from src.fusion.render_card import fmt_duration, make_env, render_all
from src.fusion.risk import load_params
from src.utils import i18n
from src.utils.paths import ROOT, load_config

NUM = re.compile(r"\d+(?:\.\d+)?")
CJK = re.compile(r"[一-鿿　-〿＀-￯]")
EN_DISCLAIMER = "Asset mapping and vulnerability data come from a lab environment and are for demonstration only."
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(ROOT / "app" / "streamlit_app.py")


def numbers(text: str) -> Counter:
    return Counter(NUM.findall(text))


def use(monkeypatch, lang: str | None):
    if lang is None:
        monkeypatch.delenv(i18n.ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(i18n.ENV_VAR, lang)
    i18n.configure(None)


def test_default_language_is_english_and_can_be_overridden(monkeypatch):
    cfg = load_config()
    assert cfg["ui"]["lang"] == "en"
    use(monkeypatch, None)
    assert i18n.get_lang() == "en" and i18n.t("告警") == "Alert"
    i18n.configure({"ui": {"lang": "zh"}})           # Config passed by the caller
    assert i18n.get_lang() == "zh" and i18n.t("告警") == "告警"
    monkeypatch.setenv(i18n.ENV_VAR, "en")           # The environment variable takes precedence
    assert i18n.get_lang() == "en"
    monkeypatch.setenv(i18n.ENV_VAR, "fr")
    with pytest.raises(ValueError):
        i18n.get_lang()
    use(monkeypatch, "en")
    with pytest.raises(KeyError):                    # A missing English translation raises an error instead of silently falling back to Chinese
        i18n.t("这句话没有译文")
    assert i18n.t_data("没有译文的数据文字") == "没有译文的数据文字" and i18n.t_data(None) is None


def test_disclaimer_is_translated_exactly(monkeypatch):
    use(monkeypatch, "en")
    assert i18n.disclaimer() == EN_DISCLAIMER
    use(monkeypatch, "zh")
    assert i18n.disclaimer() == "资产映射与漏洞数据来自实验环境，仅用于演示"


def test_catalog_is_complete_and_keeps_placeholders_and_digits():
    """The translation table matches the display text in code and templates one-to-one; placeholders are identical; translations neither add nor drop numbers; translations contain no Chinese."""
    catalog = yaml.safe_load(i18n.CATALOG.read_text(encoding="utf-8"))
    need = i18n_keys.required_keys()
    assert sorted(need - set(catalog)) == [] and sorted(set(catalog) - need) == []
    fields = lambda s: sorted(f for _, f, _, _ in string.Formatter().parse(s) if f)  # noqa: E731
    for zh, en in catalog.items():
        assert fields(zh) == fields(en), zh
        assert numbers(zh) == numbers(en), zh
        assert not CJK.search(en), en
    assert len(catalog) > 150


def fusion_inputs():
    cfg = load_config()
    fcfg = cfg["fusion"]
    graph = AssetGraph.from_yaml(ROOT / fcfg["asset_map"])
    params = load_params(ROOT / fcfg["params"])
    vulns = json.loads((ROOT / cfg["vuln"]["vulns_enriched_json"]).read_text(encoding="utf-8"))["vulns"]
    real = json.loads((ROOT / fcfg["alerts_json"]).read_text(encoding="utf-8"))
    mock = json.loads((ROOT / fcfg["mock_alerts_json"]).read_text(encoding="utf-8"))
    return graph, params, vulns, {"real": real, "mock": mock}


def strip_text(obj):
    """Drop the fields that are only display text (disclaimer, generation time) and compare the rest item by item."""
    if isinstance(obj, dict):
        return {k: strip_text(v) for k, v in obj.items() if k not in ("disclaimer", "generated_at")}
    if isinstance(obj, list):
        return [strip_text(v) for v in obj]
    return obj


@pytest.mark.parametrize("source", ["real", "mock"])
def test_fusion_output_and_cards_have_identical_numbers_in_both_languages(monkeypatch, tmp_path, source):
    """Generate the same alerts once in Chinese and once in English: the fusion JSON is identical item by item apart from the disclaimer (field names, ranks, all values);
    the numbers in the cards and the index page are exactly the same; no Chinese remains in the English version."""
    graph, params, vulns, alerts = fusion_inputs()
    results, texts = {}, {}
    for lang in ("zh", "en"):
        use(monkeypatch, lang)
        results[lang] = prioritize(alerts[source], vulns, graph, params)
        # The generation time is not compared: if the two renders straddle a second boundary, the time shown on the page differs
        results[lang]["generated_at"] = "2026-01-01T00:00:00+00:00"
        out = tmp_path / lang
        render_all(results[lang], make_env(ROOT / "templates"), out, ["md", "html"], 8)
        texts[lang] = {p.name: p.read_text(encoding="utf-8") for p in sorted(out.iterdir())}
    assert strip_text(results["zh"]) == strip_text(results["en"])
    assert results["en"]["disclaimer"] == EN_DISCLAIMER and results["en"]["alerts"][0]["disclaimer"] == EN_DISCLAIMER
    assert "仅用于演示" in results["zh"]["disclaimer"]
    assert list(texts["zh"]) == list(texts["en"]) and len(texts["en"]) == 2 * (min(8, len(alerts[source])) + 1)
    for name in texts["zh"]:
        assert numbers(texts["zh"][name]) == numbers(texts["en"][name]), name
        assert not CJK.search(texts["en"][name]), (name, CJK.findall(texts["en"][name])[:5])
        assert EN_DISCLAIMER in texts["en"][name] and "仅用于演示" in texts["zh"][name]
    card = next(v for k, v in texts["en"].items() if k.startswith("01_") and k.endswith(".md"))
    assert "Share of total sensor error" in card and "## Recommended actions" in card and "priority rank 1 /" in card
    assert '<html lang="en">' in texts["en"]["index.html"] and '<html lang="zh-CN">' in texts["zh"]["index.html"]
    if source == "mock":
        assert "[MOCK]" in texts["en"]["index.md"] and "not model output" in texts["en"]["index.md"]
    else:
        assert "include false alarms" in texts["en"]["index.md"] and "MOCK" not in texts["en"]["index.md"]


def test_duration_text(monkeypatch):
    use(monkeypatch, "en")
    assert fmt_duration(237) == "3 min 57 s" and fmt_duration(3600) == "1 h" and fmt_duration(0) == "0 s"
    use(monkeypatch, "zh")
    assert fmt_duration(237) == "3 分 57 秒"


def run_app(monkeypatch, lang: str | None):
    use(monkeypatch, lang)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    return at


def test_streamlit_is_english_by_default_with_same_values(monkeypatch):
    """Without UI_LANG the config default (English) is shown; compared with the Chinese UI, the values, alert IDs and ranks in the list are exactly the same."""
    en = run_app(monkeypatch, None)
    assert en.title[0].value == "Process anomaly + vulnerability context: fused alerts"
    assert any(EN_DISCLAIMER in c.value for c in en.caption) and not en.warning
    assert any("include false alarms" in i.value for i in en.info)
    df_en = en.dataframe[0].value
    assert list(df_en.columns) == ["Rank", "vs. default parameters", "Alert", "Start time", "Priority", "Score / threshold", "S",
                                   "Subsystem", "E_sub", "Path risk", "KEV vulns needing review"]
    shown = "\n".join([*(m.value for m in en.markdown), *(c.value for c in en.caption), *(i.value for i in en.info),
                       *(h.value for h in en.header), *(s.value for s in en.subheader)])
    assert not CJK.search(shown), CJK.findall(shown)[:10]
    assert en.sidebar.radio[0].label == "Alert source" and [s.label for s in en.sidebar.slider][0].startswith("α (")

    zh = run_app(monkeypatch, "zh")
    df_zh = zh.dataframe[0].value
    assert list(df_zh["告警"]) == list(df_en["Alert"]) and list(df_zh["排名"]) == list(df_en["Rank"])
    for a, b in (("Priority", "Priority"), ("分数 / 阈值", "Score / threshold"), ("S", "S"), ("E_sub", "E_sub"),
                 ("需核实的 KEV 漏洞", "KEV vulns needing review"), ("起始时间", "Start time")):
        assert list(df_zh[a]) == list(df_en[b])
    for a, b in ((zh.dataframe[1], en.dataframe[1]), (zh.table[0], en.table[0])):   # Asset risk and subsystem exposure risk tables: identical values
        assert a.value.shape == b.value.shape
        assert numbers(a.value.to_csv()) == numbers(b.value.to_csv())
    sub = en.table[0].value
    assert sub.loc["P4", "E_sub"] == "0.000 (no control chain: HIL simulation)"
    assert sub.loc["P3", "E_sub"] == "0.000 (not assessed: no upstream asset scanned)"
    assert sub.loc["P4", "Name"] == "HIL simulation"

    mock = run_app(monkeypatch, "en")
    mock.sidebar.radio[0].set_value("mock 报警").run()
    assert not mock.exception and any("MOCK" in w.value and "not model output" in w.value for w in mock.warning)
    assert list(mock.dataframe[0].value["Alert"])[0] == "MOCK-004"


def test_cases_text_has_identical_numbers_in_both_languages(monkeypatch):
    """Case document: for the same input, the numbers are identical in Chinese and English, and the English version contains no Chinese."""
    def alert(aid, rank, score, subs):
        return {"alert_id": aid, "rank": rank, "priority": 1.4122, "S": 1.0, "E_sub": 0.677, "start": "2020-07-28 12:43:22",
                "end": "2020-07-28 12:47:18", "duration_s": 237, "max_score": score, "threshold": 0.2369,
                "topk": [{"sensor": "P4_ST_PT01", "share": 0.248}, {"sensor": "P1_PIT01", "share": 0.22}],
                "subsystems": [{"id": k, "weight": w, "criticality": 0.9, "E_sub": 1.0} for k, w in subs.items()],
                "path": [{"subsystem": "P1", "path": ["ews_01", "plc_p1"]}], "unassessed_assets": ["historian", "plc_p1"],
                "vulns": [{"asset": "ews_01", "name": "Ghostcat", "nvt_oid": "1.2.3"}], "needs_review": [1, 2]}
    events = pd.DataFrame([{"event_id": 34, "file": "test4.csv.gz", "start_time": "2020-07-28 12:43:03",
                            "end_time": "2020-07-28 12:47:16", "duration_s": 254, "subsystems": "P1", "delay_s": 19},
                           {"event_id": 35, "file": "test4.csv.gz", "start_time": "2020-07-28 14:00:00",
                            "end_time": "2020-07-28 14:02:00", "duration_s": 121, "subsystems": "P1", "delay_s": 0}])
    real = NS(category="evaluable", hit=False, primary_event_id=34, file="test4.csv.gz", event_subsystems="P1")
    false = NS(category="non_overlap", hit=None, primary_event_id=None, file="test4.csv.gz", event_subsystems=float("nan"))
    picked = {2: (real, alert("X-2", 50, 5.4883, {"P1": 0.68, "P4": 0.32})), 3: (false, alert("X-3", 28, 0.5164, {"P1": 1.0}))}
    out = {}
    for lang in ("zh", "en"):
        use(monkeypatch, lang)
        lines = []
        for k, (title, _) in zip((2, 3), make_cases.CASES[1:]):
            row, a = picked[k]
            lines += make_cases.describe(i18n.t("案例 {k}　").format(k=k) + i18n.t(title), row, a, events, 1703, f"case{k}.md")
        out[lang] = "\n".join(lines + make_cases.observations(picked))
    assert numbers(out["zh"]) == numbers(out["en"]) and not CJK.search(out["en"]), CJK.findall(out["en"])[:10]
    assert "Case 2 — a real attack with wrong attribution: X-2" in out["en"] and "is a false alarm" in out["en"]
    assert "案例 2　归因错误的真实攻击：X-2" in out["zh"] and "capped at 1" in out["en"]
    # P4 is coupled with P1, P2 and P3 (official HAI description: the pumps and valves of P3 are controlled by the pumped-storage hydropower model)
    assert "P4 与 P1、P2、P3 都有物理耦合" in out["zh"] and "建议同时排查 P1 / P2 / P3。" in out["zh"]
    assert "coupled with P1, P2 and P3" in out["en"] and "investigating P1 / P2 / P3 as well" in out["en"]


def test_p4_note_and_exposure_text_on_real_cards(monkeypatch, tmp_path):
    """A real alert card involving P4: the note states that P4 is coupled with P1, P2 and P3; P4's E_sub wording matches the subsystem table."""
    graph, params, vulns, alerts = fusion_inputs()
    assert "P1、P2、P3" in graph.subsystems["P4"].note and graph.subsystems["P4"].note.count("P1/P2/P3") == 2
    for lang, note, esub in (("zh", "建议同时排查 P1/P2/P3 相关资产", "E_sub = 0.000（无控制链：HIL 仿真）"),
                             ("en", "the assets related to P1/P2/P3 should be investigated as well",
                              "E_sub = 0.000 (no control chain: HIL simulation)")):
        use(monkeypatch, lang)
        meta = prioritize(alerts["real"], vulns, graph, params)
        one = next(a for a in meta["alerts"] if any(s["id"] == "P4" for s in a["subsystems"]))
        out = tmp_path / lang
        render_all({**meta, "alerts": [one]}, make_env(ROOT / "templates"), out, ["md", "html"])
        for ext in ("md", "html"):
            text = (out / f"{one['rank']:02d}_{one['alert_id']}.{ext}").read_text(encoding="utf-8")
            assert note in text and esub in text and "no associated assets" not in text and "无关联资产" not in text


def test_committed_cases_and_card_snapshots_are_in_the_default_language():
    """The tracked case document and the three case cards are generated in the default language (English), with the accurately translated disclaimer."""
    report = ROOT / load_config()["ml"]["report_dir"]
    files = [report / "cases.md", *sorted((report / "cases").glob("*.md"))]
    assert len(files) == 4
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert EN_DISCLAIMER in text and not CJK.search(text), (path.name, CJK.findall(text)[:10])
