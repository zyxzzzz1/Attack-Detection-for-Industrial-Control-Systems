"""E5: Streamlit demo -- alert list + alert cards + asset risk; sidebar parameters recompute live (writes no files).

Data: by default the real alerts fusion.alerts_json (all alarm segments of LSTM-AE); the sidebar can switch to the mock
alerts fusion.mock_alerts_json; plus vuln.vulns_enriched_json and the asset graph. All alerts take part in fusion scoring;
the list and the cards show only the top fusion.max_rendered_alerts by priority. Cards reuse templates/alert_card.md.j2,
so their content matches the offline cards.

Usage:
    .venv/bin/streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # streamlit runs this as a script, so the project root has to be added manually
    sys.path.insert(0, str(ROOT))

from src.fusion.asset_graph import AssetGraph  # noqa: E402
from src.fusion.prioritize import prioritize  # noqa: E402
from src.fusion.render_card import card_context, exposure, make_env, shown_subsystems  # noqa: E402
from src.fusion.risk import load_params  # noqa: E402
from src.utils.disclaimer import DISCLAIMER  # noqa: E402


# Assessment status of the alert-level E_sub -> text of the "path risk" column
PATH_RISK_TEXT = {"assessed": "Assessed", "partial": "Possibly underestimated", "not_assessed": "Not assessed: no upstream asset scanned"}
SOURCE_REAL, SOURCE_MOCK = "Real alerts (LSTM-AE)", "Mock alerts"


@st.cache_data
def load_data(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    fcfg = cfg["fusion"]
    sources = {}
    for label, key in ((SOURCE_REAL, "alerts_json"), (SOURCE_MOCK, "mock_alerts_json")):
        path = ROOT / fcfg[key]
        if path.exists():
            sources[label] = json.loads(path.read_text(encoding="utf-8"))
    return {
        "cfg": cfg,
        "sources": sources,
        "vulns": json.loads((ROOT / cfg["vuln"]["vulns_enriched_json"]).read_text(encoding="utf-8"))["vulns"],
        "params": load_params(ROOT / fcfg["params"]),
    }


def fmt_risk(x) -> str:
    return "Not assessed" if x is None else f"{x:.3f}"


def subsystem_esub_text(s: dict) -> str:
    """Display text for a subsystem's E_sub: "not assessed" when all upstream assets are unscanned; "no control chain" when there are no assets at all (P4), which does not count as "not assessed"."""
    if s["E_sub_assessed"]:
        return f"{s['E_sub']:.3f}"
    if s["unassessed_assets"]:
        return f"{s['E_sub']:.3f}" + " (not assessed: no upstream asset scanned)"
    return f"{s['E_sub']:.3f}" + " (no control chain: HIL simulation)"


def ranking_table(result: dict, baseline: dict, limit: int | None = None) -> pd.DataFrame:
    base_rank = {a["alert_id"]: a["rank"] for a in baseline["alerts"]}
    rows = []
    for a in result["alerts"][:limit or None]:
        delta = base_rank[a["alert_id"]] - a["rank"]
        rows.append({
            "Rank": a["rank"],
            "vs. default parameters": "—" if delta == 0 else f"{'↑' if delta > 0 else '↓'}{abs(delta)}",
            "Alert": a["alert_id"],
            "Start time": a["start"],
            "Priority": round(a["priority"], 3),
            "Score / threshold": round(a["max_score"] / a["threshold"], 2),
            "S": round(a["S"], 3),
            "Subsystem": ", ".join(f"{s['id']} {s['weight']:.0%}" for s in shown_subsystems(a)) or "-",
            "E_sub": round(a["E_sub"], 3),
            "Path risk": PATH_RISK_TEXT[exposure(a)],
            "KEV vulns needing review": len(a["needs_review"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    data = load_data(str(ROOT / "config" / "config.yaml"))
    cfg, defaults = data["cfg"], data["params"]
    st.set_page_config(page_title="Fusion alert demo", layout="wide")
    graph = AssetGraph.from_yaml(ROOT / cfg["fusion"]["asset_map"])

    st.title("Process anomaly + vulnerability context: fused alerts")
    if not data["sources"]:
        st.error("No alert file is available: run python -m scripts.export_alerts (real) or python -m scripts.make_mock_alerts (mock) first.")
        st.stop()
    with st.sidebar:
        st.header("Alert data")
        source = st.radio("Alert source", list(data["sources"]), index=0,
                          help="Real alerts are read by default (all alarm segments of LSTM-AE on the HAI test set); mock alerts are hand-made examples.")
    alerts = data["sources"][source]
    limit = cfg["fusion"].get("max_rendered_alerts") or None
    is_mock = any(a.get("is_mock", False) for a in alerts)
    if is_mock:
        st.warning("[MOCK] " + "Alerts come from mock data; scores and times are made-up values, not model output.")
    else:
        st.info("Real alerts: all {n} alarm segments of {model} on the HAI 21.03 test set. They are the raw output of the model, have not been confirmed by a human, and include false alarms."
                .format(model=alerts[0].get("model", "model"), n=len(alerts)))
    st.caption("Disclaimer: " + DISCLAIMER)

    with st.sidebar:
        st.header("Fusion scoring parameters")
        st.caption("Affects only the calculations on this page; fusion.yaml and all output files are left unchanged.")
        alpha = st.slider("α (exposure risk weight)", 0.0, 3.0, float(defaults["alpha"]), 0.1)
        beta = st.slider("β (EPSS weight)", 0.0, 3.0, float(defaults["beta"]), 0.1)
        gamma = st.slider("γ (KEV multiplier)", 1.0, 3.0, float(defaults["gamma"]), 0.1)
        min_qod = st.select_slider("min_qod", options=[0, 30, 50, 70, 80, 95, 99], value=int(defaults["min_qod"]))
    params = {**defaults, "alpha": alpha, "beta": beta, "gamma": gamma, "min_qod": min_qod}

    result = prioritize(alerts, data["vulns"], graph, params)
    baseline = result if params == defaults else prioritize(alerts, data["vulns"], graph, defaults)
    n_total = len(result["alerts"])
    n_shown = min(n_total, limit or n_total)

    tab_list, tab_card, tab_assets = st.tabs(["Alert list", "Alert card", "Assets and subsystems"])

    with tab_list:
        if n_shown < n_total:
            st.caption("{n} alerts in total; only the {m} with the highest priority are listed (fusion.max_rendered_alerts in the config).".format(n=n_total, m=n_shown))
        st.dataframe(ranking_table(result, baseline, limit), hide_index=True, use_container_width=True)
        st.caption("Order: by Priority (unrounded value), descending. Alerts with equal Priority are ordered by S, descending, and then by alert ID, ascending.")
        st.caption('"Possibly underestimated": some of the assets on the path have not been assessed (not scanned); they are counted as 0. "Not assessed": no upstream asset has been scanned; E_sub is recorded as 0.'
                   + ' "vs. default parameters" compares with the parameters in fusion.yaml.')

    with tab_card:
        ids = [a["alert_id"] for a in result["alerts"][:n_shown]]
        chosen = st.selectbox("Select an alert (sorted by priority)", ids)
        alert = next(a for a in result["alerts"] if a["alert_id"] == chosen)
        env = make_env(ROOT / cfg["fusion"]["templates_dir"])
        st.markdown(env.get_template("alert_card.md.j2").render(**card_context(alert, result)))

    with tab_assets:
        left, right = st.columns(2)
        with left:
            st.subheader("Asset risk R_asset")
            st.dataframe(pd.DataFrame([
                {"Asset": aid, "R_asset": fmt_risk(a["R_asset"]), "Status": a["status"],
                 "Vulnerabilities counted": a["n_vulns"], "KEV vulns needing review": a["n_needs_review"]}
                for aid, a in result["asset_risk"].items()]), hide_index=True, use_container_width=True)
        with right:
            st.subheader("Subsystem exposure risk E_sub")
            # Cells of st.table wrap automatically (the "unassessed upstream" column is long; st.dataframe would truncate it)
            st.table(pd.DataFrame([
                {"Subsystem": sid, "Name": graph.subsystems[sid].name,
                 "E_sub": subsystem_esub_text(s),
                 "Source asset": s["max_asset"] or "-", "Unassessed upstream": ", ".join(s["unassessed_assets"]) or "-"}
                for sid, s in result["subsystem_risk"].items()]).set_index("Subsystem"))
        topo = ROOT / cfg["report"]["extension_dir"] / "asset_topology.png"
        if topo.exists():
            st.image(str(topo), caption="Asset topology (generated by python -m scripts.make_extension_report)")


main()
