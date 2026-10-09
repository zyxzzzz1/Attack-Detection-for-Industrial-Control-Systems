"""E5：Streamlit 演示——告警列表 + 告警卡片 + 资产风险，侧边栏调参实时重算（不写任何文件）。

数据：默认读真实报警 fusion.alerts_json（LSTM-AE 的全部报警段），侧边栏可切换到 mock 报警 fusion.mock_alerts_json；
加上 vuln.vulns_enriched_json 与资产图。全部报警都参与融合打分，列表和卡片只显示优先级最高的前
fusion.max_rendered_alerts 条。卡片复用 templates/alert_card.md.j2，与离线生成的卡片内容一致。

用法：
    .venv/bin/streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # streamlit 以脚本方式运行，需要手动加入项目根目录
    sys.path.insert(0, str(ROOT))

from src.fusion.asset_graph import AssetGraph  # noqa: E402
from src.fusion.prioritize import prioritize  # noqa: E402
from src.fusion.render_card import card_context, exposure, make_env, shown_subsystems  # noqa: E402
from src.fusion.risk import load_params  # noqa: E402
from src.utils import i18n  # noqa: E402
from src.utils.i18n import disclaimer, t, t_data  # noqa: E402


# 告警级 E_sub 的评估状态 → "路径风险"一列的文字
PATH_RISK_TEXT = {"assessed": "已评估", "partial": "可能低估", "not_assessed": "未评估：上游资产均未扫描"}
SOURCE_REAL, SOURCE_MOCK = "真实报警（LSTM-AE）", "mock 报警"   # 内部键（中文源文），显示时经 t()


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
    return t("未评估") if x is None else f"{x:.3f}"


def subsystem_esub_text(s: dict) -> str:
    """子系统 E_sub 的显示文字：上游资产全部未扫描时写明未评估；没有任何资产（P4）时写明无控制链，不算"未评估"。"""
    if s["E_sub_assessed"]:
        return f"{s['E_sub']:.3f}"
    if s["unassessed_assets"]:
        return f"{s['E_sub']:.3f}" + t("（未评估：上游资产均未扫描）")
    return f"{s['E_sub']:.3f}" + t("（无控制链：HIL 仿真）")


def ranking_table(result: dict, baseline: dict, limit: int | None = None) -> pd.DataFrame:
    base_rank = {a["alert_id"]: a["rank"] for a in baseline["alerts"]}
    rows = []
    for a in result["alerts"][:limit or None]:
        delta = base_rank[a["alert_id"]] - a["rank"]
        rows.append({
            t("排名"): a["rank"],
            t("相对默认参数"): "—" if delta == 0 else f"{'↑' if delta > 0 else '↓'}{abs(delta)}",
            t("告警"): a["alert_id"],
            t("起始时间"): a["start"],
            "Priority": round(a["priority"], 3),
            t("分数 / 阈值"): round(a["max_score"] / a["threshold"], 2),
            "S": round(a["S"], 3),
            t("子系统"): t("、").join(f"{s['id']} {s['weight']:.0%}" for s in shown_subsystems(a)) or "-",
            "E_sub": round(a["E_sub"], 3),
            t("路径风险"): t(PATH_RISK_TEXT[exposure(a)]),
            t("需核实的 KEV 漏洞"): len(a["needs_review"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    data = load_data(str(ROOT / "config" / "config.yaml"))
    cfg, defaults = data["cfg"], data["params"]
    i18n.configure(cfg)   # 显示语言（ui.lang）
    st.set_page_config(page_title=t("融合告警演示"), layout="wide")
    graph = AssetGraph.from_yaml(ROOT / cfg["fusion"]["asset_map"])

    st.title(t("过程异常 + 漏洞上下文 融合告警"))
    if not data["sources"]:
        st.error(t("没有可用的报警文件：先运行 python -m scripts.export_alerts（真实）或 python -m scripts.make_mock_alerts（mock）。"))
        st.stop()
    with st.sidebar:
        st.header(t("报警数据"))
        source = st.radio(t("报警来源"), list(data["sources"]), index=0, format_func=t,
                          help=t("默认读取真实报警（LSTM-AE 在 HAI 测试集上的全部报警段）；mock 报警是手工构造的示例。"))
    alerts = data["sources"][source]
    limit = cfg["fusion"].get("max_rendered_alerts") or None
    is_mock = any(a.get("is_mock", False) for a in alerts)
    if is_mock:
        st.warning("[MOCK] " + t("报警来自 mock 数据，分数与时间为虚构值，不是模型输出。"))
    else:
        st.info(t("真实报警：{model} 在 HAI 21.03 测试集上的全部 {n} 条报警段，是模型的原始输出，未经人工确认，其中包含误报。")
                .format(model=alerts[0].get("model", "model"), n=len(alerts)))
    st.caption(t("免责声明：") + disclaimer())

    with st.sidebar:
        st.header(t("融合打分参数"))
        st.caption(t("只影响本页面的计算，不改写 fusion.yaml 和任何输出文件。"))
        alpha = st.slider(t("α（暴露风险权重）"), 0.0, 3.0, float(defaults["alpha"]), 0.1)
        beta = st.slider(t("β（EPSS 权重）"), 0.0, 3.0, float(defaults["beta"]), 0.1)
        gamma = st.slider(t("γ（KEV 倍数）"), 1.0, 3.0, float(defaults["gamma"]), 0.1)
        min_qod = st.select_slider("min_qod", options=[0, 30, 50, 70, 80, 95, 99], value=int(defaults["min_qod"]))
    params = {**defaults, "alpha": alpha, "beta": beta, "gamma": gamma, "min_qod": min_qod}

    result = prioritize(alerts, data["vulns"], graph, params)
    baseline = result if params == defaults else prioritize(alerts, data["vulns"], graph, defaults)
    n_total = len(result["alerts"])
    n_shown = min(n_total, limit or n_total)

    tab_list, tab_card, tab_assets = st.tabs([t("告警列表"), t("告警卡片"), t("资产与子系统")])

    with tab_list:
        if n_shown < n_total:
            st.caption(t("共 {n} 条报警，只列出优先级最高的前 {m} 条（config 的 fusion.max_rendered_alerts）。").format(n=n_total, m=n_shown))
        st.dataframe(ranking_table(result, baseline, limit), hide_index=True, use_container_width=True)
        st.caption(t("排序：按 Priority（未四舍五入的值）降序；Priority 相同时按 S 降序，仍相同时按告警编号升序。"))
        st.caption(t("\"可能低估\"：路径上有一部分资产未评估（未扫描），按 0 计算。\"未评估\"：上游资产均未扫描，E_sub 记为 0。")
                   + t("\"相对默认参数\"对比 fusion.yaml 中的参数。"))

    with tab_card:
        ids = [a["alert_id"] for a in result["alerts"][:n_shown]]
        chosen = st.selectbox(t("选择告警（按优先级排序）"), ids)
        alert = next(a for a in result["alerts"] if a["alert_id"] == chosen)
        env = make_env(ROOT / cfg["fusion"]["templates_dir"])
        st.markdown(env.get_template("alert_card.md.j2").render(**card_context(alert, result)))

    with tab_assets:
        left, right = st.columns(2)
        with left:
            st.subheader(t("资产风险 R_asset"))
            st.dataframe(pd.DataFrame([
                {t("资产"): aid, "R_asset": fmt_risk(a["R_asset"]), t("状态"): a["status"],
                 t("参与计算漏洞数"): a["n_vulns"], t("需核实的 KEV 漏洞"): a["n_needs_review"]}
                for aid, a in result["asset_risk"].items()]), hide_index=True, use_container_width=True)
        with right:
            st.subheader(t("子系统暴露风险 E_sub"))
            # st.table 的单元格自动换行（"未评估上游"一列较长，st.dataframe 会截断）
            st.table(pd.DataFrame([
                {t("子系统"): sid, t("名称"): t_data(graph.subsystems[sid].name),
                 "E_sub": subsystem_esub_text(s),
                 t("来源资产"): s["max_asset"] or "-", t("未评估上游"): t("、").join(s["unassessed_assets"]) or "-"}
                for sid, s in result["subsystem_risk"].items()]).set_index(t("子系统")))
        topo = ROOT / cfg["report"]["extension_dir"] / "asset_topology.png"
        if topo.exists():
            st.image(str(topo), caption=t("资产拓扑（python -m scripts.make_extension_report 生成）"))


main()
