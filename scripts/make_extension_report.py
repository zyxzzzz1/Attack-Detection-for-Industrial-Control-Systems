"""E6: generate the extension report material into reports/extension/ (all based on real scan data and the asset map, no mock alerts).

Output:
- asset_topology.png   asset topology (Purdue levels, reachability edges, assessed or not, R_asset)
- qod_distribution.png QoD distribution (either side of the min_qod threshold, kept / dropped)
- kev_table.md         table of vulnerabilities in KEV (QoD, CVSS, EPSS, whether used in risk computation)
- key_numbers.md       key numbers (parsing, intel, QoD filtering, R_asset / E_sub, risk saturation), including the QoD distribution table

Usage:
    python -m scripts.make_extension_report [--output-dir DIR]
"""

import argparse
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

from src.fusion.asset_graph import AssetGraph  # noqa: E402
from src.fusion.risk import compute_risk, load_params, vuln_risk  # noqa: E402
from src.utils.i18n import DISCLAIMER_ZH, disclaimer, t, t_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WRAP_ASSET, WRAP_SUBSYSTEM = 30, 28   # Maximum characters per line inside a topology box
DISCLAIMER = DISCLAIMER_ZH  # For the md documents (Chinese); the disclaimer on figures uses disclaimer() per ui.lang

# Colors (dataviz reference palette, light): blue = kept / assessed; gray + hatching = dropped / not assessed
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BLUE_LIGHT, GRAY, HATCH = "#2a78d6", "#cde2fb", "#a3a29d", "#d6d5d0"
plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],  # The .ttc is registered in matplotlib as JP but contains Simplified Chinese glyphs
    "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "savefig.dpi": 200,
})


# ---------------------------------------------------------------- data

def load_inputs(cfg: dict) -> dict:
    enriched = json.loads((ROOT / cfg["vuln"]["vulns_enriched_json"]).read_text(encoding="utf-8"))
    parsed = json.loads((ROOT / cfg["vuln"]["vulns_json"]).read_text(encoding="utf-8"))
    graph = AssetGraph.from_yaml(ROOT / cfg["fusion"]["asset_map"])
    params = load_params(ROOT / cfg["fusion"]["params"])
    return {"parsed": parsed, "enriched": enriched, "graph": graph, "params": params,
            "risk": compute_risk(enriched["vulns"], graph, params)}


def qod_counts(vulns: list[dict]) -> list[tuple[int, int]]:
    return sorted(Counter(v["qod"] for v in vulns).items())


# ---------------------------------------------------------------- figures

def plot_qod(vulns: list[dict], min_qod: int, path: Path) -> None:
    counts = qod_counts(vulns)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = list(range(len(counts)))
    for x, (q, n) in zip(xs, counts):
        kept = q >= min_qod
        ax.bar(x, n, width=0.7, color=BLUE if kept else GRAY, hatch=None if kept else "//",
               edgecolor=SURFACE if kept else "#6f6e69", linewidth=0 if kept else 0.6)
        ax.text(x, n, f"{n}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(xs, [str(q) for q, _ in counts])
    # Threshold position: between the last value < min_qod and the first value >= min_qod
    cut = sum(q < min_qod for q, _ in counts) - 0.5
    ax.axvline(cut, color=INK2, linestyle="--", linewidth=1)
    ax.set_ylim(0, max(n for _, n in counts) * 1.12)
    ax.text(cut + 0.08, ax.get_ylim()[1] * 0.6, f"min_qod = {min_qod}", color=INK2, fontsize=9)
    kept_n = sum(n for q, n in counts if q >= min_qod)
    ax.set_title(t("漏洞 QoD 分布（共 {n} 条；QoD ≥ {q} 保留 {kept} 条）").format(n=len(vulns), q=min_qod, kept=kept_n),
                 loc="left", fontsize=12)
    ax.set_xlabel(t("QoD（检测质量）"))
    ax.set_ylabel(t("漏洞条数"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=BLUE, label=t("保留（QoD ≥ {q}）").format(q=min_qod)),
                       Patch(facecolor=GRAY, hatch="//", edgecolor="#6f6e69", label=t("不参与风险计算"))],
              frameon=False, loc="upper right")
    fig.text(0.01, 0.01, disclaimer(), fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def topology_positions(graph: AssetGraph) -> dict[str, tuple[float, float]]:
    """Lay out rows by Purdue level. L1 is ordered by name; higher levels are ordered by the mean x of their downstream nodes (barycenter) to reduce edge crossings.
    Subsystems are placed directly below their PLC; subsystems without a PLC (P4) go on the far right."""
    children: dict[str, list[str]] = {a: [] for a in graph.assets}
    for src, dst in graph.map.edges:
        children[src].append(dst)
    levels: dict[int, list[str]] = {}
    for aid, a in graph.assets.items():
        levels.setdefault(a.purdue_level, []).append(aid)
    width = max(len(v) for v in levels.values())
    pos: dict[str, tuple[float, float]] = {}
    for lvl in sorted(levels):
        ids = levels[lvl]

        def bary(aid):
            xs = [pos[c][0] for c in children[aid] if c in pos]
            return (sum(xs) / len(xs) if xs else float("inf"), aid)

        ids.sort(key=bary)
        off = (width - len(ids)) / 2
        for i, aid in enumerate(ids):
            pos[aid] = (off + i, float(lvl))
    right = max(p[0] for p in pos.values())
    for sid, sub in graph.subsystems.items():
        plcs = [a for a in sub.assets if graph.assets[a].purdue_level == 1]
        if plcs:
            pos[f"sub:{sid}"] = (pos[plcs[0]][0], 0.0)
        else:
            right += 1.15
            pos[f"sub:{sid}"] = (right, 0.0)
    return pos


def asset_head(asset) -> str:
    """The line under the asset name in a topology box: role · IP; assets without an IP show only the role."""
    return t_data(asset.role) + (f" · {asset.ip}" if asset.ip else "")


def plot_topology(graph: AssetGraph, risk: dict, path: Path) -> None:
    pos = topology_positions(graph)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    w, h = 0.78, 0.46

    def box(x, y, fill, edge, hatch=None, dashed=False):
        style = dict(boxstyle="round,pad=0.02,rounding_size=0.06")
        if hatch:
            # Hatching on its own (light) layer, border drawn separately, so the hatching does not cover the text
            ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, facecolor=fill, edgecolor=HATCH,
                                        hatch=hatch, linewidth=0, zorder=2, **style))
            fill = "none"
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, facecolor=fill, edgecolor=edge,
                                    linewidth=1.2, linestyle="--" if dashed else "-", zorder=2, **style))

    for src, dst in graph.map.edges:
        (x1, y1), (x2, y2) = pos[src], pos[dst]
        ax.annotate("", xy=(x2, y2 + h / 2), xytext=(x1, y1 - h / 2),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1, shrinkA=2, shrinkB=2), zorder=1)
    for sid, sub in graph.subsystems.items():
        x, y = pos[f"sub:{sid}"]
        for a in sub.assets:
            if graph.assets[a].purdue_level == 1:
                ax.plot([pos[a][0], x], [pos[a][1] - h / 2, y + h / 2], color=GRID, lw=1.5, zorder=1)

    for aid, a in graph.assets.items():
        x, y = pos[aid]
        r = risk["assets"][aid]
        if r["assessed"]:
            box(x, y, BLUE_LIGHT, BLUE)
            detail = f"R_asset = {r['R_asset']:.3f}"
        else:
            box(x, y, SURFACE, "#6f6e69", hatch="//", dashed=True)
            detail = t("未评估")
        head = asset_head(a)
        label_bg = dict(facecolor=SURFACE if not r["assessed"] else BLUE_LIGHT, edgecolor="none", pad=1.5)
        ax.text(x, y + 0.09, aid, ha="center", va="center", fontsize=10, weight="bold", zorder=3, bbox=label_bg)
        # English role names are long: wrap when wider than the box (Chinese is unaffected)
        ax.text(x, y - 0.08 - 0.03 * (len(head) > WRAP_ASSET),
                textwrap.fill(head, WRAP_ASSET) + f"\n{detail}", ha="center", va="center", fontsize=7.5,
                color=INK2, zorder=3, linespacing=1.3, bbox=label_bg)

    for sid, sub in graph.subsystems.items():
        x, y = pos[f"sub:{sid}"]
        e = risk["subsystems"][sid]
        if not e["upstream_assets"]:
            label = t("无控制链（HIL）")
        elif not e["E_sub_assessed"]:
            label = t("E_sub 未评估")
        else:
            label = f"E_sub = {e['E_sub']:.3f}"
        box(x, y, "#f0efec", GRID)
        ax.text(x, y + 0.08, textwrap.fill(f"{sid} {t_data(sub.name)}", WRAP_SUBSYSTEM), ha="center", va="center", fontsize=8.5, weight="bold", zorder=3)
        crit = t("关键度 {c}").format(c=sub.criticality)
        sep = "\n" if len(crit) + len(label) + 3 > WRAP_ASSET else " · "   # Split into two lines when one line does not fit
        ax.text(x, y - (0.12 if sep == "\n" else 0.1), f"{crit}{sep}{label}", ha="center", va="center", fontsize=7.5,
                color=INK2, zorder=3, linespacing=1.25)

    ys = sorted({p[1] for p in pos.values()})
    for y in ys:
        name = {3: t("L3 运营"), 2: t("L2 监控"), 1: t("L1 控制"), 0: t("L0 过程")}.get(int(y), f"L{int(y)}")
        ax.text(-0.75, y, name, ha="right", va="center", fontsize=9, color=INK2)
    xs = [p[0] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.3, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
    ax.axis("off")
    ax.set_title(t("资产拓扑与暴露风险（箭头：可下发指令 / 写数据）"), loc="left", fontsize=12)
    ax.legend(handles=[Patch(facecolor=BLUE_LIGHT, edgecolor=BLUE, label=t("已扫描评估")),
                       Patch(facecolor=SURFACE, edgecolor="#6f6e69", hatch="//", linestyle="--",
                             label=t("未扫描（未评估，计算中按 0）"))],
              frameon=False, loc="upper right", fontsize=8.5)
    fig.text(0.01, 0.01, disclaimer(), fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- tables and numbers

def kev_table(vulns: list[dict], params: dict, kev_version: str) -> str:
    rows = [v for v in vulns if v["in_kev"]]
    rows.sort(key=lambda v: (-(v["qod"] or 0), -v["cvss"], v["name"]))
    lines = ["# 命中 CISA KEV 的漏洞", "", f"> 免责声明：{DISCLAIMER}", "",
             f"KEV 版本 {kev_version}；共 {len(rows)} 条。QoD < {params['min_qod']} 的不参与风险计算，列入\"需人工核实\"。", "",
             "| KEV CVE | 漏洞 | 端口 | CVSS | EPSS | QoD | solution_type | r_v | 参与风险计算 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for v in rows:
        used = (v["qod"] or 0) >= params["min_qod"]
        lines.append(f"| {', '.join(v['kev_cves'])} | {v['name'].replace('|', '/')} | {v['port']} | {v['cvss']} | "
                     f"{v['epss']:.3f} | {v['qod']} | {v['solution_type']} | "
                     f"{vuln_risk(v, params['beta'], params['gamma']):.3f} | {'是' if used else '否（需人工核实）'} |")
    return "\n".join(lines) + "\n"


def key_numbers(d: dict) -> str:
    parsed, enriched, risk, params = d["parsed"], d["enriched"], d["risk"], d["params"]
    vs = enriched["vulns"]
    st = parsed["stats"]
    info = enriched["intel"]
    kept = [v for v in vs if (v["qod"] or 0) >= params["min_qod"]]
    saturated = sum(r["r_v"] >= 1.0 for r in risk["vulns"])
    lines = ["# 扩展部分关键数字", "", f"> 免责声明：{DISCLAIMER}", "",
             "## 漏洞采集（OpenVAS）", "",
             f"- 报告 {len(parsed['sources'])} 份，主机 {', '.join(h for s in parsed['sources'] for h in s['hosts'])}",
             f"- 结果 {st['results_total']} 条，丢弃 Log {st['log_dropped']} 条，去重 {st['duplicates_dropped']} 条，写入 {st['vulns']} 条",
             f"- Critical（CVSS ≥ 9.0）{sum(v['cvss'] >= 9.0 for v in vs)} 条", "",
             "## 威胁情报", "",
             f"- KEV 版本 {info['kev_catalog_version']}，EPSS 日期 {', '.join(info['epss_dates'])}",
             f"- 唯一 CVE {info['cves_total']} 个，EPSS 无记录 {len(info['cves_without_epss'])} 个，无 CVE 的漏洞 {sum(not v['cves'] for v in vs)} 条",
             f"- 命中 KEV {sum(v['in_kev'] for v in vs)} 条（{len(info['cves_in_kev'])} 个 CVE）",
             "- EPSS 分布（按漏洞取 CVE 最大值）："
             + "，".join(f"> {t}: {sum(v['epss'] > t for v in vs)}" for t in (0.0, 0.01, 0.1, 0.5, 0.9)), "",
             "## QoD 过滤", "",
             f"- min_qod = {params['min_qod']}：保留 {len(kept)} / {len(vs)} 条（丢弃 {1 - len(kept) / len(vs):.0%}），"
             f"其中命中 KEV {sum(v['in_kev'] for v in kept)} 条；需人工核实 {len(risk['needs_review'])} 条", "",
             "| QoD | 条数 | 是否保留 |", "|---|---|---|"]
    lines += [f"| {q} | {n} | {'是' if q >= params['min_qod'] else '否'} |" for q, n in qod_counts(vs)]
    lines += ["", "## 资产风险与子系统暴露风险", "",
              f"参数 α={params['alpha']}，β={params['beta']}，γ={params['gamma']}", "",
              "| 资产 | R_asset | 状态 | 参与计算漏洞数 | 需核实 |", "|---|---|---|---|---|"]
    for aid, a in risk["assets"].items():
        r = "未评估" if a["R_asset"] is None else f"{a['R_asset']:.3f}"
        lines.append(f"| {aid} | {r} | {a['status']} | {a['n_vulns']} | {a['n_needs_review']} |")
    lines += ["", "| 子系统 | E_sub | 来源资产 | 未评估的上游资产 |", "|---|---|---|---|"]
    for sid, s in risk["subsystems"].items():
        e = f"{s['E_sub']:.3f}" if s["E_sub_assessed"] else ("0（无关联资产）" if not s["upstream_assets"] else "未评估（按 0）")
        lines.append(f"| {sid} | {e} | {s['max_asset'] or '-'} | {'、'.join(s['unassessed_assets']) or '-'} |")
    lines += ["", "## 风险饱和", "",
              f"- 参与计算的 {len(risk['vulns'])} 条漏洞中，{saturated} 条 r_v 截断到 1.0；"
              "R_asset 与 E_sub 因此封顶，只有一台靶机被评估，E_sub 只能取 0 或 1。",
              "- 这是 min(1, ·) 公式与单靶机共同导致的，作为局限性写入报告（E3 第二台靶机已跳过）。", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="生成扩展部分报告素材")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = args.output_dir or ROOT / cfg["report"]["extension_dir"]
    out.mkdir(parents=True, exist_ok=True)
    d = load_inputs(cfg)
    plot_topology(d["graph"], d["risk"], out / "asset_topology.png")
    plot_qod(d["enriched"]["vulns"], d["params"]["min_qod"], out / "qod_distribution.png")
    (out / "kev_table.md").write_text(
        kev_table(d["enriched"]["vulns"], d["params"], d["enriched"]["intel"]["kev_catalog_version"]), encoding="utf-8")
    (out / "key_numbers.md").write_text(key_numbers(d), encoding="utf-8")
    print(f"[保存] {out}：asset_topology.png, qod_distribution.png, kev_table.md, key_numbers.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
