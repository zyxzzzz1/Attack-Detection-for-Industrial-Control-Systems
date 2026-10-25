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
from src.utils.disclaimer import DISCLAIMER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WRAP_ASSET, WRAP_SUBSYSTEM = 30, 28   # Maximum characters per line inside a topology box

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
    ax.set_title("QoD distribution of vulnerabilities ({n} in total; {kept} kept with QoD ≥ {q})".format(n=len(vulns), q=min_qod, kept=kept_n),
                 loc="left", fontsize=12)
    ax.set_xlabel("QoD (quality of detection)")
    ax.set_ylabel("Number of vulnerabilities")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=BLUE, label="Kept (QoD ≥ {q})".format(q=min_qod)),
                       Patch(facecolor=GRAY, hatch="//", edgecolor="#6f6e69", label="Excluded from the risk calculation")],
              frameon=False, loc="upper right")
    fig.text(0.01, 0.01, DISCLAIMER, fontsize=8, color=INK2)
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
    return asset.role + (f" · {asset.ip}" if asset.ip else "")


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
            detail = "Not assessed"
        head = asset_head(a)
        label_bg = dict(facecolor=SURFACE if not r["assessed"] else BLUE_LIGHT, edgecolor="none", pad=1.5)
        ax.text(x, y + 0.09, aid, ha="center", va="center", fontsize=10, weight="bold", zorder=3, bbox=label_bg)
        # Long role names: wrap when wider than the box
        ax.text(x, y - 0.08 - 0.03 * (len(head) > WRAP_ASSET),
                textwrap.fill(head, WRAP_ASSET) + f"\n{detail}", ha="center", va="center", fontsize=7.5,
                color=INK2, zorder=3, linespacing=1.3, bbox=label_bg)

    for sid, sub in graph.subsystems.items():
        x, y = pos[f"sub:{sid}"]
        e = risk["subsystems"][sid]
        if not e["upstream_assets"]:
            label = "no control chain (HIL)"
        elif not e["E_sub_assessed"]:
            label = "E_sub not assessed"
        else:
            label = f"E_sub = {e['E_sub']:.3f}"
        box(x, y, "#f0efec", GRID)
        ax.text(x, y + 0.08, textwrap.fill(f"{sid} {sub.name}", WRAP_SUBSYSTEM), ha="center", va="center", fontsize=8.5, weight="bold", zorder=3)
        crit = "criticality {c}".format(c=sub.criticality)
        sep = "\n" if len(crit) + len(label) + 3 > WRAP_ASSET else " · "   # Split into two lines when one line does not fit
        ax.text(x, y - (0.12 if sep == "\n" else 0.1), f"{crit}{sep}{label}", ha="center", va="center", fontsize=7.5,
                color=INK2, zorder=3, linespacing=1.25)

    ys = sorted({p[1] for p in pos.values()})
    for y in ys:
        name = {3: "L3 Operations", 2: "L2 Supervisory", 1: "L1 Control", 0: "L0 Process"}.get(int(y), f"L{int(y)}")
        ax.text(-0.75, y, name, ha="right", va="center", fontsize=9, color=INK2)
    xs = [p[0] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.3, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
    ax.axis("off")
    ax.set_title("Asset topology and exposure risk (arrows: can send commands / write data)", loc="left", fontsize=12)
    ax.legend(handles=[Patch(facecolor=BLUE_LIGHT, edgecolor=BLUE, label="Scanned and assessed"),
                       Patch(facecolor=SURFACE, edgecolor="#6f6e69", hatch="//", linestyle="--",
                             label="Not scanned (not assessed; counted as 0)")],
              frameon=False, loc="upper right", fontsize=8.5)
    fig.text(0.01, 0.01, DISCLAIMER, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- tables and numbers

def kev_table(vulns: list[dict], params: dict, kev_version: str) -> str:
    rows = [v for v in vulns if v["in_kev"]]
    rows.sort(key=lambda v: (-(v["qod"] or 0), -v["cvss"], v["name"]))
    lines = ["# Vulnerabilities in CISA KEV", "", f"> Disclaimer: {DISCLAIMER}", "",
             f"KEV version {kev_version}; {len(rows)} entries. Those with QoD < {params['min_qod']} are excluded from the risk computation and listed under \"needs manual verification\".", "",
             "| KEV CVE | Vulnerability | Port | CVSS | EPSS | QoD | solution_type | r_v | Used in risk computation |",
             "|---|---|---|---|---|---|---|---|---|"]
    for v in rows:
        used = (v["qod"] or 0) >= params["min_qod"]
        lines.append(f"| {', '.join(v['kev_cves'])} | {v['name'].replace('|', '/')} | {v['port']} | {v['cvss']} | "
                     f"{v['epss']:.3f} | {v['qod']} | {v['solution_type']} | "
                     f"{vuln_risk(v, params['beta'], params['gamma']):.3f} | {'yes' if used else 'no (needs manual verification)'} |")
    return "\n".join(lines) + "\n"


def key_numbers(d: dict) -> str:
    parsed, enriched, risk, params = d["parsed"], d["enriched"], d["risk"], d["params"]
    vs = enriched["vulns"]
    st = parsed["stats"]
    info = enriched["intel"]
    kept = [v for v in vs if (v["qod"] or 0) >= params["min_qod"]]
    saturated = sum(r["r_v"] >= 1.0 for r in risk["vulns"])
    lines = ["# Extension key numbers", "", f"> Disclaimer: {DISCLAIMER}", "",
             "## Vulnerability collection (OpenVAS)", "",
             f"- {len(parsed['sources'])} reports, hosts {', '.join(h for s in parsed['sources'] for h in s['hosts'])}",
             f"- {st['results_total']} results, {st['log_dropped']} Log dropped, {st['duplicates_dropped']} duplicates dropped, {st['vulns']} written",
             f"- Critical (CVSS ≥ 9.0): {sum(v['cvss'] >= 9.0 for v in vs)}", "",
             "## Threat intelligence", "",
             f"- KEV version {info['kev_catalog_version']}, EPSS date {', '.join(info['epss_dates'])}",
             f"- {info['cves_total']} unique CVEs, {len(info['cves_without_epss'])} without an EPSS record, {sum(not v['cves'] for v in vs)} vulnerabilities without CVEs",
             f"- {sum(v['in_kev'] for v in vs)} in KEV ({len(info['cves_in_kev'])} CVEs)",
             "- EPSS distribution (per vulnerability, max over its CVEs): "
             + ", ".join(f"> {t}: {sum(v['epss'] > t for v in vs)}" for t in (0.0, 0.01, 0.1, 0.5, 0.9)), "",
             "## QoD filtering", "",
             f"- min_qod = {params['min_qod']}: kept {len(kept)} / {len(vs)} ({1 - len(kept) / len(vs):.0%} dropped), "
             f"of which {sum(v['in_kev'] for v in kept)} in KEV; {len(risk['needs_review'])} need manual verification", "",
             "| QoD | Count | Kept |", "|---|---|---|"]
    lines += [f"| {q} | {n} | {'yes' if q >= params['min_qod'] else 'no'} |" for q, n in qod_counts(vs)]
    lines += ["", "## Asset risk and subsystem exposure risk", "",
              f"Parameters α={params['alpha']}, β={params['beta']}, γ={params['gamma']}", "",
              "| Asset | R_asset | Status | Vulns used | Needs review |", "|---|---|---|---|---|"]
    for aid, a in risk["assets"].items():
        r = "Not assessed" if a["R_asset"] is None else f"{a['R_asset']:.3f}"
        lines.append(f"| {aid} | {r} | {a['status']} | {a['n_vulns']} | {a['n_needs_review']} |")
    lines += ["", "| Subsystem | E_sub | Source asset | Unassessed upstream assets |", "|---|---|---|---|"]
    for sid, s in risk["subsystems"].items():
        e = f"{s['E_sub']:.3f}" if s["E_sub_assessed"] else ("0 (no associated assets)" if not s["upstream_assets"] else "not assessed (counted as 0)")
        lines.append(f"| {sid} | {e} | {s['max_asset'] or '-'} | {', '.join(s['unassessed_assets']) or '-'} |")
    lines += ["", "## Risk saturation", "",
              f"- Of the {len(risk['vulns'])} vulnerabilities used, {saturated} have r_v clipped to 1.0; "
              "R_asset and E_sub are therefore capped, and with only one target assessed, E_sub can only be 0 or 1.",
              "- This is caused jointly by the min(1, ·) formula and the single target, and is written into the report as a limitation (the second target of E3 was skipped).", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate the extension report material")
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
    print(f"[SAVED] {out}: asset_topology.png, qod_distribution.png, kev_table.md, key_numbers.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
