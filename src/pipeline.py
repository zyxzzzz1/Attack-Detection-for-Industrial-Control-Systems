"""One-command pipeline of the extension part: parse -> intel -> prioritize -> render.

1. parse:      vuln.report_xmls (OpenVAS XML, possibly several) -> vuln.vulns_json
2. intel:      vulns_json + KEV/EPSS -> vuln.vulns_enriched_json (with --offline only the data/intel/ cache is used)
3. prioritize: fusion.alerts_json + vulns_enriched_json + asset graph -> fusion.enriched_alerts_json
4. render:     enriched_alerts_json -> fusion.cards_dir (Markdown/HTML cards + index page)

The alert input defaults to the real alerts (all LSTM-AE alarm segments, exported by scripts.export_alerts); fusion scoring covers all alerts,
cards are generated only for the top fusion.max_rendered_alerts by priority. --mock uses the mock alerts instead, writing to
fusion.mock_enriched_alerts_json and fusion.mock_cards_dir without overwriting the real-alert output.

Stops at the first failing step and returns non-zero; all paths and parameters come from config/*.yaml.

Usage:
    python -m src.pipeline [--offline] [--mock] [--format md|html|both] [--config config/config.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.fusion.asset_graph import AssetGraph
from src.fusion.prioritize import prioritize
from src.fusion.render_card import make_env, render_all
from src.fusion.risk import load_params
from src.utils import i18n
from src.vuln.intel import IntelError, enrich
from src.vuln.parse_report import ReportIntegrityError, parse_reports

ROOT = Path(__file__).resolve().parents[1]


def _strip_time(data):
    return {k: v for k, v in data.items() if k != "generated_at"} if isinstance(data, dict) else data


def _write_json(path: Path, data) -> bool:
    """Write JSON; if an existing file has the same content apart from generated_at, leave it untouched (so committed snapshots are not changed pointlessly).
    Returns whether the file was written."""
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = None
        if old is not None and _strip_time(old) == _strip_time(data):
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


MAX_RANKING_PRINTED = 10  # Number of ranked alerts printed to the terminal in step 3 (there are over a thousand real alerts)


def run(cfg: dict, offline: bool = False, formats: tuple[str, ...] = ("md", "html"), session=None,
        mock: bool = False) -> dict:
    """Run the four steps in order and return a summary of each; raises on error."""
    vcfg, fcfg = cfg["vuln"], cfg["fusion"]
    summary = {}
    i18n.configure(cfg)   # Display language (ui.lang): affects only the text of the cards and the index page

    # 1. parse
    srcs = [ROOT / x for x in vcfg["report_xmls"]]
    parsed = parse_reports(srcs)
    _write_json(ROOT / vcfg["vulns_json"], parsed)
    summary["parse"] = {"reports": len(srcs), **parsed["stats"]}
    print(f"[1/4 parse] {len(srcs)} 份报告 → {parsed['stats']['vulns']} 条漏洞 "
          f"（丢弃 Log {parsed['stats']['log_dropped']}、重复 {parsed['stats']['duplicates_dropped']}）")

    # 2. intel
    enriched = enrich(parsed, cfg["intel"], session=session, offline=offline)
    _write_json(ROOT / vcfg["vulns_enriched_json"], enriched)
    info = enriched["intel"]
    n_kev = sum(v["in_kev"] for v in enriched["vulns"])
    summary["intel"] = {"kev_vulns": n_kev, "kev_version": info["kev_catalog_version"],
                        "epss_dates": info["epss_dates"]}
    print(f"[2/4 intel] {'离线' if offline else '在线'}；KEV {info['kev_catalog_version']}，"
          f"EPSS {info['epss_dates']}；命中 KEV {n_kev} 条")

    # 3. prioritize
    graph = AssetGraph.from_yaml(ROOT / fcfg["asset_map"])
    params = load_params(ROOT / fcfg["params"])
    prefix = "mock_" if mock else ""
    alerts_path = ROOT / fcfg[f"{prefix}alerts_json"]
    if not alerts_path.exists():
        hint = ("可运行 python -m scripts.make_mock_alerts" if mock
                else "先运行 python -m scripts.export_alerts 导出真实报警，或加 --mock 使用 mock 报警")
        raise FileNotFoundError(f"报警文件不存在：{alerts_path}（{hint}）")
    alerts = json.loads(alerts_path.read_text(encoding="utf-8"))
    result = prioritize(alerts, enriched["vulns"], graph, params)
    _write_json(ROOT / fcfg[f"{prefix}enriched_alerts_json"], result)
    summary["prioritize"] = {"alerts": len(result["alerts"]), "is_mock": result["is_mock"],
                             "ranking": [(a["alert_id"], a["priority"]) for a in result["alerts"]]}
    shown = result["alerts"][:MAX_RANKING_PRINTED]
    print(f"[3/4 prioritize] {len(result['alerts'])} 条报警{'（MOCK）' if result['is_mock'] else ''}："
          + " > ".join(f"{a['alert_id']} {a['priority']:.3f}" for a in shown)
          + (f" > …（其余 {len(result['alerts']) - len(shown)} 条略）" if len(result["alerts"]) > len(shown) else ""))

    # 4. render: generate cards only for the top N by priority
    limit = fcfg.get("max_rendered_alerts") or None
    cards_dir = ROOT / fcfg[f"{prefix}cards_dir"]
    written = render_all(result, make_env(ROOT / fcfg["templates_dir"]), cards_dir, list(formats), limit)
    n_cards = min(len(result["alerts"]), limit or len(result["alerts"]))
    summary["render"] = {"files": len(written), "cards": n_cards}
    print(f"[4/4 render] {len(written)} 个文件（前 {n_cards} / {len(result['alerts'])} 条报警的卡片 + 索引页）→ {cards_dir}")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="扩展部分一键流水线：parse → intel → prioritize → render")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--offline", action="store_true", help="intel 只用 data/intel/ 缓存，不联网")
    p.add_argument("--format", choices=["md", "html", "both"], default="both")
    p.add_argument("--mock", action="store_true", help="使用 mock 报警（默认使用真实报警）")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    formats = ("md", "html") if args.format == "both" else (args.format,)
    try:
        run(cfg, offline=args.offline, formats=formats, mock=args.mock)
    except ReportIntegrityError as e:
        print(f"[失败 parse] {e}", file=sys.stderr)
        return 4
    except IntelError as e:
        print(f"[失败 intel] {e}", file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError) as e:
        print(f"[失败] {e}", file=sys.stderr)
        return 2
    print(f"[完成] {i18n.disclaimer()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
