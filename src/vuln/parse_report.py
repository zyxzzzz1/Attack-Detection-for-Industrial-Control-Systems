"""OpenVAS XML report -> normalized vulnerability records vulns.json (claude.md section 7, M7).

Rules:
- Before parsing, a three-count check: actual number of results == result_count/filtered == result_count/full;
  a mismatch means results were filtered or truncated by paging during export, so exit with an error.
- No QoD filtering: everything is kept with its qod field (filtering happens during fusion scoring, using min_qod from fusion.yaml).
- Log results (severity <= 0) are not written.
- Deduplicate by (asset_ip, port, nvt_oid): keep the highest severity, then the highest QoD.
- Several reports are supported (one per target): each is checked, then merged, deduplicating across reports as well.

Usage:
    python -m src.vuln.parse_report [--input XML [XML ...]] [--output JSON]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
DISCLAIMER = "资产映射与漏洞数据来自实验环境，仅用于演示"


class ReportIntegrityError(Exception):
    """Report count check failed (incomplete export)."""


def load_report(path: Path) -> etree._Element:
    """Read a report, run the three-count check, and return the inner <report> node."""
    root = etree.parse(str(path)).getroot()
    report = root.find(".//report/report")
    if report is None:
        raise ReportIntegrityError(f"{path} 中找不到内层 <report> 节点")
    n = len(report.findall("results/result"))
    filtered = (report.findtext("result_count/filtered") or "").strip()
    full = (report.findtext("result_count/full") or "").strip()
    if not (str(n) == filtered == full):
        raise ReportIntegrityError(
            f"条数不一致：导出 {n} 条，filtered={filtered or '缺失'}，full={full or '缺失'}；报告不完整，请重新导出"
        )
    return report


def _text(el: etree._Element, path: str) -> str:
    return (el.findtext(path) or "").strip()


def parse_result(result: etree._Element) -> dict:
    """Convert a single <result> node into a normalized record (Log results included; the caller decides whether to drop them)."""
    nvt = result.find("nvt")
    solution = nvt.find("solution")
    severity_type = nvt.find("severities/severity")
    cvss_version = None
    if severity_type is not None and severity_type.get("type"):
        # cvss_base_v2 / cvss_base_v3 → v2 / v3
        cvss_version = severity_type.get("type").removeprefix("cvss_base_")
    qod = _text(result, "qod/value")
    return {
        "asset_ip": _text(result, "host"),
        "port": _text(result, "port"),
        "nvt_oid": nvt.get("oid"),
        "name": _text(nvt, "name"),
        "cvss": float(_text(result, "severity") or 0),
        "cves": [ref.get("id") for ref in nvt.iterfind("refs/ref[@type='cve']")],
        "solution": (solution.text or "").strip() if solution is not None else "",
        "qod": int(qod) if qod else None,
        "threat": _text(result, "threat"),
        "solution_type": (solution.get("type") or None) if solution is not None else None,
        "cvss_version": cvss_version,
    }


def _rel(path: Path) -> str:
    """Write paths inside the project as relative paths, so snapshots compare across machines."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_reports(paths: list[Path]) -> dict:
    """Parse and merge several reports (each with its own three-count check), deduplicating across reports by (asset_ip, port, nvt_oid) as well;
    vulns are sorted by cvss, descending."""
    sources, records = [], []
    for path in paths:
        report = load_report(path)
        recs = [parse_result(r) for r in report.iterfind("results/result")]
        records += recs
        hosts = sorted({r["asset_ip"] for r in recs})
        sources.append({"report_id": report.get("id"), "file": _rel(path), "hosts": hosts,
                        "results_total": len(recs)})
    kept = [r for r in records if r["cvss"] > 0]

    best: dict[tuple, dict] = {}
    for rec in kept:
        key = (rec["asset_ip"], rec["port"], rec["nvt_oid"])
        cur = best.get(key)
        if cur is None or (rec["cvss"], rec["qod"] or 0) > (cur["cvss"], cur["qod"] or 0):
            best[key] = rec
    vulns = sorted(best.values(), key=lambda r: (-r["cvss"], r["asset_ip"], r["port"], r["nvt_oid"]))

    return {
        "disclaimer": DISCLAIMER,
        "sources": sources,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": {
            "results_total": len(records),
            "log_dropped": len(records) - len(kept),
            "duplicates_dropped": len(kept) - len(vulns),
            "vulns": len(vulns),
            "vulns_by_host": {h: sum(v["asset_ip"] == h for v in vulns)
                              for h in sorted({v["asset_ip"] for v in vulns})},
        },
        "vulns": vulns,
    }


def parse_report(path: Path) -> dict:
    """Convenience entry point for a single report."""
    return parse_reports([path])


def main() -> int:
    p = argparse.ArgumentParser(description="OpenVAS XML 报告 → vulns.json")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--input", type=Path, nargs="+", help="一份或多份报告，默认取 config.yaml 的 vuln.report_xmls")
    p.add_argument("--output", type=Path, help="默认取 config.yaml 的 vuln.vulns_json")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["vuln"]
    srcs = args.input or [ROOT / x for x in cfg["report_xmls"]]
    dst = args.output or ROOT / cfg["vulns_json"]

    try:
        data = parse_reports(srcs)
    except ReportIntegrityError as e:
        print(f"[失败] {e}", file=sys.stderr)
        return 4

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    s = data["stats"]
    print(
        f"[保存] {dst}：共 {s['results_total']} 条结果，丢弃 Log {s['log_dropped']} 条、"
        f"重复 {s['duplicates_dropped']} 条，写入 {s['vulns']} 条"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
