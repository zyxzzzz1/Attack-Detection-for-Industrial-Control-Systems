"""Threat-intel enrichment: CISA KEV + FIRST EPSS (claude.md section 7, M8).

Input vulns.json (output of parse_report, left unchanged), output vulns_enriched.json. Each vulnerability gains:
- in_kev:   true if any CVE is in KEV; false if there are no CVEs
- kev_cves: the specific CVEs that are in KEV (used by the alert cards)
- epss:     maximum EPSS over all CVEs; 0.0 if there are no CVEs or no EPSS record

Data sources (checked against the official documentation on 2026-09-30):
- KEV:  official JSON feed, top level {catalogVersion, dateReleased, count, vulnerabilities[{cveID, ...}]}
- EPSS: GET https://api.first.org/data/v1/epss?cve=A,B,...
        the cve parameter is at most 2000 characters (including commas); limit defaults to 100; returns data[{cve, epss, percentile, date}],
        where epss/percentile are strings; the public API is rate-limited to 1000 requests/minute.

Cache:
- data/intel/kev.json: the official JSON saved as-is, re-downloaded only when the file is older than kev_max_age_hours.
- data/intel/epss.csv: columns cve,epss,percentile,date,fetched_at; cached CVEs are not queried again.
  CVEs without an EPSS record (e.g. rejected IDs) are cached too, with epss empty, to avoid querying them every time;
  empty records are re-queried once after epss_missing_max_age_days days (offline mode still uses them directly).

Usage:
    python -m src.vuln.intel [--input JSON] [--output JSON] [--offline]
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

from src.utils.disclaimer import DISCLAIMER

ROOT = Path(__file__).resolve().parents[2]
EPSS_FIELDS = ["cve", "epss", "percentile", "date", "fetched_at"]


class IntelError(Exception):
    """Failed to obtain threat intel (network, format, or missing offline cache)."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def http_get_json(session, url: str, params: dict | None, timeout: float, retries: int,
                  backoff_s: float) -> dict:
    """GET with timeout and retries; retries on connection errors, timeouts, 429 and 5xx; other 4xx fail immediately."""
    last = None
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                last = IntelError(f"{url} returned HTTP {resp.status_code}")
            elif resp.status_code != 200:
                raise IntelError(f"{url} returned HTTP {resp.status_code}")
            else:
                return resp.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e
        if attempt < retries:
            time.sleep(backoff_s * 2 ** attempt)
    raise IntelError(f"request to {url} failed (retried {retries} times): {last}")


# ---------------------------------------------------------------- KEV

def load_kev(cfg: dict, session, offline: bool) -> dict:
    """Return the official KEV JSON; read the cache when it has not expired or when offline."""
    cache = ROOT / cfg["kev_cache"]
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < cfg["kev_max_age_hours"] * 3600
    if offline or fresh:
        if not cache.exists():
            raise IntelError(f"offline mode but the KEV cache does not exist: {cache}")
        return json.loads(cache.read_text(encoding="utf-8"))

    data = http_get_json(session, cfg["kev_url"], None, cfg["timeout_s"], cfg["retries"], cfg["backoff_s"])
    missing = {"catalogVersion", "dateReleased", "vulnerabilities"} - set(data)
    if missing:
        raise IntelError(f"unexpected KEV format, missing fields {sorted(missing)}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def kev_cve_set(kev: dict) -> set[str]:
    return {v["cveID"].strip().upper() for v in kev["vulnerabilities"]}


# ---------------------------------------------------------------- EPSS

def read_epss_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {row["cve"]: row for row in csv.DictReader(f)}


def write_epss_cache(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EPSS_FIELDS)
        w.writeheader()
        for cve in sorted(rows):
            w.writerow({k: rows[cve].get(k, "") for k in EPSS_FIELDS})


def make_batches(cves: list[str], max_chars: int, max_count: int) -> list[list[str]]:
    """Split a CVE list into batches: comma-joined length <= max_chars and count <= max_count."""
    batches, cur, cur_len = [], [], 0
    for cve in cves:
        add = len(cve) + (1 if cur else 0)
        if cur and (cur_len + add > max_chars or len(cur) >= max_count):
            batches.append(cur)
            cur, cur_len, add = [], 0, len(cve)
        cur.append(cve)
        cur_len += add
    if cur:
        batches.append(cur)
    return batches


def fetch_epss(cves: list[str], cfg: dict, session) -> dict[str, dict]:
    """Query EPSS in batches, returning {cve: row}; CVEs without an API record also get a row (with empty epss)."""
    fetched_at = _now().isoformat(timespec="seconds")
    out = {}
    batches = make_batches(cves, cfg["epss_max_param_chars"], cfg["epss_batch_size"])
    for i, batch in enumerate(batches, 1):
        params = {"cve": ",".join(batch), "limit": len(batch)}
        data = http_get_json(session, cfg["epss_url"], params, cfg["timeout_s"], cfg["retries"], cfg["backoff_s"])
        if data.get("status") != "OK" or "data" not in data:
            raise IntelError(f"unexpected EPSS format: {str(data)[:200]}")
        if int(data.get("total", 0)) > len(data["data"]):
            raise IntelError(f"EPSS batch {i} truncated by paging (total={data['total']})")
        for item in data["data"]:
            cve = item["cve"].strip().upper()
            out[cve] = {"cve": cve, "epss": item["epss"], "percentile": item.get("percentile", ""),
                        "date": item.get("date", ""), "fetched_at": fetched_at}
        for cve in batch:
            out.setdefault(cve, {"cve": cve, "epss": "", "percentile": "", "date": "", "fetched_at": fetched_at})
        print(f"[EPSS] batch {i}/{len(batches)}: requested {len(batch)}, returned {len(data['data'])}")
    return out


def load_epss(cves: set[str], cfg: dict, session, offline: bool) -> dict[str, dict]:
    """Return the EPSS cache rows covering cves; only CVEs missing from the cache are queried."""
    cache_path = ROOT / cfg["epss_cache"]
    rows = read_epss_cache(cache_path)
    # Empty records (EPSS has no such CVE) older than epss_missing_max_age_days are treated as expired and re-queried
    cutoff = _now().timestamp() - cfg["epss_missing_max_age_days"] * 86400
    stale = set() if offline else {
        c for c, r in rows.items()
        if r["epss"] == "" and datetime.fromisoformat(r["fetched_at"]).timestamp() < cutoff}
    todo = sorted(cves - (set(rows) - stale))
    if todo:
        if offline:
            raise IntelError(f"offline mode but the EPSS cache is missing {len(todo)} CVEs (e.g. {todo[:3]})")
        rows.update(fetch_epss(todo, cfg, session))
        write_epss_cache(cache_path, rows)
    else:
        print(f"[EPSS] all {len(cves)} CVEs found in the cache")
    return rows


# ---------------------------------------------------------------- enrichment

def enrich_vuln(v: dict, kev: set[str], epss: dict[str, dict]) -> dict:
    cves = [c.strip().upper() for c in v["cves"]]
    kev_cves = sorted({c for c in cves if c in kev})
    scores = [float(epss[c]["epss"]) for c in cves if c in epss and epss[c]["epss"] != ""]
    return {**v, "in_kev": bool(kev_cves), "kev_cves": kev_cves, "epss": max(scores, default=0.0)}


def enrich(data: dict, cfg: dict, session=None, offline: bool = False) -> dict:
    session = session or requests.Session()
    all_cves = {c.strip().upper() for v in data["vulns"] for c in v["cves"]}
    kev = load_kev(cfg, session, offline)
    kev_set = kev_cve_set(kev)
    epss = load_epss(all_cves, cfg, session, offline)

    vulns = [enrich_vuln(v, kev_set, epss) for v in data["vulns"]]
    used = [epss[c] for c in sorted(all_cves)]
    return {
        **data,
        "disclaimer": DISCLAIMER,
        "generated_at": _now().isoformat(timespec="seconds"),
        "intel": {
            "kev_catalog_version": kev.get("catalogVersion"),
            "kev_date_released": kev.get("dateReleased"),
            "epss_dates": sorted({r["date"] for r in used if r["date"]}),
            "cves_total": len(all_cves),
            "cves_in_kev": sorted(all_cves & kev_set),
            "cves_without_epss": sorted(r["cve"] for r in used if r["epss"] == ""),
        },
        "vulns": vulns,
    }


def summarize(out: dict) -> str:
    vs = out["vulns"]
    info = out["intel"]
    lines = [
        f"{len(vs)} vulnerabilities with {info['cves_total']} CVEs; "
        f"KEV version {info['kev_catalog_version']}, EPSS date {info['epss_dates']}",
        f"{sum(v['in_kev'] for v in vs)} vulnerabilities in KEV, {len(info['cves_in_kev'])} KEV CVEs: "
        f"{info['cves_in_kev']}",
        f"{len(info['cves_without_epss'])} CVEs without an EPSS record",
    ]
    for t in (0.0, 0.01, 0.1, 0.5, 0.9):
        lines.append(f"  epss > {t}: {sum(v['epss'] > t for v in vs)}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="vulns.json + KEV/EPSS → vulns_enriched.json")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--input", type=Path, help="defaults to vuln.vulns_json in config.yaml")
    p.add_argument("--output", type=Path, help="defaults to vuln.vulns_enriched_json in config.yaml")
    p.add_argument("--offline", action="store_true", help="use only the data/intel/ cache, no network")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    src = args.input or ROOT / cfg["vuln"]["vulns_json"]
    dst = args.output or ROOT / cfg["vuln"]["vulns_enriched_json"]
    if src.resolve() == dst.resolve():
        print("[FAILED] the output path equals the input; overwriting the raw parse result is not allowed", file=sys.stderr)
        return 2

    data = json.loads(src.read_text(encoding="utf-8"))
    try:
        out = enrich(data, cfg["intel"], offline=args.offline)
    except IntelError as e:
        print(f"[FAILED] {e}", file=sys.stderr)
        return 3

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {dst}")
    print(summarize(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
