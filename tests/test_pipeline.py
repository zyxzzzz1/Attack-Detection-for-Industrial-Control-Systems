"""src/pipeline.py 的端到端测试：真实 XML 报告 + mock 报警，输出与情报缓存全部指向临时目录，不联网。"""

import copy
import csv
import json
from pathlib import Path

import pytest
import yaml

from src.pipeline import main, run
from src.vuln.intel import IntelError
from src.vuln.parse_report import parse_reports
from tests.test_intel import EPSS_URL, KEV_URL, FakeSession

ROOT = Path(__file__).resolve().parents[1]
KEV_HITS = ["CVE-2020-1938", "CVE-2012-1823"]


@pytest.fixture(scope="module")
def real_cves() -> list[str]:
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    data = parse_reports([ROOT / x for x in cfg["vuln"]["report_xmls"]])
    return sorted({c for v in data["vulns"] for c in v["cves"]})


@pytest.fixture
def cfg(tmp_path) -> dict:
    """真实配置，但所有输出与缓存改到 tmp_path（ROOT / 绝对路径 == 绝对路径）。"""
    c = copy.deepcopy(yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8")))
    c["vuln"]["vulns_json"] = str(tmp_path / "vulns.json")
    c["vuln"]["vulns_enriched_json"] = str(tmp_path / "vulns_enriched.json")
    c["intel"].update(kev_url=KEV_URL, epss_url=EPSS_URL, kev_cache=str(tmp_path / "intel" / "kev.json"),
                      epss_cache=str(tmp_path / "intel" / "epss.csv"), backoff_s=0)
    c["fusion"]["enriched_alerts_json"] = str(tmp_path / "real" / "enriched_alerts.json")
    c["fusion"]["cards_dir"] = str(tmp_path / "real" / "cards")
    c["fusion"]["mock_enriched_alerts_json"] = str(tmp_path / "enriched_alerts.json")
    c["fusion"]["mock_cards_dir"] = str(tmp_path / "cards")
    return c


def write_caches(tmp_path: Path, cves: list[str]) -> None:
    intel = tmp_path / "intel"
    intel.mkdir(parents=True, exist_ok=True)
    (intel / "kev.json").write_text(json.dumps({
        "catalogVersion": "test", "dateReleased": "2026-09-29T00:00:00Z", "count": len(KEV_HITS),
        "vulnerabilities": [{"cveID": c} for c in KEV_HITS]}), encoding="utf-8")
    with open(intel / "epss.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cve", "epss", "percentile", "date", "fetched_at"])
        w.writeheader()
        for c in cves:
            w.writerow({"cve": c, "epss": "0.5", "percentile": "0.9", "date": "2026-09-29",
                        "fetched_at": "2026-09-30T00:00:00+00:00"})


def test_offline_end_to_end(cfg, tmp_path, real_cves):
    write_caches(tmp_path, real_cves)
    s = run(cfg, offline=True, mock=True)
    assert s["parse"]["vulns"] == 540 and s["parse"]["reports"] == 1
    assert s["intel"]["kev_version"] == "test" and s["intel"]["epss_dates"] == ["2026-09-29"]
    enriched = json.loads((tmp_path / "vulns_enriched.json").read_text(encoding="utf-8"))
    assert s["intel"]["kev_vulns"] == sum(bool(set(v["cves"]) & set(KEV_HITS)) for v in enriched["vulns"])
    out = json.loads((tmp_path / "enriched_alerts.json").read_text(encoding="utf-8"))
    assert out["is_mock"] is True and len(out["alerts"]) == 5
    assert [a["rank"] for a in out["alerts"]] == [1, 2, 3, 4, 5]
    cards = sorted(p.name for p in (tmp_path / "cards").iterdir())
    assert len(cards) == 12 and "index.html" in cards and "index.md" in cards
    assert s["render"]["files"] == 12


def test_rerun_keeps_snapshot_if_unchanged(cfg, tmp_path, real_cves):
    write_caches(tmp_path, real_cves)
    run(cfg, offline=True, formats=("md",), mock=True)
    before = (tmp_path / "vulns_enriched.json").read_text(encoding="utf-8")
    run(cfg, offline=True, formats=("md",), mock=True)
    # generated_at 不同但内容相同 → 不改写
    assert (tmp_path / "vulns_enriched.json").read_text(encoding="utf-8") == before


def test_offline_without_cache_fails(cfg):
    with pytest.raises(IntelError, match="离线模式"):
        run(cfg, offline=True, mock=True)


def test_online_uses_session(cfg, tmp_path):
    session = FakeSession()
    s = run(cfg, offline=False, formats=("md",), session=session, mock=True)
    urls = {u for u, _ in session.calls}
    assert urls == {KEV_URL, EPSS_URL}
    assert s["prioritize"]["alerts"] == 5
    assert (tmp_path / "intel" / "epss.csv").exists()


def test_missing_alerts_file(cfg, tmp_path, real_cves):
    write_caches(tmp_path, real_cves)
    cfg["fusion"]["mock_alerts_json"] = str(tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="make_mock_alerts"):
        run(cfg, offline=True, mock=True)
    cfg["fusion"]["alerts_json"] = str(tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="export_alerts"):
        run(cfg, offline=True)


def test_cli_exit_codes(cfg, tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["pipeline", "--config", str(path), "--offline"])
    assert main() == 3  # 无缓存 → intel 失败


def test_real_alerts_are_default_and_only_top_n_rendered(cfg, tmp_path, real_cves):
    """默认读真实报警（data/processed/alerts.json，is_mock=false）：全部参与融合打分，只为前 N 条生成卡片；
    mock 的输出在另一个目录，互不覆盖。"""
    write_caches(tmp_path, real_cves)
    real = json.loads((ROOT / cfg["fusion"]["alerts_json"]).read_text(encoding="utf-8"))
    assert len(real) > 50 and not any(a["is_mock"] for a in real) and {a["model"] for a in real} == {"lstm_ae"}
    cfg["fusion"]["max_rendered_alerts"] = 3
    s = run(cfg, offline=True, formats=("md",))
    assert s["prioritize"]["alerts"] == len(real) and s["prioritize"]["is_mock"] is False
    assert s["render"] == {"files": 4, "cards": 3}
    out = json.loads((tmp_path / "real" / "enriched_alerts.json").read_text(encoding="utf-8"))
    assert len(out["alerts"]) == len(real) and out["is_mock"] is False
    cards = sorted(p.name for p in (tmp_path / "real" / "cards").iterdir())
    assert len(cards) == 4 and cards[-1] == "index.md" and cards[0].startswith("01_LSTMAE-")
    index = (tmp_path / "real" / "cards" / "index.md").read_text(encoding="utf-8")
    assert f"共 {len(real)} 条报警" in index and "前 3 条" in index and "MOCK" not in index and "包含误报" in index
    card = (tmp_path / "real" / "cards" / cards[0]).read_text(encoding="utf-8")
    assert f"优先级第 1 / {len(real)}" in card and "占全部传感器误差的比例" in card and "仅用于演示" in card

    # 换成更小的 N 再跑：上一次多出来的卡片被清掉
    cfg["fusion"]["max_rendered_alerts"] = 1
    run(cfg, offline=True, formats=("md",))
    assert len(list((tmp_path / "real" / "cards").iterdir())) == 2
    cfg["fusion"]["max_rendered_alerts"] = 50
    run(cfg, offline=True, formats=("md",), mock=True)
    assert len(list((tmp_path / "cards").iterdir())) == 6 and len(list((tmp_path / "real" / "cards").iterdir())) == 2
