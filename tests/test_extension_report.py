"""Tests of scripts/make_extension_report.py: generate from the real snapshots into a temporary directory."""

import sys

from scripts.make_extension_report import main


def test_generates_all_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["make_extension_report", "--output-dir", str(tmp_path)])
    assert main() == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["asset_topology.png", "kev_table.md", "key_numbers.md", "qod_distribution.png"]
    assert all((tmp_path / n).stat().st_size > 10_000 for n in names if n.endswith(".png"))
    kev = (tmp_path / "kev_table.md").read_text(encoding="utf-8")
    assert kev.count("| CVE-") == 9 and kev.count("no (needs manual verification)") == 7
    nums = (tmp_path / "key_numbers.md").read_text(encoding="utf-8")
    for s in ["540 written", "Critical (CVSS ≥ 9.0): 101", "kept 66 / 540", "| hmi_main | Not assessed |",
              "| P3 | not assessed (counted as 0) |", "for demonstration only"]:
        assert s in nums, s
    assert "MOCK" not in nums  # The report material uses only real data


def test_topology_asset_label_omits_missing_ip():
    """Topology: assets without an IP show only the role, not " · -"; those with an IP show "role · IP"."""
    from types import SimpleNamespace as NS

    from scripts.make_extension_report import asset_head
    assert asset_head(NS(role="Historian", ip=None)) == "Historian" and asset_head(NS(role="PLC", ip=None)) == "PLC"
    assert asset_head(NS(role="Engineering workstation", ip="192.168.56.11")) == "Engineering workstation · 192.168.56.11"
