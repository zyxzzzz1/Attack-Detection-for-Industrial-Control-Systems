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
    assert kev.count("| CVE-") == 9 and kev.count("否（需人工核实）") == 7
    nums = (tmp_path / "key_numbers.md").read_text(encoding="utf-8")
    for s in ["写入 540 条", "Critical（CVSS ≥ 9.0）101 条", "保留 66 / 540 条", "| hmi_main | 未评估 |",
              "| P3 | 未评估（按 0） |", "仅用于演示"]:
        assert s in nums, s
    assert "MOCK" not in nums  # The report material uses only real data


def test_topology_asset_label_omits_missing_ip(monkeypatch):
    """Topology: assets without an IP show only the role, not " · -"; those with an IP show "role · IP"."""
    from types import SimpleNamespace as NS

    from scripts.make_extension_report import asset_head
    from src.utils import i18n
    assert asset_head(NS(role="历史服务器", ip=None)) == "历史服务器" and asset_head(NS(role="PLC", ip=None)) == "PLC"
    assert asset_head(NS(role="工程师站", ip="192.168.56.11")) == "工程师站 · 192.168.56.11"
    monkeypatch.setenv(i18n.ENV_VAR, "en")
    assert asset_head(NS(role="历史服务器", ip=None)) == "Historian"
    assert asset_head(NS(role="工程师站", ip="192.168.56.11")) == "Engineering workstation · 192.168.56.11"
