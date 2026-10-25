"""Tests of scripts/compare_fusion.py: builds a before and an after fusion result."""

from scripts.compare_fusion import compare


def result(r_hmi, e3, order, is_mock=True):
    return {
        "is_mock": is_mock,
        "asset_risk": {"ews_01": {"R_asset": 1.0, "status": "scanned"},
                       "hmi_main": {"R_asset": r_hmi, "status": "scanned" if r_hmi is not None else "not_scanned"}},
        "subsystem_risk": {"P1": {"E_sub": 1.0, "max_asset": "ews_01"},
                           "P3": {"E_sub": e3, "max_asset": "hmi_main" if e3 else None}},
        "alerts": [{"alert_id": aid, "rank": i + 1, "priority": p} for i, (aid, p) in enumerate(order)],
    }


def test_compare_reports_changes():
    before = result(None, 0.0, [("A", 1.0), ("B", 0.5)])
    after = result(0.4, 0.4, [("B", 1.2), ("A", 1.0)])
    text = compare(before, after)
    assert "| hmi_main | Not assessed | 0.400 | newly assessed | not_scanned → scanned |" in text
    assert "| P3 | 0.000 | 0.400 | +0.400 | hmi_main |" in text
    assert "| B | 2 → 1 | 0.500 | 1.200 | +0.700 |" in text
    assert "Ranking changed: yes" in text and "[MOCK]" in text and "for demonstration only" in text


def test_compare_no_change_without_mock():
    same = result(None, 0.0, [("A", 1.0)], is_mock=False)
    text = compare(same, same)
    assert "Ranking changed: no" in text and "[MOCK]" not in text
    assert "| hmi_main | Not assessed | Not assessed | - |" in text
