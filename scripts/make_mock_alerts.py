"""Generate the mock alerts data/mock/alerts.json (stand-in for alerts.json before the main line is done; each with is_mock: true).

The 5 alerts cover: a single subsystem, two subsystems, just above the threshold, a severe alert (S clipped to 1), plus a cross-subsystem alert involving P4 (HIL).
All sensor names come from the actual HAI 21.03 columns (data/samples/hai_columns.txt) and are validated one by one before writing.
Scores, thresholds and times are fictitious and do not represent the real output of any model.

Usage:
    python -m scripts.make_mock_alerts [--output JSON]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.0845  # Fictitious threshold

# (alert_id, scenario, start, end, max_score / threshold, [(sensor, share)])
CASES = [
    ("MOCK-001", "单子系统（P1）", "2020-07-15 10:02:10", "2020-07-15 10:05:39", 1.5,
     [("P1_FT03", 0.40), ("P1_LIT01", 0.25), ("P1_FCV03D", 0.15), ("P1_PIT01", 0.12), ("P1_TIT01", 0.08)]),
    ("MOCK-002", "跨两个子系统（P2 + P1）", "2020-07-15 14:30:00", "2020-07-15 14:36:59", 1.8,
     [("P2_SIT01", 0.35), ("P2_VT01", 0.20), ("P1_PIT01", 0.20), ("P1_FT01", 0.15), ("P2_CO_rpm", 0.10)]),
    ("MOCK-003", "刚过阈值（P3）", "2020-07-16 03:11:45", "2020-07-16 03:12:04", 1.02,
     [("P3_LIT01", 0.45), ("P3_LCV01D", 0.30), ("P3_FIT01", 0.15), ("P3_PIT01", 0.10)]),
    ("MOCK-004", "严重报警（P2，S 截断到 1）", "2020-07-16 09:00:00", "2020-07-16 09:14:59", 3.5,
     [("P2_SIT02", 0.50), ("P2_VXT02", 0.20), ("P2_VYT02", 0.15), ("P2_SIT01", 0.10), ("P2_VT01", 0.05)]),
    ("MOCK-005", "跨子系统且涉及 P4（HIL）", "2020-07-16 17:20:30", "2020-07-16 17:23:29", 1.3,
     [("P4_ST_PT01", 0.40), ("P1_TIT01", 0.25), ("P4_ST_TT01", 0.20), ("P1_PIT01", 0.15)]),
]


def load_sensor_names(path: Path) -> set[str]:
    """Extract the sensor columns (lines marked [P?]) from hai_columns.txt."""
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*\d+\s+\[P\d+\]\s+(\S+)$", line)
        if m:
            names.add(m.group(1))
    return names


def build_alerts(sensors: set[str]) -> list[dict]:
    alerts = []
    for aid, case, start, end, ratio, topk in CASES:
        unknown = [s for s, _ in topk if s not in sensors]
        if unknown:
            raise ValueError(f"{aid} 使用了 HAI 中不存在的传感器：{unknown}")
        if abs(sum(sh for _, sh in topk) - 1) > 1e-9:
            raise ValueError(f"{aid} 的 share 之和不为 1")
        alerts.append({
            "alert_id": aid,
            "start": start,
            "end": end,
            "max_score": round(THRESHOLD * ratio, 6),
            "threshold": THRESHOLD,
            "topk": [{"sensor": s, "share": sh} for s, sh in topk],
            "is_mock": True,
            "mock_case": case,
        })
    return alerts


def main() -> int:
    p = argparse.ArgumentParser(description="生成 mock alerts.json")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--output", type=Path, help="默认取 config.yaml 的 fusion.mock_alerts_json")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["fusion"]
    alerts = build_alerts(load_sensor_names(ROOT / cfg["hai_columns_txt"]))
    dst = args.output or ROOT / cfg["mock_alerts_json"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(alerts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[保存] {dst}：{len(alerts)} 条 mock 报警")
    return 0


if __name__ == "__main__":
    sys.exit(main())
