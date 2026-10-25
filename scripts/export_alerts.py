"""Export the LSTM-AE alarm segments to alerts.json (I1), with fields matching the contract of data/mock/alerts.json. Reads no labels.

One entry per alarm segment: alert_id, start, end (HAI raw time format), max_score, threshold, topk ([{sensor, share}]),
is_mock=false, model=<model name>. Alarm segments come from alarm_segments in src/eval/alarms.py (which calls to_segments),
the same definition used by the event-level statistics of evaluate.py and the attribution of attribution.py (main-line decisions 6, 8).
share is the sensor's fraction of the total error of all sensors in the segment (not renormalized within the Top-k), so the Top-k shares sum to at most 1.

- Official (default): accepts only artifacts of a full LSTM-AE run on the cluster (not smoke, git_dirty=false, completed), written to ml.alerts.output
  (data/processed/alerts.json, tracked).
- --smoke: reads smoke artifacts and writes to <artifact dir>/smoke_reports/alerts.json, never touching data/processed/.

Usage:
    python -m scripts.export_alerts [--smoke]
"""

import argparse
import json
import sys

from src.eval.alarms import alarm_segments, sensor_shares, top_k
from src.eval.run_io import load_run
from src.utils.paths import load_config, resolve
from src.utils.reporting import ReportSource, add_common_args


def build_alerts(run, model: str, merge_gap_s: int, k: int) -> list[dict]:
    threshold = run.threshold["threshold"]
    prefix = model.upper().replace("_", "")
    alerts = []
    for seg in alarm_segments(run.scores, run.index, threshold, merge_gap_s):
        shares = sensor_shares(run.per_sensor_err, seg["global_start"], seg["global_end"])
        alerts.append({"alert_id": f"{prefix}-{len(alerts) + 1:05d}", "start": seg["start_time"], "end": seg["end_time"],
                       "max_score": seg["max_score"], "threshold": threshold,
                       "topk": [{"sensor": t["sensor"], "share": round(t["share"], 6)} for t in top_k(shares, run.features, k)],
                       "is_mock": False, "model": model})
    return alerts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export the real alerts alerts.json (LSTM-AE)")
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    ml = cfg["ml"]
    model = ml["alerts"]["model"]
    src = ReportSource(cfg, args.smoke)
    src.banner()
    if not src.usable_runs([model], required=("scores_test.npy", "per_sensor_err_test.npy", "threshold.json",
                                              "test_index.json")):
        print("[ERROR] no run available for export", file=sys.stderr)
        return 1
    run = load_run(src.root / model, per_sensor=True)
    alerts = build_alerts(run, model, ml["eval"]["merge_gap_s"], ml["alerts"]["top_k"])
    out = src.out / "alerts.json" if args.smoke else resolve(ml["alerts"]["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(alerts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[SAVED] {out} ({len(alerts)} alerts, model {model}, threshold {run.threshold['threshold']:.6g}, is_mock=false)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
