"""把 LSTM-AE 的报警段导出为 alerts.json（I1），字段与 data/mock/alerts.json 的契约一致。不读取任何标签。

每个报警段一条：alert_id、start、end（HAI 原始时间格式）、max_score、threshold、topk（[{sensor, share}]）、
is_mock=false、model=模型名。报警段由 src/eval/alarms.py 的 alarm_segments 得到（内部调用 to_segments），
与 evaluate.py 的事件级统计、attribution.py 的归因用的是同一个定义（主线补充决策 6、8）。
share 是该传感器占段内全部传感器误差的比例（不在 Top-k 内重新归一化），所以 Top-k 的 share 之和不超过 1。

- 正式（默认）：只接受集群上完整运行的 LSTM-AE 产物（非冒烟、git_dirty=false、已完成），写到 ml.alerts.output
  （data/processed/alerts.json，入库）。
- --smoke：读冒烟产物，写到 产物目录/smoke_reports/alerts.json，不会碰 data/processed/。

用法：
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
    p = argparse.ArgumentParser(description="导出真实报警 alerts.json（LSTM-AE）")
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    ml = cfg["ml"]
    model = ml["alerts"]["model"]
    src = ReportSource(cfg, args.smoke)
    src.banner()
    if not src.usable_runs([model], required=("scores_test.npy", "per_sensor_err_test.npy", "threshold.json",
                                              "test_index.json")):
        print("[错误] 没有可用于导出的运行", file=sys.stderr)
        return 1
    run = load_run(src.root / model, per_sensor=True)
    alerts = build_alerts(run, model, ml["eval"]["merge_gap_s"], ml["alerts"]["top_k"])
    out = src.out / "alerts.json" if args.smoke else resolve(ml["alerts"]["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(alerts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[保存] {out}（{len(alerts)} 条报警，模型 {model}，阈值 {run.threshold['threshold']:.6g}，is_mock=false）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
