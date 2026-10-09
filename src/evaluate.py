"""评估阶段：逐点指标 + 事件级指标。全项目只有这里（和归因命中率评估 attribution.py）读取 test 标签（约束 2）。

- 检测指标只用 attack 列；子系统标签（attack_P*）只用来给每个事件记录子系统集合（事件内各秒的并集）。
- 不使用 point-adjust（约束 4）。分数为 NaN 的秒（每个文件前 W−1 秒）不参与任何指标，数量记为 n_unscored。
- 阈值来自 threshold.json（val_thr 分位数），这里不做任何调整。

产物（运行目录下）：
    eval_metrics.json   汇总：逐点、事件级、分文件的逐点指标，以及主表的一行（main_row）
    eval_events.csv     每个攻击事件：文件、起止时间、子系统集合、是否检出、延迟（画图脚本从这里取攻击区间，不再读标签）
    eval_alarms.csv     每个报警段：文件、起止时间、是否与攻击事件重叠

用法：
    python -m src.evaluate --model pca [--smoke]
"""

import json
import math
import sys

import numpy as np
import pandas as pd

from src.eval.labels import load_test_labels
from src.eval.metrics import event_metrics, point_metrics
from src.score import TEST_INDEX
from src.threshold import THRESHOLD_FILE
from src.utils.context import stage_cli

MAIN_COLUMNS = ["model", "precision", "recall", "f1", "auroc", "event_recall", "delay_median_s", "delay_mean_s",
                "false_alarms_per_hour", "threshold", "n_unscored"]


def clean(obj):
    """NaN → None，numpy 标量 → python 标量，便于写成标准 JSON。"""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    return None if isinstance(obj, float) and math.isnan(obj) else obj


def evaluate_scores(scores: np.ndarray, index: dict, labels: dict[str, pd.DataFrame], threshold: float,
                    time_col: str, label_cols: list[str], merge_gap_s: int):
    """scores 为各 test 文件拼接的逐秒分数；index 为 test_index.json；labels 为各文件的标签表。

    冒烟测试时每个文件只打了前一部分的分，标签按同样的行数截取。返回 (汇总 dict, 事件表, 报警段表)。
    """
    main_col, sub_cols = label_cols[0], label_cols[1:]
    per_file, pairs, tables = [], [], []
    for f in index["files"]:
        df = labels[f["file"]].iloc[f["start_row"]:f["start_row"] + f["rows"]]
        if len(df) != f["rows"] or pd.Timestamp(df[time_col].iloc[0]) != pd.Timestamp(f["start_time"]):
            raise ValueError(f"{f['file']} 的标签与分数对不上（行数或起始时间不一致）")
        s = scores[f["offset"]:f["offset"] + f["rows"]]
        y = df[main_col].to_numpy()
        per_file.append({"file": f["file"], "start_row": f["start_row"], "rows": f["rows"], **point_metrics(s, y, threshold)})
        pairs.append((s, y))
        tables.append(df)

    y_all = np.concatenate([y for _, y in pairs])
    point = point_metrics(scores, y_all, threshold)
    event, events, alarms = event_metrics(pairs, threshold, merge_gap_s)

    def with_times(rows: list[dict]) -> pd.DataFrame:
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out.insert(0, "file", [index["files"][k]["file"] for k in out["file_index"]])
        for col in ("start", "end"):
            out[f"{col}_time"] = [tables[k][time_col].iloc[i] for k, i in zip(out["file_index"], out[col])]
        return out.rename(columns={"file_index": "part"})  # part = test_index.json 中的第几段

    events_df, alarms_df = with_times(events), with_times(alarms)
    if not events_df.empty:
        # 事件的子系统集合 = 事件内所有秒的子系统标签并集；并集为空的事件单列（主线补充决策 7）
        subsystems = []
        for e in events:
            block = tables[e["file_index"]].iloc[e["start"]:e["end"] + 1]
            subsystems.append("+".join(c.split("_", 1)[1] for c in sub_cols if block[c].any()))
        events_df.insert(1, "event_id", range(1, len(events_df) + 1))
        events_df["subsystems"] = subsystems
        event["n_events_without_subsystem"] = sum(s == "" for s in subsystems)
    summary = {"threshold": threshold, "point": point, "event": event, "per_file": per_file}
    return summary, events_df, alarms_df


def run(ctx) -> dict:
    hai = ctx.cfg["ml"]["hai"]
    index = json.loads((ctx.directory / TEST_INDEX).read_text(encoding="utf-8"))
    thr = json.loads((ctx.directory / THRESHOLD_FILE).read_text(encoding="utf-8"))
    scores = np.load(ctx.directory / "scores_test.npy")
    labels = load_test_labels(ctx.cfg)
    summary, events_df, alarms_df = evaluate_scores(
        scores, index, labels, thr["threshold"], hai["time_col"], hai["label_cols"],
        ctx.cfg["ml"]["eval"]["merge_gap_s"])
    point, event = summary["point"], summary["event"]
    main_row = {"model": ctx.model, **{k: point[k] for k in ("precision", "recall", "f1", "auroc")},
                **{k: event[k] for k in ("event_recall", "delay_median_s", "delay_mean_s", "false_alarms_per_hour")},
                "threshold": thr["threshold"], "n_unscored": point["n_unscored"]}
    # 阈值分位的敏感性（只作分析，不据此选阈值）：用 threshold.json 中其他分位的阈值重算同一套指标
    sensitivity = {}
    for q, value in thr.get("sensitivity", {}).items():
        s, _, _ = evaluate_scores(scores, index, labels, value, hai["time_col"], hai["label_cols"],
                                  ctx.cfg["ml"]["eval"]["merge_gap_s"])
        sensitivity[q] = {"threshold": value, **{k: s["point"][k] for k in ("precision", "recall", "f1")},
                          **{k: s["event"][k] for k in ("event_recall", "delay_median_s", "false_alarms_per_hour")}}
    summary["threshold_sensitivity"] = sensitivity
    summary.update(model=ctx.model, run_name=ctx.run_name, smoke=ctx.smoke, quantile=thr["quantile"],
                   window=index["window"], point_adjust=False, main_row=main_row)
    (ctx.directory / "eval_metrics.json").write_text(json.dumps(clean(summary), ensure_ascii=False, indent=2) + "\n",
                                                     encoding="utf-8")
    events_df.to_csv(ctx.directory / "eval_events.csv", index=False)
    alarms_df.to_csv(ctx.directory / "eval_alarms.csv", index=False)
    ctx.log.info("逐点（不使用 point-adjust）：P %.4f  R %.4f  F1 %.4f  AUROC %.4f；排除未打分 %d 秒",
                 point["precision"], point["recall"], point["f1"], point["auroc"], point["n_unscored"])
    ctx.log.info("事件：检出 %d / %d，漏检 %d，延迟中位数 %s 秒；报警段 %d 个，其中误报 %d 个（%.3f 次/小时）",
                 event["n_detected"], event["n_events"], event["n_missed"], event["delay_median_s"],
                 event["n_alarm_segments"], event["n_false_alarm_segments"], event["false_alarms_per_hour"])
    return clean(main_row)


if __name__ == "__main__":
    sys.exit(stage_cli("evaluate", "用 test 标签评估已打分的模型"))
