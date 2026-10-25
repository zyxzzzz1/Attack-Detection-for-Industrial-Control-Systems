"""Evaluation of the fusion ranking (I1): how many of the top k alerts are real, comparing three rankings.

- raw_score: by the maximum anomaly score of the alarm segment, descending.
- alpha=0: fusion priority, but with alpha = 0 (no vulnerability exposure risk E_sub, only anomaly severity S and subsystem criticality).
- alpha=1: fusion priority with alpha = 1 (the default in config/fusion.yaml).

precision@k = fraction of the top k alerts that are true alarms (overlap a labeled event). "True alarm or not" comes from eval_alarms.csv
written by the evaluate stage (ground truth is read only in evaluate.py); this module reads no ground truth, and neither scripts/export_alerts.py nor alerts.json has this column.
Alerts and the rows of eval_alarms.csv correspond one-to-one in order (same to_segments); a mismatch raises an error.

Note (constraint 7): the asset mapping is constructed by hand and the vulnerability data come from lab targets, with no causal link to the intrusions in HAI.
This comparison only shows whether real alarms rank high or low under this formula; it does not show that vulnerability context improves detection.

Output (official: reports/main/; --smoke: <artifact dir>/smoke_reports/): fusion_ranking_eval.csv / .md

Usage:
    python -m src.eval.fusion_ranking [--smoke]
"""

import argparse
import json
import sys

import pandas as pd

from src.eval.metrics import precision_at_k, wilson_interval
from src.fusion.asset_graph import AssetGraph
from src.fusion.prioritize import prioritize, severity
from src.fusion.risk import load_params
from src.utils.paths import load_config, resolve
from src.utils.reporting import ReportSource, add_common_args

COLUMNS = ["ranking", "k", "n_true", "precision", "ci_low", "ci_high", "n_alerts", "n_true_total", "base_rate",
           "n_tied_with_kth", "n_shared_with_raw_score"]


def truth_by_alert(alerts: list[dict], alarms: pd.DataFrame) -> dict[str, bool]:
    """alert_id -> whether it is a true alarm (overlaps a labeled event). Both sides must be the same set of alarm segments."""
    if len(alerts) != len(alarms):
        raise ValueError(f"alerts.json has {len(alerts)} entries and eval_alarms.csv {len(alarms)} rows: not artifacts of the same run")
    for a, row in zip(alerts, alarms.itertuples()):
        if (a["start"], a["end"]) != (row.start_time, row.end_time):
            raise ValueError(f"the start/end of {a['alert_id']} do not match eval_alarms.csv: not artifacts of the same run")
    return {a["alert_id"]: bool(t) for a, t in zip(alerts, alarms["overlaps_event"])}


def rankings(alerts: list[dict], vulns: list[dict], graph: AssetGraph, params: dict) -> dict[str, list[tuple[str, float]]]:
    """The three rankings, each as [(alert_id, ranking value), ...]. Ties are broken by alert_id (i.e. time order)."""
    out = {"raw_score": sorted(((a["alert_id"], a["max_score"]) for a in alerts), key=lambda x: (-x[1], x[0]))}
    for alpha in (0.0, 1.0):
        ranked = prioritize(alerts, vulns, graph, {**params, "alpha": alpha})["alerts"]
        out[f"alpha={alpha:g}"] = [(a["alert_id"], a["priority"]) for a in ranked]
    return out


def evaluate_rankings(ranked: dict[str, list[tuple[str, float]]], truth: dict[str, bool], ks: list[int]) -> pd.DataFrame:
    n, n_true = len(truth), sum(truth.values())
    rows = []
    reference = [i for i, _ in ranked["raw_score"]] if "raw_score" in ranked else []
    for name, order in ranked.items():
        hits = [truth[i] for i, _ in order]
        for k in ks:
            kk = min(k, n)
            shared = len({i for i, _ in order[:kk]} & set(reference[:kk]))  # Number of alerts in the top k shared with the raw-score ranking
            n_hit = sum(hits[:kk])
            lo, hi = wilson_interval(n_hit, kk)
            # Number of alerts with the same ranking value as the k-th: more than 1 means there are ties around position k, and which ones make the top k depends on tie order
            tied = sum(v == order[kk - 1][1] for _, v in order) if kk else 0
            rows.append({"ranking": name, "k": k, "n_true": n_hit, "precision": precision_at_k(hits, k),
                         "ci_low": lo, "ci_high": hi, "n_alerts": n, "n_true_total": n_true,
                         "base_rate": n_true / n if n else float("nan"), "n_tied_with_kth": tied,
                         "n_shared_with_raw_score": shared})
    return pd.DataFrame(rows, columns=COLUMNS)


def to_md(df: pd.DataFrame, model: str, n_saturated: int, smoke: bool) -> str:
    n, n_true = int(df["n_alerts"].iloc[0]), int(df["n_true_total"].iloc[0])
    lines = ["# Fusion ranking evaluation" + (" (smoke-test data, pipeline check only, do not cite)" if smoke else ""), "",
             f"{n_true} of the {n} alerts of {model} are true alarms (base rate {n_true / n:.4f}, the expected precision@k of a random ranking). "
             "precision@k is the fraction of the top k alerts that are true alarms; ci_low / ci_high are 95% Wilson confidence intervals.", "",
             "- raw_score: ranked by the maximum anomaly score within the alarm segment.",
             "- alpha=0: fusion priority without the vulnerability exposure risk (only anomaly severity S and subsystem criticality).",
             "- alpha=1: fusion priority with the vulnerability exposure risk E_sub (default parameters).", "",
             f"The anomaly severity S = min(1, max score ÷ threshold − 1) is capped once the score reaches twice the threshold: {n_saturated} alerts have S = 1. "
             "In the fusion ranking these alerts are distinguished only by subsystem weight, criticality and E_sub; n_tied_with_kth is the number of alerts with exactly the same priority as the k-th, "
             "and ties are broken by alert ID (time order). n_shared_with_raw_score is the number of alerts in the top k that are also in the top k of the raw-score ranking "
             "(equal precision does not mean the same alerts were selected).", "",
             "**Limitation (constraint 7)**: the asset mapping is constructed by hand and the vulnerability data come from lab targets, with no causal link to the intrusions in HAI. "
             "This table only describes how this ranking formula behaves on this data; it does not show that vulnerability context improves detection, and the fusion part claims no improvement of detection metrics.", "",
             "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fusion ranking evaluation: precision@k (raw score, alpha=0, alpha=1)")
    add_common_args(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    ml, fcfg = cfg["ml"], cfg["fusion"]
    model = ml["alerts"]["model"]
    src = ReportSource(cfg, args.smoke)
    src.banner()
    alerts_path = src.out / "alerts.json" if args.smoke else resolve(ml["alerts"]["output"])
    if not src.usable_runs([model], required=("eval_alarms.csv",)) or not alerts_path.exists():
        print(f"[ERROR] needs eval_alarms.csv of {model} and {alerts_path} (run scripts.export_alerts first)", file=sys.stderr)
        return 1
    alerts = json.loads(alerts_path.read_text(encoding="utf-8"))
    truth = truth_by_alert(alerts, pd.read_csv(src.root / model / "eval_alarms.csv"))
    graph = AssetGraph.from_yaml(resolve(fcfg["asset_map"]))
    params = load_params(resolve(fcfg["params"]))
    vulns = json.loads(resolve(cfg["vuln"]["vulns_enriched_json"]).read_text(encoding="utf-8"))["vulns"]

    df = evaluate_rankings(rankings(alerts, vulns, graph, params), truth, ml["fusion_eval"]["ks"])
    n_saturated = sum(severity(a["max_score"], a["threshold"]) >= 1.0 for a in alerts)
    df.to_csv(src.out / "fusion_ranking_eval.csv", index=False)
    (src.out / "fusion_ranking_eval.md").write_text(to_md(df, model, n_saturated, args.smoke), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"[GENERATED] {src.out}/fusion_ranking_eval.csv / .md ({n_saturated} alerts with capped S)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
