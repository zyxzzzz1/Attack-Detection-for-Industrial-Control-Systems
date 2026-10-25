"""Sensor attribution and its evaluation (M5, main-line decision 7). Apart from evaluate.py, the only module that reads test labels.

Within each alarm segment, the mean per_sensor_err of each sensor is normalized to a share and the Top-k is taken (k from ml.attribution.top_k).
Alarm segments and Top-k are computed in src/eval/alarms.py (label-free), shared with scripts/export_alerts.py; this module only evaluates them against labels:

- Subsystem hit rate is computed only for alarm segments overlapping an attack event: a hit if the Top-1 sensor's subsystem is in the event's subsystem set.
  When a segment overlaps several events, the event with the longest overlap decides (primary definition; ties go to the earlier event);
  a union definition is also output as a supplement (Top-1 subsystem in the union of those events' subsystem sets), together with the number of segments overlapping several events.
- An event's subsystem set = union of attack_P* over all its seconds. Events with an empty union are excluded from the hit-rate denominator and listed as "events without subsystem labels".
- The share of Top-1 falling in P4 is reported separately (P4 is a HIL simulation without a subsystem label, so a Top-1 in P4 is always a miss).
- Alarm segments not overlapping any attack are listed separately and excluded from the hit rate.
- Sensor-level Hit@k is not done for now: the HAI 21.03 CSVs have no per-sensor attack targets; to be added once the official attack-target table is available.
- Per-event hit rate (decided 2026-10-09): the primary definition takes each event's first overlapping alarm segment (the one an operator sees first)
  and checks whether its Top-1 subsystem is in the event's subsystem set; the supplementary definition merges the errors of all alarm seconds within the event and takes the Top-1.
- All hit rates come with 95% Wilson confidence intervals; hit rates are also given per alarm-segment duration group (ml.attribution.duration_bins_s).
- Multiple random seeds (ml.seed_study): computed once per seed run, reported as mean ± sample std.
- Exploratory analysis (post hoc, does not replace the main results): in-segment errors are divided per sensor by its mean error on val_thr before ranking Top-k.
  The baseline comes only from val_thr (val_sensor_contrib.csv in the run directory); this step uses no test labels; labels are only used afterwards to evaluate hit rates.

Output (official: reports/main/; --smoke: <artifact dir>/smoke_reports/):
    attribution.csv            one row per alarm segment (all compared models)
    attribution_summary.csv/.md  one row per model: hit rates, P4 share, counts of each kind of alarm segment
    attribution_heatmap_<model>.png  error share per subsystem for each detected event
    attribution_by_duration.csv  hit rates per alarm-segment duration group (one set each for the main results and the exploratory analysis)
    attribution_seeds.csv        per-seed hit rates with mean and std across the random seeds
    attribution_exploratory_valnorm.csv  exploratory analysis: hit rates before and after normalizing by the val_thr mean error

Usage:
    python -m src.attribution [--models lstm_ae lstm_vae pca] [--smoke]
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.diagnose import CONTRIB_FILE
from src.eval.alarms import alarm_segments, normalize_shares, sensor_shares, subsystem_of, subsystem_shares, top_k
from src.eval.labels import load_test_labels
from src.eval.run_io import load_run
from src.eval.metrics import wilson_interval
from src.eval.segments import overlaps, to_segments
from src.utils.paths import load_config
from src.utils.reporting import ReportSource, add_common_args

SUMMARY_COLUMNS = ["model", "n_alarm_segments", "n_overlap_attack", "n_evaluable", "n_hit", "subsystem_hit_rate",
                   "subsystem_hit_rate_ci_low", "subsystem_hit_rate_ci_high",
                   "n_multi_event_segments", "n_evaluable_union", "n_hit_union", "subsystem_hit_rate_union",
                   "n_top1_p4", "top1_p4_ratio", "n_overlap_no_subsystem_label", "n_non_overlap",
                   "n_events", "n_events_without_subsystem", "n_events_attributed",
                   "n_events_first_evaluable", "n_events_first_hit", "event_first_hit_rate",
                   "event_first_hit_rate_ci_low", "event_first_hit_rate_ci_high",
                   "n_events_merged_hit", "event_hit_rate", "event_hit_rate_ci_low", "event_hit_rate_ci_high"]
RATE_COLUMNS = ("subsystem_hit_rate", "event_first_hit_rate", "event_hit_rate")  # Columns summarized as mean ± std across seeds


def label_events(index: dict, labels: dict[str, pd.DataFrame], label_cols: list[str]) -> list[dict]:
    """Attack events per part and their subsystem sets. part is the index of the part in test_index.json; start / end are row numbers within the part."""
    main_col, sub_cols = label_cols[0], label_cols[1:]
    events = []
    for k, part in enumerate(index["files"]):
        df = labels[part["file"]].iloc[part["start_row"]:part["start_row"] + part["rows"]]
        if len(df) != part["rows"]:
            raise ValueError(f"the label rows of {part['file']} do not match the scores")
        for a, b in to_segments(df[main_col].to_numpy(), 0):
            block = df.iloc[a:b + 1]
            subs = sorted(c.split("_", 1)[1] for c in sub_cols if block[c].any())
            events.append({"event_id": len(events) + 1, "part": k, "file": part["file"], "start": a, "end": b,
                           "subsystems": subs})
    return events


def attribute_run(scores: np.ndarray, per_sensor_err: np.ndarray, index: dict, threshold: float, merge_gap_s: int,
                  events: list[dict], ks: list[int], baseline: np.ndarray | None = None) -> tuple[list[dict], dict, list[dict]]:
    """Return (attribution and evaluation per alarm segment, summary, error share per subsystem for each event).

    baseline is only for the exploratory analysis: each sensor's mean error on val_thr; when given, errors are divided by it before computing shares.
    """
    features = index["features"]
    rows, event_err = [], {}
    for seg in alarm_segments(scores, index, threshold, merge_gap_s):
        shares = sensor_shares(per_sensor_err, seg["global_start"], seg["global_end"], baseline)
        tops = {k: top_k(shares, features, k) for k in ks}
        top1 = tops[ks[0]][0] if tops[ks[0]] else None
        hit_events = [e for e in events if e["part"] == seg["part"] and overlaps((seg["start"], seg["end"]), (e["start"], e["end"]))]
        # Primary definition: the event with the longest overlap (max picks the first maximum and events are in time order, so ties go to the earlier one)
        primary = max(hit_events, key=lambda e: min(seg["end"], e["end"]) - max(seg["start"], e["start"]), default=None)
        target = primary["subsystems"] if primary else []
        union = sorted({s for e in hit_events for s in e["subsystems"]})  # Supplementary definition
        top1_sub = subsystem_of(top1["sensor"]) if top1 else None
        if not hit_events:
            category, hit = "non_overlap", None
        elif not target:
            category, hit = "overlap_no_subsystem_label", None
        else:
            category, hit = "evaluable", top1_sub in target
        rows.append({**{k: seg[k] for k in ("file", "start_time", "end_time", "duration_s", "max_score")},
                     "category": category, "event_ids": "+".join(str(e["event_id"]) for e in hit_events),
                     "n_events_overlap": len(hit_events), "primary_event_id": primary["event_id"] if primary else None,
                     "event_subsystems": "+".join(target), "union_subsystems": "+".join(union),
                     "top1_sensor": top1["sensor"] if top1 else None,
                     "top1_subsystem": top1_sub, "top1_share": top1["share"] if top1 else None, "hit": hit,
                     "hit_union": (top1_sub in union) if union else None,
                     **{f"top{k}": json.dumps(tops[k], ensure_ascii=False) for k in ks}})
        # Event level: accumulate the errors within the alarm segments overlapping the event (only the alarm seconds inside the event interval)
        for e in hit_events:
            lo = index["files"][e["part"]]["offset"] + max(seg["start"], e["start"])
            hi = index["files"][e["part"]]["offset"] + min(seg["end"], e["end"])
            block = np.asarray(per_sensor_err[lo:hi + 1], dtype=np.float64)
            acc = event_err.setdefault(e["event_id"], [np.zeros(len(features)), 0])
            acc[0] += block.sum(axis=0)
            acc[1] += len(block)

    event_rows = []
    for e in events:
        if e["event_id"] not in event_err:
            continue
        total, n = event_err[e["event_id"]]
        shares = normalize_shares(total / max(n, 1), baseline)
        by_sub = subsystem_shares(shares, features)
        top = top_k(shares, features, 1)
        event_rows.append({"event_id": e["event_id"], "file": e["file"], "subsystems": "+".join(e["subsystems"]),
                           "alarm_seconds": n, "top1_sensor": top[0]["sensor"] if top else None,
                           "top1_subsystem": subsystem_of(top[0]["sensor"]) if top else None,
                           "hit": (subsystem_of(top[0]["sensor"]) in e["subsystems"]) if top and e["subsystems"] else None,
                           **{f"share_{s}": v for s, v in sorted(by_sub.items())}})

    overlap = [r for r in rows if r["category"] != "non_overlap"]
    evaluable = [r for r in rows if r["category"] == "evaluable"]
    n_hit = sum(bool(r["hit"]) for r in evaluable)
    union_eval = [r for r in rows if r["hit_union"] is not None]
    n_hit_union = sum(bool(r["hit_union"]) for r in union_eval)
    n_p4 = sum(r["top1_subsystem"] == "P4" for r in overlap)
    ev_eval = [r for r in event_rows if r["hit"] is not None]
    summary = {
        "n_alarm_segments": len(rows), "n_overlap_attack": len(overlap), "n_evaluable": len(evaluable), "n_hit": n_hit,
        "subsystem_hit_rate": n_hit / len(evaluable) if evaluable else float("nan"),
        "n_multi_event_segments": sum(r["n_events_overlap"] > 1 for r in rows),
        "n_evaluable_union": len(union_eval), "n_hit_union": n_hit_union,
        "subsystem_hit_rate_union": n_hit_union / len(union_eval) if union_eval else float("nan"),
        "n_top1_p4": n_p4, "top1_p4_ratio": n_p4 / len(overlap) if overlap else float("nan"),
        "n_overlap_no_subsystem_label": sum(r["category"] == "overlap_no_subsystem_label" for r in rows),
        "n_non_overlap": len(rows) - len(overlap),
        "n_events": len(events), "n_events_without_subsystem": sum(not e["subsystems"] for e in events),
        "n_events_attributed": len(event_rows),
        "event_hit_rate": sum(bool(r["hit"]) for r in ev_eval) / len(ev_eval) if ev_eval else float("nan"),
    }
    return rows, summary, event_rows


def add_first_segment(event_rows: list[dict], rows: list[dict]) -> list[dict]:
    """Primary per-event definition: each event's first overlapping alarm segment (rows are in time order) and whether its Top-1 is a hit."""
    first: dict[int, dict] = {}
    for r in rows:
        for eid in (int(x) for x in r["event_ids"].split("+") if x):
            first.setdefault(eid, r)
    out = []
    for e in event_rows:
        r = first[e["event_id"]]
        subs = [x for x in e["subsystems"].split("+") if x]
        out.append({**e, "first_start_time": r["start_time"], "first_duration_s": r["duration_s"],
                    "first_top1_sensor": r["top1_sensor"], "first_top1_subsystem": r["top1_subsystem"],
                    "first_hit": (r["top1_subsystem"] in subs) if subs and r["top1_subsystem"] else None})
    return out


def rate_with_ci(hits: list) -> dict:
    """None entries in hits are excluded from the denominator. Returns n, n_hit, rate and the 95% Wilson interval."""
    valid = [bool(h) for h in hits if h is not None]
    n, k = len(valid), sum(valid)
    lo, hi = wilson_interval(k, n)
    return {"n": n, "n_hit": k, "rate": k / n if n else float("nan"), "ci_low": lo, "ci_high": hi}


def full_summary(summary: dict, rows: list[dict], event_rows: list[dict]) -> dict:
    """Summary of attribute_run plus confidence intervals and the two per-event definitions."""
    seg = rate_with_ci([r["hit"] for r in rows if r["category"] == "evaluable"])
    first = rate_with_ci([e["first_hit"] for e in event_rows])
    merged = rate_with_ci([e["hit"] for e in event_rows])
    return {**summary,
            "subsystem_hit_rate_ci_low": seg["ci_low"], "subsystem_hit_rate_ci_high": seg["ci_high"],
            "n_events_first_evaluable": first["n"], "n_events_first_hit": first["n_hit"], "event_first_hit_rate": first["rate"],
            "event_first_hit_rate_ci_low": first["ci_low"], "event_first_hit_rate_ci_high": first["ci_high"],
            "n_events_merged_hit": merged["n_hit"],
            "event_hit_rate_ci_low": merged["ci_low"], "event_hit_rate_ci_high": merged["ci_high"]}


def duration_group(duration_s: int, bins: list[int]) -> str:
    """bins = [10, 60] -> "<10 s", "10–60 s" (inclusive), ">60 s"."""
    lo, hi = bins
    return f"<{lo} s" if duration_s < lo else f"{lo}–{hi} s" if duration_s <= hi else f">{hi} s"


def by_duration(rows: list[dict], bins: list[int]) -> list[dict]:
    """Hit rates of evaluable alarm segments overlapping attacks, grouped by duration."""
    groups = [f"<{bins[0]} s", f"{bins[0]}–{bins[1]} s", f">{bins[1]} s"]
    evaluable = [r for r in rows if r["category"] == "evaluable"]
    return [{"duration_group": g,
             **rate_with_ci([r["hit"] for r in evaluable if duration_group(r["duration_s"], bins) == g])} for g in groups]


def val_thr_baseline(run_dir, features: list[str]) -> np.ndarray:
    """Mean squared error of each sensor on val_thr (written by the diagnose stage, based on val_thr only), in the order of features."""
    df = pd.read_csv(run_dir / CONTRIB_FILE).set_index("sensor")
    return df.loc[features, "mean_sq_err"].to_numpy(dtype=np.float64)


def plot_heatmap(event_rows: list[dict], model: str, path, smoke: bool) -> None:
    """Rows = detected events, columns = subsystems, color = share of the event's alarm-second error in each subsystem; dots mark the attacked subsystems according to the labels."""
    from src.utils.plotstyle import INK, SEQUENTIAL, SURFACE, apply_style, plt
    apply_style()
    df = pd.DataFrame(event_rows)
    cols = sorted(c for c in df.columns if c.startswith("share_"))
    values = df[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 3.2, 0.24 * len(df) + 1.6))
    im = ax.imshow(values, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(df.itertuples()):
        for j, col in enumerate(cols):
            if col.removeprefix("share_") in row.subsystems.split("+"):
                ax.plot(j, i, "o", ms=5, mfc=SURFACE, mec=INK, mew=1)
    ax.set_xticks(range(len(cols)), [c.removeprefix("share_") for c in cols])
    ax.set_yticks(range(len(df)), [f"{r.file.split('.')[0]} #{r.event_id}" for r in df.itertuples()], fontsize=7)
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(df) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)  # Leave a background-colored gap between adjacent cells
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).set_label("Error share")
    ax.set_title("{model}: error share by subsystem for the detected events".format(model=model) + f"{' (smoke-test data, pipeline check only)' if smoke else ''}\n"
                 + "○ = attacked subsystem according to the labels", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summary_md(df: pd.DataFrame, smoke: bool, duration: pd.DataFrame | None = None, seeds: pd.DataFrame | None = None,
               explore: pd.DataFrame | None = None) -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else f"{v:.4f}"
        return str(v)

    def table(frame: pd.DataFrame) -> list[str]:
        cols = list(frame.columns)
        return (["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                + ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)])

    lines = ["# Attribution evaluation summary" + (" (smoke-test data, pipeline check only, do not cite)" if smoke else ""), "",
             "For each alarm segment, take the share of each sensor's mean reconstruction error within the segment; a hit if the Top-1 sensor's subsystem is in the event's subsystem set. "
             "When an alarm segment overlaps several events (n_multi_event_segments of them), the primary definition uses the event with the longest overlap "
             "(n_evaluable, n_hit, subsystem_hit_rate); the union definition (Top-1 subsystem in the union of those events' subsystem sets, "
             "n_evaluable_union, n_hit_union, subsystem_hit_rate_union) is a supplement and can only have at least as many hits as the primary one. "
             "Only alarm segments overlapping an attack event whose event has subsystem labels count toward the hit rate; "
             "alarm segments overlapping no attack (n_non_overlap) and those overlapping only events without subsystem labels are listed separately. "
             "top1_p4_ratio is the fraction of attack-overlapping alarm segments whose Top-1 is in P4 (P4 has no subsystem label, so Top-1 in P4 is always a miss).", "",
             "There are two per-event hit-rate definitions, both with the detected events that have subsystem labels as the denominator:", "",
             "- **Primary, event_first_hit_rate**: take each event's first overlapping alarm segment (by segment start time) "
             "and check whether its Top-1 subsystem is in the event's subsystem set. This is the alert an operator sees first; "
             "the segment's error is computed over the whole segment (the alarm may start before the event does).",
             "- **Supplementary, event_hit_rate**: merge the errors of all alarm seconds within the event interval and take the Top-1. It uses every alarm up to the end of the event, "
             "which is not available when the alarm first appears, so it is only a supplement.", "",
             "ci_low and ci_high after each hit rate are 95% Wilson confidence intervals. The denominators are small and the intervals wide, so differences between models must be read together with the intervals.", "",
             "Sensor-level Hit@k was not done: the HAI 21.03 data files have no per-sensor attack-target annotation.", "",
             *table(df[SUMMARY_COLUMNS])]
    if duration is not None and len(duration):
        lines += ["", "## By alarm-segment duration", "",
                  "Only attack-overlapping, evaluable alarm segments. variant = main is the main result; valnorm is the exploratory analysis below.", "",
                  *table(duration)]
    if seeds is not None and len(seeds):
        lines += ["", "## Multiple random seeds", "",
                  "Attribution computed once for each seed run in ml.seed_study (identical configuration apart from the seed); the last two rows are the mean and sample std (ddof=1). "
                  "The main result is still the main-model run (the seed in the first row).", "", *table(seeds)]
    if explore is not None and len(explore):
        lines += ["", "## Exploratory analysis: normalized by the val_thr mean error (post hoc, does not replace the main results)", "",
                  "**This analysis was done after seeing the main results.** Method: the mean error of each sensor within the alarm segment is first divided by that sensor's mean error on val_thr, "
                  "then normalized and ranked for Top-k, i.e. what is compared is the \"multiple of the usual error\". The baseline comes only from val_thr and uses no test labels; "
                  "but the decision to do this step was made after seeing the hit rates on the test set, so its hit rates are optimistic "
                  "and must not be cited as the result of a method fixed in advance. The main results, alerts.json and the alert cards do not use this normalization.", "",
                  *table(explore)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sensor attribution and subsystem hit rate (M5)")
    add_common_args(p)
    p.add_argument("--models", nargs="+", help="defaults to ml.attribution.models in the config")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    ml = cfg["ml"]
    src = ReportSource(cfg, args.smoke)
    src.banner()
    models = src.usable_runs(args.models or ml["attribution"]["models"],
                             required=("scores_test.npy", "per_sensor_err_test.npy", "threshold.json", "test_index.json"))
    if not models:
        print("[INFO] no runs available, no attribution results generated")
        return 0
    labels = load_test_labels(cfg)
    attr, bins = ml["attribution"], ml["attribution"]["duration_bins_s"]

    def analyse(run, baseline=None):
        events = label_events(run.index, labels, ml["hai"]["label_cols"])
        rows, summary, event_rows = attribute_run(run.scores, run.per_sensor_err, run.index, run.threshold["threshold"],
                                                  ml["eval"]["merge_gap_s"], events, attr["top_k"], baseline)
        event_rows = add_first_segment(event_rows, rows)
        return rows, full_summary(summary, rows, event_rows), event_rows

    all_rows, summaries, duration_rows, explore_rows = [], [], [], []
    for name in models:
        run = load_run(src.root / name, per_sensor=True)
        rows, summary, event_rows = analyse(run)
        all_rows += [{"model": name, **r} for r in rows]
        summaries.append({"model": name, **summary})
        duration_rows += [{"model": name, "variant": "main", **d} for d in by_duration(rows, bins)]
        if event_rows:
            plot_heatmap(event_rows, name, src.out / f"attribution_heatmap_{name}.png", args.smoke)
            pd.DataFrame(event_rows).to_csv(src.out / f"attribution_events_{name}.csv", index=False)
        print(f"[ATTRIBUTION] {name}: alarm segments {summary['n_alarm_segments']}, overlapping attacks {summary['n_overlap_attack']}, "
              f"evaluable {summary['n_evaluable']}, hits {summary['n_hit']} (union {summary['n_hit_union']}/{summary['n_evaluable_union']}), "
              f"overlapping several events {summary['n_multi_event_segments']}, Top-1 in P4 {summary['n_top1_p4']}, "
              f"events without subsystem labels {summary['n_events_without_subsystem']}; "
              f"per event: first alarm segment {summary['n_events_first_hit']}/{summary['n_events_first_evaluable']}, "
              f"merged {summary['n_events_merged_hit']}/{summary['n_events_first_evaluable']}")

        # Exploratory analysis (post hoc): normalize by the val_thr mean error
        if attr["exploratory_val_norm"] and (run.directory / CONTRIB_FILE).exists():
            n_rows, n_summary, n_events = analyse(run, val_thr_baseline(run.directory, run.features))
            duration_rows += [{"model": name, "variant": "valnorm", **d} for d in by_duration(n_rows, bins)]
            changed = sum(a["top1_sensor"] != b["top1_sensor"] for a, b in zip(rows, n_rows) if a["category"] == "evaluable")
            for level, n_key, hit_key, rate in (("alarm segment", "n_evaluable", "n_hit", "subsystem_hit_rate"),
                                                ("event (first alarm segment)", "n_events_first_evaluable", "n_events_first_hit", "event_first_hit_rate"),
                                                ("event (merged)", "n_events_first_evaluable", "n_events_merged_hit", "event_hit_rate")):
                explore_rows.append({
                    "model": name, "level": level, "n": summary[n_key],
                    "n_hit_main": summary[hit_key], "rate_main": summary[rate],
                    "n_hit_valnorm": n_summary[hit_key], "rate_valnorm": n_summary[rate],
                    "rate_valnorm_ci_low": n_summary[f"{rate}_ci_low"], "rate_valnorm_ci_high": n_summary[f"{rate}_ci_high"],
                    "n_top1_p4_main": summary["n_top1_p4"], "n_top1_p4_valnorm": n_summary["n_top1_p4"],
                    "n_segments_top1_changed": changed})
            print(f"[EXPLORATORY] {name}: alarm-segment hits after val_thr normalization {n_summary['n_hit']}/{n_summary['n_evaluable']}"
                  f" (main result {summary['n_hit']}/{summary['n_evaluable']}); post hoc, does not replace the main results")

    # Multiple random seeds: compute once per seed run (official mode; skipped if seeds are incomplete)
    seed_df = None
    study = ml["seed_study"]
    if not args.smoke and study["model"] in models:
        names = {seed: study["model"] if seed == ml["seed"] else f"{study['run_subdir']}/{study['model']}_s{seed}"
                 for seed in study["seeds"]}
        usable = src.usable_runs(list(names.values()), required=("scores_test.npy", "per_sensor_err_test.npy",
                                                                  "threshold.json", "test_index.json"))
        if len(usable) == len(names):
            seed_rows = []
            for seed, name in names.items():
                _, summary, _ = analyse(load_run(src.root / name, per_sensor=True))
                seed_rows.append({"seed": seed, "run_name": name,
                                  **{k: summary[k] for k in ("n_alarm_segments", "n_evaluable", "n_hit", "subsystem_hit_rate",
                                                             "n_events_first_evaluable", "n_events_first_hit",
                                                             "event_first_hit_rate", "n_events_merged_hit", "event_hit_rate",
                                                             "n_top1_p4", "top1_p4_ratio")}})
            seed_df = pd.DataFrame(seed_rows)
            numeric = seed_df.drop(columns=["seed", "run_name"]).astype(float)
            seed_df = pd.concat([seed_df.astype({"seed": object}),
                                 pd.DataFrame([{"seed": "mean", "run_name": "", **numeric.mean().to_dict()},
                                               {"seed": "std", "run_name": "", **numeric.std(ddof=1).to_dict()}])],
                                ignore_index=True)
            seed_df.to_csv(src.out / "attribution_seeds.csv", index=False)
            print(f"[SEEDS] {study['model']} ({len(names)} seeds): "
                  + ", ".join(f"{c} {numeric[c].mean():.4f} ± {numeric[c].std(ddof=1):.4f}" for c in RATE_COLUMNS))
        else:
            print(f"[INFO] multi-seed runs are incomplete ({len(usable)}/{len(names)}), attribution_seeds.csv not generated")

    pd.DataFrame(all_rows).to_csv(src.out / "attribution.csv", index=False)
    sdf = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    sdf.to_csv(src.out / "attribution_summary.csv", index=False)
    ddf = pd.DataFrame(duration_rows)
    ddf.to_csv(src.out / "attribution_by_duration.csv", index=False)
    edf = pd.DataFrame(explore_rows)
    if len(edf):
        edf.to_csv(src.out / "attribution_exploratory_valnorm.csv", index=False)
    (src.out / "attribution_summary.md").write_text(summary_md(sdf, args.smoke, ddf, seed_df, edf), encoding="utf-8")
    print(f"[GENERATED] {src.out}/attribution.csv, attribution_summary.csv/.md, attribution_by_duration.csv, attribution_heatmap_*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
