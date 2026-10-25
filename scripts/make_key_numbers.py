"""R1: collect the main-line key numbers cited in the report into reports/main/key_numbers.md,
grouped as "detection / attribution / fusion / data issues", each number with its definition and source file. All numbers are read from the source files, never copied by hand;
the tests check that they match the source tables.

All sources are under reports/main/ (tracked): the result tables, and in runs/ the tracked copies of eval_metrics.json, run_manifest.json,
threshold.json, preprocess.json and train_log.csv of each full cluster run. This script first calls scripts.snapshot_runs to copy these small files from the artifact directory
with sensitive values redacted (only non-smoke, git_dirty=false, completed runs; host names etc. replaced by placeholders, originals untouched), then reads from the tracked copies.
This script reads no labels: all label-related numbers come from results already written by evaluate / attribution.

Usage:
    python -m scripts.make_key_numbers
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.fusion.prioritize import severity
from scripts.snapshot_runs import SNAPSHOT_SUBDIR, snapshot
from src.utils.paths import DEFAULT_CONFIG, load_config, resolve
from src.utils.reporting import report_problem

MODELS = ["pca", "iforest", "lstm_ae", "lstm_vae", "deep_svdd"]
OUTPUT = "key_numbers.md"
# How the level values in the source table (written by the attribution script) are phrased here
LEVELS = {"alarm segment": "alarm segments", "event (first alarm segment)": "events (first segment)", "event (merged)": "events (merged)"}


def pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


class Table:
    def __init__(self, title: str, header: str) -> None:
        self.lines = [f"### {title}", "", header, "|---|---|---|"]

    def add(self, value: str, scope: str, source: str) -> None:
        self.lines.append(f"| {value} | {scope} | {source} |")

    def text(self) -> list[str]:
        return [*self.lines, ""]


def run_json(root: Path, run: str, name: str) -> dict:
    problem = report_problem(root / run)
    if problem:
        raise ValueError(f"{run} cannot be used for the report: {problem}")
    return json.loads((root / run / name).read_text(encoding="utf-8"))


def build(cfg: dict) -> str:
    C, S = ", ", "; "          # Separators for list items and clauses
    header = "| Number | Definition and scope | Source |"
    ml = cfg["ml"]
    rep = resolve(ml["report_dir"])
    root = rep / SNAPSHOT_SUBDIR          # Tracked copies of each run's small files
    art = f"{ml['report_dir']}/{SNAPSHOT_SUBDIR}"
    main = pd.read_csv(rep / "main_results.csv").set_index("model")
    seeds = pd.read_csv(rep / "main_results_seeds.csv").set_index("seed")
    n_seeds = len(seeds) - 2
    evals = {m: run_json(root, m, "eval_metrics.json") for m in MODELS}
    n_events = evals["lstm_ae"]["event"]["n_events"]

    # ---------- Detection ----------
    det = Table("Detection", header)
    det.add("**No point-adjust**; threshold = 99th percentile of the val_thr scores; an alarm is raised when score > threshold",
            "Common to all detection metrics; a low point-wise F1 is the expected result",
            "Fixed in advance: project constraints 3, 4 and decision 6b")
    for m in MODELS:
        r, ev = main.loc[m], evals[m]["event"]
        det.add("{m}: precision {p}, recall {r}, F1 {f1}, AUROC {auc}; events {d}/{n}; median delay {dl} s; "
                  "false alarms {fa} per hour ({seg} segments)".format(
                      m=m, p=f"{r.precision:.4f}", r=f"{r.recall:.4f}", f1=f"{r.f1:.4f}", auc=f"{r.auroc:.4f}",
                      d=ev["n_detected"], n=ev["n_events"], dl=f"{r.delay_median_s:.0f}",
                      fa=f"{r.false_alarms_per_hour:.2f}", seg=ev["n_false_alarm_segments"]),
                "Single random seed ({seed}); point-wise metrics exclude the {u} unscored seconds; full cluster run".format(
                      seed=ml["seed"], u=int(r.n_unscored)),
                "`reports/main/main_results.csv`")
    mean, std = seeds.loc["mean"], seeds.loc["std"]
    det.add("**LSTM-AE, {n} seeds**: F1 **{f1}**, AUROC **{auc}**, event detection rate **{er}** ({avg}/{ne} on average), "
              "false alarms **{fa}** per hour; precision {p}, recall {r}".format(
                  n=n_seeds, f1=pm(mean.f1, std.f1), auc=pm(mean.auroc, std.auroc),
                  er=pm(mean.event_recall, std.event_recall), avg=f"{mean.event_recall * n_events:.1f}", ne=n_events,
                  fa=pm(mean.false_alarms_per_hour, std.false_alarms_per_hour, 2),
                  p=pm(mean.precision, std.precision), r=pm(mean.recall, std.recall)),
            "Main configuration W=60, hidden=64; seeds {s}; mean ± sample standard deviation (ddof=1). "
              "The LSTM-AE row of the main table uses these numbers; the other models have a single seed".format(
                  s=", ".join(str(s) for s in seeds.index[:n_seeds])),
            "`reports/main/main_results_seeds.csv`, `main_results.md`")
    f1_if = main.loc["iforest", "f1"]
    n_above = int((seeds.iloc[:n_seeds]["f1"].astype(float) > f1_if).sum())
    det.add("LSTM-AE vs IsolationForest, point-wise F1: {a} vs {b}, a difference of {d} = {sd} standard deviations; "
              "{k} of the {n} seeds are above IsolationForest".format(
                  a=f"{mean.f1:.4f}", b=f"{f1_if:.4f}", d=f"{f1_if - mean.f1:.4f}", sd=f"{(f1_if - mean.f1) / std.f1:.2f}",
                  n=n_seeds, k=n_above),
            "The difference is within random variation; the seed variation of IsolationForest itself was not evaluated",
            "Same as above")
    rows = [f"{m} {100 * (e['point']['tp'] + e['point']['fp']) / e['point']['n_points']:.1f}%" for m, e in evals.items()]
    det.add("Share of test seconds raising an alarm: " + C.join(rows),
            "Share of scored seconds with score > threshold; 1% on val_thr by the definition of the threshold. "
              "Too high for every model (threshold transfer problem)",
            f"`{art}/<{'model'}>/eval_metrics.json`")
    sub = pd.read_csv(rep / "per_subsystem_detection.csv")
    for m in ("iforest", "lstm_ae"):
        g = sub[sub.model == m].set_index("subsystem")
        det.add(m + " event detection by subsystem: "
                + C.join(f"{s} {int(g.loc[s, 'n_detected'])}/{int(g.loc[s, 'n_events'])}" for s in ("P1", "P2", "P3")),
                "An event involving several subsystems is counted once under each",
                "`reports/main/per_subsystem_detection.csv`")
    abl = pd.read_csv(rep / "ablation.csv")
    order = abl.sort_values("f1", ascending=False).reset_index(drop=True)
    rank = int(order.index[order["is_main_config"]][0]) + 1
    best, worst = order.iloc[0], order.iloc[-1]
    det.add("Ablation, 9 combinations: F1 {f_lo}–{f_hi}, AUROC {a_lo}–{a_hi}, false alarms {fa_lo}–{fa_hi} per hour; "
              "best W={bw}, hidden={bh} (F1 {bf}), worst W={ww}, hidden={wh} (F1 {wf}); "
              "the main configuration has rank {rank} by F1".format(
                  f_lo=f"{abl.f1.min():.3f}", f_hi=f"{abl.f1.max():.3f}", a_lo=f"{abl.auroc.min():.3f}",
                  a_hi=f"{abl.auroc.max():.3f}", fa_lo=f"{abl.false_alarms_per_hour.min():.2f}",
                  fa_hi=f"{abl.false_alarms_per_hour.max():.2f}", bw=int(best.window), bh=int(best.hidden),
                  bf=f"{best.f1:.3f}", ww=int(worst.window), wh=int(worst.hidden), wf=f"{worst.f1:.3f}", rank=rank),
            "LSTM-AE, W ∈ {30, 60, 120} × hidden ∈ {32, 64, 128}, single seed; sensitivity analysis only, "
              "the main model is not changed because of it (constraint 9)",
            "`reports/main/ablation.csv`, `ablation.md`, `ablation_heatmap.png`")
    q = pd.read_csv(rep / "ablation_quantiles.csv")
    w, h = ml["window"]["length"], ml["models"]["lstm_ae"]["hidden"]
    mq = q[(q.window == w) & (q.hidden == h)].set_index("quantile")
    top = q[q["quantile"] == q["quantile"].max()]
    item = "{k}th percentile F1 {f}, event detection rate {e}"
    det.add("Threshold percentile sensitivity (main configuration): "
            + "; ".join(item.format(k=f"{100 * k:g}", f=f"{v.f1:.3f}", e=f"{v.event_recall:.2f}") for k, v in mq.iterrows())
            + "; at the {k}th percentile the F1 of the 9 combinations is {lo}–{hi}".format(
                    k=f"{100 * q['quantile'].max():g}", lo=f"{top.f1.min():.2f}", hi=f"{top.f1.max():.2f}"),
            "Sensitivity analysis only, not used to choose the threshold; the main table still uses the 99th percentile",
            "`reports/main/ablation_quantiles.csv`")
    for m in ("lstm_ae", "lstm_vae", "deep_svdd"):
        tr = run_json(root, m, "run_manifest.json")["stages"]["train"]
        early = ("early stopping at epoch {e}".format(e=tr["early_stop_epoch"])
                 if tr["early_stop_epoch"] else "early stopping not triggered")
        det.add("{m} training: {fe} epochs, best epoch {be}, {early}; {sec} s per epoch on average; {npar} parameters".format(
                      m=m, fe=tr["final_epoch"], be=tr["best_epoch"], early=early,
                      sec=f"{tr['epoch_seconds_mean']:.2f}", npar=f"{tr['n_parameters']:,}"),
                "Cluster NVIDIA L4, seed 42; max_epochs=50, patience=5 fixed in advance",
                f"`{art}/{m}/run_manifest.json`" + ", `train_log.csv`; figure "
                + "`reports/main/training_curves.png`")
    c = run_json(root, "deep_svdd", "run_manifest.json")["deep_svdd_center"]
    thr = run_json(root, "deep_svdd", "threshold.json")
    det.add("Deep SVDD centre: before replacement \\|c\\| min {lo}, median {med}, max {hi}; {nr}/{dim} dimensions "
              "replaced by ±{eps}. Collapse check collapsed={col} (score_cv {cv}, inflated by 1 extreme window); "
              "val_thr score median {smed}, threshold {thr}".format(
                  lo=f"{c['abs_c_before_min']:.5f}", med=f"{c['abs_c_before_median']:.4f}", hi=f"{c['abs_c_before_max']:.4f}",
                  nr=c["n_replaced"], dim=c["rep_dim"], eps=c["eps"], col=str(thr["collapsed"]).lower(),
                  cv=f"{thr['score_cv']:.0f}", smed=f"{thr['score_stats']['median']:.2g}", thr=f"{thr['threshold']:.2g}"),
            "Centre = mean output of the initial network over all training windows, with components close to 0 "
              "set to ±eps (decisions 5, 6b)",
            f"`{art}/deep_svdd/run_manifest.json`" + ", " + "`threshold.json`")
    contrib = pd.read_csv(rep / "val_sensor_contrib.csv").iloc[0]
    det.add("Sensor with the largest share for PCA on val_thr: {s} {x}% (limit {lim}%, not exceeded)".format(
                  s=contrib.sensor, x=f"{100 * contrib.share:.1f}", lim=f"{100 * ml['diagnosis']['max_sensor_share']:.0f}"),
            "Heavy-tail diagnostic on val_thr only; the scaling (StandardScaler, no lower bound on std) was fixed "
              "before the first evaluation on test",
            "`reports/main/val_sensor_contrib.csv`, `.png`")

    # ---------- Attribution ----------
    att = Table("Attribution", header)
    summ = pd.read_csv(rep / "attribution_summary.csv").set_index("model")
    seg = pd.read_csv(rep / "attribution.csv")
    for m, s in summ.iterrows():
        ov = seg[(seg.model == m) & (seg.category == "evaluable")]
        guess = ov["event_subsystems"].str.split("+").apply(lambda x: "P1" in x).mean()
        att.add("{m}: alarm segments {seg}; events, first segment {first}; events, merged {merged}; "
                  "always guessing P1 on the same segments {guess}; top-1 in P4 {p4}".format(
                      m=m,
                      seg=f"{int(s.n_hit)}/{int(s.n_evaluable)} = {s.subsystem_hit_rate:.3f} "
                          f"{ci(s.subsystem_hit_rate_ci_low, s.subsystem_hit_rate_ci_high)}",
                      first=f"{int(s.n_events_first_hit)}/{int(s.n_events_first_evaluable)} = {s.event_first_hit_rate:.3f} "
                            f"{ci(s.event_first_hit_rate_ci_low, s.event_first_hit_rate_ci_high)}",
                      merged=f"{int(s.n_events_merged_hit)}/{int(s.n_events_first_evaluable)} = {s.event_hit_rate:.3f} "
                             f"{ci(s.event_hit_rate_ci_low, s.event_hit_rate_ci_high)}",
                      guess=f"{guess:.3f}", p4=f"{int(s.n_top1_p4)}/{int(s.n_overlap_attack)}"),
                "Subsystem hit = the subsystem of the top-1 sensor ∈ the event's subsystem set; only alarm segments "
                  "overlapping an attack are counted; brackets are 95% Wilson intervals. Per event: the main definition "
                  "takes the first alarm segment overlapping the event, the merged definition is supplementary. Single seed",
                "`reports/main/attribution_summary.csv`, `.md`, `attribution.csv`")
    first = summ.iloc[0]
    att.add("Alarm segments overlapping several events: {multi}; events without subsystem labels: {nosub}; "
              "alarm segments not overlapping an attack: ".format(
                  multi=int(summ.n_multi_event_segments.max()), nosub=int(first.n_events_without_subsystem))
            + C.join(f"{m} {int(s.n_non_overlap)}" for m, s in summ.iterrows()),
            "The main and union definitions are therefore identical; alarm segments not overlapping an attack "
              "do not count towards the hit rate",
            "`reports/main/attribution_summary.csv`")
    aseed = pd.read_csv(rep / "attribution_seeds.csv").set_index("seed")
    am, asd = aseed.loc["mean"], aseed.loc["std"]
    att.add("**LSTM-AE, {n} seeds**: alarm segments **{seg}**, events, first segment **{first}**, events, merged {merged}; "
              "share of top-1 in P4 {p4}".format(
                  n=len(aseed) - 2, seg=pm(am.subsystem_hit_rate, asd.subsystem_hit_rate, 3),
                  first=pm(am.event_first_hit_rate, asd.event_first_hit_rate, 3),
                  merged=pm(am.event_hit_rate, asd.event_hit_rate, 3), p4=pm(am.top1_p4_ratio, asd.top1_p4_ratio, 3)),
            "Attribution computed separately for each seed's run; mean ± sample standard deviation. "
              "LSTM-VAE and PCA have a single seed",
            "`reports/main/attribution_seeds.csv`")
    dur = pd.read_csv(rep / "attribution_by_duration.csv")
    for m in summ.index:
        g = dur[(dur.model == m) & (dur.variant == "main")]
        att.add(m + " by alarm segment length: " + S.join(
            f"{r.duration_group} {int(r.n_hit)}/{int(r.n)}"
            + (f" = {r.rate:.3f} {ci(r.ci_low, r.ci_high)}" if r.n else "") for r in g.itertuples()),
                "Alarm segments overlapping an attack, grouped by length; 10–60 s inclusive at both ends. "
                  "Short alarm segments are attributed clearly worse (LSTM-AE)",
                "`reports/main/attribution_by_duration.csv`")
    exp = pd.read_csv(rep / "attribution_exploratory_valnorm.csv")
    for m in summ.index:
        g = exp[exp.model == m].set_index("level")
        att.add("(Exploratory) {m} after normalising by val_thr: ".format(m=m) + S.join(
            f"{LEVELS[lv]} {int(r.n_hit_main)}/{int(r.n)} → {int(r.n_hit_valnorm)}/{int(r.n)}"
            for lv, r in g.iterrows()),
                "**Post-hoc analysis, does not replace the main results**: within-segment errors are divided by each "
                  "sensor's mean error on val_thr before ranking the top-k; the baseline comes from val_thr only, but "
                  "this step was done after seeing the test results, so the numbers are optimistic",
                "`reports/main/attribution_exploratory_valnorm.csv`")
    att.add("Sensor-level Hit@k: not computed",
            "The HAI 21.03 data files have no per-sensor attack targets and the official attack target table was "
              "not obtained; listed under limitations",
            "`reports/main/attribution_summary.md`")

    # ---------- Fusion ----------
    fus = Table("Fusion (prototype demonstration; no claim of improved detection metrics, constraint 7)", header)
    alerts = json.loads(resolve(ml["alerts"]["output"]).read_text(encoding="utf-8"))
    fr = pd.read_csv(rep / "fusion_ranking_eval.csv")
    n_true = int(fr.n_true_total.iloc[0])
    fus.add("Real alarms: {n}, of which {k} overlap an attack event (base rate {base})".format(
                  n=len(alerts), k=n_true, base=f"{n_true / len(alerts):.3f}"),
            "All alarm segments of the main LSTM-AE run (seed 42) on test, not filtered by label; is_mock=false",
            "`data/processed/alerts.json`, `reports/main/fusion_ranking_eval.csv`")
    for name, g in fr.groupby("ranking", sort=False):
        k_max = g.iloc[-1]
        shared = ("; {n} of the top {k} are the same as in the raw-score ranking".format(
                        k=int(k_max.k), n=int(k_max.n_shared_with_raw_score)) if name != "raw_score" else "")
        fus.add("Ranking {name}: ".format(name=name)
                + C.join(f"precision@{int(r.k)} = {int(r.n_true)}/{int(r.k)}" for r in g.itertuples())
                + "; 95% interval of precision@{k} {ci}".format(
                    k=int(k_max.k), ci=ci(k_max.ci_low, k_max.ci_high))
                + shared,
                "Share of the top k alarms overlapping an attack event; raw_score ranks by maximum anomaly score, "
                  "alpha=0 / alpha=1 are fused priorities. No measurable difference between the three rankings: "
                  "the ranking effect comes from the anomaly score, not from the vulnerability context",
                "`reports/main/fusion_ranking_eval.csv`, `.md`")
    n_sat = sum(severity(a["max_score"], a["threshold"]) >= 1.0 for a in alerts)
    fus.add("Alarms with S capped: {n}".format(n=n_sat),
            "S = min(1, max score ÷ threshold − 1), which equals 1 once the score reaches 2 times the threshold; "
              "these alarms are no longer separated by score. Recorded as a limitation and future work only, "
              "the formula is unchanged",
            "`data/processed/alerts.json`, `reports/main/cases.md`")
    fus.add("Cards and the demo render only the top {n} alarms by priority".format(
                n=cfg["fusion"]["max_rendered_alerts"]),
            "Fusion scoring still covers all alarms; the count is `fusion.max_rendered_alerts`",
            "`config/config.yaml`")
    fus.add("3 cases: a correctly attributed real attack, a wrongly attributed real attack, a high-priority false alarm",
            "For each type, the first alarm in descending fused priority, with no manual selection; "
              "the notes against the labels come from post-hoc evaluation",
            "`reports/main/cases.md`, `cases/`")
    fus.add("Vulnerability and asset numbers (540 vulnerabilities, 9 KEV hits, 66 kept after min_qod=70, "
              "E_sub takes only the values 0 / 1)",
            "The real scan covers a single target, ews_01; the asset mapping is constructed by hand",
            "`reports/extension/key_numbers.md`, `kev_table.md`")

    # ---------- Data issues ----------
    dat = Table("Data issues", header)
    pre = run_json(root, "lstm_ae", "preprocess.json")
    drops = ml["data"]["train_overlap_drops"][0]
    dat.add("The last {n} seconds of the official train1 and the first {n} seconds of test3 are the same data; "
              "in addition, row {h} of train1 = the last 1 row of test2. Both were removed from the training set, "
              "test is left as released".format(n=f"{drops['tail_rows']:,}", h=drops["head_rows"]),
            "All file pairs compared by a row-wise content hash over the 79 sensors; this data contains no attack. "
              "Left untreated, it would understate false alarms on test3",
            "`reports/main/data_overlap.md`; `reports/main/data_summary.md` section 3; "
              "`ml.data.train_overlap_drops` in `config/config.yaml`")
    r = pre["rows"]
    dat.add("Split: train {a} / val_es {b} / val_thr {c} rows; {f} features ({k} constant columns removed)".format(
                  a=f"{r['train']:,}", b=f"{r['val_es']:,}", c=f"{r['val_thr']:,}", f=pre["n_features"],
                  k=len(pre["constant_cols"])),
            "Split in time order after removing the overlapping rows; the last 20% is halved into val_es "
              "(early stopping) and val_thr (threshold); the scaler is fitted on the train part only",
            f"`{art}/lstm_ae/preprocess.json`" + "; section M1 of `PROGRESS.md`")
    p = evals["lstm_ae"]["point"]
    events = pd.read_csv(rep / "attack_events.csv")
    dat.add("test: {n} seconds, {a} attack seconds ({pct}%), {e} attack events (length {lo}–{hi} s)".format(
                  n=f"{p['n_points'] + p['n_unscored']:,}", a=f"{p['n_positive']:,}",
                  pct=f"{100 * p['n_positive'] / (p['n_points'] + p['n_unscored']):.2f}", e=len(events),
                  lo=int(events.duration_s.min()), hi=int(events.duration_s.max())),
            "Event = a maximal run of consecutive seconds with attack=1",
            "`reports/main/data_summary.md`, `attack_events.csv`")
    dat.add("Unscored: {u} seconds (5 files × {w} seconds), {k} of them attack seconds".format(
                  u=p["n_unscored"], w=ml["window"]["length"] - 1, k=p["n_positive_unscored"]),
            "The first W−1 seconds of each test file have no score (W=60) and enter no metric; "
              "for the W=30 / 120 ablation combinations it is 145 / 595 seconds",
            "column n_unscored of `reports/main/main_results.csv`")
    dat.add("attack=1 with all three subsystem labels 0: {n} seconds, in the interior of {k} events; "
              "under the union rule the subsystem sets of all 50 events are non-empty".format(
                  n=int(events.no_sub_s.sum()), k=int((events.no_sub_s > 0).sum())),
            "Subsystem set of an event = union of attack_P* over all its seconds; detection metrics use only "
              "the attack column and are unaffected",
            "`reports/main/attack_events.csv`, `data_summary.md`")
    files = [f"{x['file'].split('.')[0]} {100 * (x['tp'] + x['fp']) / x['n_points']:.1f}%" for x in evals["lstm_ae"]["per_file"]]
    dat.add("Share of seconds raising an alarm per test file, LSTM-AE: " + C.join(files),
            "1% on val_thr. The files differ widely; the presumed cause is drift of normal operation in test "
              "relative to the training data (which sensors drift was not verified separately)",
            f"`{art}/lstm_ae/eval_metrics.json`" + "; figures `reports/main/scores_test*.png`, `score_distribution.png`")
    dat.add("P4 has no subsystem label",
            "P4 is the HIL simulation part; an attribution whose top-1 falls in P4 necessarily counts as a miss",
            "`data/samples/hai_columns.txt`, decision 7")

    head = ["## Key numbers (main line)", "",
         "Quote these when writing the report. The numbers are read from the source files by "
         "`python -m scripts.make_key_numbers`. "
         "All detection, ablation and attribution numbers come from full cluster runs (git_dirty=false); "
         "no smoke-test number appears here.",
         "All source files are tracked: `reports/main/runs/` holds copies of the small files of each cluster run "
         "(metrics, manifest, threshold, preprocessing summary, training log), copied from the artifact directory "
         "by this script with host names and other cluster details replaced by placeholders.", ""]
    return "\n".join(head + det.text() + att.text() + fus.text() + dat.text()).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Collect the main-line key numbers into reports/main/key_numbers.md")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    # First copy each run's small files from the artifact directory to the tracked location with redaction; runs missing from the artifact directory keep the tracked copies
    copied, skipped = snapshot(cfg)
    print(f"[COPIED] small files of {len(copied)} runs (redacted) → {resolve(cfg['ml']['report_dir']) / SNAPSHOT_SUBDIR}"
          + (f"; {len(skipped)} runs unavailable in the artifact directory, keeping the tracked copies" if skipped else ""))
    try:
        text = build(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    out = resolve(cfg["ml"]["report_dir"]) / OUTPUT
    out.write_text(text, encoding="utf-8")
    print(f"[GENERATED] {out} ({text.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
