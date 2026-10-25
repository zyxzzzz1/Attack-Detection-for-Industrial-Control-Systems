"""I1: pick 3 cases from the real alerts by fixed rules, write reports/main/cases.md, and save the corresponding alert cards to reports/main/cases/.

Selection rules (walk the fusion priority ranking from the top and take the first match of each; no manual picking):
1. Real attack, correct attribution: overlaps an attack event and the Top-1 sensor's subsystem is in the event's subsystem set.
2. Real attack, wrong attribution: overlaps an attack event and the Top-1 sensor's subsystem is not in the event's subsystem set.
3. High-priority false alarm: overlaps no attack event.

Inputs: fusion.enriched_alerts_json (run python -m src.pipeline first), reports/main/attribution.csv (python -m src.attribution),
eval_events.csv of the main-model run. "Real attack or not, attribution hit or not" come from the attribution evaluation and evaluate outputs; this script reads no labels;
the alert cards themselves do not contain this information.

Usage:
    python -m scripts.make_cases
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.eval.alarms import subsystem_of
from src.fusion.render_card import card_context, make_env
from src.utils.paths import DEFAULT_CONFIG, artifact_dir, load_config, resolve
from src.utils.reporting import report_problem

CASES = [("a real attack with correct attribution", lambda r: r.category == "evaluable" and r.hit is True),
         ("a real attack with wrong attribution", lambda r: r.category == "evaluable" and r.hit is False),
         ("a high-priority false alarm", lambda r: r.category == "non_overlap")]


def load_table(cfg: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """One row per alert: fusion rank + attribution evaluation result. The two sides correspond one-to-one in alarm-segment order; a start/end mismatch raises an error."""
    ml, fcfg = cfg["ml"], cfg["fusion"]
    model = ml["alerts"]["model"]
    run_dir = artifact_dir(cfg) / model
    problem = report_problem(run_dir)
    if problem:
        raise ValueError(f"the run of {model} cannot be used for the report: {problem}")
    enriched = json.loads(resolve(fcfg["enriched_alerts_json"]).read_text(encoding="utf-8"))
    if enriched["is_mock"]:
        raise ValueError("enriched_alerts.json is the output for the mock alerts: run python -m src.pipeline (without --mock) first")
    alerts = json.loads(resolve(ml["alerts"]["output"]).read_text(encoding="utf-8"))
    attr = pd.read_csv(resolve(ml["report_dir"]) / "attribution.csv")
    attr = attr[attr["model"] == model].reset_index(drop=True)
    if len(attr) != len(alerts) or any((a["start"], a["end"]) != (r.start_time, r.end_time)
                                       for a, r in zip(alerts, attr.itertuples())):
        raise ValueError("alerts.json and attribution.csv are not the same set of alarm segments: rerun export_alerts and attribution")
    attr["alert_id"] = [a["alert_id"] for a in alerts]
    attr["hit"] = attr["hit"].map({True: True, False: False, "True": True, "False": False}).astype(object)
    attr["hit"] = attr["hit"].where(attr["hit"].notna(), None)
    rank = {a["alert_id"]: a["rank"] for a in enriched["alerts"]}
    attr["rank"] = attr["alert_id"].map(rank)
    return attr.sort_values("rank").reset_index(drop=True), enriched, pd.read_csv(run_dir / "eval_events.csv")


def describe(title: str, row, alert: dict, events: pd.DataFrame, n_total: int, card_file: str) -> list[str]:
    ratio = alert["max_score"] / alert["threshold"]
    top = alert["topk"]
    sep = ", "
    if alert["vulns"]:
        vuln_line = "- Assets on the path that have not been assessed: {assets}; {n} vulnerabilities listed (the top one is {name} on {asset}), {m} needing manual verification".format(
            assets=sep.join(alert["unassessed_assets"]) or "none", n=len(alert["vulns"]),
            asset=alert["vulns"][0]["asset"], name=alert["vulns"][0]["name"], m=len(alert["needs_review"]))
    else:
        vuln_line = "- No vulnerabilities on the path are included in the calculation"
    lines = [f"## {title}: {alert['alert_id']}", "",
             "- Fusion priority rank: {rank} / {n} (priority {p}, S {s}, E_sub {e})".format(
                 rank=alert["rank"], n=n_total, p=f"{alert['priority']:.3f}", s=f"{alert['S']:.3f}", e=f"{alert['E_sub']:.3f}"),
             "- Alert time: {start} ~ {end} ({d} s, {file})".format(
                 start=alert["start"], end=alert["end"], d=alert["duration_s"], file=row.file.split(".")[0]),
             "- Max anomaly score {score}, i.e. {ratio} times the threshold ({thr})".format(
                 score=f"{alert['max_score']:.4g}", thr=f"{alert['threshold']:.4f}", ratio=f"{ratio:.1f}"),
             "- Top-5 sensors (share of total sensor error): " + sep.join(f"{x['sensor']} {x['share']:.1%}" for x in top),
             "- Subsystems inferred by the card: " + sep.join(
                 "{id} {w} (criticality {c}, E_sub {e})".format(id=x["id"], w=f"{x['weight']:.0%}", c=x["criticality"], e=f"{x['E_sub']:.0f}")
                 for x in alert["subsystems"]),
             "- Intrusion paths given by the card: " + "; ".join(" → ".join(x["path"]) for x in alert["path"]),
             vuln_line,
             "- Alert card: " + f"[`cases/{card_file}`](cases/{card_file})", ""]
    top1_sub = subsystem_of(top[0]["sensor"])
    if row.category == "evaluable":
        ev = events[events["event_id"] == int(row.primary_event_id)].iloc[0]
        verdict = ("which **is** in the event's subsystem set: the attribution is a hit." if row.hit
                   else "which is **not** in the event's subsystem set: by the rule of Decision 7 the attribution is a miss.")
        lines += ["**Against the labels**: this alert overlaps attack event #{eid} ({start} ~ {end}, {d} s, attacked subsystem according to the labels {subs}; detection delay of this event {delay} s). The Top-1 sensor {sensor} belongs to {sub}, ".format(
                      eid=int(ev.event_id), start=ev.start_time, end=ev.end_time, d=int(ev.duration_s), subs=ev.subsystems,
                      delay=int(ev.delay_s), sensor=top[0]["sensor"], sub=top1_sub) + verdict, ""]
    else:
        same = events[events["file"] == row.file]
        before = same[pd.to_datetime(same["end_time"]) < pd.Timestamp(alert["start"])]
        after = same[pd.to_datetime(same["start_time"]) > pd.Timestamp(alert["end"])]
        gaps = []
        if len(before):
            b = before.iloc[-1]
            minutes = (pd.Timestamp(alert["start"]) - pd.Timestamp(b.end_time)).total_seconds() / 60
            gaps.append("the previous attack event #{eid} ended at {time} ({m} minutes earlier)".format(eid=int(b.event_id), time=b.end_time, m=f"{minutes:.0f}"))
        if len(after):
            n = after.iloc[0]
            minutes = (pd.Timestamp(n.start_time) - pd.Timestamp(alert["end"])).total_seconds() / 60
            gaps.append("the next attack event #{eid} started at {time} ({m} minutes later)".format(eid=int(n.event_id), time=n.start_time, m=f"{minutes:.0f}"))
        lines += ["**Against the labels**: this alert does not overlap any attack event; it is a false alarm."
                  + (" In the same file, {gaps}.".format(gaps=", ".join(gaps)) if gaps else ""), ""]
    return lines


def observations(picked: dict) -> list[str]:
    """Observations visible when the three cases are read together; each is written only if the data actually supports it."""
    out = []
    if 2 in picked:
        row, alert = picked[2]
        weights = {s["id"]: s["weight"] for s in alert["subsystems"]}
        true_subs = [x for x in str(row.event_subsystems).split("+") if x]
        share = sum(weights.get(x, 0.0) for x in true_subs)
        if share >= 0.5:
            out.append('- In Case 2 the Top-1 sensor is not in the attacked subsystem, but most of the other Top-5 sensors are: after weighting by the Top-5 error shares the card still puts {share} of the weight on {subs}, and the intrusion paths it gives belong to that subsystem. In other words, "whether Top-1 is a hit" is stricter than the information the card actually shows; this holds for this one alert only and must not be generalised.'.format(
                             share=f"{share:.0%}", subs="+".join(true_subs)))
        if "P4" in weights:
            out.append("- In Case 2 the Top-1 sensor falls in P4 (the HIL simulation part, which has no subsystem label, so by the rule it always counts as a miss). P4 is physically coupled with P1, P2 and P3, so pointing to P4 is not necessarily wrong; the available labels cannot tell. The card carries a note on P4 and recommends investigating P1 / P2 / P3 as well.")
    if 2 in picked and 3 in picked:
        (_, real), (_, false) = picked[2], picked[3]
        r_real, r_false = real["max_score"] / real["threshold"], false["max_score"] / false["threshold"]
        if false["rank"] < real["rank"] and r_false < r_real:
            out.append("- The false alarm of Case 3 (score {f} times the threshold) is ranked ahead of the real attack of Case 2 ({r} times). The reason is that the anomaly severity S = min(1, score ÷ threshold − 1) is capped at 1 as soon as the score reaches 2 times the threshold, so both alerts have the same S and the ranking is decided only by subsystem weights, criticality and E_sub. The cap on S is too low; this is a flaw of the scoring formula.".format(
                             f=f"{r_false:.1f}", r=f"{r_real:.1f}"))
    tops = {a["vulns"][0]["nvt_oid"] for _, a in picked.values() if a["vulns"]}
    assets = {v["asset"] for _, a in picked.values() for v in a["vulns"]}
    if len(picked) > 1 and len(tops) == 1 and len(assets) == 1:
        out.append("- The vulnerabilities listed on all three cards come from the same scanned asset ({asset}), and the top entry is the same one: only a single target machine has been assessed, so every alert that involves P1 or P2 receives almost the same vulnerability context, with no difference between real attacks and false alarms. This is a limitation caused by the single target machine and by risk saturation (see the M9 section of PROGRESS.md).".format(asset=next(iter(assets))))
    if 3 in picked:
        out.append("- The card of the false alarm looks exactly like the cards of the real attacks (the same intrusion paths, vulnerabilities and remediation advice). A card does not help decide whether an alert is a false alarm; it only gives a starting point for investigation on the assumption that the alert is real.")
    return ([f"## Taken together", "", *out, ""] if out else [])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pick 3 cases and write reports/main/cases.md")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    try:
        table, enriched, events = load_table(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    out = resolve(cfg["ml"]["report_dir"])
    cards = out / "cases"
    cards.mkdir(parents=True, exist_ok=True)
    for old in cards.glob("*.md"):
        old.unlink()
    env = make_env(resolve(cfg["fusion"]["templates_dir"]))
    by_id = {a["alert_id"]: a for a in enriched["alerts"]}
    n_total = len(enriched["alerts"])

    lines = [f"# Cases", "",
             "> " + "Asset mapping and vulnerability data come from a lab environment and are for demonstration only." + " "
             + "The asset mapping is constructed by hand and the vulnerability data comes from scanning a self-built target machine; neither has any causal relation to the attacks in HAI. The intrusion paths and vulnerabilities on the cards only show how the prototype brings the information together; they are not a conclusion about where these attacks came from.", "",
             'The alerts are all {n} alarm segments of the LSTM-AE main model run ({model}, seed {seed}) on the HAI 21.03 test set. The cases are selected by a fixed rule, not by hand: going down the fusion priority ranking, take the first real attack with correct attribution, the first real attack with wrong attribution, and the first false alarm. "Whether it is a real attack and whether the attribution is a hit" comes from an evaluation done afterwards with the labels; the cards do not contain this information.'.format(
                   model=cfg["ml"]["alerts"]["model"], seed=cfg["ml"]["seed"], n=n_total), ""]
    chosen, picked = [], {}
    for k, (title, rule) in enumerate(CASES, 1):
        row = next((r for r in table.itertuples() if rule(r)), None)
        if row is None:
            lines += [f"## {title}", "", "No alert meets the condition.", ""]
            continue
        alert = by_id[row.alert_id]
        card_file = f"case{k}_{alert['alert_id']}.md"
        (cards / card_file).write_text(env.get_template("alert_card.md.j2").render(**card_context(alert, enriched)),
                                       encoding="utf-8")
        lines += describe("Case {k} — ".format(k=k) + title, row, alert, events, n_total, card_file)
        chosen.append((title, alert["alert_id"], alert["rank"]))
        picked[k] = (row, alert)
    lines += observations(picked)
    (out / "cases.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    for title, aid, rank in chosen:
        print(f"[CASE] {title}: {aid} (rank {rank})")
    print(f"[GENERATED] {out / 'cases.md'}, {cards}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
