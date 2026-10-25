"""M0: per-file statistics of the HAI data, written to reports/main/data_summary.md and attack_events.csv.

Per file: row count, time range, continuity (missing / duplicated / out-of-order seconds), whether columns match data/samples/hai_columns.txt,
missing values, attack seconds and event count per label (event = contiguous run where the label is 1), and the subsystem labels of each attack event.
train files additionally get constant-column statistics. All files are compared pairwise by per-row content hash for shared data (leakage check).

Files are read one at a time, sensors as float32, released after the statistics; several files are never loaded at once (hashing reads the file again as float64 and keeps only 8 bytes per row).
test files get only descriptive statistics here (rows, times, label counts, row hashes); their sensor value distributions are not read, and nothing here is used for tuning (constraint 2).

Usage:
    python -m scripts.hai_data_summary [--config config/config.yaml] [--output-dir reports/main]
"""

import argparse
import gc
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import DEFAULT_CONFIG, hai_data_dir, load_config, resolve
from src.utils.rowhash import hash_rows, matched_runs

COL_LINE_RE = re.compile(r"^\s*\d+\s+\[[^\]]+\]\s+(\S+)\s*$")
EVENT_FIELDS = ["file", "event", "start", "end", "duration_s", "subsystems", "no_sub_s", "no_sub_runs",
                "no_sub_first_offset_s", "no_sub_last_offset_s", "no_sub_kind"]


def read_expected_columns(columns_txt: Path) -> list[str]:
    """Return the ordered column names from hai_columns.txt (ignoring # comment lines)."""
    cols = []
    for line in columns_txt.read_text(encoding="utf-8").splitlines():
        m = COL_LINE_RE.match(line)
        if m and not line.lstrip().startswith("#"):
            cols.append(m.group(1))
    return cols


def run_bounds(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return [start, end) of each contiguous run of 1s."""
    d = np.diff(np.concatenate(([0], (binary != 0).astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def run_lengths(binary: np.ndarray) -> np.ndarray:
    starts, ends = run_bounds(binary)
    return ends - starts


def attack_events(name: str, times: np.ndarray, main: np.ndarray, subs: np.ndarray, sub_names: list[str]) -> list[dict]:
    """Subsystem labels per attack event (contiguous run of the overall label): their union, and where the seconds without any subsystem label are.

    no_sub_kind: none = every second has a subsystem label; whole = the whole event has none;
    edge = unlabeled seconds touch the start or end of the event; interior = they occur only inside the event.
    """
    events = []
    for k, (s, e) in enumerate(zip(*run_bounds(main)), start=1):
        seg = subs[s:e] != 0
        no_sub = np.flatnonzero(~seg.any(axis=1))
        union = [n.removeprefix("attack_") for n, hit in zip(sub_names, seg.any(axis=0)) if hit]
        if len(no_sub) == 0:
            kind = "none"
        elif not union:
            kind = "whole"
        elif no_sub[0] == 0 or no_sub[-1] == e - s - 1:
            kind = "edge"
        else:
            kind = "interior"
        events.append({
            "file": name, "event": k, "start": str(times[s]), "end": str(times[e - 1]), "duration_s": int(e - s),
            "subsystems": "+".join(union), "no_sub_s": int(len(no_sub)),
            "no_sub_runs": int(len(run_lengths((~seg.any(axis=1)).astype(np.int8)))),
            "no_sub_first_offset_s": int(no_sub[0]) if len(no_sub) else "",
            "no_sub_last_offset_s": int(no_sub[-1]) if len(no_sub) else "",
            "no_sub_kind": kind,
        })
    return events


def summarize_file(path: Path, expected_cols: list[str], time_col: str, label_cols: list[str],
                   with_feature_stats: bool) -> dict:
    header = [c.strip() for c in pd.read_csv(path, nrows=0).columns]
    features = [c for c in header if c != time_col and c not in label_cols]
    dtypes = {c: np.float32 for c in features} | {c: np.int8 for c in label_cols if c in header}
    df = pd.read_csv(path, dtype=dtypes, skipinitialspace=True)
    df.columns = header

    t = pd.to_datetime(df[time_col])
    dt = t.diff().dt.total_seconds().to_numpy()[1:]
    gaps = np.flatnonzero(dt > 1)
    out = {
        "file": path.name, "bytes": path.stat().st_size, "rows": len(df),
        "start": str(t.iloc[0]), "end": str(t.iloc[-1]),
        "span_s": int((t.iloc[-1] - t.iloc[0]).total_seconds()) + 1,
        "duplicate_times": int(t.duplicated().sum()),
        "non_increasing_steps": int((dt <= 0).sum()),
        "gap_count": int(len(gaps)),
        "missing_seconds": int((dt[gaps] - 1).sum()),
        "gaps": [{"after": str(t.iloc[i]), "before": str(t.iloc[i + 1]), "missing_s": int(dt[i] - 1)}
                 for i in gaps],
        "n_cols": len(header), "n_features": len(features),
        "columns_match": header == expected_cols,
        "columns_missing": [c for c in expected_cols if c not in header],
        "columns_extra": [c for c in header if c not in expected_cols],
        "nan_cells": int(df[features].isna().sum().sum()),
        "labels": {}, "events": [],
    }
    for c in label_cols:
        if c not in header:
            out["labels"][c] = None
            continue
        v = df[c].to_numpy()
        runs = run_lengths(v)
        out["labels"][c] = {
            "values": sorted(int(x) for x in np.unique(v)),
            "attack_s": int((v != 0).sum()), "events": int(len(runs)),
            "min_s": int(runs.min()) if len(runs) else None,
            "median_s": float(np.median(runs)) if len(runs) else None,
            "max_s": int(runs.max()) if len(runs) else None,
        }
    main, subs = label_cols[0], [c for c in label_cols[1:] if c in header]
    if main in header and subs:
        m, sub_sum = df[main].to_numpy() != 0, (df[subs].to_numpy() != 0).sum(axis=1)
        out["label_consistency"] = {
            "main_without_sub_s": int((m & (sub_sum == 0)).sum()),
            "sub_without_main_s": int((~m & (sub_sum > 0)).sum()),
            "multi_sub_s": int((sub_sum > 1).sum()),
        }
        out["events"] = attack_events(path.name, t.astype(str).to_numpy(), df[main].to_numpy(),
                                      df[subs].to_numpy(), subs)
    if with_feature_stats:
        x = df[features]
        out["constant_cols"] = [c for c in features if x[c].min() == x[c].max()]
    del df, t
    gc.collect()
    return out


def file_row_hashes(path: Path, time_col: str, label_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-row content hashes, time strings). Hashes use only sensor columns, parsed as float64."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    features = [c for c in df.columns if c != time_col and c not in label_cols]
    out = hash_rows(df[features]), df[time_col].astype(str).to_numpy()
    del df
    gc.collect()
    return out


def find_content_overlaps(hashes: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[dict[str, int], list[dict]]:
    """Compare row hashes of all file pairs. Return (rows with duplicated content within each file, list of overlapping runs between files)."""
    internal = {n: int(len(h) - len(np.unique(h))) for n, (h, _) in hashes.items()}
    runs = []
    for a, b in combinations(hashes, 2):
        (ha, ta), (hb, tb) = hashes[a], hashes[b]
        for ia, ib, n in matched_runs(ha, hb):
            runs.append({"a": a, "b": b, "rows": n,
                         "a_start": ta[ia], "a_end": ta[ia + n - 1], "a_first_row": ia,
                         "b_start": tb[ib], "b_end": tb[ib + n - 1], "b_first_row": ib,
                         "a_rows_total": len(ha), "b_rows_total": len(hb),
                         "time_aligned": bool(ta[ia] == tb[ib])})
    return internal, runs


def _table(headers: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return lines


def _position(first_row: int, n: int, total: int) -> str:
    if n == total:
        return "whole file"
    if first_row == 0:
        return f"first {n} rows"
    if first_row + n == total:
        return f"last {n} rows"
    return f"rows {first_row}~{first_row + n - 1} (inside the file)"


def render(stats: list[dict], cfg: dict, manifest: dict | None, expected_cols: list[str],
           internal_dups: dict[str, int], overlaps: list[dict]) -> str:
    hai = cfg["ml"]["hai"]
    label_cols = hai["label_cols"]
    n_sensors = len(expected_cols) - 1 - len(label_cols)
    train = [s for s in stats if s["file"] in hai["train_files"]]
    test = [s for s in stats if s["file"] in hai["test_files"]]
    verified = {f["name"]: f["verified"] for f in manifest["files"]} if manifest else {}
    short = lambda f: f.split(".")[0]  # noqa: E731
    kind_of = lambda f: "train" if f in hai["train_files"] else "test"  # noqa: E731

    L = [f"# HAI {hai['version']} data overview (M0)", "",
         "Generated by `python -m scripts.hai_data_summary`. test files get only descriptive statistics here (rows, times, label counts, row hashes), "
         "never used for tuning, choosing the threshold or choosing models (constraint 2).", "",
         "## 1. Files and verification", ""]
    L += _table(["File", "Compressed size (bytes)", "Check against official repo", "Rows", "Columns", "Columns match reference"],
                [[s["file"], s["bytes"],
                  {True: "size and sha match", False: "mismatch"}.get(verified.get(s["file"]), "not verified"),
                  s["rows"], s["n_cols"], "yes" if s["columns_match"] else "no"] for s in stats])
    L += ["", f"- Total {sum(s['bytes'] for s in stats)} bytes ({sum(s['bytes'] for s in stats) / 1e6:.1f} MB), "
              f"{sum(s['rows'] for s in stats)} rows; train {sum(s['rows'] for s in train)} rows, "
              f"test {sum(s['rows'] for s in test)} rows."]
    if manifest:
        L.append(f"- Verified against: GitHub contents API (file size + git blob sha), checked at {manifest['checked_at']}.")
    L.append(f"- Column reference: `{hai['columns_txt']}`, {len(expected_cols)} columns = 1 time column + "
             f"{n_sensors} sensors + {len(label_cols)} label columns.")
    for s in stats:
        if not s["columns_match"]:
            L.append(f"- **{s['file']} columns differ**: missing {s['columns_missing']}, extra {s['columns_extra']}"
                     f"{'' if s['columns_missing'] or s['columns_extra'] else ' (same columns, different order)'}.")

    L += ["", "## 2. Time range and continuity", ""]
    L += _table(["File", "Start", "End", "Span (s)", "Rows", "Missing s", "Gaps", "Duplicate timestamps", "Out of order", "Missing values"],
                [[s["file"], s["start"], s["end"], s["span_s"], s["rows"], s["missing_seconds"], s["gap_count"],
                  s["duplicate_times"], s["non_increasing_steps"], s["nan_cells"]] for s in stats])
    L.append("")
    gap_lines = [f"- {s['file']}: {g['after']} → {g['before']}, {g['missing_s']} s missing"
                 for s in stats for g in s["gaps"]]
    L += gap_lines or ["- Every file is contiguous per second: no missing, duplicated or out-of-order seconds."]
    order = sorted(stats, key=lambda s: s["start"])
    L.append("- Ordered by start time: " + " → ".join(short(s["file"]) for s in order) + ".")
    tr = sorted(train, key=lambda s: s["start"])
    for a, b in zip(tr, tr[1:]):
        gap = int((pd.Timestamp(b["start"]) - pd.Timestamp(a["end"])).total_seconds()) - 1
        L.append(f"- Gap of {gap} s ({gap / 3600:.1f} h) between {a['file']} and {b['file']}: the file boundary is discontinuous, "
                 "and windows must not cross it (constraint 8).")

    # ---- Data overlap (its own section) ----
    n_pairs = len(stats) * (len(stats) - 1) // 2
    L += ["", "## 3. Data overlap between files (train and test contain the same data)", "",
          "### Method", "",
          f"- For every row of every file, take only the {n_sensors} sensor columns (not `{hai['time_col']}` or the label columns), convert them to float64 and compute a 64-bit row hash"
          " (`src/utils/rowhash.py`).",
          f"- All {len(stats)} files are compared pairwise ({n_pairs} pairs: train×test, train×train, test×test), "
          "finding rows with identical hashes and merging rows whose indices increase together on both sides into contiguous runs.",
          "- The comparison does not depend on timestamps, so shifted partial overlaps are found too; duplicated rows within each file are counted as well.", "",
          "### Results", ""]
    if overlaps:
        L += _table(["File A", "Position in A", "Time in A", "File B", "Position in B", "Time in B", "Identical rows", "Timestamps aligned"],
                    [[f"{short(o['a'])} ({kind_of(o['a'])})", _position(o["a_first_row"], o["rows"], o["a_rows_total"]),
                      f"{o['a_start']} ~ {o['a_end']}",
                      f"{short(o['b'])} ({kind_of(o['b'])})", _position(o["b_first_row"], o["rows"], o["b_rows_total"]),
                      f"{o['b_start']} ~ {o['b_end']}", o["rows"], "yes" if o["time_aligned"] else "no"]
                     for o in overlaps])
        L.append("")
    pairs_hit = {(o["a"], o["b"]) for o in overlaps}
    L.append(f"- {len(pairs_hit)} of the {n_pairs} file pairs share rows with identical content; none of the other {n_pairs - len(pairs_hit)} pairs share a single row.")
    L.append("- By type: train×test " + str(sum(1 for a, b in pairs_hit if kind_of(a) != kind_of(b)))
             + " pairs, train×train " + str(sum(1 for a, b in pairs_hit if kind_of(a) == kind_of(b) == "train"))
             + " pairs, test×test " + str(sum(1 for a, b in pairs_hit if kind_of(a) == kind_of(b) == "test")) + " pairs.")
    L.append("- Overlapping runs shifted in time: " + str(sum(1 for o in overlaps if not o["time_aligned"])) + ".")
    L.append("- Rows with duplicated content within a file: " + ", ".join(f"{short(n)} {c}" for n, c in internal_dups.items()) + ".")
    leaks = [o for o in overlaps if kind_of(o["a"]) != kind_of(o["b"])]
    attack_in_leak = 0
    for o in leaks:
        t_file, t_start, t_end = (o["b"], o["b_start"], o["b_end"]) if kind_of(o["b"]) == "test" \
            else (o["a"], o["a_start"], o["a_end"])
        ev = next(s for s in stats if s["file"] == t_file)["events"]
        attack_in_leak += sum(1 for e in ev if e["start"] <= t_end and e["end"] >= t_start)
    if leaks:
        L.append(f"- In total {sum(o['rows'] for o in leaks)} rows are identical between train and test; "
                 f"attack events intersecting these rows: {attack_in_leak} (the overlap is entirely normal data).")
    L += ["", "### Impact and handling", ""]
    if leaks:
        L += ["- Impact: if left alone, the model would already have seen this normal test data during training, the reconstruction error for those periods would be too low, and the false-alarm rate would be underestimated.",
              "- Principle: **no test row may appear in the training data, not even one.** So besides the last 43,201 rows of train1, "
              "row 1 of train1 (identical to the last row of test2) is removed as well.",
              "- Handling: **rows identical to test are removed from the training set; test stays exactly as officially released** (implemented in M1, config switch on by default). "
              "Constant columns are determined and the scaler fit after the removal; the removed rows are at the head or tail of a train file, so no new internal boundary appears.",
              "- Status: implemented in M1 (`ml.data.drop_train_overlap` on by default; `ml.data.train_overlap_drops` is head_rows 1, tail_rows 43201 "
              "for train1). Both removals confirmed (2026-10-07).",
              "- Verification: the M1 tests assert by content hash that no training window contains any test row.",
              "- Cited in the report: the data section states where and how many rows were removed; the limitations section states that the official train/test release contains duplicates, "
              "so results are not fully comparable with publications that did not do this."]
    else:
        L.append("- No rows with identical content between train and test; nothing to do.")

    # ---- Labels ----
    L += ["", "## 4. Labels", "", "Event = contiguous interval where the label is 1.", ""]
    rows = []
    for s in stats:
        for c in label_cols:
            v = s["labels"][c]
            if v is None:
                rows.append([s["file"], c, "column missing", "", "", "", "", ""])
            else:
                share = f"{100 * v['attack_s'] / s['rows']:.2f}%"
                rows.append([s["file"], c, v["attack_s"], share, v["events"],
                             *(("" if v[k] is None else f"{v[k]:g}") for k in ("min_s", "median_s", "max_s"))])
    L += _table(["File", "Label", "Attack seconds", "Share", "Events", "Shortest (s)", "Median (s)", "Longest (s)"], rows)
    main = label_cols[0]
    tot_s = sum(s["labels"][main]["attack_s"] for s in test if s["labels"][main])
    tot_e = sum(s["labels"][main]["events"] for s in test if s["labels"][main])
    tot_rows = sum(s["rows"] for s in test)
    train_att = sum(s["labels"][c]["attack_s"] for s in train for c in label_cols if s["labels"][c])
    L += ["", f"- test total: `{main}` attack {tot_s} s ({100 * tot_s / tot_rows:.2f}%), {tot_e} events.",
          f"- Attack seconds over all label columns of the train files: {train_att} (should be 0)."]
    odd = [(s["file"], c) for s in stats for c in label_cols
           if s["labels"][c] and not set(s["labels"][c]["values"]) <= {0, 1}]
    L.append("- All label values are 0/1." if not odd else f"- **Label values other than 0/1**: {odd}.")
    lc = [(s["file"], s["label_consistency"]) for s in test if "label_consistency" in s]
    L.append("- Seconds with a subsystem label of 1 but `" + main + "`=0: total " + str(sum(c["sub_without_main_s"] for _, c in lc)) + ".")
    L.append("- Seconds with several subsystem labels equal to 1: " + ", ".join(f"{short(f)} {c['multi_sub_s']}" for f, c in lc)
             + f" (total {sum(c['multi_sub_s'] for _, c in lc)}).")
    L.append("- Subsystem labels can have more events than the overall label: within one attack event the subsystem labels appear intermittently (as short as 1 s). "
             f"Detection metrics (point-wise and event-level) use only the `{main}` column and are not affected by the subsystem labels.")

    # ---- Seconds without subsystem labels ----
    events = [e for s in test for e in s["events"]]
    partial = [e for e in events if e["no_sub_s"] > 0]
    kinds = {k: [e for e in events if e["no_sub_kind"] == k] for k in ("whole", "edge", "interior")}
    multi = [e for e in events if "+" in e["subsystems"]]
    L += ["", f"### Seconds with `{main}`=1 but no subsystem label", "",
          "- Total " + str(sum(c["main_without_sub_s"] for _, c in lc)) + " s: "
          + ", ".join(f"{short(f)} {c['main_without_sub_s']}" for f, c in lc) + ".",
          f"- These seconds fall in {len(partial)} / {len(events)} attack events: {len(kinds['whole'])} events without any subsystem label; "
          f"{len(kinds['edge'])} where the unlabeled seconds touch the start or end of the event; {len(kinds['interior'])} where they occur only inside the event.", ""]
    L += _table(["File", "Event", "Start", "Duration (s)", "Subsystem union", "Unlabeled s", "Unlabeled runs", "First offset (s)", "Last offset (s)", "Kind"],
                [[short(e["file"]), e["event"], e["start"], e["duration_s"], e["subsystems"] or "(empty)", e["no_sub_s"],
                  e["no_sub_runs"], e["no_sub_first_offset_s"], e["no_sub_last_offset_s"], e["no_sub_kind"]]
                 for e in partial])
    by_union = {}
    for e in events:
        by_union[e["subsystems"] or "(empty)"] = by_union.get(e["subsystems"] or "(empty)", 0) + 1
    L += ["", "- Rule (M5): an event's subsystem set = union of attack_P* over all its seconds. Events with a non-empty union take part in the hit rate normally; "
              "events with an empty union are excluded from the hit-rate denominator and listed as \"events without subsystem labels\".",
          f"- Under this rule: of the {len(events)} events, {len(kinds['whole'])} have an empty union "
          f"and {len(multi)} have a union with several subsystems. Union distribution: " + ", ".join(f"{k} {v}" for k, v in sorted(by_union.items())) + ".",
          "- Details of all events are in `attack_events.csv`."]

    L += ["", "## 5. Constant columns of the train files", "",
          "Computed on the train files only. M1 re-determines them on \"the training part concatenated after removing the overlapping rows\"; that result is authoritative and this one is for reference only.", ""]
    for s in train:
        L.append(f"- {s['file']}: {len(s['constant_cols'])}: {', '.join(s['constant_cols']) or 'none'}")
    common = set(train[0]["constant_cols"]).intersection(*(s["constant_cols"] for s in train)) if train else set()
    L.append(f"- Columns constant in every train file: {len(common)}: {', '.join(sorted(common)) or 'none'}")
    return "\n".join(L) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Per-file statistics of the HAI data")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path, help="defaults to ml.report_dir in the config")
    args = p.parse_args()

    cfg = load_config(args.config)
    hai = cfg["ml"]["hai"]
    data_dir = hai_data_dir(cfg)
    expected = read_expected_columns(resolve(hai["columns_txt"]))
    manifest_path = data_dir / hai["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None

    stats, hashes = [], {}
    for name in hai["train_files"] + hai["test_files"]:
        s = summarize_file(data_dir / name, expected, hai["time_col"], hai["label_cols"],
                           with_feature_stats=name in hai["train_files"])
        hashes[name] = file_row_hashes(data_dir / name, hai["time_col"], hai["label_cols"])
        print(f"[STATS] {name}: {s['rows']} rows, {s['start']} ~ {s['end']}, missing seconds {s['missing_seconds']}, "
              f"columns match {s['columns_match']}")
        stats.append(s)

    internal_dups, overlaps = find_content_overlaps(hashes)
    for o in overlaps:
        print(f"[OVERLAP] {o['a']} ~ {o['b']}: {o['rows']} rows, A {o['a_start']} ~ {o['a_end']}, B {o['b_start']} ~ {o['b_end']}")

    out_dir = args.output_dir or resolve(cfg["ml"]["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_summary.md").write_text(render(stats, cfg, manifest, expected, internal_dups, overlaps),
                                             encoding="utf-8")
    events = [e for s in stats for e in s["events"]]
    pd.DataFrame(events, columns=EVENT_FIELDS).to_csv(out_dir / "attack_events.csv", index=False)
    print(f"[SAVED] {out_dir / 'data_summary.md'}, {out_dir / 'attack_events.csv'} ({len(events)} events)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
