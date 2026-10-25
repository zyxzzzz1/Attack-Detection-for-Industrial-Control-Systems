"""Print all column names of a HAI train file, marking the time column / sensor columns (by subsystem prefix) / label columns, and save them to a text file.

For inspection only (claude.md section 11: do not guess column names). Also reports row count, time range and the values of each label column,
to confirm whether the labels of the train files are all 0.

Usage:
    python -m scripts.inspect_hai_columns data/raw/hai-21.03/train1.csv.gz \
        --output data/samples/hai_columns.txt
"""

import argparse
import csv
import gzip
import re
from collections import Counter, defaultdict
from pathlib import Path

LABEL_RE = re.compile(r"^attack(_P\d+)?$", re.IGNORECASE)
PREFIX_RE = re.compile(r"^(P\d+)_")


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect the column names of a HAI CSV")
    p.add_argument("csv", type=Path, help="HAI train file (.csv or .csv.gz)")
    p.add_argument("--output", type=Path, default=Path("data/samples/hai_columns.txt"))
    args = p.parse_args()

    opener = gzip.open if args.csv.suffix == ".gz" else open
    with opener(args.csv, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        labels = [i for i, h in enumerate(header) if LABEL_RE.match(h)]
        label_values = defaultdict(Counter)
        n_rows, first_time, last_time = 0, None, None
        for row in reader:
            n_rows += 1
            first_time = first_time or row[0]
            last_time = row[0]
            for i in labels:
                label_values[header[i]][row[i]] += 1

    by_sub = defaultdict(list)
    lines = [f"# Source: {args.csv.name} (official HAI 21.03 repository https://github.com/icsdataset/hai)",
             f"# Rows {n_rows}, time {first_time} ~ {last_time}, columns {len(header)}",
             "# Marks: [time] time column, [label] label column (not a feature), [P?] subsystem of the sensor", ""]
    for i, h in enumerate(header):
        if i == 0:
            tag = "time"
        elif LABEL_RE.match(h):
            tag = "label"
        else:
            m = PREFIX_RE.match(h)
            tag = m.group(1) if m else "UNKNOWN"
            by_sub[tag].append(h)
        lines.append(f"{i:3d}  [{tag}]  {h}")

    lines += ["", "# Sensors per subsystem"]
    lines += [f"#   {k}: {len(v)}" for k, v in sorted(by_sub.items())]
    lines += ["", "# Label column value counts"]
    lines += [f"#   {k}: {dict(sorted(v.items()))}" for k, v in label_values.items()]

    text = "\n".join(lines) + "\n"
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"[SAVED] {args.output}")


if __name__ == "__main__":
    main()
