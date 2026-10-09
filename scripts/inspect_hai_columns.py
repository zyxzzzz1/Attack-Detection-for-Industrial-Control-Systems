"""打印 HAI train 文件的全部列名，标注时间列 / 传感器列（按子系统前缀）/ 标签列，并保存到文本文件。

只做检查用途（claude.md 第 11 节：不臆测列名）。同时统计行数、时间范围和各标签列的取值，
确认 train 文件的标签是否全为 0。

用法：
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
    p = argparse.ArgumentParser(description="检查 HAI CSV 列名")
    p.add_argument("csv", type=Path, help="HAI train 文件（.csv 或 .csv.gz）")
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
    lines = [f"# 来源：{args.csv.name}（HAI 21.03 官方仓库 https://github.com/icsdataset/hai）",
             f"# 行数 {n_rows}，时间 {first_time} ~ {last_time}，列数 {len(header)}",
             "# 标注：[time] 时间列，[label] 标签列（不作为特征），[P?] 传感器所属子系统", ""]
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

    lines += ["", "# 子系统传感器数"]
    lines += [f"#   {k}: {len(v)}" for k, v in sorted(by_sub.items())]
    lines += ["", "# 标签列取值统计"]
    lines += [f"#   {k}: {dict(sorted(v.items()))}" for k, v in label_values.items()]

    text = "\n".join(lines) + "\n"
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"[保存] {args.output}")


if __name__ == "__main__":
    main()
