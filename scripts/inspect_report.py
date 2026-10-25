"""Print the structure of the result nodes in an OpenVAS XML report, to confirm the fields before writing parse_report.py.

Takes the union over all result nodes: lists each sub-path with its occurrence count, attribute names and one example value,
and reports the severity distribution, QoD distribution and refs types. Read-only; no gvmd connection needed.

Usage:
    python -m scripts.inspect_report [data/samples/smoke_report.xml]
"""

import argparse
from collections import Counter
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parent.parent


def walk(el, path: str, stats: dict) -> None:
    """Recursively record path -> [occurrences, set of attribute names, example value], preserving first-seen order."""
    entry = stats.setdefault(path, [0, set(), None])
    entry[0] += 1
    entry[1].update(el.attrib.keys())
    text = (el.text or "").strip()
    if text and entry[2] is None:
        entry[2] = " ".join(text.split())[:70]
    for child in el:
        if isinstance(child.tag, str):
            walk(child, f"{path}/{child.tag}", stats)


def main() -> None:
    p = argparse.ArgumentParser(description="Show the structure of an OpenVAS XML report")
    p.add_argument("path", type=Path, nargs="?", default=ROOT / "data" / "samples" / "smoke_report.xml")
    args = p.parse_args()

    root = etree.parse(str(args.path)).getroot()
    report = root.find(".//report/report")
    results = root.findall(".//results/result")
    print(f"File: {args.path}")
    print(f"Root node: <{root.tag}>, inner report id={report.get('id') if report is not None else '?'}")
    print(f"Total results: {len(results)}")
    if not results:
        return

    stats: dict = {}
    for r in results:
        walk(r, "result", stats)

    # Sort by "first-seen order of each ancestor level", so child nodes come right below their parent
    first_seen = {path: i for i, path in enumerate(stats)}

    def order(path: str) -> tuple:
        parts = path.split("/")
        return tuple(first_seen["/".join(parts[: i + 1])] for i in range(len(parts)))

    print("\n== Structure of result nodes (indent = level | attributes | [occurrences] | example value)")
    for path in sorted(stats, key=order):
        count, attrs, sample = stats[path]
        depth = path.count("/")
        name = path.rsplit("/", 1)[-1]
        attr_s = f" @{','.join(sorted(attrs))}" if attrs else ""
        sample_s = f"  = {sample}" if sample else ""
        print(f"{'  ' * depth}{name}{attr_s}  [{count}]{sample_s}")

    print("\n== threat distribution:", dict(Counter(r.findtext("threat") for r in results)))
    sev = [float(r.findtext("severity") or 0) for r in results]
    print(f"== severity: min={min(sev)} max={max(sev)}, {sum(s >= 7 for s in sev)} with >= 7.0")
    print("== qod/value distribution:", dict(Counter(r.findtext("qod/value") for r in results)))
    print("== refs type counts:", dict(Counter(ref.get("type") for ref in root.iterfind(".//results/result/nvt/refs/ref"))))
    ports = Counter(r.findtext("port") for r in results)
    print("== ports (top 10):", dict(ports.most_common(10)))


if __name__ == "__main__":
    main()
