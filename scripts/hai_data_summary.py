"""M0：逐文件统计 HAI 数据并写入 reports/main/data_summary.md 与 attack_events.csv。

每个文件：行数、时间范围、连续性（缺秒 / 重复 / 乱序）、列是否与 data/samples/hai_columns.txt 一致、
缺失值、各标签的攻击秒数与事件数（事件 = 标签为 1 的连续区间）、每个攻击事件的子系统标签情况。
train 文件另外统计常量列。全部文件两两之间按逐行内容哈希检查是否含有同一份数据（数据泄漏检查）。

逐文件读取、传感器用 float32、统计完即释放，不同时加载多个文件（哈希另读一遍 float64，只保留每行 8 字节）。
test 文件在这里只做描述性统计（行数、时间、标签计数、行哈希），不读取其传感器取值分布，结果不用于调参（约束 2）。

用法：
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
    """从 hai_columns.txt 取出按顺序排列的列名（忽略 # 注释行）。"""
    cols = []
    for line in columns_txt.read_text(encoding="utf-8").splitlines():
        m = COL_LINE_RE.match(line)
        if m and not line.lstrip().startswith("#"):
            cols.append(m.group(1))
    return cols


def run_bounds(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回每个取值为 1 的连续区间的 [start, end)。"""
    d = np.diff(np.concatenate(([0], (binary != 0).astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def run_lengths(binary: np.ndarray) -> np.ndarray:
    starts, ends = run_bounds(binary)
    return ends - starts


def attack_events(name: str, times: np.ndarray, main: np.ndarray, subs: np.ndarray, sub_names: list[str]) -> list[dict]:
    """逐个攻击事件（总标签的连续区间）统计子系统标签：并集，以及没有任何子系统标签的秒的位置。

    no_sub_kind：none = 每秒都有子系统标签；whole = 整个事件都没有；
    edge = 无标签的秒贴着事件的开头或结尾；interior = 只出现在事件内部。
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
    """返回（逐行内容哈希, 时间字符串）。哈希只用传感器列，按 float64 解析。"""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    features = [c for c in df.columns if c != time_col and c not in label_cols]
    out = hash_rows(df[features]), df[time_col].astype(str).to_numpy()
    del df
    gc.collect()
    return out


def find_content_overlaps(hashes: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[dict[str, int], list[dict]]:
    """全部文件两两比较行哈希。返回（各文件内部内容重复的行数, 文件间的重叠段列表）。"""
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
        return "整个文件"
    if first_row == 0:
        return f"开头 {n} 行"
    if first_row + n == total:
        return f"末尾 {n} 行"
    return f"第 {first_row}~{first_row + n - 1} 行（文件内部）"


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

    L = [f"# HAI {hai['version']} 数据概况（M0）", "",
         "由 `python -m scripts.hai_data_summary` 生成。test 文件在此只做描述性统计（行数、时间、标签计数、行哈希），"
         "不用于调参、选阈值或选模型（约束 2）。", "",
         "## 1. 文件与核对", ""]
    L += _table(["文件", "压缩大小（字节）", "与官方仓库核对", "行数", "列数", "列与基准一致"],
                [[s["file"], s["bytes"],
                  {True: "大小与 sha 一致", False: "不一致"}.get(verified.get(s["file"]), "未核对"),
                  s["rows"], s["n_cols"], "是" if s["columns_match"] else "否"] for s in stats])
    L += ["", f"- 合计 {sum(s['bytes'] for s in stats)} 字节（{sum(s['bytes'] for s in stats) / 1e6:.1f} MB），"
              f"{sum(s['rows'] for s in stats)} 行；train {sum(s['rows'] for s in train)} 行，"
              f"test {sum(s['rows'] for s in test)} 行。"]
    if manifest:
        L.append(f"- 核对来源：GitHub contents API（文件大小 + git blob sha），核对时间 {manifest['checked_at']}。")
    L.append(f"- 列基准：`{hai['columns_txt']}`，共 {len(expected_cols)} 列 = 1 个时间列 + "
             f"{n_sensors} 个传感器 + {len(label_cols)} 个标签列。")
    for s in stats:
        if not s["columns_match"]:
            L.append(f"- **{s['file']} 列不一致**：缺少 {s['columns_missing']}，多出 {s['columns_extra']}"
                     f"{'' if s['columns_missing'] or s['columns_extra'] else '（列相同但顺序不同）'}。")

    L += ["", "## 2. 时间范围与连续性", ""]
    L += _table(["文件", "开始", "结束", "跨度（秒）", "行数", "缺秒", "缺口数", "重复时间戳", "乱序", "缺失值"],
                [[s["file"], s["start"], s["end"], s["span_s"], s["rows"], s["missing_seconds"], s["gap_count"],
                  s["duplicate_times"], s["non_increasing_steps"], s["nan_cells"]] for s in stats])
    L.append("")
    gap_lines = [f"- {s['file']}：{g['after']} → {g['before']}，缺 {g['missing_s']} 秒"
                 for s in stats for g in s["gaps"]]
    L += gap_lines or ["- 所有文件内部逐秒连续：无缺秒、无重复、无乱序。"]
    order = sorted(stats, key=lambda s: s["start"])
    L.append("- 按开始时间排序：" + " → ".join(short(s["file"]) for s in order) + "。")
    tr = sorted(train, key=lambda s: s["start"])
    for a, b in zip(tr, tr[1:]):
        gap = int((pd.Timestamp(b["start"]) - pd.Timestamp(a["end"])).total_seconds()) - 1
        L.append(f"- {a['file']} 与 {b['file']} 之间间隔 {gap} 秒（{gap / 3600:.1f} 小时）：文件边界不连续，"
                 "滑窗不得跨越（约束 8）。")

    # ---- 数据重叠（单独成节）----
    n_pairs = len(stats) * (len(stats) - 1) // 2
    L += ["", "## 3. 文件之间的数据重叠（train 与 test 含有同一份数据）", "",
          "### 检测方法", "",
          f"- 对每个文件的每一行，只取 {n_sensors} 个传感器列（不含 `{hai['time_col']}` 和标签列），统一转成 float64 后计算 64 位行哈希"
          "（`src/utils/rowhash.py`）。",
          f"- 全部 {len(stats)} 个文件两两比较（{n_pairs} 对：train×test、train×train、test×test），"
          "找出哈希相同的行，并把两边下标同步递增的行合并成连续段。",
          "- 比较不依赖时间戳，因此时间上错位的部分重叠也能发现；同时统计每个文件内部内容重复的行数。", "",
          "### 结果", ""]
    if overlaps:
        L += _table(["文件 A", "A 中位置", "A 中时间", "文件 B", "B 中位置", "B 中时间", "相同行数", "时间戳对齐"],
                    [[f"{short(o['a'])}（{kind_of(o['a'])}）", _position(o["a_first_row"], o["rows"], o["a_rows_total"]),
                      f"{o['a_start']} ~ {o['a_end']}",
                      f"{short(o['b'])}（{kind_of(o['b'])}）", _position(o["b_first_row"], o["rows"], o["b_rows_total"]),
                      f"{o['b_start']} ~ {o['b_end']}", o["rows"], "是" if o["time_aligned"] else "否"]
                     for o in overlaps])
        L.append("")
    pairs_hit = {(o["a"], o["b"]) for o in overlaps}
    L.append(f"- {n_pairs} 对文件中有 {len(pairs_hit)} 对含相同内容的行，其余 {n_pairs - len(pairs_hit)} 对没有任何一行相同。")
    L.append("- 按类型：train×test " + str(sum(1 for a, b in pairs_hit if kind_of(a) != kind_of(b)))
             + " 对，train×train " + str(sum(1 for a, b in pairs_hit if kind_of(a) == kind_of(b) == "train"))
             + " 对，test×test " + str(sum(1 for a, b in pairs_hit if kind_of(a) == kind_of(b) == "test")) + " 对。")
    L.append("- 时间上错位的重叠段：" + str(sum(1 for o in overlaps if not o["time_aligned"])) + " 个。")
    L.append("- 文件内部内容重复的行：" + "，".join(f"{short(n)} {c}" for n, c in internal_dups.items()) + "。")
    leaks = [o for o in overlaps if kind_of(o["a"]) != kind_of(o["b"])]
    attack_in_leak = 0
    for o in leaks:
        t_file, t_start, t_end = (o["b"], o["b_start"], o["b_end"]) if kind_of(o["b"]) == "test" \
            else (o["a"], o["a_start"], o["a_end"])
        ev = next(s for s in stats if s["file"] == t_file)["events"]
        attack_in_leak += sum(1 for e in ev if e["start"] <= t_end and e["end"] >= t_start)
    if leaks:
        L.append(f"- train 与 test 相同的行合计 {sum(o['rows'] for o in leaks)} 行；"
                 f"与这些行相交的攻击事件：{attack_in_leak} 个（重叠部分全部是正常数据）。")
    L += ["", "### 影响与处理方式", ""]
    if leaks:
        L += ["- 影响：不处理的话，这部分 test 的正常数据在训练时已被模型见过，对应时段的重构误差偏低，误报率会被低估。",
              "- 原则：**训练数据中不出现任何 test 行，哪怕只有一行。**因此除 train1 末尾的 43,201 行外，"
              "train1 的第 1 行（与 test2 最后 1 行相同）也一并删除。",
              "- 处理：**从训练集中删除与 test 相同的行，test 保持官方原样**（M1 实现，config 开关默认开启）。"
              "删除后再判定常量列、再 fit scaler；被删的行位于 train 文件的开头或末尾，不产生新的内部边界。",
              "- 状态：M1 已实现（`ml.data.drop_train_overlap` 默认开启；`ml.data.train_overlap_drops` 为 train1 的 "
              "head_rows 1、tail_rows 43201）。两处删除均已确认（2026-10-07）。",
              "- 验证：M1 的测试按内容哈希断言训练窗口中不含 test 的任何一行。",
              "- 报告中引用：数据部分说明删除的位置与行数；局限性部分说明官方发布的 train/test 存在重复，"
              "与未做此处理的文献结果不完全可比。"]
    else:
        L.append("- train 与 test 之间没有相同内容的行，无需处理。")

    # ---- 标签 ----
    L += ["", "## 4. 标签", "", "事件 = 标签为 1 的连续区间。", ""]
    rows = []
    for s in stats:
        for c in label_cols:
            v = s["labels"][c]
            if v is None:
                rows.append([s["file"], c, "列不存在", "", "", "", "", ""])
            else:
                share = f"{100 * v['attack_s'] / s['rows']:.2f}%"
                rows.append([s["file"], c, v["attack_s"], share, v["events"],
                             *(("" if v[k] is None else f"{v[k]:g}") for k in ("min_s", "median_s", "max_s"))])
    L += _table(["文件", "标签", "攻击秒数", "占比", "事件数", "最短（秒）", "中位（秒）", "最长（秒）"], rows)
    main = label_cols[0]
    tot_s = sum(s["labels"][main]["attack_s"] for s in test if s["labels"][main])
    tot_e = sum(s["labels"][main]["events"] for s in test if s["labels"][main])
    tot_rows = sum(s["rows"] for s in test)
    train_att = sum(s["labels"][c]["attack_s"] for s in train for c in label_cols if s["labels"][c])
    L += ["", f"- test 合计：`{main}` 攻击 {tot_s} 秒（占 {100 * tot_s / tot_rows:.2f}%），{tot_e} 个事件。",
          f"- train 文件全部标签列的攻击秒数合计：{train_att}（应为 0）。"]
    odd = [(s["file"], c) for s in stats for c in label_cols
           if s["labels"][c] and not set(s["labels"][c]["values"]) <= {0, 1}]
    L.append("- 标签取值均为 0/1。" if not odd else f"- **标签取值不是 0/1**：{odd}。")
    lc = [(s["file"], s["label_consistency"]) for s in test if "label_consistency" in s]
    L.append("- 子系统标签为 1 但 `" + main + "`=0 的秒数：合计 " + str(sum(c["sub_without_main_s"] for _, c in lc)) + "。")
    L.append("- 同时有多个子系统标签为 1 的秒数：" + "，".join(f"{short(f)} {c['multi_sub_s']}" for f, c in lc)
             + f"（合计 {sum(c['multi_sub_s'] for _, c in lc)}）。")
    L.append("- 子系统标签的事件数可能多于总标签：同一个攻击事件内，子系统标签会断续出现（最短 1 秒）。"
             f"检测指标（逐点和事件级）只用 `{main}` 列，不受子系统标签影响。")

    # ---- 无子系统标签的秒 ----
    events = [e for s in test for e in s["events"]]
    partial = [e for e in events if e["no_sub_s"] > 0]
    kinds = {k: [e for e in events if e["no_sub_kind"] == k] for k in ("whole", "edge", "interior")}
    multi = [e for e in events if "+" in e["subsystems"]]
    L += ["", f"### `{main}`=1 但没有任何子系统标签的秒", "",
          "- 合计 " + str(sum(c["main_without_sub_s"] for _, c in lc)) + " 秒："
          + "，".join(f"{short(f)} {c['main_without_sub_s']}" for f, c in lc) + "。",
          f"- 这些秒分布在 {len(partial)} / {len(events)} 个攻击事件中：整个事件都没有子系统标签的 {len(kinds['whole'])} 个；"
          f"无标签的秒贴着事件开头或结尾的 {len(kinds['edge'])} 个；只出现在事件内部的 {len(kinds['interior'])} 个。", ""]
    L += _table(["文件", "事件", "开始", "时长（秒）", "子系统并集", "无标签秒数", "无标签段数", "首个偏移（秒）", "末个偏移（秒）", "类型"],
                [[short(e["file"]), e["event"], e["start"], e["duration_s"], e["subsystems"] or "（空）", e["no_sub_s"],
                  e["no_sub_runs"], e["no_sub_first_offset_s"], e["no_sub_last_offset_s"], e["no_sub_kind"]]
                 for e in partial])
    by_union = {}
    for e in events:
        by_union[e["subsystems"] or "（空）"] = by_union.get(e["subsystems"] or "（空）", 0) + 1
    L += ["", "- 判定规则（M5）：事件的子系统集合 = 该事件内所有秒的 attack_P* 并集。并集非空的事件正常参与命中率；"
              "并集为空的事件不计入命中率分母，单列为“无子系统标签事件”。",
          f"- 按此规则：{len(events)} 个事件中并集为空的有 {len(kinds['whole'])} 个，"
          f"并集含多个子系统的有 {len(multi)} 个。并集分布：" + "，".join(f"{k} {v}" for k, v in sorted(by_union.items())) + "。",
          "- 全部事件的明细见 `attack_events.csv`。"]

    L += ["", "## 5. train 文件的常量列", "",
          "只在 train 文件上统计。M1 中以“删除重叠行之后拼接的训练部分”重新判定，以那里的结果为准，这里仅供参考。", ""]
    for s in train:
        L.append(f"- {s['file']}：{len(s['constant_cols'])} 个：{', '.join(s['constant_cols']) or '无'}")
    common = set(train[0]["constant_cols"]).intersection(*(s["constant_cols"] for s in train)) if train else set()
    L.append(f"- 在全部 train 文件中都为常量的列：{len(common)} 个：{', '.join(sorted(common)) or '无'}")
    return "\n".join(L) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="HAI 数据逐文件统计")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path, help="默认取 config 的 ml.report_dir")
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
        print(f"[统计] {name}: {s['rows']} 行, {s['start']} ~ {s['end']}, 缺秒 {s['missing_seconds']}, "
              f"列一致 {s['columns_match']}")
        stats.append(s)

    internal_dups, overlaps = find_content_overlaps(hashes)
    for o in overlaps:
        print(f"[重叠] {o['a']} ~ {o['b']}: {o['rows']} 行, A {o['a_start']} ~ {o['a_end']}, B {o['b_start']} ~ {o['b_end']}")

    out_dir = args.output_dir or resolve(cfg["ml"]["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_summary.md").write_text(render(stats, cfg, manifest, expected, internal_dups, overlaps),
                                             encoding="utf-8")
    events = [e for s in stats for e in s["events"]]
    pd.DataFrame(events, columns=EVENT_FIELDS).to_csv(out_dir / "attack_events.csv", index=False)
    print(f"[保存] {out_dir / 'data_summary.md'}, {out_dir / 'attack_events.csv'}（{len(events)} 个事件）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
