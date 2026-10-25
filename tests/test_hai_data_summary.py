"""Tests of the M0 data statistics and path resolution: all use small hand-made CSVs and never read the real HAI data."""

import hashlib

import numpy as np
import pandas as pd

from scripts.download_hai import check, git_blob_sha
from scripts.hai_data_summary import (file_row_hashes, find_content_overlaps, read_expected_columns, run_lengths,
                                      summarize_file)
from src.utils.paths import ROOT, artifact_dir, hai_data_dir, load_config
from src.utils.rowhash import hash_rows, matched_runs

LABELS = ["attack", "attack_P1", "attack_P2", "attack_P3"]
COLS = ["time", "P1_A", "P1_B", "P2_C", *LABELS]


def make_csv(path, times, attack=None, attack_p1=None, const=1.0, base=0):
    n = len(times)
    df = pd.DataFrame({"time": times, "P1_A": np.arange(base, base + n, dtype=float), "P1_B": const,
                       "P2_C": np.arange(base, base + n, dtype=float) * 2})
    df["attack"] = attack if attack is not None else 0
    df["attack_P1"] = attack_p1 if attack_p1 is not None else 0
    df["attack_P2"] = 0
    df["attack_P3"] = 0
    df.to_csv(path, index=False)
    return path


def seconds(start, n):
    return [str(t) for t in pd.date_range(start, periods=n, freq="s")]


def test_run_lengths():
    assert run_lengths(np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])).tolist() == [2, 1, 3]
    assert run_lengths(np.zeros(5)).tolist() == []
    assert run_lengths(np.ones(4)).tolist() == [4]


def test_summarize_continuous_file(tmp_path):
    attack = [0, 1, 1, 0, 0, 1, 1, 1, 0, 0]
    attack_p1 = [0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
    p = make_csv(tmp_path / "a.csv.gz", seconds("2020-01-01", 10), attack, attack_p1)
    s = summarize_file(p, COLS, "time", LABELS, with_feature_stats=True)
    assert s["rows"] == 10 and s["span_s"] == 10
    assert s["missing_seconds"] == 0 and s["duplicate_times"] == 0 and s["non_increasing_steps"] == 0
    assert s["columns_match"] and s["n_features"] == 3 and s["nan_cells"] == 0
    assert s["labels"]["attack"] == {"values": [0, 1], "attack_s": 5, "events": 2,
                                     "min_s": 2, "median_s": 2.5, "max_s": 3}
    # Subsystem labels appear intermittently within one event: more events than for the overall label
    assert s["labels"]["attack_P1"]["events"] == 3
    assert s["label_consistency"] == {"main_without_sub_s": 2, "sub_without_main_s": 0, "multi_sub_s": 0}
    # Event 1: unlabeled seconds touch the end; event 2: unlabeled seconds are in the interior
    e1, e2 = s["events"]
    assert (e1["start"], e1["duration_s"], e1["subsystems"], e1["no_sub_s"], e1["no_sub_kind"]) == \
        ("2020-01-01 00:00:01", 2, "P1", 1, "edge")
    assert (e2["no_sub_s"], e2["no_sub_first_offset_s"], e2["no_sub_kind"]) == (1, 1, "interior")
    assert s["constant_cols"] == ["P1_B"]


def test_summarize_detects_gap_duplicate_and_column_mismatch(tmp_path):
    times = seconds("2020-01-01 00:00:00", 3) + seconds("2020-01-01 00:00:10", 2)
    times.append(times[-1])  # Duplicated timestamp
    p = make_csv(tmp_path / "b.csv", times)
    s = summarize_file(p, COLS + ["P3_X"], "time", LABELS, with_feature_stats=False)
    assert s["gap_count"] == 1 and s["missing_seconds"] == 7
    assert s["gaps"][0]["after"] == "2020-01-01 00:00:02"
    assert s["duplicate_times"] == 1 and s["non_increasing_steps"] == 1
    assert not s["columns_match"] and s["columns_missing"] == ["P3_X"]
    assert "constant_cols" not in s


def test_events_without_any_subsystem_label(tmp_path):
    p = make_csv(tmp_path / "a.csv", seconds("2020-01-01", 6), attack=[0, 1, 1, 0, 1, 0], attack_p1=[0, 0, 0, 0, 1, 0])
    e1, e2 = summarize_file(p, COLS, "time", LABELS, False)["events"]
    assert (e1["subsystems"], e1["no_sub_kind"]) == ("", "whole")
    assert (e2["subsystems"], e2["no_sub_kind"], e2["no_sub_s"]) == ("P1", "none", 0)


def test_hash_rows_ignores_int_vs_float_dtype():
    # The same column may be parsed as int64 or float64 in different files; rows with equal values must hash the same
    a = pd.DataFrame({"x": [1, 2], "y": [0.5, 1.5]})
    b = pd.DataFrame({"x": [1.0, 2.0], "y": [0.5, 1.5]})
    assert (hash_rows(a) == hash_rows(b)).all()
    assert hash_rows(a)[0] != hash_rows(a)[1]


def test_matched_runs_finds_misaligned_and_split_segments():
    a = np.array([10, 11, 12, 13, 14, 15, 16], dtype=np.uint64)
    # b contains a[1:4] (shifted to index 5) and a[5:7] (index 0), with unrelated rows in between
    b = np.array([15, 16, 90, 91, 92, 11, 12, 13, 93], dtype=np.uint64)
    assert matched_runs(a, b) == [(1, 5, 3), (5, 0, 2)]
    assert matched_runs(a, np.array([1, 2], dtype=np.uint64)) == []


def test_content_overlap_ignores_time_and_labels(tmp_path):
    # The last 4 rows of a have the same sensor values as the first 4 rows of b, but completely different timestamps (shifted overlap), and b carries attack labels
    a = make_csv(tmp_path / "a.csv", seconds("2020-01-01", 10), base=0)
    b = make_csv(tmp_path / "b.csv", seconds("2020-03-01", 8), attack=[1] * 8, base=6)
    # c has the same timestamps as a but different content: not an overlap
    c = make_csv(tmp_path / "c.csv", seconds("2020-01-01", 5), base=100)
    hashes = {p.name: file_row_hashes(p, "time", LABELS) for p in (a, b, c)}
    internal, runs = find_content_overlaps(hashes)
    assert internal == {"a.csv": 0, "b.csv": 0, "c.csv": 0}
    assert len(runs) == 1
    r = runs[0]
    assert (r["a"], r["b"], r["rows"], r["a_first_row"], r["b_first_row"], r["time_aligned"]) == \
        ("a.csv", "b.csv", 4, 6, 0, False)
    assert r["a_start"] == "2020-01-01 00:00:06" and r["b_end"] == "2020-03-01 00:00:03"


def test_expected_columns_match_sample_file():
    cfg = load_config()
    cols = read_expected_columns(ROOT / cfg["ml"]["hai"]["columns_txt"])
    assert len(cols) == 84 and cols[0] == "time" and cols[-4:] == LABELS


def test_git_blob_sha_and_check(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello\n")
    expected = hashlib.sha1(b"blob 6\0hello\n").hexdigest()
    assert git_blob_sha(p) == expected
    assert check(p, 6, expected) is None
    assert "size" in check(p, 7, expected)
    assert check(p, 6, "0" * 40) == "sha mismatch"
    assert check(tmp_path / "missing", 1, expected) == "file does not exist"


def test_data_and_artifact_dirs_env_override(monkeypatch, tmp_path):
    cfg = load_config()
    monkeypatch.delenv("HAI_DATA_DIR", raising=False)
    monkeypatch.delenv("ARTIFACT_DIR", raising=False)
    assert hai_data_dir(cfg) == ROOT / cfg["ml"]["data_dir"]
    assert artifact_dir(cfg) == ROOT / cfg["ml"]["artifact_dir"]
    monkeypatch.setenv("HAI_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "a"))
    assert hai_data_dir(cfg) == tmp_path / "d" and artifact_dir(cfg) == tmp_path / "a"


def test_no_absolute_paths_in_tracked_code():
    """Portability: no absolute paths into the user's home directory in code or config."""
    files = [p for d in ("src", "scripts", "app", "cluster", "config") for p in (ROOT / d).rglob("*")
             if p.suffix in (".py", ".yaml", ".sh", ".sbatch", ".j2") and "__pycache__" not in p.parts]
    assert len(files) > 20
    bad = [str(p.relative_to(ROOT)) for p in files
           if "/home/" in p.read_text(encoding="utf-8") or "\u684c\u9762" in p.read_text(encoding="utf-8")]
    assert bad == []
