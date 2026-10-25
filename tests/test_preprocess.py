"""Tests of M1 preprocessing and windowing. All use small hand-made CSVs except the last group (real data, skipped when the data is absent)."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.data.load_hai import load_test_segments, load_train_segments, read_segment, split_segments
from src.data.preprocess import Preprocessor, prepare_test, prepare_train
from src.data.windowing import WindowDataset
from src.utils.paths import hai_data_dir, load_config
from src.utils.rowhash import hash_rows

LABELS = ["attack", "attack_P1", "attack_P2", "attack_P3"]
W = 5


def write_file(path, start, n, gid0, const=7.0, late_only=0.0, scale=1.0):
    """gid = global row number (a disjoint range per file), used to identify where the rows in a window came from."""
    gid = np.arange(gid0, gid0 + n, dtype=float)
    df = pd.DataFrame({
        "time": [str(t) for t in pd.date_range(start, periods=n, freq="s")],
        "P1_GID": gid,
        "P1_NOISE": np.sin(gid) * scale,
        "P2_CONST": const,
        "P3_LATE": late_only,
    })
    for c in LABELS:
        df[c] = 1  # Labels are all 1: if they were read into the features, the tests would notice
    df.to_csv(path, index=False)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Three train files (100 / 60 / 240 rows) and two test files, with time gaps between files."""
    monkeypatch.delenv("HAI_DATA_DIR", raising=False)
    write_file(tmp_path / "train1.csv", "2020-01-01", 100, 0)
    write_file(tmp_path / "train2.csv", "2020-01-05", 60, 1000)
    write_file(tmp_path / "train3.csv", "2020-01-09", 240, 2000)
    write_file(tmp_path / "test1.csv", "2020-02-01", 30, 9000, scale=50.0)
    write_file(tmp_path / "test2.csv", "2020-02-02", 3, 9500)
    c = copy.deepcopy(load_config())
    c["ml"]["data_dir"] = str(tmp_path)
    c["ml"]["hai"]["train_files"] = ["train1.csv", "train2.csv", "train3.csv"]
    c["ml"]["hai"]["test_files"] = ["test1.csv", "test2.csv"]
    c["ml"]["data"] = {"val_fraction": 0.2, "val_es_fraction": 0.5, "drop_train_overlap": False,
                       "train_overlap_drops": [{"file": "train1.csv", "head_rows": 2, "tail_rows": 10}]}
    return c


def raw_split(cfg):
    d = cfg["ml"]["data"]
    return split_segments(load_train_segments(cfg), d["val_fraction"], d["val_es_fraction"])


# ---------- Reading and splitting ----------

def test_reader_drops_time_and_label_columns(cfg):
    seg = load_train_segments(cfg)[0]
    assert seg.columns == ["P1_GID", "P1_NOISE", "P2_CONST", "P3_LATE"]
    assert seg.x.dtype == np.float32 and seg.x.shape == (100, 4)
    assert seg.start_time == "2020-01-01 00:00:00"


def test_reader_rejects_file_with_missing_seconds(tmp_path):
    df = pd.DataFrame({"time": ["2020-01-01 00:00:00", "2020-01-01 00:00:01", "2020-01-01 00:00:05"], "P1_A": [1, 2, 3]})
    df.to_csv(tmp_path / "gap.csv", index=False)
    with pytest.raises(ValueError, match="逐秒连续"):
        read_segment(tmp_path / "gap.csv", "time", LABELS, "train")


def test_split_is_chronological_last_20_percent_halved(cfg):
    raw = raw_split(cfg)
    rows = {k: sum(len(s) for s in v) for k, v in raw.items()}
    assert rows == {"train": 320, "val_es": 40, "val_thr": 40}
    # Time order train | val_es | val_thr, not shuffled: global row numbers strictly increase
    gid = np.concatenate([s.x[:, 0] for k in ("train", "val_es", "val_thr") for s in raw[k]])
    assert (np.diff(gid) > 0).all()
    assert raw["train"][-1].x[-1, 0] < raw["val_es"][0].x[0, 0] < raw["val_thr"][0].x[0, 0]
    # The split point falls inside train3: train3 is cut into three segments, and their start rows and start times are updated accordingly
    assert [s.name for s in raw["train"]] == ["train1.csv", "train2.csv", "train3.csv"]
    assert (raw["val_es"][0].start_row, raw["val_es"][0].start_time) == (160, "2020-01-09 00:02:40")
    assert (raw["val_thr"][0].start_row, raw["val_thr"][0].start_time) == (200, "2020-01-09 00:03:20")


def test_split_cut_on_file_boundary_and_across_files():
    from src.data.load_hai import Segment
    mk = lambda name, n, g: Segment(name, "train", 0, "2020-01-01 00:00:00", ["g"],  # noqa: E731
                                    np.arange(g, g + n, dtype=np.float32)[:, None])
    # The validation set spans two files: val_es consists of the tail of a and the head of b, and the file boundary is still a segment boundary
    out = split_segments([mk("a", 60, 0), mk("b", 40, 100)], val_fraction=0.6, val_es_fraction=0.5)
    assert [(s.name, len(s)) for s in out["train"]] == [("a", 40)]
    assert [(s.name, len(s)) for s in out["val_es"]] == [("a", 20), ("b", 10)]
    assert [(s.name, len(s)) for s in out["val_thr"]] == [("b", 30)]


# ---------- Windowing ----------

@pytest.mark.parametrize("stride", [1, 3, 10])
def test_windows_never_cross_any_boundary(cfg, stride):
    """File boundaries, the train/val split point, the split point inside val: the rows of any window must come from the same segment and be consecutive."""
    raw = raw_split(cfg)
    seen = {}
    for split, segs in raw.items():
        ds = WindowDataset([s.x for s in segs], W, stride)
        assert len(ds) == sum(max(0, (len(s) - W) // stride + 1) for s in segs)
        allowed = [set(s.x[:, 0].tolist()) for s in segs]
        for i in range(len(ds)):
            gid = ds[i][:, 0].numpy()
            assert gid.shape == (W,)
            assert (np.diff(gid) == 1).all(), f"{split} 窗口 {i} 跨越了边界"
            assert any(set(gid.tolist()) <= a for a in allowed)
        seen[split] = {g for i in range(len(ds)) for g in ds[i][:, 0].tolist()}
    # Windows of the three parts never share a row
    assert not (seen["train"] & seen["val_es"]) and not (seen["val_es"] & seen["val_thr"])
    assert not (seen["train"] & seen["val_thr"])


def test_get_batch_matches_getitem_and_iter_batches_covers_all(cfg):
    segs = [s.x for s in raw_split(cfg)["train"]]
    ds = WindowDataset(segs, W, 3)
    idx = np.array([0, len(ds) - 1, 31, 32, 33, 5])  # Crossing segments around 31/32/33
    batch = ds.get_batch(idx)
    assert batch.shape == (6, W, 4) and batch.dtype.is_floating_point
    for k, i in enumerate(idx):
        assert (batch[k] == ds[int(i)]).all()
    got = np.concatenate([i for i, _ in ds.iter_batches(16, shuffle=True, generator=np.random.default_rng(0))])
    assert sorted(got.tolist()) == list(range(len(ds)))
    a = [i.tolist() for i, _ in ds.iter_batches(16, shuffle=True, generator=np.random.default_rng(42))]
    b = [i.tolist() for i, _ in ds.iter_batches(16, shuffle=True, generator=np.random.default_rng(42))]
    assert a == b


def test_windows_are_views_not_expanded(cfg):
    x = raw_split(cfg)["train"][0].x
    ds = WindowDataset([x], W, 1)
    assert np.shares_memory(ds[3].numpy(), ds.segments[0])
    assert ds._views[0].base is not None


def test_first_w_minus_1_seconds_are_nan_per_file(cfg):
    """The first W−1 seconds of each test file are NaN; scores are assigned to the window's last second; files shorter than one window are all NaN."""
    test = load_test_segments(cfg)
    ds = WindowDataset([s.x for s in test], W, 1)
    assert len(ds) == 30 - W + 1  # test2 has only 3 rows, so no windows
    window_last_gid = np.array([ds[i][-1, 0].item() for i in range(len(ds))])
    per_file = ds.scatter(window_last_gid)
    assert [len(a) for a in per_file] == [30, 3]
    assert np.isnan(per_file[0][:W - 1]).all() and not np.isnan(per_file[0][W - 1:]).any()
    assert (per_file[0][W - 1:] == test[0].x[W - 1:, 0]).all()
    assert np.isnan(per_file[1]).all()
    # Same for per-sensor errors, shape (T, D)
    per_sensor = ds.scatter(np.ones((len(ds), 4)))
    assert per_sensor[0].shape == (30, 4) and np.isnan(per_sensor[0][:W - 1]).all()
    assert int(np.isnan(np.concatenate(per_file)).sum()) == (W - 1) + 3
    with pytest.raises(ValueError):
        ds.scatter(np.zeros(len(ds) + 1))


def test_scatter_with_stride_leaves_unscored_seconds_nan():
    ds = WindowDataset([np.zeros((12, 2), dtype=np.float32)], 4, 3)
    out = ds.scatter(np.arange(len(ds), dtype=float))[0]
    assert np.flatnonzero(~np.isnan(out)).tolist() == [3, 6, 9]


# ---------- Preprocessing ----------

def test_constant_columns_removed(cfg):
    data = prepare_train(cfg)
    assert data.pre.constant_cols == ["P2_CONST", "P3_LATE"]
    assert data.pre.features == ["P1_GID", "P1_NOISE"]
    assert all(s.x.shape[1] == 2 and s.x.dtype == np.float32 for v in data.segments.values() for s in v)
    assert all(s.columns == ["P1_GID", "P1_NOISE"] for s in prepare_test(cfg, data.pre))


def test_column_constant_per_file_but_different_across_files_is_kept(cfg, tmp_path):
    write_file(tmp_path / "train2.csv", "2020-01-05", 60, 1000, const=8.0)
    assert "P2_CONST" in prepare_train(cfg).pre.features


def test_constant_columns_decided_on_train_part_only(cfg, tmp_path):
    # P3_LATE changes only in the last 20 rows of train3 (inside val_thr): judged on the training part it is still constant and must be dropped
    df = pd.read_csv(tmp_path / "train3.csv")
    df.loc[df.index[-20:], "P3_LATE"] = np.arange(20)
    df.to_csv(tmp_path / "train3.csv", index=False)
    assert "P3_LATE" in prepare_train(cfg).pre.constant_cols


def test_scaler_fit_on_train_part_only(cfg, tmp_path):
    # Amplify the noise of the validation part (last 80 rows of train3) 1000x: the scaler statistics must not be affected
    df = pd.read_csv(tmp_path / "train3.csv")
    df.loc[df.index[-80:], "P1_NOISE"] *= 1000
    df.to_csv(tmp_path / "train3.csv", index=False)
    raw = raw_split(cfg)
    data = prepare_train(cfg)
    train_raw = np.concatenate([s.x for s in raw["train"]]).astype(np.float64)
    assert data.pre.n_fit_rows == 320
    np.testing.assert_allclose(data.pre.scaler.mean_, train_raw[:, :2].mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(data.pre.scaler.scale_, train_raw[:, :2].std(axis=0), rtol=1e-6)
    train_scaled = np.concatenate([s.x for s in data.segments["train"]])
    np.testing.assert_allclose(train_scaled.mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(train_scaled.std(axis=0), 1, atol=1e-4)
    val_scaled = np.concatenate([s.x for s in data.segments["val_es"] + data.segments["val_thr"]])
    assert val_scaled[:, 1].std() > 100  # The validation part did not take part in fit, so the amplified noise shows up unchanged
    assert prepare_test(cfg, data.pre)[0].x[:, 1].std() > 10


def test_preprocessor_refuses_non_train_segments(cfg):
    raw = raw_split(cfg)
    with pytest.raises(ValueError, match="train"):
        Preprocessor().fit(raw["train"] + raw["val_es"])
    with pytest.raises(ValueError, match="train"):
        Preprocessor().fit(load_test_segments(cfg))


def test_preprocessor_roundtrip_and_nan_rejected(cfg, tmp_path):
    data = prepare_train(cfg)
    data.pre.save(tmp_path / "out" / "scaler.joblib")
    loaded = Preprocessor.load(tmp_path / "out" / "scaler.joblib")
    raw = load_test_segments(cfg)[0]
    assert (loaded.transform(raw).x == data.pre.transform(raw).x).all()
    raw.x[3, 0] = np.nan
    with pytest.raises(ValueError, match="缺失"):
        loaded.transform(raw)


# ---------- Dropping rows that overlap test ----------

def test_overlap_drop_switch(cfg):
    off = load_train_segments(cfg)[0]
    cfg["ml"]["data"]["drop_train_overlap"] = True
    on = load_train_segments(cfg)[0]
    assert len(off) == 100 and len(on) == 88
    assert (on.start_row, on.start_time) == (2, "2020-01-01 00:00:02")
    assert on.x[0, 0] == 2 and on.x[-1, 0] == 89
    assert sum(len(s) for v in raw_split(cfg).values() for s in v) == 388


def test_constants_and_scaler_computed_after_drop(cfg, tmp_path):
    # P3_LATE changes only in the last 10 rows of train1, exactly the rows dropped: afterwards it is constant and the scaler never saw those rows
    df = pd.read_csv(tmp_path / "train1.csv")
    df.loc[df.index[-10:], ["P3_LATE", "P1_NOISE"]] = 1e6
    df.to_csv(tmp_path / "train1.csv", index=False)
    assert "P3_LATE" in prepare_train(cfg).pre.features
    cfg["ml"]["data"]["drop_train_overlap"] = True
    data = prepare_train(cfg)
    assert "P3_LATE" in data.pre.constant_cols
    assert data.pre.scaler.scale_[1] < 10


def test_fraction_keeps_leading_rows_of_each_file(cfg):
    segs = load_train_segments(cfg, fraction=0.5)
    assert [len(s) for s in segs] == [50, 30, 120]
    assert [s.x[0, 0] for s in segs] == [0, 1000, 2000]
    assert [len(s) for s in load_test_segments(cfg, fraction=0.5)] == [15, 1]


# ---------- Real data (runs when the full data is available locally / on the cluster) ----------

REAL = load_config()
HAVE_REAL = all((hai_data_dir(REAL) / f).exists()
                for f in REAL["ml"]["hai"]["train_files"] + REAL["ml"]["hai"]["test_files"])


@pytest.fixture(scope="module")
def real_hashes():
    """Per-row content hashes of the real data (raw values of the 79 sensors, read as float32 then converted to float64)."""
    test = {s.name: hash_rows(pd.DataFrame(s.x)) for s in load_test_segments(REAL)}
    cfg_off = copy.deepcopy(REAL)
    cfg_off["ml"]["data"]["drop_train_overlap"] = False
    return test, cfg_off


@pytest.mark.skipif(not HAVE_REAL, reason="没有完整的 HAI 数据")
def test_real_training_windows_contain_no_test_rows(real_hashes):
    """By content hash (ignoring timestamps): no window of train / val_es / val_thr contains any row of any test file,
    in particular the first 43,201 seconds of test3."""
    test, cfg_off = real_hashes
    assert REAL["ml"]["data"]["drop_train_overlap"] is True  # On by default
    all_test = np.concatenate(list(test.values()))
    test3_head = test["test3.csv.gz"][:43201]
    d, w = REAL["ml"]["data"], REAL["ml"]["window"]
    raw = split_segments(load_train_segments(REAL), d["val_fraction"], d["val_es_fraction"])
    total = 0
    for split, segs in raw.items():
        ds = WindowDataset([s.x for s in segs], w["length"], w["train_stride"] if split == "train" else w["eval_stride"])
        assert len(ds) > 0
        for s, x in zip(segs, ds.segments):  # Windows are sliced only from these arrays, so checking row by row covers all windows
            h = hash_rows(pd.DataFrame(x))
            assert not np.isin(h, test3_head).any(), f"{split}/{s.name} 含 test3 前 43,201 秒的行"
            assert not np.isin(h, all_test).any(), f"{split}/{s.name} 含 test 文件的行"
            total += len(h)
    assert total == 921603 - 43201 - 1
    assert raw["train"][0].start_row == 1 and len(raw["train"][0]) == 216001 - 43201 - 1

    # With the switch off these 43,202 rows really are found, showing the assertion above is not vacuous
    train1_off = load_train_segments(cfg_off)[0]
    h = hash_rows(pd.DataFrame(train1_off.x))
    assert int(np.isin(h, test3_head).sum()) == 43201
    assert int(np.isin(h, all_test).sum()) == 43202


@pytest.mark.skipif(not HAVE_REAL, reason="没有完整的 HAI 数据")
def test_real_split_sizes_and_features():
    data = prepare_train(REAL)
    rows = data.summary()["rows"]
    assert rows == {"train": 702721, "val_es": 87840, "val_thr": 87840}
    assert {s.name for s in data.segments["val_es"] + data.segments["val_thr"]} == {"train3.csv.gz"}
    assert len(data.pre.features) == 57 and len(data.pre.constant_cols) == 22
    assert data.pre.n_fit_rows == 702721
    assert all(s.x.dtype == np.float32 for v in data.segments.values() for s in v)
