"""M2a: PCA / IsolationForest baselines, diagnosis, threshold, and an end-to-end run of all run_model stages on synthetic data."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts import make_main_report
from src import run_model
from src.data.windowing import WindowDataset
from src.diagnose import sensor_contributions
from src.models import build_model
from src.models.classic import window_stats
from src.threshold import compute_threshold
from src.utils.paths import load_config

N, W = 3000, 20
SENSORS = ["P1_A", "P1_B", "P2_A", "P2_B", "P3_A", "P4_A"]
# (file, start row, end row, affected sensors, subsystem labels set to 1)
EVENTS = [("test1.csv.gz", 1000, 1049, ["P1_A"], ["attack_P1"]),
          ("test2.csv.gz", 500, 529, ["P1_B", "P2_A"], ["attack_P1", "attack_P2"])]
# The synthetic data has only 6 sensors and 2 latent variables: PCA keeps 80% of the variance (about 2 components) and the diagnosis limit is relaxed (a separate test tightens it)
OVERRIDES = ["--set", "ml.data.drop_train_overlap=false", "--set", f"ml.window.length={W}",
             "--set", "ml.models.pca.variance=0.8", "--set", "ml.diagnosis.max_sensor_share=0.99"]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def hai_dir(tmp_path_factory, cfg):
    """Synthetic HAI directory: 6 correlated sensors + 1 constant column; one event injected into each of test1 and test2."""
    root = tmp_path_factory.mktemp("hai")
    rng = np.random.default_rng(0)
    mix = rng.normal(size=(2, len(SENSORS)))
    hai = cfg["ml"]["hai"]
    for k, name in enumerate(hai["train_files"] + hai["test_files"]):
        x = rng.normal(size=(N, 2)) @ mix + 0.5 * rng.normal(size=(N, len(SENSORS)))
        df = pd.DataFrame(x, columns=SENSORS)
        df.insert(0, "time", [str(t) for t in pd.date_range(f"2020-01-{k + 1:02d}", periods=N, freq="s")])
        df["P1_CONST"] = 1.0
        for c in hai["label_cols"]:
            df[c] = 0
        for file, a, b, sensors, sub_cols in EVENTS:
            if file == name:
                df.loc[a:b, sensors] += 12.0
                df.loc[a:b, [hai["label_cols"][0], *sub_cols]] = 1
        df.to_csv(root / name, index=False)
    return root


@pytest.fixture
def env(monkeypatch, hai_dir, tmp_path):
    monkeypatch.setenv("HAI_DATA_DIR", str(hai_dir))
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    return tmp_path / "artifacts"


def toy_dataset(stride=1, seed=1):
    rng = np.random.default_rng(seed)
    return WindowDataset([rng.normal(size=(50, 4)).astype(np.float32), rng.normal(size=(5, 4)).astype(np.float32),
                          rng.normal(size=(31, 4)).astype(np.float32)], window=8, stride=stride)


# ---------- Models ----------

@pytest.mark.parametrize("stride", [1, 3])
def test_pca_window_score_is_mean_row_mse_and_last_second_error(cfg, stride):
    model = build_model("pca", cfg)
    info = model.fit(toy_dataset(10), None)
    assert info["n_fit_rows"] == 86 and 1 <= info["n_components"] <= 4
    ds = toy_dataset(stride, seed=2)
    scores, per_sensor = model.score(ds)
    assert scores.shape == (len(ds),) and per_sensor.shape == (len(ds), 4)
    for i in (0, len(ds) // 2, len(ds) - 1):  # Matches the result computed directly per window
        err = model.row_errors(ds[i].numpy())
        assert scores[i] == pytest.approx(err.mean(), rel=1e-4)
        np.testing.assert_allclose(per_sensor[i], err[-1], rtol=1e-4, atol=1e-7)


def test_window_stats_hand_example():
    w = np.array([[[1.0, 10.0], [3.0, 10.0], [2.0, 4.0]]])  # (1, W=3, D=2)
    np.testing.assert_allclose(window_stats(w)[0], [2, 8, np.std([1, 3, 2]), np.std([10, 10, 4]), 1, 4, 3, 10, 1, -6],
                               rtol=1e-6)


def test_iforest_scores_are_deterministic_and_flag_shifted_windows(cfg):
    train = toy_dataset(2)
    a, b = build_model("iforest", cfg), build_model("iforest", cfg)
    assert a.fit(train, None)["n_window_features"] == 20
    b.fit(train, None)
    normal = toy_dataset(1, seed=3)
    shifted = WindowDataset([s + 6 for s in normal.segments], 8, 1)
    s_normal, err = a.score(normal)
    assert err is None and s_normal.shape == (len(normal),)
    np.testing.assert_array_equal(s_normal, b.score(normal)[0])  # Fixed seed
    assert a.score(shifted)[0].mean() > s_normal.mean()


def test_iforest_train_subsampling(cfg):
    c = json.loads(json.dumps(cfg))
    c["ml"]["models"]["iforest"]["max_train_windows"] = 10
    assert build_model("iforest", c).fit(toy_dataset(1), None)["n_fit_windows"] == 10


# ---------- Diagnosis and threshold ----------

def test_sensor_contributions_ignore_unscored_rows():
    err = np.array([[np.nan, np.nan], [1.0, 3.0], [3.0, 9.0]])
    df = sensor_contributions(err, ["P1_X", "P2_Y"])
    assert df["sensor"].tolist() == ["P2_Y", "P1_X"] and df["subsystem"].tolist() == ["P2", "P1"]
    assert df["share"].tolist() == pytest.approx([0.75, 0.25]) and df["mean_sq_err"].tolist() == pytest.approx([6, 2])


def test_threshold_is_quantile_of_scored_val_thr_seconds():
    scores = np.concatenate(([np.nan] * 5, np.arange(101, dtype=float)))
    t = compute_threshold(scores, 0.99, [0.99, 0.999])
    assert t["threshold"] == pytest.approx(99.0) and (t["n_scored"], t["n_unscored"]) == (101, 5)
    assert t["sensitivity"]["0.999"] == pytest.approx(99.9) and t["score_stats"]["max"] == 100
    with pytest.raises(ValueError):
        compute_threshold(np.array([np.nan]), 0.99, [])


# ---------- End to end ----------

def test_pca_end_to_end(env, cfg):
    assert run_model.main(["--model", "pca", *OVERRIDES]) == 0
    d = env / "pca"
    manifest = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and list(manifest["stages"]) == list(run_model.STAGES)

    scores = np.load(d / "scores_test.npy")
    err = np.load(d / "per_sensor_err_test.npy")
    index = json.loads((d / "test_index.json").read_text(encoding="utf-8"))
    assert scores.shape == (5 * N,) and err.shape == (5 * N, len(SENSORS)) and index["features"] == SENSORS
    # The first W−1 seconds of each test file are NaN; everything else has a score
    unscored = np.isnan(scores).reshape(5, N)
    assert unscored[:, :W - 1].all() and not unscored[:, W - 1:].any()
    np.testing.assert_array_equal(np.isnan(err).all(axis=1), np.isnan(scores))

    # The threshold comes only from the 99% quantile of the val_thr scores
    val = np.load(d / "scores_val_thr.npy")
    thr = json.loads((d / "threshold.json").read_text(encoding="utf-8"))
    assert len(val) == 900 and np.isnan(val).sum() == W - 1
    assert thr["threshold"] == pytest.approx(np.quantile(val[~np.isnan(val)], 0.99)) and thr["source"] == "val_thr"

    m = json.loads((d / "eval_metrics.json").read_text(encoding="utf-8"))
    assert m["point"]["n_unscored"] == 5 * (W - 1) and m["point"]["n_positive"] == 80 and m["point_adjust"] is False
    assert m["event"]["n_events"] == 2 and m["event"]["event_recall"] == 1.0 and m["event"]["n_events_without_subsystem"] == 0
    assert m["point"]["auroc"] > 0.9
    assert list(m["main_row"]) == make_main_report.MAIN_COLUMNS and m["main_row"]["threshold"] == thr["threshold"]
    assert [f["file"] for f in m["per_file"]] == cfg["ml"]["hai"]["test_files"]

    events = pd.read_csv(d / "eval_events.csv")
    assert events["subsystems"].tolist() == ["P1", "P1+P2"] and events["file"].tolist() == ["test1.csv.gz", "test2.csv.gz"]
    assert events["start_time"].tolist() == ["2020-01-04 00:16:40", "2020-01-05 00:08:20"]
    alarms = pd.read_csv(d / "eval_alarms.csv")
    assert alarms["overlaps_event"].sum() >= 2 and len(alarms) == m["event"]["n_alarm_segments"]

    contrib = pd.read_csv(d / "val_sensor_contrib.csv")
    assert sorted(contrib["sensor"]) == sorted(SENSORS) and contrib["share"].sum() == pytest.approx(1)


def test_iforest_end_to_end_and_single_stage_rerun(env):
    assert run_model.main(["--model", "iforest", *OVERRIDES]) == 0
    d = env / "iforest"
    assert not (d / "per_sensor_err_test.npy").exists() and not (d / "val_sensor_contrib.csv").exists()
    manifest = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["diagnose"]["skipped"] is True
    first = json.loads((d / "eval_metrics.json").read_text(encoding="utf-8"))
    assert first["event"]["n_events"] == 2 and first["point"]["n_unscored"] == 5 * (W - 1)
    # Stages can be rerun individually: rerunning only evaluate gives the same result (scores and threshold read from the run directory)
    assert run_model.main(["--model", "iforest", "--stages", "evaluate", *OVERRIDES]) == 0
    assert json.loads((d / "eval_metrics.json").read_text(encoding="utf-8"))["main_row"] == first["main_row"]


def test_diagnosis_over_limit_stops_before_reading_test(env):
    args = ["--model", "pca", *OVERRIDES, "--set", "ml.diagnosis.max_sensor_share=0.05"]
    assert run_model.main(args) == run_model.EXIT_STOPPED_FOR_REVIEW
    d = env / "pca"
    manifest = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "stopped_for_review" and manifest["stages"]["diagnose"]["over_limit"] is True
    assert "score" not in manifest["stages"] and (d / "val_sensor_contrib.csv").exists()
    for name in ("scores_test.npy", "scores_val_thr.npy", "test_index.json", "threshold.json", "eval_metrics.json"):
        assert not (d / name).exists()
    # Allowed through after the decision: --resume loads the trained model and continues with the remaining stages
    assert run_model.main([*args, "--set", "ml.diagnosis.allow_over_limit=true", "--resume"]) == 0
    manifest = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and manifest["stages"]["train"]["resumed"] is True
    assert manifest["stages"]["train"]["n_components"] >= 1 and (d / "eval_metrics.json").exists()


def test_smoke_run_does_not_stop_on_diagnosis(env):
    assert run_model.main(["--model", "pca", "--smoke", *OVERRIDES, "--set", "ml.smoke.fraction=0.5",
                           "--set", "ml.diagnosis.max_sensor_share=0.05"]) == 0


# ---------- Report scripts ----------

def test_main_report_rejects_unofficial_runs(env, tmp_path, capsys):
    """Local runs (usually a dirty worktree, or smoke) must never enter the report; --allow-unofficial may only write elsewhere."""
    assert run_model.main(["--model", "pca", *OVERRIDES]) == 0
    manifest_path = env / "pca" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = tmp_path / "report"

    for change, reason in (({"git_dirty": True, "smoke": False}, "git_dirty"), ({"git_dirty": False, "smoke": True}, "smoke"),
                           ({"git_dirty": False, "smoke": False, "status": "failed"}, "status")):
        manifest_path.write_text(json.dumps({**manifest, **change}), encoding="utf-8")
        assert make_main_report.main(["--models", "pca", "--output-dir", str(out)]) == 0
        assert reason in capsys.readouterr().out and not (out / "main_results.csv").exists()
        assert not (out / "val_sensor_contrib.csv").exists()

    manifest_path.write_text(json.dumps({**manifest, "git_dirty": False, "smoke": False}), encoding="utf-8")
    assert make_main_report.main(["--models", "pca", "iforest", "--output-dir", str(out)]) == 0
    table = pd.read_csv(out / "main_results.csv")
    assert list(table.columns) == make_main_report.MAIN_COLUMNS and table["model"].tolist() == ["pca"]
    assert "point-adjust" in (out / "main_results.md").read_text(encoding="utf-8")
    assert (out / "val_sensor_contrib.png").stat().st_size > 0 and (out / "val_sensor_contrib.csv").exists()

    with pytest.raises(SystemExit):
        make_main_report.main(["--allow-unofficial"])


# ---------- Smoke clips and local diagnostic runs ----------

def test_smoke_adds_event_clip_so_event_metrics_run(env):
    """Smoke: each test file contributes its first part, plus the clips described by row number in the config (here covering the test1 event with some margin on both sides)."""
    clip = "ml.smoke.extra_test_clips=[{file: test1.csv.gz, start_row: 900, rows: 300}]"
    assert run_model.main(["--model", "pca", "--smoke", *OVERRIDES, "--set", "ml.smoke.fraction=0.2", "--set", clip]) == 0
    d = env / "pca"
    index = json.loads((d / "test_index.json").read_text(encoding="utf-8"))
    assert [(f["file"], f["start_row"], f["rows"]) for f in index["files"]][:3] == [
        ("test1.csv.gz", 0, 600), ("test1.csv.gz", 900, 300), ("test2.csv.gz", 0, 600)]
    scores = np.load(d / "scores_test.npy")
    assert len(scores) == 5 * 600 + 300 and np.isnan(scores[600:600 + W - 1]).all() and not np.isnan(scores[600 + W - 1])
    m = json.loads((d / "eval_metrics.json").read_text(encoding="utf-8"))
    assert m["smoke"] is True and m["point"]["n_unscored"] == 6 * (W - 1)
    assert m["event"]["n_events"] == 2 and m["event"]["n_detected"] == 2  # The test1 event is in the clip, the test2 event is in the first 20%
    events = pd.read_csv(d / "eval_events.csv")
    assert events["start_time"].tolist() == ["2020-01-04 00:16:40", "2020-01-05 00:08:20"]


def test_full_run_ignores_smoke_clips(env):
    assert run_model.main(["--model", "pca", *OVERRIDES]) == 0
    index = json.loads((env / "pca" / "test_index.json").read_text(encoding="utf-8"))
    assert [(f["start_row"], f["rows"]) for f in index["files"]] == [(0, N)] * 5


def test_local_diagnostic_run_never_reads_test_and_is_rejected_by_report(env, tmp_path, capsys, monkeypatch):
    import src.data.preprocess as preprocess
    monkeypatch.setattr(preprocess, "load_test_segments", lambda *a, **k: pytest.fail("a local diagnostic run must not read test"))
    assert run_model.main(["--model", "pca", "--local-diagnostic", *OVERRIDES]) == 0
    d = env / "local_diagnostic" / "pca"
    manifest = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["local_diagnostic"] is True and list(manifest["stages"]) == ["prepare", "train", "diagnose"]
    assert not list(d.glob("scores_*")) and not (d / "eval_metrics.json").exists()
    contrib = pd.read_csv(d / "val_sensor_contrib.csv")
    assert {"train_std", "val_thr_absz_p99", "val_thr_absz_max", "val_thr_frac_absz_gt5"} <= set(contrib.columns)
    assert (contrib["train_std"] > 0).all() and (contrib["val_thr_absz_max"] >= contrib["val_thr_absz_p99"]).all()

    for bad in (["--stages", "prepare,score"], ["--smoke"]):
        with pytest.raises(SystemExit):
            run_model.main(["--model", "pca", "--local-diagnostic", *bad, *OVERRIDES])

    # Even when placed where an official run directory would be, with every other condition met, the report scripts refuse it
    (env / "pca").mkdir()
    (env / "pca" / "val_sensor_contrib.csv").write_text((d / "val_sensor_contrib.csv").read_text())
    (env / "pca" / "eval_metrics.json").write_text(json.dumps({"main_row": {}}))
    (env / "pca" / "run_manifest.json").write_text(json.dumps({**manifest, "git_dirty": False, "status": "completed"}))
    capsys.readouterr()
    assert make_main_report.main(["--models", "pca", "--output-dir", str(tmp_path / "report")]) == 0
    assert capsys.readouterr().out.count("local_diagnostic") == 2 and not list((tmp_path / "report").iterdir())
