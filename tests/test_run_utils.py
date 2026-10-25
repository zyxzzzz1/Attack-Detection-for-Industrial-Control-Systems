"""C0: tests of the shared run-environment code (config overrides, hashing, manifest, thread / worker counts, random seeds) and the run_model entry point."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src import run_model
from src.utils import run as run_utils
from src.utils.paths import load_config
from src.utils.run import (RunManifest, ablation_grid, apply_overrides, config_hash, git_commit, resolve_num_workers,
                           seed_everything)

MANIFEST_REQUIRED = {  # Fields required by "Portability" in CLAUDE.md and by C0
    "git_commit", "config_hash", "hyperparameters", "python", "versions", "device", "device_name", "hostname",
    "started_at", "finished_at", "final_epoch", "early_stop_epoch", "slurm_job_id", "slurm_array_task_id", "node",
    "gpu_nvidia_smi", "segments", "status", "smoke", "seed",
}


def test_apply_overrides_parses_yaml_and_does_not_mutate():
    cfg = {"ml": {"window": {"length": 60}, "models": {"lstm_ae": {}}}}
    out = apply_overrides(cfg, ["ml.window.length=30", "ml.models.lstm_ae.hidden=128", "ml.x.flag=true", "ml.name=abc"])
    assert out["ml"]["window"]["length"] == 30 and out["ml"]["models"]["lstm_ae"]["hidden"] == 128
    assert out["ml"]["x"]["flag"] is True and out["ml"]["name"] == "abc"
    assert cfg == {"ml": {"window": {"length": 60}, "models": {"lstm_ae": {}}}}
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["ml.window.length"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["ml.window.length.x=1"])


def test_config_hash_ignores_key_order_and_tracks_content():
    a = {"x": 1, "y": {"b": 2, "a": [1, 2]}}
    b = {"y": {"a": [1, 2], "b": 2}, "x": 1}
    assert config_hash(a) == config_hash(b) and len(config_hash(a)) == 64
    assert config_hash(a) != config_hash(apply_overrides(a, ["x=2"]))


def test_ablation_grid_is_3x3_with_fixed_numbering():
    cfg = load_config()
    grid = ablation_grid(cfg)
    assert [(g["window"], g["hidden"]) for g in grid] == [(w, h) for w in (30, 60, 120) for h in (32, 64, 128)]
    assert [g["index"] for g in grid] == list(range(9))
    assert grid[4]["run_name"] == "ablation/lstm_ae_w60_h64"  # Task 4 = main-model configuration
    assert grid[8]["overrides"] == ["ml.window.length=120", "ml.models.lstm_ae.hidden=128"]
    # The keys targeted by overrides really exist in the config structure
    out = apply_overrides(cfg, grid[0]["overrides"])
    assert out["ml"]["window"]["length"] == 30 and out["ml"]["models"]["lstm_ae"]["hidden"] == 32


def test_num_workers_capped_by_slurm_cpus(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert run_utils.available_cpus() == 4
    assert resolve_num_workers(16) == 4 and resolve_num_workers(2) == 2 and resolve_num_workers(0) == 0
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert resolve_num_workers(10_000) == run_utils.available_cpus()


def test_seed_everything_is_reproducible_and_sets_cudnn_flags():
    import random
    seed_everything(42)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(42)
    assert a == (random.random(), np.random.rand(), torch.rand(1).item())
    assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False


def test_git_commit_falls_back_to_file(tmp_path):
    assert git_commit(tmp_path) == {"commit": None, "dirty": None, "source": None}
    (tmp_path / ".git_commit").write_text("abc123 dirty\n", encoding="utf-8")
    assert git_commit(tmp_path) == {"commit": "abc123", "dirty": True, "source": ".git_commit"}
    (tmp_path / ".git_commit").write_text("abc123\n", encoding="utf-8")
    assert git_commit(tmp_path)["dirty"] is False


def test_manifest_fields_slurm_and_resume_segments(tmp_path, monkeypatch):
    cfg = load_config()
    monkeypatch.setenv("SLURM_JOB_ID", "1001")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "4")
    monkeypatch.setenv("SLURMD_NODENAME", "gpu-node-07")
    monkeypatch.setattr(run_utils, "nvidia_smi", lambda: ["NVIDIA L4, 23034 MiB, 550.90"])
    m = RunManifest(tmp_path)
    m.start(cfg, model="lstm_ae", run_name="lstm_ae", smoke=False, overrides=["ml.window.length=60"], argv=["--resume"])
    assert MANIFEST_REQUIRED <= set(m.data)
    assert (m.data["slurm_job_id"], m.data["slurm_array_task_id"], m.data["node"]) == ("1001", "4", "gpu-node-07")
    assert m.data["gpu_nvidia_smi"] == ["NVIDIA L4, 23034 MiB, 550.90"]
    assert m.data["config_hash"] == config_hash(cfg) and m.data["hyperparameters"] == cfg["ml"]
    assert set(m.data["versions"]) == {"torch", "sklearn", "numpy", "pandas"}
    assert m.data["status"] == "running" and m.data["finished_at"] is None
    m.update(final_epoch=7)
    m.finish("interrupted")  # The first job segment is killed at its time limit

    # The second job segment (resubmitted, resuming): keeps the first start time and the recorded epochs, with one record per segment
    monkeypatch.setenv("SLURM_JOB_ID", "1002")
    m2 = RunManifest(tmp_path)
    m2.start(cfg, model="lstm_ae", run_name="lstm_ae", smoke=False, overrides=[], argv=["--resume"])
    m2.update(final_epoch=23, early_stop_epoch=18)
    m2.stage("train", seconds=1.0)
    m2.finish("completed")
    saved = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved["started_at"] == m.data["started_at"] and saved["status"] == "completed"
    assert [s["slurm"]["job_id"] for s in saved["segments"]] == ["1001", "1002"]
    assert [s["status"] for s in saved["segments"]] == ["interrupted", "completed"]
    assert (saved["final_epoch"], saved["early_stop_epoch"]) == (23, 18) and "train" in saved["stages"]


# ---------- run_model entry point (synthetic data) ----------

@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    """Write 8 tiny files under the real file names and point HAI_DATA_DIR / ARTIFACT_DIR at them."""
    cfg = load_config()
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()
    for k, name in enumerate(cfg["ml"]["hai"]["train_files"] + cfg["ml"]["hai"]["test_files"]):
        n = 400
        df = pd.DataFrame({"time": [str(t) for t in pd.date_range(f"2020-01-{k + 1:02d}", periods=n, freq="s")],
                           "P1_A": rng.normal(size=n), "P2_B": rng.normal(size=n), "P3_C": 1.0})
        for c in cfg["ml"]["hai"]["label_cols"]:
            df[c] = 0
        df.to_csv(data / name, index=False)
    monkeypatch.setenv("HAI_DATA_DIR", str(data))
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    return tmp_path / "artifacts"


def test_run_model_prepare_writes_manifest_scaler_and_log(tiny_data):
    assert run_model.main(["--model", "pca", "--stages", "prepare", "--set", "ml.data.drop_train_overlap=false"]) == 0
    d = tiny_data / "pca"
    assert {"run_manifest.json", "run.log", "scaler.joblib", "preprocess.json"} <= {p.name for p in d.iterdir()}
    m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == "completed" and m["smoke"] is False and MANIFEST_REQUIRED <= set(m)
    assert m["overrides"] == ["ml.data.drop_train_overlap=false"]
    assert m["hyperparameters"]["data"]["drop_train_overlap"] is False
    assert m["stages"]["prepare"]["rows"] == {"train": 960, "val_es": 120, "val_thr": 120}
    assert m["stages"]["prepare"]["n_features"] == 2
    assert "阶段 prepare 完成" in (d / "run.log").read_text(encoding="utf-8")


def test_run_model_ablation_index_sets_run_name_and_overrides(tiny_data):
    assert run_model.main(["--ablation-index", "0", "--set", "ml.data.drop_train_overlap=false", "--resume"]) == 0
    m = json.loads((tiny_data / "ablation" / "lstm_ae_w30_h32" / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["model"] == "lstm_ae" and m["hyperparameters"]["window"]["length"] == 30
    assert m["hyperparameters"]["models"]["lstm_ae"]["hidden"] == 32
    # Different combinations have different config hashes
    assert run_model.main(["--ablation-index", "1", "--set", "ml.data.drop_train_overlap=false"]) == 0
    m1 = json.loads((tiny_data / "ablation" / "lstm_ae_w30_h64" / "run_manifest.json").read_text(encoding="utf-8"))
    assert m1["config_hash"] != m["config_hash"]


def test_run_model_failure_is_recorded_in_manifest(tiny_data):
    # Window longer than every segment: the prepare stage fails and the manifest records failed
    with pytest.raises(RuntimeError, match="窗口"):
        run_model.main(["--model", "pca", "--set", "ml.data.drop_train_overlap=false", "--set", "ml.window.length=5000"])
    m = json.loads((tiny_data / "pca" / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == "failed" and m["segments"][-1]["status"] == "failed"


def test_run_model_rejects_unknown_model_and_stage(tiny_data):
    for argv in (["--model", "nope"], ["--model", "pca", "--stages", "nope"], ["--ablation-index", "9"], []):
        with pytest.raises(SystemExit):
            run_model.main(argv)


def test_smoke_flag_limits_data_and_epochs(tiny_data):
    assert run_model.main(["--model", "lstm_ae", "--smoke", "--set", "ml.data.drop_train_overlap=false",
                           "--set", "ml.smoke.fraction=0.5", "--set", "ml.window.length=10"]) == 0
    m = json.loads((tiny_data / "lstm_ae" / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["smoke"] is True and m["hyperparameters"]["train"]["max_epochs"] == 1
    assert sum(m["stages"]["prepare"]["rows"].values()) == 600
