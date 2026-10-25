"""M2b / M2c: structure, score definitions, resumable training, early stopping and collapse check of LSTM-AE, LSTM-VAE and Deep SVDD (synthetic data; models on the auto-detected device, CPU or GPU)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src import run_model
from src.data.windowing import WindowDataset
from src.models import MODEL_CLASSES, build_model
from src.threshold import compute_threshold
from src.utils.paths import load_config
from src.utils.run import apply_overrides, seed_everything

W, D = 12, 5
TORCH_MODELS = ["lstm_ae", "lstm_vae", "deep_svdd"]


def small_cfg(*overrides):
    return apply_overrides(load_config(), [
        "ml.models.lstm_ae.hidden=8", "ml.models.lstm_vae.hidden=8", "ml.models.lstm_vae.latent=4",
        "ml.models.deep_svdd.hidden=8", "ml.models.deep_svdd.rep_dim=4", "ml.train.max_epochs=3",
        "ml.train.batch_size=32", "ml.train.eval_batch_size=64", *overrides])


def datasets(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(900)[:, None]
    x = (np.sin(t / 7 + np.arange(D)) + 0.1 * rng.normal(size=(900, D))).astype(np.float32)
    return WindowDataset([x[:600]], W, 4), WindowDataset([x[600:750]], W, 1), WindowDataset([x[750:]], W, 1)


def fit(name, tmp_path, cfg=None, resume=False):
    seed_everything(42)
    model = build_model(name, cfg or small_cfg())
    model.bind(directory=tmp_path, resume=resume, log=model.log)
    train, val, _ = datasets()
    return model, model.fit(train, val)


def test_all_five_models_registered():
    assert list(MODEL_CLASSES) == list(load_config()["ml"]["models"])


@pytest.mark.parametrize("name", TORCH_MODELS)
def test_fit_score_save_load(name, tmp_path):
    model, info = fit(name, tmp_path)
    assert info["final_epoch"] == 3 and 1 <= info["best_epoch"] <= 3 and info["early_stop_epoch"] is None
    log = pd.read_csv(tmp_path / "train_log.csv")
    assert log["epoch"].tolist() == [1, 2, 3] and {"train_loss", "val_es_loss", "train_seconds", "val_seconds"} <= set(log.columns)
    assert log["is_best"].sum() >= 1 and log.loc[log["val_es_loss"].idxmin(), "epoch"] == info["best_epoch"]
    for f in ("last.pt", "best.pt", "epoch_001.pt", "epoch_003.pt"):  # Every epoch has a checkpoint
        assert (tmp_path / "checkpoints" / f).exists()

    _, _, test = datasets()
    scores, err = model.score(test)
    assert scores.shape == (len(test),) and np.isfinite(scores).all() and (scores >= 0).all()
    if name == "deep_svdd":
        assert err is None
    else:
        assert err.shape == (len(test), D) and (err >= 0).all()
    np.testing.assert_array_equal(scores, model.score(test)[0])  # Scoring is deterministic (the VAE does not sample when scoring)

    model.save(tmp_path)
    loaded = build_model(name, small_cfg())
    assert loaded.exists(tmp_path)
    loaded.load(tmp_path)
    np.testing.assert_allclose(loaded.score(test)[0], scores, rtol=1e-6)


def test_lstm_ae_structure_and_score_definition(tmp_path):
    model, _ = fit("lstm_ae", tmp_path)
    net = model.net
    assert net.encoder.hidden_size == 8 and net.decoder.input_size == 8 and net.out.out_features == D
    _, _, test = datasets()
    x = test.get_batch(np.arange(5)).to(model.device)
    with torch.no_grad():
        recon = net(x)
        _, (h, _) = net.encoder(x)
        expected, _ = net.decoder(h[-1].unsqueeze(1).repeat(1, W, 1))  # The last hidden state is repeated W times and fed to the decoder
        np.testing.assert_allclose(recon.cpu().numpy(), net.out(expected).cpu().numpy(), rtol=1e-5, atol=1e-6)
    scores, err = model.score(test)
    sq = ((recon - x) ** 2).cpu().numpy()
    np.testing.assert_allclose(scores[:5], sq.mean(axis=(1, 2)), rtol=1e-5)  # Score = window MSE
    np.testing.assert_allclose(err[:5], sq[:, -1, :], rtol=1e-5, atol=1e-7)  # Per-sensor squared error at the last second


def test_main_model_config_is_fixed():
    cfg = load_config()["ml"]
    assert cfg["window"]["length"] == 60 and cfg["models"]["lstm_ae"]["hidden"] == 64  # Constraint 9
    assert cfg["train"] == {"optimizer": "adam", "lr": 0.001, "batch_size": 256, "max_epochs": 50, "patience": 5,
                            "eval_batch_size": 1024}
    assert cfg["models"]["lstm_vae"]["beta"] == 1.0


def test_lstm_vae_loss_is_recon_plus_beta_kl(tmp_path):
    train, _, _ = datasets()
    x = train.get_batch(np.arange(16))
    losses = {}
    for beta in (0.0, 1.0, 3.0):
        seed_everything(1)
        model = build_model("lstm_vae", small_cfg(f"ml.models.lstm_vae.beta={beta}"))
        model._init_net(D, W)
        x = x.to(model.device)
        loss, parts = model.batch_loss(x, training=False)
        losses[beta] = (loss.item(), parts)
    recon = 0.5 * losses[0.0][1]["mse"] * W * D
    kl = losses[1.0][1]["kl"]
    assert kl > 0 and losses[0.0][0] == pytest.approx(recon, rel=1e-5)
    assert losses[1.0][0] == pytest.approx(recon + kl, rel=1e-5) and losses[3.0][0] == pytest.approx(recon + 3 * kl, rel=1e-5)
    # Samples during training, not during evaluation
    seed_everything(1)
    a = model.batch_loss(x, training=True)[0].item()
    assert a != model.batch_loss(x, training=True)[0].item()
    assert model.batch_loss(x, training=False)[0].item() == model.batch_loss(x, training=False)[0].item()


def test_deep_svdd_has_no_bias_and_fixed_center(tmp_path):
    seed_everything(42)
    model = build_model("deep_svdd", small_cfg("ml.models.deep_svdd.eps=0.1"))
    model.bind(directory=tmp_path, resume=False, log=model.log)
    train, val, test = datasets()
    model._init_net(D, W)
    names = [n for n, _ in model.net.named_parameters()]
    assert names and not any("bias" in n for n in names)
    assert not any(isinstance(m, (torch.nn.BatchNorm1d, torch.nn.LayerNorm)) for m in model.net.modules())
    model.on_fit_start(train)
    c0 = model.net.center.detach().cpu().clone()
    assert (c0.abs() >= 0.1 - 1e-7).all()  # Components close to 0 are set to ±eps
    with torch.no_grad():  # Center = mean of the initial output (components not adjusted by eps)
        mean = torch.cat([model.net(x.to(model.device)) for _, x in train.iter_batches(64)]).mean(dim=0).cpu()
    big = mean.abs() >= 0.1
    np.testing.assert_allclose(c0[big].numpy(), mean[big].numpy(), rtol=1e-4, atol=1e-6)
    assert torch.equal(torch.sign(c0[~big & (mean != 0)]), torch.sign(mean[~big & (mean != 0)]))

    model.fit(train, val)
    assert not torch.equal(model.net.center.cpu(), torch.zeros(4))
    scores, err = model.score(test)
    with torch.no_grad():
        x = test.get_batch(np.arange(4)).to(model.device)
        np.testing.assert_allclose(scores[:4], ((model.net(x) - model.net.center) ** 2).sum(dim=1).cpu().numpy(), rtol=1e-5)
    assert err is None and scores.std() > 0


def test_deep_svdd_center_survives_training(tmp_path):
    captured = {}
    cls = MODEL_CLASSES["deep_svdd"]
    seed_everything(42)
    model = cls(small_cfg())
    model.bind(directory=tmp_path, resume=False, log=model.log)
    original = model.on_fit_start
    model.on_fit_start = lambda ds: (original(ds), captured.setdefault("c", model.net.center.clone()))
    train, val, _ = datasets()
    model.fit(train, val)
    assert torch.equal(model.net.center, captured["c"])  # Not updated during training


@pytest.mark.parametrize("name", TORCH_MODELS)
def test_resume_after_interruption_matches_uninterrupted_run(name, tmp_path):
    """Interrupted after epoch 2, then resumed to the end: weights and training log are bit-identical to an uninterrupted run."""
    cfg = small_cfg("ml.train.max_epochs=4")
    straight, info_a = fit(name, tmp_path / "a", cfg)

    seed_everything(42)
    model = build_model(name, cfg)
    model.bind(directory=tmp_path / "b", resume=True, log=model.log)

    def interrupt(epoch):
        if epoch == 2:
            raise KeyboardInterrupt
    model._epoch_end_hook = interrupt
    train, val, test = datasets()
    with pytest.raises(KeyboardInterrupt):
        model.fit(train, val)
    assert pd.read_csv(tmp_path / "b" / "train_log.csv")["epoch"].tolist() == [1, 2]

    seed_everything(7)  # A new process has a different RNG state: resuming must restore from the checkpoint rather than rely on the process's initial state
    resumed, info_b = fit(name, tmp_path / "b", cfg, resume=True)
    assert info_b["final_epoch"] == 4 and info_b["best_epoch"] == info_a["best_epoch"]
    for (k, a), (_, b) in zip(straight.net.state_dict().items(), resumed.net.state_dict().items()):
        assert torch.equal(a.cpu(), b.cpu()), k
    cols = ["epoch", "train_loss", "val_es_loss", "is_best"]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "a" / "train_log.csv")[cols],
                                  pd.read_csv(tmp_path / "b" / "train_log.csv")[cols])
    np.testing.assert_array_equal(straight.score(test)[0], resumed.score(test)[0])


def test_resume_when_training_already_finished_does_not_train_again(tmp_path):
    fit("lstm_ae", tmp_path)
    model = build_model("lstm_ae", small_cfg())
    model.bind(directory=tmp_path, resume=True, log=model.log)
    model._epoch_end_hook = lambda epoch: pytest.fail("不应再训练")
    train, val, _ = datasets()
    assert model.fit(train, val)["final_epoch"] == 3


def test_early_stopping_restores_best_weights(tmp_path):
    """Learning rate large enough to diverge: the validation loss stops decreasing, training stops after patience epochs, and the final weights are those of the epoch with the best validation loss."""
    cfg = small_cfg("ml.train.max_epochs=30", "ml.train.patience=2", "ml.train.lr=0.5")
    model, info = fit("lstm_ae", tmp_path, cfg)
    log = pd.read_csv(tmp_path / "train_log.csv")
    assert info["early_stop_epoch"] == info["final_epoch"] == info["best_epoch"] + 2 < 30
    assert len(log) == info["final_epoch"] and log["val_es_loss"].idxmin() + 1 == info["best_epoch"]
    best = torch.load(tmp_path / "checkpoints" / f"epoch_{info['best_epoch']:03d}.pt", weights_only=True)["model"]
    for k, v in model.net.state_dict().items():
        assert torch.equal(v.cpu(), best[k].cpu()), k


def test_collapse_flag_from_val_thr_score_variation():
    assert compute_threshold(np.full(100, 0.37), 0.99, [])["score_cv"] < 1e-12
    assert compute_threshold(np.array([1.0, 1.0 + 1e-7] * 50), 0.99, [])["score_cv"] < 1e-3
    assert compute_threshold(np.random.default_rng(0).exponential(size=1000), 0.99, [])["score_cv"] > 0.5


# ---------- End to end (run_model, synthetic data) ----------

@pytest.fixture(scope="module")
def hai_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("hai")
    hai = load_config()["ml"]["hai"]
    rng = np.random.default_rng(0)
    n = 1500
    for k, name in enumerate(hai["train_files"] + hai["test_files"]):
        t = np.arange(n)[:, None]
        x = np.sin(t / 9 + np.arange(4)) + 0.1 * rng.normal(size=(n, 4))
        df = pd.DataFrame(x, columns=["P1_A", "P1_B", "P2_A", "P3_A"])
        df.insert(0, "time", [str(s) for s in pd.date_range(f"2020-02-{k + 1:02d}", periods=n, freq="s")])
        for c in hai["label_cols"]:
            df[c] = 0
        if name == "test1.csv.gz":
            df.loc[700:759, ["P1_A", "P1_B"]] += 6.0
            df.loc[700:759, [hai["label_cols"][0], "attack_P1"]] = 1
        df.to_csv(root / name, index=False)
    return root


E2E = ["--set", "ml.data.drop_train_overlap=false", "--set", "ml.window.length=12", "--set", "ml.train.max_epochs=2",
       "--set", "ml.models.lstm_ae.hidden=8", "--set", "ml.models.lstm_vae.hidden=8", "--set", "ml.models.deep_svdd.hidden=8"]


@pytest.mark.parametrize("name", TORCH_MODELS)
def test_run_model_end_to_end(name, hai_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("HAI_DATA_DIR", str(hai_dir))
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    assert run_model.main(["--model", name, "--resume", *E2E]) == 0
    d = tmp_path / name
    manifest = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and list(manifest["stages"]) == list(run_model.STAGES)
    assert manifest["final_epoch"] == 2 and manifest["early_stop_epoch"] is None
    assert manifest["stages"]["train"]["epoch_seconds_mean"] >= 0
    assert manifest["stages"]["diagnose"].get("gated", False) is False  # Only the PCA diagnosis stops the run
    for f in ("model.pt", "train_log.csv", "scores_val_thr.npy", "scores_test.npy", "threshold.json", "eval_metrics.json"):
        assert (d / f).exists(), f
    assert (d / "per_sensor_err_test.npy").exists() == (name != "deep_svdd")
    scores = np.load(d / "scores_test.npy")
    assert np.isnan(scores).reshape(5, 1500)[:, :11].all() and np.isnan(scores).sum() == 5 * 11
    thr = json.loads((d / "threshold.json").read_text(encoding="utf-8"))
    assert thr["collapsed"] is False and thr["score_stats"]["var"] > 0
    m = json.loads((d / "eval_metrics.json").read_text(encoding="utf-8"))
    assert m["event"]["n_events"] == 1 and m["main_row"]["model"] == name

    # A chained follow-up job: the previous segment finished with the same config -> nothing to do, no stage records added for a new segment
    before = (d / "run_manifest.json").read_text(encoding="utf-8")
    assert run_model.main(["--model", name, "--resume", *E2E]) == 0
    assert (d / "run_manifest.json").read_text(encoding="utf-8") == before
    # A changed config (e.g. more epochs) is not skipped; an existing model.pt is loaded directly
    assert run_model.main(["--model", name, "--resume", *E2E, "--set", "ml.eval.merge_gap_s=5"]) == 0
    after = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(after["segments"]) == 2 and after["stages"]["train"]["resumed"] is True
