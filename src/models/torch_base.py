"""Training / scoring loop shared by the LSTM models (LSTM-AE, LSTM-VAE, Deep SVDD).

- Adam, early stopping on the val_es loss (main-line decision 4); device auto-detected, hyperparameters never change with the device.
- Resumable training: after each epoch save checkpoints/last.pt (model, optimizer, early-stopping state, RNG state, training log) and
  checkpoints/epoch_NNN.pt (model weights only); the weights with the best validation loss are also saved as checkpoints/best.pt.
  After training the best weights are written to model.pt. With --resume: load model.pt if present; otherwise continue from the next epoch if last.pt exists.
  The shuffle order of each epoch depends only on (seed, epoch) and the RNG state is saved with the checkpoint, so resuming after an interruption gives the same result as an uninterrupted run (bit-identical on CPU).
- train_log.csv has one row per epoch, including training / validation time.

Subclasses implement build_net, batch_loss and batch_scores, and optionally on_fit_start (Deep SVDD uses it to set the center).
"""

import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch

from src.data.windowing import WindowDataset
from src.utils.run import detect_device

MODEL_FILE = "model.pt"
CKPT_DIR = "checkpoints"
TRAIN_LOG = "train_log.csv"


def atomic_save(obj, path: Path) -> None:
    """Write a temporary file and then rename it, so a job killed mid-save never leaves a corrupt checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


class TorchWindowModel:
    name = ""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.params = cfg["ml"]["models"][self.name]
        self.train_cfg = cfg["ml"]["train"]
        self.device = detect_device()
        self.net: torch.nn.Module | None = None
        self.directory: Path | None = None
        self.resume = False
        self.log = logging.getLogger("run")
        self.manifest = None
        self.extra: dict = {}  # Model-specific information to write into the manifest (saved with the checkpoint, so it survives resuming)

    # ---------- Implemented by subclasses ----------

    def build_net(self, n_features: int, window: int) -> torch.nn.Module:
        raise NotImplementedError

    def batch_loss(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict]:
        """Return (mean loss of this batch, components to log)."""
        raise NotImplementedError

    def batch_scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (per-window scores (B,), per-sensor errors at the window's last second (B, D) or None)."""
        raise NotImplementedError

    def on_fit_start(self, train_ds: WindowDataset) -> None:
        """Called once before training from scratch (not when resuming)."""

    # ---------- Shared loop ----------

    def bind(self, directory: Path, resume: bool, log, manifest=None) -> None:
        self.directory, self.resume, self.log, self.manifest = Path(directory), resume, log, manifest

    def exists(self, directory: Path) -> bool:
        return (Path(directory) / MODEL_FILE).exists()

    def _init_net(self, n_features: int, window: int) -> None:
        self.meta = {"n_features": n_features, "window": window, "params": dict(self.params)}
        self.net = self.build_net(n_features, window).to(self.device)

    def save(self, directory: Path) -> None:
        state = {k: v.cpu() for k, v in self.net.state_dict().items()}
        atomic_save({"model": state, "meta": self.meta, "name": self.name}, Path(directory) / MODEL_FILE)

    def load(self, directory: Path) -> None:
        blob = torch.load(Path(directory) / MODEL_FILE, map_location="cpu", weights_only=True)
        self._init_net(blob["meta"]["n_features"], blob["meta"]["window"])
        self.net.load_state_dict(blob["model"])

    def _epoch_end_hook(self, epoch: int) -> None:
        """Called after each epoch's checkpoint is written; tests use it to simulate an interrupted job."""

    def _mean_loss(self, ds: WindowDataset, batch_size: int) -> tuple[float, dict]:
        self.net.eval()
        total, parts, n = 0.0, {}, 0
        with torch.no_grad():
            for _, x in ds.iter_batches(batch_size):
                loss, comp = self.batch_loss(x.to(self.device), training=False)
                total += loss.item() * len(x)
                for k, v in comp.items():
                    parts[k] = parts.get(k, 0.0) + float(v) * len(x)
                n += len(x)
        return total / n, {k: v / n for k, v in parts.items()}

    def fit(self, train_ds: WindowDataset, val_es_ds: WindowDataset) -> dict:
        tc, seed = self.train_cfg, self.cfg["ml"]["seed"]
        if tc["optimizer"] != "adam":
            raise ValueError("only Adam is implemented (ml.train.optimizer = adam)")
        ckpt_dir = self.directory / CKPT_DIR
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        last_path, best_path, log_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt", self.directory / TRAIN_LOG

        self._init_net(train_ds.n_features, train_ds.window)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=tc["lr"])
        state = {"epoch": 0, "best_val": float("inf"), "best_epoch": 0, "bad_epochs": 0, "stopped_early": False,
                 "history": []}
        if self.resume and last_path.exists():
            # Load onto CPU: torch.set_rng_state only accepts CPU state; weights and optimizer state are moved to the model's device by load_state_dict
            ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
            self.net.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            torch.set_rng_state(ckpt["torch_rng"])
            if ckpt.get("cuda_rng") is not None and self.device.type == "cuda":
                torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
            state = ckpt["state"]
            self.extra = state.get("extra", {})
            self.log.info("resuming from checkpoint: %d epochs done, best validation loss %.6g (epoch %d)",
                          state["epoch"], state["best_val"], state["best_epoch"])
        else:
            self.on_fit_start(train_ds)
            state["extra"] = self.extra
            log_path.unlink(missing_ok=True)
        if self.manifest is not None and self.extra:
            self.manifest.update(**self.extra)

        while state["epoch"] < tc["max_epochs"] and not state["stopped_early"]:
            epoch = state["epoch"] + 1
            t0 = time.time()
            self.net.train()
            total, parts, n = 0.0, {}, 0
            order_rng = np.random.default_rng([seed, epoch])  # The shuffle order depends only on (seed, epoch)
            for _, x in train_ds.iter_batches(tc["batch_size"], shuffle=True, generator=order_rng):
                x = x.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss, comp = self.batch_loss(x, training=True)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(x)
                for k, v in comp.items():
                    parts[k] = parts.get(k, 0.0) + float(v) * len(x)
                n += len(x)
            train_loss, t1 = total / n, time.time()
            if not np.isfinite(train_loss):
                raise RuntimeError(f"training loss of epoch {epoch} is not finite ({train_loss})")
            val_loss, val_parts = self._mean_loss(val_es_ds, tc["eval_batch_size"])
            t2 = time.time()

            is_best = val_loss < state["best_val"]
            if is_best:
                state.update(best_val=val_loss, best_epoch=epoch, bad_epochs=0)
            else:
                state["bad_epochs"] += 1
            state["epoch"] = epoch
            state["stopped_early"] = state["bad_epochs"] >= tc["patience"]
            row = {"epoch": epoch, "train_loss": train_loss, "val_es_loss": val_loss, "is_best": int(is_best),
                   "train_seconds": round(t1 - t0, 2), "val_seconds": round(t2 - t1, 2), "lr": tc["lr"],
                   **{f"train_{k}": v / n for k, v in parts.items()}, **{f"val_es_{k}": v for k, v in val_parts.items()}}
            state["history"].append(row)

            weights = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            if is_best:
                atomic_save({"model": weights, "epoch": epoch, "val_es_loss": val_loss}, best_path)
            atomic_save({"model": weights, "epoch": epoch}, ckpt_dir / f"epoch_{epoch:03d}.pt")
            atomic_save({"model": weights, "optimizer": optimizer.state_dict(), "state": state,
                         "torch_rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None}, last_path)
            with open(log_path, "w", newline="", encoding="utf-8") as f:  # Rewrite the whole log every epoch, so resuming never produces duplicate rows
                writer = csv.DictWriter(f, fieldnames=list(state["history"][0]))
                writer.writeheader()
                writer.writerows(state["history"])
            self.log.info("epoch %d/%d  train %.6g  val_es %.6g%s  training %.1f s  validation %.1f s", epoch, tc["max_epochs"],
                          train_loss, val_loss, "  *" if is_best else "", t1 - t0, t2 - t1)
            if self.manifest is not None:
                self.manifest.update(final_epoch=epoch, early_stop_epoch=epoch if state["stopped_early"] else None,
                                     best_epoch=state["best_epoch"])
            self._epoch_end_hook(epoch)

        best = torch.load(best_path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(best["model"])
        seconds = [r["train_seconds"] + r["val_seconds"] for r in state["history"]]
        return {"final_epoch": state["epoch"], "best_epoch": state["best_epoch"], "best_val_es_loss": state["best_val"],
                "early_stop_epoch": state["epoch"] if state["stopped_early"] else None,
                "epoch_seconds_mean": float(np.mean(seconds)), "epoch_seconds_max": float(np.max(seconds)),
                "n_parameters": sum(p.numel() for p in self.net.parameters()), "device": str(self.device), **self.extra}

    def score(self, ds: WindowDataset) -> tuple[np.ndarray, np.ndarray | None]:
        self.net.eval()
        scores, per_sensor = [], []
        with torch.no_grad():
            for _, x in ds.iter_batches(self.train_cfg["eval_batch_size"]):
                s, e = self.batch_scores(x.to(self.device))
                scores.append(s.float().cpu().numpy())
                if e is not None:
                    per_sensor.append(e.float().cpu().numpy())
        if not scores:
            return np.empty(0, dtype=np.float32), None
        return np.concatenate(scores), np.concatenate(per_sensor) if per_sensor else None
