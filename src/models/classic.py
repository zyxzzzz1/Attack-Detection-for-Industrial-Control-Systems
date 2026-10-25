"""Classic baselines: PCA (reconstruction error) and IsolationForest (main-line decision 5).

PCA uses per-second vectors (D dimensions) as input rather than flattened windows (W×D dimensions):
- a flattened window has 3,420 dimensions at W=60 and adjacent windows overlap heavily, so most principal components would describe time shifts rather than relations between sensors;
- per-second PCA is a baseline without temporal information, so its gap to the LSTM models directly reflects what temporal modeling buys;
- per-sensor errors come for free, used for the heavy-tail diagnosis on val_thr and the M5 attribution comparison.
To use the same window setup as the other models, window score = mean of the per-second reconstruction MSE over the W seconds of the window (same definition as the LSTM-AE window MSE),
per_sensor_err = squared error of each sensor at the window's last second.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from src.data.windowing import WindowDataset
from src.utils.run import available_cpus

MODEL_FILE = "model.joblib"
STAT_NAMES = ("mean", "std", "min", "max", "last_minus_first")


class _JoblibModel:
    """Save / load self.state (a dict) with joblib."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.state: dict = {}

    def bind(self, directory: Path, resume: bool, log, manifest=None) -> None:
        """Classic baselines are fit in one go and need no checkpoints."""

    def save(self, directory: Path) -> None:
        joblib.dump(self.state, Path(directory) / MODEL_FILE)

    def load(self, directory: Path) -> None:
        self.state = joblib.load(Path(directory) / MODEL_FILE)

    @staticmethod
    def exists(directory: Path) -> bool:
        return (Path(directory) / MODEL_FILE).exists()


class PCAModel(_JoblibModel):
    name = "pca"

    def fit(self, train_ds: WindowDataset, val_es_ds: WindowDataset | None = None) -> dict:
        """Fit on all per-second vectors of the train part, keeping the variance fraction given by variance. No early stopping; val_es_ds is unused."""
        params = self.cfg["ml"]["models"]["pca"]
        if params["input"] != "per_second":
            raise ValueError("PCA only implements per-second vector input (ml.models.pca.input = per_second)")
        x = np.concatenate(train_ds.segments).astype(np.float64)
        pca = PCA(n_components=params["variance"], svd_solver="full").fit(x)
        self.state = {"pca": pca}
        return {"n_fit_rows": len(x), "n_components": int(pca.n_components_),
                "explained_variance": float(pca.explained_variance_ratio_.sum())}

    def row_errors(self, x: np.ndarray) -> np.ndarray:
        """Per-second, per-sensor squared reconstruction errors, shape (n, D)."""
        pca = self.state["pca"]
        x = x.astype(np.float64)
        return (x - pca.inverse_transform(pca.transform(x))) ** 2

    def score(self, ds: WindowDataset) -> tuple[np.ndarray, np.ndarray]:
        scores, per_sensor = [], []
        for seg, count in zip(ds.segments, ds.counts):
            if count == 0:
                continue
            err = self.row_errors(seg)
            csum = np.concatenate(([0.0], np.cumsum(err.mean(axis=1))))
            starts = np.arange(count) * ds.stride
            scores.append((csum[starts + ds.window] - csum[starts]) / ds.window)
            per_sensor.append(err[starts + ds.window - 1])
        if not scores:
            return np.empty(0, dtype=np.float32), np.empty((0, ds.n_features), dtype=np.float32)
        return np.concatenate(scores).astype(np.float32), np.concatenate(per_sensor).astype(np.float32)


def window_stats(windows: np.ndarray) -> np.ndarray:
    """(B, W, D) -> (B, 5·D): mean, std, min, max, last−first of each sensor within the window (order in STAT_NAMES)."""
    stats = [windows.mean(axis=1), windows.std(axis=1), windows.min(axis=1), windows.max(axis=1),
             windows[:, -1] - windows[:, 0]]
    return np.concatenate(stats, axis=1).astype(np.float32)


class IForestModel(_JoblibModel):
    name = "iforest"

    def _features(self, ds: WindowDataset, idx: np.ndarray) -> np.ndarray:
        batch = self.cfg["ml"]["models"]["iforest"]["feature_batch"]
        return np.concatenate([window_stats(ds.get_batch(idx[i:i + batch]).numpy())
                               for i in range(0, len(idx), batch)])

    def fit(self, train_ds: WindowDataset, val_es_ds: WindowDataset | None = None) -> dict:
        """Fit on the statistical features of the training windows. When there are more than max_train_windows windows, subsample randomly (fixed seed)."""
        params = self.cfg["ml"]["models"]["iforest"]
        seed = self.cfg["ml"]["seed"]
        idx = np.arange(len(train_ds))
        limit = params["max_train_windows"]
        if limit and len(idx) > limit:
            idx = np.sort(np.random.default_rng(seed).choice(idx, size=limit, replace=False))
        x = self._features(train_ds, idx)
        forest = IsolationForest(n_estimators=params["n_estimators"], max_samples=min(params["max_samples"], len(x)),
                                 random_state=seed, n_jobs=available_cpus()).fit(x)
        self.state = {"forest": forest}
        return {"n_fit_windows": len(x), "n_window_features": x.shape[1], "n_estimators": params["n_estimators"]}

    def score(self, ds: WindowDataset) -> tuple[np.ndarray, None]:
        """Score = −score_samples (higher is more anomalous). No per-sensor errors are produced."""
        batch = self.cfg["ml"]["models"]["iforest"]["feature_batch"]
        forest, idx = self.state["forest"], np.arange(len(ds))
        out = [-forest.score_samples(window_stats(ds.get_batch(idx[i:i + batch]).numpy()))
               for i in range(0, len(idx), batch)]
        return (np.concatenate(out) if out else np.empty(0)).astype(np.float32), None
