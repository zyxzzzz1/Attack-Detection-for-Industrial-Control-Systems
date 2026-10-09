"""传统基线：PCA（重构误差）与 IsolationForest（主线补充决策 5）。

PCA 的输入选逐秒向量（D 维），不用展平窗口（W×D 维）：
- 展平窗口在 W=60 时有 3,420 维，相邻窗口高度重复，主成分大部分用来描述时间上的平移，而不是传感器之间的关系；
- 逐秒 PCA 是不含时间信息的基线，和 LSTM 类模型的差别正好反映"时间建模"带来的收益；
- 逐传感器误差可以直接得到，用于 val_thr 上的重尾特征诊断和 M5 的归因对比。
为了和其他模型使用同一套窗口设置，窗口分数 = 窗口内 W 秒逐秒重构 MSE 的均值（与 LSTM-AE 的窗口 MSE 口径相同），
per_sensor_err = 窗口最后一秒各传感器的平方误差。
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
    """用 joblib 保存 / 读取 self.state（字典）。"""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.state: dict = {}

    def bind(self, directory: Path, resume: bool, log, manifest=None) -> None:
        """传统基线一次拟合完成，不需要 checkpoint。"""

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
        """在 train 部分的全部逐秒向量上拟合，保留 variance 比例的方差。不需要早停，val_es_ds 不使用。"""
        params = self.cfg["ml"]["models"]["pca"]
        if params["input"] != "per_second":
            raise ValueError("PCA 只实现了逐秒向量输入（ml.models.pca.input = per_second）")
        x = np.concatenate(train_ds.segments).astype(np.float64)
        pca = PCA(n_components=params["variance"], svd_solver="full").fit(x)
        self.state = {"pca": pca}
        return {"n_fit_rows": len(x), "n_components": int(pca.n_components_),
                "explained_variance": float(pca.explained_variance_ratio_.sum())}

    def row_errors(self, x: np.ndarray) -> np.ndarray:
        """逐秒、逐传感器的平方重构误差，形状 (n, D)。"""
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
    """(B, W, D) → (B, 5·D)：每个传感器在窗口内的 mean、std、min、max、last−first（顺序见 STAT_NAMES）。"""
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
        """在训练窗口的统计特征上拟合。窗口数超过 max_train_windows 时随机下采样（固定种子）。"""
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
        """分数 = −score_samples（越大越异常）。不输出逐传感器误差。"""
        batch = self.cfg["ml"]["models"]["iforest"]["feature_batch"]
        forest, idx = self.state["forest"], np.arange(len(ds))
        out = [-forest.score_samples(window_stats(ds.get_batch(idx[i:i + batch]).numpy()))
               for i in range(0, len(idx), batch)]
        return (np.concatenate(out) if out else np.empty(0)).astype(np.float32), None
