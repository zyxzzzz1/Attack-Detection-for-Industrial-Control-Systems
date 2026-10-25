"""Model registry. All models implement a uniform interface (main-line decision 3):

    fit(train_ds, val_es_ds) -> dict | None                 # returns a training summary to record in the manifest
    score(ds) -> (scores, per_sensor_err | None)            # per window; scores has shape (n_windows,), per_sensor_err (n_windows, D)
    save(directory) / load(directory)                       # each model chooses its own file names

    bind(directory, resume, log, manifest)                  # called by train.py before training: run directory, whether to resume, etc.
    exists(directory)                                       # whether the run directory already holds a trained model

The caller (score.py) assigns scores to each window's last second with WindowDataset.scatter.
"""

from src.models.classic import IForestModel, PCAModel
from src.models.deep_svdd import DeepSVDDModel
from src.models.lstm_ae import LSTMAEModel
from src.models.lstm_vae import LSTMVAEModel

MODEL_CLASSES = {"pca": PCAModel, "iforest": IForestModel, "lstm_ae": LSTMAEModel, "lstm_vae": LSTMVAEModel,
                 "deep_svdd": DeepSVDDModel}


def build_model(name: str, cfg: dict):
    if name not in MODEL_CLASSES:
        raise NotImplementedError(f"模型 {name} 尚未实现（已实现：{', '.join(MODEL_CLASSES)}）")
    return MODEL_CLASSES[name](cfg)


def load_model(name: str, cfg: dict, directory):
    model = build_model(name, cfg)
    model.load(directory)
    return model
