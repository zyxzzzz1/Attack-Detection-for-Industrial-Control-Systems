"""模型注册表。所有模型实现统一接口（主线补充决策 3）：

    fit(train_ds, val_es_ds) -> dict | None                 # 返回要记入 manifest 的训练摘要
    score(ds) -> (scores, per_sensor_err | None)            # 逐窗口；scores 形状 (窗口数,)，per_sensor_err 形状 (窗口数, D)
    save(directory) / load(directory)                       # 模型文件名由各模型自己决定

    bind(directory, resume, log, manifest)                  # 训练前由 train.py 调用：运行目录、是否续训等
    exists(directory)                                       # 运行目录里是否已有训练好的模型

分数由调用方（score.py）用 WindowDataset.scatter 赋给窗口最后一秒。
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
