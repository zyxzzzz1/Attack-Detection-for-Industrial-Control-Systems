"""一次运行的上下文：各阶段（prepare、train、diagnose、score、threshold、evaluate）之间共享的对象。

每个阶段都可以单独执行（--stages），所以需要的东西不在内存里时从运行目录读取或重新计算：
数据重新读取并 fit 预处理（确定性的，结果与 prepare 阶段相同），模型从运行目录读取。
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.data.preprocess import TrainData, prepare_train
from src.data.windowing import WindowDataset
from src.utils.run import RunManifest


@dataclass
class RunContext:
    cfg: dict
    model: str
    run_name: str
    directory: Path
    smoke: bool
    resume: bool
    manifest: RunManifest
    log: object
    data: TrainData | None = None
    state: dict = field(default_factory=dict)  # 阶段之间传递的内存对象（模型、分数等）

    @property
    def fraction(self) -> float:
        return self.cfg["ml"]["smoke"]["fraction"] if self.smoke else 1.0

    def ensure_data(self) -> TrainData:
        if self.data is None:
            self.data = prepare_train(self.cfg, self.fraction)
        return self.data

    def dataset(self, split: str) -> WindowDataset:
        """train 用训练步长，val_es / val_thr 用评估步长。"""
        w = self.cfg["ml"]["window"]
        return self.ensure_data().dataset(split, w["length"], w["train_stride"] if split == "train" else w["eval_stride"])

    def ensure_model(self):
        if "model" not in self.state:
            from src.models import load_model
            self.state["model"] = load_model(self.model, self.cfg, self.directory)
        return self.state["model"]


def stage_cli(stages: str, description: str) -> int:
    """单个阶段脚本（python -m src.train 等）的入口：转给 run_model，只执行指定的阶段。"""
    from src.run_model import main
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(f"{description}\n等价于 python -m src.run_model --stages {stages} ...；参数如下。\n")
    return main([*sys.argv[1:], "--stages", stages])
