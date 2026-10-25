"""Context of one run: objects shared between the stages (prepare, train, diagnose, score, threshold, evaluate).

Each stage can run on its own (--stages), so anything not in memory is read from the run directory or recomputed:
data are re-read and preprocessing refit (deterministic, same result as the prepare stage), the model is loaded from the run directory.
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
    state: dict = field(default_factory=dict)  # In-memory objects passed between stages (model, scores, etc.)

    @property
    def fraction(self) -> float:
        return self.cfg["ml"]["smoke"]["fraction"] if self.smoke else 1.0

    def ensure_data(self) -> TrainData:
        if self.data is None:
            self.data = prepare_train(self.cfg, self.fraction)
        return self.data

    def dataset(self, split: str) -> WindowDataset:
        """train uses the training stride; val_es / val_thr use the evaluation stride."""
        w = self.cfg["ml"]["window"]
        return self.ensure_data().dataset(split, w["length"], w["train_stride"] if split == "train" else w["eval_stride"])

    def ensure_model(self):
        if "model" not in self.state:
            from src.models import load_model
            self.state["model"] = load_model(self.model, self.cfg, self.directory)
        return self.state["model"]


def stage_cli(stages: str, description: str) -> int:
    """Entry point of the single-stage scripts (python -m src.train etc.): forwards to run_model and runs only the given stage."""
    from src.run_model import main
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(f"{description}\nEquivalent to python -m src.run_model --stages {stages} ...; arguments below.\n")
    return main([*sys.argv[1:], "--stages", stages])
