"""Read the scoring artifacts in a run directory (label-free)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.run import MANIFEST_NAME


@dataclass
class RunArtifacts:
    directory: Path
    manifest: dict
    index: dict            # test_index.json: position of each part in the concatenated array, feature names, window length
    threshold: dict        # threshold.json
    scores: np.ndarray     # (T,)
    per_sensor_err: np.ndarray | None  # (T, D), memory-mapped on demand

    @property
    def features(self) -> list[str]:
        return self.index["features"]


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run(directory: Path, per_sensor: bool = False) -> RunArtifacts:
    d = Path(directory)
    err_path = d / "per_sensor_err_test.npy"
    if per_sensor and not err_path.exists():
        raise FileNotFoundError(f"{d} has no per_sensor_err_test.npy (only reconstruction models produce per-sensor errors)")
    return RunArtifacts(directory=d, manifest=read_json(d / MANIFEST_NAME), index=read_json(d / "test_index.json"),
                        threshold=read_json(d / "threshold.json"), scores=np.load(d / "scores_test.npy"),
                        per_sensor_err=np.load(err_path, mmap_mode="r") if per_sensor else None)
