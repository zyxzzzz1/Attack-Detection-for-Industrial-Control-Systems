"""The single entry point for reading labels. Only src/evaluate.py and src/attribution.py (attribution hit-rate evaluation) may import this module.

tests/test_label_isolation.py scans every .py under src/: apart from the whitelisted files, no file may reference this module
or contain the string "attack". Label column names come from ml.hai.label_cols in the config (attack, attack_P1~P3).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import hai_data_dir


def read_labels(path: Path, time_col: str, label_cols: list[str]) -> pd.DataFrame:
    """Read the time column and label columns (int8) of one file."""
    header = [c.strip() for c in pd.read_csv(path, nrows=0).columns]
    df = pd.read_csv(path, skipinitialspace=True, header=0, names=header, usecols=[time_col, *label_cols],
                     dtype={c: np.int8 for c in label_cols})
    return df[[time_col, *label_cols]]


def load_test_labels(cfg: dict) -> dict[str, pd.DataFrame]:
    """Read the labels of all test files in config order, keyed by file name. Called only in the final evaluation (constraint 2)."""
    hai = cfg["ml"]["hai"]
    return {name: read_labels(hai_data_dir(cfg) / name, hai["time_col"], hai["label_cols"])
            for name in hai["test_files"]}
