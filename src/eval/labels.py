"""标签读取的唯一入口。只允许 src/evaluate.py 与 src/attribution.py（归因命中率评估）导入本模块。

tests/test_label_isolation.py 扫描 src/ 下全部 .py：除白名单文件外，任何文件不得引用本模块，
也不得出现字符串 "attack"。标签列名来自 config 的 ml.hai.label_cols（attack、attack_P1~P3）。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import hai_data_dir


def read_labels(path: Path, time_col: str, label_cols: list[str]) -> pd.DataFrame:
    """读取一个文件的时间列与标签列（int8）。"""
    header = [c.strip() for c in pd.read_csv(path, nrows=0).columns]
    df = pd.read_csv(path, skipinitialspace=True, header=0, names=header, usecols=[time_col, *label_cols],
                     dtype={c: np.int8 for c in label_cols})
    return df[[time_col, *label_cols]]


def load_test_labels(cfg: dict) -> dict[str, pd.DataFrame]:
    """按配置顺序读取全部 test 文件的标签，键为文件名。只在最终评估时调用（约束 2）。"""
    hai = cfg["ml"]["hai"]
    return {name: read_labels(hai_data_dir(cfg) / name, hai["time_col"], hai["label_cols"])
            for name in hai["test_files"]}
