"""配置读取与路径解析：所有路径来自 config，相对路径相对项目根目录；数据目录和产物目录可用环境变量覆盖。

环境变量（本地与集群共用一份代码，约定见 CLAUDE.md"可移植性"）：
- HAI_DATA_DIR  覆盖 ml.data_dir
- ARTIFACT_DIR  覆盖 ml.artifact_dir
"""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """相对路径按项目根目录解析，绝对路径原样返回。"""
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def hai_data_dir(cfg: dict) -> Path:
    return resolve(os.environ.get("HAI_DATA_DIR") or cfg["ml"]["data_dir"])


def artifact_dir(cfg: dict) -> Path:
    return resolve(os.environ.get("ARTIFACT_DIR") or cfg["ml"]["artifact_dir"])
