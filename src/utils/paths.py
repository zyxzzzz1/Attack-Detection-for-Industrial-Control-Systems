"""Config loading and path resolution: all paths come from the config, relative paths are relative to the project root; the data and artifact directories can be overridden by environment variables.

Environment variables (one codebase for local and cluster, see "Portability" in CLAUDE.md):
- HAI_DATA_DIR  overrides ml.data_dir
- ARTIFACT_DIR  overrides ml.artifact_dir
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
    """Resolve relative paths against the project root; absolute paths are returned unchanged."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def hai_data_dir(cfg: dict) -> Path:
    return resolve(os.environ.get("HAI_DATA_DIR") or cfg["ml"]["data_dir"])


def artifact_dir(cfg: dict) -> Path:
    return resolve(os.environ.get("ARTIFACT_DIR") or cfg["ml"]["artifact_dir"])
