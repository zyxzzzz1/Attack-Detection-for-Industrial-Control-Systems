"""运行环境的公共部分：配置覆盖与哈希、随机种子、设备检测、线程 / 进程数、运行目录与 run_manifest.json。

本地与集群共用（可移植性约定见 CLAUDE.md）。超参数不因设备不同而改变；这里只记录运行环境。
"""

import copy
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.utils.paths import ROOT, artifact_dir

MANIFEST_NAME = "run_manifest.json"
GIT_COMMIT_FILE = ".git_commit"  # push_to_cluster.sh 写入：集群上的代码目录没有 .git 时从这里读取提交号
SLURM_KEYS = {"job_id": "SLURM_JOB_ID", "array_job_id": "SLURM_ARRAY_JOB_ID", "array_task_id": "SLURM_ARRAY_TASK_ID",
              "job_name": "SLURM_JOB_NAME", "partition": "SLURM_JOB_PARTITION", "node": "SLURMD_NODENAME",
              "cpus_per_task": "SLURM_CPUS_PER_TASK"}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------- 配置 ----------

def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """按 "a.b.c=value" 覆盖配置（value 按 YAML 解析），返回新字典，不修改原配置。中间层级不存在时创建。"""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"覆盖项必须写成 key=value：{item!r}")
        key, value = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"{key} 的上级 {part} 不是字典，无法覆盖")
        node[parts[-1]] = yaml.safe_load(value)
    return out


def config_hash(cfg: dict) -> str:
    """配置内容的 sha256（键排序后的 JSON），与键的书写顺序无关。"""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def ablation_grid(cfg: dict) -> list[dict]:
    """消融网格（行优先：先遍历 windows，再遍历 hiddens）。每项含 job array 任务号、运行名与配置覆盖项。"""
    ab = cfg["ml"]["ablation"]
    grid = []
    for w in ab["windows"]:
        for h in ab["hiddens"]:
            grid.append({"index": len(grid), "window": w, "hidden": h,
                         "run_name": f"ablation/{ab['model']}_w{w}_h{h}",
                         "overrides": [f"{ab['window_key']}={w}", f"{ab['hidden_key']}={h}"]})
    return grid


# ---------- 随机种子、设备、线程 ----------

def seed_everything(seed: int) -> None:
    """固定 python、numpy、torch 的随机种子；有 GPU 时令 cuDNN 走确定性路径。"""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_device():
    """有 GPU 用第一块 GPU（只申请一块，不写多 GPU 代码），否则用 CPU。"""
    import torch
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def available_cpus() -> int:
    """作业可用的 CPU 数：优先取 SLURM_CPUS_PER_TASK，其次取进程的 CPU 亲和集合。"""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return max(1, int(slurm))
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def resolve_num_workers(requested: int) -> int:
    """DataLoader 进程数不超过作业申请的 CPU 数。"""
    return max(0, min(int(requested), available_cpus()))


def limit_threads() -> int:
    """把 torch 的线程数限制在作业申请的 CPU 数以内（集群节点的核数远多于申请数）。"""
    import torch
    n = available_cpus()
    torch.set_num_threads(n)
    return n


# ---------- 环境信息 ----------

def git_commit(root: Path = ROOT) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                check=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True,
                                    check=True, timeout=10).stdout.strip())
        return {"commit": commit, "dirty": dirty, "source": "git"}
    except (OSError, subprocess.SubprocessError):
        pass
    f = root / GIT_COMMIT_FILE
    if f.exists():
        parts = f.read_text(encoding="utf-8").split()
        if parts:
            return {"commit": parts[0], "dirty": "dirty" in parts[1:], "source": GIT_COMMIT_FILE}
    return {"commit": None, "dirty": None, "source": None}


def nvidia_smi() -> list[str] | None:
    """nvidia-smi 查询到的 GPU（型号、显存、驱动版本）；没有 nvidia-smi 时返回 None。"""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True, timeout=20).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment_info() -> dict:
    import pandas
    import sklearn
    import torch
    device = detect_device()
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "versions": {"torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
                     "pandas": pandas.__version__},
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else cpu_model(),
        "cuda_version": torch.version.cuda if device.type == "cuda" else None,
        "gpu_nvidia_smi": nvidia_smi(),
        "cpus_available": available_cpus(),
        "slurm": {k: os.environ.get(v) for k, v in SLURM_KEYS.items()},
    }


# ---------- 运行目录、日志、manifest ----------

def run_dir(cfg: dict, run_name: str) -> Path:
    """运行目录 = 产物根目录 / 运行名（主模型为模型名，消融为 ablation/<名字>）。"""
    d = artifact_dir(cfg) / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_logging(directory: Path, name: str = "run") -> logging.Logger:
    """日志同时写到标准输出和运行目录下的 run.log（追加，续训时接着写）。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(directory / "run.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


class RunManifest:
    """run_manifest.json：一次运行的可复现信息。

    顶层字段描述整次运行（最近一段作业的环境、配置、结果）；segments 逐段记录每次启动
    （断点续训、--dependency 串接的多段作业各占一项），因此被中断的作业也留有记录。
    """

    def __init__(self, directory: Path) -> None:
        self.path = directory / MANIFEST_NAME
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def start(self, cfg: dict, *, model: str, run_name: str, smoke: bool, overrides: list[str], argv: list[str]) -> None:
        env = environment_info()
        git = git_commit()
        started = now()
        segment = {"started_at": started, "finished_at": None, "status": "running", "hostname": env["hostname"],
                   "device_name": env["device_name"], "slurm": env["slurm"], "git_commit": git["commit"], "argv": argv}
        previous = self.data.get("segments", [])
        self.data = {
            "model": model, "run_name": run_name, "smoke": smoke,
            "status": "running",
            "started_at": self.data.get("started_at") or started, "finished_at": None,
            "git_commit": git["commit"], "git_dirty": git["dirty"], "git_source": git["source"],
            "config_hash": config_hash(cfg), "overrides": overrides,
            "hyperparameters": copy.deepcopy(cfg["ml"]),
            "seed": cfg["ml"]["seed"],
            **env,
            "slurm_job_id": env["slurm"]["job_id"], "slurm_array_task_id": env["slurm"]["array_task_id"],
            "node": env["slurm"]["node"] or env["hostname"],
            "final_epoch": self.data.get("final_epoch"), "early_stop_epoch": self.data.get("early_stop_epoch"),
            "stages": self.data.get("stages", {}),
            "segments": [*previous, segment],
        }
        self.save()

    def update(self, **fields) -> None:
        self.data.update(fields)
        self.save()

    def stage(self, name: str, **fields) -> None:
        self.data.setdefault("stages", {})[name] = {**fields, "finished_at": now()}
        self.save()

    def finish(self, status: str) -> None:
        finished = now()
        self.data["status"], self.data["finished_at"] = status, finished
        self.data["segments"][-1].update(status=status, finished_at=finished)
        self.save()

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)
