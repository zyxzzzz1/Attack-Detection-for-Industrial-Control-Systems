"""按 config/cluster.yaml 生成集群作业脚本（cluster/*.sbatch）和 shell 可读的 cluster/cluster_env.sh。

cluster.yaml 中仍为 TODO 的项会列出来。影响作业的 TODO（集群上的路径）会写进 cluster_env.sh，
作业脚本在集群上运行时检查到就拒绝执行；本地 --smoke 不受影响。

个人信息（登录用户名、集群上的绝对路径）不入库：入库的 cluster.yaml 中这些项是 TODO，真实值放在不入库的
config/cluster.local.yaml。本脚本生成两套文件：
- cluster/*.sbatch、cluster/cluster_env.sh（入库）：只按 cluster.yaml 生成，路径为 TODO，只用于本地 --smoke 和测试；
- cluster/local/（不入库）：有 cluster.local.yaml 时按合并后的配置生成，scripts/push_to_cluster.sh 把它们推送到
  集群上的 cluster/ 目录，scripts/push_to_cluster.sh 和 pull_artifacts.sh 也从这里读取集群路径。

用法：
    python -m scripts.make_cluster_jobs [--check]      # --check：只检查生成结果与已有文件是否一致，不写文件
"""

import argparse
import shlex
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.utils.paths import DEFAULT_CONFIG, ROOT, load_config
from src.utils.run import ablation_grid

CLUSTER_YAML = ROOT / "config" / "cluster.yaml"
CLUSTER_LOCAL_YAML = ROOT / "config" / "cluster.local.yaml"
CLUSTER_DIR = ROOT / "cluster"
LOCAL_SUBDIR = "local"
BASELINE_MODELS = ["pca", "iforest"]
# 影响作业能否在集群上运行的配置项（其余 TODO 只是待补充的说明信息；作业时长为 TODO 时提交必须带 --time）
BLOCKING = ("paths.",)


def deep_merge(base: dict, override: dict) -> dict:
    """override 中的项覆盖 base 的同名项（字典逐层合并），返回新字典。"""
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cluster(path: Path = CLUSTER_YAML, local: Path | None = CLUSTER_LOCAL_YAML) -> dict:
    """cluster.yaml，加上本机私有的 cluster.local.yaml（存在时）覆盖。local=None 时只读入库的那一份。"""
    cluster = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if local is not None and Path(local).exists():
        cluster = deep_merge(cluster, yaml.safe_load(Path(local).read_text(encoding="utf-8")) or {})
    return cluster


def find_todos(node, prefix="") -> list[str]:
    """递归找出取值以 TODO 开头的配置项，返回点分路径。"""
    if isinstance(node, dict):
        return [k for key, value in node.items() for k in find_todos(value, f"{prefix}{key}.")]
    return [prefix[:-1]] if isinstance(node, str) and node.strip().upper().startswith("TODO") else []


def render_all(cluster: dict, cfg: dict) -> dict[str, str]:
    """返回 {文件名: 内容}。"""
    todos = find_todos(cluster)
    blocking = [t for t in todos if t.startswith(BLOCKING)]
    env = Environment(loader=FileSystemLoader(CLUSTER_DIR / "templates"), undefined=StrictUndefined,
                      keep_trailing_newline=True)
    n_ablation = len(ablation_grid(cfg))
    array = f"0-{n_ablation - 1}%{cluster['jobs']['ablation']['array_concurrency']}"
    common = {"paths": cluster["paths"], "jobs": cluster["jobs"], "todo_keys": blocking}
    out = {
        "train_one.sbatch": env.get_template("train_one.sbatch.j2").render(
            name="hai-train", job=cluster["jobs"]["train"], array=None, **common),
        "baselines.sbatch": env.get_template("baselines.sbatch.j2").render(
            name="hai-baselines", job=cluster["jobs"]["baselines"], array=None, baseline_models=BASELINE_MODELS, **common),
        "ablation_array.sbatch": env.get_template("ablation_array.sbatch.j2").render(
            name="hai-ablation", job=cluster["jobs"]["ablation"], array=array, **common),
        "pytest.sbatch": env.get_template("pytest.sbatch.j2").render(**common),
    }
    pytest_job = cluster["jobs"]["pytest"]
    values = {
        "CLUSTER_SSH_ALIAS": cluster["cluster"]["ssh_alias"],
        "CLUSTER_CODE_DIR": cluster["paths"]["code_dir"],
        "CLUSTER_VENV_DIR": cluster["paths"]["venv_dir"],
        "CLUSTER_DATA_DIR": cluster["paths"]["data_dir"],
        "CLUSTER_ARTIFACT_DIR": cluster["paths"]["artifact_dir"],
        "CLUSTER_PYTHON": cluster["cluster"]["python"],
        "CLUSTER_PYTEST_SRUN_ARGS": f"--ntasks=1 --cpus-per-task={pytest_job['cpus_per_task']} --mem={pytest_job['mem']} "
                                    f"--time={pytest_job['time']}",
        "CLUSTER_ABLATION_ARRAY": array,
        "CLUSTER_TODO_COUNT": str(len(blocking)),
        "CLUSTER_TODO_KEYS": " ".join(blocking),
    }
    lines = ["# 由 scripts/make_cluster_jobs.py 按 config/cluster.yaml 生成，不要手改。",
             "# 供 cluster/job_common.sh、scripts/push_to_cluster.sh、scripts/pull_artifacts.sh source。"]
    lines += [f"{k}={shlex.quote(v)}" for k, v in values.items()]
    out["cluster_env.sh"] = "\n".join(lines) + "\n"
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="生成集群作业脚本")
    p.add_argument("--cluster-config", type=Path, default=CLUSTER_YAML)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--check", action="store_true", help="只检查已有文件是否与配置一致")
    args = p.parse_args()

    cfg = load_config(args.config)
    cluster = load_cluster(args.cluster_config, local=None)
    targets = [(CLUSTER_DIR, cluster)]
    has_local = args.cluster_config == CLUSTER_YAML and CLUSTER_LOCAL_YAML.exists()
    if has_local:  # 真实的作业脚本（不入库）
        targets.append((CLUSTER_DIR / LOCAL_SUBDIR, load_cluster(args.cluster_config)))
    stale = []
    for directory, conf in targets:
        directory.mkdir(parents=True, exist_ok=True)
        for name, text in render_all(conf, cfg).items():
            path = directory / name
            if args.check:
                if not path.exists() or path.read_text(encoding="utf-8") != text:
                    stale.append(str(path.relative_to(ROOT)))
            else:
                path.write_text(text, encoding="utf-8")
                print(f"[生成] {path.relative_to(ROOT)}")
    final = targets[-1][1]
    todos = find_todos(final)
    if not has_local:
        print("[提示] 没有 config/cluster.local.yaml：只生成了入库的作业脚本（路径为 TODO，只能用于本地 --smoke）。"
              "要在集群上运行，先按 config/cluster.local.example.yaml 写一份。")
    if todos:
        print(f"[提示] {'合并后的配置' if has_local else 'cluster.yaml'} 还有 {len(todos)} 个 TODO：")
        for t in todos:
            print(f"  - {t}{'（影响作业，补全前不能在集群上提交）' if t.startswith(BLOCKING) else ''}")
    if stale:
        print(f"[过期] 与配置不一致，需要重新生成：{stale}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
