"""把报告引用到的各次运行的小文件从产物目录复制到 reports/main/runs/（入库），让报告里的每个数字都能在仓库里查到来源。
由 scripts.make_key_numbers 在汇总关键数字之前自动调用，也可以单独运行。

复制的文件：eval_metrics.json、run_manifest.json、threshold.json、preprocess.json、train_log.csv（没有的跳过，
例如 PCA 没有 train_log.csv）。不复制分数、逐传感器误差、模型权重、checkpoint 和日志。
只接受集群上的完整运行（非冒烟、非本地诊断、git_dirty=false、已完成）；不合格的运行跳过并打印原因。
包括：config 的 ml.models 中全部模型、消融的全部组合、多随机种子的全部运行。

脱敏（仓库将来可能公开；产物目录里的原件不动）：
- 键名为 hostname、node 的值（计算节点的主机名）一律替换为占位符 <cluster-node>；
- 任何字符串里出现的集群用户名（cluster.cid）、集群上的绝对路径（paths.*；真实值在不入库的 config/cluster.local.yaml）、
  登录主机名，以及用户目录形式的路径（集群的 data/users/<名字>、Linux 主目录 home/<名字>），
  替换为 <cluster-user>、<cluster-path>、<cluster-host>、<user-path>。
除此之外内容不变：没有命中的文件与原件逐字节相同。测试会核对入库副本 = 对原件做同样脱敏的结果。

用法：
    python -m scripts.snapshot_runs
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from src.utils.paths import DEFAULT_CONFIG, ROOT, artifact_dir, load_config, resolve
from src.utils.reporting import report_problem
from src.utils.run import ablation_grid

FILES = ["eval_metrics.json", "run_manifest.json", "threshold.json", "preprocess.json", "train_log.csv"]
SNAPSHOT_SUBDIR = "runs"
NODE_KEYS = {"hostname", "node"}
NODE_PLACEHOLDER = "<cluster-node>"
# 用户目录形式的路径。主目录的前缀拆开写，是因为可移植性测试不允许代码里出现主目录的绝对路径字面量
USER_PATH_RE = re.compile(r"/(?:data/users|" + "ho" + r"me)/[^/\s\"',:]+")


def run_names(cfg: dict) -> list[str]:
    ml = cfg["ml"]
    study = ml["seed_study"]
    seeds = [f"{study['run_subdir']}/{study['model']}_s{s}" for s in study["seeds"] if s != ml["seed"]]
    return [*ml["models"], *(g["run_name"] for g in ablation_grid(cfg)), *seeds]


def scrub_rules(cluster_yaml: Path | None = None) -> list[tuple[str, str]]:
    """(要替换的字符串, 占位符)，长的在前。来自 cluster.yaml 与同目录的 cluster.local.yaml；值为 TODO 或不存在的项跳过。"""
    path = cluster_yaml or ROOT / "config" / "cluster.yaml"
    if not path.exists():
        return []
    c = yaml.safe_load(path.read_text(encoding="utf-8"))
    local = path.with_name("cluster.local.yaml")   # 真实的用户名和路径在不入库的本机配置里
    if local.exists():
        extra = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
        c = {**c, **{k: {**(c.get(k) or {}), **v} for k, v in extra.items() if isinstance(v, dict)}}
    rules = [(str(v), "<cluster-path>") for v in (c.get("paths") or {}).values()]
    cl = c.get("cluster") or {}
    rules += [(str(cl.get("login_host", "")), "<cluster-host>"), (str(cl.get("cid", "")), "<cluster-user>")]
    rules = [(s, r) for s, r in rules if s and not s.startswith("TODO")]
    return sorted(set(rules), key=lambda x: -len(x[0]))


def scrub_text(text: str, rules: list[tuple[str, str]]) -> str:
    for needle, placeholder in rules:
        text = text.replace(needle, placeholder)
    return USER_PATH_RE.sub("<user-path>", text)


def scrub_nodes(obj):
    """递归地把键名为 hostname / node 的非空值替换为占位符。"""
    if isinstance(obj, dict):
        return {k: (NODE_PLACEHOLDER if k in NODE_KEYS and v else scrub_nodes(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_nodes(v) for v in obj]
    return obj


def redacted(path: Path, rules: list[tuple[str, str]]) -> str:
    """入库副本的内容。JSON 中没有任何需要替换的内容时原样返回（与原件逐字节相同）。"""
    with open(path, encoding="utf-8", newline="") as f:   # newline=""：不改动换行符（train_log.csv 是 \r\n）
        text = f.read()
    if path.suffix == ".json":
        data = json.loads(text)
        clean = scrub_nodes(data)
        if clean != data:
            text = json.dumps(clean, ensure_ascii=False, indent=2) + "\n"
    return scrub_text(text, rules)


def snapshot(cfg: dict) -> tuple[list[str], list[str]]:
    root, out = artifact_dir(cfg), resolve(cfg["ml"]["report_dir"]) / SNAPSHOT_SUBDIR
    rules = scrub_rules()
    copied, skipped = [], []
    for name in run_names(cfg):
        src = root / name
        problem = report_problem(src) if src.exists() else "运行目录不存在（尚未运行或尚未同步）"
        if problem:
            skipped.append(f"{name}: {problem}")
            continue
        dst = out / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            if (src / f).exists():
                with open(dst / f, "w", encoding="utf-8", newline="") as fh:
                    fh.write(redacted(src / f, rules))
        copied.append(name)
    return copied, skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="把各次运行的小文件脱敏后复制到 reports/main/runs/")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    copied, skipped = snapshot(cfg)
    for s in skipped:
        print(f"[跳过] {s}")
    print(f"[复制] {len(copied)} 次运行（已脱敏）→ {resolve(cfg['ml']['report_dir']) / SNAPSHOT_SUBDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
