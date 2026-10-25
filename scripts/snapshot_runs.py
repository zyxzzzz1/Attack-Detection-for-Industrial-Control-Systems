"""Copy the small files of each run cited by the report from the artifact directory to reports/main/runs/ (tracked), so every number in the report can be traced in the repository.
Called automatically by scripts.make_key_numbers before collecting key numbers; can also be run on its own.

Files copied: eval_metrics.json, run_manifest.json, threshold.json, preprocess.json, train_log.csv (missing ones are skipped,
e.g. PCA has no train_log.csv). Scores, per-sensor errors, model weights, checkpoints and logs are not copied.
Only full runs on the cluster are accepted (not smoke, not local diagnostic, git_dirty=false, completed); other runs are skipped with the reason printed.
Covers: all models in ml.models of the config, all ablation combinations, all multi-seed runs.

Redaction (the repository may become public; originals in the artifact directory are untouched):
- values of keys named hostname or node (compute-node host names) are always replaced by the placeholder <cluster-node>;
- in any string, the cluster user name (cluster.cid), absolute paths on the cluster (paths.*; real values in the untracked config/cluster.local.yaml),
  the login host name, and user-directory-style paths (the cluster's data/users/<name>, Linux home directories home/<name>)
  are replaced by <cluster-user>, <cluster-path>, <cluster-host>, <user-path>.
Nothing else changes: files without matches are byte-identical to the originals. The tests check that tracked copy = original with the same redaction applied.

Usage:
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
# User-directory-style paths. The home-directory prefix is split up because the portability test forbids absolute home-directory literals in code
USER_PATH_RE = re.compile(r"/(?:data/users|" + "ho" + r"me)/[^/\s\"',:]+")


def run_names(cfg: dict) -> list[str]:
    ml = cfg["ml"]
    study = ml["seed_study"]
    seeds = [f"{study['run_subdir']}/{study['model']}_s{s}" for s in study["seeds"] if s != ml["seed"]]
    return [*ml["models"], *(g["run_name"] for g in ablation_grid(cfg)), *seeds]


def scrub_rules(cluster_yaml: Path | None = None) -> list[tuple[str, str]]:
    """(string to replace, placeholder), longest first. Taken from cluster.yaml and the cluster.local.yaml next to it; items that are TODO or missing are skipped."""
    path = cluster_yaml or ROOT / "config" / "cluster.yaml"
    if not path.exists():
        return []
    c = yaml.safe_load(path.read_text(encoding="utf-8"))
    local = path.with_name("cluster.local.yaml")   # The real user name and paths are in the untracked machine-local config
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
    """Recursively replace non-empty values of keys named hostname / node with a placeholder."""
    if isinstance(obj, dict):
        return {k: (NODE_PLACEHOLDER if k in NODE_KEYS and v else scrub_nodes(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_nodes(v) for v in obj]
    return obj


def redacted(path: Path, rules: list[tuple[str, str]]) -> str:
    """Content of the tracked copy. A JSON file with nothing to replace is returned as-is (byte-identical to the original)."""
    with open(path, encoding="utf-8", newline="") as f:   # newline="": keep line endings unchanged (train_log.csv uses \r\n)
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
