"""C0: generation of the cluster job scripts (resource parameters from cluster.yaml) and local --smoke execution."""

import copy
import os
import re
import subprocess

import pytest
import yaml

from scripts.make_cluster_jobs import CLUSTER_DIR, CLUSTER_YAML, find_todos, render_all
from src.utils.paths import ROOT, load_config


@pytest.fixture(scope="module")
def cluster():
    with open(CLUSTER_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def filled(cluster):
    """Fill the TODOs with dummy values to simulate a completed cluster.yaml."""
    c = copy.deepcopy(cluster)
    c["paths"] = {"code_dir": "/cluster/proj/code", "venv_dir": "/cluster/proj/code/.venv",
                  "data_dir": "/cluster/proj/data", "artifact_dir": "/cluster/proj/artifacts"}
    for job, t in (("train", "06:00:00"), ("baselines", "02:00:00"), ("ablation", "12:00:00")):
        c["jobs"][job]["time"] = t
    return c


def sbatch_lines(text):
    return [line for line in text.splitlines() if line.startswith("#SBATCH")]


def test_generated_files_are_up_to_date(cluster):
    for name, text in render_all(cluster, load_config()).items():
        assert (CLUSTER_DIR / name).read_text(encoding="utf-8") == text, f"{name} 过期：运行 python -m scripts.make_cluster_jobs"


def test_resources_come_from_cluster_yaml(cluster):
    files = render_all(filled(cluster), load_config())
    train = sbatch_lines(files["train_one.sbatch"])
    assert train == ["#SBATCH --job-name=hai-train", "#SBATCH --partition=long", "#SBATCH --gres=gpu:L4:1",
                     "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=4", "#SBATCH --mem=32G", "#SBATCH --time=06:00:00",
                     "#SBATCH --output=/cluster/proj/artifacts/slurm_logs/%x_%j.out"]
    # One task per job: two processes writing the same run directory would corrupt each other's artifacts
    assert all("#SBATCH --ntasks=1" in sbatch_lines(text) for name, text in files.items() if name.endswith(".sbatch"))
    base = sbatch_lines(files["baselines.sbatch"])
    assert "#SBATCH --cpus-per-task=16" in base and "#SBATCH --mem=64G" in base
    assert not any("gres" in line for line in base)  # Baselines request no GPU
    abl = sbatch_lines(files["ablation_array.sbatch"])
    assert "#SBATCH --array=0-8%3" in abl and "#SBATCH --gres=gpu:L4:1" in abl
    assert "#SBATCH --output=/cluster/proj/artifacts/slurm_logs/%x_%A_%a.out" in abl
    # Changing the yaml changes the generated output (nothing hard-coded)
    c = filled(cluster)
    c["jobs"]["ablation"]["array_concurrency"] = 2
    c["jobs"]["train"]["mem"] = "48G"
    files2 = render_all(c, load_config())
    assert "#SBATCH --array=0-8%2" in files2["ablation_array.sbatch"] and "#SBATCH --mem=48G" in files2["train_one.sbatch"]
    # On the cluster the shared script is sourced by absolute path; only one GPU is requested
    assert 'source "/cluster/proj/code/cluster/job_common.sh"' in files["train_one.sbatch"]
    assert "CLUSTER_VENV_DIR=/cluster/proj/code/.venv" in files["cluster_env.sh"]
    assert "CLUSTER_TODO_COUNT=0" in files["cluster_env.sh"]
    assert all(line.endswith(":1") for f in files.values() for line in sbatch_lines(f) if "gres" in line)


def test_todo_handling(cluster):
    todos = find_todos(cluster)
    files = render_all(cluster, load_config())
    blocking = [t for t in todos if t.startswith("paths.")]
    assert f"CLUSTER_TODO_COUNT={len(blocking)}" in files["cluster_env.sh"]
    # While the job time limit is still TODO no --time line is written (it must be given at submission), and --time=TODO is never generated
    if "jobs.train.time" in todos:
        assert not any("--time" in line for line in sbatch_lines(files["train_one.sbatch"]))
    done = "".join(render_all(filled(cluster), load_config()).values())
    assert "=TODO" not in done and "TODO/" not in done and "not yet set" not in done
    assert find_todos({"a": {"b": "TODO 稍后", "c": 1}, "d": "ok", "e": ["TODO"]}) == ["a.b"]


def test_job_refuses_to_run_outside_slurm_without_smoke(tmp_path, cluster):
    """A completed script executed directly on the login node must refuse to run."""
    files = render_all(filled(cluster), load_config())
    work = tmp_path / "code" / "cluster"
    work.mkdir(parents=True)
    (work / "job_common.sh").write_text((CLUSTER_DIR / "job_common.sh").read_text(encoding="utf-8"), encoding="utf-8")
    (work / "cluster_env.sh").write_text(files["cluster_env.sh"], encoding="utf-8")
    script = files["train_one.sbatch"].replace("/cluster/proj/code", str(tmp_path / "code"))
    (work / "train_one.sbatch").write_text(script, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM")}
    r = subprocess.run(["bash", str(work / "train_one.sbatch"), "pca"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "登录节点" in r.stderr


def test_sync_scripts_fail_fast(cluster, tmp_path):
    """Exit immediately while paths are still TODO; once filled in, an unreachable alias prompts to check the VPN (simulated with a nonexistent alias)."""
    for script in ("push_to_cluster.sh", "pull_artifacts.sh"):
        text = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert "检查 Chalmers VPN" in text and "BatchMode=yes" in text and "CLUSTER_SSH_ALIAS" in text
        assert subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)]).returncode == 0
    # Paths are TODO in the tracked config: with only the tracked files (no cluster/local/), the sync scripts exit right away without connecting to the cluster.
    # Run on a copy in a temporary directory; not in the project directory, which may contain a real cluster/local/ and would make the script actually connect to the cluster.
    assert "paths.artifact_dir" in find_todos(cluster) and "paths.code_dir" in find_todos(cluster)
    (tmp_path / "cluster").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "cluster" / "cluster_env.sh").write_text((CLUSTER_DIR / "cluster_env.sh").read_text(encoding="utf-8"), encoding="utf-8")
    for script in ("push_to_cluster.sh", "pull_artifacts.sh"):
        copy = tmp_path / "scripts" / script
        copy.write_text((ROOT / "scripts" / script).read_text(encoding="utf-8"), encoding="utf-8")
        r = subprocess.run(["bash", str(copy)], capture_output=True, text=True, timeout=60)
        assert r.returncode == 2 and "TODO" in r.stderr and "cluster.local.yaml" in r.stderr
    pull = (ROOT / "scripts" / "pull_artifacts.sh").read_text(encoding="utf-8")
    for pattern in ("scores_*.npy", "per_sensor_err_test.npy", "threshold.json", "train_log.csv", "run_manifest.json"):
        assert f"--include='{pattern}'" in pull
    assert "--with-checkpoints" in pull and pull.index("WITH_CKPT}\" == \"1\"") < pull.index("--include='*.pt'")


def test_push_refuses_dirty_worktree_unless_allowed(tmp_path, cluster):
    """Uncommitted changes in the worktree: refuse to push and list the files; --allow-dirty lets it through (and only then connects to the cluster). A clean worktree goes straight to connecting."""
    files = render_all(filled(cluster), load_config())
    repo = tmp_path / "repo"
    (repo / "cluster").mkdir(parents=True)
    (repo / "scripts").mkdir()
    env_text = files["cluster_env.sh"].replace("CLUSTER_SSH_ALIAS=minerva", "CLUSTER_SSH_ALIAS=no-such-host.invalid")
    (repo / "cluster" / "cluster_env.sh").write_text(env_text, encoding="utf-8")
    script = repo / "scripts" / "push_to_cluster.sh"
    script.write_text((ROOT / "scripts" / "push_to_cluster.sh").read_text(encoding="utf-8"), encoding="utf-8")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    run = lambda *a: subprocess.run(a, cwd=repo, capture_output=True, text=True, timeout=60)  # noqa: E731
    assert run(*git, "init", "-q").returncode == 0 and run(*git, "add", "-A").returncode == 0
    assert run(*git, "commit", "-qm", "init").returncode == 0

    clean = run("bash", str(script))
    assert clean.returncode == 1 and "检查 Chalmers VPN" in clean.stderr  # Clean: passes the check and gets stuck at connecting

    (repo / "cluster" / "cluster_env.sh").write_text(env_text + "# 改动\n", encoding="utf-8")
    (repo / "new_file.py").write_text("x = 1\n", encoding="utf-8")
    dirty = run("bash", str(script))
    assert dirty.returncode == 3 and "拒绝推送" in dirty.stderr
    assert "cluster/cluster_env.sh" in dirty.stderr and "new_file.py" in dirty.stderr
    assert "VPN" not in dirty.stderr and not (repo / ".git_commit").exists()

    allowed = run("bash", str(script), "--allow-dirty")
    assert allowed.returncode == 1 and "检查 Chalmers VPN" in allowed.stderr
    # Pushing data is not affected by the worktree state
    assert run("bash", str(script), "--data").returncode == 1


def test_sync_scripts_report_vpn_when_unreachable(tmp_path, cluster):
    files = render_all(filled(cluster), load_config())
    (tmp_path / "cluster").mkdir()
    (tmp_path / "scripts").mkdir()
    env_text = files["cluster_env.sh"].replace("CLUSTER_SSH_ALIAS=minerva", "CLUSTER_SSH_ALIAS=no-such-host.invalid")
    (tmp_path / "cluster" / "cluster_env.sh").write_text(env_text, encoding="utf-8")
    # push_to_cluster.sh needs a git repository (it checks the worktree first); its message when it cannot connect is covered in test_push_refuses_dirty_worktree_unless_allowed
    script = "pull_artifacts.sh"
    (tmp_path / "scripts" / script).write_text((ROOT / "scripts" / script).read_text(encoding="utf-8"), encoding="utf-8")
    r = subprocess.run(["bash", str(tmp_path / "scripts" / script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "检查 Chalmers VPN" in r.stderr


@pytest.mark.parametrize("script,args,expect_dirs", [
    ("train_one.sbatch", ["lstm_ae"], ["lstm_ae"]),
    ("baselines.sbatch", [], ["pca", "iforest"]),
    ("ablation_array.sbatch", ["8"], ["ablation/lstm_ae_w120_h128"]),
])
def test_job_bodies_run_locally_in_smoke_mode(tmp_path, script, args, expect_dirs):
    """Run the body of a job script locally (--smoke, no scheduler) on synthetic data. Requires .venv in the project root."""
    if not (ROOT / ".venv" / "bin" / "activate").exists():
        pytest.skip("没有本地 .venv（集群上的环境在别处），跳过本地冒烟")
    import numpy as np
    import pandas as pd
    cfg = load_config()
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()
    for k, name in enumerate(cfg["ml"]["hai"]["train_files"] + cfg["ml"]["hai"]["test_files"]):
        n = 30000
        df = pd.DataFrame({"time": [str(t) for t in pd.date_range(f"2020-01-{k + 1:02d}", periods=n, freq="s")],
                           "P1_A": rng.normal(size=n), "P2_B": rng.normal(size=n)})
        for c in cfg["ml"]["hai"]["label_cols"]:
            df[c] = 0
        df.to_csv(data / name, index=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM")}
    env.update(HAI_DATA_DIR=str(data), ARTIFACT_DIR=str(tmp_path / "artifacts"))
    r = subprocess.run(["bash", str(CLUSTER_DIR / script), "--smoke", *args, "--set", "ml.data.drop_train_overlap=false",
                        "--set", "ml.smoke.fraction=0.2"],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for d in expect_dirs:
        assert (tmp_path / "artifacts" / d / "run_manifest.json").exists()
        assert (tmp_path / "artifacts" / d / "run.log").exists()


def test_personal_cluster_settings_are_not_tracked(tmp_path):
    """The user name and absolute cluster paths live only in the untracked cluster.local.yaml; in the tracked config and job scripts they are TODO."""
    from scripts.make_cluster_jobs import CLUSTER_LOCAL_YAML, LOCAL_SUBDIR, deep_merge, load_cluster
    tracked = load_cluster(local=None)
    assert str(tracked["cluster"]["cid"]).startswith("TODO")
    assert all(str(v).startswith("TODO") for v in tracked["paths"].values())
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "config/cluster.local.yaml" in ignore and f"cluster/{LOCAL_SUBDIR}/" in ignore
    listed = subprocess.run(["git", "ls-files", "config", "cluster"], cwd=ROOT, capture_output=True, text=True)
    if listed.returncode == 0 and listed.stdout:      # There is no .git on the cluster, so skip this item
        assert "cluster.local.yaml" not in listed.stdout.replace("cluster.local.example.yaml", "")
        assert f"cluster/{LOCAL_SUBDIR}/" not in listed.stdout
    # The real values of the machine-local config never appear in tracked files (without a local config there is nothing to compare, so only the generic user-directory pattern is checked)
    secrets = []
    if CLUSTER_LOCAL_YAML.exists():
        local = yaml.safe_load(CLUSTER_LOCAL_YAML.read_text(encoding="utf-8"))
        secrets = [str(local["cluster"]["cid"]), *map(str, local["paths"].values())]
    files = [CLUSTER_YAML, ROOT / "config" / "cluster.local.example.yaml", ROOT / "PROGRESS.md", ROOT / "README.md",
             *(p for p in CLUSTER_DIR.iterdir() if p.is_file()), *(ROOT / "docs").glob("*.md")]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not [x for x in secrets if x in text], path
        assert not re.search(r"/data/users/[a-z0-9]+", text), path

    # Merge rule: the local config overrides same-named keys level by level, everything else unchanged; after merging no job-affecting TODO remains
    merged = deep_merge(tracked, {"cluster": {"cid": "someone"}, "paths": {k: f"/x/{k}" for k in tracked["paths"]}})
    assert merged["cluster"]["cid"] == "someone" and merged["cluster"]["ssh_alias"] == tracked["cluster"]["ssh_alias"]
    assert merged["jobs"] == tracked["jobs"] and not [t for t in find_todos(merged) if t.startswith("paths.")]
    assert "CLUSTER_CODE_DIR=/x/code_dir" in render_all(merged, load_config())["cluster_env.sh"]
    assert str(tracked["paths"]["code_dir"]).startswith("TODO")   # deep_merge does not modify the original dicts
