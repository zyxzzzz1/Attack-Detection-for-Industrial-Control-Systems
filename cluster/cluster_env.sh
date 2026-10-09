# 由 scripts/make_cluster_jobs.py 按 config/cluster.yaml 生成，不要手改。
# 供 cluster/job_common.sh、scripts/push_to_cluster.sh、scripts/pull_artifacts.sh source。
CLUSTER_SSH_ALIAS=minerva
CLUSTER_CODE_DIR='TODO 写在 config/cluster.local.yaml'
CLUSTER_VENV_DIR='TODO 写在 config/cluster.local.yaml'
CLUSTER_DATA_DIR='TODO 写在 config/cluster.local.yaml'
CLUSTER_ARTIFACT_DIR='TODO 写在 config/cluster.local.yaml'
CLUSTER_PYTHON=python3
CLUSTER_PYTEST_SRUN_ARGS='--ntasks=1 --cpus-per-task=4 --mem=8G --time=00:20:00'
CLUSTER_ABLATION_ARRAY=0-8%3
CLUSTER_TODO_COUNT=4
CLUSTER_TODO_KEYS='paths.code_dir paths.venv_dir paths.data_dir paths.artifact_dir'
