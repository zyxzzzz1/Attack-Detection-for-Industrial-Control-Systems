#!/bin/bash
#SBATCH --job-name=hai-baselines
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:05:00
#SBATCH --output=TODO 写在 config/cluster.local.yaml/slurm_logs/%x_%j.out
#
# 由 scripts/make_cluster_jobs.py 按 config/cluster.yaml 生成，不要手改（改 cluster/templates/ 或 cluster.yaml 后重新生成）。
# 注意：cluster.yaml 还有影响作业的 TODO（paths.code_dir、paths.venv_dir、paths.data_dir、paths.artifact_dir），补全并重新生成之前不能在集群上提交。
#
# PCA 与 IsolationForest：训练、打分、阈值、评估。只用 CPU。
# 用法：
#   sbatch [--time=HH:MM:SS] cluster/baselines.sbatch [run_model 的其他参数]
#   bash cluster/baselines.sbatch --smoke [...]                 # 本地冒烟，不经过调度器
set -euo pipefail

if [[ "${1:-}" == "--smoke" ]]; then SMOKE=1; shift; else SMOKE=0; fi
if [[ "${SMOKE}" == "1" ]]; then
    # shellcheck source=/dev/null
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/job_common.sh"
else
    # sbatch 会把脚本复制到 spool 目录，所以用代码目录的绝对路径
    # shellcheck source=/dev/null
    source "TODO 写在 config/cluster.local.yaml/cluster/job_common.sh"
fi

for MODEL in pca iforest; do
    python -m src.run_model --model "${MODEL}" --resume "${SMOKE_ARGS[@]}" "$@"
done
