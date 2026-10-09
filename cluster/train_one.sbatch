#!/bin/bash
#SBATCH --job-name=hai-train
#SBATCH --partition=long
#SBATCH --gres=gpu:L4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=TODO 写在 config/cluster.local.yaml/slurm_logs/%x_%j.out
#
# 由 scripts/make_cluster_jobs.py 按 config/cluster.yaml 生成，不要手改（改 cluster/templates/ 或 cluster.yaml 后重新生成）。
# 注意：cluster.yaml 还有影响作业的 TODO（paths.code_dir、paths.venv_dir、paths.data_dir、paths.artifact_dir），补全并重新生成之前不能在集群上提交。
#
# 训练、打分、阈值、评估一个模型。作业被中断后原样重新提交即可从 checkpoint 继续（--resume）。
# 用法：
#   sbatch [--time=HH:MM:SS] cluster/train_one.sbatch <模型名> [run_model 的其他参数，如 --set ml.train.max_epochs=1]
#   bash cluster/train_one.sbatch --smoke <模型名> [...]        # 本地冒烟，不经过调度器
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

MODEL="${1:?用法：train_one.sbatch [--smoke] <模型名> [--set key=value ...]}"
shift
python -m src.run_model --model "${MODEL}" --resume "${SMOKE_ARGS[@]}" "$@"
