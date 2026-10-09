#!/bin/bash
#SBATCH --job-name=hai-ablation
#SBATCH --partition=long
#SBATCH --gres=gpu:L4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --array=0-8%3
#SBATCH --output=TODO 写在 config/cluster.local.yaml/slurm_logs/%x_%A_%a.out
#
# 由 scripts/make_cluster_jobs.py 按 config/cluster.yaml 生成，不要手改（改 cluster/templates/ 或 cluster.yaml 后重新生成）。
# 注意：cluster.yaml 还有影响作业的 TODO（paths.code_dir、paths.venv_dir、paths.data_dir、paths.artifact_dir），补全并重新生成之前不能在集群上提交。
#
# LSTM-AE 消融（M4）：job array，任务号 → (W, hidden) 的对应关系见 python -m src.run_model --list-ablation。
# 每个任务的产物在 产物目录/ablation/lstm_ae_w<W>_h<hidden>/。阈值分位 {99, 99.5, 99.9} 的敏感性分析
# 只需用各任务保存的 val_thr 分数重新取分位，不需要额外的 array 任务。
# 用法：
#   sbatch [--time=HH:MM:SS] cluster/ablation_array.sbatch [run_model 的其他参数]
#   sbatch --array=4 cluster/ablation_array.sbatch              # 只重跑某一个任务（被中断的任务会从 checkpoint 继续）
#   bash cluster/ablation_array.sbatch --smoke [任务号，默认 0]  # 本地冒烟，不经过调度器
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

if [[ "${SMOKE}" == "1" ]]; then
    INDEX="${1:-0}"
    [[ $# -gt 0 ]] && shift
else
    INDEX="${SLURM_ARRAY_TASK_ID:?需要作为 job array 提交}"
fi
python -m src.run_model --ablation-index "${INDEX}" --resume "${SMOKE_ARGS[@]}" "$@"
