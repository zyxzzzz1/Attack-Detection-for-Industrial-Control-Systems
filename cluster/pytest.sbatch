#!/bin/bash
#SBATCH --job-name=hai-pytest
#SBATCH --gres=gpu:L4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=TODO 写在 config/cluster.local.yaml/slurm_logs/%x_%j.out
#
# 由 scripts/make_cluster_jobs.py 按 config/cluster.yaml 生成，不要手改。
# 环境检查：在一个 GPU 节点上跑完整 pytest，并确认 torch 能用 GPU。默认分区（不写 -p）。
# 集群上的第一个作业是 srun 版的 pytest（见 docs/cluster_runbook.md 第 5 步）；本脚本是它的 GPU 版补充。
# 用法：
#   sbatch cluster/pytest.sbatch
#   bash cluster/pytest.sbatch --smoke                          # 本地，不经过调度器
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

python -m pytest -q
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
if [[ "${SMOKE}" != "1" ]]; then
    python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else '申请了 GPU 但 torch 用不了：检查 torch 的 CUDA 构建与驱动版本（env/README.md）')"
fi
