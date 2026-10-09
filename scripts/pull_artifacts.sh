#!/bin/bash
# 把集群上各运行目录的结果拉回本地 artifacts/（或 ARTIFACT_DIR）。
# 默认拉回：scores_*.npy、per_sensor_err_test.npy、threshold.json、train_log.csv、run_manifest.json、
#           评估结果（eval_*.json / *.csv）、test_index.json、val_sensor_contrib.csv、预处理摘要与 scaler、
#           日志（run.log、slurm_logs/）。
# 默认不拉 checkpoint 和模型权重；加 --with-checkpoints 才拉。
#
# 用法：
#   scripts/pull_artifacts.sh [--with-checkpoints] [--dry-run] [运行名 ...]
#   例：scripts/pull_artifacts.sh lstm_ae pca      # 只拉这两个运行目录
#       scripts/pull_artifacts.sh ablation         # 只拉消融
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 集群路径等个人信息不入库：真实值在 cluster/local/cluster_env.sh（由 make_cluster_jobs 按 config/cluster.local.yaml 生成）；
# 没有时退回入库的 cluster/cluster_env.sh（路径为 TODO，下面会因此拒绝执行）。
CLUSTER_ENV="${ROOT}/cluster/cluster_env.sh"
if [[ -f "${ROOT}/cluster/local/cluster_env.sh" ]]; then CLUSTER_ENV="${ROOT}/cluster/local/cluster_env.sh"; fi
# shellcheck source=/dev/null
source "${CLUSTER_ENV}"

RSYNC_OPTS=(-az --prune-empty-dirs --info=stats1)
WITH_CKPT=0
RUNS=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run) RSYNC_OPTS+=(--dry-run -v) ;;
        --with-checkpoints) WITH_CKPT=1 ;;
        -*) echo "未知参数：${arg}" >&2; exit 2 ;;
        *) RUNS+=("${arg}") ;;
    esac
done

if [[ "${CLUSTER_ARTIFACT_DIR}" == TODO* ]]; then
    echo "[错误] 集群产物路径还是 TODO：在 config/cluster.local.yaml 中填写后运行 python -m scripts.make_cluster_jobs。" >&2
    exit 2
fi
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${CLUSTER_SSH_ALIAS}" true 2>/dev/null; then
    echo "[错误] 连不上 ${CLUSTER_SSH_ALIAS}：检查 Chalmers VPN（以及 ~/.ssh/config 中的 ${CLUSTER_SSH_ALIAS} 别名和密钥）。" >&2
    exit 1
fi

FILTERS=(--include='*/'
         --include='scores_*.npy' --include='per_sensor_err_test.npy'
         --include='threshold.json' --include='train_log.csv' --include='run_manifest.json'
         --include='eval_*.json' --include='eval_*.csv' --include='preprocess.json' --include='scaler.joblib'
         --include='test_index.json' --include='val_sensor_contrib.csv'
         --include='run.log' --include='slurm_logs/*.out')
if [[ "${WITH_CKPT}" == "1" ]]; then
    FILTERS+=(--include='*.pt' --include='*.joblib')
fi
FILTERS+=(--exclude='*')

LOCAL="${ARTIFACT_DIR:-${ROOT}/artifacts}"
mkdir -p "${LOCAL}"
if [[ ${#RUNS[@]} -eq 0 ]]; then
    rsync "${RSYNC_OPTS[@]}" "${FILTERS[@]}" "${CLUSTER_SSH_ALIAS}:${CLUSTER_ARTIFACT_DIR}/" "${LOCAL}/"
else
    for run in "${RUNS[@]}"; do
        mkdir -p "${LOCAL}/${run}"
        rsync "${RSYNC_OPTS[@]}" "${FILTERS[@]}" "${CLUSTER_SSH_ALIAS}:${CLUSTER_ARTIFACT_DIR}/${run}/" "${LOCAL}/${run}/"
    done
fi
echo "[完成] 产物已拉回 ${LOCAL}"
find "${LOCAL}" -name run_manifest.json -not -path '*/smoke/*' | sort | sed 's/^/  /'
