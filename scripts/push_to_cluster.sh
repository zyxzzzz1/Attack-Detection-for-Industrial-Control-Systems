#!/bin/bash
# 把代码和配置推送到集群（rsync）。只推送 git 跟踪的文件：
#   - 不含 data/raw/、data/intel/、artifacts/、.venv/（都不在 git 里）；
#   - 含 data/samples/、data/mock/ 等入库的小文件，集群上的完整 pytest 需要它们。
# 同时写入 .git_commit（提交号），供集群上的 run_manifest.json 记录。
# 工作区有未提交的修改（含未跟踪的新文件）时拒绝推送并列出这些文件，防止集群上运行的代码与提交号对不上；
# 加 --allow-dirty 才放行，此时 .git_commit 和 run_manifest.json 中标记 dirty=true。
#
# 用法：
#   scripts/push_to_cluster.sh [--dry-run] [--allow-dirty]     # 推送代码
#   scripts/push_to_cluster.sh --data          # 推送 HAI 数据（集群登录节点不能上网时用）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 集群路径等个人信息不入库：真实值在 cluster/local/cluster_env.sh（由 make_cluster_jobs 按 config/cluster.local.yaml 生成）；
# 没有时退回入库的 cluster/cluster_env.sh（路径为 TODO，下面会因此拒绝执行）。
CLUSTER_ENV="${ROOT}/cluster/cluster_env.sh"
if [[ -f "${ROOT}/cluster/local/cluster_env.sh" ]]; then CLUSTER_ENV="${ROOT}/cluster/local/cluster_env.sh"; fi
# shellcheck source=/dev/null
source "${CLUSTER_ENV}"

MODE=code
ALLOW_DIRTY=0
RSYNC_OPTS=(-az --info=stats1)
for arg in "$@"; do
    case "${arg}" in
        --dry-run) RSYNC_OPTS+=(--dry-run -v) ;;
        --data) MODE=data ;;
        --allow-dirty) ALLOW_DIRTY=1 ;;
        *) echo "未知参数：${arg}" >&2; exit 2 ;;
    esac
done

for key in CLUSTER_CODE_DIR CLUSTER_DATA_DIR; do
    if [[ "${!key}" == TODO* ]]; then
        echo "[错误] 集群路径还是 TODO（${key}）：在 config/cluster.local.yaml 中填写（示例见 config/cluster.local.example.yaml），" >&2
        echo "       再运行 python -m scripts.make_cluster_jobs 生成 cluster/local/。" >&2
        exit 2
    fi
done

# 工作区检查放在连接之前：不需要 VPN 就能发现问题
cd "${ROOT}"
DIRTY="$(git status --porcelain)"
if [[ "${MODE}" == "code" && -n "${DIRTY}" && "${ALLOW_DIRTY}" != "1" ]]; then
    echo "[错误] 工作区有未提交的修改，拒绝推送（集群上运行的代码会与提交号对不上）：" >&2
    echo "${DIRTY}" | sed 's/^/    /' >&2
    echo "       请先提交；确实要推送未提交的代码时加 --allow-dirty（manifest 中会标记 dirty=true）。" >&2
    exit 3
fi

if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${CLUSTER_SSH_ALIAS}" true 2>/dev/null; then
    echo "[错误] 连不上 ${CLUSTER_SSH_ALIAS}：检查 Chalmers VPN（以及 ~/.ssh/config 中的 ${CLUSTER_SSH_ALIAS} 别名和密钥）。" >&2
    exit 1
fi

if [[ "${MODE}" == "data" ]]; then
    LOCAL_DATA="${HAI_DATA_DIR:-${ROOT}/data/raw/hai-21.03}"
    ssh "${CLUSTER_SSH_ALIAS}" "mkdir -p '${CLUSTER_DATA_DIR}'"
    rsync "${RSYNC_OPTS[@]}" --include='*.csv.gz' --include='download_manifest.json' --exclude='*' \
        "${LOCAL_DATA}/" "${CLUSTER_SSH_ALIAS}:${CLUSTER_DATA_DIR}/"
    echo "[完成] 数据已推送到 ${CLUSTER_SSH_ALIAS}:${CLUSTER_DATA_DIR}（在集群上用 srun 运行核对，见 runbook）"
    exit 0
fi

COMMIT="$(git rev-parse HEAD)"
if [[ -n "${DIRTY}" ]]; then
    COMMIT="${COMMIT} dirty"
    echo "[警告] --allow-dirty：推送含未提交修改的代码，run_manifest.json 会记为 dirty=true。不要用于正式运行。" >&2
fi
LIST="$(mktemp)"
trap 'rm -f "${LIST}" "${ROOT}/.git_commit"' EXIT
if [[ -n "${DIRTY}" ]]; then
    git ls-files -co --exclude-standard > "${LIST}"   # 放行时连同未跟踪的新文件一起推（仍遵守 .gitignore）
else
    git ls-files > "${LIST}"
fi
echo "${COMMIT}" > "${ROOT}/.git_commit"
echo ".git_commit" >> "${LIST}"

ssh "${CLUSTER_SSH_ALIAS}" "mkdir -p '${CLUSTER_CODE_DIR}'"
rsync "${RSYNC_OPTS[@]}" --files-from="${LIST}" "${ROOT}/" "${CLUSTER_SSH_ALIAS}:${CLUSTER_CODE_DIR}/"
# 入库的作业脚本里路径是 TODO；真实的作业脚本和 cluster_env.sh 在 cluster/local/（不入库），推送到集群上的 cluster/ 覆盖它们。
# 这些文件由入库的模板和 config/cluster.local.yaml 生成，不属于代码改动，不影响 dirty 标记。
if [[ -d "${ROOT}/cluster/local" ]]; then
    rsync "${RSYNC_OPTS[@]}" --include='*.sbatch' --include='cluster_env.sh' --exclude='*' \
        "${ROOT}/cluster/local/" "${CLUSTER_SSH_ALIAS}:${CLUSTER_CODE_DIR}/cluster/"
fi
echo "[完成] 代码已推送到 ${CLUSTER_SSH_ALIAS}:${CLUSTER_CODE_DIR}（提交 ${COMMIT}）"
echo "[提示] 本地删除的文件不会在集群上自动删除；需要时在集群上手动清理。"
