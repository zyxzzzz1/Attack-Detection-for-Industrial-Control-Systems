# 作业脚本的公共部分，由 cluster/*.sbatch source。不要直接执行。
# 调用前需设置：SMOKE（1 = 本地冒烟，不经过调度器；0 = 集群作业）。
# 设置：ROOT、SMOKE_ARGS，导出 HAI_DATA_DIR / ARTIFACT_DIR，激活虚拟环境并切到代码目录。

_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_common_dir}/cluster_env.sh"

if [[ "${SMOKE}" == "1" ]]; then
    # 冒烟：用脚本所在的代码目录和其中的 .venv；产物写到 smoke/ 子目录，与正式产物分开
    ROOT="$(cd "${_common_dir}/.." && pwd)"
    VENV_DIR="${ROOT}/.venv"
    if [[ -n "${SLURM_JOB_ID:-}" && "${CLUSTER_TODO_COUNT}" == "0" ]]; then
        # 在集群上通过 srun 做冒烟（登录节点上不允许直接跑）：数据和产物用集群上的目录
        export HAI_DATA_DIR="${HAI_DATA_DIR:-${CLUSTER_DATA_DIR}}"
        export ARTIFACT_DIR="${ARTIFACT_DIR:-${CLUSTER_ARTIFACT_DIR}/smoke}"
    else
        export ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT}/artifacts/smoke}"
    fi
    SMOKE_ARGS=(--smoke)
else
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "[错误] 登录节点上禁止直接运行计算任务：请用 sbatch 或 srun 提交（本地冒烟请加 --smoke）。" >&2
        exit 2
    fi
    if [[ "${CLUSTER_TODO_COUNT}" != "0" ]]; then
        echo "[错误] config/cluster.yaml 还有 ${CLUSTER_TODO_COUNT} 个影响作业的 TODO：${CLUSTER_TODO_KEYS}" >&2
        echo "       补全后运行 python -m scripts.make_cluster_jobs 重新生成，再推送到集群。" >&2
        exit 2
    fi
    ROOT="${CLUSTER_CODE_DIR}"
    VENV_DIR="${CLUSTER_VENV_DIR}"
    export HAI_DATA_DIR="${CLUSTER_DATA_DIR}"
    export ARTIFACT_DIR="${CLUSTER_ARTIFACT_DIR}"
    SMOKE_ARGS=()
fi

cd "${ROOT}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
mkdir -p "${ARTIFACT_DIR}"

# Minerva 上没有申请 GPU 的作业也能看到节点上的显卡（设备没有被隔离）：没申请就把显卡藏起来，
# 避免 CPU 作业（PCA、IsolationForest、CPU 版 pytest）占用别人申请的 GPU。申请了 GPU 时 Slurm 会设置这个变量。
if [[ -n "${SLURM_JOB_ID:-}" && -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi

# 线程数不超过申请的 CPU 数
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-4}}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONUNBUFFERED=1

echo "[作业] $(date '+%F %T') 主机 $(hostname) 作业 ${SLURM_JOB_ID:-本地} array ${SLURM_ARRAY_TASK_ID:--} 冒烟 ${SMOKE}"
echo "[作业] 代码 ${ROOT}  数据 ${HAI_DATA_DIR:-（config 默认）}  产物 ${ARTIFACT_DIR}  python $(python --version 2>&1)"
