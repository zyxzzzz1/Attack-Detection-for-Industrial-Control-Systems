# Shared part of the job scripts, sourced by cluster/*.sbatch. Do not execute directly.
# Must be set before sourcing: SMOKE (1 = local smoke run without the scheduler; 0 = cluster job).
# Sets ROOT and SMOKE_ARGS, exports HAI_DATA_DIR / ARTIFACT_DIR, activates the virtualenv and cds into the code directory.

_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_common_dir}/cluster_env.sh"

if [[ "${SMOKE}" == "1" ]]; then
    # Smoke: use the code directory containing this script and its .venv; artifacts go to the smoke/ subdirectory, separate from official ones
    ROOT="$(cd "${_common_dir}/.." && pwd)"
    VENV_DIR="${ROOT}/.venv"
    if [[ -n "${SLURM_JOB_ID:-}" && "${CLUSTER_TODO_COUNT}" == "0" ]]; then
        # Smoke run on the cluster via srun (not allowed directly on the login node): data and artifacts use the cluster directories
        export HAI_DATA_DIR="${HAI_DATA_DIR:-${CLUSTER_DATA_DIR}}"
        export ARTIFACT_DIR="${ARTIFACT_DIR:-${CLUSTER_ARTIFACT_DIR}/smoke}"
    else
        export ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT}/artifacts/smoke}"
    fi
    SMOKE_ARGS=(--smoke)
else
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "[ERROR] running compute tasks directly on a login node is forbidden: submit with sbatch or srun (add --smoke for a local smoke run)." >&2
        exit 2
    fi
    if [[ "${CLUSTER_TODO_COUNT}" != "0" ]]; then
        echo "[ERROR] config/cluster.yaml still has ${CLUSTER_TODO_COUNT} job-affecting TODOs: ${CLUSTER_TODO_KEYS}" >&2
        echo "        fill them in, regenerate with python -m scripts.make_cluster_jobs and push to the cluster again." >&2
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

# On Minerva, jobs that did not request a GPU can still see the node's GPUs (devices are not isolated): hide them when none was requested,
# so CPU jobs (PCA, IsolationForest, CPU pytest) do not use GPUs requested by others. Slurm sets this variable when a GPU is requested.
if [[ -n "${SLURM_JOB_ID:-}" && -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi

# Thread count must not exceed the number of requested CPUs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-4}}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONUNBUFFERED=1

echo "[JOB] $(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-local} array ${SLURM_ARRAY_TASK_ID:--} smoke ${SMOKE}"
echo "[JOB] code ${ROOT}  data ${HAI_DATA_DIR:-(config default)}  artifacts ${ARTIFACT_DIR}  python $(python --version 2>&1)"
