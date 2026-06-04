#!/usr/bin/env bash
set -euo pipefail

# Run the bundled replay workload with the MARS-modified vLLM engine.
# Configure via environment variables; CLI args are forwarded to run_experiment.py.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST_HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
source "${SCRIPT_DIR}/common_runtime.sh"

PYTHON_BIN="${PYTHON_BIN:-$(mars_default_python "${HOST_HOME}/miniconda3/envs/mars/bin/python")}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN=/path/to/python, or create the mars conda env." >&2
  exit 1
fi

export PYTHONPATH="${MARS_DIR}:${MARS_DIR}/vllm-continuum${PYTHONPATH:+:${PYTHONPATH}}"
mars_ensure_vllm_flash_attn_extensions "${MARS_DIR}" "${PYTHON_BIN}"

# Model and engine configuration.
MODEL_PATH="${MODEL_PATH:-${HOST_HOME}/models/Qwen3-Coder-30B-A3B-Instruct}"
DTYPE="${DTYPE:-auto}"

# Configuration for gpt_oss_120b model
_is_gpt_oss_120b_model() {
  local model_path_lc compact
  model_path_lc="${1,,}"
  compact="$(echo "${model_path_lc}" | tr -cd '[:alnum:]')"
  if [[ "${model_path_lc}" == *"gpt-oss"* && "${model_path_lc}" == *"120b"* ]]; then
    return 0
  fi
  [[ "${compact}" == *"gptoss120b"* ]]
}

if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
  MAX_MODEL_LEN_RAW="${MAX_MODEL_LEN}"
elif _is_gpt_oss_120b_model "${MODEL_PATH}"; then
  MAX_MODEL_LEN_RAW="131072"
else
  MAX_MODEL_LEN_RAW="262144"
fi


# argparse `type=int` cannot parse separators like "131,072".
MAX_MODEL_LEN="${MAX_MODEL_LEN_RAW//,/}"
if ! [[ "${MAX_MODEL_LEN}" =~ ^[0-9]+$ ]]; then
  echo "Invalid MAX_MODEL_LEN: ${MAX_MODEL_LEN_RAW} (must be an integer)." >&2
  exit 1
fi
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ROPE_SCALING="${ROPE_SCALING:-}"

# Workload and scheduling configuration.
TASKS_PATH="${TASKS_PATH:-${MARS_DIR}/benchmark/hybrid_workloads/test_workload.jsonl}"
NUM_REQUESTS="${NUM_REQUESTS:-400}"
RPS="${RPS:-0.2}"
EMIT_MODE="${EMIT_MODE:-rps}"
GPU_TIMEOUT_S="${GPU_TIMEOUT_S:-60000}"
KV_EVENTS="${KV_EVENTS:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
MAX_SEQ_LEN_TO_CAPTURE="${MAX_SEQ_LEN_TO_CAPTURE:-}"
CUDA_GRAPH_SIZES="${CUDA_GRAPH_SIZES:-}"
ALLOW_CONCURRENT_RUNS="${ALLOW_CONCURRENT_RUNS:-0}"
MARS_LONG_KV_RATIO="${MARS_LONG_KV_RATIO:-0.95}"
THREAD_POOL="${THREAD_POOL:-on}"

# CPU_GPU and CPU_TOOL should not overlap with each other
CPU_TOOL="${CPU_TOOL:-16-23}"
CPU_GPU="${CPU_GPU:-0-1}"

# Available scheduling policies: fcfs/autellix/infercept/continuum/continuum_dy/mars
SCHEDULING_POLICY="${SCHEDULING_POLICY:-mars}"

# Optional Autellix tail fallback: after N finished jobs, switch to FCFS.
# Set AUTELLIX_TAIL_FCFS=1 to enable. If the threshold is omitted, the last
# 100 requests use FCFS (threshold = NUM_REQUESTS - 100).
AUTELLIX_TAIL_FCFS="${AUTELLIX_TAIL_FCFS:-}"
AUTELLIX_TAIL_FCFS_AFTER_FINISHED="${AUTELLIX_TAIL_FCFS_AFTER_FINISHED:-${AUTELLIX_FCFS_AFTER_FINISHED:-}}"
case "${AUTELLIX_TAIL_FCFS}" in
  1|on|true|yes|y|TRUE|ON|True|YES|Y)
    AUTELLIX_TAIL_FCFS_ENABLED=1
    ;;
  "")
    if [[ -n "${AUTELLIX_TAIL_FCFS_AFTER_FINISHED}" ]]; then
      AUTELLIX_TAIL_FCFS_ENABLED=1
    else
      AUTELLIX_TAIL_FCFS_ENABLED=0
    fi
    ;;
  *)
    AUTELLIX_TAIL_FCFS_ENABLED=0
    ;;
esac
if [[ "${SCHEDULING_POLICY}" == "autellix" && "${AUTELLIX_TAIL_FCFS_ENABLED}" == "1" ]]; then
  if [[ -z "${AUTELLIX_TAIL_FCFS_AFTER_FINISHED}" ]]; then
    AUTELLIX_TAIL_FCFS_AFTER_FINISHED=$((NUM_REQUESTS - 100))
    if (( AUTELLIX_TAIL_FCFS_AFTER_FINISHED < 0 )); then
      AUTELLIX_TAIL_FCFS_AFTER_FINISHED=0
    fi
  fi
else
  AUTELLIX_TAIL_FCFS_AFTER_FINISHED=0
fi

RUNNER="${MARS_DIR}/experiments/replay/vllm_tool_baseline/run_experiment.py"

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  echo "Expected a local HF model directory (e.g. \${HOME}/models/Qwen3-Coder-30B-A3B-Instruct)." >&2
  exit 1
fi

if [[ ! -e "${TASKS_PATH}" ]]; then
  echo "Tasks file not found: ${TASKS_PATH}" >&2
  exit 1
fi

# Running multiple baseline replays on one GPU frequently causes opaque
# startup failures (e.g., low free-memory errors). Keep single-run default.
# Check only active GPU compute processes on guarded GPU(s), so runs on
# different GPUs can proceed concurrently.
if [[ "${ALLOW_CONCURRENT_RUNS}" != "1" && "${ALLOW_CONCURRENT_RUNS}" != "on" && "${ALLOW_CONCURRENT_RUNS}" != "true" ]]; then
  GPU_GUARD_DEVICES="${GPU_GUARD_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
  GPU_GUARD_DEVICES="${GPU_GUARD_DEVICES// /}"
  GPU_PIDS_RAW=""
  if [[ -n "${GPU_GUARD_DEVICES}" && "${GPU_GUARD_DEVICES}" != "all" ]]; then
    IFS=',' read -r -a _gpu_guard_list <<< "${GPU_GUARD_DEVICES}"
    GPU_PIDS_RAW="$(
      for gpu_id in "${_gpu_guard_list[@]}"; do
        [[ -z "${gpu_id}" ]] && continue
        nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true
      done
    )"
  else
    GPU_PIDS_RAW="$(
      nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true
    )"
  fi
  EXISTING_RUN_PIDS="$(
    printf "%s\n" "${GPU_PIDS_RAW}" \
      | awk '{print $1}' \
      | sed '/^[0-9][0-9]*$/!d' \
      | sort -u \
      | while read -r pid; do
          cmd="$(ps -o cmd= -p "${pid}" 2>/dev/null || true)"
          [[ -z "${cmd}" ]] && continue
          if [[ "${cmd}" == *"run_experiment.py"* || "${cmd}" == *"VLLM::EngineCore"* || "${cmd}" == *"vllm"* ]]; then
            printf "%s " "${pid}"
          fi
        done
  )"
  EXISTING_RUN_PIDS="$(echo "${EXISTING_RUN_PIDS}" | sed 's/[[:space:]]*$//')"
  if [[ -n "${EXISTING_RUN_PIDS}" ]]; then
    echo "Another GPU baseline run is already active on guarded GPU(s) [${GPU_GUARD_DEVICES}] (PID(s): ${EXISTING_RUN_PIDS})." >&2
    echo "Stop old runs first, or set ALLOW_CONCURRENT_RUNS=1 to bypass this guard." >&2
    exit 1
  fi
fi

EXTRA_ARGS=()
if [[ "${KV_EVENTS}" == "1" || "${KV_EVENTS}" == "on" || "${KV_EVENTS}" == "true" ]]; then
  EXTRA_ARGS+=(--kv-events)
fi
if [[ "${ENFORCE_EAGER}" == "1" || "${ENFORCE_EAGER}" == "on" || "${ENFORCE_EAGER}" == "true" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
else
  EXTRA_ARGS+=(--no-enforce-eager)
fi
if [[ -n "${MAX_SEQ_LEN_TO_CAPTURE}" ]]; then
  EXTRA_ARGS+=(--max-seq-len-to-capture "${MAX_SEQ_LEN_TO_CAPTURE}")
fi
if [[ -n "${CUDA_GRAPH_SIZES}" ]]; then
  EXTRA_ARGS+=(--cuda-graph-sizes "${CUDA_GRAPH_SIZES}")
fi
if [[ -n "${ROPE_SCALING}" ]]; then
  EXTRA_ARGS+=(--rope-scaling "${ROPE_SCALING}")
fi
if [[ -n "${CPU_TOOL}" ]]; then
  EXTRA_ARGS+=(--cpu-tool "${CPU_TOOL}")
fi
if [[ -n "${CPU_GPU}" ]]; then
  EXTRA_ARGS+=(--cpu-gpu "${CPU_GPU}")
fi

exec "${PYTHON_BIN}" "${RUNNER}" \
  --tasks "${TASKS_PATH}" \
  --model-path "${MODEL_PATH}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --emit-mode "${EMIT_MODE}" \
  --num-requests "${NUM_REQUESTS}" \
  --rps "${RPS}" \
  --gpu-timeout-s "${GPU_TIMEOUT_S}" \
  --scheduling-policy "${SCHEDULING_POLICY}" \
  --autellix-tail-fcfs-after-finished "${AUTELLIX_TAIL_FCFS_AFTER_FINISHED}" \
  --log-stats \
  --mars-active-pool-size 30 \
  --mars-long-prefill-tokens 50000 \
  --mars-long-kv-ratio "${MARS_LONG_KV_RATIO}" \
  --thread_pool "${THREAD_POOL}" \
  --mars-cpu-backlog-low 1 \
  --mars-cpu-backlog-high 4 \
  "${EXTRA_ARGS[@]}" \
  "$@"
