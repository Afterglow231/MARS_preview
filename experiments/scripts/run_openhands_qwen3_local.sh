#!/usr/bin/env bash
set -euo pipefail

# Start a local MARS/vLLM OpenAI-compatible server, then run the bundled
# OpenHands workload runner against it. Set USE_EXISTING_VLLM=1 to skip local
# server startup and use BASE_URL instead.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST_HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
source "${SCRIPT_DIR}/common_runtime.sh"

MARS_PYTHON="${MARS_PYTHON:-$(mars_default_python "${HOST_HOME}/miniconda3/envs/mars/bin/python")}"
OPENHANDS_PYTHON="${OPENHANDS_PYTHON:-$(mars_default_python "${HOST_HOME}/.venvs/openhands313/bin/python")}"
OPENHANDS_RUNNER="${MARS_DIR}/experiments/openhands/run_task_workload.py"

# Resources.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES

# CPU_GPU and CPU_TOOL should not overlap with each other
CPU_TOOL="${CPU_TOOL:-16-23}"
CPU_GPU="${CPU_GPU:-0-1}"
CPU_CORES="${CPU_CORES:-${CPU_GPU},${CPU_TOOL}}"
TP="${TP:-}"

# Model and vLLM.
MODEL_PATH="${MODEL_PATH:-${HOST_HOME}/models/Qwen3-Coder-30B-A3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-30B-A3B-Instruct}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
DISABLE_UVICORN_ACCESS_LOG="${DISABLE_UVICORN_ACCESS_LOG:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
MAX_SEQ_LEN_TO_CAPTURE="${MAX_SEQ_LEN_TO_CAPTURE:-}"
CUDA_GRAPH_SIZES="${CUDA_GRAPH_SIZES:-}"
ROPE_SCALING="${ROPE_SCALING:-}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-}"

# Scheduling.
SCHEDULING_POLICY="${SCHEDULING_POLICY:-mars}"
THREAD_POOL="${THREAD_POOL:-on}"
ALLOW_CONCURRENT_RUNS="${ALLOW_CONCURRENT_RUNS:-0}"
AUTELLIX_TAIL_FCFS="${AUTELLIX_TAIL_FCFS:-}"
AUTELLIX_TAIL_FCFS_AFTER_FINISHED="${AUTELLIX_TAIL_FCFS_AFTER_FINISHED:-${AUTELLIX_FCFS_AFTER_FINISHED:-}}"
MARS_ACTIVE_POOL_SIZE="${MARS_ACTIVE_POOL_SIZE:-8}"
MARS_LONG_PREFILL_TOKENS="${MARS_LONG_PREFILL_TOKENS:-50000}"
MARS_CPU_BACKLOG_LOW="${MARS_CPU_BACKLOG_LOW:-1}"
MARS_CPU_BACKLOG_HIGH="${MARS_CPU_BACKLOG_HIGH:-4}"
MARS_KV_TARGET_RATIO="${MARS_KV_TARGET_RATIO:-0.9}"
MARS_KV_STAT_INTERVAL_S="${MARS_KV_STAT_INTERVAL_S:-0.2}"
MARS_WINDOW_MIN="${MARS_WINDOW_MIN:-2}"
MARS_WINDOW_INIT="${MARS_WINDOW_INIT:-8}"
MARS_WINDOW_INC="${MARS_WINDOW_INC:-1}"
MARS_WINDOW_DEC_FACTOR="${MARS_WINDOW_DEC_FACTOR:-0.7}"
MARS_CONTROL_INTERVAL_S="${MARS_CONTROL_INTERVAL_S:-0.2}"
MARS_CPU_QUEUE_WAIT_HIGH_S="${MARS_CPU_QUEUE_WAIT_HIGH_S:-5.0}"
MARS_CPU_QUEUE_WAIT_LOW_S="${MARS_CPU_QUEUE_WAIT_LOW_S:-2.0}"
MARS_KV_MAX_STALE_S="${MARS_KV_MAX_STALE_S:-2.0}"
MARS_LONG_MAX_INFLIGHT="${MARS_LONG_MAX_INFLIGHT:-1}"
MARS_TAIL_MAX_INFLIGHT="${MARS_TAIL_MAX_INFLIGHT:-3}"
MARS_TAIL_KV_BUDGET_RATIO="${MARS_TAIL_KV_BUDGET_RATIO:-1.0}"
MARS_NO_KV_ACTIVE_LIMIT="${MARS_NO_KV_ACTIVE_LIMIT:-2}"

# Workload.
TASKS_PATH="${TASKS_PATH:-${MARS_DIR}/benchmark/hybrid_workloads/test_workload.jsonl}"
NUM_REQUESTS="${NUM_REQUESTS:-}"
MAX_REQUESTS="${MAX_REQUESTS:-}"
RPS="${RPS:-0.2}"
MAX_WORKERS="${MAX_WORKERS:-0}"
SHUFFLE="${SHUFFLE:-0}"
SEED="${SEED:-0}"
EMIT_MODE="${EMIT_MODE:-rps}"
GPU_TIMEOUT_S="${GPU_TIMEOUT_S:-60000}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-2048}"
LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-3000}"
LLM_NUM_RETRIES="${LLM_NUM_RETRIES:-3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-40}"
TERMINAL_NO_CHANGE_TIMEOUT_S="${TERMINAL_NO_CHANGE_TIMEOUT_S:-30}"
TERMINAL_TYPE="${TERMINAL_TYPE:-}"
PREFILL_TOKENIZER="${PREFILL_TOKENIZER:-${MODEL_PATH}}"
API_KEY="${API_KEY:-EMPTY}"

# Output.
OUTPUT_ROOT="${OUTPUT_ROOT:-${MARS_DIR}/outputs/openhands}"
RUN_LABEL="${RUN_LABEL:-${RUN_NAME:-openhands_qwen3_317}}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/${SCHEDULING_POLICY}/${RUN_LABEL}/${TIMESTAMP}}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${OUT_DIR}/workspaces}"
LAUNCHER_LOG_PATH="${LAUNCHER_LOG_PATH:-${OUT_DIR}/launcher.log}"
WORKLOAD_LOG_PATH="${WORKLOAD_LOG_PATH:-${OUT_DIR}/openhands_workload.log}"
VLLM_LOG_PATH="${VLLM_LOG_PATH:-${OUT_DIR}/vllm_server.log}"

# Server lifecycle.
USE_EXISTING_VLLM="${USE_EXISTING_VLLM:-0}"
BASE_URL="${BASE_URL:-}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-}"
VLLM_DP_MASTER_PORT="${VLLM_DP_MASTER_PORT:-}"
KEEP_VLLM_SERVER="${KEEP_VLLM_SERVER:-0}"
SERVER_TERM_GRACE_S="${SERVER_TERM_GRACE_S:-10}"
KV_EVENTS="${KV_EVENTS:-1}"
KV_EVENTS_ENDPOINT="${KV_EVENTS_ENDPOINT:-}"
KV_EVENTS_CONFIG="${KV_EVENTS_CONFIG:-}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-}"

die() {
  echo "$*" >&2
  exit 1
}

log_kv() {
  printf '%s=%s\n' "$1" "$2"
}

resolve_autellix_tail_threshold() {
  local enabled=0
  if mars_bool_true "${AUTELLIX_TAIL_FCFS}"; then
    enabled=1
  elif [[ -z "${AUTELLIX_TAIL_FCFS}" && -n "${AUTELLIX_TAIL_FCFS_AFTER_FINISHED}" ]]; then
    enabled=1
  fi

  if [[ "${SCHEDULING_POLICY}" != "autellix" || "${enabled}" != "1" ]]; then
    echo 0
    return 0
  fi
  if [[ -n "${AUTELLIX_TAIL_FCFS_AFTER_FINISHED}" ]]; then
    echo "${AUTELLIX_TAIL_FCFS_AFTER_FINISHED}"
    return 0
  fi
  if [[ -n "${MAX_REQUESTS}" ]]; then
    local threshold=$((MAX_REQUESTS - 100))
    (( threshold < 0 )) && threshold=0
    echo "${threshold}"
  else
    echo 0
  fi
}

existing_guarded_run_pids() {
  local gpu_spec="${GPU_GUARD_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
  gpu_spec="${gpu_spec// /}"
  command -v nvidia-smi >/dev/null 2>&1 || return 0

  local raw_pids
  if [[ -n "${gpu_spec}" && "${gpu_spec}" != "all" ]]; then
    IFS=',' read -r -a _gpu_guard_list <<< "${gpu_spec}"
    raw_pids="$(
      for gpu_id in "${_gpu_guard_list[@]}"; do
        [[ -z "${gpu_id}" ]] && continue
        nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true
      done
    )"
  else
    raw_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
  fi

  printf '%s\n' "${raw_pids}" \
    | awk '{print $1}' \
    | sed '/^[0-9][0-9]*$/!d' \
    | sort -u \
    | while read -r pid; do
        cmd="$(ps -o cmd= -p "${pid}" 2>/dev/null || true)"
        [[ -z "${cmd}" ]] && continue
        if [[ "${cmd}" == *"run_task_workload.py"* \
          || "${cmd}" == *"vllm.entrypoints.openai.api_server"* \
          || "${cmd}" == *"vllm_api_server_mars.py"* \
          || "${cmd}" == *"VLLM::EngineCore"* ]]; then
          printf '%s ' "${pid}"
        fi
      done \
    | sed 's/[[:space:]]*$//'
}

preflight() {
  [[ -x "${MARS_PYTHON}" ]] || die "MARS_PYTHON not found or not executable: ${MARS_PYTHON}"
  [[ -x "${OPENHANDS_PYTHON}" ]] || die "OPENHANDS_PYTHON not found or not executable: ${OPENHANDS_PYTHON}"
  [[ -f "${OPENHANDS_RUNNER}" ]] || die "OpenHands workload runner not found: ${OPENHANDS_RUNNER}"
  [[ -f "${TASKS_PATH}" ]] || die "Tasks file not found: ${TASKS_PATH}"

  if [[ -n "${NUM_REQUESTS}" && -z "${MAX_REQUESTS}" ]]; then
    MAX_REQUESTS="${NUM_REQUESTS}"
  fi

  if ! mars_bool_true "${USE_EXISTING_VLLM}"; then
    [[ -d "${MODEL_PATH}" ]] || die "Local model path not found: ${MODEL_PATH}"
    mars_ensure_vllm_flash_attn_extensions "${MARS_DIR}" "${MARS_PYTHON}"
  elif [[ -z "${BASE_URL}" ]]; then
    die "BASE_URL is required when USE_EXISTING_VLLM=1"
  fi
}

configure_runtime() {
  [[ -z "${TP}" ]] && TP="$(mars_infer_tp_from_devices "${CUDA_VISIBLE_DEVICES}")"
  AUTELLIX_TAIL_FCFS_AFTER_FINISHED="$(resolve_autellix_tail_threshold)"

  if mars_bool_true "${KV_EVENTS}" && [[ -z "${KV_EVENTS_CONFIG}" ]]; then
    [[ -z "${KV_EVENTS_ENDPOINT}" ]] && KV_EVENTS_ENDPOINT="tcp://127.0.0.1:$(mars_pick_free_port "${MARS_PYTHON}")"
    local bind_endpoint="${KV_EVENTS_ENDPOINT/tcp:\/\/127.0.0.1:/tcp://*:}"
    [[ "${bind_endpoint}" == "${KV_EVENTS_ENDPOINT}" ]] && bind_endpoint="${KV_EVENTS_ENDPOINT/tcp:\/\/localhost:/tcp://*:}"
    KV_EVENTS_CONFIG="$(printf '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"%s","topic":"kv-events"}' "${bind_endpoint}")"
  fi

  ADDITIONAL_CONFIG="$(
    mars_merge_additional_config \
      "${MARS_PYTHON}" \
      "${ADDITIONAL_CONFIG}" \
      "${MARS_ACTIVE_POOL_SIZE}" \
      "${MARS_CPU_BACKLOG_LOW}" \
      "${MARS_CPU_BACKLOG_HIGH}"
  )"
}

SERVER_PID=""
SERVER_IS_PROCESS_GROUP=0
CLEANUP_DONE=0

cleanup() {
  [[ "${CLEANUP_DONE}" == "1" ]] && return 0
  CLEANUP_DONE=1
  if [[ -z "${SERVER_PID}" ]] || mars_bool_true "${KEEP_VLLM_SERVER}"; then
    return 0
  fi

  if [[ "${SERVER_IS_PROCESS_GROUP}" == "1" ]]; then
    kill -- "-${SERVER_PID}" >/dev/null 2>&1 || true
    sleep "${SERVER_TERM_GRACE_S}" || true
    kill -KILL -- "-${SERVER_PID}" >/dev/null 2>&1 || true
  else
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  wait "${SERVER_PID}" >/dev/null 2>&1 || true
}

handle_signal() {
  local exit_code="$1"
  trap - EXIT
  cleanup
  exit "${exit_code}"
}

build_vllm_command() {
  local entrypoint="vllm.entrypoints.openai.api_server"
  VLLM_CMD=("${MARS_PYTHON}" -m "${entrypoint}")
  REQUIRE_KV_STATS_READY=0
  if [[ "${SCHEDULING_POLICY}" == "mars" ]]; then
    entrypoint="${MARS_DIR}/experiments/openhands/vllm_api_server_mars.py"
    VLLM_CMD=("${MARS_PYTHON}" "${entrypoint}")
    REQUIRE_KV_STATS_READY=1
  fi
  VLLM_ENTRYPOINT="${entrypoint}"

  VLLM_CMD+=(
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --model "${MODEL_PATH}"
    --served-model-name "${MODEL_NAME}"
    --dtype "${DTYPE}"
    --tensor-parallel-size "${TP}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --enable-force-include-usage
    --disable-log-stats
    --scheduling-policy "${SCHEDULING_POLICY}"
    --autellix-tail-fcfs-after-finished "${AUTELLIX_TAIL_FCFS_AFTER_FINISHED}"
  )
  [[ -n "${KV_EVENTS_CONFIG}" ]] && VLLM_CMD+=(--kv-events-config "${KV_EVENTS_CONFIG}")
  [[ -n "${ADDITIONAL_CONFIG}" ]] && VLLM_CMD+=(--additional-config "${ADDITIONAL_CONFIG}")
  [[ -n "${COMPILATION_CONFIG}" ]] && VLLM_CMD+=(--compilation-config "${COMPILATION_CONFIG}")
  mars_bool_true "${DISABLE_UVICORN_ACCESS_LOG}" && VLLM_CMD+=(--disable-uvicorn-access-log)
  mars_bool_true "${ENFORCE_EAGER}" && VLLM_CMD+=(--enforce-eager)
  [[ -n "${MAX_SEQ_LEN_TO_CAPTURE}" ]] && VLLM_CMD+=(--max-seq-len-to-capture "${MAX_SEQ_LEN_TO_CAPTURE}")
  [[ -n "${CUDA_GRAPH_SIZES}" ]] && VLLM_CMD+=(--cuda-graph-sizes "${CUDA_GRAPH_SIZES}")
  [[ -n "${ROPE_SCALING}" ]] && VLLM_CMD+=(--rope-scaling "${ROPE_SCALING}")
  mars_bool_true "${ENABLE_AUTO_TOOL_CHOICE}" && VLLM_CMD+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
  mars_bool_true "${TRUST_REMOTE_CODE}" && VLLM_CMD+=(--trust-remote-code)
}

check_local_vllm_slot() {
  GPU_GUARD_DEVICES="${GPU_GUARD_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
  GPU_GUARD_DEVICES="${GPU_GUARD_DEVICES// /}"
  if ! mars_bool_true "${ALLOW_CONCURRENT_RUNS}"; then
    local pids
    pids="$(existing_guarded_run_pids)"
    [[ -n "${pids}" ]] && die "Another OpenHands/vLLM run is active on GPU(s) [${GPU_GUARD_DEVICES}] (PID(s): ${pids}). Set ALLOW_CONCURRENT_RUNS=1 to bypass."
  fi
}

start_local_vllm() {
  [[ -z "${VLLM_PORT}" ]] && VLLM_PORT="$(mars_pick_free_port "${MARS_PYTHON}")"
  check_local_vllm_slot
  mars_assert_local_port_available "${MARS_PYTHON}" "${VLLM_HOST}" "${VLLM_PORT}" "vLLM"
  [[ -n "${VLLM_DP_MASTER_PORT}" ]] && mars_assert_local_port_range_available "${MARS_PYTHON}" "${VLLM_HOST}" "${VLLM_DP_MASTER_PORT}" 10 "vLLM DP master"

  BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
  build_vllm_command

  echo "== Starting local vLLM =="
  log_kv CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"
  log_kv CPU_CORES "${CPU_CORES}"
  log_kv MODEL_PATH "${MODEL_PATH}"
  log_kv BASE_URL "${BASE_URL}"
  log_kv VLLM_ENTRYPOINT "${VLLM_ENTRYPOINT}"
  log_kv TP "${TP}"
  log_kv SCHEDULING_POLICY "${SCHEDULING_POLICY}"
  log_kv KV_EVENTS_ENDPOINT "${KV_EVENTS_ENDPOINT}"
  log_kv VLLM_LOG_PATH "${VLLM_LOG_PATH}"

  local launch_cmd=(
    env
    -u VLLM_PORT
    PYTHONPATH="${MARS_DIR}:${MARS_DIR}/vllm-continuum${PYTHONPATH:+:${PYTHONPATH}}"
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    RUN_OUTPUT_DIR="${OUT_DIR}"
  )
  [[ -n "${VLLM_DP_MASTER_PORT}" ]] && launch_cmd+=(VLLM_DP_MASTER_PORT="${VLLM_DP_MASTER_PORT}")
  if command -v taskset >/dev/null 2>&1 && [[ -n "${CPU_CORES}" ]]; then
    launch_cmd+=(taskset -c "${CPU_CORES}")
  fi
  launch_cmd+=("${VLLM_CMD[@]}")

  if command -v setsid >/dev/null 2>&1; then
    setsid "${launch_cmd[@]}" >"${VLLM_LOG_PATH}" 2>&1 &
    SERVER_IS_PROCESS_GROUP=1
  else
    "${launch_cmd[@]}" >"${VLLM_LOG_PATH}" 2>&1 &
  fi
  SERVER_PID=$!

  mars_wait_for_vllm "${MARS_PYTHON}" "${BASE_URL}" 300 "${SERVER_PID}" "${VLLM_LOG_PATH}" "${REQUIRE_KV_STATS_READY}"
  echo "== vLLM ready =="
  log_kv BASE_URL "${BASE_URL}"
}

build_workload_command() {
  WORKLOAD_EXTRA_ARGS=()
  [[ -n "${MAX_REQUESTS}" ]] && WORKLOAD_EXTRA_ARGS+=(--max-requests "${MAX_REQUESTS}")
  mars_bool_true "${SHUFFLE}" && WORKLOAD_EXTRA_ARGS+=(--shuffle)
  [[ -n "${TOP_P}" ]] && WORKLOAD_EXTRA_ARGS+=(--top-p "${TOP_P}")
  [[ -n "${TERMINAL_TYPE}" ]] && WORKLOAD_EXTRA_ARGS+=(--terminal-type "${TERMINAL_TYPE}")
  [[ -n "${CPU_CORES}" ]] && WORKLOAD_EXTRA_ARGS+=(--cpu-affinity "${CPU_CORES}")
  [[ -n "${PREFILL_TOKENIZER}" ]] && WORKLOAD_EXTRA_ARGS+=(--prefill-tokenizer "${PREFILL_TOKENIZER}")
  if mars_bool_true "${KV_EVENTS}"; then
    WORKLOAD_EXTRA_ARGS+=(--kv-events)
    [[ -n "${KV_EVENTS_ENDPOINT}" ]] && WORKLOAD_EXTRA_ARGS+=(--kv-events-endpoint "${KV_EVENTS_ENDPOINT}")
  fi

  WORKLOAD_CMD=(
    env
    PYTHONPATH="${MARS_DIR}/experiments/openhands:${MARS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  )
  if command -v taskset >/dev/null 2>&1 && [[ -n "${CPU_CORES}" ]]; then
    WORKLOAD_CMD+=(taskset -c "${CPU_CORES}")
  fi
  WORKLOAD_CMD+=(
    "${MARS_PYTHON}"
    "${OPENHANDS_RUNNER}"
    --tasks "${TASKS_PATH}"
    --out-dir "${OUT_DIR}"
    --workspace-root "${WORKSPACE_ROOT}"
    --model "${MODEL_NAME}"
    --base-url "${BASE_URL}"
    --api-key "${API_KEY}"
    --openhands-python "${OPENHANDS_PYTHON}"
    --rps "${RPS}"
    --emit-mode "${EMIT_MODE}"
    --max-workers "${MAX_WORKERS}"
    --seed "${SEED}"
    --gpu-timeout-s "${GPU_TIMEOUT_S}"
    --scheduling-policy "${SCHEDULING_POLICY}"
    --temperature "${TEMPERATURE}"
    --max-output-tokens "${MAX_OUTPUT_TOKENS}"
    --llm-timeout-s "${LLM_TIMEOUT_S}"
    --llm-num-retries "${LLM_NUM_RETRIES}"
    --max-iterations "${MAX_ITERATIONS}"
    --terminal-no-change-timeout-s "${TERMINAL_NO_CHANGE_TIMEOUT_S}"
    --thread-pool "${THREAD_POOL}"
    --cpu-gpu "${CPU_GPU}"
    --cpu-tool "${CPU_TOOL}"
    --autellix-tail-fcfs-after-finished "${AUTELLIX_TAIL_FCFS_AFTER_FINISHED}"
    --mars-active-pool-size "${MARS_ACTIVE_POOL_SIZE}"
    --mars-long-prefill-tokens "${MARS_LONG_PREFILL_TOKENS}"
    --mars-cpu-backlog-low "${MARS_CPU_BACKLOG_LOW}"
    --mars-cpu-backlog-high "${MARS_CPU_BACKLOG_HIGH}"
    --mars-kv-target-ratio "${MARS_KV_TARGET_RATIO}"
    --mars-kv-stat-interval-s "${MARS_KV_STAT_INTERVAL_S}"
    --mars-window-min "${MARS_WINDOW_MIN}"
    --mars-window-init "${MARS_WINDOW_INIT}"
    --mars-window-inc "${MARS_WINDOW_INC}"
    --mars-window-dec-factor "${MARS_WINDOW_DEC_FACTOR}"
    --mars-control-interval-s "${MARS_CONTROL_INTERVAL_S}"
    --mars-cpu-queue-wait-high-s "${MARS_CPU_QUEUE_WAIT_HIGH_S}"
    --mars-cpu-queue-wait-low-s "${MARS_CPU_QUEUE_WAIT_LOW_S}"
    --mars-kv-max-stale-s "${MARS_KV_MAX_STALE_S}"
    --mars-long-max-inflight "${MARS_LONG_MAX_INFLIGHT}"
    --mars-tail-max-inflight "${MARS_TAIL_MAX_INFLIGHT}"
    --mars-tail-kv-budget-ratio "${MARS_TAIL_KV_BUDGET_RATIO}"
    --mars-no-kv-active-limit "${MARS_NO_KV_ACTIVE_LIMIT}"
    "${WORKLOAD_EXTRA_ARGS[@]}"
    "$@"
  )
}

run_workload() {
  echo "== Running OpenHands workload =="
  log_kv TASKS_PATH "${TASKS_PATH}"
  log_kv OUT_DIR "${OUT_DIR}"
  log_kv BASE_URL "${BASE_URL}"
  log_kv RPS "${RPS}"
  log_kv MAX_REQUESTS "${MAX_REQUESTS:-all}"
  log_kv MAX_WORKERS "${MAX_WORKERS}"
  log_kv CPU_CORES "${CPU_CORES}"
  log_kv SCHEDULING_POLICY "${SCHEDULING_POLICY}"
  log_kv WORKLOAD_LOG_PATH "${WORKLOAD_LOG_PATH}"

  set +e
  "${WORKLOAD_CMD[@]}" 2>&1 | tee -a "${WORKLOAD_LOG_PATH}"
  local rc="${PIPESTATUS[0]}"
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    echo "OpenHands workload failed with exit code ${rc}." >&2
    echo "Check ${WORKLOAD_LOG_PATH} and ${VLLM_LOG_PATH} for details." >&2
    exit "${rc}"
  fi
  [[ -f "${OUT_DIR}/events.jsonl" ]] || die "OpenHands workload exited without producing events.jsonl. Check ${WORKLOAD_LOG_PATH} and ${VLLM_LOG_PATH}."

  echo "OpenHands workload finished successfully."
  echo "Artifacts: ${OUT_DIR}"
}

main() {
  preflight
  mkdir -p "${OUT_DIR}" "${WORKSPACE_ROOT}"
  : > "${LAUNCHER_LOG_PATH}"
  : > "${WORKLOAD_LOG_PATH}"
  mars_bool_true "${USE_EXISTING_VLLM}" || : > "${VLLM_LOG_PATH}"

  exec > >(tee -a "${LAUNCHER_LOG_PATH}") 2>&1
  trap cleanup EXIT
  trap 'handle_signal 129' HUP
  trap 'handle_signal 130' INT
  trap 'handle_signal 131' QUIT
  trap 'handle_signal 143' TERM

  configure_runtime
  if ! mars_bool_true "${USE_EXISTING_VLLM}"; then
    start_local_vllm
  fi
  build_workload_command "$@"
  run_workload
}

main "$@"
