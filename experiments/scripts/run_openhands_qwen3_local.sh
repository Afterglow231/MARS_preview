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
OPENHANDS_PYTHON="${OPENHANDS_PYTHON:-$(mars_default_openhands_python "${HOST_HOME}")}"
OPENHANDS_RUNNER="${MARS_DIR}/experiments/openhands/run_task_workload.py"

# Resources.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

# CPU_GPU and CPU_TOOL should not overlap with each other
CPU_TOOL="${CPU_TOOL:-16-23}"
CPU_GPU="${CPU_GPU:-0-1}"
CPU_CORES="${CPU_CORES:-${CPU_GPU},${CPU_TOOL}}"
TP="${TP:-}"

# Model and vLLM.
MODEL_PATH="${MODEL_PATH:-${HOST_HOME}/models/Qwen3-Coder-30B-A3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-30B-A3B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
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
TOP_P="${TOP_P:-1.0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-2048}"
LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-3000}"
LLM_NUM_RETRIES="${LLM_NUM_RETRIES:-3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-40}"
TERMINAL_NO_CHANGE_TIMEOUT_S="${TERMINAL_NO_CHANGE_TIMEOUT_S:-30}"
TERMINAL_TYPE="${TERMINAL_TYPE:-}"
PREFILL_TOKENIZER="${PREFILL_TOKENIZER:-${MODEL_PATH}}"
API_KEY="${API_KEY:-EMPTY}"
MARS_REQUIRE_BWRAP="${MARS_REQUIRE_BWRAP:-1}"
MARS_ALLOW_HOST_MODELS_READ="${MARS_ALLOW_HOST_MODELS_READ:-0}"

# Optional common protocol warmup.  The LightLLM launcher runs the same probe
# before delegating here and sets this to 0 to avoid a duplicate warmup.
OPENAI_COMPAT_WARMUP="${OPENAI_COMPAT_WARMUP:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
WARMUP_MAX_TOKENS="${WARMUP_MAX_TOKENS:-128}"
OPENAI_COMPAT_PROBE_SCRIPT="${MARS_DIR}/experiments/baselines/lightllm/openai_compat_probe.py"
PROVENANCE_SCRIPT="${MARS_DIR}/experiments/baselines/lightllm/capture_provenance.py"

# Output.
OUTPUT_ROOT="${OUTPUT_ROOT:-${MARS_DIR}/outputs/openhands}"
RUN_LABEL="${RUN_LABEL:-${RUN_NAME:-openhands_qwen3_317}}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/${SCHEDULING_POLICY}/${RUN_LABEL}/${TIMESTAMP}}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${OUT_DIR}/workspaces}"
LAUNCHER_LOG_PATH="${LAUNCHER_LOG_PATH:-${OUT_DIR}/launcher.log}"
WORKLOAD_LOG_PATH="${WORKLOAD_LOG_PATH:-${OUT_DIR}/openhands_workload.log}"
VLLM_LOG_PATH="${VLLM_LOG_PATH:-${OUT_DIR}/vllm_server.log}"
WARMUP_ARTIFACT="${WARMUP_ARTIFACT:-${OUT_DIR}/warmup.jsonl}"
PROVENANCE_PATH="${PROVENANCE_PATH:-${OUT_DIR}/provenance.json}"
MODEL_HASH_MODE="${MODEL_HASH_MODE:-full}"

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
  mars_gpu_compute_pids "${gpu_spec}"
}

preflight() {
  [[ -x "${MARS_PYTHON}" ]] || die "MARS_PYTHON not found or not executable: ${MARS_PYTHON}"
  [[ -x "${OPENHANDS_PYTHON}" ]] || die "OPENHANDS_PYTHON not found or not executable: ${OPENHANDS_PYTHON}"
  if ! "${OPENHANDS_PYTHON}" - <<'PY' >/dev/null 2>&1
import openhands.sdk  # noqa: F401
import pydantic  # noqa: F401
PY
  then
    die "OPENHANDS_PYTHON cannot import openhands.sdk and pydantic: ${OPENHANDS_PYTHON}"
  fi
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

assert_fresh_workload_output() {
  local artifact existing=""
  for artifact in run_config.json events.jsonl traces.jsonl summary.json; do
    [[ ! -e "${OUT_DIR}/${artifact}" ]] || die \
      "Refusing to overwrite existing workload artifact: ${OUT_DIR}/${artifact}"
  done

  if [[ -d "${OUT_DIR}" ]]; then
    existing="$(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)"
  fi
  if ! mars_bool_true "${USE_EXISTING_VLLM}" && [[ -n "${existing}" ]]; then
    die "Refusing to reuse non-empty OUT_DIR: ${OUT_DIR}"
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
SERVER_LAUNCH_COMMAND=()
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
    if [[ -n "${pids}" ]]; then
      die "Selected GPU(s) [${GPU_GUARD_DEVICES}] are not idle (compute PID(s): ${pids}). Use free GPUs or set ALLOW_CONCURRENT_RUNS=1 explicitly."
    fi
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

  if mars_bool_true "${VLLM_NCU_PROFILE:-0}"; then
    local ncu_exe="${NCU_BIN:-/opt/nvidia/nsight-compute/2025.4.1/ncu}"
    if [[ ! -x "${ncu_exe}" ]]; then
      if command -v "${ncu_exe}" >/dev/null 2>&1; then
        ncu_exe="$(command -v "${ncu_exe}")"
      else
        die "VLLM_NCU_PROFILE=1 but NCU_BIN=${ncu_exe} is not executable/on PATH"
      fi
    fi
    [[ -n "${VLLM_NCU_OUTPUT_PREFIX:-}" ]] || die "VLLM_NCU_PROFILE=1 requires VLLM_NCU_OUTPUT_PREFIX"

    echo "== Profiling local vLLM with NCU =="
    log_kv VLLM_NCU_OUTPUT_PREFIX "${VLLM_NCU_OUTPUT_PREFIX}"
    log_kv NCU_FORWARD_MODE "${NCU_FORWARD_MODE:-pd_mixed}"
    log_kv NCU_NVTX_INCLUDE "${NCU_NVTX_INCLUDE:-regex:vllm_forward_step.*mode=${NCU_FORWARD_MODE:-pd_mixed}.*/}"
    log_kv NCU_LAUNCH_SKIP "${NCU_LAUNCH_SKIP:-0}"
    log_kv NCU_LAUNCH_COUNT "${NCU_LAUNCH_COUNT:-20}"

    local ncu_cmd=(
      "${ncu_exe}"
      --force-overwrite
      --target-processes all
      --kernel-name-base demangled
      --replay-mode "${NCU_REPLAY_MODE:-kernel}"
      --null-stdin
      --nvtx
      --nvtx-include "${NCU_NVTX_INCLUDE:-regex:vllm_forward_step.*mode=${NCU_FORWARD_MODE:-pd_mixed}.*/}"
      --launch-skip "${NCU_LAUNCH_SKIP:-0}"
      --launch-count "${NCU_LAUNCH_COUNT:-20}"
      -o "${VLLM_NCU_OUTPUT_PREFIX}"
    )
    if mars_bool_true "${VLLM_NCU_KILL_AFTER_PROFILE:-0}"; then
      ncu_cmd+=(--kill 1)
    fi
    if [[ -n "${NCU_KERNEL_NAME:-}" ]]; then
      ncu_cmd+=(--kernel-name "${NCU_KERNEL_NAME}")
    fi

    local section
    local ncu_sections
    IFS=',' read -r -a ncu_sections <<< "${NCU_SECTIONS:-SpeedOfLight,SpeedOfLight_RooflineChart,SpeedOfLight_HierarchicalTensorRooflineChart,ComputeWorkloadAnalysis,MemoryWorkloadAnalysis,MemoryWorkloadAnalysis_Chart,MemoryWorkloadAnalysis_Tables,Occupancy,SchedulerStats,WarpStateStats,PmSampling_WarpStates,LaunchStats}"
    for section in "${ncu_sections[@]}"; do
      section="$(echo "${section}" | xargs)"
      [[ -z "${section}" ]] && continue
      ncu_cmd+=(--section "${section}")
    done
    if [[ -n "${NCU_EXTRA_ARGS:-}" ]]; then
      local ncu_extra_args
      read -r -a ncu_extra_args <<< "${NCU_EXTRA_ARGS}"
      ncu_cmd+=("${ncu_extra_args[@]}")
    fi

    launch_cmd=("${ncu_cmd[@]}" -- "${launch_cmd[@]}")
  fi

  if command -v setsid >/dev/null 2>&1; then
    setsid "${launch_cmd[@]}" >"${VLLM_LOG_PATH}" 2>&1 &
    SERVER_IS_PROCESS_GROUP=1
  else
    "${launch_cmd[@]}" >"${VLLM_LOG_PATH}" 2>&1 &
  fi
  SERVER_LAUNCH_COMMAND=("${launch_cmd[@]}")
  SERVER_PID=$!

  mars_wait_for_vllm "${MARS_PYTHON}" "${BASE_URL}" 300 "${SERVER_PID}" "${VLLM_LOG_PATH}" "${REQUIRE_KV_STATS_READY}"
  echo "== vLLM ready =="
  log_kv BASE_URL "${BASE_URL}"
}

write_local_vllm_provenance() {
  mars_bool_true "${USE_EXISTING_VLLM}" && return 0
  [[ -f "${PROVENANCE_SCRIPT}" ]] || die "Provenance helper not found: ${PROVENANCE_SCRIPT}"

  local repo_url repo_commit normalized_top_p
  repo_url="$(git -C "${MARS_DIR}" remote get-url origin 2>/dev/null || echo local)"
  repo_commit="$(git -C "${MARS_DIR}" rev-parse HEAD)"
  normalized_top_p="${TOP_P:-1.0}"
  "${MARS_PYTHON}" "${PROVENANCE_SCRIPT}" \
    --output "${PROVENANCE_PATH}" \
    --backend-name mars-vllm \
    --runtime source \
    --repo-url "${repo_url}" \
    --ref "$(git -C "${MARS_DIR}" branch --show-current 2>/dev/null || true)" \
    --expected-commit "${repo_commit}" \
    --source-dir "${MARS_DIR}" \
    --python "${MARS_PYTHON}" \
    --include-packages false \
    --orchestrator-python "${MARS_PYTHON}" \
    --openhands-python "${OPENHANDS_PYTHON}" \
    --client-source "common_runtime.sh=${SCRIPT_DIR}/common_runtime.sh" \
    --client-source "run_openhands_qwen3_local.sh=${BASH_SOURCE[0]}" \
    --client-source "run_task_workload.py=${OPENHANDS_RUNNER}" \
    --client-source "command_guard.py=${MARS_DIR}/experiments/openhands/command_guard.py" \
    --client-source "common.py=${MARS_DIR}/experiments/openhands/common.py" \
    --client-source "llm_trace_bridge.py=${MARS_DIR}/experiments/openhands/llm_trace_bridge.py" \
    --client-source "openhands_runner.py=${MARS_DIR}/experiments/openhands/openhands_runner.py" \
    --client-source "request_bootstrap.py=${MARS_DIR}/experiments/openhands/request_bootstrap.py" \
    --client-source "request_worker.py=${MARS_DIR}/experiments/openhands/request_worker.py" \
    --client-source "sandbox_shell.sh=${MARS_DIR}/experiments/openhands/sandbox_shell.sh" \
    --client-source "tool_trace_bridge.py=${MARS_DIR}/experiments/openhands/tool_trace_bridge.py" \
    --client-source "trace_writer.py=${MARS_DIR}/experiments/openhands/trace_writer.py" \
    --client-source "openai_compat_probe.py=${OPENAI_COMPAT_PROBE_SCRIPT}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --nccl-port "${VLLM_DP_MASTER_PORT:-0}" \
    --server-log "${VLLM_LOG_PATH}" \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --model-hash-mode "${MODEL_HASH_MODE}" \
    --tasks-path "${TASKS_PATH}" \
    --rps "${RPS}" \
    --max-requests "${MAX_REQUESTS}" \
    --shuffle "${SHUFFLE}" \
    --seed "${SEED}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${normalized_top_p}" \
    --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
    --llm-timeout-s "${LLM_TIMEOUT_S}" \
    --llm-num-retries "${LLM_NUM_RETRIES}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --emit-mode "${EMIT_MODE}" \
    --max-workers "${MAX_WORKERS}" \
    --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
    --tp "${TP}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --cpu-gpu "${CPU_GPU}" \
    --cpu-tool "${CPU_TOOL}" \
    --cpu-cores "${CPU_CORES}" \
    --warmup-enabled "${OPENAI_COMPAT_WARMUP}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --warmup-max-tokens "${WARMUP_MAX_TOKENS}" \
    --warmup-artifact "${WARMUP_ARTIFACT}" \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --auto-tool-choice-enabled "${ENABLE_AUTO_TOOL_CHOICE}" \
    --kv-events "${KV_EVENTS}" \
    --triton-autotune-level -1 \
    --load-worker -1 \
    --upstream-patched true \
    -- "${SERVER_LAUNCH_COMMAND[@]}"
}

run_openai_compat_warmup() {
  if ! mars_bool_true "${OPENAI_COMPAT_WARMUP}"; then
    return 0
  fi
  [[ -f "${OPENAI_COMPAT_PROBE_SCRIPT}" ]] || die \
    "OpenAI compatibility probe not found: ${OPENAI_COMPAT_PROBE_SCRIPT}"
  echo "== Running common streaming/tool-call warmup =="
  "${MARS_PYTHON}" "${OPENAI_COMPAT_PROBE_SCRIPT}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_NAME}" \
    --api-key "${API_KEY}" \
    --wait-timeout-s 30 \
    --request-timeout-s "${LLM_TIMEOUT_S}" \
    --requests "${WARMUP_REQUESTS}" \
    --max-tokens "${WARMUP_MAX_TOKENS}" \
    --output "${WARMUP_ARTIFACT}"
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
    MARS_HOST_HOME="${HOST_HOME}"
    MARS_HOST_CONDA_ROOT="${HOST_HOME}/miniconda3"
    MARS_HOST_VENVS_ROOT="${HOST_HOME}/.venvs"
    MARS_HOST_MODELS_ROOT="${HOST_HOME}/models"
    MARS_REQUIRE_BWRAP="${MARS_REQUIRE_BWRAP}"
    MARS_ALLOW_HOST_MODELS_READ="${MARS_ALLOW_HOST_MODELS_READ}"
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
  log_kv WORKSPACE_ROOT "${WORKSPACE_ROOT}"
  log_kv MARS_REQUIRE_BWRAP "${MARS_REQUIRE_BWRAP}"
  log_kv MARS_ALLOW_HOST_MODELS_READ "${MARS_ALLOW_HOST_MODELS_READ}"
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
  assert_fresh_workload_output
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
  write_local_vllm_provenance
  run_openai_compat_warmup
  build_workload_command "$@"
  run_workload
}

main "$@"
