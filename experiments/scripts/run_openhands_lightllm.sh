#!/usr/bin/env bash
set -euo pipefail

# Run the frozen OpenHands workload against an unmodified LightLLM server.
# LightLLM remains an external serving backend; no upstream source is vendored
# or patched and no vLLM-private telemetry is synthesized.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST_HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
source "${SCRIPT_DIR}/common_runtime.sh"
source "${MARS_DIR}/experiments/baselines/lightllm/upstream.env"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
[[ $# -eq 0 ]] || { echo "Unknown arguments: $*" >&2; exit 2; }

die() {
  echo "$*" >&2
  exit 1
}

log_kv() {
  printf '%s=%s\n' "$1" "$2"
}

print_command() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
}

# Runtime identity.
LIGHTLLM_RUNTIME="${LIGHTLLM_RUNTIME:-docker}"
LIGHTLLM_DOCKER_BIN="${LIGHTLLM_DOCKER_BIN:-docker}"
EXTERNAL_ROOT_DEFAULT="${XDG_CACHE_HOME:-${HOST_HOME}/.cache}/mars/baselines"
EXTERNAL_ROOT="${MARS_EXTERNAL_BASELINES_ROOT:-${EXTERNAL_ROOT_DEFAULT}}"
LIGHTLLM_INSTALL_ROOT="${EXTERNAL_ROOT}/lightllm/${LIGHTLLM_COMMIT}"
LIGHTLLM_SOURCE_DIR="${LIGHTLLM_SOURCE_DIR:-${LIGHTLLM_INSTALL_ROOT}/source}"
LIGHTLLM_ENV_DIR="${LIGHTLLM_ENV_DIR:-${LIGHTLLM_INSTALL_ROOT}/env}"
LIGHTLLM_PYTHON="${LIGHTLLM_PYTHON:-${LIGHTLLM_ENV_DIR}/bin/python}"

# Shared client and hardware configuration.  Defaults deliberately match the
# maintained MARS OpenHands launcher and make sampling limits explicit.
MARS_PYTHON="${MARS_PYTHON:-$(mars_default_python "${HOST_HOME}/miniconda3/envs/mars/bin/python")}"
OPENHANDS_PYTHON="${OPENHANDS_PYTHON:-$(mars_default_openhands_python "${HOST_HOME}")}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CPU_TOOL="${CPU_TOOL:-16-23}"
CPU_GPU="${CPU_GPU:-0-1}"
CPU_CORES="${CPU_CORES:-${CPU_GPU},${CPU_TOOL}}"
TP="${TP:-$(mars_infer_tp_from_devices "${CUDA_VISIBLE_DEVICES}")}"

MODEL_PATH="${MODEL_PATH:-${HOST_HOME}/models/Qwen3-Coder-30B-A3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-30B-A3B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
# Keep the official image/default behavior. Level 1 performs potentially long
# hardware-specific tuning and is an explicit experimental override.
LIGHTLLM_TRITON_AUTOTUNE_LEVEL="${LIGHTLLM_TRITON_AUTOTUNE_LEVEL:-0}"
LIGHTLLM_LOADWORKER="${LIGHTLLM_LOADWORKER:-18}"
LIGHTLLM_SHM_SIZE="${LIGHTLLM_SHM_SIZE:-16g}"
LIGHTLLM_DISABLE_LOG_STATS="${LIGHTLLM_DISABLE_LOG_STATS:-0}"

TASKS_PATH="${TASKS_PATH:-${MARS_DIR}/benchmark/hybrid_workloads/test_workload.jsonl}"
NUM_REQUESTS="${NUM_REQUESTS:-}"
MAX_REQUESTS="${MAX_REQUESTS:-${NUM_REQUESTS}}"
RPS="${RPS:-0.2}"
MAX_WORKERS="${MAX_WORKERS:-0}"
SHUFFLE="${SHUFFLE:-0}"
SEED="${SEED:-0}"
EMIT_MODE="${EMIT_MODE:-rps}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-2048}"
LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-3000}"
LLM_NUM_RETRIES="${LLM_NUM_RETRIES:-3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-40}"
GPU_TIMEOUT_S="${GPU_TIMEOUT_S:-60000}"
API_KEY="${API_KEY:-EMPTY}"

WARMUP_ENABLED="${WARMUP_ENABLED:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
WARMUP_MAX_TOKENS="${WARMUP_MAX_TOKENS:-128}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1200}"
SERVER_REQUEST_TIMEOUT_S="${SERVER_REQUEST_TIMEOUT_S:-300}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${MARS_DIR}/outputs/openhands}"
RUN_LABEL="${RUN_LABEL:-${RUN_NAME:-openhands_qwen3_lightllm}}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/lightllm_native/${RUN_LABEL}/${TIMESTAMP}}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${OUT_DIR}/workspaces}"
LIGHTLLM_LOG_PATH="${LIGHTLLM_LOG_PATH:-${OUT_DIR}/lightllm_server.log}"
LIGHTLLM_LAUNCHER_LOG_PATH="${LIGHTLLM_LAUNCHER_LOG_PATH:-${OUT_DIR}/lightllm_launcher.log}"
WARMUP_ARTIFACT="${WARMUP_ARTIFACT:-${OUT_DIR}/warmup.jsonl}"
READINESS_ARTIFACT="${READINESS_ARTIFACT:-${OUT_DIR}/readiness.jsonl}"
PROVENANCE_PATH="${PROVENANCE_PATH:-${OUT_DIR}/provenance.json}"
MODEL_HASH_MODE="${MODEL_HASH_MODE:-full}"

LIGHTLLM_HOST="${LIGHTLLM_HOST:-127.0.0.1}"
LIGHTLLM_PORT="${LIGHTLLM_PORT:-}"
LIGHTLLM_NCCL_PORT="${LIGHTLLM_NCCL_PORT:-}"
BASE_URL=""
ALLOW_CONCURRENT_RUNS="${ALLOW_CONCURRENT_RUNS:-0}"
KEEP_LIGHTLLM_SERVER="${KEEP_LIGHTLLM_SERVER:-0}"
SERVER_TERM_GRACE_S="${SERVER_TERM_GRACE_S:-20}"
SERVER_ONLY="${SERVER_ONLY:-0}"

PROBE_SCRIPT="${MARS_DIR}/experiments/baselines/lightllm/openai_compat_probe.py"
PROVENANCE_SCRIPT="${MARS_DIR}/experiments/baselines/lightllm/capture_provenance.py"
MARS_OPENHANDS_LAUNCHER="${MARS_DIR}/experiments/scripts/run_openhands_qwen3_local.sh"

normalize_dtype() {
  case "${DTYPE,,}" in
    bf16|bfloat16) echo bfloat16 ;;
    fp16|float16|half) echo float16 ;;
    fp32|float32) echo float32 ;;
    *) die "LightLLM dtype must be explicit: bf16, fp16, or fp32 (got ${DTYPE})" ;;
  esac
}

assert_gpu_contract() {
  local devices="${CUDA_VISIBLE_DEVICES// /}"
  [[ -n "${devices}" && "${devices,,}" != "all" ]] || die \
    "CUDA_VISIBLE_DEVICES must be an explicit comma-separated list for a fair run."
  IFS=',' read -r -a gpu_list <<< "${devices}"
  [[ "${#gpu_list[@]}" -eq "${TP}" ]] || die \
    "TP=${TP} but CUDA_VISIBLE_DEVICES exposes ${#gpu_list[@]} device(s): ${devices}"
  local gpu
  for gpu in "${gpu_list[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || die "Only physical GPU indices are supported: ${gpu}"
  done
}

assert_idle_gpus() {
  mars_bool_true "${ALLOW_CONCURRENT_RUNS}" && return 0
  local devices="${CUDA_VISIBLE_DEVICES// /}"
  local pids
  pids="$(mars_gpu_compute_pids "${devices}")"
  [[ -z "${pids}" ]] || die \
    "Selected GPU(s) [${devices}] are not idle (compute PID(s): ${pids}). Use free GPUs or set ALLOW_CONCURRENT_RUNS=1 explicitly."
}

verify_runtime() {
  case "${LIGHTLLM_RUNTIME}" in
    docker)
      command -v "${LIGHTLLM_DOCKER_BIN}" >/dev/null 2>&1 || die "Docker executable not found."
      "${LIGHTLLM_DOCKER_BIN}" info >/dev/null 2>&1 || die \
        "Cannot access Docker. Run setup_lightllm_baseline.sh --runtime source and set LIGHTLLM_RUNTIME=source."
      local revision
      revision="$(
        "${LIGHTLLM_DOCKER_BIN}" image inspect \
          --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' \
          "${LIGHTLLM_IMAGE}" 2>/dev/null || true
      )"
      [[ "${revision}" == "${LIGHTLLM_COMMIT}" ]] || die \
        "Pinned image missing or wrong revision. Run experiments/scripts/setup_lightllm_baseline.sh."
      ;;
    source)
      command -v realpath >/dev/null 2>&1 || die "realpath is required for source verification."
      local canonical_source canonical_env canonical_python
      canonical_source="$(realpath -m -- "${LIGHTLLM_SOURCE_DIR}")"
      canonical_env="$(realpath -m -- "${LIGHTLLM_ENV_DIR}")"
      canonical_python="$(realpath -m -- "${LIGHTLLM_PYTHON}")"
      case "${canonical_source}" in
        "${MARS_DIR}"|"${MARS_DIR}"/*) die "LightLLM source must remain outside this repository." ;;
      esac
      case "${canonical_env}" in
        "${MARS_DIR}"|"${MARS_DIR}"/*) die "LightLLM environment must remain outside this repository." ;;
      esac
      LIGHTLLM_SOURCE_DIR="${canonical_source}"
      LIGHTLLM_ENV_DIR="${canonical_env}"
      LIGHTLLM_PYTHON="${canonical_python}"
      [[ -x "${LIGHTLLM_PYTHON}" ]] || die "LightLLM Python not found: ${LIGHTLLM_PYTHON}"
      [[ -x "${LIGHTLLM_ENV_DIR}/bin/hypercorn" ]] || die \
        "LightLLM HTTP entrypoint not found: ${LIGHTLLM_ENV_DIR}/bin/hypercorn"
      [[ -d "${LIGHTLLM_SOURCE_DIR}/.git" ]] || die "LightLLM checkout not found: ${LIGHTLLM_SOURCE_DIR}"
      local commit dirty module_path
      commit="$(git -C "${LIGHTLLM_SOURCE_DIR}" rev-parse HEAD)"
      [[ "${commit}" == "${LIGHTLLM_COMMIT}" ]] || die \
        "LightLLM commit mismatch: expected ${LIGHTLLM_COMMIT}, found ${commit}"
      dirty="$(git -C "${LIGHTLLM_SOURCE_DIR}" status --porcelain --untracked-files=all)"
      [[ -z "${dirty}" ]] || die "LightLLM checkout is modified; refusing a patched baseline."
      module_path="$(
        PYTHONDONTWRITEBYTECODE=1 "${LIGHTLLM_PYTHON}" - <<'PY'
import pathlib
import lightllm
print(pathlib.Path(lightllm.__file__).resolve())
PY
      )"
      [[ "${module_path}" == "${LIGHTLLM_SOURCE_DIR}"/* ]] || die \
        "LightLLM imports from ${module_path}, expected ${LIGHTLLM_SOURCE_DIR}"
      ;;
    *) die "LIGHTLLM_RUNTIME must be docker or source (got ${LIGHTLLM_RUNTIME})" ;;
  esac
}

preflight() {
  [[ -x "${MARS_PYTHON}" ]] || die "MARS_PYTHON not found: ${MARS_PYTHON}"
  [[ -x "${OPENHANDS_PYTHON}" ]] || die "OPENHANDS_PYTHON not found: ${OPENHANDS_PYTHON}"
  [[ -d "${MODEL_PATH}" ]] || die "Model directory not found: ${MODEL_PATH}"
  [[ -f "${TASKS_PATH}" ]] || die "Tasks file not found: ${TASKS_PATH}"
  [[ -f "${PROBE_SCRIPT}" && -f "${PROVENANCE_SCRIPT}" ]] || die "LightLLM integration helpers are missing."
  [[ -f "${MARS_OPENHANDS_LAUNCHER}" ]] || die \
    "MARS OpenHands launcher not found: ${MARS_OPENHANDS_LAUNCHER}"
  [[ "${TOOL_CALL_PARSER}" == "qwen3_coder" ]] || die \
    "This Qwen3-Coder baseline requires upstream parser qwen3_coder."
  [[ "${EMIT_MODE}" == "rps" ]] || die \
    "OpenHands currently implements fixed-RPS emission only; EMIT_MODE=${EMIT_MODE} is unsupported."
  [[ "${MAX_WORKERS}" -ge 0 ]] || die "MAX_WORKERS must be >= 0"
  [[ "${WARMUP_REQUESTS}" -ge 0 ]] || die "WARMUP_REQUESTS must be >= 0"
  assert_gpu_contract
  DTYPE="$(normalize_dtype)"
  verify_runtime
  assert_idle_gpus
}

assert_fresh_output_dir() {
  local existing=""
  if [[ -e "${OUT_DIR}" && ! -d "${OUT_DIR}" ]]; then
    die "OUT_DIR exists and is not a directory: ${OUT_DIR}"
  fi
  if [[ -d "${OUT_DIR}" ]]; then
    existing="$(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)"
  fi
  if [[ -n "${existing}" ]]; then
    die "Refusing to reuse non-empty OUT_DIR: ${OUT_DIR}"
  fi
}

build_server_command() {
  [[ -n "${LIGHTLLM_PORT}" ]] || LIGHTLLM_PORT="$(mars_pick_free_port "${MARS_PYTHON}")"
  [[ -n "${LIGHTLLM_NCCL_PORT}" ]] || LIGHTLLM_NCCL_PORT="$(mars_pick_free_port "${MARS_PYTHON}")"
  [[ "${LIGHTLLM_NCCL_PORT}" != "${LIGHTLLM_PORT}" ]] || die "HTTP and NCCL ports must differ."
  mars_assert_local_port_available "${MARS_PYTHON}" "${LIGHTLLM_HOST}" "${LIGHTLLM_PORT}" "LightLLM HTTP"
  mars_assert_local_port_available "${MARS_PYTHON}" "${LIGHTLLM_HOST}" "${LIGHTLLM_NCCL_PORT}" "LightLLM NCCL"
  BASE_URL="http://${LIGHTLLM_HOST}:${LIGHTLLM_PORT}/v1"

  LIGHTLLM_API_ARGS=(
    --host "${LIGHTLLM_HOST}"
    --port "${LIGHTLLM_PORT}"
    --model_dir "${MODEL_PATH}"
    --model_name "${MODEL_NAME}"
    --tp "${TP}"
    --data_type "${DTYPE}"
    --mem_fraction "${GPU_MEMORY_UTILIZATION}"
    --max_req_total_len "${MAX_MODEL_LEN}"
    --nccl_port "${LIGHTLLM_NCCL_PORT}"
    --tool_call_parser "${TOOL_CALL_PARSER}"
  )
  mars_bool_true "${TRUST_REMOTE_CODE}" && LIGHTLLM_API_ARGS+=(--trust_remote_code)
  mars_bool_true "${LIGHTLLM_DISABLE_LOG_STATS}" && LIGHTLLM_API_ARGS+=(--disable_log_stats)

  if [[ "${LIGHTLLM_RUNTIME}" == "docker" ]]; then
    CONTAINER_NAME="mars-lightllm-${UID}-$$"
    local container_model_dir="/models/model"
    local container_api_args=("${LIGHTLLM_API_ARGS[@]}")
    local i
    for ((i = 0; i < ${#container_api_args[@]}; i++)); do
      if [[ "${container_api_args[$i]}" == "${MODEL_PATH}" ]]; then
        container_api_args[$i]="${container_model_dir}"
      fi
    done
    SERVER_COMMAND=(
      "${LIGHTLLM_DOCKER_BIN}" run --detach --rm
      --name "${CONTAINER_NAME}"
      --platform "${LIGHTLLM_PLATFORM}"
      --network host
      --shm-size "${LIGHTLLM_SHM_SIZE}"
      # Docker parses --gpus as CSV. Embedded quotes keep a multi-GPU device
      # list in one value (the form documented by NVIDIA/Docker).
      --gpus "\"device=${CUDA_VISIBLE_DEVICES// /}\""
      --cpuset-cpus "${CPU_CORES}"
      --env "LOADWORKER=${LIGHTLLM_LOADWORKER}"
      --env "LIGHTLLM_TRITON_AUTOTUNE_LEVEL=${LIGHTLLM_TRITON_AUTOTUNE_LEVEL}"
      --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1
      --env TOKENIZERS_PARALLELISM=false
      --volume "${MODEL_PATH}:${container_model_dir}:ro"
      "${LIGHTLLM_IMAGE}"
      python -m lightllm.server.api_server
      "${container_api_args[@]}"
    )
  else
    SERVER_COMMAND=(
      env
      "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
      "PATH=${LIGHTLLM_ENV_DIR}/bin:${PATH}"
      "LOADWORKER=${LIGHTLLM_LOADWORKER}"
      "LIGHTLLM_TRITON_AUTOTUNE_LEVEL=${LIGHTLLM_TRITON_AUTOTUNE_LEVEL}"
      HF_HUB_OFFLINE=1
      TRANSFORMERS_OFFLINE=1
      TOKENIZERS_PARALLELISM=false
    )
    if command -v taskset >/dev/null 2>&1 && [[ -n "${CPU_CORES}" ]]; then
      SERVER_COMMAND+=(taskset -c "${CPU_CORES}")
    fi
    SERVER_COMMAND+=(
      "${LIGHTLLM_PYTHON}" -m lightllm.server.api_server
      "${LIGHTLLM_API_ARGS[@]}"
    )
  fi
}

CONTAINER_NAME=""
CONTAINER_ID=""
SERVER_PID=""
LOG_PID=""
READY_PID=""
CLEANUP_DONE=0

server_alive() {
  if [[ "${LIGHTLLM_RUNTIME}" == "docker" ]]; then
    [[ -n "${CONTAINER_NAME}" ]] || return 1
    [[ "$("${LIGHTLLM_DOCKER_BIN}" inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" == "true" ]]
  else
    [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1
  fi
}

cleanup() {
  [[ "${CLEANUP_DONE}" == "1" ]] && return 0
  CLEANUP_DONE=1
  [[ -z "${READY_PID}" ]] || kill "${READY_PID}" >/dev/null 2>&1 || true
  if ! mars_bool_true "${KEEP_LIGHTLLM_SERVER}"; then
    if [[ "${LIGHTLLM_RUNTIME}" == "docker" && -n "${CONTAINER_NAME}" ]]; then
      "${LIGHTLLM_DOCKER_BIN}" stop --time "${SERVER_TERM_GRACE_S}" "${CONTAINER_NAME}" >/dev/null 2>&1 || true
      "${LIGHTLLM_DOCKER_BIN}" rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    elif [[ -n "${SERVER_PID}" ]]; then
      kill -- "-${SERVER_PID}" >/dev/null 2>&1 || kill "${SERVER_PID}" >/dev/null 2>&1 || true
      local deadline=$((SECONDS + SERVER_TERM_GRACE_S))
      while kill -0 "${SERVER_PID}" >/dev/null 2>&1 && ((SECONDS < deadline)); do
        sleep 0.2
      done
      kill -KILL -- "-${SERVER_PID}" >/dev/null 2>&1 || kill -KILL "${SERVER_PID}" >/dev/null 2>&1 || true
      wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
  fi
  [[ -z "${LOG_PID}" ]] || kill "${LOG_PID}" >/dev/null 2>&1 || true
  [[ -z "${LOG_PID}" ]] || wait "${LOG_PID}" >/dev/null 2>&1 || true
}

handle_signal() {
  local code="$1"
  trap - EXIT
  cleanup
  exit "${code}"
}

start_server() {
  : > "${LIGHTLLM_LOG_PATH}"
  echo "== Starting pinned upstream LightLLM =="
  log_kv LIGHTLLM_RUNTIME "${LIGHTLLM_RUNTIME}"
  log_kv LIGHTLLM_COMMIT "${LIGHTLLM_COMMIT}"
  log_kv LIGHTLLM_IMAGE "${LIGHTLLM_IMAGE}"
  log_kv CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"
  log_kv TP "${TP}"
  log_kv DTYPE "${DTYPE}"
  log_kv GPU_MEMORY_UTILIZATION "${GPU_MEMORY_UTILIZATION}"
  log_kv MAX_MODEL_LEN "${MAX_MODEL_LEN}"
  log_kv BASE_URL "${BASE_URL}"
  log_kv LIGHTLLM_LOG_PATH "${LIGHTLLM_LOG_PATH}"
  print_command "${SERVER_COMMAND[@]}"

  if [[ "${LIGHTLLM_RUNTIME}" == "docker" ]]; then
    CONTAINER_ID="$("${SERVER_COMMAND[@]}")"
    "${LIGHTLLM_DOCKER_BIN}" logs --follow "${CONTAINER_NAME}" > "${LIGHTLLM_LOG_PATH}" 2>&1 &
    LOG_PID=$!
  else
    if command -v setsid >/dev/null 2>&1; then
      setsid "${SERVER_COMMAND[@]}" > "${LIGHTLLM_LOG_PATH}" 2>&1 &
    else
      "${SERVER_COMMAND[@]}" > "${LIGHTLLM_LOG_PATH}" 2>&1 &
    fi
    SERVER_PID=$!
  fi
}

wait_until_ready() {
  echo "== Waiting for /v1/models =="
  "${MARS_PYTHON}" "${PROBE_SCRIPT}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_NAME}" \
    --api-key "${API_KEY}" \
    --wait-timeout-s "${SERVER_READY_TIMEOUT_S}" \
    --request-timeout-s "${SERVER_REQUEST_TIMEOUT_S}" \
    --requests 0 \
    --max-tokens "${WARMUP_MAX_TOKENS}" \
    --output "${READINESS_ARTIFACT}" &
  READY_PID=$!
  while kill -0 "${READY_PID}" >/dev/null 2>&1; do
    if ! server_alive; then
      kill "${READY_PID}" >/dev/null 2>&1 || true
      wait "${READY_PID}" >/dev/null 2>&1 || true
      READY_PID=""
      echo "LightLLM exited before readiness. Last log lines:" >&2
      tail -80 "${LIGHTLLM_LOG_PATH}" >&2 || true
      return 1
    fi
    sleep 1
  done
  set +e
  wait "${READY_PID}"
  local rc=$?
  set -e
  READY_PID=""
  if [[ "${rc}" -ne 0 ]]; then
    echo "LightLLM readiness failed. Last log lines:" >&2
    tail -80 "${LIGHTLLM_LOG_PATH}" >&2 || true
    return "${rc}"
  fi
  echo "== LightLLM ready =="
}

write_provenance() {
  local source_args=()
  local image_args=()
  local python_args=()
  if [[ "${LIGHTLLM_RUNTIME}" == "source" ]]; then
    source_args=(--source-dir "${LIGHTLLM_SOURCE_DIR}")
    python_args=(--python "${LIGHTLLM_PYTHON}" --include-packages true)
  else
    image_args=(--image "${LIGHTLLM_IMAGE}" --image-digest "${LIGHTLLM_IMAGE#*@}")
  fi
  "${MARS_PYTHON}" "${PROVENANCE_SCRIPT}" \
    --output "${PROVENANCE_PATH}" \
    --backend-name lightllm_native \
    --runtime "${LIGHTLLM_RUNTIME}" \
    --repo-url "${LIGHTLLM_REPO_URL}" \
    --ref "${LIGHTLLM_REF}" \
    --expected-commit "${LIGHTLLM_COMMIT}" \
    "${source_args[@]}" \
    "${image_args[@]}" \
    "${python_args[@]}" \
    --orchestrator-python "${MARS_PYTHON}" \
    --openhands-python "${OPENHANDS_PYTHON}" \
    --client-source "common_runtime.sh=${SCRIPT_DIR}/common_runtime.sh" \
    --client-source "run_openhands_qwen3_local.sh=${MARS_OPENHANDS_LAUNCHER}" \
    --client-source "run_task_workload.py=${MARS_DIR}/experiments/openhands/run_task_workload.py" \
    --client-source "command_guard.py=${MARS_DIR}/experiments/openhands/command_guard.py" \
    --client-source "common.py=${MARS_DIR}/experiments/openhands/common.py" \
    --client-source "llm_trace_bridge.py=${MARS_DIR}/experiments/openhands/llm_trace_bridge.py" \
    --client-source "openhands_runner.py=${MARS_DIR}/experiments/openhands/openhands_runner.py" \
    --client-source "request_bootstrap.py=${MARS_DIR}/experiments/openhands/request_bootstrap.py" \
    --client-source "request_worker.py=${MARS_DIR}/experiments/openhands/request_worker.py" \
    --client-source "sandbox_shell.sh=${MARS_DIR}/experiments/openhands/sandbox_shell.sh" \
    --client-source "tool_trace_bridge.py=${MARS_DIR}/experiments/openhands/tool_trace_bridge.py" \
    --client-source "trace_writer.py=${MARS_DIR}/experiments/openhands/trace_writer.py" \
    --client-source "openai_compat_probe.py=${PROBE_SCRIPT}" \
    --host "${LIGHTLLM_HOST}" \
    --port "${LIGHTLLM_PORT}" \
    --nccl-port "${LIGHTLLM_NCCL_PORT}" \
    --server-log "${LIGHTLLM_LOG_PATH}" \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --model-hash-mode "${MODEL_HASH_MODE}" \
    --tasks-path "${TASKS_PATH}" \
    --rps "${RPS}" \
    --max-requests "${MAX_REQUESTS}" \
    --shuffle "${SHUFFLE}" \
    --seed "${SEED}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
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
    --warmup-enabled "${WARMUP_ENABLED}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --warmup-max-tokens "${WARMUP_MAX_TOKENS}" \
    --warmup-artifact "${WARMUP_ARTIFACT}" \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --auto-tool-choice-enabled true \
    --kv-events false \
    --triton-autotune-level "${LIGHTLLM_TRITON_AUTOTUNE_LEVEL}" \
    --load-worker "${LIGHTLLM_LOADWORKER}" \
    --upstream-patched false \
    -- "${SERVER_COMMAND[@]}"
}

run_warmup() {
  if ! mars_bool_true "${WARMUP_ENABLED}"; then
    : > "${WARMUP_ARTIFACT}"
    echo "Warmup disabled by WARMUP_ENABLED=${WARMUP_ENABLED}."
    return 0
  fi
  echo "== Running common streaming/tool-call warmup =="
  "${MARS_PYTHON}" "${PROBE_SCRIPT}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_NAME}" \
    --api-key "${API_KEY}" \
    --wait-timeout-s 30 \
    --request-timeout-s "${SERVER_REQUEST_TIMEOUT_S}" \
    --requests "${WARMUP_REQUESTS}" \
    --max-tokens "${WARMUP_MAX_TOKENS}" \
    --output "${WARMUP_ARTIFACT}"
}

run_workload() {
  echo "== Running the shared OpenHands workload =="
  env \
    USE_EXISTING_VLLM=1 \
    BASE_URL="${BASE_URL}" \
    KV_EVENTS=0 \
    SCHEDULING_POLICY=lightllm_native \
    MARS_PYTHON="${MARS_PYTHON}" \
    OPENHANDS_PYTHON="${OPENHANDS_PYTHON}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    CPU_TOOL="${CPU_TOOL}" \
    CPU_GPU="${CPU_GPU}" \
    CPU_CORES="${CPU_CORES}" \
    TP="${TP}" \
    MODEL_PATH="${MODEL_PATH}" \
    MODEL_NAME="${MODEL_NAME}" \
    DTYPE="${DTYPE}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    TASKS_PATH="${TASKS_PATH}" \
    MAX_REQUESTS="${MAX_REQUESTS}" \
    RPS="${RPS}" \
    MAX_WORKERS="${MAX_WORKERS}" \
    SHUFFLE="${SHUFFLE}" \
    SEED="${SEED}" \
    EMIT_MODE="${EMIT_MODE}" \
    TEMPERATURE="${TEMPERATURE}" \
    TOP_P="${TOP_P}" \
    MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS}" \
    LLM_TIMEOUT_S="${LLM_TIMEOUT_S}" \
    LLM_NUM_RETRIES="${LLM_NUM_RETRIES}" \
    MAX_ITERATIONS="${MAX_ITERATIONS}" \
    GPU_TIMEOUT_S="${GPU_TIMEOUT_S}" \
    API_KEY="${API_KEY}" \
    OUT_DIR="${OUT_DIR}" \
    WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
    VLLM_LOG_PATH="${LIGHTLLM_LOG_PATH}" \
    OPENAI_COMPAT_WARMUP=0 \
    bash "${MARS_OPENHANDS_LAUNCHER}"
}

dry_run() {
  build_server_command
  echo "# LightLLM server (upstream commit ${LIGHTLLM_COMMIT})"
  print_command "${SERVER_COMMAND[@]}"
  echo "# Shared workload delegation"
  print_command env USE_EXISTING_VLLM=1 BASE_URL="${BASE_URL}" KV_EVENTS=0 \
    SCHEDULING_POLICY=lightllm_native OUT_DIR="${OUT_DIR}" bash "${MARS_OPENHANDS_LAUNCHER}"
}

main() {
  assert_gpu_contract
  DTYPE="$(normalize_dtype)"
  if [[ "${DRY_RUN}" == "1" ]]; then
    dry_run
    return 0
  fi

  preflight
  assert_fresh_output_dir
  mkdir -p "${OUT_DIR}" "${WORKSPACE_ROOT}"
  : > "${LIGHTLLM_LAUNCHER_LOG_PATH}"
  exec > >(tee -a "${LIGHTLLM_LAUNCHER_LOG_PATH}") 2>&1
  trap cleanup EXIT
  trap 'handle_signal 129' HUP
  trap 'handle_signal 130' INT
  trap 'handle_signal 131' QUIT
  trap 'handle_signal 143' TERM

  build_server_command
  start_server
  wait_until_ready
  write_provenance
  run_warmup

  if mars_bool_true "${SERVER_ONLY}"; then
    echo "SERVER_ONLY=1; LightLLM is ready at ${BASE_URL}. Waiting for termination."
    if [[ "${LIGHTLLM_RUNTIME}" == "docker" ]]; then
      local container_status
      container_status="$("${LIGHTLLM_DOCKER_BIN}" wait "${CONTAINER_NAME}")"
      [[ "${container_status}" =~ ^[0-9]+$ ]] || die \
        "Docker returned an invalid LightLLM container status: ${container_status}"
      if [[ "${container_status}" -ne 0 ]]; then
        echo "LightLLM container exited with status ${container_status}. Last log lines:" >&2
        tail -80 "${LIGHTLLM_LOG_PATH}" >&2 || true
        return "${container_status}"
      fi
    else
      wait "${SERVER_PID}"
    fi
    return 0
  fi

  run_workload
  echo "LightLLM baseline finished successfully."
  echo "Artifacts: ${OUT_DIR}"
}

main
