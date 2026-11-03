#!/usr/bin/env bash
set -euo pipefail

# Start one local MARS/vLLM server and run OpenCode and/or Codex against the
# same task stream. Set USE_EXISTING_VLLM=1 and BASE_URL=... to reuse a server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_MARS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MARS_DIR="${MARS_DIR_OVERRIDE:-${LAUNCHER_MARS_DIR}}"
MARS_DIR="$(cd "${MARS_DIR}" && pwd)"
HOST_ROOT="${HOST_ROOT:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
source "${SCRIPT_DIR}/common_runtime.sh"

MARS_PYTHON="${MARS_PYTHON:-$(mars_default_python "${HOST_ROOT}/miniconda3/envs/mars/bin/python")}"
MARS_BWRAP_BIN="${MARS_BWRAP_BIN:-$(command -v bwrap || true)}"
MARS_BWRAP_BIN="${MARS_BWRAP_BIN:-/usr/bin/bwrap}"
export MARS_BWRAP_BIN
AGENT_CLI_ADAPTER_DIR="${AGENT_CLI_ADAPTER_DIR:-${LAUNCHER_MARS_DIR}/experiments/agent_cli}"
source "${AGENT_CLI_ADAPTER_DIR}/upstream.env"
RUNNER="${AGENT_CLI_ADAPTER_DIR}/run_workload.py"
SETUP_SCRIPT="${SCRIPT_DIR}/setup_agent_cli.sh"
MARS_AGENT_BASELINES_ROOT="${MARS_AGENT_BASELINES_ROOT:-${HOST_ROOT}/external/mars_agent_baselines}"
export MARS_AGENT_BASELINES_ROOT
if [[ "${MARS_BWRAP_BIN}" == "/usr/bin/bwrap" \
      && ! -x "${MARS_BWRAP_BIN}" \
      && -x "${MARS_AGENT_BASELINES_ROOT}/toolchain/bubblewrap/usr/bin/bwrap" ]]; then
  MARS_BWRAP_BIN="${MARS_AGENT_BASELINES_ROOT}/toolchain/bubblewrap/usr/bin/bwrap"
  export MARS_BWRAP_BIN
fi
MARS_NODE_BIN="${MARS_NODE_BIN:-$(command -v node || true)}"
MARS_NODE_BIN="${MARS_NODE_BIN:-${MARS_AGENT_BASELINES_ROOT}/toolchain/node-v${NODE_VERSION}-linux-x64/bin/node}"
export MARS_NODE_BIN
if [[ -x "${MARS_NODE_BIN}" ]]; then
  PATH="$(dirname "${MARS_NODE_BIN}"):${PATH}"
  export PATH
fi
# Reuse the maintained OpenHands client harness (including task-input
# localization) while MARS_DIR_OVERRIDE selects the server/scheduler checkout.
MARS_OPENHANDS_DIR="${MARS_OPENHANDS_DIR:-${LAUNCHER_MARS_DIR}/experiments/openhands}"
export MARS_OPENHANDS_DIR

# Agents and workload.
AGENTS="${AGENTS:-opencode,codex}"
TASKS_PATH="${TASKS_PATH:-${MARS_DIR}/benchmark/hybrid_workloads/test_workload.jsonl}"
MAX_REQUESTS="${MAX_REQUESTS:-}"
RPS="${RPS:-0.2}"
MAX_WORKERS="${MAX_WORKERS:-0}"
SHUFFLE="${SHUFFLE:-0}"
SEED="${SEED:-0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
# Match the maintained OpenHands runner's two timeout layers. LLM_TIMEOUT_S is
# the per-LLM HTTP timeout; GPU_TIMEOUT_S is the hard wall-clock timeout for a
# complete agent request. REQUEST_TIMEOUT_S remains a compatibility alias for
# older ad-hoc invocations, but never controls the LLM timeout.
LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-3000}"
LLM_NUM_RETRIES="${LLM_NUM_RETRIES:-3}"
GPU_TIMEOUT_S="${GPU_TIMEOUT_S:-${REQUEST_TIMEOUT_S:-60000}}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-2048}"
API_KEY="${API_KEY:-EMPTY}"
OPENAI_COMPAT_WARMUP="${OPENAI_COMPAT_WARMUP:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
WARMUP_MAX_TOKENS="${WARMUP_MAX_TOKENS:-128}"
OPENAI_COMPAT_PROBE_SCRIPT="${OPENAI_COMPAT_PROBE_SCRIPT:-${LAUNCHER_MARS_DIR}/experiments/baselines/lightllm/openai_compat_probe.py}"
RESUME="${RESUME:-0}"
RERUN_FAILURES="${RERUN_FAILURES:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_UNSANDBOXED_AGENT="${ALLOW_UNSANDBOXED_AGENT:-${ALLOW_UNSANDBOXED_OPENCODE:-0}}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
CODEX_ENABLE_SUBAGENTS="${CODEX_ENABLE_SUBAGENTS:-0}"

# Model and server.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
TP="${TP:-}"
MODEL_PATH="${MODEL_PATH:-${HOST_ROOT}/models/Qwen3-Coder-30B-A3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-30B-A3B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
# Available server policies: fcfs/autellix/infercept/continuum/continuum_dy/lightllm/mars
SCHEDULING_POLICY="${SCHEDULING_POLICY:-mars}"
MARS_ACTIVE_POOL_SIZE="${MARS_ACTIVE_POOL_SIZE:-8}"
MARS_CPU_BACKLOG_LOW="${MARS_CPU_BACKLOG_LOW:-1}"
MARS_CPU_BACKLOG_HIGH="${MARS_CPU_BACKLOG_HIGH:-4}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
CPU_SERVER="${CPU_SERVER:-0-1}"
CPU_AGENT="${CPU_AGENT:-2-9}"
CPU_GPU="${CPU_GPU:-${CPU_SERVER}}"
CPU_TOOL="${CPU_TOOL:-${CPU_AGENT}}"
REQUIRE_DISJOINT_SERVER_AGENT_CPU="${REQUIRE_DISJOINT_SERVER_AGENT_CPU:-1}"
ALLOW_CONCURRENT_RUNS="${ALLOW_CONCURRENT_RUNS:-0}"
RUN_LOCK_PATH="${RUN_LOCK_PATH:-/tmp/mars_agent_cli_exclusive_${UID}.lock}"

# Exact OpenHands MARS admission/window controller settings. These are carried
# through for every policy; the shared harness activates them only for mars.
EMIT_MODE="${EMIT_MODE:-rps}"
THREAD_POOL="${THREAD_POOL:-on}"
AUTELLIX_FCFS_AFTER_FINISHED="${AUTELLIX_FCFS_AFTER_FINISHED:-0}"
MARS_LONG_PREFILL_TOKENS="${MARS_LONG_PREFILL_TOKENS:-50000}"
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
MARS_TAIL_MAX_INFLIGHT="${MARS_TAIL_MAX_INFLIGHT:-4}"
MARS_TAIL_KV_BUDGET_RATIO="${MARS_TAIL_KV_BUDGET_RATIO:-1.0}"
MARS_NO_KV_ACTIVE_LIMIT="${MARS_NO_KV_ACTIVE_LIMIT:-2}"
CONCUR_WINDOW_INIT="${CONCUR_WINDOW_INIT:-1}"
CONCUR_MAX_ACTIVE_AGENTS="${CONCUR_MAX_ACTIVE_AGENTS:-0}"
CONCUR_CONTROL_INTERVAL_S="${CONCUR_CONTROL_INTERVAL_S:-0.2}"
CONCUR_ADDITIVE_INCREASE="${CONCUR_ADDITIVE_INCREASE:-2}"
CONCUR_MULTIPLICATIVE_DECREASE="${CONCUR_MULTIPLICATIVE_DECREASE:-0.5}"
CONCUR_UTIL_LOW="${CONCUR_UTIL_LOW:-0.2}"
CONCUR_UTIL_HIGH="${CONCUR_UTIL_HIGH:-0.5}"
CONCUR_HIT_RATE_THRESHOLD="${CONCUR_HIT_RATE_THRESHOLD:-0.2}"
PREFILL_TOKENIZER="${PREFILL_TOKENIZER:-${MODEL_PATH}}"
# Client-side admission policy. Defaults to the server policy; use
# continuum_con/continuum_dy_con/fcfs_con to add CONCUR admission control.
WORKLOAD_SCHEDULING_POLICY="${WORKLOAD_SCHEDULING_POLICY:-${SCHEDULING_POLICY}}"

KV_EVENTS="${KV_EVENTS:-1}"
KV_EVENTS_ENDPOINT="${KV_EVENTS_ENDPOINT:-}"
KV_EVENTS_CONFIG="${KV_EVENTS_CONFIG:-}"

USE_EXISTING_VLLM="${USE_EXISTING_VLLM:-0}"
BASE_URL="${BASE_URL:-}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-}"
KEEP_VLLM_SERVER="${KEEP_VLLM_SERVER:-0}"
SERVER_TERM_GRACE_S="${SERVER_TERM_GRACE_S:-10}"

# Results follow the OpenHands policy/RUN_LABEL/TIMESTAMP hierarchy. A
# single-agent run writes artifacts directly in RUN_DIR; only an explicitly
# requested multi-agent debug run receives per-agent child directories.
OUTPUT_ROOT="${OUTPUT_ROOT:-${MARS_DIR}/outputs/agent_cli}"
RUN_LABEL="${RUN_LABEL:-qwen3_dynamic_agents}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${SCHEDULING_POLICY}/${RUN_LABEL}/${TIMESTAMP}}"
VLLM_LOG_PATH="${VLLM_LOG_PATH:-${RUN_DIR}/vllm_server.log}"
LAUNCHER_LOG_PATH="${LAUNCHER_LOG_PATH:-${RUN_DIR}/launcher.log}"
LOCAL_INPUTS_DIR="${LOCAL_INPUTS_DIR:-${RUN_DIR}/_localized_inputs}"
WARMUP_ARTIFACT="${WARMUP_ARTIFACT:-${RUN_DIR}/warmup.jsonl}"

die() {
  echo "$*" >&2
  exit 1
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

normalize_existing_openai_base_url() {
  "${MARS_PYTHON}" - "$1" <<'PY'
import sys
import urllib.parse

value = sys.argv[1].strip()
parsed = urllib.parse.urlsplit(value)
if parsed.scheme not in {"http", "https"} or not parsed.hostname:
    raise SystemExit("BASE_URL must be an absolute http(s) URL")
if parsed.query or parsed.fragment:
    raise SystemExit("BASE_URL must not contain a query string or fragment")
path = parsed.path.rstrip("/")
if not path.endswith("/v1"):
    path = f"{path}/v1" if path else "/v1"
print(urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", "")))
PY
}

SERVER_PID=""
SERVER_IS_PROCESS_GROUP=0
CLEANUP_DONE=0
RUN_LOCK_FD=""

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

preflight() {
  [[ -x "${MARS_PYTHON}" ]] || die "MARS_PYTHON is not executable: ${MARS_PYTHON}"
  [[ -x "${MARS_NODE_BIN}" ]] || die "Pinned Node runtime is missing: ${MARS_NODE_BIN}"
  [[ "$("${MARS_NODE_BIN}" --version)" == "v${NODE_VERSION}" ]] || die \
    "Unexpected Node runtime: $("${MARS_NODE_BIN}" --version); expected v${NODE_VERSION}"
  [[ -f "${RUNNER}" ]] || die "Runner is missing: ${RUNNER}"
  [[ -f "${TASKS_PATH}" ]] || die "Tasks file is missing: ${TASKS_PATH}"
  local agent
  IFS=',' read -r -a REQUESTED_AGENTS <<< "${AGENTS// /}"
  (( ${#REQUESTED_AGENTS[@]} > 0 )) || die "AGENTS is empty"
  for agent in "${REQUESTED_AGENTS[@]}"; do
    case "${agent}" in
      opencode|codex) ;;
      *) die "Unsupported agent in AGENTS: ${agent}" ;;
    esac
  done
  if ! mars_bool_true "${ALLOW_UNSANDBOXED_AGENT}" \
    && ! mars_bool_true "${DRY_RUN}"; then
    [[ -x "${MARS_BWRAP_BIN}" ]] || die \
      "Agent CLIs require bubblewrap; set MARS_BWRAP_BIN or ALLOW_UNSANDBOXED_AGENT=1 for debugging"
    "${MARS_BWRAP_BIN}" \
      --ro-bind /usr /usr \
      --ro-bind /bin /bin \
      --ro-bind /lib /lib \
      --ro-bind /lib64 /lib64 \
      --proc /proc \
      /bin/true >/dev/null 2>&1 || die \
      "bubblewrap is installed but unusable; set ALLOW_UNSANDBOXED_AGENT=1 only for debugging"
  fi
  if [[ -n "${CPU_AGENT}" ]]; then
    [[ -x /usr/bin/taskset ]] || die "CPU_AGENT requires /usr/bin/taskset"
    /usr/bin/taskset -c "${CPU_AGENT}" /bin/true >/dev/null 2>&1 || die \
      "CPU_AGENT is invalid or unavailable in this cpuset: ${CPU_AGENT}"
  fi
  if ! mars_bool_true "${USE_EXISTING_VLLM}" \
    && mars_bool_true "${REQUIRE_DISJOINT_SERVER_AGENT_CPU}" \
    && [[ -n "${CPU_SERVER}" && -n "${CPU_AGENT}" ]]; then
    "${MARS_PYTHON}" - "${CPU_SERVER}" "${CPU_AGENT}" <<'PY' || die \
      "CPU_SERVER and CPU_AGENT must be disjoint"
import sys

def expand(spec):
    result = set()
    for part in spec.split(","):
        bounds = part.strip().split("-", 1)
        start = int(bounds[0])
        end = int(bounds[-1])
        result.update(range(start, end + 1))
    return result

raise SystemExit(1 if expand(sys.argv[1]) & expand(sys.argv[2]) else 0)
PY
  fi
  if ! mars_bool_true "${DRY_RUN}"; then
    [[ -x "${MARS_AGENT_BASELINES_ROOT}/runtime/opencode/node_modules/.bin/opencode" ]] || die \
      "OpenCode runtime is missing; run ${SETUP_SCRIPT}"
    [[ -x "${MARS_AGENT_BASELINES_ROOT}/runtime/codex/node_modules/.bin/codex" ]] || die \
      "Codex runtime is missing; run ${SETUP_SCRIPT}"
  fi
  if mars_bool_true "${USE_EXISTING_VLLM}"; then
    [[ -n "${BASE_URL}" ]] || die "BASE_URL is required with USE_EXISTING_VLLM=1"
    local normalized_base_url
    normalized_base_url="$(normalize_existing_openai_base_url "${BASE_URL}")" || \
      die "Invalid BASE_URL for USE_EXISTING_VLLM=1: ${BASE_URL}"
    BASE_URL="${normalized_base_url}"
  elif ! mars_bool_true "${DRY_RUN}"; then
    [[ -d "${MODEL_PATH}" ]] || die "Model directory is missing: ${MODEL_PATH}"
    mars_ensure_vllm_flash_attn_extensions "${MARS_DIR}" "${MARS_PYTHON}"
  fi
  if [[ -e "${RUN_DIR}" && -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
    && ! mars_bool_true "${RESUME}" && ! mars_bool_true "${DRY_RUN}"; then
    die "Refusing to reuse non-empty RUN_DIR without RESUME=1: ${RUN_DIR}"
  fi
  return 0
}

acquire_run_lock() {
  mars_bool_true "${ALLOW_CONCURRENT_RUNS}" && return 0
  command -v flock >/dev/null 2>&1 || die "flock is required for exclusive agent-CLI runs"
  mkdir -p "$(dirname "${RUN_LOCK_PATH}")"
  exec {RUN_LOCK_FD}>"${RUN_LOCK_PATH}"
  flock -n "${RUN_LOCK_FD}" || die \
    "Another agent-CLI launcher holds ${RUN_LOCK_PATH}; wait for it or use ALLOW_CONCURRENT_RUNS=1 explicitly"
}

build_server_command() {
  [[ -z "${TP}" ]] && TP="$(mars_infer_tp_from_devices "${CUDA_VISIBLE_DEVICES}")"
  [[ -z "${VLLM_PORT}" ]] && VLLM_PORT="$(mars_pick_free_port "${MARS_PYTHON}")"
  BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
  if mars_bool_true "${KV_EVENTS}" && [[ -z "${KV_EVENTS_CONFIG}" ]]; then
    [[ -n "${KV_EVENTS_ENDPOINT}" ]] || \
      KV_EVENTS_ENDPOINT="tcp://127.0.0.1:$(mars_pick_free_port "${MARS_PYTHON}")"
    local kv_bind_endpoint="${KV_EVENTS_ENDPOINT/tcp:\/\/127.0.0.1:/tcp://*:}"
    [[ "${kv_bind_endpoint}" == "${KV_EVENTS_ENDPOINT}" ]] && \
      kv_bind_endpoint="${KV_EVENTS_ENDPOINT/tcp:\/\/localhost:/tcp://*:}"
    KV_EVENTS_CONFIG="$(printf '{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"%s\",\"topic\":\"kv-events\"}' "${kv_bind_endpoint}")"
  fi
  ADDITIONAL_CONFIG="$(
    mars_merge_additional_config \
      "${MARS_PYTHON}" \
      "${ADDITIONAL_CONFIG}" \
      "${MARS_ACTIVE_POOL_SIZE}" \
      "${MARS_CPU_BACKLOG_LOW}" \
      "${MARS_CPU_BACKLOG_HIGH}"
  )"
  # Always use the maintained server entrypoint.  Besides the MARS-only KV
  # stats route it installs the strict Qwen Responses provider used by native
  # Codex.  All four scheduling policies must expose the same model protocol.
  local entrypoint="${MARS_DIR}/experiments/openhands/vllm_api_server_mars.py"
  SERVER_CMD=("${MARS_PYTHON}" "${entrypoint}")
  REQUIRE_KV_STATS_READY=0
  if [[ "${SCHEDULING_POLICY}" == "mars" ]]; then
    REQUIRE_KV_STATS_READY=1
  fi
  SERVER_CMD+=(
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
    --additional-config "${ADDITIONAL_CONFIG}"
  )
  [[ -n "${KV_EVENTS_CONFIG}" ]] && SERVER_CMD+=(--kv-events-config "${KV_EVENTS_CONFIG}")
  mars_bool_true "${ENABLE_AUTO_TOOL_CHOICE}" && SERVER_CMD+=(
    --enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}"
  )
  mars_bool_true "${TRUST_REMOTE_CODE}" && SERVER_CMD+=(--trust-remote-code)
  mars_bool_true "${ENFORCE_EAGER}" && SERVER_CMD+=(--enforce-eager)
  return 0
}

start_server() {
  local pids
  if ! mars_bool_true "${ALLOW_CONCURRENT_RUNS}"; then
    pids="$(mars_gpu_compute_pids "${CUDA_VISIBLE_DEVICES}")"
    [[ -z "${pids}" ]] || die \
      "Selected GPU(s) are busy (compute PIDs: ${pids}); choose free GPUs or set ALLOW_CONCURRENT_RUNS=1"
  fi
  mars_assert_local_port_available "${MARS_PYTHON}" "${VLLM_HOST}" "${VLLM_PORT}" "vLLM"
  local launch=(
    env
    -u VLLM_PORT
    PYTHONPATH="${MARS_DIR}:${MARS_DIR}/vllm-continuum${PYTHONPATH:+:${PYTHONPATH}}"
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    RUN_OUTPUT_DIR="${RUN_DIR}"
  )
  if command -v taskset >/dev/null 2>&1 && [[ -n "${CPU_SERVER}" ]]; then
    launch+=(taskset -c "${CPU_SERVER}")
  fi
  launch+=("${SERVER_CMD[@]}")
  echo "== Starting MARS/vLLM =="
  print_command "${launch[@]}"
  if command -v setsid >/dev/null 2>&1; then
    setsid "${launch[@]}" >"${VLLM_LOG_PATH}" 2>&1 &
    SERVER_IS_PROCESS_GROUP=1
  else
    "${launch[@]}" >"${VLLM_LOG_PATH}" 2>&1 &
  fi
  SERVER_PID=$!
  mars_wait_for_vllm \
    "${MARS_PYTHON}" "${BASE_URL}" 300 "${SERVER_PID}" "${VLLM_LOG_PATH}" \
    "${REQUIRE_KV_STATS_READY}" "${RUN_DIR}/kv_cache_stats_ready.json"
  echo "vLLM ready: ${BASE_URL}"
}

run_openai_compat_warmup() {
  mars_bool_true "${OPENAI_COMPAT_WARMUP}" || return 0
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

write_server_provenance() {
  local path="${RUN_DIR}/server_provenance.txt"
  local diff_path="${RUN_DIR}/mars_tracked_diff.patch"
  local status_path="${RUN_DIR}/mars_git_status.txt"
  git -C "${MARS_DIR}" diff --binary --no-ext-diff >"${diff_path}" 2>/dev/null || true
  git -C "${MARS_DIR}" status --porcelain=v1 >"${status_path}" 2>/dev/null || true
  {
    printf 'captured_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'base_url=%s\n' "${BASE_URL}"
    printf 'model_path=%s\n' "${MODEL_PATH}"
    printf 'model_name=%s\n' "${MODEL_NAME}"
    printf 'scheduling_policy=%s\n' "${SCHEDULING_POLICY}"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
    printf 'cpu_server=%s\n' "${CPU_SERVER}"
    printf 'cpu_agent=%s\n' "${CPU_AGENT}"
    printf 'node_bin=%s\n' "${MARS_NODE_BIN}"
    printf 'node_version=%s\n' "$("${MARS_NODE_BIN}" --version)"
    printf 'bubblewrap_bin=%s\n' "${MARS_BWRAP_BIN}"
    printf 'bubblewrap_version=%s\n' "$("${MARS_BWRAP_BIN}" --version 2>/dev/null || echo unavailable)"
    printf 'launcher_sha256=%s\n' "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
    printf 'common_runtime_sha256=%s\n' "$(sha256sum "${SCRIPT_DIR}/common_runtime.sh" | awk '{print $1}')"
    printf 'vllm_entrypoint_sha256=%s\n' "$(sha256sum "${MARS_DIR}/experiments/openhands/vllm_api_server_mars.py" | awk '{print $1}')"
    printf 'qwen_responses_compat_sha256=%s\n' "$(sha256sum "${MARS_DIR}/experiments/openhands/qwen_responses_compat.py" | awk '{print $1}')"
    printf 'run_lock_path=%s\n' "${RUN_LOCK_PATH}"
    printf 'mars_repo_commit=%s\n' "$(git -C "${MARS_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'mars_repo_dirty_files=%s\n' "$(git -C "${MARS_DIR}" status --porcelain 2>/dev/null | wc -l)"
    printf 'mars_tracked_diff_path=%s\n' "${diff_path}"
    printf 'mars_tracked_diff_sha256=%s\n' "$(sha256sum "${diff_path}" | awk '{print $1}')"
    printf 'mars_git_status_path=%s\n' "${status_path}"
    printf 'mars_git_status_sha256=%s\n' "$(sha256sum "${status_path}" | awk '{print $1}')"
    printf 'vllm_repo_commit=%s\n' "$(git -C "${MARS_DIR}/vllm-continuum" rev-parse HEAD 2>/dev/null || echo unknown)"
    if mars_bool_true "${USE_EXISTING_VLLM}"; then
      printf 'server_command=external_server_not_launched_by_this_script\n'
    else
      printf 'server_command='
      printf '%q ' "${SERVER_CMD[@]}"
      printf '\n'
    fi
  } >"${path}"
}

build_agent_command() {
  local agent="$1"
  local agent_out="${RUN_DIR}"
  if (( ${#REQUESTED_AGENTS[@]} > 1 )); then
    agent_out="${RUN_DIR}/${agent}"
  fi
  AGENT_CMD=(
    "${MARS_PYTHON}"
    "${RUNNER}"
    --agent "${agent}"
    --tasks "${TASKS_PATH}"
    --out-dir "${agent_out}"
    --workspace-root "${agent_out}/workspaces"
    --local-inputs-dir "${LOCAL_INPUTS_DIR}"
    --model "${MODEL_NAME}"
    --base-url "${BASE_URL}"
    --api-key "${API_KEY}"
    --rps "${RPS}"
    --emit-mode "${EMIT_MODE}"
    --max-workers "${MAX_WORKERS}"
    --seed "${SEED}"
    --context-window "${MAX_MODEL_LEN}"
    --max-output-tokens "${MAX_OUTPUT_TOKENS}"
    --max-iterations "${MAX_ITERATIONS}"
    --temperature "${TEMPERATURE}"
    --llm-timeout-s "${LLM_TIMEOUT_S}"
    --llm-num-retries "${LLM_NUM_RETRIES}"
    --gpu-timeout-s "${GPU_TIMEOUT_S}"
    --scheduling-policy "${WORKLOAD_SCHEDULING_POLICY}"
    --concur-window-init "${CONCUR_WINDOW_INIT}"
    --concur-max-active-agents "${CONCUR_MAX_ACTIVE_AGENTS}"
    --concur-control-interval-s "${CONCUR_CONTROL_INTERVAL_S}"
    --concur-additive-increase "${CONCUR_ADDITIVE_INCREASE}"
    --concur-multiplicative-decrease "${CONCUR_MULTIPLICATIVE_DECREASE}"
    --concur-util-low "${CONCUR_UTIL_LOW}"
    --concur-util-high "${CONCUR_UTIL_HIGH}"
    --concur-hit-rate-threshold "${CONCUR_HIT_RATE_THRESHOLD}"
    --thread-pool "${THREAD_POOL}"
    --cpu-gpu "${CPU_GPU}"
    --cpu-tool "${CPU_TOOL}"
    --autellix-tail-fcfs-after-finished "${AUTELLIX_FCFS_AFTER_FINISHED}"
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
    --prefill-tokenizer "${PREFILL_TOKENIZER}"
  )
  [[ -n "${TOP_P}" ]] && AGENT_CMD+=(--top-p "${TOP_P}")
  if [[ "${agent}" == "codex" ]] && mars_bool_true "${CODEX_ENABLE_SUBAGENTS}"; then
    AGENT_CMD+=(--codex-enable-subagents)
  fi
  if mars_bool_true "${KV_EVENTS}"; then
    AGENT_CMD+=(--kv-events --kv-events-endpoint "${KV_EVENTS_ENDPOINT}")
  fi
  [[ -n "${CPU_AGENT}" ]] && AGENT_CMD+=(--cpu-affinity "${CPU_AGENT}")
  [[ -n "${MAX_REQUESTS}" ]] && AGENT_CMD+=(--max-requests "${MAX_REQUESTS}")
  mars_bool_true "${SHUFFLE}" && AGENT_CMD+=(--shuffle)
  mars_bool_true "${RESUME}" && AGENT_CMD+=(--resume)
  mars_bool_true "${RERUN_FAILURES}" && AGENT_CMD+=(--rerun-failures)
  mars_bool_true "${ALLOW_UNSANDBOXED_AGENT}" && AGENT_CMD+=(--allow-unsandboxed-agent)
  return 0
}

main() {
  preflight
  if ! mars_bool_true "${USE_EXISTING_VLLM}"; then
    build_server_command
  fi

  if mars_bool_true "${DRY_RUN}"; then
    echo "RUN_DIR=${RUN_DIR}"
    echo "BASE_URL=${BASE_URL:-<local-server-url-assigned-at-runtime>}"
    if ! mars_bool_true "${USE_EXISTING_VLLM}"; then
      echo "Server command:"
      print_command "${SERVER_CMD[@]}"
    fi
    local dry_agent
    for dry_agent in "${REQUESTED_AGENTS[@]}"; do
      build_agent_command "${dry_agent}"
      echo "${dry_agent} command:"
      print_command "${AGENT_CMD[@]}"
    done
    return 0
  fi

  mkdir -p "${RUN_DIR}" "${LOCAL_INPUTS_DIR}"
  acquire_run_lock
  : >"${LAUNCHER_LOG_PATH}"
  mars_bool_true "${USE_EXISTING_VLLM}" || : >"${VLLM_LOG_PATH}"
  exec > >(tee -a "${LAUNCHER_LOG_PATH}") 2>&1
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  write_server_provenance
  if ! mars_bool_true "${USE_EXISTING_VLLM}"; then
    start_server
  fi
  run_openai_compat_warmup
  echo "== Agent CLI workload =="
  echo "agents=${AGENTS} tasks=${TASKS_PATH} rps=${RPS} server_policy=${SCHEDULING_POLICY} workload_policy=${WORKLOAD_SCHEDULING_POLICY} cpu_agent=${CPU_AGENT}"
  echo "llm_timeout_s=${LLM_TIMEOUT_S} llm_num_retries=${LLM_NUM_RETRIES} gpu_timeout_s=${GPU_TIMEOUT_S}"
  local agent
  local agent_rc
  local overall_rc=0
  for agent in "${REQUESTED_AGENTS[@]}"; do
    build_agent_command "${agent}"
    echo "== Running ${agent} =="
    print_command "${AGENT_CMD[@]}"
    set +e
    "${AGENT_CMD[@]}" 2>&1 | tee -a "${RUN_DIR}/${agent}_workload.log"
    agent_rc="${PIPESTATUS[0]}"
    set -e
    if [[ "${agent_rc}" != "0" ]]; then
      overall_rc="${agent_rc}"
      echo "${agent} reached terminal state with failures (exit ${agent_rc})"
      mars_bool_true "${STOP_ON_FAILURE}" && break
    fi
  done
  if [[ "${overall_rc}" != "0" ]]; then
    echo "Dynamic-agent run finished with failures: ${RUN_DIR}"
    return "${overall_rc}"
  fi
  echo "Completed dynamic-agent run: ${RUN_DIR}"
}

main "$@"
