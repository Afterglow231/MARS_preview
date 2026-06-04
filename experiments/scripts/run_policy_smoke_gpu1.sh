#!/usr/bin/env bash
set -euo pipefail

# Smoke-test all bundled scheduling policies on one GPU.
# Defaults: vLLM replay uses 200 requests; OpenHands uses 50 requests.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST_HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
source "${SCRIPT_DIR}/common_runtime.sh"

BASELINE_SCRIPT="${SCRIPT_DIR}/run_baseline_continuum.sh"
OPENHANDS_SCRIPT="${SCRIPT_DIR}/run_openhands_qwen3_local.sh"
SOURCE_TASKS_PATH="${SOURCE_TASKS_PATH:-${MARS_DIR}/benchmark/hybrid_workloads/test_workload.jsonl}"
MARS_PYTHON="${MARS_PYTHON:-$(mars_default_python "${HOST_HOME}/miniconda3/envs/mars/bin/python")}"

GPU="${GPU:-1}"
POLICIES="${POLICIES:-continuum continuum_dy autellix fcfs infercept mars}"
TARGETS="${TARGETS:-vllm openhands}"
RPS="${RPS:-0.2}"
MAX_WORKERS="${MAX_WORKERS:-0}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-10}"
DRY_RUN="${DRY_RUN:-0}"

VLLM_REQUESTS="${VLLM_REQUESTS:-${REQUESTS:-200}}"
OPENHANDS_REQUESTS="${OPENHANDS_REQUESTS:-${REQUESTS:-50}}"

CPU_TOOL="${CPU_TOOL:-16-23}"
CPU_GPU="${CPU_GPU:-0-1}"
CPU_CORES="${CPU_CORES:-${CPU_GPU},${CPU_TOOL}}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${MARS_DIR}/outputs/policy_smoke_gpu${GPU}}"
RUN_ROOT="${OUT_ROOT}/${RUN_ID}"
VLLM_TASKS_SLICE="${RUN_ROOT}/test_workload_first${VLLM_REQUESTS}_vllm.jsonl"
OPENHANDS_TASKS_SLICE="${RUN_ROOT}/test_workload_first${OPENHANDS_REQUESTS}_openhands.jsonl"
SUMMARY_PATH="${RUN_ROOT}/summary.tsv"

die() {
  echo "$*" >&2
  exit 1
}

words_to_array() {
  local raw="$1"
  raw="${raw//,/ }"
  printf '%s\n' "${raw}"
}

require_positive_int() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 )) || die "${name} must be a positive integer, got: ${value}"
}

target_enabled() {
  local needle="$1"
  local target
  for target in "${TARGET_LIST[@]}"; do
    [[ "${target}" == "${needle}" ]] && return 0
  done
  return 1
}

request_count_for_target() {
  case "$1" in
    vllm) echo "${VLLM_REQUESTS}" ;;
    openhands) echo "${OPENHANDS_REQUESTS}" ;;
    *) die "Unknown target: $1" ;;
  esac
}

slice_path_for_target() {
  case "$1" in
    vllm) echo "${VLLM_TASKS_SLICE}" ;;
    openhands) echo "${OPENHANDS_TASKS_SLICE}" ;;
    *) die "Unknown target: $1" ;;
  esac
}

make_task_slice() {
  local limit="$1"
  local out_path="$2"

  mkdir -p "$(dirname "${out_path}")"
  awk -v limit="${limit}" '
    NF {
      print
      n += 1
      if (n >= limit) exit
    }
    END {
      if (n < limit) {
        printf("Requested %d tasks, but only found %d non-empty rows.\n", limit, n) > "/dev/stderr"
        exit 1
      }
    }
  ' "${SOURCE_TASKS_PATH}" > "${out_path}"

  PYTHONDONTWRITEBYTECODE=1 python3 - "${out_path}" "${limit}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
rows = 0
for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    try:
        json.loads(line)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    rows += 1
if rows != expected:
    raise SystemExit(f"{path} has {rows} JSONL rows, expected {expected}")
PY
}

preflight() {
  [[ -f "${SOURCE_TASKS_PATH}" ]] || die "Source tasks file not found: ${SOURCE_TASKS_PATH}"
  [[ -x "${MARS_PYTHON}" ]] || die "MARS_PYTHON not found or not executable: ${MARS_PYTHON}"
  require_positive_int "VLLM_REQUESTS" "${VLLM_REQUESTS}"
  require_positive_int "OPENHANDS_REQUESTS" "${OPENHANDS_REQUESTS}"

  local target
  for target in "${TARGET_LIST[@]}"; do
    case "${target}" in
      vllm) [[ -f "${BASELINE_SCRIPT}" ]] || die "Baseline script not found: ${BASELINE_SCRIPT}" ;;
      openhands) [[ -f "${OPENHANDS_SCRIPT}" ]] || die "OpenHands script not found: ${OPENHANDS_SCRIPT}" ;;
      *) die "Unknown target: ${target} (expected: vllm, openhands)" ;;
    esac
  done

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i "${GPU}" --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true
  fi
  mars_ensure_vllm_flash_attn_extensions "${MARS_DIR}" "${MARS_PYTHON}"
}

prepare_workloads() {
  local target
  for target in "${TARGET_LIST[@]}"; do
    make_task_slice "$(request_count_for_target "${target}")" "$(slice_path_for_target "${target}")"
  done
}

run_vllm_policy() {
  local policy="$1"
  local run_dir="${RUN_ROOT}/vllm/${policy}"
  mkdir -p "${run_dir}"

  (
    env \
      CUDA_VISIBLE_DEVICES="${GPU}" \
      GPU_GUARD_DEVICES="${GPU}" \
      PYTHON_BIN="${MARS_PYTHON}" \
      TP=1 \
      TASKS_PATH="${VLLM_TASKS_SLICE}" \
      NUM_REQUESTS="${VLLM_REQUESTS}" \
      RPS="${RPS}" \
      SCHEDULING_POLICY="${policy}" \
      AUTELLIX_TAIL_FCFS=0 \
      AUTELLIX_TAIL_FCFS_AFTER_FINISHED=0 \
      AUTELLIX_FCFS_AFTER_FINISHED=0 \
      CPU_TOOL="${CPU_TOOL}" \
      CPU_GPU="${CPU_GPU}" \
      bash "${BASELINE_SCRIPT}" \
        --output-dir "${run_dir}"
  ) 2>&1 | tee "${run_dir}/driver.log"
  return "${PIPESTATUS[0]}"
}

run_openhands_policy() {
  local policy="$1"
  local run_dir="${RUN_ROOT}/openhands/${policy}"
  mkdir -p "${run_dir}"

  (
    env \
      CUDA_VISIBLE_DEVICES="${GPU}" \
      GPU_GUARD_DEVICES="${GPU}" \
      MARS_PYTHON="${MARS_PYTHON}" \
      TP=1 \
      TASKS_PATH="${OPENHANDS_TASKS_SLICE}" \
      NUM_REQUESTS="${OPENHANDS_REQUESTS}" \
      MAX_REQUESTS="${OPENHANDS_REQUESTS}" \
      SHUFFLE=0 \
      RPS="${RPS}" \
      MAX_WORKERS="${MAX_WORKERS}" \
      SCHEDULING_POLICY="${policy}" \
      AUTELLIX_TAIL_FCFS=0 \
      AUTELLIX_TAIL_FCFS_AFTER_FINISHED=0 \
      AUTELLIX_FCFS_AFTER_FINISHED=0 \
      CPU_CORES="${CPU_CORES}" \
      CPU_TOOL="${CPU_TOOL}" \
      CPU_GPU="${CPU_GPU}" \
      OUT_DIR="${run_dir}" \
      RUN_LABEL="policy_smoke_${policy}" \
      bash "${OPENHANDS_SCRIPT}"
  ) 2>&1 | tee "${run_dir}/driver.log"
  return "${PIPESTATUS[0]}"
}

run_target_policy() {
  local target="$1"
  local policy="$2"
  case "${target}" in
    vllm) run_vllm_policy "${policy}" ;;
    openhands) run_openhands_policy "${policy}" ;;
    *) die "Unknown target: ${target}" ;;
  esac
}

run_one() {
  local target="$1"
  local policy="$2"
  local run_dir="${RUN_ROOT}/${target}/${policy}"
  local requests
  local start end rc status duration

  requests="$(request_count_for_target "${target}")"
  start="$(date +%s)"

  echo
  echo "== ${target} / ${policy} =="
  echo "requests=${requests}"
  echo "run_dir=${run_dir}"

  set +e
  run_target_policy "${target}" "${policy}"
  rc=$?
  set -e

  end="$(date +%s)"
  duration=$((end - start))
  [[ "${rc}" -eq 0 ]] && status="pass" || status="fail"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${target}" "${policy}" "${requests}" "${status}" "${rc}" "${duration}" "${run_dir}" \
    | tee -a "${SUMMARY_PATH}"

  if [[ "${rc}" -ne 0 ]] && mars_bool_true "${STOP_ON_FAILURE}"; then
    exit "${rc}"
  fi
  return "${rc}"
}

print_config() {
  echo "== Policy smoke matrix =="
  echo "MARS_DIR=${MARS_DIR}"
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "GPU=${GPU}"
  target_enabled vllm && echo "VLLM_REQUESTS=${VLLM_REQUESTS}" && echo "VLLM_TASKS_SLICE=${VLLM_TASKS_SLICE}"
  target_enabled openhands && echo "OPENHANDS_REQUESTS=${OPENHANDS_REQUESTS}" && echo "OPENHANDS_TASKS_SLICE=${OPENHANDS_TASKS_SLICE}"
  echo "RPS=${RPS}"
  echo "CPU_CORES=${CPU_CORES}"
  echo "CPU_GPU=${CPU_GPU}"
  echo "CPU_TOOL=${CPU_TOOL}"
  echo "POLICIES=${POLICIES}"
  echo "TARGETS=${TARGETS}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "SUMMARY_PATH=${SUMMARY_PATH}"
}

main() {
  read -r -a POLICY_LIST <<< "$(words_to_array "${POLICIES}")"
  read -r -a TARGET_LIST <<< "$(words_to_array "${TARGETS}")"
  ((${#POLICY_LIST[@]} > 0)) || die "POLICIES is empty"
  ((${#TARGET_LIST[@]} > 0)) || die "TARGETS is empty"

  preflight
  prepare_workloads
  printf 'target\tpolicy\trequests\tstatus\texit_code\tduration_s\trun_dir\n' > "${SUMMARY_PATH}"
  print_config

  if mars_bool_true "${DRY_RUN}"; then
    echo "DRY_RUN=1; preflight complete, skipping policy runs."
    echo "Artifacts: ${RUN_ROOT}"
    return 0
  fi

  local failures=0
  local target policy
  for target in "${TARGET_LIST[@]}"; do
    for policy in "${POLICY_LIST[@]}"; do
      if ! run_one "${target}" "${policy}"; then
        failures=$((failures + 1))
      fi
      if (( SLEEP_BETWEEN_RUNS > 0 )); then
        sleep "${SLEEP_BETWEEN_RUNS}"
      fi
    done
  done

  echo
  echo "== Summary =="
  cat "${SUMMARY_PATH}"
  echo "Artifacts: ${RUN_ROOT}"

  if (( failures > 0 )); then
    echo "Completed with ${failures} failed run(s)." >&2
    exit 1
  fi
}

main "$@"
