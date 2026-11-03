#!/usr/bin/env bash
set -euo pipefail

# One real-model request through the selected pinned native frontend(s). This
# is the production protocol gate before a policy/RPS sweep. The default keeps
# the historical Codex+OpenCode gate; set AGENTS=codex for the Codex-only path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAUNCHER="${MARS_DIR}/experiments/scripts/run_agent_cli_local.sh"
TASKS_PATH="${TASKS_PATH:-${MARS_DIR}/experiments/agent_cli/fixtures/native_gpu_smoke.jsonl}"
SMOKE_ID="${SMOKE_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MARS_DIR}/outputs/native_agent_gpu_smoke}"
RUN_LABEL="${RUN_LABEL:-qwen3_native_frontend_gate}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-fcfs}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${SCHEDULING_POLICY}/${RUN_LABEL}/${SMOKE_ID}}"
DRY_RUN="${DRY_RUN:-0}"
AGENTS="${AGENTS:-codex,opencode}"

die() {
  echo "$*" >&2
  exit 1
}

bool_true() {
  case "${1:-}" in
    1|on|true|yes|y|TRUE|ON|True|YES|Y) return 0 ;;
    *) return 1 ;;
  esac
}

[[ -f "${TASKS_PATH}" ]] || die "Smoke workload not found: ${TASKS_PATH}"
[[ -f "${LAUNCHER}" ]] || die "Native launcher not found: ${LAUNCHER}"
[[ ! -e "${RUN_DIR}" ]] || die "Refusing to reuse smoke RUN_DIR: ${RUN_DIR}"

env \
  AGENTS="${AGENTS}" \
  TASKS_PATH="${TASKS_PATH}" \
  MAX_REQUESTS=1 \
  MAX_WORKERS=1 \
  RPS=0.2 \
  MAX_ITERATIONS="${MAX_ITERATIONS:-100}" \
  MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-2048}" \
  LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-3000}" \
  GPU_TIMEOUT_S="${GPU_TIMEOUT_S:-1800}" \
  SCHEDULING_POLICY="${SCHEDULING_POLICY}" \
  WORKLOAD_SCHEDULING_POLICY=fcfs \
  KV_EVENTS=0 \
  OPENAI_COMPAT_WARMUP=1 \
  WARMUP_REQUESTS=1 \
  WARMUP_MAX_TOKENS=32 \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_LABEL="${RUN_LABEL}" \
  TIMESTAMP="${SMOKE_ID}" \
  RUN_DIR="${RUN_DIR}" \
  STOP_ON_FAILURE=1 \
  RESUME=0 \
  RERUN_FAILURES=0 \
  DRY_RUN="${DRY_RUN}" \
  bash "${LAUNCHER}"

if bool_true "${DRY_RUN}"; then
  echo "SMOKE_DRY_RUN_DIR=${RUN_DIR}"
  exit 0
fi

python3 - "${RUN_DIR}" "${AGENTS}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1]).resolve()
agents = tuple(item.strip() for item in sys.argv[2].split(",") if item.strip())
errors = []
expected_protocol = {
    "codex": "codex_app_server_v2",
    "opencode": "opencode_legacy_server_sse",
}
expected_raw = {
    "codex": "codex_app_server.raw.jsonl",
    "opencode": "native_events.jsonl",
}
for agent in agents:
    if agent not in expected_protocol:
        errors.append(f"unsupported smoke agent: {agent}")
        continue
    agent_root = root if len(agents) == 1 else root / agent
    summary_path = agent_root / "summary.json"
    traces_path = agent_root / "traces.jsonl"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{summary_path}: {type(exc).__name__}: {exc}")
        continue
    run = summary.get("run") or {}
    if run.get("requested_count", summary.get("requested_count")) != 1:
        errors.append(f"{summary_path}: requested_count is not 1")
    if run.get("completed_requests", summary.get("count")) != 1:
        errors.append(f"{summary_path}: completed_requests is not 1")
    if run.get("completed_all_requests", summary.get("completed_all_requests")) is not True:
        errors.append(f"{summary_path}: completed_all_requests is not true")
    try:
        traces = [json.loads(line) for line in traces_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        errors.append(f"{traces_path}: {type(exc).__name__}: {exc}")
        continue
    if len(traces) != 1:
        errors.append(f"{traces_path}: expected one trace, found {len(traces)}")
        continue
    trace = traces[0]
    if trace.get("status") != "completed" or trace.get("native_terminal_status") != "completed":
        errors.append(
            f"{traces_path}: status={trace.get('status')!r}, "
            f"native_terminal_status={trace.get('native_terminal_status')!r}"
        )
    if trace.get("frontend_protocol") != expected_protocol[agent]:
        errors.append(f"{traces_path}: wrong frontend_protocol={trace.get('frontend_protocol')!r}")
    if not trace.get("native_session_id"):
        errors.append(f"{traces_path}: native_session_id is missing")
    rounds = trace.get("logical_llm_rounds") or trace.get("llm_calls") or []
    attempts = trace.get("backend_attempts") or []
    tools = trace.get("tool_calls") or []
    if len(rounds) < 2:
        errors.append(
            f"{traces_path}: expected at least two native logical LLM rounds, "
            f"found {len(rounds)}"
        )
    if not attempts:
        errors.append(f"{traces_path}: no backend attempts")
    if attempts and any(
        item.get("http_status") != 200 or item.get("status") != "completed"
        for item in attempts
    ):
        errors.append(
            f"{traces_path}: non-200 backend attempt statuses="
            f"{[(item.get('http_status'), item.get('status')) for item in attempts]}"
        )
    if not any(item.get("status") == "completed" for item in tools):
        errors.append(f"{traces_path}: no completed native tool call")
    unresolved_tools = [
        item.get("tool_call_id")
        for item in tools
        if item.get("status") in {None, "observed", "queued", "running"}
    ]
    if unresolved_tools:
        errors.append(f"{traces_path}: unresolved native tools={unresolved_tools}")
    if tools and trace.get("tool_total_s") is None:
        errors.append(f"{traces_path}: tool_total_s is unknown despite observed tools")
    if tools and trace.get("tool_wait_total_s") is None:
        errors.append(f"{traces_path}: tool_wait_total_s is unknown despite observed tools")
    request_runs = agent_root / "_request_runs"
    raw_attempt_matches = (
        list(request_runs.rglob("backend_attempts.jsonl"))
        if request_runs.is_dir()
        else []
    )
    if len(raw_attempt_matches) != 1:
        errors.append(
            f"{request_runs}: expected one authoritative backend_attempts.jsonl, "
            f"found {len(raw_attempt_matches)}"
        )
    else:
        raw_attempt_path = raw_attempt_matches[0]
        try:
            raw_attempt_rows = [
                json.loads(line)
                for line in raw_attempt_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except Exception as exc:
            errors.append(
                f"{raw_attempt_path}: {type(exc).__name__}: {exc}"
            )
        else:
            raw_started = [
                str(item.get("backend_attempt_id"))
                for item in raw_attempt_rows
                if item.get("event") == "backend_attempt_started"
            ]
            raw_ended_rows = [
                item
                for item in raw_attempt_rows
                if item.get("event") == "backend_attempt_ended"
            ]
            raw_ended = [
                str(item.get("backend_attempt_id")) for item in raw_ended_rows
            ]
            projected = [
                str(item.get("backend_attempt_id") or item.get("attempt_id"))
                for item in attempts
            ]
            if not raw_started:
                errors.append(f"{raw_attempt_path}: no physical backend attempts")
            if Counter(raw_started) != Counter(raw_ended):
                errors.append(
                    f"{raw_attempt_path}: raw start/end IDs differ: "
                    f"started={raw_started!r} ended={raw_ended!r}"
                )
            if Counter(raw_started) != Counter(projected):
                errors.append(
                    f"{raw_attempt_path}: raw/projected IDs differ: "
                    f"raw={raw_started!r} projected={projected!r}"
                )
            if any(
                item.get("http_status") != 200 or item.get("error") is not None
                for item in raw_ended_rows
            ):
                errors.append(
                    f"{raw_attempt_path}: non-200/error physical attempts="
                    f"{[(item.get('backend_attempt_id'), item.get('http_status'), item.get('error')) for item in raw_ended_rows]}"
                )
    raw_matches = list(request_runs.rglob(expected_raw[agent])) if request_runs.is_dir() else []
    if len(raw_matches) != 1 or raw_matches[0].stat().st_size == 0:
        errors.append(
            f"{request_runs}: expected one non-empty {expected_raw[agent]}, "
            f"found {len(raw_matches)}"
        )

server_log = root / "vllm_server.log"
try:
    server_text = server_log.read_text(encoding="utf-8", errors="replace")
except Exception as exc:
    errors.append(f"{server_log}: {type(exc).__name__}: {exc}")
else:
    fatal_markers = (
        "Qwen Responses compatibility rejected the request",
        "JSONDecodeError",
        "Traceback (most recent call last)",
    )
    for marker in fatal_markers:
        if marker in server_text:
            errors.append(f"{server_log}: contains fatal marker {marker!r}")

if errors:
    print("Native GPU smoke validation failed:", file=sys.stderr)
    for error in errors:
        print(f"  {error}", file=sys.stderr)
    raise SystemExit(1)
print(f"Native {'/'.join(agents)} GPU smoke passed: {root}")
PY

echo "SMOKE_RUN_DIR=${RUN_DIR}"
