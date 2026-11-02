#!/usr/bin/env python3
"""Run OpenCode/Codex through the exact OpenHands workload harness.

The agent-specific code in this module is deliberately limited to pinning,
endpoint validation, provenance, and the per-request backend callback.  Fixed
RPS emission, MARS admission, progress/events/traces, and summary aggregation
all come from ``experiments/openhands/run_task_workload.py`` so the three agent
frontends have one experiment contract.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from event_normalizer import load_jsonl, summarize_dynamicity
from docker_proxy import cleanup_scoped_docker_project
from request_worker import _llm_retry_policy, _working_bwrap


OPENHANDS_DIR = Path(
    os.environ.get(
        "MARS_OPENHANDS_DIR",
        str(Path(__file__).resolve().parents[1] / "openhands"),
    )
).expanduser().resolve()
if str(OPENHANDS_DIR) not in sys.path:
    sys.path.insert(0, str(OPENHANDS_DIR))

import run_task_workload as openhands_workload  # noqa: E402
from security_policy import (  # noqa: E402
    blocked_security_task_decision,
    build_policy_rejected_result,
)


NATIVE_WORKER_ENTRY = Path(__file__).resolve().with_name("native_worker_entry.py")
AGENT_RUNTIME_BUNDLE_FILES = (
    "backend_proxy.py",
    "codex_app_server.py",
    "docker_proxy.py",
    "event_normalizer.py",
    "frontend_contract.py",
    "lifecycle_collector.py",
    "native_request_worker.py",
    "native_worker_entry.py",
    "opencode_server.py",
    "request_worker.py",
    "sandbox_paths.py",
    "trace_proxy.py",
)


EXTERNAL_ROOT = Path(
    os.environ.get(
        "MARS_AGENT_BASELINES_ROOT",
        str(Path.home() / "external" / "mars_agent_baselines"),
    )
).expanduser()


PINNED = {
    "opencode": {
        "version": "1.18.23",
        "commit": "ef2880f379129aa048be9e9353e30aa168d42c17",
        "binary": str(EXTERNAL_ROOT / "runtime/opencode/node_modules/.bin/opencode"),
        "source": str(EXTERNAL_ROOT / "opencode"),
    },
    "codex": {
        "version": "0.149.1",
        "commit": "ff29a44391deccde0aba0f8390337d7f3c319ea4",
        "binary": str(EXTERNAL_ROOT / "runtime/codex/node_modules/.bin/codex"),
        "source": str(EXTERNAL_ROOT / "codex"),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, choices=sorted(PINNED))
    parser.add_argument("--agent-bin", default=None)
    parser.add_argument("--agent-source-dir", default=None)
    parser.add_argument("--allow-unpinned-agent", action="store_true")
    parser.add_argument(
        "--allow-blocked-security-tasks",
        action="store_true",
        help=(
            "Allow security-sensitive TerminalBench tasks rejected by the shared-system "
            "profile. Use only in an approved isolated environment."
        ),
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--local-inputs-dir", default=None)
    parser.add_argument("--no-localize-task-inputs", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--rps", type=float, default=0.2)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-window", type=int, default=262144)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--llm-timeout-s", type=float, default=3000.0)
    parser.add_argument("--llm-num-retries", type=int, default=3)
    parser.add_argument(
        "--codex-enable-subagents",
        action="store_true",
        help="Enable Codex collaboration instructions and the built-in agents tool.",
    )
    parser.add_argument(
        "--gpu-timeout-s",
        "--request-timeout-s",
        dest="gpu_timeout_s",
        type=float,
        default=60000.0,
        help="OpenHands-compatible whole-request worker timeout.",
    )
    parser.add_argument("--terminal-no-change-timeout-s", type=int, default=30)
    parser.add_argument("--terminal-type", choices=("tmux", "subprocess"), default=None)
    parser.add_argument("--scheduling-policy", default="fcfs")
    parser.add_argument("--concur-window-init", type=int, default=1)
    parser.add_argument("--concur-max-active-agents", type=int, default=0)
    parser.add_argument("--concur-control-interval-s", type=float, default=0.2)
    parser.add_argument("--concur-additive-increase", type=int, default=2)
    parser.add_argument("--concur-multiplicative-decrease", type=float, default=0.5)
    parser.add_argument("--concur-util-low", type=float, default=0.2)
    parser.add_argument("--concur-util-high", type=float, default=0.5)
    parser.add_argument("--concur-hit-rate-threshold", type=float, default=0.2)
    parser.add_argument("--emit-mode", default="rps")
    parser.add_argument("--thread-pool", default="on")
    parser.add_argument("--cpu-gpu", default=None)
    parser.add_argument("--cpu-tool", default=None)
    parser.add_argument("--kv-events", action="store_true")
    parser.add_argument("--kv-events-endpoint", default=None)
    parser.add_argument("--autellix-tail-fcfs-after-finished", type=int, default=0)
    parser.add_argument("--mars-active-pool-size", type=int, default=8)
    parser.add_argument("--mars-long-prefill-tokens", type=int, default=50000)
    parser.add_argument("--mars-cpu-backlog-low", type=float, default=1.0)
    parser.add_argument("--mars-cpu-backlog-high", type=float, default=4.0)
    parser.add_argument("--mars-kv-target-ratio", type=float, default=0.9)
    parser.add_argument("--mars-kv-stat-interval-s", type=float, default=0.2)
    parser.add_argument("--mars-window-min", type=int, default=2)
    parser.add_argument("--mars-window-init", type=int, default=8)
    parser.add_argument("--mars-window-inc", type=int, default=1)
    parser.add_argument("--mars-window-dec-factor", type=float, default=0.7)
    parser.add_argument("--mars-control-interval-s", type=float, default=0.2)
    parser.add_argument("--mars-cpu-queue-wait-high-s", type=float, default=5.0)
    parser.add_argument("--mars-cpu-queue-wait-low-s", type=float, default=2.0)
    parser.add_argument("--mars-kv-max-stale-s", type=float, default=2.0)
    parser.add_argument("--mars-long-max-inflight", type=int, default=1)
    parser.add_argument("--mars-tail-max-inflight", type=int, default=4)
    parser.add_argument("--mars-tail-kv-budget-ratio", type=float, default=1.0)
    parser.add_argument("--mars-no-kv-active-limit", type=int, default=2)
    parser.add_argument("--prefill-tokenizer", default=None)
    parser.add_argument(
        "--cpu-affinity",
        default=None,
        help="Pin each request sandbox and all of its descendants with taskset -c.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Accepted for compatibility; interrupted formal attempts are never mixed/resumed.",
    )
    parser.add_argument("--rerun-failures", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--allow-unsandboxed-agent",
        action="store_true",
        help="Debug only: run an agent without outer bubblewrap PID/filesystem isolation.",
    )
    parser.add_argument(
        "--allow-unsandboxed-opencode",
        dest="allow_unsandboxed_agent",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        args.base_url = _normalize_openai_base_url(args.base_url)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _task_sha(task: dict[str, Any]) -> str:
    return _sha256_bytes(json.dumps(task, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def _version(binary: Path) -> str:
    env = os.environ.copy()
    node_bin = env.get("MARS_NODE_BIN", "").strip()
    if node_bin:
        node_dir = str(Path(node_bin).expanduser().absolute().parent)
        path_entries = env.get("PATH", "").split(os.pathsep)
        if node_dir not in path_entries:
            env["PATH"] = os.pathsep.join([node_dir, *filter(None, path_entries)])
    result = subprocess.run(
        [str(binary), "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    text = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(f"failed to query {binary} version: {text}")
    return text


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _normalize_openai_base_url(base_url: str) -> str:
    """Return one canonical OpenAI-compatible base URL ending in ``/v1``.

    The endpoint preflight, OpenHands harness, and per-request trace proxy must
    receive the exact same base path.  In particular, accepting a bare
    ``http://host:port`` only in preflight would make OpenCode later request
    ``/chat/completions`` instead of ``/v1/chat/completions``.
    """

    value = str(base_url).strip()
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            "--base-url must be an absolute http(s) URL, for example "
            "http://127.0.0.1:8000/v1"
        )
    if parsed.query or parsed.fragment:
        raise ValueError("--base-url must not contain a query string or fragment")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path, "", "")
    )


def _endpoint(path: str, suffix: str) -> str:
    return _normalize_openai_base_url(path) + suffix


def _preflight_endpoint(base_url: str, model: str, agent: str, api_key: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    request = urllib.request.Request(_endpoint(base_url, "/models"), headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read())
    model_ids = [
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id") is not None
    ]
    if model not in model_ids:
        raise RuntimeError(f"model {model!r} is not served by {base_url}; available={model_ids}")

    def probe_route(suffix: str) -> int:
        probe = urllib.request.Request(
            _endpoint(base_url, suffix),
            data=b"{}",
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(probe, timeout=10) as response:
                return int(response.status)
        except urllib.error.HTTPError as exc:
            return int(exc.code)

    chat_status = probe_route("/chat/completions")
    if chat_status in {404, 405} or chat_status >= 500:
        raise RuntimeError(
            f"agent adapter requires Chat Completions, but "
            f"{_endpoint(base_url, '/chat/completions')} returned {chat_status}"
        )
    responses_status = probe_route("/responses") if agent == "codex" else None
    if agent == "codex" and (
        responses_status in {404, 405}
        or (responses_status is not None and responses_status >= 500)
    ):
        raise RuntimeError(
            "native Codex requires the Responses API, but "
            f"{_endpoint(base_url, '/responses')} returned {responses_status}"
        )
    return {
        "model_ids": model_ids,
        "chat_completions_probe_status": chat_status,
        "native_responses_probe_status": responses_status,
        "codex_native_responses": agent == "codex",
        "qwen_responses_provider": "server_side_strict_chat_tool_parser" if agent == "codex" else None,
    }


def _adapter_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = [
        "backend_proxy.py",
        "codex_app_server.py",
        "docker_proxy.py",
        "event_normalizer.py",
        "frontend_contract.py",
        "lifecycle_collector.py",
        "native_worker_entry.py",
        "native_request_worker.py",
        "opencode_server.py",
        "request_worker.py",
        "run_workload.py",
        "sandbox_paths.py",
        "trace_proxy.py",
        "upstream.env",
    ]
    hashes = {name: _sha256_path(root / name) for name in names}
    for name in (
        "command_guard.py",
        "common.py",
        "llm_trace_bridge.py",
        "openhands_runner.py",
        "qwen_responses_compat.py",
        "request_bootstrap.py",
        "request_worker.py",
        "run_task_workload.py",
        "sandbox_paths.py",
        "sandbox_shell.sh",
        "tool_trace_bridge.py",
        "trace_writer.py",
        "vllm_api_server_mars.py",
    ):
        path = OPENHANDS_DIR / name
        hashes[f"openhands/{name}"] = _sha256_path(path)
    return hashes


def _prepare_agent_runtime_bundle(out_dir: Path) -> tuple[Path, dict[str, str]]:
    source_dir = Path(__file__).resolve().parent
    bundle_dir = (out_dir / "_agent_runtime_bundle").resolve()
    bundle_dir.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for name in AGENT_RUNTIME_BUNDLE_FILES:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing native adapter runtime file: {source}")
        destination = bundle_dir / name
        shutil.copy2(source, destination)
        hashes[name] = _sha256_path(destination)
    return bundle_dir, hashes


def _validate_infrastructure_artifacts(
    traces: list[dict[str, Any]],
    *,
    out_dir: Path,
    require_sandbox: bool,
) -> list[str]:
    """Reject incomplete transport/provenance without scoring agent quality."""

    errors: list[str] = []
    request_dirs: dict[str, Path] = {}
    request_runs_root = out_dir / "_request_runs"
    if request_runs_root.is_dir():
        for result_path in request_runs_root.rglob("result.json"):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{result_path}: unreadable result: {type(exc).__name__}: {exc}")
                continue
            trace = result.get("trace") if isinstance(result, dict) else None
            request_id = trace.get("request_id") if isinstance(trace, dict) else None
            if not request_id:
                errors.append(f"{result_path}: result has no trace request_id")
                continue
            request_id = str(request_id)
            if request_id in request_dirs:
                errors.append(f"duplicate per-request result for {request_id}")
            request_dirs[request_id] = result_path.parent

    infrastructure_statuses = {
        "worker_error",
        "worker_timeout",
        "worker_future_exception",
    }
    for trace in traces:
        request_id = str(trace.get("request_id") or "<missing-request-id>")
        status = str(trace.get("status") or "unknown")
        if status == "policy_rejected":
            request_dir = request_dirs.get(request_id)
            if request_dir is None:
                errors.append(f"{request_id}: policy rejection result directory is missing")
                continue
            policy_path = request_dir / "policy_decision.json"
            try:
                policy = json.loads(policy_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(
                    f"{request_id}: policy decision artifact is invalid: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if (
                policy.get("decision") != "reject"
                or policy.get("task_id") != trace.get("task_id")
            ):
                errors.append(
                    f"{request_id}: policy decision does not match trace: {policy!r}"
                )
            if trace.get("workspace_materialized") is not False:
                errors.append(
                    f"{request_id}: rejected request reports a materialized workspace"
                )
            continue
        if status in infrastructure_statuses:
            errors.append(f"{request_id}: infrastructure status={status}")
        if not trace.get("native_session_id"):
            errors.append(f"{request_id}: native_session_id is missing")
        if require_sandbox and trace.get("outer_sandbox") != "bubblewrap_pid_filesystem":
            errors.append(
                f"{request_id}: outer_sandbox={trace.get('outer_sandbox')!r}"
            )

        request_dir = request_dirs.get(request_id)
        if request_dir is None:
            errors.append(f"{request_id}: per-request result directory is missing")
            continue
        raw_path = request_dir / "backend_attempts.jsonl"
        if not raw_path.is_file():
            errors.append(f"{request_id}: {raw_path} is missing")
            continue
        if str(trace.get("benchmark") or "").strip().lower() == "terminalbench":
            cleanup_path = request_dir / "docker_cleanup.json"
            try:
                cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(
                    f"{request_id}: Docker cleanup artifact is invalid: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                cleanup_valid = (
                    cleanup.get("status") == "completed"
                    and cleanup.get("returncode") == 0
                ) or (
                    cleanup.get("status") == "not_started"
                    and cleanup.get("returncode") is None
                )
                if not cleanup_valid:
                    errors.append(
                        f"{request_id}: scoped Docker cleanup failed: {cleanup!r}"
                    )
        raw_rows = load_jsonl(raw_path)
        started = [
            str(row.get("backend_attempt_id"))
            for row in raw_rows
            if row.get("event") == "backend_attempt_started"
        ]
        ended = [
            str(row.get("backend_attempt_id"))
            for row in raw_rows
            if row.get("event") == "backend_attempt_ended"
        ]
        raw_ended_by_id = {
            str(row.get("backend_attempt_id")): row
            for row in raw_rows
            if row.get("event") == "backend_attempt_ended"
        }
        if collections.Counter(started) != collections.Counter(ended):
            errors.append(
                f"{request_id}: raw backend start/end IDs differ: "
                f"started={started!r} ended={ended!r}"
            )
        projected = [
            str(item.get("backend_attempt_id") or item.get("attempt_id"))
            for item in (trace.get("backend_attempts") or [])
        ]
        if collections.Counter(started) != collections.Counter(projected):
            errors.append(
                f"{request_id}: raw/projected backend IDs differ: "
                f"raw={started!r} projected={projected!r}"
            )
        for attempt in trace.get("backend_attempts") or []:
            attempt_id = str(
                attempt.get("backend_attempt_id") or attempt.get("attempt_id")
            )
            raw_end = raw_ended_by_id.get(attempt_id)
            if attempt.get("end_time_s") is None or attempt.get("status") not in {
                "completed",
                "error",
            }:
                errors.append(
                    f"{request_id}: projected backend attempt is not terminal: "
                    f"{attempt_id} status={attempt.get('status')!r}"
                )
            if raw_end is not None and attempt.get("http_status") != raw_end.get(
                "http_status"
            ):
                errors.append(
                    f"{request_id}: backend HTTP status mismatch for {attempt_id}: "
                    f"projected={attempt.get('http_status')!r} "
                    f"raw={raw_end.get('http_status')!r}"
                )
        if status == "completed":
            rounds = trace.get("logical_llm_rounds") or trace.get("llm_calls") or []
            if not rounds:
                errors.append(f"{request_id}: completed trace has no logical LLM rounds")
            for round_item in rounds:
                if (
                    round_item.get("end_time_s") is None
                    or round_item.get("status") != "completed"
                ):
                    errors.append(
                        f"{request_id}: completed trace has nonterminal LLM round: "
                        f"{round_item.get('round_id')!r} "
                        f"status={round_item.get('status')!r}"
                    )
            if not started:
                errors.append(f"{request_id}: completed trace has no backend attempts")
            unresolved_tools = [
                item.get("tool_call_id")
                for item in (trace.get("tool_calls") or [])
                if item.get("end_time_s") is None
                or item.get("status") in {None, "observed", "queued", "running"}
            ]
            if unresolved_tools:
                errors.append(
                    f"{request_id}: completed trace has unresolved tools="
                    f"{unresolved_tools!r}"
                )
    return errors


_BACKEND_CONFIG: dict[str, Any] = {}


def _agent_request_backend(
    *,
    args: argparse.Namespace,
    task: dict[str, Any],
    request_id: str,
    arrival_time_s: float,
    workspace_root: Path,
    request_run_dir: Path,
    runtime_bundle_dir: Path,
) -> dict[str, Any]:
    """Run one native frontend request behind the OpenHands worker boundary.

    The shared OpenHands harness owns emission, admission and aggregation.  A
    fresh Python subprocess owns the complete native-agent lifecycle so a
    request timeout can terminate the worker and every descendant it spawned.
    """
    config = _BACKEND_CONFIG
    policy_decision = blocked_security_task_decision(
        task,
        allow_blocked_security_tasks=bool(
            getattr(args, "allow_blocked_security_tasks", False)
        ),
    )
    if policy_decision is not None:
        return build_policy_rejected_result(
            task=task,
            decision=policy_decision,
            request_id=request_id,
            arrival_time_s=arrival_time_s,
            workspace_root=workspace_root,
            request_run_dir=request_run_dir,
            agent=str(config["agent"]),
            model=args.model,
            task_sha256=_task_sha(task),
        )

    request_run_dir.mkdir(parents=True, exist_ok=True)
    task_json = request_run_dir / "task.json"
    config_json = request_run_dir / "native_worker_config.json"
    result_json = request_run_dir / "result.json"
    stdout_path = request_run_dir / "stdout.txt"
    stderr_path = request_run_dir / "stderr.txt"

    _atomic_json(task_json, task)
    _atomic_json(
        config_json,
        {
            "agent": str(config["agent"]),
            "agent_bin": str(config["agent_bin"]),
            "task_sha256": _task_sha(task),
            "request_id": request_id,
            # Match OpenHands and the vLLM scheduler metadata exactly.
            "mars_job_id": request_id,
            "arrival_time_s": float(arrival_time_s),
            "workspace_root": str(workspace_root),
            "local_inputs_dir": str(config["local_inputs_dir"]),
            "request_run_dir": str(request_run_dir),
            "model": args.model,
            "base_url": args.base_url,
            "context_window": int(config["context_window"]),
            "max_output_tokens": int(args.max_output_tokens),
            "max_iterations": int(args.max_iterations),
            "temperature": float(args.temperature),
            "top_p": args.top_p,
            "timeout_s": float(args.gpu_timeout_s),
            "llm_timeout_s": float(args.llm_timeout_s),
            "llm_num_retries": int(args.llm_num_retries),
            "codex_enable_subagents": bool(
                getattr(args, "codex_enable_subagents", False)
            ),
            "terminal_no_change_timeout_s": args.terminal_no_change_timeout_s,
            "admission_gate_path": (
                str(
                    Path(args.out_dir).expanduser().resolve()
                    / "_concur_gates"
                    / f"{request_id}.active"
                )
                if str(getattr(args, "scheduling_policy", "") or "").lower()
                in {"continuum_dy_con", "continuum_con", "fcfs_con"}
                else None
            ),
            "require_bwrap": bool(config["require_bwrap"]),
            "allow_blocked_security_tasks": bool(
                getattr(args, "allow_blocked_security_tasks", False)
            ),
            "cpu_affinity": args.cpu_affinity,
            "benchmark_root": config.get("benchmark_root"),
        },
    )

    adapter_bundle_raw = config.get("adapter_bundle_dir")
    if adapter_bundle_raw:
        adapter_bundle_dir = Path(str(adapter_bundle_raw)).resolve()
        worker_entry = adapter_bundle_dir / "native_worker_entry.py"
    else:
        worker_entry = NATIVE_WORKER_ENTRY.resolve()
        adapter_bundle_dir = worker_entry.parent
    expected_bundle_hashes = config.get("adapter_bundle_sha256") or {}
    for name, expected_hash in expected_bundle_hashes.items():
        path = adapter_bundle_dir / str(name)
        if not path.is_file() or _sha256_path(path) != str(expected_hash):
            raise RuntimeError(f"Frozen native adapter bundle changed during run: {path}")
    for name, expected_hash in (config.get("openhands_bundle_sha256") or {}).items():
        path = runtime_bundle_dir / str(name)
        if not path.is_file() or _sha256_path(path) != str(expected_hash):
            raise RuntimeError(f"Frozen OpenHands runtime bundle changed during run: {path}")
    if not worker_entry.is_file():
        raise FileNotFoundError(f"native agent worker entry not found: {worker_entry}")
    cmd = [
        sys.executable,
        str(worker_entry),
        "--task-json",
        str(task_json),
        "--config-json",
        str(config_json),
        "--result-json",
        str(result_json),
    ]
    env = os.environ.copy()
    # Keep credentials out of argv and the persistent per-request config.
    env["MARS_NATIVE_AGENT_API_KEY"] = str(args.api_key)
    module_paths = [str(adapter_bundle_dir), str(runtime_bundle_dir)]
    if env.get("PYTHONPATH"):
        module_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(module_paths)
    env["MARS_OPENHANDS_DIR"] = str(runtime_bundle_dir)

    def _failed_result(
        *,
        status: str,
        error_type: str,
        message: str,
        stdout_text: str,
        stderr_text: str,
        returncode: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        end_time_s = openhands_workload.now_s()
        error_payload: dict[str, Any] = {
            "type": error_type,
            "message": message,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout_preview": (stdout_text or "")[:2000],
            "stderr_preview": (stderr_text or "")[:2000],
        }
        event_payload: dict[str, Any] = {
            "ts": end_time_s,
            "event": status,
            "request_id": request_id,
            "dataset_request_id": task.get("request_id"),
            "benchmark": task.get("benchmark"),
            "task_id": task.get("task_id"),
        }
        if returncode is not None:
            error_payload["returncode"] = int(returncode)
            event_payload["returncode"] = int(returncode)
        if timeout_s is not None:
            error_payload["timeout_s"] = float(timeout_s)
            event_payload["timeout_s"] = float(timeout_s)
        return {
            "trace": {
                "request_id": request_id,
                "dataset_request_id": task.get("request_id"),
                "benchmark": task.get("benchmark"),
                "task_id": task.get("task_id"),
                "workspace_dir": str((workspace_root / request_id).resolve()),
                "workspace_file_count": 0,
                "arrival_time_s": arrival_time_s,
                "start_time_s": arrival_time_s,
                "end_time_s": end_time_s,
                "queue_wait_s": None,
                "e2e_latency_s": float(max(0.0, end_time_s - arrival_time_s)),
                "gpu_total_s": None,
                "tool_total_s": None,
                "tool_wait_total_s": None,
                "wait_total_s": None,
                "status": status,
                "conversation_status": None,
                "error": error_payload,
                "llm_calls": [],
                "tool_calls": [],
                "rounds": [],
                "orphan_tools": [],
                "agent_messages": [],
                "agent_errors": [],
                "conversation_errors": [],
            },
            "events": [event_payload],
        }

    timeout_s = float(args.gpu_timeout_s)
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(request_run_dir),
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout_text, stderr_text = proc.communicate(
            timeout=(timeout_s if timeout_s > 0 else None)
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        # This is the same process-group teardown used by OpenHands requests.
        openhands_workload._terminate_worker_process(proc)
        stdout_text, stderr_text = proc.communicate()

    stdout_path.write_text(stdout_text or "", encoding="utf-8")
    stderr_path.write_text(stderr_text or "", encoding="utf-8")

    parent_cleanup_error: str | None = None
    if str(task.get("benchmark") or "").strip().lower() == "terminalbench":
        cleanup_descriptor_path = request_run_dir / "docker_cleanup_descriptor.json"
        try:
            if cleanup_descriptor_path.is_file():
                cleanup_scoped_docker_project(
                    workspace_dir=(workspace_root / request_id).resolve(),
                    runtime_root=request_run_dir / "parent_docker_cleanup_runtime",
                    cleanup_descriptor_path=cleanup_descriptor_path,
                )
            else:
                _atomic_json(
                    request_run_dir / "docker_cleanup.json",
                    {"status": "not_started", "returncode": None},
                )
        except Exception as exc:
            parent_cleanup_error = f"{type(exc).__name__}: {exc}"
            _atomic_json(
                request_run_dir / "parent_docker_cleanup_error.json",
                {
                    "schema": "mars_parent_docker_cleanup_error_v1",
                    "error": parent_cleanup_error,
                },
            )

    if timed_out:
        return _failed_result(
            status="worker_timeout",
            error_type="WorkerProcessTimeout",
            message=f"worker exceeded timeout after {timeout_s:.1f}s",
            stdout_text=stdout_text or "",
            stderr_text=stderr_text or "",
            returncode=proc.returncode,
            timeout_s=timeout_s,
        )
    if proc.returncode != 0:
        return _failed_result(
            status="worker_error",
            error_type="WorkerProcessError",
            message=f"worker exited with code {proc.returncode}",
            stdout_text=stdout_text or "",
            stderr_text=stderr_text or "",
            returncode=proc.returncode,
        )
    if parent_cleanup_error is not None:
        return _failed_result(
            status="worker_error",
            error_type="DockerCleanupError",
            message=parent_cleanup_error,
            stdout_text=stdout_text or "",
            stderr_text=stderr_text or "",
            returncode=proc.returncode,
        )

    try:
        result = json.loads(result_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _failed_result(
            status="worker_error",
            error_type="WorkerResultError",
            message=f"failed to read worker result: {type(exc).__name__}: {exc}",
            stdout_text=stdout_text or "",
            stderr_text=stderr_text or "",
            returncode=proc.returncode,
        )
    if not isinstance(result, dict) or not isinstance(result.get("trace"), dict):
        return _failed_result(
            status="worker_error",
            error_type="WorkerResultError",
            message="invalid worker result: expected an object with a trace object",
            stdout_text=stdout_text or "",
            stderr_text=stderr_text or "",
            returncode=proc.returncode,
        )
    return result


def _openhands_argv(args: argparse.Namespace, *, workspace_root: Path) -> list[str]:
    local_inputs_dir = (
        Path(args.local_inputs_dir).expanduser().resolve()
        if args.local_inputs_dir
        else (Path(args.out_dir).expanduser().resolve() / "_localized_inputs").resolve()
    )
    argv = [
        str(OPENHANDS_DIR / "run_task_workload.py"),
        "--tasks", args.tasks,
        "--out-dir", args.out_dir,
        "--workspace-root", str(workspace_root),
        "--local-inputs-dir", str(local_inputs_dir),
        "--model", args.model,
        "--base-url", args.base_url,
        "--api-key", args.api_key,
        "--openhands-python", sys.executable,
        "--rps", str(args.rps),
        "--max-workers", str(args.max_workers),
        "--seed", str(args.seed),
        "--temperature", str(args.temperature),
        "--max-output-tokens", str(args.max_output_tokens),
        "--llm-timeout-s", str(int(args.llm_timeout_s)),
        "--llm-num-retries", str(args.llm_num_retries),
        "--max-iterations", str(args.max_iterations),
        "--terminal-no-change-timeout-s", str(args.terminal_no_change_timeout_s),
        "--cpu-affinity", str(args.cpu_affinity or ""),
        "--emit-mode", args.emit_mode,
        "--gpu-timeout-s", str(args.gpu_timeout_s),
        "--scheduling-policy", args.scheduling_policy,
        "--concur-window-init", str(args.concur_window_init),
        "--concur-max-active-agents", str(args.concur_max_active_agents),
        "--concur-control-interval-s", str(args.concur_control_interval_s),
        "--concur-additive-increase", str(args.concur_additive_increase),
        "--concur-multiplicative-decrease", str(args.concur_multiplicative_decrease),
        "--concur-util-low", str(args.concur_util_low),
        "--concur-util-high", str(args.concur_util_high),
        "--concur-hit-rate-threshold", str(args.concur_hit_rate_threshold),
        "--thread-pool", args.thread_pool,
        "--cpu-gpu", str(args.cpu_gpu or ""),
        "--cpu-tool", str(args.cpu_tool or args.cpu_affinity or ""),
        "--autellix-tail-fcfs-after-finished", str(args.autellix_tail_fcfs_after_finished),
        "--mars-active-pool-size", str(args.mars_active_pool_size),
        "--mars-long-prefill-tokens", str(args.mars_long_prefill_tokens),
        "--mars-cpu-backlog-low", str(args.mars_cpu_backlog_low),
        "--mars-cpu-backlog-high", str(args.mars_cpu_backlog_high),
        "--mars-kv-target-ratio", str(args.mars_kv_target_ratio),
        "--mars-kv-stat-interval-s", str(args.mars_kv_stat_interval_s),
        "--mars-window-min", str(args.mars_window_min),
        "--mars-window-init", str(args.mars_window_init),
        "--mars-window-inc", str(args.mars_window_inc),
        "--mars-window-dec-factor", str(args.mars_window_dec_factor),
        "--mars-control-interval-s", str(args.mars_control_interval_s),
        "--mars-cpu-queue-wait-high-s", str(args.mars_cpu_queue_wait_high_s),
        "--mars-cpu-queue-wait-low-s", str(args.mars_cpu_queue_wait_low_s),
        "--mars-kv-max-stale-s", str(args.mars_kv_max_stale_s),
        "--mars-long-max-inflight", str(args.mars_long_max_inflight),
        "--mars-tail-max-inflight", str(args.mars_tail_max_inflight),
        "--mars-tail-kv-budget-ratio", str(args.mars_tail_kv_budget_ratio),
        "--mars-no-kv-active-limit", str(args.mars_no_kv_active_limit),
    ]
    if args.codex_enable_subagents:
        argv.append("--codex-enable-subagents")
    if args.top_p is not None:
        argv += ["--top-p", str(args.top_p)]
    if args.allow_blocked_security_tasks:
        argv.append("--allow-blocked-security-tasks")
    if args.max_requests is not None:
        argv += ["--max-requests", str(args.max_requests)]
    if args.shuffle:
        argv.append("--shuffle")
    if args.no_localize_task_inputs:
        argv.append("--no-localize-task-inputs")
    if args.terminal_type:
        argv += ["--terminal-type", args.terminal_type]
    if args.prefill_tokenizer:
        argv += ["--prefill-tokenizer", args.prefill_tokenizer]
    if args.kv_events:
        argv.append("--kv-events")
        if args.kv_events_endpoint:
            argv += ["--kv-events-endpoint", args.kv_events_endpoint]
    return argv


def main() -> None:
    args = _parse_args()
    if args.rps <= 0:
        raise SystemExit("--rps must be positive")
    if args.max_workers < 0:
        raise SystemExit("--max-workers must be >= 0")
    if args.resume or args.rerun_failures:
        raise SystemExit(
            "OpenHands-aligned formal attempts are immutable and cannot mix results across "
            "server lifetimes; choose a new TIMESTAMP/SWEEP_ID after an interruption."
        )

    bwrap_available = _working_bwrap()
    if not args.allow_unsandboxed_agent and not bwrap_available:
        raise SystemExit(
            "working bubblewrap is required (set MARS_BWRAP_BIN when it is not on PATH); "
            "use --allow-unsandboxed-agent only for debugging"
        )
    pin = PINNED[args.agent]
    agent_bin = Path(args.agent_bin or pin["binary"]).expanduser().absolute()
    source_dir = Path(args.agent_source_dir or pin["source"]).expanduser().resolve()
    tasks_path = Path(args.tasks).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    workspace_root = (
        Path(args.workspace_root).expanduser().resolve()
        if args.workspace_root
        else (out_dir / "workspaces").resolve()
    )
    local_inputs_dir = (
        Path(args.local_inputs_dir).expanduser().resolve()
        if args.local_inputs_dir
        else (out_dir / "_localized_inputs").resolve()
    )
    benchmark_root_raw = os.environ.get("MARS_BENCHMARK_ROOT")
    benchmark_root = (
        Path(benchmark_root_raw).expanduser().resolve()
        if benchmark_root_raw
        else None
    )
    if not OPENHANDS_DIR.joinpath("run_task_workload.py").is_file():
        raise SystemExit(f"OpenHands workload harness not found: {OPENHANDS_DIR}")
    if not agent_bin.is_file() or not os.access(agent_bin, os.X_OK):
        raise SystemExit(f"agent binary is missing or not executable: {agent_bin}")
    if not tasks_path.is_file():
        raise SystemExit(f"tasks file not found: {tasks_path}")
    for artifact in ("run_config.json", "events.jsonl", "traces.jsonl", "summary.json"):
        if (out_dir / artifact).exists():
            raise SystemExit(f"refusing to overwrite formal run artifact: {out_dir / artifact}")

    version = _version(agent_bin)
    source_commit = _git_commit(source_dir)
    if not args.allow_unpinned_agent:
        if pin["version"] not in version:
            raise SystemExit(f"expected {args.agent} {pin['version']}, got {version!r}")
        if source_commit != pin["commit"]:
            raise SystemExit(f"expected {args.agent} source commit {pin['commit']}, got {source_commit}")

    endpoint_probe = _preflight_endpoint(args.base_url, args.model, args.agent, args.api_key)
    requested_count = sum(1 for line in tasks_path.open("r", encoding="utf-8") if line.strip())
    if args.max_requests is not None:
        requested_count = min(requested_count, args.max_requests)
    metadata = {
        "schema": "mars_native_agent_openhands_contract_v3",
        "agent": args.agent,
        "agent_version": version,
        "agent_commit": source_commit,
        "agent_bin": str(agent_bin),
        "agent_source_dir": str(source_dir),
        "context_window": args.context_window,
        "top_p": args.top_p,
        "llm_num_retries": (
            args.llm_num_retries if args.agent == "codex" else None
        ),
        "requested_llm_num_retries": args.llm_num_retries,
        "llm_retry_policy": _llm_retry_policy(args.agent, args.llm_num_retries),
        "codex_enable_subagents": (
            bool(args.codex_enable_subagents) if args.agent == "codex" else None
        ),
        "codex_namespace_tool_compat": (
            "flatten_restore_v1"
            if args.agent == "codex" and args.codex_enable_subagents
            else None
        ),
        "outer_sandbox": (
            "bubblewrap_pid_filesystem" if bwrap_available else "disabled_debug_only"
        ),
        "bubblewrap_bin": os.environ.get("MARS_BWRAP_BIN"),
        "node_bin": os.environ.get("MARS_NODE_BIN"),
        "frontend_protocol": (
            "codex_app_server_v2" if args.agent == "codex" else "opencode_legacy_server_sse"
        ),
        "backend_protocol": "responses" if args.agent == "codex" else "chat_completions",
        "backend_model_adapter": (
            "qwen_responses_server_side" if args.agent == "codex" else "native_chat"
        ),
        "protocol_translation_location": (
            "request_scoped_namespace_compat_proxy"
            if args.agent == "codex" and args.codex_enable_subagents
            else "inference_server"
            if args.agent == "codex"
            else "none"
        ),
        "native_terminal_status": True,
        "benchmark_root": str(benchmark_root) if benchmark_root else None,
        "endpoint_probe": endpoint_probe,
        "requested_count": requested_count,
        "openhands_harness": str(OPENHANDS_DIR / "run_task_workload.py"),
        "openhands_contract": {
            "emission": "shared_fixed_rps",
            "mars_admission": "shared_external_window_kv_cpu",
            "concur_admission": "shared_external_kv_aimd_generation_gate",
            "trace_summary": "shared_all_terminal_traces",
        },
    }
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    local_inputs_dir.mkdir(parents=True, exist_ok=True)
    adapter_bundle_dir, adapter_bundle_hashes = _prepare_agent_runtime_bundle(
        out_dir
    )
    adapter_hashes = _adapter_hashes()
    openhands_bundle_hashes = {
        key.removeprefix("openhands/"): value
        for key, value in adapter_hashes.items()
        if key.startswith("openhands/")
        and key.removeprefix("openhands/")
        in openhands_workload.LOCAL_RUNTIME_BUNDLE_FILES
    }
    metadata["agent_runtime_bundle"] = str(adapter_bundle_dir)
    _atomic_json(
        out_dir / "provenance.json",
        {
            **metadata,
            "expected_version": pin["version"],
            "expected_commit": pin["commit"],
            "binary_sha256": _sha256_path(agent_bin),
            "adapter_sha256": adapter_hashes,
            "agent_runtime_bundle_sha256": adapter_bundle_hashes,
            "tasks": str(tasks_path),
            "tasks_sha256": _sha256_path(tasks_path),
        },
    )

    _BACKEND_CONFIG.clear()
    _BACKEND_CONFIG.update(
        {
            "agent": args.agent,
            "agent_bin": str(agent_bin),
            "context_window": args.context_window,
            "local_inputs_dir": str(local_inputs_dir),
            "benchmark_root": str(benchmark_root) if benchmark_root else None,
            "require_bwrap": not args.allow_unsandboxed_agent,
            "adapter_bundle_dir": str(adapter_bundle_dir),
            "adapter_bundle_sha256": adapter_bundle_hashes,
            "openhands_bundle_sha256": openhands_bundle_hashes,
        }
    )

    original_backend = openhands_workload._run_request_subprocess
    original_atomic_write_json = openhands_workload.atomic_write_json
    original_argv = sys.argv[:]

    def _write_with_agent_metadata(path: Path, payload: Any) -> None:
        if Path(path).name == "run_config.json" and isinstance(payload, dict):
            payload = dict(payload)
            config = dict(payload.get("config") or {})
            config["wait_semantics"] = (
                "request queue wait plus native tool enqueue-to-start; unavailable tool wait is null"
            )
            config["top_p"] = args.top_p
            config["requested_llm_num_retries"] = args.llm_num_retries
            config["llm_num_retries"] = (
                args.llm_num_retries if args.agent == "codex" else None
            )
            config["llm_retry_policy"] = _llm_retry_policy(
                args.agent, args.llm_num_retries
            )
            config["agent_cli"] = metadata
            payload["config"] = config
        original_atomic_write_json(path, payload)

    openhands_workload._run_request_subprocess = _agent_request_backend
    openhands_workload.atomic_write_json = _write_with_agent_metadata
    sys.argv = _openhands_argv(args, workspace_root=workspace_root)
    try:
        openhands_workload.main()
    finally:
        sys.argv = original_argv
        openhands_workload._run_request_subprocess = original_backend
        openhands_workload.atomic_write_json = original_atomic_write_json

    traces = load_jsonl(out_dir / "traces.jsonl")
    infrastructure_errors = _validate_infrastructure_artifacts(
        traces,
        out_dir=out_dir,
        require_sandbox=not args.allow_unsandboxed_agent,
    )
    _atomic_json(
        out_dir / "infrastructure_validation.json",
        {
            "schema": "mars_agent_cli_infrastructure_validation_v1",
            "valid": not infrastructure_errors,
            "error_count": len(infrastructure_errors),
            "errors": infrastructure_errors,
        },
    )
    _atomic_json(
        out_dir / "dynamicity_summary.json",
        {
            "schema": "mars_agent_cli_dynamicity_artifact_v1",
            "agent": args.agent,
            "count": len(traces),
            "dynamicity": summarize_dynamicity(traces),
        },
    )
    if infrastructure_errors:
        print("Agent CLI infrastructure validation failed:", file=sys.stderr)
        for error in infrastructure_errors:
            print(f"  {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
