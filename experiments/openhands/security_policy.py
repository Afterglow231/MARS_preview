#!/usr/bin/env python3
"""Fail-closed admission policy for security-sensitive benchmark requests."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping


POLICY_ID = "shared_terminalbench_security_deny_v1"
BLOCKED_TERMINALBENCH_TASK_IDS = frozenset(
    {
        "spring-messaging-vul",
        "sql-injection-attack",
    }
)


def blocked_security_task_decision(
    task: Mapping[str, Any],
    *,
    allow_blocked_security_tasks: bool = False,
) -> dict[str, Any] | None:
    """Return a rejection decision for a blocked task, otherwise ``None``."""

    if allow_blocked_security_tasks:
        return None
    benchmark = str(task.get("benchmark") or "").strip().lower()
    payload = task.get("payload") if isinstance(task.get("payload"), Mapping) else {}
    task_id = str(task.get("task_id") or payload.get("task_id") or "").strip().lower()
    if benchmark != "terminalbench" or task_id not in BLOCKED_TERMINALBENCH_TASK_IDS:
        return None
    return {
        "schema": "mars_security_policy_decision_v1",
        "policy_id": POLICY_ID,
        "decision": "reject",
        "benchmark": benchmark,
        "task_id": task_id,
        "reason": (
            "Security-sensitive TerminalBench task is disabled on the shared-system "
            "execution profile. Use --allow-blocked-security-tasks only in an "
            "approved isolated environment."
        ),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".policy.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _task_sha256(task: Mapping[str, Any]) -> str:
    raw = json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_policy_rejected_result(
    *,
    task: Mapping[str, Any],
    decision: Mapping[str, Any],
    request_id: str,
    arrival_time_s: float,
    workspace_root: Path,
    request_run_dir: Path,
    agent: str,
    model: str,
    task_sha256: str | None = None,
) -> dict[str, Any]:
    """Persist and return a terminal result without starting an agent or workspace."""

    started = time.time()
    ended = time.time()
    queue_wait_s = max(0.0, started - float(arrival_time_s))
    workspace_dir = (workspace_root / request_id).resolve()
    request_run_dir.mkdir(parents=True, exist_ok=True)

    decision_payload = {
        **dict(decision),
        "request_id": request_id,
        "dataset_request_id": task.get("request_id"),
        "timestamp_s": started,
        "override_flag": "--allow-blocked-security-tasks",
    }
    _write_json_atomic(request_run_dir / "task.json", task)
    _write_json_atomic(request_run_dir / "policy_decision.json", decision_payload)
    (request_run_dir / "stdout.txt").write_text("", encoding="utf-8")
    (request_run_dir / "stderr.txt").write_text("", encoding="utf-8")

    trace = {
        "schema": "mars_security_policy_trace_v1",
        "agent": agent,
        "request_id": request_id,
        "mars_job_id": request_id,
        "dataset_request_id": task.get("request_id"),
        "benchmark": task.get("benchmark"),
        "task_id": task.get("task_id"),
        "workspace_dir": str(workspace_dir),
        "workspace_file_count": 0,
        "workspace_materialized": False,
        "outer_sandbox": "not_started_policy_rejected",
        "docker_access": "not_started_policy_rejected",
        "sandbox_readonly_paths": [],
        "native_session_id": None,
        "arrival_time_s": float(arrival_time_s),
        "start_time_s": started,
        "end_time_s": ended,
        "queue_wait_s": queue_wait_s,
        "e2e_latency_s": max(0.0, ended - float(arrival_time_s)),
        "gpu_total_s": 0.0,
        "tool_total_s": 0.0,
        "tool_wait_total_s": 0.0,
        "wait_total_s": queue_wait_s,
        "status": "policy_rejected",
        "conversation_status": "policy_rejected",
        "native_terminal_status": "not_started",
        "terminal_source": "security_admission_policy",
        "returncode": None,
        "error": None,
        "policy_rejection": decision_payload,
        "verification_required": False,
        "initial_verification_tool_count": 0,
        "verification_tool_count": 0,
        "verification_nudge_count": 0,
        "verification_nudges": [],
        "backend_attempts": [],
        "logical_llm_rounds": [],
        "llm_calls": [],
        "tool_calls": [],
        "rounds": [],
        "orphan_tools": [],
        "agent_messages": [],
        "agent_errors": [],
        "conversation_errors": [],
        "dynamic_graph": None,
    }
    common = {
        "request_id": request_id,
        "dataset_request_id": task.get("request_id"),
        "benchmark": task.get("benchmark"),
        "task_id": task.get("task_id"),
    }
    events = [
        {
            "ts": started,
            "event": "request_start",
            "queue_wait_s": queue_wait_s,
            **common,
        },
        {
            "ts": ended,
            "event": "policy_rejected",
            "policy_id": decision_payload["policy_id"],
            "reason": decision_payload["reason"],
            **common,
        },
        {
            "ts": ended,
            "event": "request_done",
            "status": "policy_rejected",
            "conversation_status": "policy_rejected",
            "queue_wait_s": queue_wait_s,
            "e2e_latency_s": trace["e2e_latency_s"],
            "gpu_total_s": 0.0,
            "tool_total_s": 0.0,
            "tool_wait_total_s": 0.0,
            "wait_total_s": queue_wait_s,
            "verification_tool_count": 0,
            "llm_call_count": 0,
            "backend_attempt_count": 0,
            "tool_call_count": 0,
            **common,
        },
    ]
    result = {
        "schema": "mars_security_policy_request_result_v1",
        "task_sha256": task_sha256 or _task_sha256(task),
        "agent": agent,
        "model": model,
        "trace": trace,
        "events": events,
    }
    _write_json_atomic(request_run_dir / "result.json", result)
    return result


__all__ = [
    "BLOCKED_TERMINALBENCH_TASK_IDS",
    "POLICY_ID",
    "blocked_security_task_decision",
    "build_policy_rejected_result",
]
