#!/usr/bin/env python3
"""Run one benchmark request through a native Codex/OpenCode frontend.

The frontend owns its session, model/tool loop, tools, follow-up turns and
terminal state.  MARS/vLLM is only the model provider.  This worker owns the
request workspace, outer isolation, transparent backend observation and the
one-way projection into the shared OpenHands trace contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from backend_proxy import BackendTraceProxy
from codex_app_server import (
    CodexAppServerDriver,
    CodexAppServerError,
    CodexTurnResult,
    build_codex_app_server_command,
)
from docker_proxy import ScopedDockerProxy
from frontend_contract import AgentRequestContext, FrontendTerminalStatus
from lifecycle_collector import LifecycleCollector, LifecycleStateError
from opencode_server import (
    OpenCodePhaseResult,
    OpenCodePromptTimeout,
    OpenCodeServerDriver,
    OpenCodeSnapshot,
)
from request_worker import (
    _agent_sandbox_command,
    _cpu_affinity_command,
    _is_bwrap_command,
    _llm_retry_policy,
    _runtime_env,
    _verification_tool_count,
    _working_bwrap,
    _write_codex_config,
    _write_codex_rules,
    _write_json,
    _write_opencode_config,
)
from sandbox_paths import task_readonly_paths


OPENHANDS_DIR = Path(
    os.environ.get(
        "MARS_OPENHANDS_DIR",
        str(Path(__file__).resolve().parents[1] / "openhands"),
    )
).expanduser().resolve()
if str(OPENHANDS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(OPENHANDS_DIR))

from request_bootstrap import (  # noqa: E402
    build_user_request,
    build_verification_nudge,
    materialize_workspace_init,
    requires_tool_verification,
)
from security_policy import (  # noqa: E402
    blocked_security_task_decision,
    build_policy_rejected_result,
)


_CODEX_TOOL_TYPES = {
    "commandExecution": "command_execution",
    "fileChange": "file_change",
    "mcpToolCall": "mcp_tool_call",
    "dynamicToolCall": "dynamic_tool_call",
    "collabAgentToolCall": "collab_agent_tool_call",
    "webSearch": "web_search",
}


def _codex_function_tool_name(name: Any) -> str:
    value = str(name or "function")
    if value in {"exec_command", "shell_command", "bash", "write_stdin"}:
        return "command_execution"
    if value in {"apply_patch", "file_change"}:
        return "file_change"
    return value


def _remaining(deadline_monotonic_s: float | None) -> float:
    if deadline_monotonic_s is None:
        return 365.0 * 24.0 * 3600.0
    return max(0.001, float(deadline_monotonic_s) - time.monotonic())


def _native_timestamp(message: Mapping[str, Any], params: Mapping[str, Any]) -> float | None:
    for raw in (
        message.get("emittedAtMs"),
        params.get("startedAtMs"),
        params.get("completedAtMs"),
    ):
        if isinstance(raw, (int, float)):
            return float(raw) / 1000.0
    return None


def _usage(raw: Any) -> dict[str, int | None]:
    value = raw if isinstance(raw, Mapping) else {}
    return {
        "prompt_tokens": (
            int(value.get("inputTokens"))
            if isinstance(value.get("inputTokens"), (int, float))
            else None
        ),
        "completion_tokens": (
            int(value.get("outputTokens"))
            if isinstance(value.get("outputTokens"), (int, float))
            else None
        ),
        "reasoning_tokens": (
            int(value.get("reasoningOutputTokens"))
            if isinstance(value.get("reasoningOutputTokens"), (int, float))
            else None
        ),
    }


def _reconcile_backend_attempts_from_raw(
    trace: dict[str, Any],
    raw_path: Path,
) -> int:
    """Close projected attempts from their request-scoped physical audit.

    The JSONL proxy audit is written before its lifecycle callback.  It is
    therefore the authoritative fallback when a frontend/proxy notification
    race leaves an already-known projected attempt nonterminal.  No attempt or
    round edge is created here; only an existing attempt with a matching raw
    terminal record may be completed.
    """
    if not raw_path.is_file():
        return 0
    raw_ended: dict[str, dict[str, Any]] = {}
    try:
        with raw_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if row.get("event") != "backend_attempt_ended":
                    continue
                attempt_id = str(row.get("backend_attempt_id") or "")
                if attempt_id:
                    raw_ended[attempt_id] = row
    except OSError:
        return 0

    targets: list[dict[str, Any]] = []
    seen_objects: set[int] = set()
    for item in trace.get("backend_attempts") or []:
        if isinstance(item, dict) and id(item) not in seen_objects:
            seen_objects.add(id(item))
            targets.append(item)
    for round_item in trace.get("rounds") or []:
        if not isinstance(round_item, Mapping):
            continue
        for item in round_item.get("backend_attempts") or []:
            if isinstance(item, dict) and id(item) not in seen_objects:
                seen_objects.add(id(item))
                targets.append(item)

    reconciled_ids: set[str] = set()
    for attempt in targets:
        attempt_id = str(
            attempt.get("backend_attempt_id") or attempt.get("attempt_id") or ""
        )
        raw_end = raw_ended.get(attempt_id)
        if raw_end is None:
            continue
        http_status = raw_end.get("http_status")
        error_obj = raw_end.get("error")
        success = isinstance(http_status, int) and 200 <= http_status < 300
        expected_status = "completed" if success and not error_obj else "error"
        if (
            attempt.get("end_time_s") is not None
            and attempt.get("status") == expected_status
            and attempt.get("http_status") == http_status
        ):
            continue
        attempt["end_time_s"] = raw_end.get("ts")
        attempt["end_monotonic_s"] = raw_end.get("monotonic_s")
        attempt["duration_s"] = raw_end.get("duration_s")
        attempt["status"] = expected_status
        attempt["http_status"] = http_status if isinstance(http_status, int) else None
        attempt["error"] = (
            json.dumps(error_obj, ensure_ascii=False) if error_obj else None
        )
        metadata = attempt.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.update(
                {
                    "native_response_id": raw_end.get("response_id"),
                    "ttft_s": raw_end.get("ttft_s"),
                    "terminal_reconciled_from_raw_audit": True,
                }
            )
        reconciled_ids.add(attempt_id)

    top_level = [
        item
        for item in (trace.get("backend_attempts") or [])
        if isinstance(item, Mapping)
    ]
    unknown = sum(
        not isinstance(item.get("duration_s"), (int, float)) for item in top_level
    )
    trace["backend_attempt_duration_unknown_count"] = unknown
    trace["backend_attempt_total_s"] = (
        sum(float(item["duration_s"]) for item in top_level)
        if top_level and unknown == 0
        else None
    )
    if reconciled_ids:
        trace["backend_attempt_reconciliation"] = {
            "source": "request_scoped_raw_proxy_audit",
            "attempt_ids": sorted(reconciled_ids),
        }
    return len(reconciled_ids)


def _codex_final_output(result: CodexTurnResult | None) -> str | None:
    if result is None:
        return None
    items = result.turn.get("items")
    if not isinstance(items, list):
        return None
    texts: list[str] = []
    for item in items:
        if not isinstance(item, Mapping) or item.get("type") != "agentMessage":
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
    return "\n".join(texts) or None


class _CodexLifecycleBridge:
    """Map native app-server notifications and physical HTTP attempts live."""

    def __init__(self, collector: LifecycleCollector, *, max_iterations: int) -> None:
        self.collector = collector
        self.max_iterations = max_iterations
        self._lock = threading.RLock()
        self.phase = "initial"
        self.thread_id: str | None = None
        self.active_turn_id: str | None = None
        self.phase_by_turn: dict[str, str] = {}
        self.active_round_by_turn: dict[str, str] = {}
        self.last_round_by_turn: dict[str, str] = {}
        self.round_count_by_turn: dict[str, int] = {}
        self.attempt_round: dict[str, str] = {}
        self.pending_backend_events: list[dict[str, Any]] = []
        self.call_parent: dict[str, str] = {}
        self.tool_state: dict[str, str] = {}
        self.last_round_id: str | None = None

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def _turn_id(self, params: Mapping[str, Any]) -> str | None:
        raw = params.get("turnId")
        if isinstance(raw, str) and raw:
            return raw
        turn = params.get("turn")
        raw = turn.get("id") if isinstance(turn, Mapping) else None
        return str(raw) if isinstance(raw, str) and raw else None

    def _ensure_round(self, turn_id: str | None = None) -> str | None:
        # A backend request can race ahead of the app-server's turn/started
        # notification.  Do not mint a fake native turn id in that window: the
        # backend rows are buffered by ``backend_event`` and rebound after the
        # real turn arrives.
        resolved_turn_id = turn_id or self.active_turn_id
        if resolved_turn_id is None:
            return None
        existing = self.active_round_by_turn.get(resolved_turn_id)
        if existing is not None:
            return existing
        ordinal = self.round_count_by_turn.get(resolved_turn_id, 0) + 1
        self.round_count_by_turn[resolved_turn_id] = ordinal
        round_id = f"{resolved_turn_id}:llm_{ordinal:04d}"
        self.active_round_by_turn[resolved_turn_id] = round_id
        self.last_round_id = round_id
        self.collector.llm_round_started(
            round_id,
            api_kind="responses",
            phase=self.phase_by_turn.get(resolved_turn_id, self.phase),
            metadata={
                "native_turn_id": resolved_turn_id,
                "causality": "backend_request_start",
            },
        )
        return round_id

    def _record_backend_event_locked(self, row: Mapping[str, Any]) -> bool:
        kind = str(row.get("event") or "")
        attempt_id = str(row.get("backend_attempt_id") or "")
        if not attempt_id:
            return True
        if kind == "backend_attempt_started":
            round_id = self._ensure_round()
            if round_id is None:
                return False
            if attempt_id in self.attempt_round:
                return True
            self.attempt_round[attempt_id] = round_id
            self.collector.backend_attempt_started(
                attempt_id,
                round_id=round_id,
                protocol=str(row.get("protocol") or "responses"),
                metadata={
                    "frontend_correlation_headers": row.get(
                        "frontend_correlation_headers"
                    ),
                    "input_sha256": row.get("input_sha256"),
                },
            )
            return True
        if kind == "backend_attempt_first_token":
            round_id = self.attempt_round.get(attempt_id)
            if round_id is None:
                return False
            try:
                self.collector.llm_round_first_token(round_id)
            except LifecycleStateError as exc:
                if not any(
                    marker in str(exc)
                    for marker in ("already recorded", "already ended")
                ):
                    raise
            return True
        if kind == "backend_attempt_ended":
            if attempt_id not in self.attempt_round:
                return False
            status_code = row.get("http_status")
            success = isinstance(status_code, int) and 200 <= status_code < 300
            error_obj = row.get("error")
            self.collector.backend_attempt_ended(
                attempt_id,
                status="completed" if success and not error_obj else "error",
                http_status=status_code if isinstance(status_code, int) else None,
                error=(json.dumps(error_obj, ensure_ascii=False) if error_obj else None),
                metadata={
                    "native_response_id": row.get("response_id"),
                    "ttft_s": row.get("ttft_s"),
                },
            )
            return True
        return True

    def _drain_pending_backend_events_locked(self) -> None:
        if self.active_turn_id is None or not self.pending_backend_events:
            return
        pending = self.pending_backend_events
        self.pending_backend_events = []
        # Normally the proxy callback is ordered.  The retry loop permits one
        # extra pass so even an observed first-token/end that raced ahead of its
        # start is attached only after that start has a real turn-bound round.
        while pending:
            deferred: list[dict[str, Any]] = []
            progress = False
            for row in pending:
                if self._record_backend_event_locked(row):
                    progress = True
                else:
                    deferred.append(row)
            if not progress:
                self.pending_backend_events.extend(deferred)
                return
            pending = deferred

    def backend_event(self, row: dict[str, Any]) -> None:
        kind = str(row.get("event") or "")
        attempt_id = str(row.get("backend_attempt_id") or "")
        if not attempt_id:
            return
        with self._lock:
            # A final physical attempt-end can arrive just after Codex emits
            # turn/completed and clears active_turn_id.  Its attempt already
            # has trustworthy round linkage, so record it immediately rather
            # than buffering it forever waiting for a new turn.  Only events
            # without an established attempt->round edge need an active turn.
            if (
                self.active_turn_id is None
                and attempt_id not in self.attempt_round
            ):
                self.pending_backend_events.append(dict(row))
                return
            if not self._record_backend_event_locked(row):
                self.pending_backend_events.append(dict(row))
                return
            if kind == "backend_attempt_started":
                self._drain_pending_backend_events_locked()

    def close_active_rounds(self, *, status: str, error: str | None = None) -> None:
        """End every observed logical round before an abnormal frontend terminal."""

        with self._lock:
            for turn_id, round_id in list(self.active_round_by_turn.items()):
                self.collector.llm_round_ended(
                    round_id,
                    status=status,
                    native_response_id=None,
                    error=error,
                    metadata={
                        "native_turn_id": turn_id,
                        "terminal_closure": True,
                    },
                )
                self.active_round_by_turn.pop(turn_id, None)
                self.last_round_by_turn[turn_id] = round_id
                self.last_round_id = round_id
            # Rows observed without turn/started have no trustworthy native
            # linkage.  Their physical audit remains in backend_attempts.jsonl;
            # manufacturing a logical/native edge here would be misleading.
            self.pending_backend_events.clear()
            self.active_turn_id = None

    def pending_tool_ids(self) -> list[str]:
        with self._lock:
            return sorted(
                tool_id
                for tool_id, state in self.tool_state.items()
                if state != "terminal"
            )

    def native_event(self, message: dict[str, Any], captured_at_s: float) -> None:
        del captured_at_s
        method = str(message.get("method") or "")
        raw_params = message.get("params")
        params = raw_params if isinstance(raw_params, Mapping) else {}
        thread_id = params.get("threadId")
        if not isinstance(thread_id, str):
            thread = params.get("thread")
            thread_id = thread.get("id") if isinstance(thread, Mapping) else None
        self.collector.record_native_event(
            message,
            source="codex_app_server",
            event_type=method,
            phase=self.phase,
            native_session_id=thread_id if isinstance(thread_id, str) else self.thread_id,
            native_timestamp_s=_native_timestamp(message, params),
        )
        with self._lock:
            if method == "thread/started" and isinstance(thread_id, str):
                self.thread_id = thread_id
                return
            turn_id = self._turn_id(params)
            if method == "turn/started" and turn_id is not None:
                self.active_turn_id = turn_id
                self.phase_by_turn[turn_id] = self.phase
                self._drain_pending_backend_events_locked()
                return
            if method == "rawResponseItem/completed":
                item = params.get("item")
                if not isinstance(item, Mapping):
                    return
                item_type = item.get("type")
                if item_type == "function_call_output":
                    raw_call_id = item.get("call_id")
                    call_id = (
                        raw_call_id
                        if isinstance(raw_call_id, str) and raw_call_id
                        else None
                    )
                    if call_id is None or self.tool_state.get(call_id) == "terminal":
                        return
                    if call_id not in self.tool_state:
                        # Without the matching native function_call there is
                        # no trustworthy name or parent edge.  Preserve the raw
                        # event but do not manufacture a tool record.
                        return
                    self.collector.tool_ended(
                        call_id,
                        status="completed",
                        parent_round_id=self.call_parent.get(call_id),
                        native_tool_call_id=call_id,
                        phase=self.phase_by_turn.get(turn_id or "", self.phase),
                        metadata={
                            "completion_source": "raw_function_call_output",
                            "linkage_quality": "native_call_id",
                        },
                        use_enqueue_as_start_if_missing=True,
                    )
                    self.tool_state[call_id] = "terminal"
                    return
                if item_type != "function_call":
                    return
                round_id = self._ensure_round(turn_id)
                if round_id is None:
                    return
                raw_call_id = item.get("call_id")
                native_call_id = (
                    raw_call_id
                    if isinstance(raw_call_id, str) and raw_call_id
                    else None
                )
                raw_item_id = item.get("id")
                response_item_id = (
                    raw_item_id
                    if isinstance(raw_item_id, str) and raw_item_id
                    else None
                )
                # A response item id is usable as a local observation key, but
                # it is not a native tool-call id when call_id is absent.
                tool_call_id = native_call_id or response_item_id
                if tool_call_id is None:
                    return
                if native_call_id is not None:
                    self.call_parent[native_call_id] = round_id
                if response_item_id is not None:
                    self.call_parent[response_item_id] = round_id
                if tool_call_id not in self.tool_state:
                    name = _codex_function_tool_name(item.get("name"))
                    self.collector.tool_queued(
                        tool_call_id,
                        tool_name=name,
                        parent_round_id=round_id,
                        native_tool_call_id=native_call_id,
                        phase=self.phase_by_turn.get(turn_id or "", self.phase),
                        metadata={
                            "response_item_id": response_item_id,
                            "linkage_quality": (
                                "native_call_id"
                                if native_call_id is not None
                                else "native_response_item_id"
                            ),
                        },
                    )
                    self.tool_state[tool_call_id] = "queued"
                return
            if method == "rawResponse/completed":
                resolved_turn_id = turn_id or self.active_turn_id
                round_id = self._ensure_round(resolved_turn_id)
                if round_id is None:
                    return
                response_id = params.get("responseId")
                native_response_id = (
                    response_id
                    if isinstance(response_id, str) and response_id
                    else None
                )
                self.collector.llm_round_ended(
                    round_id,
                    status="completed",
                    native_response_id=native_response_id,
                    usage=_usage(params.get("usage")),
                    metadata={"native_turn_id": resolved_turn_id},
                )
                if resolved_turn_id is not None:
                    self.active_round_by_turn.pop(resolved_turn_id, None)
                    self.last_round_by_turn[resolved_turn_id] = round_id
                self.last_round_id = round_id
                return
            if method not in {"item/started", "item/completed"}:
                if method == "turn/completed" and turn_id == self.active_turn_id:
                    self.active_turn_id = None
                return
            item = params.get("item")
            if not isinstance(item, Mapping):
                return
            native_type = str(item.get("type") or "")
            tool_name = _CODEX_TOOL_TYPES.get(native_type)
            if tool_name is None:
                return
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id:
                return
            if item_id in self.call_parent:
                round_id = self.call_parent[item_id]
                linkage_quality = "native_call_id"
            elif turn_id and turn_id in self.active_round_by_turn:
                round_id = self.active_round_by_turn[turn_id]
                linkage_quality = "native_turn_active_round"
            elif turn_id and turn_id in self.last_round_by_turn:
                round_id = self.last_round_by_turn[turn_id]
                linkage_quality = "native_turn_last_round"
            else:
                round_id = None
                linkage_quality = "unresolved"
            phase = self.phase_by_turn.get(turn_id or "", self.phase)
            item_metadata = {
                "native_item": dict(item),
                "linkage_quality": linkage_quality,
            }
            if method == "item/started":
                if item_id in self.tool_state and self.tool_state[item_id] == "queued":
                    self.collector.tool_started(
                        item_id,
                        parent_round_id=round_id,
                        native_tool_call_id=item_id,
                        phase=phase,
                        metadata=item_metadata,
                    )
                elif item_id not in self.tool_state:
                    self.collector.tool_started(
                        item_id,
                        tool_name=tool_name,
                        parent_round_id=round_id,
                        native_tool_call_id=item_id,
                        phase=phase,
                        metadata=item_metadata,
                    )
                self.tool_state[item_id] = "running"
                return
            status = str(item.get("status") or "completed")
            error = item.get("error")
            if item_id not in self.tool_state:
                self.collector.tool_ended(
                    item_id,
                    status=status,
                    tool_name=tool_name,
                    parent_round_id=round_id,
                    native_tool_call_id=item_id,
                    error=str(error) if error is not None else None,
                    phase=phase,
                    metadata=item_metadata,
                )
            elif self.tool_state[item_id] != "terminal":
                self.collector.tool_ended(
                    item_id,
                    status=status,
                    parent_round_id=round_id,
                    native_tool_call_id=item_id,
                    error=str(error) if error is not None else None,
                    phase=phase,
                    metadata=item_metadata,
                )
            self.tool_state[item_id] = "terminal"


def _opencode_session_id(event: Mapping[str, Any]) -> str | None:
    properties = event.get("properties")
    if not isinstance(properties, Mapping):
        return None
    raw = properties.get("sessionID")
    if isinstance(raw, str) and raw:
        return raw
    info = properties.get("info")
    if isinstance(info, Mapping):
        raw = info.get("sessionID") or info.get("id")
        if isinstance(raw, str) and raw:
            return raw
    part = properties.get("part")
    if isinstance(part, Mapping):
        raw = part.get("sessionID")
        if isinstance(raw, str) and raw:
            return raw
    return None


def _opencode_native_timestamp(event: Mapping[str, Any]) -> float | None:
    properties = event.get("properties")
    if not isinstance(properties, Mapping):
        return None
    part = properties.get("part")
    if isinstance(part, Mapping):
        state = part.get("state")
        timing = state.get("time") if isinstance(state, Mapping) else None
        if isinstance(timing, Mapping):
            for key in ("end", "start"):
                raw = timing.get(key)
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    return float(raw) / 1000.0
    info = properties.get("info")
    if isinstance(info, Mapping):
        timing = info.get("time")
        if isinstance(timing, Mapping):
            for key in ("completed", "created"):
                raw = timing.get(key)
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    return float(raw) / 1000.0
    return None


def _opencode_usage(raw: Any) -> dict[str, int | None]:
    value = raw if isinstance(raw, Mapping) else {}

    def integer(key: str) -> int | None:
        item = value.get(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return int(item)
        return None

    return {
        "prompt_tokens": integer("input"),
        "completion_tokens": integer("output"),
        "reasoning_tokens": integer("reasoning"),
    }


class _OpenCodeLifecycleBridge:
    """Project native OpenCode SSE into the OpenHands-compatible lifecycle.

    ``step-start``/``step-finish`` own logical model rounds.  Transport calls
    observed by :class:`BackendTraceProxy` are children of those rounds, so an
    SDK retry never becomes a fake extra agent iteration.  The
    ``x-session-affinity`` header keeps concurrent child-agent traffic
    separated without guessing from timestamps.
    """

    _TERMINAL_TOOL_STATES = {"completed", "error"}

    def __init__(self, collector: LifecycleCollector, *, max_iterations: int) -> None:
        self.collector = collector
        self.max_iterations = int(max_iterations)
        self._lock = threading.RLock()
        self.phase = "initial"
        self.root_session_id: str | None = None
        self.session_parent: dict[str, str | None] = {}
        self.session_phase: dict[str, str] = {}
        self.round_sequence: dict[str, int] = {}
        self.phase_round_count: dict[str, int] = {}
        self.active_round_by_session: dict[str, str] = {}
        self.round_by_message: dict[tuple[str, str], str] = {}
        self.last_round_by_message: dict[tuple[str, str], str] = {}
        self.round_message: dict[str, tuple[str, str] | None] = {}
        self.round_response_id: dict[str, str] = {}
        self.attempt_round: dict[str, str] = {}
        self.tool_state: dict[str, str] = {}
        self.session_errors: dict[str, str] = {}
        self.iteration_limited = False
        self._abort_callback: Any | None = None
        self._abort_started = False

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase
            self._abort_started = False
            if self.root_session_id is not None:
                self.session_phase[self.root_session_id] = phase

    def set_root_session(self, session_id: str) -> None:
        with self._lock:
            self.root_session_id = session_id
            self.session_parent.setdefault(session_id, None)
            self.session_phase[session_id] = self.phase

    def set_abort_callback(self, callback: Any) -> None:
        with self._lock:
            self._abort_callback = callback

    def _phase_for_session(self, session_id: str) -> str:
        current: str | None = session_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            phase = self.session_phase.get(current)
            if phase:
                return phase
            current = self.session_parent.get(current)
        return self.phase

    def _ensure_round(
        self,
        session_id: str,
        *,
        message_id: str | None = None,
        causality: str,
    ) -> str:
        active = self.active_round_by_session.get(session_id)
        if active is not None:
            if message_id:
                key = (session_id, message_id)
                self.round_by_message[key] = active
                self.round_message[active] = key
            return active
        ordinal = self.round_sequence.get(session_id, 0) + 1
        self.round_sequence[session_id] = ordinal
        round_id = f"{session_id}:llm_{ordinal:04d}"
        phase = self._phase_for_session(session_id)
        self.phase_round_count[phase] = self.phase_round_count.get(phase, 0) + 1
        self.active_round_by_session[session_id] = round_id
        key = (session_id, message_id) if message_id else None
        if key is not None:
            self.round_by_message[key] = round_id
        self.round_message[round_id] = key
        self.collector.llm_round_started(
            round_id,
            api_kind="chat_completions",
            phase=phase,
            metadata={
                "native_session_id": session_id,
                "native_message_id": message_id,
                "causality": causality,
            },
        )
        return round_id

    def backend_event(self, row: dict[str, Any]) -> None:
        kind = str(row.get("event") or "")
        attempt_id = str(row.get("backend_attempt_id") or "")
        if not attempt_id:
            return
        with self._lock:
            headers = row.get("frontend_correlation_headers")
            headers = headers if isinstance(headers, Mapping) else {}
            session_id = str(
                headers.get("x-session-affinity")
                or headers.get("x-session-id")
                or self.root_session_id
                or "session_pending"
            )
            if kind == "backend_attempt_started":
                round_id = self._ensure_round(
                    session_id, causality="backend_request_start"
                )
                self.attempt_round[attempt_id] = round_id
                self.collector.backend_attempt_started(
                    attempt_id,
                    round_id=round_id,
                    protocol=str(row.get("protocol") or "chat_completions"),
                    metadata={
                        "native_session_id": session_id,
                        "frontend_correlation_headers": dict(headers),
                        "input_sha256": row.get("input_sha256"),
                    },
                )
                return
            round_id = self.attempt_round.get(attempt_id)
            if round_id is None:
                return
            if kind == "backend_attempt_first_token":
                try:
                    self.collector.llm_round_first_token(round_id)
                except LifecycleStateError as exc:
                    if "already recorded" not in str(exc):
                        raise
                return
            if kind != "backend_attempt_ended":
                return
            status_code = row.get("http_status")
            error_obj = row.get("error")
            success = isinstance(status_code, int) and 200 <= status_code < 300
            response_id = row.get("response_id")
            if success and isinstance(response_id, str) and response_id:
                self.round_response_id[round_id] = response_id
            self.collector.backend_attempt_ended(
                attempt_id,
                status="completed" if success and not error_obj else "error",
                http_status=status_code if isinstance(status_code, int) else None,
                error=(json.dumps(error_obj, ensure_ascii=False) if error_obj else None),
                metadata={
                    "native_response_id": response_id,
                    "ttft_s": row.get("ttft_s"),
                },
            )

    def native_event(self, event: dict[str, Any], captured_at_s: float) -> None:
        del captured_at_s
        kind = str(event.get("type") or "")
        session_id = _opencode_session_id(event)
        self.collector.record_native_event(
            event,
            source="opencode_server_sse",
            event_type=kind,
            phase=self._phase_for_session(session_id) if session_id else self.phase,
            native_session_id=session_id,
            native_timestamp_s=_opencode_native_timestamp(event),
        )
        properties = event.get("properties")
        props = properties if isinstance(properties, Mapping) else {}
        with self._lock:
            if kind == "session.created":
                info = props.get("info")
                if isinstance(info, Mapping):
                    child = info.get("id")
                    parent = info.get("parentID")
                    if isinstance(child, str) and child:
                        parent_id = str(parent) if isinstance(parent, str) and parent else None
                        self.session_parent[child] = parent_id
                        self.session_phase[child] = (
                            self._phase_for_session(parent_id)
                            if parent_id is not None
                            else self.phase
                        )
                return
            if session_id is None:
                return
            if kind == "session.error":
                error = props.get("error")
                self.session_errors[session_id] = json.dumps(
                    error, ensure_ascii=False, default=str
                )
                return
            if kind == "message.updated":
                info = props.get("info")
                if not isinstance(info, Mapping) or info.get("role") != "assistant":
                    return
                message_id = info.get("id")
                if isinstance(message_id, str) and message_id:
                    self.session_phase.setdefault(session_id, self.phase)
                    error = info.get("error")
                    if error is not None:
                        self.session_errors[session_id] = json.dumps(
                            error, ensure_ascii=False, default=str
                        )
                return
            if kind != "message.part.updated":
                return
            part = props.get("part")
            if not isinstance(part, Mapping):
                return
            part_type = str(part.get("type") or "")
            message_id = part.get("messageID")
            message_id = str(message_id) if isinstance(message_id, str) and message_id else None
            if part_type == "step-start":
                self._ensure_round(
                    session_id,
                    message_id=message_id,
                    causality="native_step_start",
                )
                return
            if part_type != "step-finish":
                return
            round_id = self._ensure_round(
                session_id,
                message_id=message_id,
                causality="native_step_finish_without_start",
            )
            reason = str(part.get("reason") or "unknown")
            self.collector.llm_round_ended(
                round_id,
                status="completed",
                native_response_id=self.round_response_id.get(round_id)
                or (str(part.get("id")) if part.get("id") is not None else None),
                usage=_opencode_usage(part.get("tokens")),
                metadata={
                    "native_session_id": session_id,
                    "native_message_id": message_id,
                    "finish_reason": reason,
                    "native_step_finish_part_id": part.get("id"),
                    "cost": part.get("cost"),
                },
            )
            self.active_round_by_session.pop(session_id, None)
            if message_id:
                key = (session_id, message_id)
                self.round_by_message.pop(key, None)
                self.last_round_by_message[key] = round_id
            phase = self._phase_for_session(session_id)
            if (
                self.max_iterations > 0
                and self.phase_round_count.get(phase, 0) >= self.max_iterations
                and reason in {"tool-calls", "unknown"}
            ):
                self.iteration_limited = True
                self._request_abort_locked()

    def _request_abort_locked(self) -> None:
        if self._abort_started or self._abort_callback is None:
            return
        self._abort_started = True
        callback = self._abort_callback

        def abort() -> None:
            try:
                callback()
            except Exception:
                # The native phase/error reconciliation remains authoritative;
                # an abort transport failure is captured by driver stderr/error.
                pass

        threading.Thread(
            target=abort,
            name="mars-opencode-max-iterations-abort",
            daemon=True,
        ).start()

    def tool_event(self, event: dict[str, Any], captured_at_s: float) -> None:
        del captured_at_s
        with self._lock:
            session_id = str(event.get("sessionID") or "")
            message_id = str(event.get("messageID") or "")
            call_id = str(event.get("callID") or event.get("partID") or "")
            if not session_id or not message_id or not call_id:
                return
            tool_id = f"{session_id}:{message_id}:{call_id}"
            status = str(event.get("status") or "unknown")
            phase = str(event.get("phase") or self._phase_for_session(session_id))
            parent_round = (
                self.round_by_message.get((session_id, message_id))
                or self.last_round_by_message.get((session_id, message_id))
                or self.active_round_by_session.get(session_id)
            )
            tool_name = str(event.get("tool") or "unknown")
            state = event.get("state")
            metadata = {
                "native_session_id": session_id,
                "native_parent_session_id": event.get("parentSessionID"),
                "native_message_id": message_id,
                "native_part_id": event.get("partID"),
                "native_state": dict(state) if isinstance(state, Mapping) else state,
                "linkage_quality": (
                    "native_session_message_id"
                    if parent_round is not None
                    else "unresolved"
                ),
            }
            prior = self.tool_state.get(tool_id)
            if status == "pending" and prior is None:
                self.collector.tool_queued(
                    tool_id,
                    tool_name=tool_name,
                    parent_round_id=parent_round,
                    native_tool_call_id=call_id,
                    phase=phase,
                    metadata=metadata,
                )
                self.tool_state[tool_id] = "queued"
                return
            if status == "running" and prior not in {"running", "terminal"}:
                self.collector.tool_started(
                    tool_id,
                    tool_name=tool_name if prior is None else None,
                    parent_round_id=parent_round,
                    native_tool_call_id=call_id,
                    phase=phase,
                    metadata=metadata,
                )
                self.tool_state[tool_id] = "running"
                return
            if status not in self._TERMINAL_TOOL_STATES or prior == "terminal":
                return
            error: str | None = None
            if isinstance(state, Mapping) and state.get("error") is not None:
                error = str(state.get("error"))
            self.collector.tool_ended(
                tool_id,
                status=status,
                tool_name=tool_name if prior is None else None,
                parent_round_id=parent_round,
                native_tool_call_id=call_id,
                error=error,
                phase=phase,
                metadata=metadata,
            )
            self.tool_state[tool_id] = "terminal"

    def finalize_open_rounds(self, *, status: str, error: str | None = None) -> None:
        with self._lock:
            for session_id, round_id in tuple(self.active_round_by_session.items()):
                round_error = self.session_errors.get(session_id) or error
                self.collector.llm_round_ended(
                    round_id,
                    status=status,
                    native_response_id=self.round_response_id.get(round_id),
                    error=round_error,
                    metadata={
                        "native_session_id": session_id,
                        "reconciled_without_step_finish": True,
                    },
                )
                self.active_round_by_session.pop(session_id, None)


@dataclass(frozen=True, slots=True)
class _OpenCodeRunResult:
    status: str
    timed_out: bool
    iteration_limited: bool
    root_session_id: str
    phases: tuple[OpenCodePhaseResult, ...]
    snapshot: OpenCodeSnapshot
    final_output: str | None
    errors: tuple[dict[str, Any], ...]
    stdout_tail: str
    stderr_tail: str


def _opencode_final_output(snapshot: OpenCodeSnapshot) -> str | None:
    messages = snapshot.messages_by_session.get(snapshot.root_session_id, ())
    for row in reversed(messages):
        info = row.get("info") if isinstance(row, Mapping) else None
        if not isinstance(info, Mapping) or info.get("role") != "assistant":
            continue
        parts = row.get("parts")
        if not isinstance(parts, (list, tuple)):
            continue
        texts = [
            str(part.get("text"))
            for part in parts
            if isinstance(part, Mapping)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
            and str(part.get("text")).strip()
        ]
        if texts:
            return "\n".join(texts)
    return None


def _native_graph(trace: Mapping[str, Any]) -> dict[str, Any]:
    llm_calls = list(trace.get("llm_calls") or [])
    tools = list(trace.get("tool_calls") or [])
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    emitted: dict[str, list[str]] = {}
    for index, call in enumerate(llm_calls):
        round_id = str(call.get("round_id") or call.get("llm_call_id") or f"llm_{index:04d}")
        nodes.append(
            {
                "node_id": round_id,
                "kind": "llm",
                "round_index": index,
                "phase": call.get("phase"),
                "native_response_id": call.get("native_response_id"),
            }
        )
        emitted[round_id] = []
    for index, tool in enumerate(tools):
        tool_id = str(tool.get("tool_call_id") or f"tool_{index:04d}")
        parent = tool.get("parent_round_id")
        nodes.append(
            {
                "node_id": tool_id,
                "kind": "tool",
                "tool_name": tool.get("tool_name"),
                "status": tool.get("status"),
                "native_tool_call_id": tool.get("native_tool_call_id"),
            }
        )
        if isinstance(parent, str) and parent:
            metadata = tool.get("metadata")
            linkage_quality = (
                metadata.get("linkage_quality")
                if isinstance(metadata, Mapping)
                else None
            )
            edges.append(
                {
                    "source": parent,
                    "target": tool_id,
                    "kind": "emits",
                    "causality": linkage_quality or "native_id",
                }
            )
            emitted.setdefault(parent, []).append(tool_id)
    signature = "->".join(
        "llm["
        + ",".join(
            sorted(
                str(next(node.get("tool_name") for node in nodes if node["node_id"] == tool_id))
                for tool_id in emitted.get(str(call.get("round_id") or call.get("llm_call_id")), [])
            )
        )
        + "]"
        for call in llm_calls
    )
    return {
        "schema": "mars_native_dynamic_agent_graph_v2",
        "nodes": nodes,
        "edges": edges,
        "round_count": len(llm_calls),
        "tool_call_count": len(tools),
        "max_tools_emitted_by_round": max((len(value) for value in emitted.values()), default=0),
        "topology_signature": signature,
        "topology_sha256": hashlib.sha256(signature.encode()).hexdigest(),
        "edge_semantics": (
            "native message/call IDs where available; unresolved tools remain "
            "orphans; no timestamp-nearest inference"
        ),
    }


def _request_events(
    collector: LifecycleCollector,
    trace: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    verification_nudges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    snapshot = collector.snapshot()
    common = {
        "request_id": collector.request.request_id,
        "dataset_request_id": task.get("request_id"),
        "benchmark": task.get("benchmark"),
        "task_id": task.get("task_id"),
    }
    lifecycle_events = [
        {**event, **common}
        for event in snapshot.get("lifecycle_events") or []
    ]
    events = list(lifecycle_events)
    compatibility_names = {
        "llm_round_start": "gpu_submit",
        "llm_round_first_token": "gpu_first_token",
        "llm_round_end": "gpu_end",
    }
    for event in lifecycle_events:
        projected = compatibility_names.get(str(event.get("event") or ""))
        if projected is None:
            continue
        events.append(
            {
                **event,
                "event": projected,
                "llm_call_id": event.get("round_id"),
                "projection_source": "native_logical_round",
            }
        )
    events.extend({**event, **common} for event in verification_nudges)
    events.append(
        {
            "ts": trace.get("end_time_s"),
            "event": "request_done",
            "status": trace.get("status"),
            "conversation_status": trace.get("conversation_status"),
            "native_terminal_status": trace.get("native_terminal_status"),
            "queue_wait_s": trace.get("queue_wait_s"),
            "e2e_latency_s": trace.get("e2e_latency_s"),
            "gpu_total_s": trace.get("gpu_total_s"),
            "tool_total_s": trace.get("tool_total_s"),
            "tool_wait_total_s": trace.get("tool_wait_total_s"),
            "wait_total_s": trace.get("wait_total_s"),
            "verification_tool_count": trace.get("verification_tool_count"),
            "llm_call_count": len(trace.get("llm_calls") or []),
            "backend_attempt_count": len(trace.get("backend_attempts") or []),
            "tool_call_count": len(trace.get("tool_calls") or []),
            **common,
        }
    )
    return sorted(events, key=lambda event: float(event.get("ts") or 0.0))


def _run_codex(
    *,
    agent_bin: Path,
    workspace_dir: Path,
    runtime_root: Path,
    local_inputs_dir: Path,
    readonly_paths: tuple[Path, ...],
    require_bwrap: bool,
    cpu_affinity: str | None,
    env: dict[str, str],
    collector: LifecycleCollector,
    bridge: _CodexLifecycleBridge,
    request_run_dir: Path,
    model: str,
    prompt: str,
    job_id: str,
    max_iterations: int,
    deadline_monotonic_s: float | None,
    verification_required: bool,
    verification_prompt: str,
) -> tuple[CodexTurnResult, CodexTurnResult | None, bool, list[dict[str, Any]], int]:
    native = build_codex_app_server_command(agent_bin)
    command = _agent_sandbox_command(
        native,
        agent_bin=agent_bin,
        workspace_dir=workspace_dir,
        runtime_root=runtime_root,
        local_inputs_dir=local_inputs_dir,
        readonly_paths=readonly_paths,
        require_bwrap=require_bwrap,
    )
    sandboxed = _is_bwrap_command(command)
    command = _cpu_affinity_command(command, cpu_affinity)
    _write_json(request_run_dir / "command.json", command)
    nudges: list[dict[str, Any]] = []
    initial: CodexTurnResult
    verification: CodexTurnResult | None = None
    with CodexAppServerDriver(
        command=command,
        raw_jsonl_path=request_run_dir / "codex_app_server.raw.jsonl",
        wire_audit_path=request_run_dir / "codex_app_server.wire.jsonl",
        stderr_path=request_run_dir / "codex_app_server.stderr.txt",
        cwd=workspace_dir,
        env=env,
        event_callback=bridge.native_event,
    ) as driver:
        thread = driver.start_thread(
            model=model,
            model_provider="mars",
            cwd=workspace_dir,
            approval_policy="never",
            sandbox="danger-full-access",
            ephemeral=True,
            experimental_raw_events=True,
        )
        thread_obj = thread.get("thread") if isinstance(thread, Mapping) else None
        thread_id = thread_obj.get("id") if isinstance(thread_obj, Mapping) else None
        collector.frontend_started(
            native_session_id=str(thread_id) if thread_id is not None else None,
            phase="initial",
        )
        bridge.set_phase("initial")
        try:
            initial = driver.run_turn(
                prompt,
                phase="initial",
                responsesapi_client_metadata={"mars_job_id": job_id},
                client_user_message_id=f"{collector.request.request_id}:initial",
                timeout_s=_remaining(deadline_monotonic_s),
                max_raw_responses=max_iterations,
            )
        except BaseException as exc:
            bridge.close_active_rounds(
                status="error", error=f"{type(exc).__name__}: {exc}"
            )
            raise
        initial_tool_count = _verification_tool_count(collector.snapshot()["tool_calls"])
        if (
            verification_required
            and initial.status == "completed"
            and initial_tool_count <= 0
            and not initial.timed_out
            and not getattr(initial, "iteration_limited", False)
        ):
            nudge = {
                "ts": time.time(),
                "event": "verification_nudge",
                "attempt": 1,
                "phase": "verification",
                "native_session_id": initial.thread_id,
            }
            nudges.append(nudge)
            bridge.set_phase("verification")
            try:
                verification = driver.run_verification_turn(
                    verification_prompt,
                    responsesapi_client_metadata={"mars_job_id": job_id},
                    client_user_message_id=f"{collector.request.request_id}:verification:1",
                    timeout_s=_remaining(deadline_monotonic_s),
                    max_raw_responses=max_iterations,
                )
            except BaseException as exc:
                bridge.close_active_rounds(
                    status="error", error=f"{type(exc).__name__}: {exc}"
                )
                raise
        terminal_result = verification or initial
        pending_tool_ids = bridge.pending_tool_ids()
        if terminal_result.status == "completed" and pending_tool_ids:
            raise CodexAppServerError(
                "Codex turn completed with unresolved native tool calls: "
                + ", ".join(pending_tool_ids)
            )
        native_error = (
            str(terminal_result.error)
            if terminal_result.error is not None
            else None
        )
        if terminal_result.timed_out:
            terminal_status = FrontendTerminalStatus.TIMEOUT
            round_status = "timeout"
            terminal_reason = native_error or "Codex turn timed out"
        elif terminal_result.iteration_limited:
            terminal_status = FrontendTerminalStatus.MAX_ITERATIONS
            round_status = "max_iterations"
            terminal_reason = native_error or "Codex turn reached max iterations"
        elif terminal_result.status == "completed":
            terminal_status = FrontendTerminalStatus.COMPLETED
            round_status = "completed"
            terminal_reason = native_error
        elif terminal_result.status == "interrupted":
            terminal_status = FrontendTerminalStatus.INTERRUPTED
            round_status = "interrupted"
            terminal_reason = native_error or "Codex turn was interrupted"
        else:
            terminal_status = FrontendTerminalStatus.ERROR
            round_status = str(terminal_result.status or "error")
            terminal_reason = native_error or f"Codex turn ended as {round_status}"
        if terminal_status != FrontendTerminalStatus.COMPLETED:
            bridge.close_active_rounds(
                status=round_status,
                error=terminal_reason,
            )
        # Freeze the measurement boundary while the native frontend is still
        # alive, immediately after its matching turn/completed.  app-server
        # shutdown and all container/proxy teardown happen after this call.
        collector.frontend_terminal(
            terminal_status,
            reason=terminal_reason,
            native_session_id=terminal_result.thread_id,
            final_output=_codex_final_output(terminal_result),
            metadata={
                "native_turn_id": terminal_result.turn_id,
                "native_turn_status": terminal_result.status,
                "timed_out": terminal_result.timed_out,
                "iteration_limited": terminal_result.iteration_limited,
            },
        )
    return initial, verification, sandboxed, nudges, initial_tool_count


def _run_opencode(
    *,
    agent_bin: Path,
    workspace_dir: Path,
    runtime_root: Path,
    local_inputs_dir: Path,
    readonly_paths: tuple[Path, ...],
    require_bwrap: bool,
    cpu_affinity: str | None,
    env: dict[str, str],
    collector: LifecycleCollector,
    bridge: _OpenCodeLifecycleBridge,
    request_run_dir: Path,
    model: str,
    prompt: str,
    max_iterations: int,
    deadline_monotonic_s: float | None,
    verification_required: bool,
    verification_prompt: str,
) -> tuple[_OpenCodeRunResult, bool, list[dict[str, Any]], int]:
    sandbox_state = {"enabled": False}

    def command_builder(native: tuple[str, ...]) -> list[str]:
        command = _agent_sandbox_command(
            list(native),
            agent_bin=agent_bin,
            workspace_dir=workspace_dir,
            runtime_root=runtime_root,
            local_inputs_dir=local_inputs_dir,
            readonly_paths=readonly_paths,
            require_bwrap=require_bwrap,
        )
        sandbox_state["enabled"] = _is_bwrap_command(command)
        return _cpu_affinity_command(command, cpu_affinity)

    nudges: list[dict[str, Any]] = []
    initial: OpenCodePhaseResult | None = None
    verification: OpenCodePhaseResult | None = None
    timed_out = False
    integrity_error: str | None = None
    driver = OpenCodeServerDriver(
        workspace_dir=workspace_dir,
        model=model,
        provider_id="mars",
        agent="build",
        binary=agent_bin,
        command_builder=command_builder,
        env=env,
        cwd=workspace_dir,
        event_callback=bridge.native_event,
        tool_event_callback=bridge.tool_event,
        event_settle_s=0.0,
    )
    _write_json(request_run_dir / "command.json", list(driver.command))
    with driver:
        root_session_id = driver.start()
        bridge.set_root_session(root_session_id)
        bridge.set_abort_callback(driver.abort)
        collector.frontend_started(
            native_session_id=root_session_id,
            phase="initial",
        )
        bridge.set_phase("initial")
        try:
            initial = driver.run_prompt(
                prompt,
                phase="initial",
                timeout_s=_remaining(deadline_monotonic_s),
            )
        except OpenCodePromptTimeout:
            timed_out = True
            phases = driver.snapshot().phases
            if phases:
                initial = phases[-1]
        if initial is None:
            raise RuntimeError("OpenCode produced no native initial phase result")
        initial_tool_count = _verification_tool_count(
            collector.snapshot()["tool_calls"]
        )
        if (
            verification_required
            and initial.status == "completed"
            and initial_tool_count <= 0
            and not timed_out
            and not bridge.iteration_limited
        ):
            nudge = {
                "ts": time.time(),
                "event": "verification_nudge",
                "attempt": 1,
                "phase": "verification",
                "native_session_id": root_session_id,
            }
            nudges.append(nudge)
            bridge.set_phase("verification")
            try:
                verification = driver.run_verification_prompt(
                    verification_prompt,
                    phase="verification",
                    timeout_s=_remaining(deadline_monotonic_s),
                )
            except OpenCodePromptTimeout:
                timed_out = True
                phases = driver.snapshot().phases
                if phases:
                    verification = phases[-1]

        terminal_phase = verification or initial
        if terminal_phase is None:
            raise RuntimeError("OpenCode produced no native terminal phase")
        open_rounds = bool(bridge.active_round_by_session)
        if bridge.iteration_limited:
            bridge.finalize_open_rounds(
                status="interrupted", error="maximum logical iterations reached"
            )
            terminal_status = FrontendTerminalStatus.MAX_ITERATIONS
            native_status = "max_iterations"
        elif timed_out or terminal_phase.timed_out:
            bridge.finalize_open_rounds(
                status="interrupted", error="native OpenCode phase timed out"
            )
            terminal_status = FrontendTerminalStatus.TIMEOUT
            native_status = "timeout"
        elif terminal_phase.status == "completed" and not open_rounds:
            terminal_status = FrontendTerminalStatus.COMPLETED
            native_status = "completed"
        else:
            if open_rounds and terminal_phase.status == "completed":
                integrity_error = (
                    "OpenCode root became idle with an unfinished logical LLM round"
                )
            bridge.finalize_open_rounds(
                status="error",
                error=integrity_error or "native OpenCode phase failed",
            )
            terminal_status = FrontendTerminalStatus.ERROR
            native_status = "error"

        # Root idle/error/timeout is the native measurement boundary.  Reconcile
        # messages and shut down the request-local server only after this event,
        # exactly as OpenHands excludes sandbox cleanup from request latency.
        collector.frontend_terminal(
            terminal_status,
            reason=(
                integrity_error
                or (
                    json.dumps(terminal_phase.abort_error, ensure_ascii=False)
                    if terminal_phase.abort_error
                    else None
                )
            ),
            native_session_id=root_session_id,
            metadata={
                "native_phase": terminal_phase.phase,
                "native_phase_status": terminal_phase.status,
                "busy_seen": terminal_phase.busy_seen,
                "retry_count": terminal_phase.retry_count,
                "timed_out": timed_out or terminal_phase.timed_out,
                "iteration_limited": bridge.iteration_limited,
            },
        )
        snapshot = driver.reconcile(best_effort=True)
        result = _OpenCodeRunResult(
            status=native_status,
            timed_out=timed_out or terminal_phase.timed_out,
            iteration_limited=bridge.iteration_limited,
            root_session_id=root_session_id,
            phases=snapshot.phases,
            snapshot=snapshot,
            final_output=_opencode_final_output(snapshot),
            errors=tuple(dict(item) for item in snapshot.errors),
            stdout_tail=driver.stdout_tail,
            stderr_tail=driver.stderr_tail,
        )
        (request_run_dir / "opencode_server.stdout.txt").write_text(
            result.stdout_tail, encoding="utf-8"
        )
        (request_run_dir / "opencode_server.stderr.txt").write_text(
            result.stderr_tail, encoding="utf-8"
        )
    return result, sandbox_state["enabled"], nudges, initial_tool_count


def run_request(
    *,
    agent: str,
    agent_bin: Path,
    task: dict[str, Any],
    task_sha256: str,
    request_id: str,
    mars_job_id: str,
    arrival_time_s: float,
    workspace_root: Path,
    local_inputs_dir: Path,
    request_run_dir: Path,
    model: str,
    base_url: str,
    api_key: str,
    context_window: int,
    max_output_tokens: int,
    max_iterations: int,
    temperature: float,
    top_p: float | None = None,
    timeout_s: float,
    require_bwrap: bool,
    llm_timeout_s: float = 3000.0,
    llm_num_retries: int = 3,
    terminal_no_change_timeout_s: int | None = 30,
    benchmark_root: Path | None = None,
    cpu_affinity: str | None = None,
    admission_gate_path: Path | None = None,
    allow_blocked_security_tasks: bool = False,
    codex_enable_subagents: bool = False,
) -> dict[str, Any]:
    if agent not in {"codex", "opencode"}:
        raise ValueError(f"unsupported agent: {agent}")
    policy_decision = blocked_security_task_decision(
        task,
        allow_blocked_security_tasks=allow_blocked_security_tasks,
    )
    if policy_decision is not None:
        return build_policy_rejected_result(
            task=task,
            decision=policy_decision,
            request_id=request_id,
            arrival_time_s=arrival_time_s,
            workspace_root=workspace_root,
            request_run_dir=request_run_dir,
            agent=agent,
            model=model,
            task_sha256=task_sha256,
        )

    # Match OpenHands' worker boundary: queue wait ends as soon as the request
    # worker begins, before workspace materialization, Docker preparation, or
    # frontend startup.
    request_started_wall = time.time()
    request_started_mono = time.monotonic()
    request_run_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = (workspace_root / request_id).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    _write_json(request_run_dir / "task.json", task)
    prompt = build_user_request(task)
    (request_run_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    workspace_file_count = materialize_workspace_init(task, workspace_dir)
    (workspace_dir / ".mars_runtime" / "artifacts").mkdir(parents=True, exist_ok=True)

    arrival_age_s = max(0.0, request_started_wall - float(arrival_time_s))
    arrival_monotonic_s = request_started_mono - arrival_age_s
    deadline = request_started_mono + float(timeout_s) if float(timeout_s) > 0 else None
    env = _runtime_env(request_run_dir, workspace_dir, api_key)
    runtime_root = request_run_dir / "runtime"
    local_inputs_dir.mkdir(parents=True, exist_ok=True)

    allowed_roots = [local_inputs_dir]
    if benchmark_root is not None:
        allowed_roots.append(benchmark_root)
    readonly_paths = task_readonly_paths(task, allowed_roots=allowed_roots)
    docker_proxy: ScopedDockerProxy | None = None
    docker_tempdir: tempfile.TemporaryDirectory[str] | None = None
    docker_access = "none"
    if str(task.get("benchmark") or "").strip().lower() == "terminalbench":
        try:
            docker_tempdir = tempfile.TemporaryDirectory(
                prefix=f"mars-native-dp-{os.getuid()}-{request_id[-8:]}-", dir="/tmp"
            )
            docker_runtime = Path(docker_tempdir.name) / "runtime"
            docker_runtime.mkdir(parents=True, exist_ok=True)
            docker_proxy = ScopedDockerProxy(
                workspace_dir=workspace_dir,
                runtime_root=docker_runtime,
                cleanup_descriptor_path=(
                    request_run_dir / "docker_cleanup_descriptor.json"
                ),
            )
            docker_proxy.install_wrapper(env)
            readonly_paths = tuple(
                sorted({*readonly_paths, docker_proxy.task_dir, docker_runtime}, key=str)
            )
            docker_access = "scoped_terminalbench_proxy"
        except Exception as exc:
            docker_access = "unavailable_missing_asset"
            (request_run_dir / "docker_proxy_unavailable.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )

    request_context = AgentRequestContext(
        request_id=request_id,
        job_id=mars_job_id,
        prompt=prompt,
        workspace_dir=workspace_dir,
        model=model,
        backend_base_url=base_url,
        arrival_time_s=arrival_time_s,
        arrival_monotonic_s=arrival_monotonic_s,
        dataset_request_id=(str(task.get("request_id")) if task.get("request_id") is not None else None),
        benchmark=(str(task.get("benchmark")) if task.get("benchmark") is not None else None),
        task_id=(str(task.get("task_id")) if task.get("task_id") is not None else None),
        readonly_paths=readonly_paths,
        max_iterations=max_iterations,
        deadline_monotonic_s=deadline,
        runtime_env=env,
        metadata={"agent": agent, "public_latency_clock": "wall_openhands"},
    )
    verification_required = requires_tool_verification(task)
    verification_nudges: list[dict[str, Any]] = []
    native_terminal_status: str | None = None
    terminal_source = "native_frontend"
    terminal_result: Any = None
    # Record configured isolation even if the native frontend exits before the
    # phase helper can return its per-command flag.
    outer_sandbox_enabled = bool(require_bwrap and _working_bwrap())
    control_error: dict[str, Any] | None = None
    cleanup_error: dict[str, Any] | None = None
    initial_verification_tool_count = 0
    agent_messages: list[dict[str, Any]] = []
    agent_errors: list[dict[str, Any]] = []
    conversation_errors: list[dict[str, Any]] = []
    final_output: str | None = None

    collector = LifecycleCollector(
        request_context,
        native_events_path=request_run_dir / "native_events.jsonl",
        lifecycle_events_path=request_run_dir / "lifecycle_events.jsonl",
        live_events_path=request_run_dir / "live_events.jsonl",
    )
    collector.request_started(
        observed_time_s=request_started_wall,
        observed_monotonic_s=request_started_mono,
    )
    bridge: Any
    if agent == "codex":
        bridge = _CodexLifecycleBridge(collector, max_iterations=max_iterations)
    else:
        bridge = _OpenCodeLifecycleBridge(collector, max_iterations=max_iterations)

    proxy = BackendTraceProxy(
        upstream_base_url=base_url,
        trace_path=request_run_dir / "backend_attempts.jsonl",
        request_id=request_id,
        mars_job_id=mars_job_id,
        upstream_timeout_s=llm_timeout_s,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        responses_namespace_tool_compat=(
            agent == "codex" and bool(codex_enable_subagents)
        ),
        admission_gate_path=admission_gate_path,
        on_event=(bridge.backend_event if bridge is not None else None),
    )
    try:
        with ExitStack() as stack:
            if docker_proxy is not None:
                stack.enter_context(docker_proxy)
            stack.enter_context(proxy)
            if agent == "codex":
                config_path = Path(env["CODEX_HOME"]) / "config.toml"
                _write_codex_config(
                    config_path,
                    proxy_base_url=proxy.base_url,
                    model=model,
                    context_window=context_window,
                    llm_timeout_s=llm_timeout_s,
                    llm_num_retries=llm_num_retries,
                    sandbox_mode="danger-full-access" if require_bwrap else "workspace-write",
                    enable_subagents=codex_enable_subagents,
                )
                _write_codex_rules(Path(env["CODEX_HOME"]))
                (
                    initial,
                    verification,
                    outer_sandbox_enabled,
                    verification_nudges,
                    initial_verification_tool_count,
                ) = _run_codex(
                    agent_bin=agent_bin,
                    workspace_dir=workspace_dir,
                    runtime_root=runtime_root,
                    local_inputs_dir=local_inputs_dir,
                    readonly_paths=readonly_paths,
                    require_bwrap=require_bwrap,
                    cpu_affinity=cpu_affinity,
                    env=env,
                    collector=collector,
                    bridge=bridge,
                    request_run_dir=request_run_dir,
                    model=model,
                    prompt=prompt,
                    job_id=mars_job_id,
                    max_iterations=max_iterations,
                    deadline_monotonic_s=deadline,
                    verification_required=verification_required,
                    verification_prompt=build_verification_nudge(task),
                )
                terminal_result = verification or initial
                native_terminal_status = terminal_result.status
                final_output = _codex_final_output(terminal_result)
                for phase_name, phase_result in (
                    ("initial", initial),
                    ("verification", verification),
                ):
                    if phase_result is None:
                        continue
                    items = phase_result.turn.get("items")
                    if isinstance(items, list):
                        for item in items:
                            if not isinstance(item, Mapping):
                                continue
                            if item.get("type") == "agentMessage":
                                agent_messages.append(
                                    {
                                        "phase": phase_name,
                                        "turn_id": phase_result.turn_id,
                                        "item": dict(item),
                                        "preview": item.get("text"),
                                    }
                                )
                    if phase_result.error is not None:
                        agent_errors.append(
                            {
                                "phase": phase_name,
                                "turn_id": phase_result.turn_id,
                                "error": str(phase_result.error),
                            }
                        )
                conversation_errors = list(agent_errors)
            else:
                config_path = Path(env["XDG_CONFIG_HOME"]) / "opencode.json"
                _write_opencode_config(
                    config_path,
                    proxy_base_url=proxy.base_url,
                    model=model,
                    context_window=context_window,
                    max_output_tokens=max_output_tokens,
                    max_iterations=max_iterations,
                    temperature=temperature,
                )
                env["OPENCODE_CONFIG"] = str(config_path)
                (
                    opencode_result,
                    outer_sandbox_enabled,
                    verification_nudges,
                    initial_verification_tool_count,
                ) = _run_opencode(
                    agent_bin=agent_bin,
                    workspace_dir=workspace_dir,
                    runtime_root=runtime_root,
                    local_inputs_dir=local_inputs_dir,
                    readonly_paths=readonly_paths,
                    require_bwrap=require_bwrap,
                    cpu_affinity=cpu_affinity,
                    env=env,
                    collector=collector,
                    bridge=bridge,
                    request_run_dir=request_run_dir,
                    model=model,
                    prompt=prompt,
                    max_iterations=max_iterations,
                    deadline_monotonic_s=deadline,
                    verification_required=verification_required,
                    verification_prompt=build_verification_nudge(task),
                )
                terminal_result = opencode_result
                native_terminal_status = opencode_result.status
                final_output = opencode_result.final_output
                agent_errors = [dict(item) for item in opencode_result.errors]
                conversation_errors = [dict(item) for item in opencode_result.errors]
                for session_id, rows in opencode_result.snapshot.messages_by_session.items():
                    for row in rows:
                        agent_messages.append(
                            {
                                "session_id": session_id,
                                "message": dict(row),
                            }
                        )
    except BaseException as exc:
        error_record = {"type": type(exc).__name__, "message": str(exc)}
        if collector.snapshot().get("terminal") is not None:
            cleanup_error = error_record
        else:
            control_error = error_record
            terminal_source = "frontend_transport_failure"
            collector.frontend_terminal(
                FrontendTerminalStatus.ERROR,
                reason=f"{type(exc).__name__}: {exc}",
                metadata={"native_terminal_missing": True},
            )
    finally:
        # Terminal time was frozen above.  These operations are intentionally
        # excluded from request e2e latency.
        if docker_tempdir is not None:
            try:
                audit_source = Path(docker_tempdir.name) / "docker_proxy_audit.jsonl"
                if audit_source.is_file():
                    shutil.copy2(audit_source, request_run_dir / "docker_proxy_audit.jsonl")
                docker_tempdir.cleanup()
            except Exception as exc:
                cleanup_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "phase": "docker_cleanup",
                }

    snapshot = collector.snapshot()
    verification_tool_count = _verification_tool_count(snapshot["tool_calls"])
    trace = collector.snapshot_openhands_trace()
    native_status = native_terminal_status or (snapshot.get("terminal") or {}).get("status")
    published_status = str(trace.get("status") or "error")
    if control_error is not None:
        published_status = "worker_error"
    elif terminal_result is not None and getattr(terminal_result, "timed_out", False):
        published_status = "worker_timeout"
    elif terminal_result is not None and getattr(terminal_result, "iteration_limited", False):
        published_status = "max_iterations"
    elif native_status == "completed":
        published_status = "completed"
    elif native_status == "interrupted":
        published_status = "interrupted"
    elif native_status in {"failed", "error"}:
        published_status = "error"
    if verification_required and verification_tool_count <= 0 and published_status == "completed":
        published_status = "verification_missing"

    trace.update(
        {
            "schema": "mars_native_agent_trace_v3",
            "agent": agent,
            "mars_job_id": mars_job_id,
            "workspace_file_count": workspace_file_count,
            "outer_sandbox": (
                "bubblewrap_pid_filesystem"
                if outer_sandbox_enabled
                else "disabled_debug_only"
            ),
            "docker_access": docker_access,
            "sandbox_readonly_paths": [str(path) for path in readonly_paths],
            "cpu_affinity": cpu_affinity,
            "llm_timeout_s": llm_timeout_s,
            "llm_num_retries": llm_num_retries if agent == "codex" else None,
            "codex_enable_subagents": (
                bool(codex_enable_subagents) if agent == "codex" else None
            ),
            "codex_namespace_tool_compat": (
                "flatten_restore_v1"
                if agent == "codex" and codex_enable_subagents
                else None
            ),
            "requested_llm_num_retries": llm_num_retries,
            "llm_retry_policy": _llm_retry_policy(agent, llm_num_retries),
            "top_p": top_p,
            "terminal_no_change_timeout_s": terminal_no_change_timeout_s,
            "status": published_status,
            "native_terminal_status": native_status,
            "terminal_source": terminal_source,
            "conversation_status": native_status,
            "error": control_error or trace.get("error"),
            "cleanup_error": cleanup_error,
            "verification_required": verification_required,
            "initial_verification_tool_count": initial_verification_tool_count,
            "verification_tool_count": verification_tool_count,
            "verification_nudge_count": len(verification_nudges),
            "verification_nudges": verification_nudges,
            "agent_session_id": trace.get("native_session_id"),
            "agent_messages": agent_messages,
            "agent_errors": agent_errors + ([control_error] if control_error else []),
            "conversation_errors": conversation_errors,
            "final_output": final_output or trace.get("final_output"),
            "frontend_protocol": (
                "codex_app_server_v2" if agent == "codex" else "opencode_legacy_server_sse"
            ),
            "backend_protocol": "responses" if agent == "codex" else "chat_completions",
            "dynamic_graph": None,
        }
    )
    _reconcile_backend_attempts_from_raw(
        trace,
        request_run_dir / "backend_attempts.jsonl",
    )
    trace["dynamic_graph"] = _native_graph(trace)
    events = _request_events(
        collector, trace, task=task, verification_nudges=verification_nudges
    )
    result = {
        "schema": "mars_native_agent_request_result_v3",
        "task_sha256": task_sha256,
        "agent": agent,
        "model": model,
        "trace": trace,
        "events": events,
    }
    _write_json(request_run_dir / "result.json", result)
    collector.close()
    return result


__all__ = ["run_request"]
