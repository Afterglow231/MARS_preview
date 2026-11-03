#!/usr/bin/env python3

"""Thread-safe real-time lifecycle collector shared by agent frontends."""

from __future__ import annotations

import copy
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

try:  # Works both as a package and in the isolated flat runtime bundle.
    from .frontend_contract import (
        AgentRequestContext,
        FrontendOutcome,
        FrontendTerminalStatus,
    )
except ImportError:  # pragma: no cover - exercised by flat-bundle integration.
    from frontend_contract import (  # type: ignore
        AgentRequestContext,
        FrontendOutcome,
        FrontendTerminalStatus,
    )


NATIVE_EVENT_SCHEMA = "mars_agent_native_event_v1"
LIFECYCLE_EVENT_SCHEMA = "mars_agent_lifecycle_event_v1"


class LifecycleStateError(RuntimeError):
    """Raised when an adapter emits an impossible lifecycle transition."""


@dataclass(frozen=True, slots=True)
class _Stamp:
    time_s: float
    monotonic_s: float


class _AppendOnlyJSONL:
    """A flushed append-only writer; it never truncates an existing artifact."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._closed = False

    def write_line(self, line: str) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleStateError(f"JSONL writer is closed: {self.path}")
            self._fp.write(line + "\n")
            self._fp.flush()

    def write(self, row: Mapping[str, Any]) -> str:
        line = _strict_json_dumps(row)
        self.write_line(line)
        return line

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._fp.flush()
            self._fp.close()
            self._closed = True


def _strict_json_dumps(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
        allow_nan=False,
    )


def _json_ready(value: Any) -> Any:
    """Normalize generic Mapping/tuple inputs while retaining strict JSON."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _json_copy(value: Any) -> Any:
    """Detach state from caller-owned objects and reject non-JSON payloads."""

    return json.loads(_strict_json_dumps(value))


def _nonempty(value: str, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _optional_finite(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite when provided")
    return number


def _duration(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    value = float(end) - float(start)
    if value < 0:
        raise LifecycleStateError(
            f"monotonic event order moved backwards: start={start}, end={end}"
        )
    return value


def _complete_sum(records: list[dict[str, Any]], field: str) -> float | None:
    if not records:
        return 0.0
    values = [record.get(field) for record in records]
    if any(value is None for value in values):
        return None
    return float(sum(float(value) for value in values))


class LifecycleCollector:
    """Collect one request's native and normalized events in real time.

    Native events are serialized immediately and retained internally as their
    exact JSON strings, so later mutation of the adapter payload cannot rewrite
    history.  Lifecycle durations use collector-observed monotonic timestamps.
    Native/source timestamps are provenance only and remain null when absent.
    """

    def __init__(
        self,
        request: AgentRequestContext,
        *,
        native_events_path: Path | None = None,
        lifecycle_events_path: Path | None = None,
        live_events_path: Path | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.request = request
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._lock = threading.RLock()
        self._closed = False
        self._native_sequence = 0
        self._lifecycle_sequence = 0
        self._native_event_lines: list[str] = []
        self._lifecycle_events: list[dict[str, Any]] = []
        self._request_start: dict[str, Any] | None = None
        self._frontend_start: dict[str, Any] | None = None
        self._terminal: dict[str, Any] | None = None
        self._native_session_id: str | None = None
        self._llm_rounds: dict[str, dict[str, Any]] = {}
        self._backend_attempts: dict[str, dict[str, Any]] = {}
        self._tools: dict[str, dict[str, Any]] = {}
        self._active_tool_ids: set[str] = set()

        paths = [
            Path(path).resolve(strict=False)
            for path in (native_events_path, lifecycle_events_path, live_events_path)
            if path is not None
        ]
        if len(paths) != len(set(paths)):
            raise ValueError("native, lifecycle, and live JSONL paths must be distinct")
        self._native_writer = (
            _AppendOnlyJSONL(Path(native_events_path))
            if native_events_path is not None
            else None
        )
        self._lifecycle_writer = (
            _AppendOnlyJSONL(Path(lifecycle_events_path))
            if lifecycle_events_path is not None
            else None
        )
        self._live_writer = (
            _AppendOnlyJSONL(Path(live_events_path))
            if live_events_path is not None
            else None
        )

    def __enter__(self) -> "LifecycleCollector":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def _stamp(self) -> _Stamp:
        monotonic_s = float(self._monotonic_clock())
        time_s = float(self._wall_clock())
        if not math.isfinite(monotonic_s) or not math.isfinite(time_s):
            raise ValueError("collector clocks must return finite values")
        return _Stamp(time_s=time_s, monotonic_s=monotonic_s)

    def _ensure_open(self) -> None:
        if self._closed:
            raise LifecycleStateError("collector is closed")

    def _ensure_running(self) -> None:
        self._ensure_open()
        if self._terminal is not None:
            raise LifecycleStateError("frontend is already terminal")

    def _emit_lifecycle(
        self, event: str, stamp: _Stamp, **fields: Any
    ) -> dict[str, Any]:
        self._lifecycle_sequence += 1
        row = {
            "schema": LIFECYCLE_EVENT_SCHEMA,
            "sequence": self._lifecycle_sequence,
            "ts": stamp.time_s,
            "monotonic_s": stamp.monotonic_s,
            "event": event,
            "request_id": self.request.request_id,
            **fields,
        }
        detached = _json_copy(row)
        self._lifecycle_events.append(detached)
        if self._lifecycle_writer is not None:
            self._lifecycle_writer.write(detached)
        return detached

    def _write_live(self, row: Mapping[str, Any]) -> None:
        if self._live_writer is not None:
            self._live_writer.write(row)

    def record_native_event(
        self,
        raw: Mapping[str, Any],
        *,
        source: str,
        event_type: str | None = None,
        phase: str | None = None,
        native_session_id: str | None = None,
        native_timestamp_s: float | None = None,
    ) -> int:
        with self._lock:
            # Raw audit events may continue through process teardown.  They do
            # not alter the already-frozen frontend terminal timestamp.
            self._ensure_open()
            stamp = self._stamp()
            self._native_sequence += 1
            row = {
                "schema": NATIVE_EVENT_SCHEMA,
                "sequence": self._native_sequence,
                "request_id": self.request.request_id,
                "source": _nonempty(source, "source"),
                "event_type": event_type,
                "phase": phase,
                "native_session_id": native_session_id,
                "native_timestamp_s": _optional_finite(
                    native_timestamp_s, "native_timestamp_s"
                ),
                "observed_time_s": stamp.time_s,
                "observed_monotonic_s": stamp.monotonic_s,
                "raw": _json_copy(raw),
            }
            line = _strict_json_dumps(row)
            self._native_event_lines.append(line)
            if self._native_writer is not None:
                self._native_writer.write_line(line)
            return self._native_sequence

    def request_started(
        self,
        *,
        observed_time_s: float | None = None,
        observed_monotonic_s: float | None = None,
    ) -> None:
        with self._lock:
            self._ensure_running()
            if self._request_start is not None:
                raise LifecycleStateError("request_started was already emitted")
            if (observed_time_s is None) != (observed_monotonic_s is None):
                raise ValueError(
                    "observed_time_s and observed_monotonic_s must be supplied together"
                )
            if observed_time_s is None:
                stamp = self._stamp()
            else:
                wall = _optional_finite(observed_time_s, "observed_time_s")
                mono = _optional_finite(
                    observed_monotonic_s, "observed_monotonic_s"
                )
                assert wall is not None and mono is not None
                stamp = _Stamp(time_s=wall, monotonic_s=mono)
            # OpenHands publishes queue/e2e latency in the wall-clock domain.
            # Preserve monotonic observations for diagnostics, but use the same
            # public formula so cross-frontend means have identical boundaries.
            if (
                self.request.metadata.get("public_latency_clock") == "wall_openhands"
                and self.request.arrival_time_s is not None
            ):
                queue_wait_s = max(
                    0.0, stamp.time_s - float(self.request.arrival_time_s)
                )
            else:
                queue_wait_s = _duration(
                    self.request.arrival_monotonic_s, stamp.monotonic_s
                )
            self._request_start = {
                "time_s": stamp.time_s,
                "monotonic_s": stamp.monotonic_s,
                "queue_wait_s": queue_wait_s,
            }
            self._emit_lifecycle(
                "request_start", stamp, queue_wait_s=queue_wait_s
            )

    def frontend_started(
        self, *, native_session_id: str | None = None, phase: str = "initial"
    ) -> None:
        with self._lock:
            self._ensure_running()
            if self._frontend_start is not None:
                raise LifecycleStateError("frontend_started was already emitted")
            stamp = self._stamp()
            self._native_session_id = native_session_id or self._native_session_id
            self._frontend_start = {
                "time_s": stamp.time_s,
                "monotonic_s": stamp.monotonic_s,
                "phase": phase,
                "native_session_id": self._native_session_id,
            }
            self._emit_lifecycle(
                "frontend_start",
                stamp,
                phase=phase,
                native_session_id=self._native_session_id,
            )

    def llm_round_started(
        self,
        round_id: str,
        *,
        model: str | None = None,
        api_kind: str | None = None,
        phase: str = "initial",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._ensure_running()
            round_id = _nonempty(round_id, "round_id")
            if round_id in self._llm_rounds:
                raise LifecycleStateError(f"duplicate logical round: {round_id}")
            stamp = self._stamp()
            record = {
                "round_id": round_id,
                "llm_call_id": round_id,
                "model": model or self.request.model,
                "api_kind": api_kind,
                "phase": phase,
                "submit_time_s": stamp.time_s,
                "submit_monotonic_s": stamp.monotonic_s,
                "first_token_time_s": None,
                "first_token_monotonic_s": None,
                "end_time_s": None,
                "end_monotonic_s": None,
                "ttft_s": None,
                "duration_s": None,
                "status": "running",
                "native_response_id": None,
                "response_id": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "reasoning_tokens": None,
                "error": None,
                "metadata": _json_copy(metadata or {}),
            }
            self._llm_rounds[round_id] = record
            self._emit_lifecycle(
                "llm_round_start",
                stamp,
                round_id=round_id,
                model=record["model"],
                api_kind=api_kind,
                phase=phase,
            )

    def llm_round_first_token(self, round_id: str) -> None:
        with self._lock:
            self._ensure_running()
            record = self._llm_rounds.get(round_id)
            if record is None:
                raise LifecycleStateError(f"unknown logical round: {round_id}")
            if record["end_monotonic_s"] is not None:
                raise LifecycleStateError(f"logical round already ended: {round_id}")
            if record["first_token_monotonic_s"] is not None:
                raise LifecycleStateError(
                    f"first token already recorded for logical round: {round_id}"
                )
            stamp = self._stamp()
            record["first_token_time_s"] = stamp.time_s
            record["first_token_monotonic_s"] = stamp.monotonic_s
            record["ttft_s"] = _duration(
                record["submit_monotonic_s"], stamp.monotonic_s
            )
            self._emit_lifecycle(
                "llm_round_first_token",
                stamp,
                round_id=round_id,
                ttft_s=record["ttft_s"],
            )

    def llm_round_ended(
        self,
        round_id: str,
        *,
        status: str,
        native_response_id: str | None = None,
        error: str | None = None,
        usage: Mapping[str, int | None] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._ensure_running()
            record = self._llm_rounds.get(round_id)
            if record is None:
                raise LifecycleStateError(f"unknown logical round: {round_id}")
            if record["end_monotonic_s"] is not None:
                raise LifecycleStateError(f"logical round already ended: {round_id}")
            stamp = self._stamp()
            record["end_time_s"] = stamp.time_s
            record["end_monotonic_s"] = stamp.monotonic_s
            record["duration_s"] = _duration(
                record["submit_monotonic_s"], stamp.monotonic_s
            )
            record["status"] = _nonempty(status, "status")
            record["native_response_id"] = native_response_id
            record["response_id"] = native_response_id
            record["error"] = error
            if usage is not None:
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                ):
                    value = usage.get(key)
                    record[key] = None if value is None else int(value)
            if metadata:
                record["metadata"].update(_json_copy(metadata))
            self._emit_lifecycle(
                "llm_round_end",
                stamp,
                round_id=round_id,
                native_response_id=native_response_id,
                duration_s=record["duration_s"],
                status=record["status"],
                error=error,
            )

    def backend_attempt_started(
        self,
        attempt_id: str,
        *,
        round_id: str,
        protocol: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._ensure_running()
            attempt_id = _nonempty(attempt_id, "attempt_id")
            if attempt_id in self._backend_attempts:
                raise LifecycleStateError(f"duplicate backend attempt: {attempt_id}")
            if round_id not in self._llm_rounds:
                raise LifecycleStateError(
                    f"backend attempt references unknown logical round: {round_id}"
                )
            stamp = self._stamp()
            record = {
                "attempt_id": attempt_id,
                "backend_attempt_id": attempt_id,
                "round_id": round_id,
                "llm_call_id": round_id,
                "protocol": protocol,
                "start_time_s": stamp.time_s,
                "start_monotonic_s": stamp.monotonic_s,
                "end_time_s": None,
                "end_monotonic_s": None,
                "duration_s": None,
                "status": "running",
                "http_status": None,
                "error": None,
                "metadata": _json_copy(metadata or {}),
            }
            self._backend_attempts[attempt_id] = record
            self._emit_lifecycle(
                "backend_attempt_start",
                stamp,
                attempt_id=attempt_id,
                round_id=round_id,
                protocol=protocol,
            )

    def backend_attempt_ended(
        self,
        attempt_id: str,
        *,
        status: str,
        http_status: int | None = None,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            # The native frontend can publish its terminal notification after
            # consuming the final SSE event but just before the request proxy's
            # ``finally`` block records the physical HTTP attempt end.  The
            # public request clock is already frozen at that point, but the
            # existing backend attempt still needs its audit closure.  Allow
            # this terminal-only update after frontend completion; starts and
            # every other lifecycle mutation remain forbidden after terminal.
            record = self._backend_attempts.get(attempt_id)
            if record is None:
                raise LifecycleStateError(f"unknown backend attempt: {attempt_id}")
            if record["end_monotonic_s"] is not None:
                raise LifecycleStateError(f"backend attempt already ended: {attempt_id}")
            stamp = self._stamp()
            record["end_time_s"] = stamp.time_s
            record["end_monotonic_s"] = stamp.monotonic_s
            record["duration_s"] = _duration(
                record["start_monotonic_s"], stamp.monotonic_s
            )
            record["status"] = _nonempty(status, "status")
            record["http_status"] = None if http_status is None else int(http_status)
            record["error"] = error
            if metadata:
                record["metadata"].update(_json_copy(metadata))
            self._emit_lifecycle(
                "backend_attempt_end",
                stamp,
                attempt_id=attempt_id,
                round_id=record["round_id"],
                duration_s=record["duration_s"],
                status=record["status"],
                http_status=record["http_status"],
                error=error,
            )

    def _new_tool_record(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        parent_round_id: str | None,
        native_tool_call_id: str | None,
        phase: str,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "tool_call_id": tool_call_id,
            "action_id": tool_call_id,
            "native_tool_call_id": native_tool_call_id,
            "tool_name": _nonempty(tool_name, "tool_name"),
            "parent_round_id": parent_round_id,
            "llm_call_id": parent_round_id,
            "llm_response_id": None,
            "phase": phase,
            "enqueued_time_s": None,
            "enqueued_monotonic_s": None,
            "start_time_s": None,
            "start_monotonic_s": None,
            "end_time_s": None,
            "end_monotonic_s": None,
            "wait_s": None,
            "duration_s": None,
            "status": "observed",
            "error": None,
            "metadata": _json_copy(metadata or {}),
        }

    def tool_queued(
        self,
        tool_call_id: str,
        *,
        tool_name: str,
        parent_round_id: str | None = None,
        native_tool_call_id: str | None = None,
        phase: str = "initial",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._ensure_running()
            tool_call_id = _nonempty(tool_call_id, "tool_call_id")
            if tool_call_id in self._tools:
                raise LifecycleStateError(f"duplicate tool call: {tool_call_id}")
            stamp = self._stamp()
            record = self._new_tool_record(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                parent_round_id=parent_round_id,
                native_tool_call_id=native_tool_call_id,
                phase=phase,
                metadata=metadata,
            )
            record["enqueued_time_s"] = stamp.time_s
            record["enqueued_monotonic_s"] = stamp.monotonic_s
            record["status"] = "queued"
            self._tools[tool_call_id] = record
            self._emit_lifecycle(
                "tool_enqueued",
                stamp,
                tool_call_id=tool_call_id,
                native_tool_call_id=native_tool_call_id,
                tool_name=record["tool_name"],
                parent_round_id=parent_round_id,
                phase=phase,
            )

    def tool_started(
        self,
        tool_call_id: str,
        *,
        tool_name: str | None = None,
        parent_round_id: str | None = None,
        native_tool_call_id: str | None = None,
        phase: str = "initial",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._ensure_running()
            tool_call_id = _nonempty(tool_call_id, "tool_call_id")
            record = self._tools.get(tool_call_id)
            if record is None:
                if tool_name is None:
                    raise LifecycleStateError(
                        f"tool_name is required when start has no queue event: {tool_call_id}"
                    )
                record = self._new_tool_record(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    parent_round_id=parent_round_id,
                    native_tool_call_id=native_tool_call_id,
                    phase=phase,
                    metadata=metadata,
                )
                self._tools[tool_call_id] = record
            if record["start_monotonic_s"] is not None:
                raise LifecycleStateError(f"tool already started: {tool_call_id}")
            if record["end_monotonic_s"] is not None:
                raise LifecycleStateError(f"tool already ended: {tool_call_id}")
            stamp = self._stamp()
            record["start_time_s"] = stamp.time_s
            record["start_monotonic_s"] = stamp.monotonic_s
            record["wait_s"] = _duration(
                record["enqueued_monotonic_s"], stamp.monotonic_s
            )
            record["status"] = "running"
            if parent_round_id is not None:
                record["parent_round_id"] = parent_round_id
                record["llm_call_id"] = parent_round_id
            if native_tool_call_id is not None:
                record["native_tool_call_id"] = native_tool_call_id
            if metadata:
                record["metadata"].update(_json_copy(metadata))
            self._active_tool_ids.add(tool_call_id)
            lifecycle_row = self._emit_lifecycle(
                "tool_start",
                stamp,
                tool_call_id=tool_call_id,
                native_tool_call_id=record["native_tool_call_id"],
                tool_name=record["tool_name"],
                parent_round_id=record["parent_round_id"],
                phase=record["phase"],
                wait_s=record["wait_s"],
            )
            # This is the compatibility seam consumed by LiveToolStats.  The
            # write is flushed before this method returns.
            self._write_live(
                {
                    "ts": lifecycle_row["ts"],
                    "event": "tool_start",
                    "request_id": self.request.request_id,
                    "phase": record["phase"],
                    "tool_call_id": tool_call_id,
                    "native_tool_call_id": record["native_tool_call_id"],
                    "tool_name": record["tool_name"],
                    "wait_s": record["wait_s"],
                }
            )

    def tool_ended(
        self,
        tool_call_id: str,
        *,
        status: str,
        tool_name: str | None = None,
        parent_round_id: str | None = None,
        native_tool_call_id: str | None = None,
        error: str | None = None,
        phase: str = "initial",
        metadata: Mapping[str, Any] | None = None,
        use_enqueue_as_start_if_missing: bool = False,
    ) -> None:
        with self._lock:
            self._ensure_running()
            tool_call_id = _nonempty(tool_call_id, "tool_call_id")
            record = self._tools.get(tool_call_id)
            if record is None:
                if tool_name is None:
                    raise LifecycleStateError(
                        f"tool_name is required when end has no prior event: {tool_call_id}"
                    )
                record = self._new_tool_record(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    parent_round_id=parent_round_id,
                    native_tool_call_id=native_tool_call_id,
                    phase=phase,
                    metadata=metadata,
                )
                self._tools[tool_call_id] = record
            if record["end_monotonic_s"] is not None:
                raise LifecycleStateError(f"tool already ended: {tool_call_id}")
            stamp = self._stamp()
            if (
                use_enqueue_as_start_if_missing
                and record["start_monotonic_s"] is None
            ):
                # Some native frameworks expose function-call and
                # function-call-output but no separate execution-start event
                # for tiny in-process tools.  In that case the only measured
                # execution envelope is enqueue-to-output.  Treat enqueue as
                # the execution start (there is no separately exposed native
                # queue) and label that inferred boundary explicitly instead
                # of silently leaving a terminal tool queued forever.
                record["start_time_s"] = (
                    record["enqueued_time_s"]
                    if record["enqueued_time_s"] is not None
                    else stamp.time_s
                )
                record["start_monotonic_s"] = (
                    record["enqueued_monotonic_s"]
                    if record["enqueued_monotonic_s"] is not None
                    else stamp.monotonic_s
                )
                record["wait_s"] = 0.0
                record["metadata"]["timing_boundary"] = (
                    "native_function_call_to_output_no_start_event"
                )
                record["metadata"]["wait_boundary"] = (
                    "no_separate_native_queue_assume_start_at_function_call"
                )
            record["end_time_s"] = stamp.time_s
            record["end_monotonic_s"] = stamp.monotonic_s
            record["duration_s"] = _duration(
                record["start_monotonic_s"], stamp.monotonic_s
            )
            record["status"] = _nonempty(status, "status")
            record["error"] = error
            if parent_round_id is not None:
                record["parent_round_id"] = parent_round_id
                record["llm_call_id"] = parent_round_id
            if native_tool_call_id is not None:
                record["native_tool_call_id"] = native_tool_call_id
            if metadata:
                record["metadata"].update(_json_copy(metadata))
            self._active_tool_ids.discard(tool_call_id)
            lifecycle_row = self._emit_lifecycle(
                "tool_end",
                stamp,
                tool_call_id=tool_call_id,
                native_tool_call_id=record["native_tool_call_id"],
                tool_name=record["tool_name"],
                parent_round_id=record["parent_round_id"],
                phase=record["phase"],
                duration_s=record["duration_s"],
                status=record["status"],
                error=error,
            )
            self._write_live(
                {
                    "ts": lifecycle_row["ts"],
                    "event": "tool_end",
                    "request_id": self.request.request_id,
                    "phase": record["phase"],
                    "tool_call_id": tool_call_id,
                    "native_tool_call_id": record["native_tool_call_id"],
                    "tool_name": record["tool_name"],
                    "duration_s": record["duration_s"],
                    "status": record["status"],
                    "error": error,
                }
            )

    def frontend_terminal(
        self,
        status: FrontendTerminalStatus | str,
        *,
        reason: str | None = None,
        native_session_id: str | None = None,
        final_output: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FrontendOutcome:
        with self._lock:
            self._ensure_open()
            if self._terminal is not None:
                raise LifecycleStateError("frontend terminal event was already emitted")
            terminal_status = (
                status
                if isinstance(status, FrontendTerminalStatus)
                else FrontendTerminalStatus(status)
            )
            if terminal_status is FrontendTerminalStatus.COMPLETED:
                unresolved_tools = sorted(
                    tool_id
                    for tool_id, record in self._tools.items()
                    if record.get("end_monotonic_s") is None
                )
                if unresolved_tools:
                    raise LifecycleStateError(
                        "completed frontend has unresolved tools: "
                        + ", ".join(unresolved_tools)
                    )
            stamp = self._stamp()
            self._native_session_id = native_session_id or self._native_session_id
            self._terminal = {
                "status": terminal_status.value,
                "reason": reason,
                "native_session_id": self._native_session_id,
                "final_output": final_output,
                "time_s": stamp.time_s,
                "monotonic_s": stamp.monotonic_s,
                "metadata": _json_copy(metadata or {}),
            }
            self._emit_lifecycle(
                "frontend_terminal",
                stamp,
                status=terminal_status.value,
                reason=reason,
                native_session_id=self._native_session_id,
                active_tool_ids=sorted(self._active_tool_ids),
            )
            return FrontendOutcome(
                request_id=self.request.request_id,
                status=terminal_status,
                terminal_monotonic_s=stamp.monotonic_s,
                terminal_time_s=stamp.time_s,
                native_session_id=self._native_session_id,
                reason=reason,
                final_output=final_output,
                metadata=metadata or {},
            )

    def _metrics_locked(self) -> dict[str, Any]:
        terminal_monotonic_s = (
            self._terminal.get("monotonic_s") if self._terminal is not None else None
        )
        frontend_monotonic_s = (
            self._frontend_start.get("monotonic_s")
            if self._frontend_start is not None
            else None
        )
        queue_wait_s = (
            self._request_start.get("queue_wait_s")
            if self._request_start is not None
            else None
        )
        e2e_latency_monotonic_s = _duration(
            self.request.arrival_monotonic_s, terminal_monotonic_s
        )
        terminal_time_s = (
            self._terminal.get("time_s") if self._terminal is not None else None
        )
        if (
            self.request.metadata.get("public_latency_clock") == "wall_openhands"
            and self.request.arrival_time_s is not None
            and terminal_time_s is not None
        ):
            e2e_latency_s = max(
                0.0, float(terminal_time_s) - float(self.request.arrival_time_s)
            )
        else:
            e2e_latency_s = e2e_latency_monotonic_s
        frontend_latency_s = _duration(frontend_monotonic_s, terminal_monotonic_s)
        llm_records = list(self._llm_rounds.values())
        attempt_records = list(self._backend_attempts.values())
        tool_records = list(self._tools.values())
        gpu_total_s = _complete_sum(llm_records, "duration_s")
        backend_attempt_total_s = _complete_sum(attempt_records, "duration_s")
        tool_total_s = _complete_sum(tool_records, "duration_s")
        tool_wait_total_s = _complete_sum(tool_records, "wait_s")
        wait_total_s = (
            float(queue_wait_s) + float(tool_wait_total_s)
            if queue_wait_s is not None and tool_wait_total_s is not None
            else None
        )
        return {
            "queue_wait_s": queue_wait_s,
            "e2e_latency_s": e2e_latency_s,
            "e2e_latency_monotonic_s": e2e_latency_monotonic_s,
            "frontend_latency_s": frontend_latency_s,
            "gpu_total_s": gpu_total_s,
            "backend_attempt_total_s": backend_attempt_total_s,
            "tool_total_s": tool_total_s,
            "tool_wait_total_s": tool_wait_total_s,
            "wait_total_s": wait_total_s,
            "llm_call_count": len(llm_records),
            "backend_attempt_count": len(attempt_records),
            "tool_call_count": len(tool_records),
            "llm_duration_unknown_count": sum(
                record["duration_s"] is None for record in llm_records
            ),
            "backend_attempt_duration_unknown_count": sum(
                record["duration_s"] is None for record in attempt_records
            ),
            "tool_duration_unknown_count": sum(
                record["duration_s"] is None for record in tool_records
            ),
            "tool_wait_unknown_count": sum(
                record["wait_s"] is None for record in tool_records
            ),
            "active_tool_count": len(self._active_tool_ids),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return a detached point-in-time view safe for concurrent readers."""

        with self._lock:
            snapshot = {
                "request_id": self.request.request_id,
                "job_id": self.request.job_id,
                "request_start": self._request_start,
                "frontend_start": self._frontend_start,
                "terminal": self._terminal,
                "native_session_id": self._native_session_id,
                "logical_llm_rounds": list(self._llm_rounds.values()),
                "backend_attempts": list(self._backend_attempts.values()),
                "tool_calls": list(self._tools.values()),
                "active_tool_ids": sorted(self._active_tool_ids),
                "lifecycle_events": self._lifecycle_events,
                "native_events": [
                    json.loads(line) for line in self._native_event_lines
                ],
                "metrics": self._metrics_locked(),
            }
            return copy.deepcopy(snapshot)

    def snapshot_openhands_trace(self) -> dict[str, Any]:
        """Build an OpenHands-shaped trace without collapsing attempt semantics."""

        with self._lock:
            llm_calls = copy.deepcopy(list(self._llm_rounds.values()))
            backend_attempts = copy.deepcopy(list(self._backend_attempts.values()))
            tool_calls = copy.deepcopy(list(self._tools.values()))
            response_by_round = {
                str(call.get("round_id")): call.get("native_response_id")
                for call in llm_calls
            }
            tools_by_round: dict[str, list[dict[str, Any]]] = {}
            for tool in tool_calls:
                parent = tool.get("parent_round_id")
                if isinstance(parent, str) and parent:
                    tools_by_round.setdefault(parent, []).append(tool)
                tool.setdefault("tool_kind", tool.get("tool_name"))
                tool.setdefault("summary", None)
                tool["llm_response_id"] = (
                    response_by_round.get(parent)
                    if isinstance(parent, str)
                    else None
                )
                metadata = tool.get("metadata")
                state = metadata.get("native_state") if isinstance(metadata, Mapping) else None
                output = state.get("output") if isinstance(state, Mapping) else None
                preview = None
                if output is not None:
                    preview = str(output)
                    if len(preview) > 1000:
                        preview = preview[:1000] + "..."
                tool.setdefault("observation_preview", preview)
            for call in llm_calls:
                round_id = str(call.get("round_id"))
                call.setdefault("message_count", None)
                call["tool_count"] = len(tools_by_round.get(round_id, ()))
            rounds: list[dict[str, Any]] = []
            known_round_ids: set[str] = set()
            for index, llm_call in enumerate(llm_calls):
                round_id = str(llm_call["round_id"])
                known_round_ids.add(round_id)
                rounds.append(
                    {
                        "round_index": index,
                        "round_id": round_id,
                        "llm": llm_call,
                        "backend_attempts": [
                            attempt
                            for attempt in backend_attempts
                            if attempt.get("round_id") == round_id
                        ],
                        "tools": [
                            tool
                            for tool in tool_calls
                            if tool.get("parent_round_id") == round_id
                        ],
                    }
                )
            orphan_tools = [
                tool
                for tool in tool_calls
                if tool.get("parent_round_id") not in known_round_ids
            ]
            metrics = self._metrics_locked()
            terminal = copy.deepcopy(self._terminal)
            start = copy.deepcopy(self._request_start)
            status = terminal.get("status") if terminal is not None else "running"
            reason = terminal.get("reason") if terminal is not None else None
            trace = {
                "request_id": self.request.request_id,
                "dataset_request_id": self.request.dataset_request_id,
                "benchmark": self.request.benchmark,
                "task_id": self.request.task_id,
                "workspace_dir": str(self.request.workspace_dir),
                "arrival_time_s": self.request.arrival_time_s,
                "start_time_s": start.get("time_s") if start is not None else None,
                "end_time_s": terminal.get("time_s") if terminal is not None else None,
                **metrics,
                "status": status,
                "conversation_status": status,
                "native_session_id": self._native_session_id,
                "error": (
                    None
                    if status
                    in {
                        FrontendTerminalStatus.COMPLETED.value,
                        FrontendTerminalStatus.NEEDS_CONFIRMATION.value,
                    }
                    else ({"type": "FrontendTerminal", "message": reason} if reason else None)
                ),
                "llm_calls": llm_calls,
                "backend_attempts": backend_attempts,
                "tool_calls": tool_calls,
                "rounds": rounds,
                "orphan_tools": orphan_tools,
                "terminal": terminal,
            }
            return copy.deepcopy(trace)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for writer in (
                self._native_writer,
                self._lifecycle_writer,
                self._live_writer,
            ):
                if writer is not None:
                    writer.close()
            self._closed = True


__all__ = [
    "LIFECYCLE_EVENT_SCHEMA",
    "NATIVE_EVENT_SCHEMA",
    "LifecycleCollector",
    "LifecycleStateError",
]
