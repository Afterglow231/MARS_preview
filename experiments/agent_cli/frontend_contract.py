#!/usr/bin/env python3

"""Frontend-neutral contract for agent lifecycle adapters.

The scheduler owns request arrival and admission.  An ``AgentFrontend`` owns
the native agent session, its logical model/tool loop, and the terminal
decision.  The model server remains a backend and may make more than one
transport attempt for a single logical model round.

This module intentionally contains no Codex, OpenCode, or OpenHands imports so
it can be copied into the isolated per-request runtime bundle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


class FrontendTerminalStatus(str, Enum):
    """Terminal states owned by the native frontend, not by its subprocess."""

    COMPLETED = "completed"
    ERROR = "error"
    NEEDS_CONFIRMATION = "needs_confirmation"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    MAX_ITERATIONS = "max_iterations"
    INTERRUPTED = "interrupted"


def _require_nonempty(value: str, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _finite_optional(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite when provided")
    return number


def _freeze(value: Any) -> Any:
    """Make caller-owned configuration containers read-only by construction."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class FrontendCapabilities:
    """Declares which native signals an adapter can emit in real time."""

    name: str
    version: str | None = None
    streams_native_events: bool = True
    streams_logical_llm_rounds: bool = True
    streams_backend_attempts: bool = True
    streams_tool_queue: bool = False
    streams_tool_execution: bool = True
    streams_terminal_status: bool = True
    supports_session_resume: bool = True
    supports_parallel_tools: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_nonempty(self.name, "name"))


@dataclass(frozen=True, slots=True)
class AgentRequestContext:
    """Immutable request data shared by every frontend implementation.

    ``arrival_time_s`` is wall-clock time for persisted traces.
    ``arrival_monotonic_s`` is the scheduler's monotonic observation and is the
    only valid basis for queue/e2e durations.  Either may be unavailable for a
    replay; derived metrics then remain ``None`` rather than using a fabricated
    timestamp.
    """

    request_id: str
    job_id: str
    prompt: str
    workspace_dir: Path
    model: str
    backend_base_url: str
    arrival_time_s: float | None = None
    arrival_monotonic_s: float | None = None
    dataset_request_id: str | None = None
    benchmark: str | None = None
    task_id: str | None = None
    readonly_paths: Sequence[Path] = ()
    max_iterations: int | None = None
    deadline_monotonic_s: float | None = None
    runtime_env: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_id", _require_nonempty(self.request_id, "request_id")
        )
        object.__setattr__(self, "job_id", _require_nonempty(self.job_id, "job_id"))
        object.__setattr__(self, "model", _require_nonempty(self.model, "model"))
        object.__setattr__(
            self,
            "backend_base_url",
            _require_nonempty(self.backend_base_url, "backend_base_url"),
        )
        object.__setattr__(self, "workspace_dir", Path(self.workspace_dir))
        object.__setattr__(
            self, "readonly_paths", tuple(Path(path) for path in self.readonly_paths)
        )
        object.__setattr__(
            self,
            "arrival_time_s",
            _finite_optional(self.arrival_time_s, "arrival_time_s"),
        )
        object.__setattr__(
            self,
            "arrival_monotonic_s",
            _finite_optional(self.arrival_monotonic_s, "arrival_monotonic_s"),
        )
        object.__setattr__(
            self,
            "deadline_monotonic_s",
            _finite_optional(self.deadline_monotonic_s, "deadline_monotonic_s"),
        )
        if self.max_iterations is not None and int(self.max_iterations) <= 0:
            raise ValueError("max_iterations must be positive when provided")
        if self.max_iterations is not None:
            object.__setattr__(self, "max_iterations", int(self.max_iterations))
        object.__setattr__(self, "runtime_env", _freeze(dict(self.runtime_env)))
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class FrontendOutcome:
    """Native terminal outcome captured before artifact/container cleanup."""

    request_id: str
    status: FrontendTerminalStatus
    terminal_monotonic_s: float
    terminal_time_s: float
    native_session_id: str | None = None
    reason: str | None = None
    final_output: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_id", _require_nonempty(self.request_id, "request_id")
        )
        if not isinstance(self.status, FrontendTerminalStatus):
            object.__setattr__(self, "status", FrontendTerminalStatus(self.status))
        terminal_monotonic_s = _finite_optional(
            self.terminal_monotonic_s, "terminal_monotonic_s"
        )
        terminal_time_s = _finite_optional(self.terminal_time_s, "terminal_time_s")
        assert terminal_monotonic_s is not None
        assert terminal_time_s is not None
        object.__setattr__(self, "terminal_monotonic_s", terminal_monotonic_s)
        object.__setattr__(self, "terminal_time_s", terminal_time_s)
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))


@runtime_checkable
class AgentEventSink(Protocol):
    """Real-time event sink implemented by ``LifecycleCollector``."""

    def record_native_event(
        self,
        raw: Mapping[str, Any],
        *,
        source: str,
        event_type: str | None = None,
        phase: str | None = None,
        native_session_id: str | None = None,
        native_timestamp_s: float | None = None,
    ) -> int: ...

    def request_started(self) -> None: ...

    def frontend_started(
        self, *, native_session_id: str | None = None, phase: str = "initial"
    ) -> None: ...

    def llm_round_started(
        self,
        round_id: str,
        *,
        model: str | None = None,
        api_kind: str | None = None,
        phase: str = "initial",
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...

    def llm_round_ended(
        self,
        round_id: str,
        *,
        status: str,
        native_response_id: str | None = None,
        error: str | None = None,
        usage: Mapping[str, int | None] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...

    def llm_round_first_token(self, round_id: str) -> None: ...

    def backend_attempt_started(
        self,
        attempt_id: str,
        *,
        round_id: str,
        protocol: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...

    def backend_attempt_ended(
        self,
        attempt_id: str,
        *,
        status: str,
        http_status: int | None = None,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...

    def tool_queued(
        self,
        tool_call_id: str,
        *,
        tool_name: str,
        parent_round_id: str | None = None,
        native_tool_call_id: str | None = None,
        phase: str = "initial",
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...

    def tool_started(
        self,
        tool_call_id: str,
        *,
        tool_name: str | None = None,
        parent_round_id: str | None = None,
        native_tool_call_id: str | None = None,
        phase: str = "initial",
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...

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
    ) -> None: ...

    def frontend_terminal(
        self,
        status: FrontendTerminalStatus | str,
        *,
        reason: str | None = None,
        native_session_id: str | None = None,
        final_output: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FrontendOutcome: ...


@runtime_checkable
class AgentFrontend(Protocol):
    """Frontend adapter that owns one complete native agent lifecycle."""

    capabilities: FrontendCapabilities

    def run(
        self, request: AgentRequestContext, events: AgentEventSink
    ) -> FrontendOutcome: ...

    def cancel(self, reason: str) -> None: ...

    def close(self) -> None: ...


__all__ = [
    "AgentEventSink",
    "AgentFrontend",
    "AgentRequestContext",
    "FrontendCapabilities",
    "FrontendOutcome",
    "FrontendTerminalStatus",
]
