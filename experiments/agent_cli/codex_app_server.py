#!/usr/bin/env python3
"""Native Codex app-server stdio driver for the MARS agent baseline.

This module deliberately stops at Codex's public app-server boundary.  Codex
owns the thread, turn, model/tool loop, conversation history, and final turn
state.  The driver only:

* transports JSON-RPC messages over the app-server's JSONL stdio protocol;
* exposes native notifications to a live observer;
* waits for the matching ``turn/completed`` notification;
* requests ``turn/interrupt`` when the evaluator deadline expires; and
* preserves the exact app-server stdout stream as an immutable JSONL artifact.

It contains no Responses/Chat translation and no inferred agent state.
"""

from __future__ import annotations

import copy
import json
import os
import queue
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, TypeAlias


JsonObject: TypeAlias = dict[str, Any]
EventCallback: TypeAlias = Callable[[JsonObject, float], Any]
ServerRequestHandler: TypeAlias = Callable[[str, Mapping[str, Any], Mapping[str, Any]], Any]


class CodexAppServerError(RuntimeError):
    """Base error for transport and lifecycle failures."""


class CodexProtocolError(CodexAppServerError):
    """The app-server emitted an invalid or internally inconsistent envelope."""


class CodexProcessExited(CodexAppServerError):
    """The process exited before the requested native lifecycle event arrived."""

    def __init__(self, returncode: int | None, stderr_tail: str = "") -> None:
        message = f"Codex app-server exited before native turn completion (returncode={returncode})"
        if stderr_tail:
            message += f"; stderr tail: {stderr_tail}"
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


class CodexRpcError(CodexAppServerError):
    """A JSON-RPC response contained an error object."""

    def __init__(
        self,
        *,
        method: str,
        request_id: int | str,
        code: int | None,
        message: str,
        data: Any = None,
    ) -> None:
        super().__init__(f"Codex RPC {method!r} failed ({code}): {message}")
        self.method = method
        self.request_id = request_id
        self.code = code
        self.rpc_message = message
        self.data = data


class CodexRpcTimeout(CodexAppServerError):
    """A JSON-RPC request did not receive its matching response in time."""

    def __init__(self, method: str, request_id: int | str, timeout_s: float) -> None:
        super().__init__(
            f"Timed out after {timeout_s:.3f}s waiting for Codex RPC "
            f"{method!r} response id={request_id!r}"
        )
        self.method = method
        self.request_id = request_id
        self.timeout_s = timeout_s


class CodexTurnTimeout(CodexAppServerError):
    """No matching ``turn/completed`` arrived after interrupt grace elapsed."""

    def __init__(
        self,
        *,
        thread_id: str,
        turn_id: str,
        timeout_s: float,
        interrupt_grace_s: float,
        interrupt_error: BaseException | None = None,
        trigger: str = "turn deadline",
    ) -> None:
        message = (
            f"Codex turn {turn_id!r} on thread {thread_id!r} reached {trigger} "
            f"and emitted no matching turn/completed within "
            f"{interrupt_grace_s:.3f}s after turn/interrupt"
        )
        if interrupt_error is not None:
            message += f"; interrupt RPC error: {interrupt_error}"
        super().__init__(message)
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.timeout_s = timeout_s
        self.interrupt_grace_s = interrupt_grace_s
        self.interrupt_error = interrupt_error
        self.trigger = trigger


class CodexEventCallbackError(CodexAppServerError):
    """The live event observer failed; tracing loss is never silently ignored."""


class CodexServerRequestRejected(Exception):
    """Ask the driver to reject a server-initiated JSON-RPC request."""

    def __init__(self, message: str, *, code: int = -32601, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


@dataclass(frozen=True, slots=True)
class CodexAppServerEvent:
    """One unmodified server notification and its host receive timestamp."""

    captured_at_s: float
    message: JsonObject

    @property
    def method(self) -> str:
        return str(self.message.get("method") or "")


@dataclass(frozen=True, slots=True)
class CodexTurnResult:
    """Native terminal state for exactly one app-server turn."""

    phase: str
    thread_id: str
    turn_id: str
    status: str
    turn: JsonObject
    events: tuple[CodexAppServerEvent, ...]
    started_at_s: float
    completed_at_s: float
    timed_out: bool
    iteration_limited: bool
    interrupt_requested: bool
    interrupt_error: str | None = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.completed_at_s - self.started_at_s)

    @property
    def error(self) -> Any:
        return self.turn.get("error")

    @property
    def raw_response_count(self) -> int:
        return sum(event.method == "rawResponse/completed" for event in self.events)


@dataclass(frozen=True, slots=True)
class CodexConversationResult:
    """Initial and optional verification turns from one native thread."""

    thread_id: str
    initial: CodexTurnResult
    verification: CodexTurnResult | None


@dataclass(frozen=True, slots=True)
class _RpcReply:
    message: JsonObject


@dataclass(frozen=True, slots=True)
class _FatalReply:
    error: BaseException


@dataclass(frozen=True, slots=True)
class _TerminalTurn:
    message: JsonObject
    captured_at_s: float


class _TurnWaitDeadline(Exception):
    pass


def build_codex_app_server_command(
    binary: str | os.PathLike[str],
    *,
    config_overrides: Sequence[str] = (),
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build the pinned CLI's native stdio app-server command."""

    command = [str(binary), "app-server"]
    for override in config_overrides:
        command.extend(["--config", str(override)])
    command.extend(["--listen", "stdio://"])
    command.extend(str(arg) for arg in extra_args)
    return command


class CodexAppServerDriver:
    """Synchronous, thread-safe controller for one Codex app-server process.

    A dedicated reader thread drains stdout continuously.  ``event_callback``
    runs on that reader thread and must be quick and non-blocking.  It receives
    the exact notification object and the host receive timestamp.  Callback
    failures are fatal so a formal benchmark cannot silently lose trace data.

    One driver intentionally owns one native thread.  Calls to ``run_turn`` and
    ``run_verification_turn`` therefore append to the same Codex conversation.
    Concurrent turns are rejected.
    """

    _TERMINAL_STATUSES = {"completed", "interrupted", "failed"}

    def __init__(
        self,
        *,
        command: Sequence[str],
        raw_jsonl_path: str | os.PathLike[str],
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        stderr_path: str | os.PathLike[str] | None = None,
        wire_audit_path: str | os.PathLike[str] | None = None,
        event_callback: EventCallback | None = None,
        server_request_handler: ServerRequestHandler | None = None,
        rpc_timeout_s: float = 30.0,
        interrupt_grace_s: float = 10.0,
        shutdown_timeout_s: float = 3.0,
        client_name: str = "mars_agent_baseline",
        client_title: str = "MARS Agent Baseline",
        client_version: str = "1",
        experimental_api: bool = True,
        overwrite_logs: bool = False,
        start_new_session: bool = True,
    ) -> None:
        if not command:
            raise ValueError("Codex app-server command must not be empty")
        if rpc_timeout_s <= 0:
            raise ValueError("rpc_timeout_s must be positive")
        if interrupt_grace_s <= 0:
            raise ValueError("interrupt_grace_s must be positive")
        if shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")

        self.command = tuple(str(part) for part in command)
        self.cwd = str(cwd) if cwd is not None else None
        self.env = dict(env) if env is not None else None
        self.raw_jsonl_path = Path(raw_jsonl_path)
        self.stderr_path = Path(stderr_path) if stderr_path is not None else None
        self.wire_audit_path = (
            Path(wire_audit_path) if wire_audit_path is not None else None
        )
        self.event_callback = event_callback
        self.server_request_handler = server_request_handler
        self.rpc_timeout_s = float(rpc_timeout_s)
        self.interrupt_grace_s = float(interrupt_grace_s)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self.client_info = {
            "name": client_name,
            "title": client_title,
            "version": client_version,
        }
        self.experimental_api = bool(experimental_api)
        self.overwrite_logs = bool(overwrite_logs)
        self.start_new_session = bool(start_new_session)

        self._process: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._raw_fp: TextIO | None = None
        self._stderr_fp: TextIO | None = None
        self._wire_fp: TextIO | None = None
        self._stderr_tail: deque[str] = deque(maxlen=80)

        self._write_lock = threading.Lock()
        self._wire_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._state_condition = threading.Condition()
        self._active_turn_lock = threading.Lock()
        self._pending: dict[int | str, queue.Queue[_RpcReply | _FatalReply]] = {}
        self._expired_request_ids: set[int | str] = set()
        self._events_by_turn: dict[tuple[str, str], list[CodexAppServerEvent]] = {}
        self._raw_response_counts: dict[tuple[str, str], int] = {}
        self._terminal_turns: dict[tuple[str, str], _TerminalTurn] = {}
        self._unexpected_server_requests: list[JsonObject] = []

        self._next_request_id = 0
        self._wire_sequence = 0
        self._fatal_error: BaseException | None = None
        self._callback_error: BaseException | None = None
        self._started = False
        self._closing = False
        self._closed = False
        self.initialize_result: JsonObject | None = None
        self.thread_start_result: JsonObject | None = None
        self.thread_id: str | None = None

    @property
    def process(self) -> subprocess.Popen[str] | None:
        return self._process

    @property
    def unexpected_server_requests(self) -> tuple[JsonObject, ...]:
        return tuple(copy.deepcopy(self._unexpected_server_requests))

    def __enter__(self) -> "CodexAppServerDriver":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()

    def _open_log(self, path: Path) -> TextIO:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if self.overwrite_logs else "x"
        return path.open(mode, encoding="utf-8", buffering=1)

    def start(self) -> JsonObject:
        """Launch app-server and complete initialize/initialized once."""

        if self._closed:
            raise CodexAppServerError("Codex app-server driver is already closed")
        if self._started:
            assert self.initialize_result is not None
            return copy.deepcopy(self.initialize_result)

        self._raw_fp = self._open_log(self.raw_jsonl_path)
        try:
            if self.stderr_path is not None:
                self._stderr_fp = self._open_log(self.stderr_path)
            if self.wire_audit_path is not None:
                self._wire_fp = self._open_log(self.wire_audit_path)
            self._process = subprocess.Popen(
                list(self.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self.cwd,
                env=self.env,
                start_new_session=self.start_new_session,
            )
        except BaseException:
            self._close_log_files()
            raise

        self._started = True
        self._stdout_thread = threading.Thread(
            target=self._stdout_reader,
            name=f"codex-app-server-stdout-{self._process.pid}",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_reader,
            name=f"codex-app-server-stderr-{self._process.pid}",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        initialize_params: JsonObject = {"clientInfo": dict(self.client_info)}
        if self.experimental_api:
            initialize_params["capabilities"] = {"experimentalApi": True}
        try:
            result = self._request(
                "initialize", initialize_params, timeout_s=self.rpc_timeout_s
            )
            self._send_notification("initialized")
        except BaseException:
            self.close()
            raise
        if not isinstance(result, dict):
            self.close()
            raise CodexProtocolError("initialize result must be a JSON object")
        self.initialize_result = copy.deepcopy(result)
        return copy.deepcopy(result)

    def start_thread(
        self,
        *,
        model: str | None = None,
        model_provider: str | None = None,
        cwd: str | os.PathLike[str] | None = None,
        approval_policy: str | None = "never",
        sandbox: str | None = None,
        permissions: str | None = None,
        ephemeral: bool = True,
        experimental_raw_events: bool = True,
        extra_params: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> JsonObject:
        """Create the single native Codex thread owned by this driver."""

        self.start()
        if self.thread_id is not None:
            raise CodexAppServerError(
                f"driver already owns Codex thread {self.thread_id!r}"
            )
        if sandbox is not None and permissions is not None:
            raise ValueError("thread/start cannot combine sandbox and permissions")

        params: JsonObject = dict(extra_params or {})
        if model is not None:
            params["model"] = model
        if model_provider is not None:
            params["modelProvider"] = model_provider
        if cwd is not None:
            params["cwd"] = str(cwd)
        if approval_policy is not None:
            params["approvalPolicy"] = approval_policy
        if sandbox is not None:
            params["sandbox"] = sandbox
        if permissions is not None:
            params["permissions"] = permissions
        params["ephemeral"] = bool(ephemeral)
        if experimental_raw_events:
            if not self.experimental_api:
                raise ValueError(
                    "experimental_raw_events requires initialize experimentalApi=true"
                )
            params["experimentalRawEvents"] = True

        result = self._request(
            "thread/start",
            params,
            timeout_s=self.rpc_timeout_s if timeout_s is None else float(timeout_s),
        )
        thread = result.get("thread") if isinstance(result, dict) else None
        thread_id = thread.get("id") if isinstance(thread, dict) else None
        if not isinstance(thread_id, str) or not thread_id:
            raise CodexProtocolError(
                "thread/start result is missing non-empty result.thread.id"
            )
        self.thread_id = thread_id
        self.thread_start_result = copy.deepcopy(result)
        return copy.deepcopy(result)

    def run_turn(
        self,
        input_value: str | Sequence[Mapping[str, Any]],
        *,
        phase: str = "initial",
        responsesapi_client_metadata: Mapping[str, str] | None = None,
        client_user_message_id: str | None = None,
        timeout_s: float,
        max_raw_responses: int | None = None,
        interrupt_grace_s: float | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> CodexTurnResult:
        """Run one native turn and wait only for its matching turn/completed.

        If ``timeout_s`` expires, the driver sends ``turn/interrupt`` and still
        waits for the matching native terminal notification.  It never invents
        a completion status from the process return code or timeout itself.
        """

        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.thread_id is None:
            raise CodexAppServerError("start_thread must complete before run_turn")
        if not self._active_turn_lock.acquire(blocking=False):
            raise CodexAppServerError("concurrent turns on one Codex thread are unsupported")
        try:
            return self._run_turn_locked(
                input_value,
                phase=phase,
                responsesapi_client_metadata=responsesapi_client_metadata,
                client_user_message_id=client_user_message_id,
                timeout_s=float(timeout_s),
                max_raw_responses=self._validate_max_raw_responses(
                    max_raw_responses
                ),
                interrupt_grace_s=(
                    self.interrupt_grace_s
                    if interrupt_grace_s is None
                    else float(interrupt_grace_s)
                ),
                extra_params=extra_params,
            )
        finally:
            self._active_turn_lock.release()

    def run_verification_turn(
        self,
        input_value: str | Sequence[Mapping[str, Any]],
        *,
        timeout_s: float,
        responsesapi_client_metadata: Mapping[str, str] | None = None,
        client_user_message_id: str | None = None,
        max_raw_responses: int | None = None,
        interrupt_grace_s: float | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> CodexTurnResult:
        """Append a verification nudge to the exact same native thread."""

        return self.run_turn(
            input_value,
            phase="verification",
            responsesapi_client_metadata=responsesapi_client_metadata,
            client_user_message_id=client_user_message_id,
            timeout_s=timeout_s,
            max_raw_responses=max_raw_responses,
            interrupt_grace_s=interrupt_grace_s,
            extra_params=extra_params,
        )

    def run_conversation(
        self,
        initial_input: str | Sequence[Mapping[str, Any]],
        *,
        timeout_s: float,
        initial_metadata: Mapping[str, str] | None = None,
        verification_input: str | Sequence[Mapping[str, Any]] | None = None,
        verification_metadata: Mapping[str, str] | None = None,
        max_raw_responses: int | None = None,
        interrupt_grace_s: float | None = None,
    ) -> CodexConversationResult:
        """Convenience wrapper for an initial and optional same-thread turn."""

        initial = self.run_turn(
            initial_input,
            phase="initial",
            responsesapi_client_metadata=initial_metadata,
            timeout_s=timeout_s,
            max_raw_responses=max_raw_responses,
            interrupt_grace_s=interrupt_grace_s,
        )
        verification = None
        if verification_input is not None:
            verification = self.run_verification_turn(
                verification_input,
                responsesapi_client_metadata=verification_metadata,
                timeout_s=timeout_s,
                max_raw_responses=max_raw_responses,
                interrupt_grace_s=interrupt_grace_s,
            )
        assert self.thread_id is not None
        return CodexConversationResult(
            thread_id=self.thread_id,
            initial=initial,
            verification=verification,
        )

    def _run_turn_locked(
        self,
        input_value: str | Sequence[Mapping[str, Any]],
        *,
        phase: str,
        responsesapi_client_metadata: Mapping[str, str] | None,
        client_user_message_id: str | None,
        timeout_s: float,
        max_raw_responses: int | None,
        interrupt_grace_s: float,
        extra_params: Mapping[str, Any] | None,
    ) -> CodexTurnResult:
        if interrupt_grace_s <= 0:
            raise ValueError("interrupt_grace_s must be positive")
        assert self.thread_id is not None
        started_at_s = time.time()
        deadline = time.monotonic() + timeout_s
        params: JsonObject = dict(extra_params or {})
        params["threadId"] = self.thread_id
        params["input"] = self._normalize_input(input_value)
        if responsesapi_client_metadata is not None:
            params["responsesapiClientMetadata"] = {
                str(key): str(value)
                for key, value in responsesapi_client_metadata.items()
            }
        if client_user_message_id is not None:
            params["clientUserMessageId"] = client_user_message_id

        result = self._request(
            "turn/start",
            params,
            timeout_s=self._remaining_or_raise(deadline, "turn/start"),
        )
        turn = result.get("turn") if isinstance(result, dict) else None
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        if not isinstance(turn_id, str) or not turn_id:
            raise CodexProtocolError(
                "turn/start result is missing non-empty result.turn.id"
            )
        key = (self.thread_id, turn_id)

        timed_out = False
        iteration_limited = False
        interrupt_requested = False
        interrupt_error: BaseException | None = None
        try:
            terminal, limit_reached = self._wait_for_terminal_or_limit(
                key,
                deadline,
                max_raw_responses=max_raw_responses,
            )
        except _TurnWaitDeadline:
            timed_out = True
            terminal = self._terminal_if_present(key)
            if terminal is None:
                interrupt_requested = True
                terminal, interrupt_error = self._interrupt_and_wait(
                    key,
                    turn_id=turn_id,
                    interrupt_grace_s=interrupt_grace_s,
                    timeout_s=timeout_s,
                    trigger=f"turn deadline ({timeout_s:.3f}s)",
                )
        else:
            if limit_reached:
                # Terminal wins if it raced with the threshold notification and
                # was already observed before the interrupt is issued.
                terminal = self._terminal_if_present(key)
                if terminal is None:
                    iteration_limited = True
                    interrupt_requested = True
                    assert max_raw_responses is not None
                    terminal, interrupt_error = self._interrupt_and_wait(
                        key,
                        turn_id=turn_id,
                        interrupt_grace_s=interrupt_grace_s,
                        timeout_s=timeout_s,
                        trigger=f"max_raw_responses={max_raw_responses}",
                    )

        assert terminal is not None
        self._raise_callback_error_if_any()
        terminal_params = terminal.message.get("params")
        terminal_turn = (
            terminal_params.get("turn") if isinstance(terminal_params, dict) else None
        )
        if not isinstance(terminal_turn, dict):
            raise CodexProtocolError("matching turn/completed has no params.turn object")
        status = terminal_turn.get("status")
        if status not in self._TERMINAL_STATUSES:
            raise CodexProtocolError(
                f"matching turn/completed has non-terminal status {status!r}"
            )
        events = self._copy_turn_events(key)
        return CodexTurnResult(
            phase=phase,
            thread_id=self.thread_id,
            turn_id=turn_id,
            status=str(status),
            turn=copy.deepcopy(terminal_turn),
            events=events,
            started_at_s=started_at_s,
            completed_at_s=terminal.captured_at_s,
            timed_out=timed_out,
            iteration_limited=iteration_limited,
            interrupt_requested=interrupt_requested,
            interrupt_error=str(interrupt_error) if interrupt_error is not None else None,
        )

    @staticmethod
    def _validate_max_raw_responses(value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("max_raw_responses must be a positive integer when provided")
        return value

    @staticmethod
    def _normalize_input(
        input_value: str | Sequence[Mapping[str, Any]],
    ) -> list[JsonObject]:
        if isinstance(input_value, str):
            if not input_value:
                raise ValueError("turn input text must not be empty")
            return [{"type": "text", "text": input_value}]
        normalized: list[JsonObject] = []
        for item in input_value:
            if not isinstance(item, Mapping):
                raise TypeError("each turn input item must be a mapping")
            normalized.append(copy.deepcopy(dict(item)))
        if not normalized:
            raise ValueError("turn input must contain at least one item")
        return normalized

    def _remaining_or_raise(self, deadline: float, method: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CodexRpcTimeout(method, "not-sent", 0.0)
        return min(self.rpc_timeout_s, remaining)

    def _wait_for_terminal_or_limit(
        self,
        key: tuple[str, str],
        deadline: float,
        *,
        max_raw_responses: int | None,
    ) -> tuple[_TerminalTurn | None, bool]:
        with self._state_condition:
            while True:
                terminal = self._terminal_turns.get(key)
                if terminal is not None:
                    return terminal, False
                if (
                    max_raw_responses is not None
                    and self._raw_response_counts.get(key, 0) >= max_raw_responses
                ):
                    return None, True
                if self._fatal_error is not None:
                    raise self._fatal_error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _TurnWaitDeadline
                self._state_condition.wait(timeout=remaining)

    def _wait_for_terminal(
        self, key: tuple[str, str], deadline: float
    ) -> _TerminalTurn:
        terminal, _limit_reached = self._wait_for_terminal_or_limit(
            key,
            deadline,
            max_raw_responses=None,
        )
        assert terminal is not None
        return terminal

    def _interrupt_and_wait(
        self,
        key: tuple[str, str],
        *,
        turn_id: str,
        interrupt_grace_s: float,
        timeout_s: float,
        trigger: str,
    ) -> tuple[_TerminalTurn, BaseException | None]:
        assert self.thread_id is not None
        interrupt_deadline = time.monotonic() + interrupt_grace_s
        interrupt_error: BaseException | None = None
        try:
            self._request(
                "turn/interrupt",
                {"threadId": self.thread_id, "turnId": turn_id},
                timeout_s=min(
                    self.rpc_timeout_s,
                    max(0.001, interrupt_deadline - time.monotonic()),
                ),
            )
        except BaseException as exc:
            interrupt_error = exc
        try:
            terminal = self._wait_for_terminal(key, interrupt_deadline)
        except _TurnWaitDeadline as exc:
            raise CodexTurnTimeout(
                thread_id=self.thread_id,
                turn_id=turn_id,
                timeout_s=timeout_s,
                interrupt_grace_s=interrupt_grace_s,
                interrupt_error=interrupt_error,
                trigger=trigger,
            ) from exc
        return terminal, interrupt_error

    def _terminal_if_present(self, key: tuple[str, str]) -> _TerminalTurn | None:
        with self._state_condition:
            return self._terminal_turns.get(key)

    def _copy_turn_events(
        self, key: tuple[str, str]
    ) -> tuple[CodexAppServerEvent, ...]:
        with self._state_condition:
            return tuple(
                CodexAppServerEvent(
                    captured_at_s=event.captured_at_s,
                    message=copy.deepcopy(event.message),
                )
                for event in self._events_by_turn.get(key, ())
            )

    def _next_id(self) -> int:
        with self._pending_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            return request_id

    def _request(
        self, method: str, params: Mapping[str, Any], *, timeout_s: float
    ) -> JsonObject:
        self._ensure_running()
        if timeout_s <= 0:
            raise CodexRpcTimeout(method, "not-sent", timeout_s)
        request_id = self._next_id()
        response_queue: queue.Queue[_RpcReply | _FatalReply] = queue.Queue(maxsize=1)
        with self._pending_lock:
            if self._fatal_error is not None:
                raise self._fatal_error
            self._pending[request_id] = response_queue
        try:
            self._send_message(
                {"method": method, "id": request_id, "params": dict(params)}
            )
            try:
                reply = response_queue.get(timeout=timeout_s)
            except queue.Empty as exc:
                with self._pending_lock:
                    self._expired_request_ids.add(request_id)
                raise CodexRpcTimeout(method, request_id, timeout_s) from exc
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

        if isinstance(reply, _FatalReply):
            raise reply.error
        response = reply.message
        if "error" in response:
            error = response.get("error")
            if not isinstance(error, dict):
                raise CodexProtocolError(
                    f"RPC response id={request_id!r} has a non-object error"
                )
            raise CodexRpcError(
                method=method,
                request_id=request_id,
                code=error.get("code") if isinstance(error.get("code"), int) else None,
                message=str(error.get("message") or "unknown RPC error"),
                data=error.get("data"),
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise CodexProtocolError(
                f"RPC response for {method!r} id={request_id!r} has no object result"
            )
        return result

    def _send_notification(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> None:
        message: JsonObject = {"method": method}
        if params is not None:
            message["params"] = dict(params)
        self._send_message(message)

    def _send_message(self, message: JsonObject) -> None:
        self._ensure_running()
        try:
            encoded = json.dumps(
                message,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise CodexProtocolError(f"outbound JSON-RPC message is not serializable: {exc}") from exc
        captured_at_s = time.time()
        with self._write_lock:
            process = self._process
            if process is None or process.stdin is None:
                raise CodexAppServerError("Codex app-server stdin is unavailable")
            try:
                process.stdin.write(encoded + "\n")
                # Record before flush: once flush returns, the reader thread may
                # already have received and logged the response.
                self._record_wire("to_server", message, captured_at_s)
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                error = self._process_exit_error()
                self._set_fatal(error)
                raise error from exc

    def _stdout_reader(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        stream = self._process.stdout
        try:
            for raw_line in stream:
                captured_at_s = time.time()
                if self._raw_fp is not None:
                    self._raw_fp.write(raw_line)
                    self._raw_fp.flush()
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._set_fatal(
                        CodexProtocolError(
                            f"invalid JSON from Codex app-server stdout: {exc}: {line[:500]!r}"
                        )
                    )
                    continue
                if not isinstance(message, dict):
                    self._set_fatal(
                        CodexProtocolError(
                            f"non-object JSON-RPC envelope from Codex: {type(message).__name__}"
                        )
                    )
                    continue
                self._record_wire("from_server", message, captured_at_s)
                self._route_inbound(message, captured_at_s)
        finally:
            if not self._closing:
                self._set_fatal(self._process_exit_error())

    def _stderr_reader(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr_tail.append(line.rstrip("\r\n"))
            if self._stderr_fp is not None:
                self._stderr_fp.write(line)
                self._stderr_fp.flush()

    def _route_inbound(self, message: JsonObject, captured_at_s: float) -> None:
        method = message.get("method")
        if isinstance(method, str):
            if "id" in message:
                self._handle_server_request(message, captured_at_s)
            else:
                self._handle_notification(message, captured_at_s)
            return

        if "id" not in message or not ({"result", "error"} & message.keys()):
            self._set_fatal(
                CodexProtocolError(f"unrecognized JSON-RPC envelope: {message!r}")
            )
            return
        request_id = message.get("id")
        with self._pending_lock:
            response_queue = self._pending.get(request_id)
            expired = request_id in self._expired_request_ids
            if expired:
                self._expired_request_ids.discard(request_id)
        if expired:
            return
        if response_queue is None:
            self._set_fatal(
                CodexProtocolError(f"response for unknown JSON-RPC id {request_id!r}")
            )
            return
        try:
            response_queue.put_nowait(_RpcReply(copy.deepcopy(message)))
        except queue.Full:
            self._set_fatal(
                CodexProtocolError(f"duplicate JSON-RPC response id {request_id!r}")
            )

    def _handle_notification(self, message: JsonObject, captured_at_s: float) -> None:
        method = str(message.get("method"))
        params = message.get("params")
        params_obj = params if isinstance(params, dict) else {}
        key = self._event_key(params_obj)
        event = CodexAppServerEvent(
            captured_at_s=captured_at_s,
            message=copy.deepcopy(message),
        )

        terminal_error: CodexProtocolError | None = None
        with self._state_condition:
            if key is not None:
                self._events_by_turn.setdefault(key, []).append(event)
                if method == "rawResponse/completed":
                    self._raw_response_counts[key] = (
                        self._raw_response_counts.get(key, 0) + 1
                    )
            if method == "turn/completed":
                if key is None:
                    terminal_error = CodexProtocolError(
                        "turn/completed is missing params.threadId or params.turn.id"
                    )
                else:
                    turn = params_obj.get("turn")
                    status = turn.get("status") if isinstance(turn, dict) else None
                    if status not in self._TERMINAL_STATUSES:
                        terminal_error = CodexProtocolError(
                            f"turn/completed {key!r} has invalid status {status!r}"
                        )
                    elif key in self._terminal_turns:
                        terminal_error = CodexProtocolError(
                            f"duplicate turn/completed for {key!r}"
                        )
                    else:
                        self._terminal_turns[key] = _TerminalTurn(
                            message=copy.deepcopy(message),
                            captured_at_s=captured_at_s,
                        )

        if self.event_callback is not None:
            try:
                self.event_callback(copy.deepcopy(message), captured_at_s)
            except BaseException as exc:
                callback_error = CodexEventCallbackError(
                    f"Codex event callback failed for {method!r}: {exc}"
                )
                callback_error.__cause__ = exc
                self._callback_error = callback_error
                self._set_fatal(callback_error)

        if terminal_error is not None:
            self._set_fatal(terminal_error)
        with self._state_condition:
            self._state_condition.notify_all()

    @staticmethod
    def _event_key(params: Mapping[str, Any]) -> tuple[str, str] | None:
        thread_id = params.get("threadId")
        turn_id = params.get("turnId")
        if not isinstance(turn_id, str):
            turn = params.get("turn")
            turn_id = turn.get("id") if isinstance(turn, Mapping) else None
        if not isinstance(thread_id, str) or not isinstance(turn_id, str):
            return None
        if not thread_id or not turn_id:
            return None
        return thread_id, turn_id

    def _handle_server_request(
        self, message: JsonObject, captured_at_s: float
    ) -> None:
        del captured_at_s
        method = str(message.get("method") or "")
        request_id = message.get("id")
        params = message.get("params")
        params_obj = params if isinstance(params, dict) else {}
        self._unexpected_server_requests.append(copy.deepcopy(message))
        if self.server_request_handler is None:
            self._send_rpc_error(
                request_id,
                code=-32601,
                message=(
                    f"Unsupported server request {method!r} in non-interactive "
                    "MARS benchmark driver"
                ),
            )
            return
        try:
            result = self.server_request_handler(method, params_obj, message)
        except CodexServerRequestRejected as exc:
            self._send_rpc_error(
                request_id, code=exc.code, message=str(exc), data=exc.data
            )
        except BaseException as exc:
            self._send_rpc_error(
                request_id,
                code=-32603,
                message=f"MARS server-request handler failed: {exc}",
            )
        else:
            self._send_message({"id": request_id, "result": result})

    def _send_rpc_error(
        self,
        request_id: Any,
        *,
        code: int,
        message: str,
        data: Any = None,
    ) -> None:
        error: JsonObject = {"code": int(code), "message": message}
        if data is not None:
            error["data"] = data
        self._send_message({"id": request_id, "error": error})

    def _record_wire(
        self, direction: str, message: Mapping[str, Any], captured_at_s: float
    ) -> None:
        if self._wire_fp is None:
            return
        with self._wire_lock:
            sequence = self._wire_sequence
            self._wire_sequence += 1
            record = {
                "schema": "codex_app_server_wire_v1",
                "sequence": sequence,
                "direction": direction,
                "captured_at_s": captured_at_s,
                "message": message,
            }
            self._wire_fp.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            self._wire_fp.flush()

    def _set_fatal(self, error: BaseException) -> None:
        with self._state_condition:
            if self._fatal_error is None:
                self._fatal_error = error
            fatal = self._fatal_error
            self._state_condition.notify_all()
        with self._pending_lock:
            pending = list(self._pending.values())
        for response_queue in pending:
            try:
                response_queue.put_nowait(_FatalReply(fatal))
            except queue.Full:
                pass

    def _raise_callback_error_if_any(self) -> None:
        if self._callback_error is not None:
            raise self._callback_error

    def _ensure_running(self) -> None:
        if not self._started:
            raise CodexAppServerError("Codex app-server has not been started")
        if self._closed or self._closing:
            raise CodexAppServerError("Codex app-server is closing or closed")
        if self._fatal_error is not None:
            raise self._fatal_error
        process = self._process
        if process is None:
            raise CodexAppServerError("Codex app-server process is unavailable")
        returncode = process.poll()
        if returncode is not None:
            error = self._process_exit_error(returncode)
            self._set_fatal(error)
            raise error

    def _process_exit_error(self, returncode: int | None = None) -> CodexProcessExited:
        process = self._process
        if returncode is None and process is not None:
            returncode = process.poll()
            if returncode is None:
                try:
                    returncode = process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    returncode = process.poll()
        stderr_tail = "\n".join(self._stderr_tail)[-4000:]
        return CodexProcessExited(returncode, stderr_tail)

    def close(self) -> None:
        """Close stdin, then terminate only this app-server process group."""

        if self._closed:
            return
        self._closing = True
        process = self._process
        if process is not None:
            with self._write_lock:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
            try:
                process.wait(timeout=self.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                self._signal_process_group(signal.SIGTERM)
                try:
                    process.wait(timeout=self.shutdown_timeout_s)
                except subprocess.TimeoutExpired:
                    self._signal_process_group(signal.SIGKILL)
                    try:
                        process.wait(timeout=self.shutdown_timeout_s)
                    except subprocess.TimeoutExpired:
                        pass

        current = threading.current_thread()
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None and thread is not current:
                thread.join(timeout=self.shutdown_timeout_s)
        self._close_log_files()
        self._closed = True

    def _signal_process_group(self, sig: signal.Signals) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            if self.start_new_session and process.pid > 1:
                os.killpg(process.pid, sig)
            else:
                process.send_signal(sig)
        except ProcessLookupError:
            pass

    def _close_log_files(self) -> None:
        for fp in (self._raw_fp, self._stderr_fp, self._wire_fp):
            if fp is not None:
                try:
                    fp.flush()
                    fp.close()
                except OSError:
                    pass
        self._raw_fp = None
        self._stderr_fp = None
        self._wire_fp = None


__all__ = [
    "CodexAppServerDriver",
    "CodexAppServerError",
    "CodexAppServerEvent",
    "CodexConversationResult",
    "CodexEventCallbackError",
    "CodexProcessExited",
    "CodexProtocolError",
    "CodexRpcError",
    "CodexRpcTimeout",
    "CodexServerRequestRejected",
    "CodexTurnResult",
    "CodexTurnTimeout",
    "build_codex_app_server_command",
]
