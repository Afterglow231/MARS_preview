#!/usr/bin/env python3
"""Native OpenCode server/session driver for the MARS agent baseline.

OpenCode, rather than this module, owns the conversation, model/tool loop,
child sessions, retries, and final answer.  The driver starts one authenticated
``opencode --pure serve`` process per benchmark request and uses only the
native legacy HTTP/SSE surface:

* subscribe to ``/event`` before creating the session;
* create one root session and append every prompt to that same session;
* observe native busy/retry/idle and root/child tool transitions in real time;
* automatically answer permission requests and reject interactive questions;
* reconcile the authoritative message and child-session state after each turn;
* abort the root session on the evaluator deadline; and
* terminate only the per-request server process group during cleanup.

There is intentionally no Chat/Responses translation or locally implemented
tool loop in this module.  vLLM is merely the provider endpoint configured in
the OpenCode runtime environment.
"""

from __future__ import annotations

import base64
import copy
import http.client
import json
import os
import secrets
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias


JsonObject: TypeAlias = dict[str, Any]
EventCallback: TypeAlias = Callable[[JsonObject, float], Any]
ToolEventCallback: TypeAlias = Callable[[JsonObject, float], Any]
CommandBuilder: TypeAlias = Callable[[Sequence[str]], Sequence[str]]
VerificationPredicate: TypeAlias = Callable[["OpenCodeSnapshot"], bool]


DEFAULT_OPENCODE_BINARY = Path(
    os.environ.get(
        "MARS_AGENT_BASELINES_ROOT",
        str(Path.home() / "external" / "mars_agent_baselines"),
    )
) / "runtime/opencode/node_modules/.bin/opencode"


class OpenCodeServerError(RuntimeError):
    """Base error for native OpenCode server/session failures."""


class OpenCodeProtocolError(OpenCodeServerError):
    """The native HTTP/SSE API returned a malformed or inconsistent value."""


class OpenCodeHttpError(OpenCodeServerError):
    """A native OpenCode HTTP endpoint returned a non-success response."""

    def __init__(self, method: str, path: str, status: int, body: str) -> None:
        preview = body[-2000:] if body else ""
        message = f"OpenCode {method} {path} returned HTTP {status}"
        if preview:
            message += f": {preview}"
        super().__init__(message)
        self.method = method
        self.path = path
        self.status = int(status)
        self.body = body


class OpenCodeProcessExited(OpenCodeServerError):
    """The per-request server exited before native session completion."""

    def __init__(self, returncode: int | None, stderr_tail: str = "") -> None:
        message = (
            "OpenCode server exited before native session completion "
            f"(returncode={returncode})"
        )
        if stderr_tail:
            message += f"; stderr tail: {stderr_tail}"
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


class OpenCodeEventStreamError(OpenCodeServerError):
    """The SSE stream disconnected or emitted invalid JSON."""


class OpenCodeEventCallbackError(OpenCodeServerError):
    """A live event observer failed, making the trace incomplete."""


class OpenCodePromptTimeout(OpenCodeServerError):
    """A prompt exceeded its deadline and was aborted."""

    def __init__(
        self,
        *,
        session_id: str,
        phase: str,
        timeout_s: float,
        abort_error: str | None,
    ) -> None:
        message = (
            f"OpenCode session {session_id!r} phase {phase!r} exceeded "
            f"{timeout_s:.3f}s and was aborted"
        )
        if abort_error:
            message += f"; abort error: {abort_error}"
        super().__init__(message)
        self.session_id = session_id
        self.phase = phase
        self.timeout_s = float(timeout_s)
        self.abort_error = abort_error


@dataclass(frozen=True, slots=True)
class OpenCodeServerEvent:
    """One unmodified SSE payload and its host receive timestamps."""

    captured_at_s: float
    captured_monotonic_s: float
    message: JsonObject

    @property
    def type(self) -> str:
        return str(self.message.get("type") or "")


@dataclass(frozen=True, slots=True)
class OpenCodeToolRecord:
    """Final state of one native tool part from any tracked session."""

    key: str
    session_id: str
    parent_session_id: str | None
    message_id: str
    part_id: str
    call_id: str
    tool_name: str
    phase: str
    status: str
    enqueued_time_s: float | None
    start_time_s: float | None
    end_time_s: float | None
    input: Any
    output: Any
    error: Any
    metadata: Any
    transitions: tuple[str, ...]

    @property
    def wait_s(self) -> float | None:
        if self.enqueued_time_s is None or self.start_time_s is None:
            return None
        return max(0.0, self.start_time_s - self.enqueued_time_s)

    @property
    def duration_s(self) -> float | None:
        if self.start_time_s is None or self.end_time_s is None:
            return None
        return max(0.0, self.end_time_s - self.start_time_s)


@dataclass(frozen=True, slots=True)
class OpenCodePhaseResult:
    """Native completion observation for one prompt on the root session."""

    phase: str
    session_id: str
    status: str
    started_at_s: float
    completed_at_s: float
    busy_seen: bool
    retry_count: int
    timed_out: bool
    abort_requested: bool
    abort_error: str | None
    event_start_index: int
    event_end_index: int

    @property
    def duration_s(self) -> float:
        return max(0.0, self.completed_at_s - self.started_at_s)


@dataclass(frozen=True, slots=True)
class OpenCodeSnapshot:
    """Reconciled same-session state exposed to a verification predicate."""

    root_session_id: str
    sessions: Mapping[str, JsonObject]
    messages_by_session: Mapping[str, tuple[JsonObject, ...]]
    tools: tuple[OpenCodeToolRecord, ...]
    errors: tuple[JsonObject, ...]
    phases: tuple[OpenCodePhaseResult, ...]


@dataclass(frozen=True, slots=True)
class OpenCodeConversationResult:
    """One full OpenCode conversation, including optional verification turns."""

    root_session_id: str
    status: str
    phases: tuple[OpenCodePhaseResult, ...]
    events: tuple[OpenCodeServerEvent, ...]
    tools: tuple[OpenCodeToolRecord, ...]
    sessions: Mapping[str, JsonObject]
    messages_by_session: Mapping[str, tuple[JsonObject, ...]]
    errors: tuple[JsonObject, ...]
    final_output: str | None
    started_at_s: float
    completed_at_s: float
    timed_out: bool
    process_returncode: int | None
    stdout_tail: str
    stderr_tail: str

    @property
    def duration_s(self) -> float:
        return max(0.0, self.completed_at_s - self.started_at_s)


@dataclass(slots=True)
class _MutableTool:
    key: str
    session_id: str
    parent_session_id: str | None
    message_id: str
    part_id: str
    call_id: str
    tool_name: str
    phase: str
    status: str = "unknown"
    enqueued_time_s: float | None = None
    start_time_s: float | None = None
    end_time_s: float | None = None
    input: Any = None
    output: Any = None
    error: Any = None
    metadata: Any = None
    transitions: list[str] | None = None

    def freeze(self) -> OpenCodeToolRecord:
        return OpenCodeToolRecord(
            key=self.key,
            session_id=self.session_id,
            parent_session_id=self.parent_session_id,
            message_id=self.message_id,
            part_id=self.part_id,
            call_id=self.call_id,
            tool_name=self.tool_name,
            phase=self.phase,
            status=self.status,
            enqueued_time_s=self.enqueued_time_s,
            start_time_s=self.start_time_s,
            end_time_s=self.end_time_s,
            input=copy.deepcopy(self.input),
            output=copy.deepcopy(self.output),
            error=copy.deepcopy(self.error),
            metadata=copy.deepcopy(self.metadata),
            transitions=tuple(self.transitions or ()),
        )


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_opencode_server_command(
    binary: str | os.PathLike[str] = DEFAULT_OPENCODE_BINARY,
    *,
    host: str = "127.0.0.1",
    port: int,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build the pinned CLI's isolated native server command."""

    if not (0 < int(port) < 65536):
        raise ValueError("port must be between 1 and 65535")
    if host != "127.0.0.1":
        raise ValueError("formal OpenCode baseline server must bind to 127.0.0.1")
    return [
        str(binary),
        "--pure",
        "serve",
        "--hostname",
        host,
        "--port",
        str(int(port)),
        *(str(arg) for arg in extra_args),
    ]


class OpenCodeServerDriver:
    """Controller for one authenticated OpenCode server and one root session.

    ``command_builder`` receives the fully resolved native server argv and may
    wrap it in an externally prepared sandbox command (for example bwrap with a
    PID namespace).  Alternatively ``command`` and ``env`` may be supplied
    directly.  ``{host}`` and ``{port}`` placeholders in a custom command are
    expanded after the secure endpoint is selected.

    Passing ``base_url`` attaches to a test or externally managed server and
    disables subprocess management; production callers should omit it.
    """

    _TERMINAL_TOOL_STATES = {"completed", "error"}

    def __init__(
        self,
        *,
        workspace_dir: str | os.PathLike[str],
        model: str,
        provider_id: str = "mars",
        agent: str = "build",
        binary: str | os.PathLike[str] = DEFAULT_OPENCODE_BINARY,
        command: Sequence[str] | None = None,
        command_builder: CommandBuilder | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        base_url: str | None = None,
        event_callback: EventCallback | None = None,
        tool_event_callback: ToolEventCallback | None = None,
        startup_timeout_s: float = 30.0,
        http_timeout_s: float = 30.0,
        abort_grace_s: float = 5.0,
        shutdown_timeout_s: float = 3.0,
        event_settle_s: float = 0.05,
        permission_reply: str = "once",
        reject_questions: bool = True,
        start_new_session: bool = True,
    ) -> None:
        workspace = Path(workspace_dir).absolute()
        if not str(model).strip():
            raise ValueError("model must not be empty")
        if not str(provider_id).strip():
            raise ValueError("provider_id must not be empty")
        if not str(agent).strip():
            raise ValueError("agent must not be empty")
        if host != "127.0.0.1":
            raise ValueError("OpenCode server must bind to 127.0.0.1")
        if command is not None and command_builder is not None:
            raise ValueError("command and command_builder are mutually exclusive")
        for value, name in (
            (startup_timeout_s, "startup_timeout_s"),
            (http_timeout_s, "http_timeout_s"),
            (abort_grace_s, "abort_grace_s"),
            (shutdown_timeout_s, "shutdown_timeout_s"),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if event_settle_s < 0:
            raise ValueError("event_settle_s must be non-negative")
        if permission_reply not in {"once", "always", "reject"}:
            raise ValueError("permission_reply must be once, always, or reject")

        self.workspace_dir = workspace
        self.model = str(model)
        self.provider_id = str(provider_id)
        self.agent = str(agent)
        self.binary = str(binary)
        self.host = host
        self.port = int(port or _reserve_loopback_port())
        if not (0 < self.port < 65536):
            raise ValueError("port must be between 1 and 65535")
        self.username = username or f"mars_{secrets.token_urlsafe(9)}"
        self.password = password or secrets.token_urlsafe(32)
        self.base_url = base_url.rstrip("/") if base_url else None
        self.event_callback = event_callback
        self.tool_event_callback = tool_event_callback
        self.startup_timeout_s = float(startup_timeout_s)
        self.http_timeout_s = float(http_timeout_s)
        self.abort_grace_s = float(abort_grace_s)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self.event_settle_s = float(event_settle_s)
        self.permission_reply = permission_reply
        self.reject_questions = bool(reject_questions)
        self.start_new_session = bool(start_new_session)
        self.cwd = str(Path(cwd).absolute()) if cwd is not None else str(workspace)

        selected_env = dict(os.environ if env is None else env)
        selected_env["OPENCODE_SERVER_USERNAME"] = self.username
        selected_env["OPENCODE_SERVER_PASSWORD"] = self.password
        # Experimental mode currently implies background subagents unless this
        # flag is explicitly false.  A benchmark request must be quiescent when
        # its root session becomes idle.
        selected_env["OPENCODE_EXPERIMENTAL_BACKGROUND_SUBAGENTS"] = "false"
        self.env = selected_env

        if command is None:
            argv = build_opencode_server_command(
                self.binary, host=self.host, port=self.port
            )
        else:
            argv = [
                str(arg).replace("{host}", self.host).replace("{port}", str(self.port))
                for arg in command
            ]
        if command_builder is not None:
            argv = [str(arg) for arg in command_builder(tuple(argv))]
        if not argv and self.base_url is None:
            raise ValueError("OpenCode server command must not be empty")
        self.command = tuple(argv)

        if self.base_url is not None:
            parsed = urllib.parse.urlsplit(self.base_url)
            if parsed.scheme != "http" or parsed.hostname not in {
                "127.0.0.1",
                "localhost",
            }:
                raise ValueError("base_url must be a loopback http endpoint")
            self.host = str(parsed.hostname)
            if parsed.port is None:
                raise ValueError("base_url must include an explicit port")
            self.port = int(parsed.port)

        self._managed_process = self.base_url is None
        self._process: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stdout_tail: deque[str] = deque(maxlen=200)
        self._stderr_tail: deque[str] = deque(maxlen=200)
        self._sse_thread: threading.Thread | None = None
        self._sse_connection: http.client.HTTPConnection | None = None
        self._sse_response: http.client.HTTPResponse | None = None
        self._condition = threading.Condition(threading.RLock())
        self._conversation_lock = threading.Lock()
        self._events: list[OpenCodeServerEvent] = []
        self._tools: dict[str, _MutableTool] = {}
        self._tool_order: list[str] = []
        self._sessions: dict[str, JsonObject] = {}
        self._session_parent: dict[str, str | None] = {}
        self._session_status: dict[str, str] = {}
        self._messages_by_session: dict[str, list[JsonObject]] = {}
        self._errors: list[JsonObject] = []
        self._root_errors: list[JsonObject] = []
        self._deferred_permissions: dict[str, JsonObject] = {}
        self._deferred_questions: dict[str, JsonObject] = {}
        self._callback_error: BaseException | None = None
        self._sse_error: BaseException | None = None
        self._server_connected = False
        self._sse_stop = threading.Event()
        self._started = False
        self._closed = False
        self._closing = False
        self.root_session_id: str | None = None
        self._active_phase: str | None = None
        self._phase_activity = False
        self._phase_busy_seen = False
        self._phase_idle = False
        self._phase_retry_count = 0
        self._phase_event_start = 0
        self._phases: list[OpenCodePhaseResult] = []

    def __enter__(self) -> "OpenCodeServerDriver":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()

    @property
    def process(self) -> subprocess.Popen[str] | None:
        return self._process

    @property
    def events(self) -> tuple[OpenCodeServerEvent, ...]:
        with self._condition:
            return tuple(copy.deepcopy(self._events))

    @property
    def stdout_tail(self) -> str:
        return "".join(self._stdout_tail)[-12000:]

    @property
    def stderr_tail(self) -> str:
        return "".join(self._stderr_tail)[-12000:]

    def start(self) -> str:
        """Start the server, connect SSE, then create the root session."""

        if self._closed:
            raise OpenCodeServerError("OpenCode driver is already closed")
        if self._started:
            assert self.root_session_id is not None
            return self.root_session_id
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        if self._managed_process:
            self._launch_process()
        self._wait_until_ready()
        self._start_sse()
        self._wait_for_server_connected()
        session = self._api_json(
            "POST",
            "/session",
            body={
                "title": f"MARS request {self.workspace_dir.name}",
                "permission": [
                    {"permission": "question", "pattern": "*", "action": "deny"},
                    {"permission": "plan_enter", "pattern": "*", "action": "deny"},
                    {"permission": "plan_exit", "pattern": "*", "action": "deny"},
                ],
            },
        )
        if not isinstance(session, dict):
            raise OpenCodeProtocolError("POST /session did not return an object")
        session_id = session.get("id")
        if not isinstance(session_id, str) or not session_id:
            raise OpenCodeProtocolError("POST /session response is missing id")
        with self._condition:
            self.root_session_id = session_id
            self._track_session_locked(session)
            self._started = True
            self._condition.notify_all()
        self._drain_deferred_actions()
        return session_id

    def run_prompt(
        self,
        prompt: str,
        *,
        phase: str = "initial",
        timeout_s: float,
    ) -> OpenCodePhaseResult:
        """Append one prompt and wait for its native root-session idle event."""

        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        if not str(phase).strip():
            raise ValueError("phase must not be empty")
        if not self._conversation_lock.acquire(blocking=False):
            raise OpenCodeServerError(
                "concurrent prompts on one session are unsupported"
            )
        try:
            if not self._started:
                self.start()
            return self._run_prompt_locked(
                prompt, phase=str(phase), timeout_s=float(timeout_s)
            )
        finally:
            self._conversation_lock.release()

    def run_verification_prompt(
        self,
        prompt: str,
        *,
        timeout_s: float,
        phase: str = "verification",
    ) -> OpenCodePhaseResult:
        """Append a verification nudge to the exact same root session."""

        return self.run_prompt(prompt, phase=phase, timeout_s=timeout_s)

    def run_conversation(
        self,
        initial_prompt: str,
        *,
        timeout_s: float,
        verification_prompts: Sequence[str] = (),
        verification_needed: VerificationPredicate | None = None,
    ) -> OpenCodeConversationResult:
        """Run one initial turn and zero or more same-session verifications.

        ``timeout_s`` is one whole-conversation budget, not a per-turn reset.
        A predicate is evaluated against a reconciled snapshot before each
        verification prompt.  Without a predicate every supplied verification
        prompt is sent.
        """

        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        started_at_s = time.time()
        deadline = time.monotonic() + float(timeout_s)
        timed_out = False
        try:
            self.run_prompt(
                initial_prompt,
                phase="initial",
                timeout_s=self._remaining(deadline),
            )
            self.reconcile()
            for index, prompt in enumerate(verification_prompts, start=1):
                if verification_needed is not None and not verification_needed(
                    self.snapshot()
                ):
                    break
                self.run_verification_prompt(
                    prompt,
                    phase=f"verification_{index}",
                    timeout_s=self._remaining(deadline),
                )
                self.reconcile()
        except OpenCodePromptTimeout:
            timed_out = True
        self.reconcile(best_effort=True)
        completed_at_s = time.time()
        snapshot = self.snapshot()
        if timed_out:
            status = "timeout"
        elif self._root_errors:
            status = "error"
        else:
            status = "completed"
        return OpenCodeConversationResult(
            root_session_id=snapshot.root_session_id,
            status=status,
            phases=snapshot.phases,
            events=self.events,
            tools=snapshot.tools,
            sessions=snapshot.sessions,
            messages_by_session=snapshot.messages_by_session,
            errors=snapshot.errors,
            final_output=self._final_output(snapshot),
            started_at_s=started_at_s,
            completed_at_s=completed_at_s,
            timed_out=timed_out,
            process_returncode=(
                self._process.poll() if self._process is not None else None
            ),
            stdout_tail=self.stdout_tail,
            stderr_tail=self.stderr_tail,
        )

    def snapshot(self) -> OpenCodeSnapshot:
        if self.root_session_id is None:
            raise OpenCodeServerError("root session has not been created")
        with self._condition:
            sessions = {
                session_id: copy.deepcopy(info)
                for session_id, info in self._sessions.items()
            }
            messages = {
                session_id: tuple(copy.deepcopy(rows))
                for session_id, rows in self._messages_by_session.items()
            }
            tools = tuple(self._tools[key].freeze() for key in self._tool_order)
            errors = tuple(copy.deepcopy(self._errors))
            phases = tuple(self._phases)
        return OpenCodeSnapshot(
            root_session_id=self.root_session_id,
            sessions=sessions,
            messages_by_session=messages,
            tools=tools,
            errors=errors,
            phases=phases,
        )

    def reconcile(self, *, best_effort: bool = False) -> OpenCodeSnapshot:
        """Fetch authoritative root/child messages and repair missed tool state."""

        if self.root_session_id is None:
            raise OpenCodeServerError("root session has not been created")
        pending = [self.root_session_id]
        visited: set[str] = set()
        failures: list[BaseException] = []
        while pending:
            session_id = pending.pop(0)
            if session_id in visited:
                continue
            visited.add(session_id)
            try:
                children = self._api_json(
                    "GET",
                    f"/session/{urllib.parse.quote(session_id, safe='')}/children",
                )
                messages = self._api_json(
                    "GET", f"/session/{urllib.parse.quote(session_id, safe='')}/message"
                )
            except BaseException as exc:
                failures.append(exc)
                continue
            if not isinstance(children, list) or not all(
                isinstance(item, dict) for item in children
            ):
                failures.append(
                    OpenCodeProtocolError(
                        f"children response for {session_id!r} is not an object list"
                    )
                )
                continue
            if not isinstance(messages, list) or not all(
                isinstance(item, dict) for item in messages
            ):
                failures.append(
                    OpenCodeProtocolError(
                        f"messages response for {session_id!r} is not an object list"
                    )
                )
                continue
            with self._condition:
                for child in children:
                    child_id = child.get("id")
                    if isinstance(child_id, str) and child_id:
                        self._track_session_locked(child)
                        pending.append(child_id)
                self._messages_by_session[session_id] = copy.deepcopy(messages)
                phase = self._active_phase or (
                    self._phases[-1].phase if self._phases else "initial"
                )
                for row in messages:
                    info = row.get("info")
                    parts = row.get("parts")
                    if isinstance(info, dict):
                        error = info.get("error")
                        if error is not None and info.get("role") == "assistant":
                            self._record_message_error_locked(info, error)
                    if isinstance(parts, list):
                        for part in parts:
                            if isinstance(part, dict) and part.get("type") == "tool":
                                self._update_tool_locked(
                                    part,
                                    captured_at_s=time.time(),
                                    phase=phase,
                                    source="reconcile",
                                )
                self._condition.notify_all()
        self._drain_deferred_actions()
        if failures and not best_effort:
            raise OpenCodeServerError(
                "OpenCode reconciliation failed: "
                + "; ".join(str(item) for item in failures)
            )
        return self.snapshot()

    def abort(self) -> bool:
        """Ask OpenCode to cancel the active root session."""

        if self.root_session_id is None:
            return False
        result = self._api_json(
            "POST",
            f"/session/{urllib.parse.quote(self.root_session_id, safe='')}/abort",
            body=None,
        )
        return bool(result)

    def _run_prompt_locked(
        self, prompt: str, *, phase: str, timeout_s: float
    ) -> OpenCodePhaseResult:
        if timeout_s <= 0:
            raise OpenCodePromptTimeout(
                session_id=self.root_session_id or "unknown",
                phase=phase,
                timeout_s=max(0.0, timeout_s),
                abort_error="deadline expired before prompt submission",
            )
        self._ensure_healthy()
        assert self.root_session_id is not None
        started_at_s = time.time()
        deadline = time.monotonic() + timeout_s
        with self._condition:
            self._active_phase = phase
            self._phase_activity = False
            self._phase_busy_seen = False
            self._phase_idle = False
            self._phase_retry_count = 0
            self._phase_event_start = len(self._events)
        try:
            self._api_json(
                "POST",
                (
                    f"/session/{urllib.parse.quote(self.root_session_id, safe='')}"
                    "/prompt_async"
                ),
                body={
                    "model": {
                        "providerID": self.provider_id,
                        "modelID": self.model,
                    },
                    "agent": self.agent,
                    "parts": [{"type": "text", "text": prompt}],
                },
                timeout_s=min(self.http_timeout_s, self._remaining(deadline)),
                expected_statuses={204},
            )
            self._wait_for_phase_idle(deadline)
            if self.event_settle_s:
                time.sleep(min(self.event_settle_s, self._remaining(deadline)))
            completed_at_s = time.time()
            with self._condition:
                result = OpenCodePhaseResult(
                    phase=phase,
                    session_id=self.root_session_id,
                    status="error" if self._root_errors else "completed",
                    started_at_s=started_at_s,
                    completed_at_s=completed_at_s,
                    busy_seen=self._phase_busy_seen,
                    retry_count=self._phase_retry_count,
                    timed_out=False,
                    abort_requested=False,
                    abort_error=None,
                    event_start_index=self._phase_event_start,
                    event_end_index=len(self._events),
                )
                self._phases.append(result)
                self._active_phase = None
            return result
        except TimeoutError:
            return self._timeout_phase(
                phase=phase,
                started_at_s=started_at_s,
                timeout_s=timeout_s,
            )

    def _timeout_phase(
        self, *, phase: str, started_at_s: float, timeout_s: float
    ) -> OpenCodePhaseResult:
        abort_error: str | None = None
        try:
            self.abort()
        except BaseException as exc:
            abort_error = str(exc)
        grace_deadline = time.monotonic() + self.abort_grace_s
        try:
            self._wait_for_phase_idle(grace_deadline, allow_no_activity=True)
        except (TimeoutError, OpenCodeServerError):
            pass
        self.reconcile(best_effort=True)
        completed_at_s = time.time()
        with self._condition:
            result = OpenCodePhaseResult(
                phase=phase,
                session_id=self.root_session_id or "unknown",
                status="timeout",
                started_at_s=started_at_s,
                completed_at_s=completed_at_s,
                busy_seen=self._phase_busy_seen,
                retry_count=self._phase_retry_count,
                timed_out=True,
                abort_requested=True,
                abort_error=abort_error,
                event_start_index=self._phase_event_start,
                event_end_index=len(self._events),
            )
            self._phases.append(result)
            self._active_phase = None
        raise OpenCodePromptTimeout(
            session_id=self.root_session_id or "unknown",
            phase=phase,
            timeout_s=timeout_s,
            abort_error=abort_error,
        )

    def _wait_for_phase_idle(
        self, deadline: float, *, allow_no_activity: bool = False
    ) -> None:
        with self._condition:
            while True:
                self._raise_async_error_locked()
                if self._phase_idle and (self._phase_activity or allow_no_activity):
                    active_children = [
                        session_id
                        for session_id, status in self._session_status.items()
                        if session_id in self._session_parent
                        and status in {"busy", "retry"}
                    ]
                    if not active_children:
                        return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                self._condition.wait(timeout=min(remaining, 0.25))
                self._ensure_process_alive_locked()

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        return remaining

    def _launch_process(self) -> None:
        try:
            self._process = subprocess.Popen(
                list(self.command),
                cwd=self.cwd,
                env=self.env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=self.start_new_session,
            )
        except OSError as exc:
            raise OpenCodeServerError(
                f"failed to launch OpenCode server command {self.command!r}: {exc}"
            ) from exc
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._stdout_thread = threading.Thread(
            target=self._drain_pipe,
            args=(self._process.stdout, self._stdout_tail),
            name="opencode-server-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_pipe,
            args=(self._process.stderr, self._stderr_tail),
            name="opencode-server-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    @staticmethod
    def _drain_pipe(pipe: Any, destination: deque[str]) -> None:
        try:
            for line in pipe:
                destination.append(line)
        except (OSError, ValueError):
            pass
        finally:
            try:
                pipe.close()
            except OSError:
                pass

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            with self._condition:
                self._ensure_process_alive_locked()
            try:
                status = self._api_json(
                    "GET",
                    "/session/status",
                    timeout_s=min(0.5, max(0.05, deadline - time.monotonic())),
                )
                if isinstance(status, dict):
                    return
                last_error = OpenCodeProtocolError(
                    "GET /session/status did not return an object"
                )
            except (OpenCodeHttpError, urllib.error.URLError, OSError) as exc:
                last_error = exc
            time.sleep(0.05)
        raise OpenCodeServerError(
            f"OpenCode server was not ready within {self.startup_timeout_s:.3f}s"
            + (f": {last_error}" if last_error else "")
            + (f"; stderr tail: {self.stderr_tail}" if self.stderr_tail else "")
        )

    def _start_sse(self) -> None:
        path = self._path_with_directory("/event")
        connection = http.client.HTTPConnection(
            self.host, self.port, timeout=self.http_timeout_s
        )
        connection.request(
            "GET",
            path,
            headers={
                "Accept": "text/event-stream",
                "Authorization": self._authorization_header(),
                "Cache-Control": "no-cache",
            },
        )
        response = connection.getresponse()
        if response.status != 200:
            body = response.read().decode("utf-8", errors="replace")
            response.close()
            connection.close()
            raise OpenCodeHttpError("GET", path, response.status, body)
        content_type = response.getheader("Content-Type", "")
        if "text/event-stream" not in content_type:
            response.close()
            connection.close()
            raise OpenCodeProtocolError(
                f"GET /event returned unexpected Content-Type {content_type!r}"
            )
        if connection.sock is not None:
            connection.sock.settimeout(None)
        self._sse_connection = connection
        self._sse_response = response
        self._sse_thread = threading.Thread(
            target=self._sse_loop,
            name="opencode-server-sse",
            daemon=True,
        )
        self._sse_thread.start()

    def _wait_for_server_connected(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        with self._condition:
            while not self._server_connected:
                self._raise_async_error_locked()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OpenCodeServerError(
                        "OpenCode /event emitted no server.connected before timeout"
                    )
                self._condition.wait(timeout=min(remaining, 0.25))
                self._ensure_process_alive_locked()

    def _sse_loop(self) -> None:
        response = self._sse_response
        assert response is not None
        data_lines: list[str] = []
        try:
            while not self._sse_stop.is_set():
                raw_line = response.readline()
                if raw_line == b"":
                    if not self._sse_stop.is_set():
                        raise OpenCodeEventStreamError(
                            "OpenCode /event stream closed before driver shutdown"
                        )
                    break
                line = raw_line.decode("utf-8", errors="strict").rstrip("\r\n")
                if line == "":
                    if data_lines:
                        self._consume_sse_data("\n".join(data_lines))
                        data_lines.clear()
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    value = line[5:]
                    if value.startswith(" "):
                        value = value[1:]
                    data_lines.append(value)
            if data_lines and not self._sse_stop.is_set():
                self._consume_sse_data("\n".join(data_lines))
        except BaseException as exc:
            if not self._sse_stop.is_set():
                error = (
                    exc
                    if isinstance(exc, OpenCodeServerError)
                    else OpenCodeEventStreamError(str(exc))
                )
                with self._condition:
                    self._sse_error = error
                    self._condition.notify_all()

    def _consume_sse_data(self, data: str) -> None:
        try:
            value = json.loads(data)
        except json.JSONDecodeError as exc:
            raise OpenCodeEventStreamError(
                f"OpenCode /event emitted invalid JSON: {exc}: {data[-1000:]}"
            ) from exc
        if not isinstance(value, dict):
            raise OpenCodeEventStreamError(
                "OpenCode /event payload must be a JSON object"
            )
        captured_at_s = time.time()
        captured_monotonic_s = time.monotonic()
        callback_event: JsonObject | None = None
        tool_callbacks: list[JsonObject] = []
        actions: list[tuple[str, str]] = []
        with self._condition:
            event = OpenCodeServerEvent(
                captured_at_s=captured_at_s,
                captured_monotonic_s=captured_monotonic_s,
                message=copy.deepcopy(value),
            )
            self._events.append(event)
            callback_event = copy.deepcopy(value)
            tool_callbacks, actions = self._apply_event_locked(
                value, captured_at_s=captured_at_s
            )
            self._condition.notify_all()
        self._invoke_callbacks(callback_event, tool_callbacks, captured_at_s)
        for action, request_id in actions:
            self._perform_interactive_action(action, request_id)

    def _apply_event_locked(
        self, event: JsonObject, *, captured_at_s: float
    ) -> tuple[list[JsonObject], list[tuple[str, str]]]:
        kind = str(event.get("type") or "")
        properties = event.get("properties")
        props = properties if isinstance(properties, dict) else {}
        tool_callbacks: list[JsonObject] = []
        actions: list[tuple[str, str]] = []
        if kind == "server.connected":
            self._server_connected = True
            return tool_callbacks, actions

        if kind == "session.created":
            info = props.get("info")
            if isinstance(info, dict):
                parent_id = info.get("parentID")
                if parent_id is None or self._is_tracked_locked(str(parent_id)):
                    self._track_session_locked(info)
            return tool_callbacks, actions

        session_id = self._event_session_id(event)
        if session_id is None:
            return tool_callbacks, actions
        tracked = self._is_tracked_locked(session_id)

        if kind in {"permission.asked", "permission.updated"}:
            request_id = props.get("id")
            if isinstance(request_id, str) and request_id:
                if tracked:
                    actions.append(("permission", request_id))
                else:
                    self._deferred_permissions[request_id] = copy.deepcopy(props)
            return tool_callbacks, actions
        if kind == "question.asked":
            request_id = props.get("id")
            if isinstance(request_id, str) and request_id:
                if tracked:
                    actions.append(("question", request_id))
                else:
                    self._deferred_questions[request_id] = copy.deepcopy(props)
            return tool_callbacks, actions
        if not tracked:
            return tool_callbacks, actions

        if kind == "session.status":
            status_obj = props.get("status")
            status = status_obj.get("type") if isinstance(status_obj, dict) else None
            if isinstance(status, str):
                self._session_status[session_id] = status
                if (
                    self._active_phase is not None
                    and session_id == self.root_session_id
                ):
                    if status in {"busy", "retry"}:
                        self._phase_activity = True
                    if status == "busy":
                        self._phase_busy_seen = True
                    elif status == "retry":
                        self._phase_retry_count += 1
                    elif status == "idle":
                        self._phase_idle = True
            return tool_callbacks, actions

        if kind == "session.idle":
            self._session_status[session_id] = "idle"
            if self._active_phase is not None and session_id == self.root_session_id:
                self._phase_idle = True
            return tool_callbacks, actions

        if kind == "session.error":
            error_record = {
                "source": "session.error",
                "sessionID": session_id,
                "error": copy.deepcopy(props.get("error")),
                "captured_at_s": captured_at_s,
                "phase": self._active_phase,
            }
            self._errors.append(error_record)
            if session_id == self.root_session_id:
                self._root_errors.append(error_record)
                if self._active_phase is not None:
                    self._phase_activity = True
            return tool_callbacks, actions

        if kind == "message.updated":
            info = props.get("info")
            if isinstance(info, dict):
                if (
                    self._active_phase is not None
                    and session_id == self.root_session_id
                ):
                    self._phase_activity = True
                error = info.get("error")
                if error is not None and info.get("role") == "assistant":
                    self._record_message_error_locked(info, error)
            return tool_callbacks, actions

        if kind == "message.part.updated":
            part = props.get("part")
            if isinstance(part, dict):
                if (
                    self._active_phase is not None
                    and session_id == self.root_session_id
                ):
                    self._phase_activity = True
                if part.get("type") == "tool":
                    callback = self._update_tool_locked(
                        part,
                        captured_at_s=captured_at_s,
                        phase=self._active_phase or "initial",
                        source="sse",
                    )
                    if callback is not None:
                        tool_callbacks.append(callback)
                    state = part.get("state")
                    metadata = (
                        state.get("metadata") if isinstance(state, dict) else None
                    )
                    part_metadata = part.get("metadata")
                    for candidate in (metadata, part_metadata):
                        if not isinstance(candidate, dict):
                            continue
                        child_id = candidate.get("sessionId") or candidate.get(
                            "sessionID"
                        )
                        if isinstance(child_id, str) and child_id:
                            parent_id = (
                                candidate.get("parentSessionId")
                                or candidate.get("parentSessionID")
                                or session_id
                            )
                            self._session_parent[child_id] = str(parent_id)
                            self._session_status.setdefault(child_id, "idle")
            return tool_callbacks, actions
        return tool_callbacks, actions

    def _update_tool_locked(
        self,
        part: JsonObject,
        *,
        captured_at_s: float,
        phase: str,
        source: str,
    ) -> JsonObject | None:
        state_obj = part.get("state")
        state = state_obj if isinstance(state_obj, dict) else {}
        session_id = str(part.get("sessionID") or "")
        message_id = str(part.get("messageID") or "")
        part_id = str(part.get("id") or "")
        call_id = str(part.get("callID") or part_id)
        if not session_id or not message_id or not part_id:
            return None
        key = f"{session_id}:{message_id}:{call_id or part_id}"
        record = self._tools.get(key)
        if record is None:
            record = _MutableTool(
                key=key,
                session_id=session_id,
                parent_session_id=self._session_parent.get(session_id),
                message_id=message_id,
                part_id=part_id,
                call_id=call_id,
                tool_name=str(part.get("tool") or "unknown"),
                phase=phase,
                transitions=[],
            )
            self._tools[key] = record
            self._tool_order.append(key)
        status = str(state.get("status") or "unknown")
        timing = state.get("time") if isinstance(state.get("time"), dict) else {}
        native_start = self._milliseconds_to_seconds(timing.get("start"))
        native_end = self._milliseconds_to_seconds(timing.get("end"))
        if status == "pending" and record.enqueued_time_s is None:
            record.enqueued_time_s = captured_at_s
        if status in {"running", "completed", "error"}:
            candidate = native_start or captured_at_s
            if record.start_time_s is None or candidate < record.start_time_s:
                record.start_time_s = candidate
            if record.enqueued_time_s is None:
                record.enqueued_time_s = record.start_time_s
        if status in self._TERMINAL_TOOL_STATES:
            candidate_end = native_end or captured_at_s
            if record.end_time_s is None or candidate_end > record.end_time_s:
                record.end_time_s = candidate_end
        record.status = status
        if "input" in state:
            record.input = copy.deepcopy(state.get("input"))
        if "output" in state:
            record.output = copy.deepcopy(state.get("output"))
        if "error" in state:
            record.error = copy.deepcopy(state.get("error"))
        if "metadata" in state:
            record.metadata = copy.deepcopy(state.get("metadata"))
        transitions = record.transitions
        assert transitions is not None
        if status in {"pending", "running", "completed", "error"} and (
            status not in transitions
        ):
            transitions.append(status)
            return {
                "type": "opencode.tool.lifecycle",
                "source": source,
                "phase": record.phase,
                "sessionID": session_id,
                "parentSessionID": record.parent_session_id,
                "messageID": message_id,
                "partID": part_id,
                "callID": call_id,
                "tool": record.tool_name,
                "status": status,
                "captured_at_s": captured_at_s,
                "state": copy.deepcopy(state),
            }
        return None

    @staticmethod
    def _milliseconds_to_seconds(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) / 1000.0

    def _track_session_locked(self, info: Mapping[str, Any]) -> None:
        session_id = info.get("id")
        if not isinstance(session_id, str) or not session_id:
            return
        parent_id = info.get("parentID")
        self._sessions[session_id] = copy.deepcopy(dict(info))
        self._session_parent[session_id] = (
            str(parent_id) if isinstance(parent_id, str) and parent_id else None
        )
        self._session_status.setdefault(session_id, "idle")

    def _is_tracked_locked(self, session_id: str) -> bool:
        if session_id == self.root_session_id:
            return True
        current: str | None = session_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            parent = self._session_parent.get(current)
            if parent == self.root_session_id:
                return True
            current = parent
        return False

    @staticmethod
    def _event_session_id(event: Mapping[str, Any]) -> str | None:
        props = event.get("properties")
        if not isinstance(props, Mapping):
            return None
        session_id = props.get("sessionID")
        if isinstance(session_id, str) and session_id:
            return session_id
        info = props.get("info")
        if isinstance(info, Mapping):
            value = info.get("sessionID") or info.get("id")
            if isinstance(value, str) and value:
                return value
        part = props.get("part")
        if isinstance(part, Mapping):
            value = part.get("sessionID")
            if isinstance(value, str) and value:
                return value
        return None

    def _record_message_error_locked(self, info: JsonObject, error: Any) -> None:
        record = {
            "source": "message.updated",
            "sessionID": info.get("sessionID"),
            "messageID": info.get("id"),
            "error": copy.deepcopy(error),
            "phase": self._active_phase,
        }
        identity = (
            record["source"],
            record["sessionID"],
            record["messageID"],
            json.dumps(record["error"], sort_keys=True, default=str),
        )
        for existing in self._errors:
            existing_identity = (
                existing.get("source"),
                existing.get("sessionID"),
                existing.get("messageID"),
                json.dumps(existing.get("error"), sort_keys=True, default=str),
            )
            if existing_identity == identity:
                return
        self._errors.append(record)
        if info.get("sessionID") == self.root_session_id:
            self._root_errors.append(record)

    def _invoke_callbacks(
        self,
        raw_event: JsonObject,
        tool_events: Sequence[JsonObject],
        captured_at_s: float,
    ) -> None:
        try:
            if self.event_callback is not None:
                self.event_callback(copy.deepcopy(raw_event), captured_at_s)
            if self.tool_event_callback is not None:
                for event in tool_events:
                    self.tool_event_callback(copy.deepcopy(event), captured_at_s)
        except BaseException as exc:
            error = OpenCodeEventCallbackError(
                f"OpenCode live event callback failed: {exc}"
            )
            error.__cause__ = exc
            with self._condition:
                self._callback_error = error
                self._condition.notify_all()

    def _drain_deferred_actions(self) -> None:
        actions: list[tuple[str, str]] = []
        with self._condition:
            for request_id, props in tuple(self._deferred_permissions.items()):
                session_id = props.get("sessionID")
                if isinstance(session_id, str) and self._is_tracked_locked(session_id):
                    actions.append(("permission", request_id))
                    del self._deferred_permissions[request_id]
            for request_id, props in tuple(self._deferred_questions.items()):
                session_id = props.get("sessionID")
                if isinstance(session_id, str) and self._is_tracked_locked(session_id):
                    actions.append(("question", request_id))
                    del self._deferred_questions[request_id]
        for action, request_id in actions:
            self._perform_interactive_action(action, request_id)

    def _perform_interactive_action(self, action: str, request_id: str) -> None:
        try:
            escaped = urllib.parse.quote(request_id, safe="")
            if action == "permission":
                self._api_json(
                    "POST",
                    f"/permission/{escaped}/reply",
                    body={"reply": self.permission_reply},
                )
            elif action == "question" and self.reject_questions:
                self._api_json("POST", f"/question/{escaped}/reject", body=None)
        except BaseException as exc:
            with self._condition:
                self._sse_error = OpenCodeServerError(
                    f"failed to handle OpenCode {action} request {request_id!r}: {exc}"
                )
                self._condition.notify_all()

    def _api_json(
        self,
        method: str,
        path: str,
        *,
        body: Any = ...,
        timeout_s: float | None = None,
        expected_statuses: set[int] | None = None,
    ) -> Any:
        request_path = self._path_with_directory(path)
        url = f"http://{self.host}:{self.port}{request_path}"
        headers = {
            "Accept": "application/json",
            "Authorization": self._authorization_header(),
        }
        data: bytes | None = None
        if body is not ...:
            data = json.dumps(body).encode("utf-8") if body is not None else b""
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url, data=data, headers=headers, method=method.upper()
        )
        allowed = expected_statuses or set(range(200, 300))
        try:
            with urllib.request.urlopen(
                request, timeout=timeout_s or self.http_timeout_s
            ) as response:
                status = int(response.status)
                payload = response.read()
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise OpenCodeHttpError(
                method.upper(), request_path, exc.code, response_body
            ) from exc
        if status not in allowed:
            raise OpenCodeHttpError(
                method.upper(),
                request_path,
                status,
                payload.decode("utf-8", errors="replace"),
            )
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise OpenCodeProtocolError(
                f"OpenCode {method.upper()} {request_path} returned invalid JSON"
            ) from exc

    def _path_with_directory(self, path: str) -> str:
        separator = "&" if "?" in path else "?"
        return (
            path
            + separator
            + urllib.parse.urlencode({"directory": str(self.workspace_dir)})
        )

    def _authorization_header(self) -> str:
        token = base64.b64encode(
            f"{self.username}:{self.password}".encode("utf-8")
        ).decode("ascii")
        return f"Basic {token}"

    def _ensure_healthy(self) -> None:
        with self._condition:
            self._raise_async_error_locked()
            self._ensure_process_alive_locked()
        if not self._started:
            raise OpenCodeServerError("OpenCode server/session has not been started")
        if self._closing or self._closed:
            raise OpenCodeServerError("OpenCode driver is closing or closed")

    def _raise_async_error_locked(self) -> None:
        if self._callback_error is not None:
            raise self._callback_error
        if self._sse_error is not None:
            raise self._sse_error

    def _ensure_process_alive_locked(self) -> None:
        if not self._managed_process:
            return
        process = self._process
        if process is None:
            if self._closing or self._closed:
                return
            raise OpenCodeServerError("OpenCode server process was not launched")
        returncode = process.poll()
        if returncode is not None and not self._closing:
            raise OpenCodeProcessExited(returncode, self.stderr_tail)

    @staticmethod
    def _final_output(snapshot: OpenCodeSnapshot) -> str | None:
        messages = snapshot.messages_by_session.get(snapshot.root_session_id, ())
        for row in reversed(messages):
            info = row.get("info") if isinstance(row, Mapping) else None
            if not isinstance(info, Mapping) or info.get("role") != "assistant":
                continue
            parts = row.get("parts")
            if not isinstance(parts, Sequence):
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

    def close(self) -> None:
        """Stop SSE and terminate only this request's server process group."""

        if self._closed:
            return
        self._closing = True
        self._sse_stop.set()
        response = self._sse_response
        connection = self._sse_connection
        if response is not None:
            try:
                response.close()
            except OSError:
                pass
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        process = self._process
        if process is not None and process.poll() is None:
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
        for thread in (
            self._sse_thread,
            self._stdout_thread,
            self._stderr_thread,
        ):
            if thread is not None and thread is not current:
                thread.join(timeout=self.shutdown_timeout_s)
        self._closed = True
        self._closing = False

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


__all__ = [
    "DEFAULT_OPENCODE_BINARY",
    "OpenCodeConversationResult",
    "OpenCodeEventCallbackError",
    "OpenCodeEventStreamError",
    "OpenCodeHttpError",
    "OpenCodePhaseResult",
    "OpenCodeProcessExited",
    "OpenCodePromptTimeout",
    "OpenCodeProtocolError",
    "OpenCodeServerDriver",
    "OpenCodeServerError",
    "OpenCodeServerEvent",
    "OpenCodeSnapshot",
    "OpenCodeToolRecord",
    "build_opencode_server_command",
]
