#!/usr/bin/env python3

from __future__ import annotations

import base64
import json
import queue
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from opencode_server import (
    OpenCodeServerDriver,
    build_opencode_server_command,
)


def _session(session_id: str, parent_id: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": session_id,
        "directory": "/workspace",
        "title": session_id,
        "time": {"created": 1, "updated": 1},
    }
    if parent_id is not None:
        value["parentID"] = parent_id
    return value


def _status(session_id: str, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "session.status",
        "properties": {"sessionID": session_id, "status": {"type": status, **extra}},
    }


def _tool(
    session_id: str,
    message_id: str,
    part_id: str,
    call_id: str,
    tool_name: str,
    status: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {"status": status, "input": {"command": part_id}}
    if status == "pending":
        state["raw"] = ""
    else:
        state["time"] = {"start": start_ms}
        if end_ms is not None:
            state["time"]["end"] = end_ms
        if status == "completed":
            state.update({"output": f"output:{part_id}", "title": part_id})
        elif status == "error":
            state["error"] = f"error:{part_id}"
    if metadata is not None:
        state["metadata"] = metadata
    return {
        "type": "message.part.updated",
        "properties": {
            "part": {
                "id": part_id,
                "sessionID": session_id,
                "messageID": message_id,
                "type": "tool",
                "callID": call_id,
                "tool": tool_name,
                "state": state,
            }
        },
    }


def _assistant_message(
    session_id: str,
    message_id: str,
    text: str,
    *,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "id": message_id,
        "sessionID": session_id,
        "role": "assistant",
        "time": {"created": 1, "completed": 2},
    }
    if error is not None:
        info["error"] = error
    return {
        "info": info,
        "parts": [
            {
                "id": f"text-{message_id}",
                "sessionID": session_id,
                "messageID": message_id,
                "type": "text",
                "text": text,
            }
        ],
    }


class _MockState:
    def __init__(self) -> None:
        self.username = "test-user"
        self.password = "test-password"
        self.root = _session("ses_root")
        self.children: dict[str, list[dict[str, Any]]] = {"ses_root": []}
        self.messages: dict[str, list[dict[str, Any]]] = {"ses_root": []}
        self.prompt_scripts: list[Callable[["_MockState"], None]] = []
        self.abort_script: Callable[["_MockState"], None] | None = None
        self.event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.event_connected = threading.Event()
        self.operations: list[str] = []
        self.prompts: list[dict[str, Any]] = []
        self.permission_replies: list[tuple[str, dict[str, Any]]] = []
        self.question_rejections: list[str] = []
        self.abort_calls = 0
        self.auth_failures = 0
        self.lock = threading.Lock()

    @property
    def authorization(self) -> str:
        token = base64.b64encode(
            f"{self.username}:{self.password}".encode("utf-8")
        ).decode("ascii")
        return f"Basic {token}"

    def emit(self, event: dict[str, Any]) -> None:
        self.event_queue.put(event)

    def run_script(self, script: Callable[["_MockState"], None]) -> None:
        def target() -> None:
            time.sleep(0.01)
            script(self)

        threading.Thread(target=target, daemon=True).start()


class _MockServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, state: _MockState) -> None:
        super().__init__(("127.0.0.1", 0), _MockHandler)
        self.state = state


class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _MockServer

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _authorized(self) -> bool:
        if self.headers.get("Authorization") == self.server.state.authorization:
            return True
        self.server.state.auth_failures += 1
        self.send_response(401)
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def _path(self) -> str:
        return urllib.parse.urlsplit(self.path).path

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length) if length else b""
        return json.loads(data) if data else None

    def _json(self, status: int, value: Any) -> None:
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _no_content(self) -> None:
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._authorized():
            return
        path = self._path()
        state = self.server.state
        if path == "/session/status":
            self._json(200, {})
            return
        if path == "/event":
            with state.lock:
                state.operations.append("event_connect")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(
                    b'data: {"type":"server.connected","properties":{}}\n\n'
                )
                self.wfile.flush()
                state.event_connected.set()
                while True:
                    event = state.event_queue.get()
                    if event is None:
                        break
                    data = (
                        "data: " + json.dumps(event, separators=(",", ":")) + "\n\n"
                    ).encode("utf-8")
                    self.wfile.write(data)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        pieces = path.strip("/").split("/")
        if len(pieces) == 3 and pieces[0] == "session":
            session_id, resource = pieces[1], pieces[2]
            if resource == "children":
                self._json(200, state.children.get(session_id, []))
                return
            if resource == "message":
                self._json(200, state.messages.get(session_id, []))
                return
        self._json(404, {"error": path})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        path = self._path()
        state = self.server.state
        if path == "/session":
            body = self._read_json()
            with state.lock:
                state.operations.append("session_create")
                state.session_create_body = body
            self._json(200, state.root)
            return
        pieces = path.strip("/").split("/")
        if len(pieces) == 3 and pieces[0] == "session" and pieces[2] == "prompt_async":
            body = self._read_json()
            with state.lock:
                state.operations.append(f"prompt:{pieces[1]}")
                state.prompts.append(body)
                index = len(state.prompts) - 1
            self._no_content()
            state.run_script(state.prompt_scripts[index])
            return
        if len(pieces) == 3 and pieces[:1] == ["session"] and pieces[2] == "abort":
            self._read_json()
            state.abort_calls += 1
            self._json(200, True)
            if state.abort_script is not None:
                state.run_script(state.abort_script)
            return
        if len(pieces) == 3 and pieces[0] == "permission" and pieces[2] == "reply":
            body = self._read_json()
            state.permission_replies.append((pieces[1], body))
            self._json(200, True)
            return
        if len(pieces) == 3 and pieces[0] == "question" and pieces[2] == "reject":
            self._read_json()
            state.question_rejections.append(pieces[1])
            self._json(200, True)
            return
        self._json(404, {"error": path})


@pytest.fixture
def mock_server() -> tuple[_MockState, str]:
    state = _MockState()
    server = _MockServer(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state, url
    finally:
        state.event_queue.put(None)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _driver(
    tmp_path: Path,
    state: _MockState,
    url: str,
    **kwargs: Any,
) -> OpenCodeServerDriver:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return OpenCodeServerDriver(
        workspace_dir=workspace,
        model="qwen/model",
        base_url=url,
        username=state.username,
        password=state.password,
        startup_timeout_s=2,
        http_timeout_s=2,
        abort_grace_s=1,
        event_settle_s=0,
        **kwargs,
    )


def test_sse_precedes_create_and_two_prompts_share_one_session(
    tmp_path: Path, mock_server: tuple[_MockState, str]
) -> None:
    state, url = mock_server
    base_ms = int(time.time() * 1000)

    def initial(mock: _MockState) -> None:
        mock.emit(_status("ses_root", "busy"))
        mock.emit(_status("ses_root", "retry", attempt=1, message="retry", next=1))
        mock.emit(_tool("ses_root", "msg-1", "part-1", "call-1", "bash", "pending"))
        mock.emit(
            _tool(
                "ses_root",
                "msg-1",
                "part-1",
                "call-1",
                "bash",
                "running",
                start_ms=base_ms + 10,
            )
        )
        # OpenCode may replace state.time.start on a later metadata update.  The
        # adapter must preserve the first running timestamp.
        mock.emit(
            _tool(
                "ses_root",
                "msg-1",
                "part-1",
                "call-1",
                "bash",
                "completed",
                start_ms=base_ms + 40,
                end_ms=base_ms + 70,
            )
        )
        mock.emit(_status("ses_root", "idle"))

    def verification(mock: _MockState) -> None:
        mock.emit(_status("ses_root", "busy"))
        mock.emit(_tool("ses_root", "msg-2", "part-2", "call-2", "read", "pending"))
        mock.emit(
            _tool(
                "ses_root",
                "msg-2",
                "part-2",
                "call-2",
                "read",
                "running",
                start_ms=base_ms + 100,
            )
        )
        mock.emit(
            _tool(
                "ses_root",
                "msg-2",
                "part-2",
                "call-2",
                "read",
                "completed",
                start_ms=base_ms + 100,
                end_ms=base_ms + 130,
            )
        )
        mock.emit(_status("ses_root", "idle"))

    state.prompt_scripts = [initial, verification]
    state.messages["ses_root"] = [
        _assistant_message("ses_root", "msg-1", "initial answer"),
        _assistant_message("ses_root", "msg-2", "verified answer"),
    ]
    raw_types: list[str] = []
    transitions: list[tuple[str, str]] = []
    driver = _driver(
        tmp_path,
        state,
        url,
        event_callback=lambda raw, ts: raw_types.append(str(raw.get("type"))),
        tool_event_callback=lambda event, ts: transitions.append(
            (str(event["callID"]), str(event["status"]))
        ),
    )
    with driver:
        result = driver.run_conversation(
            "solve",
            timeout_s=3,
            verification_prompts=["verify"],
            verification_needed=lambda snapshot: len(snapshot.phases) == 1,
        )

    assert state.operations[:2] == ["event_connect", "session_create"]
    assert state.operations.count("session_create") == 1
    assert state.operations.count("prompt:ses_root") == 2
    assert [item["parts"][0]["text"] for item in state.prompts] == [
        "solve",
        "verify",
    ]
    assert all(
        item["model"] == {"providerID": "mars", "modelID": "qwen/model"}
        for item in state.prompts
    )
    assert state.session_create_body["permission"] == [
        {"permission": "question", "pattern": "*", "action": "deny"},
        {"permission": "plan_enter", "pattern": "*", "action": "deny"},
        {"permission": "plan_exit", "pattern": "*", "action": "deny"},
    ]
    assert result.root_session_id == "ses_root"
    assert result.status == "completed"
    assert [phase.phase for phase in result.phases] == ["initial", "verification_1"]
    assert result.phases[0].busy_seen
    assert result.phases[0].retry_count == 1
    assert result.final_output == "verified answer"
    assert result.tools[0].start_time_s == pytest.approx((base_ms + 10) / 1000)
    # Epoch-sized millisecond timestamps lose a few sub-microseconds when
    # converted to binary floating-point seconds.
    assert result.tools[0].duration_s == pytest.approx(0.060, abs=1e-6)
    assert transitions == [
        ("call-1", "pending"),
        ("call-1", "running"),
        ("call-1", "completed"),
        ("call-2", "pending"),
        ("call-2", "running"),
        ("call-2", "completed"),
    ]
    assert raw_types[0] == "server.connected"
    assert state.auth_failures == 0


def test_parallel_root_and_child_tools_permissions_and_reconcile(
    tmp_path: Path, mock_server: tuple[_MockState, str]
) -> None:
    state, url = mock_server
    base_ms = int(time.time() * 1000)
    child = _session("ses_child", "ses_root")
    state.children["ses_root"] = [child]
    state.children["ses_child"] = []
    state.messages["ses_root"] = [_assistant_message("ses_root", "root-msg", "done")]
    state.messages["ses_child"] = [
        _assistant_message("ses_child", "child-msg", "child done")
    ]

    def parallel(mock: _MockState) -> None:
        mock.emit(_status("ses_root", "busy"))
        mock.emit({"type": "session.created", "properties": {"info": child}})
        mock.emit(_tool("ses_root", "root-msg", "root-a", "call-a", "bash", "pending"))
        mock.emit(_tool("ses_root", "root-msg", "root-b", "call-b", "task", "pending"))
        mock.emit(
            _tool(
                "ses_root",
                "root-msg",
                "root-a",
                "call-a",
                "bash",
                "running",
                start_ms=base_ms + 10,
            )
        )
        mock.emit(
            _tool(
                "ses_root",
                "root-msg",
                "root-b",
                "call-b",
                "task",
                "running",
                start_ms=base_ms + 11,
                metadata={"sessionId": "ses_child", "parentSessionId": "ses_root"},
            )
        )
        mock.emit(_status("ses_child", "busy"))
        mock.emit(
            {
                "type": "permission.asked",
                "properties": {
                    "id": "per_child",
                    "sessionID": "ses_child",
                    "permission": "bash",
                    "patterns": ["*"],
                },
            }
        )
        mock.emit(
            _tool("ses_child", "child-msg", "child-a", "child-call", "read", "pending")
        )
        mock.emit(
            _tool(
                "ses_child",
                "child-msg",
                "child-a",
                "child-call",
                "read",
                "running",
                start_ms=base_ms + 20,
            )
        )
        mock.emit(
            _tool(
                "ses_child",
                "child-msg",
                "child-a",
                "child-call",
                "read",
                "completed",
                start_ms=base_ms + 20,
                end_ms=base_ms + 30,
            )
        )
        mock.emit(
            {
                "type": "question.asked",
                "properties": {
                    "id": "que_child",
                    "sessionID": "ses_child",
                    "questions": [],
                },
            }
        )
        mock.emit(_status("ses_child", "idle"))
        mock.emit(
            _tool(
                "ses_root",
                "root-msg",
                "root-a",
                "call-a",
                "bash",
                "completed",
                start_ms=base_ms + 10,
                end_ms=base_ms + 50,
            )
        )
        mock.emit(
            _tool(
                "ses_root",
                "root-msg",
                "root-b",
                "call-b",
                "task",
                "completed",
                start_ms=base_ms + 11,
                end_ms=base_ms + 55,
                metadata={"sessionId": "ses_child", "parentSessionId": "ses_root"},
            )
        )
        mock.emit(_status("ses_root", "idle"))

    state.prompt_scripts = [parallel]
    live: list[tuple[str, str, str]] = []
    driver = _driver(
        tmp_path,
        state,
        url,
        tool_event_callback=lambda event, ts: live.append(
            (str(event["sessionID"]), str(event["callID"]), str(event["status"]))
        ),
    )
    with driver:
        result = driver.run_conversation("parallel", timeout_s=3)

    assert result.status == "completed"
    assert set(result.sessions) == {"ses_root", "ses_child"}
    assert set(result.messages_by_session) == {"ses_root", "ses_child"}
    assert {tool.call_id for tool in result.tools} == {"call-a", "call-b", "child-call"}
    assert (
        next(
            tool for tool in result.tools if tool.call_id == "child-call"
        ).parent_session_id
        == "ses_root"
    )
    assert ("ses_child", "child-call", "running") in live
    assert state.permission_replies == [("per_child", {"reply": "once"})]
    assert state.question_rejections == ["que_child"]


def test_root_session_error_is_native_terminal_error(
    tmp_path: Path, mock_server: tuple[_MockState, str]
) -> None:
    state, url = mock_server
    error = {"name": "APIError", "data": {"message": "provider failed"}}
    state.messages["ses_root"] = [
        _assistant_message("ses_root", "msg-error", "", error=error)
    ]

    def fail(mock: _MockState) -> None:
        mock.emit(_status("ses_root", "busy"))
        mock.emit(
            {
                "type": "session.error",
                "properties": {"sessionID": "ses_root", "error": error},
            }
        )
        mock.emit(
            {
                "type": "message.updated",
                "properties": {
                    "info": {
                        "id": "msg-error",
                        "sessionID": "ses_root",
                        "role": "assistant",
                        "error": error,
                    }
                },
            }
        )
        mock.emit(_status("ses_root", "idle"))

    state.prompt_scripts = [fail]
    driver = _driver(tmp_path, state, url)
    with driver:
        result = driver.run_conversation("fail", timeout_s=3)

    assert result.status == "error"
    assert result.phases[0].status == "error"
    assert {item["source"] for item in result.errors} == {
        "session.error",
        "message.updated",
    }


def test_timeout_aborts_and_reconciles_running_tool(
    tmp_path: Path, mock_server: tuple[_MockState, str]
) -> None:
    state, url = mock_server
    base_ms = int(time.time() * 1000)
    state.messages["ses_root"] = [_assistant_message("ses_root", "msg", "aborted")]

    def hangs(mock: _MockState) -> None:
        mock.emit(_status("ses_root", "busy"))
        mock.emit(_tool("ses_root", "msg", "part", "call", "bash", "pending"))
        mock.emit(
            _tool(
                "ses_root",
                "msg",
                "part",
                "call",
                "bash",
                "running",
                start_ms=base_ms,
            )
        )

    def abort(mock: _MockState) -> None:
        mock.emit(
            _tool(
                "ses_root",
                "msg",
                "part",
                "call",
                "bash",
                "error",
                start_ms=base_ms,
                end_ms=base_ms + 100,
            )
        )
        mock.emit(_status("ses_root", "idle"))

    state.prompt_scripts = [hangs]
    state.abort_script = abort
    driver = _driver(tmp_path, state, url)
    with driver:
        result = driver.run_conversation("hang", timeout_s=0.15)

    assert result.status == "timeout"
    assert result.timed_out
    assert state.abort_calls == 1
    assert result.phases[0].timed_out
    assert result.phases[0].abort_requested
    assert result.tools[0].status == "error"
    assert result.tools[0].transitions == ("pending", "running", "error")


def test_command_builder_wraps_resolved_pinned_server_and_env(tmp_path: Path) -> None:
    seen: list[tuple[str, ...]] = []

    def wrap(command: Any) -> list[str]:
        seen.append(tuple(command))
        return ["/usr/bin/env", *command]

    driver = OpenCodeServerDriver(
        workspace_dir=tmp_path,
        model="model",
        binary="/pinned/opencode",
        port=32123,
        command_builder=wrap,
        env={"PATH": "/bin"},
        username="user",
        password="password",
    )
    try:
        assert seen == [
            (
                "/pinned/opencode",
                "--pure",
                "serve",
                "--hostname",
                "127.0.0.1",
                "--port",
                "32123",
            )
        ]
        assert driver.command == ("/usr/bin/env", *seen[0])
        assert driver.env["OPENCODE_SERVER_USERNAME"] == "user"
        assert driver.env["OPENCODE_SERVER_PASSWORD"] == "password"
        assert driver.env["OPENCODE_EXPERIMENTAL_BACKGROUND_SUBAGENTS"] == "false"
    finally:
        driver.close()


def test_command_constructor_rejects_non_loopback() -> None:
    assert build_opencode_server_command("opencode", port=1234)[1:3] == [
        "--pure",
        "serve",
    ]
    with pytest.raises(ValueError, match="127.0.0.1"):
        build_opencode_server_command("opencode", host="0.0.0.0", port=1234)
