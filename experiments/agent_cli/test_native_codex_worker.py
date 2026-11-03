from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import native_request_worker as native_worker
import smoke_test
from codex_app_server import CodexTurnResult
from event_normalizer import load_jsonl
from frontend_contract import AgentRequestContext
from lifecycle_collector import LifecycleCollector
from native_request_worker import run_request
from request_worker import _working_bwrap
from run_workload import PINNED


CODEX_BINARY = Path(PINNED["codex"]["binary"])


def _require_native_runtime() -> None:
    if not CODEX_BINARY.exists():
        pytest.skip(f"pinned Codex binary is missing: {CODEX_BINARY}")
    if not _working_bwrap():
        pytest.skip("bubblewrap is required for the native Codex integration gate")


def _run_codex(
    root: Path,
    server: smoke_test.MockOpenAIServer,
    *,
    task: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    return run_request(
        agent="codex",
        agent_bin=CODEX_BINARY,
        task=task,
        task_sha256=f"{name}-sha256",
        request_id=name,
        mars_job_id=name,
        arrival_time_s=time.time(),
        workspace_root=root / "workspaces",
        local_inputs_dir=root / "localized_inputs",
        request_run_dir=root / "request",
        model=smoke_test.MODEL,
        base_url=server.base_url,
        api_key="EMPTY",
        context_window=32768,
        max_output_tokens=1024,
        max_iterations=5,
        temperature=0.0,
        timeout_s=120.0,
        require_bwrap=True,
    )


def _collector(tmp_path: Path, name: str) -> LifecycleCollector:
    collector = LifecycleCollector(
        AgentRequestContext(
            request_id=name,
            job_id=name,
            prompt="gate",
            workspace_dir=tmp_path / name / "workspace",
            model=smoke_test.MODEL,
            backend_base_url="http://127.0.0.1:1/v1",
            arrival_time_s=time.time(),
            arrival_monotonic_s=time.monotonic(),
            max_iterations=5,
        )
    )
    collector.request_started()
    return collector


def _native(method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"method": method, "params": params, "emittedAtMs": int(time.time() * 1000)}


def _responses_sse(events: list[dict[str, Any]]) -> bytes:
    return b"".join(
        b"event: "
        + str(event["type"]).encode()
        + b"\n"
        + b"data: "
        + json.dumps(event, separators=(",", ":")).encode()
        + b"\n\n"
        for event in events
    )


def _text_response(response_id: str, text: str) -> bytes:
    item_id = f"item-{response_id}"
    return _responses_sse(
        [
            {
                "type": "response.created",
                "response": {"id": response_id, "status": "in_progress"},
            },
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "item_id": item_id,
                "delta": text,
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "id": item_id,
                    "content": [{"type": "output_text", "text": text}],
                    "status": "completed",
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 2,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 12,
                    },
                },
            },
        ]
    )


def _spawn_agent_response() -> bytes:
    alias = "multi_agent_v1__spawn_agent"
    arguments = json.dumps(
        {"message": "Return a short child result.", "fork_context": False},
        separators=(",", ":"),
    )
    return _responses_sse(
        [
            {
                "type": "response.created",
                "response": {"id": "resp-root-spawn", "status": "in_progress"},
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "item-root-spawn",
                    "call_id": "call-root-spawn",
                    "name": alias,
                    "arguments": "",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "item-root-spawn",
                "delta": arguments,
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "item_id": "item-root-spawn",
                "arguments": arguments,
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "item-root-spawn",
                    "call_id": "call-root-spawn",
                    "name": alias,
                    "arguments": arguments,
                    "status": "completed",
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp-root-spawn",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 5,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 15,
                    },
                },
            },
        ]
    )


def _wait_agent_response(agent_id: str) -> bytes:
    alias = "multi_agent_v1__wait_agent"
    arguments = json.dumps({"targets": [agent_id]}, separators=(",", ":"))
    return _responses_sse(
        [
            {
                "type": "response.created",
                "response": {"id": "resp-root-wait", "status": "in_progress"},
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "item-root-wait",
                    "call_id": "call-root-wait",
                    "name": alias,
                    "arguments": "",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "item-root-wait",
                "delta": arguments,
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "item_id": "item-root-wait",
                "arguments": arguments,
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "item-root-wait",
                    "call_id": "call-root-wait",
                    "name": alias,
                    "arguments": arguments,
                    "status": "completed",
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp-root-wait",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 5,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 15,
                    },
                },
            },
        ]
    )


def test_native_codex_subagent_namespace_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not CODEX_BINARY.exists():
        pytest.skip(f"pinned Codex binary is missing: {CODEX_BINARY}")
    toolchain_root = CODEX_BINARY.parents[4] / "toolchain"
    node_candidates = sorted(toolchain_root.glob("node-*/bin/node"))
    if not node_candidates:
        pytest.skip(f"pinned Node binary is missing below: {toolchain_root}")
    monkeypatch.setenv(
        "PATH",
        f"{node_candidates[-1].parent}:{os.environ.get('PATH', '')}",
    )

    requests: list[dict[str, Any]] = []
    request_metadata: list[dict[str, Any]] = []
    request_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            metadata_raw = self.headers.get("X-Codex-Turn-Metadata") or "{}"
            metadata = json.loads(metadata_raw)
            with request_lock:
                requests.append(payload)
                request_metadata.append(metadata)
            outputs = [
                item
                for item in payload.get("input") or []
                if isinstance(item, dict) and item.get("type") == "function_call_output"
            ]
            output_call_ids = {str(item.get("call_id") or "") for item in outputs}
            if metadata.get("subagent_kind"):
                body = _text_response("resp-child-done", "child complete")
            elif "call-root-wait" in output_call_ids:
                body = _text_response("resp-root-done", "root complete")
            elif "call-root-spawn" in output_call_ids:
                spawn_output = next(
                    item.get("output")
                    for item in outputs
                    if item.get("call_id") == "call-root-spawn"
                )
                parsed_output = json.loads(str(spawn_output))
                body = _wait_agent_response(str(parsed_output["agent_id"]))
            else:
                body = _spawn_agent_response()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = tmp_path / "codex-subagent"
    task = {
        "schema": "mas_task_v1",
        "request_id": "native-codex-subagent",
        "benchmark": "smoke",
        "task_id": "native-codex-subagent-1",
        "payload": {"text": "Spawn one subagent, wait for it, then finish."},
        "workspace_init": [{"path": "README.txt", "content": "subagent gate\n"}],
    }
    try:
        result = run_request(
            agent="codex",
            agent_bin=CODEX_BINARY,
            task=task,
            task_sha256="native-codex-subagent-sha256",
            request_id="native_codex_subagent",
            mars_job_id="native_codex_subagent",
            arrival_time_s=time.time(),
            workspace_root=root / "workspaces",
            local_inputs_dir=root / "localized_inputs",
            request_run_dir=root / "request",
            model=smoke_test.MODEL,
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            api_key="EMPTY",
            context_window=32768,
            max_output_tokens=1024,
            max_iterations=10,
            temperature=0.0,
            timeout_s=120.0,
            require_bwrap=False,
            codex_enable_subagents=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    trace = result["trace"]
    assert trace["status"] == "completed", trace.get("error")
    assert trace["codex_enable_subagents"] is True
    assert trace["codex_namespace_tool_compat"] == "flatten_restore_v1"
    assert trace["dynamic_graph"]["round_count"] >= 4
    assert any(metadata.get("subagent_kind") for metadata in request_metadata)
    assert len(requests) >= 4
    assert all(
        not any(tool.get("type") == "namespace" for tool in request["tools"])
        for request in requests
    )
    assert any(
        tool.get("name") == "multi_agent_v1__spawn_agent"
        for tool in requests[0]["tools"]
    )
    assert any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for request in requests[1:]
        for item in request.get("input") or []
    )
    backend_rows = load_jsonl(root / "request" / "backend_attempts.jsonl")
    started = [row for row in backend_rows if row.get("event") == "backend_attempt_started"]
    assert started
    assert all(row["responses_namespace_tool_compat"] is True for row in started)
    assert started[0]["flattened_namespace_count"] >= 1


def test_backend_events_wait_for_real_codex_turn_id(tmp_path: Path) -> None:
    collector = _collector(tmp_path, "pending-backend")
    bridge = native_worker._CodexLifecycleBridge(collector, max_iterations=5)
    bridge.backend_event(
        {
            "event": "backend_attempt_started",
            "backend_attempt_id": "attempt-1",
            "protocol": "responses",
        }
    )
    bridge.backend_event(
        {
            "event": "backend_attempt_first_token",
            "backend_attempt_id": "attempt-1",
        }
    )
    bridge.backend_event(
        {
            "event": "backend_attempt_ended",
            "backend_attempt_id": "attempt-1",
            "http_status": 200,
        }
    )
    assert collector.snapshot()["logical_llm_rounds"] == []

    bridge.native_event(
        _native(
            "turn/started",
            {
                "threadId": "thread-real",
                "turn": {"id": "turn-real", "status": "inProgress"},
            },
        ),
        time.time(),
    )
    snapshot = collector.snapshot()
    assert len(snapshot["logical_llm_rounds"]) == 1
    logical_round = snapshot["logical_llm_rounds"][0]
    assert logical_round["round_id"] == "turn-real:llm_0001"
    assert logical_round["metadata"]["native_turn_id"] == "turn-real"
    assert "turn_pending" not in str(snapshot)
    assert snapshot["backend_attempts"][0]["round_id"] == logical_round["round_id"]
    assert snapshot["backend_attempts"][0]["status"] == "completed"
    assert logical_round["first_token_time_s"] is not None

    bridge.native_event(
        _native(
            "rawResponse/completed",
            {
                "threadId": "thread-real",
                "turnId": "turn-real",
                "responseId": "response-real",
                "usage": {},
            },
        ),
        time.time(),
    )
    ended = collector.snapshot()["logical_llm_rounds"][0]
    assert ended["status"] == "completed"
    assert ended["native_response_id"] == "response-real"


def test_backend_end_after_turn_completed_is_not_left_pending(tmp_path: Path) -> None:
    collector = _collector(tmp_path, "terminal-backend-race")
    bridge = native_worker._CodexLifecycleBridge(collector, max_iterations=5)
    bridge.native_event(
        _native(
            "turn/started",
            {
                "threadId": "thread-real",
                "turn": {"id": "turn-real", "status": "inProgress"},
            },
        ),
        time.time(),
    )
    bridge.backend_event(
        {
            "event": "backend_attempt_started",
            "backend_attempt_id": "backend-final",
            "protocol": "responses",
        }
    )
    bridge.native_event(
        _native(
            "rawResponse/completed",
            {
                "threadId": "thread-real",
                "turnId": "turn-real",
                "responseId": "response-real",
                "usage": {},
            },
        ),
        time.time(),
    )
    bridge.native_event(
        _native(
            "turn/completed",
            {
                "threadId": "thread-real",
                "turn": {"id": "turn-real", "status": "completed"},
            },
        ),
        time.time(),
    )
    assert bridge.active_turn_id is None

    bridge.backend_event(
        {
            "event": "backend_attempt_ended",
            "backend_attempt_id": "backend-final",
            "protocol": "responses",
            "http_status": 200,
            "response_id": "response-real",
            "ttft_s": 0.1,
            "error": None,
        }
    )

    attempt = collector.snapshot()["backend_attempts"][0]
    assert attempt["status"] == "completed"
    assert attempt["http_status"] == 200
    assert bridge.pending_backend_events == []


def test_raw_backend_audit_reconciles_existing_projection(tmp_path: Path) -> None:
    top_level = {
        "attempt_id": "backend_0008",
        "backend_attempt_id": "backend_0008",
        "status": "running",
        "end_time_s": None,
        "end_monotonic_s": None,
        "duration_s": None,
        "http_status": None,
        "error": None,
        "metadata": {},
    }
    nested = dict(top_level)
    trace = {
        "backend_attempts": [top_level],
        "rounds": [{"backend_attempts": [nested]}],
        "backend_attempt_duration_unknown_count": 1,
        "backend_attempt_total_s": None,
    }
    raw_path = tmp_path / "backend_attempts.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "event": "backend_attempt_ended",
                "backend_attempt_id": "backend_0008",
                "ts": 20.0,
                "monotonic_s": 10.0,
                "duration_s": 2.5,
                "ttft_s": 0.1,
                "http_status": 200,
                "response_id": "response-8",
                "error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count = native_worker._reconcile_backend_attempts_from_raw(trace, raw_path)

    assert count == 1
    for attempt in (top_level, nested):
        assert attempt["status"] == "completed"
        assert attempt["http_status"] == 200
        assert attempt["duration_s"] == 2.5
        assert attempt["metadata"]["terminal_reconciled_from_raw_audit"] is True
    assert trace["backend_attempt_duration_unknown_count"] == 0
    assert trace["backend_attempt_total_s"] == 2.5


def test_missing_native_ids_remain_null_and_do_not_create_cross_turn_edges(
    tmp_path: Path,
) -> None:
    collector = _collector(tmp_path, "missing-native-id")
    bridge = native_worker._CodexLifecycleBridge(collector, max_iterations=5)
    bridge.native_event(
        _native(
            "turn/started",
            {
                "threadId": "thread-real",
                "turn": {"id": "turn-real", "status": "inProgress"},
            },
        ),
        time.time(),
    )
    bridge.backend_event(
        {
            "event": "backend_attempt_started",
            "backend_attempt_id": "attempt-1",
            "protocol": "responses",
        }
    )
    bridge.native_event(
        _native(
            "rawResponseItem/completed",
            {
                "threadId": "thread-real",
                "turnId": "turn-real",
                "item": {
                    "type": "function_call",
                    "id": "response-item-only",
                    "name": "exec_command",
                },
            },
        ),
        time.time(),
    )
    queued = collector.snapshot()["tool_calls"][0]
    assert queued["tool_call_id"] == "response-item-only"
    assert queued["native_tool_call_id"] is None

    bridge.native_event(
        _native(
            "rawResponse/completed",
            {
                "threadId": "thread-real",
                "turnId": "turn-real",
                "usage": {},
            },
        ),
        time.time(),
    )
    bridge.native_event(
        _native(
            "turn/completed",
            {
                "threadId": "thread-real",
                "turn": {"id": "turn-real", "status": "completed"},
            },
        ),
        time.time(),
    )
    bridge.native_event(
        _native(
            "item/started",
            {
                "threadId": "thread-real",
                "item": {
                    "type": "commandExecution",
                    "id": "orphan-native-tool",
                    "status": "inProgress",
                },
            },
        ),
        time.time(),
    )
    orphan = [
        item
        for item in collector.snapshot()["tool_calls"]
        if item["tool_call_id"] == "orphan-native-tool"
    ][0]
    assert orphan["native_tool_call_id"] == "orphan-native-tool"
    assert orphan["parent_round_id"] is None


def test_raw_function_call_output_closes_internal_codex_tool_span(
    tmp_path: Path,
) -> None:
    collector = _collector(tmp_path, "internal-tool-output")
    bridge = native_worker._CodexLifecycleBridge(collector, max_iterations=5)
    bridge.native_event(
        _native(
            "turn/started",
            {
                "threadId": "thread-internal",
                "turn": {"id": "turn-internal", "status": "inProgress"},
            },
        ),
        time.time(),
    )
    bridge.native_event(
        _native(
            "rawResponseItem/completed",
            {
                "threadId": "thread-internal",
                "turnId": "turn-internal",
                "item": {
                    "type": "function_call",
                    "id": "fc-internal",
                    "call_id": "call-internal",
                    "name": "update_plan",
                    "arguments": '{"plan":[]}',
                },
            },
        ),
        time.time(),
    )
    assert bridge.pending_tool_ids() == ["call-internal"]
    bridge.native_event(
        _native(
            "rawResponseItem/completed",
            {
                "threadId": "thread-internal",
                "turnId": "turn-internal",
                "item": {
                    "type": "function_call_output",
                    "call_id": "call-internal",
                    "output": "Plan updated",
                },
            },
        ),
        time.time(),
    )

    assert bridge.pending_tool_ids() == []
    tool = collector.snapshot()["tool_calls"][0]
    assert tool["tool_name"] == "update_plan"
    assert tool["status"] == "completed"
    assert tool["duration_s"] is not None
    assert tool["wait_s"] == 0.0
    assert (
        tool["metadata"]["timing_boundary"]
        == "native_function_call_to_output_no_start_event"
    )
    assert (
        tool["metadata"]["wait_boundary"]
        == "no_separate_native_queue_assume_start_at_function_call"
    )
    assert tool["metadata"]["completion_source"] == "raw_function_call_output"


@pytest.mark.parametrize(
    ("native_status", "timed_out", "iteration_limited", "round_status", "terminal_status"),
    [
        ("failed", False, False, "failed", "error"),
        ("interrupted", False, False, "interrupted", "interrupted"),
        ("interrupted", True, False, "timeout", "timeout"),
        ("interrupted", False, True, "max_iterations", "max_iterations"),
    ],
)
def test_run_codex_closes_active_round_on_abnormal_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_status: str,
    timed_out: bool,
    iteration_limited: bool,
    round_status: str,
    terminal_status: str,
) -> None:
    collector = _collector(tmp_path, f"terminal-{round_status}")
    bridge = native_worker._CodexLifecycleBridge(collector, max_iterations=5)

    class FakeDriver:
        def __init__(self, **kwargs: Any) -> None:
            self.callback = kwargs["event_callback"]

        def __enter__(self) -> "FakeDriver":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def start_thread(self, **_kwargs: Any) -> dict[str, Any]:
            self.callback(
                _native("thread/started", {"thread": {"id": "thread-terminal"}}),
                time.time(),
            )
            return {"thread": {"id": "thread-terminal"}}

        def run_turn(self, *_args: Any, **_kwargs: Any) -> CodexTurnResult:
            self.callback(
                _native(
                    "turn/started",
                    {
                        "threadId": "thread-terminal",
                        "turn": {"id": "turn-terminal", "status": "inProgress"},
                    },
                ),
                time.time(),
            )
            bridge.backend_event(
                {
                    "event": "backend_attempt_started",
                    "backend_attempt_id": "attempt-terminal",
                    "protocol": "responses",
                }
            )
            terminal_turn = {
                "id": "turn-terminal",
                "status": native_status,
                "items": [],
                "error": ({"message": "native failure"} if native_status == "failed" else None),
            }
            self.callback(
                _native(
                    "turn/completed",
                    {"threadId": "thread-terminal", "turn": terminal_turn},
                ),
                time.time(),
            )
            now = time.time()
            return CodexTurnResult(
                phase="initial",
                thread_id="thread-terminal",
                turn_id="turn-terminal",
                status=native_status,
                turn=terminal_turn,
                events=(),
                started_at_s=now - 0.1,
                completed_at_s=now,
                timed_out=timed_out,
                iteration_limited=iteration_limited,
                interrupt_requested=timed_out or iteration_limited,
            )

    monkeypatch.setattr(native_worker, "CodexAppServerDriver", FakeDriver)
    monkeypatch.setattr(
        native_worker,
        "_agent_sandbox_command",
        lambda command, **_kwargs: list(command),
    )
    root = tmp_path / f"run-{round_status}"
    for path in (root, root / "workspace", root / "runtime", root / "inputs"):
        path.mkdir(parents=True, exist_ok=True)
    native_worker._run_codex(
        agent_bin=Path("/bin/true"),
        workspace_dir=root / "workspace",
        runtime_root=root / "runtime",
        local_inputs_dir=root / "inputs",
        readonly_paths=(),
        require_bwrap=False,
        cpu_affinity=None,
        env={},
        collector=collector,
        bridge=bridge,
        request_run_dir=root,
        model=smoke_test.MODEL,
        prompt="gate",
        job_id="gate",
        max_iterations=5,
        deadline_monotonic_s=None,
        verification_required=False,
        verification_prompt="unused",
    )
    snapshot = collector.snapshot()
    assert snapshot["terminal"]["status"] == terminal_status
    assert len(snapshot["logical_llm_rounds"]) == 1
    logical_round = snapshot["logical_llm_rounds"][0]
    assert logical_round["status"] == round_status
    assert logical_round["end_time_s"] is not None
    assert logical_round["native_response_id"] is None
    assert logical_round["metadata"]["terminal_closure"] is True
    assert bridge.active_round_by_turn == {}


def test_run_codex_rejects_completed_turn_with_unresolved_native_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _collector(tmp_path, "unresolved-tool-terminal")
    bridge = native_worker._CodexLifecycleBridge(collector, max_iterations=5)

    class FakeDriver:
        def __init__(self, **kwargs: Any) -> None:
            self.callback = kwargs["event_callback"]

        def __enter__(self) -> "FakeDriver":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def start_thread(self, **_kwargs: Any) -> dict[str, Any]:
            self.callback(
                _native("thread/started", {"thread": {"id": "thread-pending"}}),
                time.time(),
            )
            return {"thread": {"id": "thread-pending"}}

        def run_turn(self, *_args: Any, **_kwargs: Any) -> CodexTurnResult:
            self.callback(
                _native(
                    "turn/started",
                    {
                        "threadId": "thread-pending",
                        "turn": {"id": "turn-pending", "status": "inProgress"},
                    },
                ),
                time.time(),
            )
            self.callback(
                _native(
                    "rawResponseItem/completed",
                    {
                        "threadId": "thread-pending",
                        "turnId": "turn-pending",
                        "item": {
                            "type": "function_call",
                            "id": "fc-pending",
                            "call_id": "call-pending",
                            "name": "update_plan",
                            "arguments": '{"plan":[]}',
                        },
                    },
                ),
                time.time(),
            )
            terminal_turn = {
                "id": "turn-pending",
                "status": "completed",
                "items": [],
                "error": None,
            }
            self.callback(
                _native(
                    "turn/completed",
                    {"threadId": "thread-pending", "turn": terminal_turn},
                ),
                time.time(),
            )
            now = time.time()
            return CodexTurnResult(
                phase="initial",
                thread_id="thread-pending",
                turn_id="turn-pending",
                status="completed",
                turn=terminal_turn,
                events=(),
                started_at_s=now - 0.1,
                completed_at_s=now,
                timed_out=False,
                iteration_limited=False,
                interrupt_requested=False,
            )

    monkeypatch.setattr(native_worker, "CodexAppServerDriver", FakeDriver)
    monkeypatch.setattr(
        native_worker,
        "_agent_sandbox_command",
        lambda command, **_kwargs: list(command),
    )
    root = tmp_path / "run-unresolved-tool"
    for path in (root, root / "workspace", root / "runtime", root / "inputs"):
        path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(
        native_worker.CodexAppServerError,
        match="unresolved native tool calls: call-pending",
    ):
        native_worker._run_codex(
            agent_bin=Path("/bin/true"),
            workspace_dir=root / "workspace",
            runtime_root=root / "runtime",
            local_inputs_dir=root / "inputs",
            readonly_paths=(),
            require_bwrap=False,
            cpu_affinity=None,
            env={},
            collector=collector,
            bridge=bridge,
            request_run_dir=root,
            model=smoke_test.MODEL,
            prompt="gate",
            job_id="gate",
            max_iterations=5,
            deadline_monotonic_s=None,
            verification_required=False,
            verification_prompt="unused",
        )


def test_sleeping_tool_span_is_visible_before_codex_turn_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native tool start must be flushed while the tool is still running."""

    _require_native_runtime()
    original_select_tool = smoke_test._select_tool

    def select_waiting_tool(
        payload: dict[str, Any], *, shell_delay_s: float = 0.0
    ) -> tuple[str, dict[str, Any]]:
        name, arguments = original_select_tool(
            payload, shell_delay_s=shell_delay_s
        )
        # The stock smoke mock intentionally yields after one second.  Waiting
        # longer than the two-second command gives this lifecycle gate a real
        # item/completed event as well as an observable in-flight interval.
        if name == "exec_command":
            arguments["yield_time_ms"] = 5000
        return name, arguments

    monkeypatch.setattr(smoke_test, "_select_tool", select_waiting_tool)
    task = {
        "schema": "mas_task_v1",
        "request_id": "native-codex-live-tool",
        "benchmark": "smoke",
        "task_id": "native-codex-live-tool-1",
        "payload": {"text": "Inspect the workspace with a shell command, then finish."},
        "workspace_init": [{"path": "README.txt", "content": "native Codex gate\n"}],
    }
    root = tmp_path / "live"
    result_box: dict[str, Any] = {}
    errors: list[BaseException] = []

    with smoke_test.MockOpenAIServer(shell_delay_s=2.0) as server:
        def invoke() -> None:
            try:
                result_box["result"] = _run_codex(
                    root, server, task=task, name="native_codex_live_tool"
                )
            except BaseException as exc:  # re-raised in the test thread
                errors.append(exc)

        worker = threading.Thread(target=invoke)
        worker.start()
        live_path = root / "request" / "live_events.jsonl"
        observed_rows: list[dict[str, Any]] | None = None
        deadline = time.monotonic() + 90.0
        while worker.is_alive() and time.monotonic() < deadline:
            rows = load_jsonl(live_path)
            if any(row.get("event") == "tool_start" for row in rows) and not any(
                row.get("event") == "tool_end" for row in rows
            ):
                assert worker.is_alive()
                observed_rows = rows
                break
            time.sleep(0.01)
        worker.join(timeout=150.0)

    assert not worker.is_alive(), "native Codex request hung"
    if errors:
        raise errors[0]
    assert observed_rows is not None, "tool_start was not observable in flight"
    assert [row["event"] for row in observed_rows] == ["tool_start"]

    result = result_box["result"]
    trace = result["trace"]
    assert trace["status"] == "completed", trace.get("error")
    assert trace["native_terminal_status"] == "completed"
    assert trace["dynamic_graph"]["round_count"] == 2
    assert len(trace["tool_calls"]) == 1
    tool = trace["tool_calls"][0]
    assert tool["status"] == "completed"
    assert float(tool["duration_s"]) >= 1.5
    final_live_rows = load_jsonl(root / "request" / "live_events.jsonl")
    assert [row["event"] for row in final_live_rows] == ["tool_start", "tool_end"]
    assert float(final_live_rows[1]["ts"]) > float(final_live_rows[0]["ts"])


def test_verification_turn_reuses_the_exact_native_codex_thread(tmp_path: Path) -> None:
    _require_native_runtime()
    task = {
        "schema": "mas_task_v1",
        "request_id": "native-codex-same-thread-verification",
        "benchmark": "repobench",
        "task_id": "native-codex-same-thread-verification-1",
        "payload": {
            "text": "Answer from the prompt first; inspect the workspace only if reminded."
        },
        "workspace_init": [
            {"path": "EVIDENCE.txt", "content": "same-thread verification\n"}
        ],
    }
    root = tmp_path / "verification"
    with smoke_test.MockOpenAIServer(verification_nudge=True) as server:
        result = _run_codex(
            root, server, task=task, name="native_codex_same_thread_verification"
        )

    trace = result["trace"]
    assert trace["status"] == "completed", trace.get("error")
    assert trace["verification_nudge_count"] == 1
    assert trace["initial_verification_tool_count"] == 0
    assert trace["verification_tool_count"] == 1
    assert len(trace["tool_calls"]) == 1
    assert trace["tool_calls"][0]["phase"] == "verification"

    wire_rows = load_jsonl(root / "request" / "codex_app_server.wire.jsonl")
    outbound = [
        row["message"]
        for row in wire_rows
        if row.get("direction") == "to_server" and isinstance(row.get("message"), dict)
    ]
    inbound = [
        row["message"]
        for row in wire_rows
        if row.get("direction") == "from_server" and isinstance(row.get("message"), dict)
    ]
    thread_starts = [row for row in outbound if row.get("method") == "thread/start"]
    turn_starts = [row for row in outbound if row.get("method") == "turn/start"]
    thread_started = [row for row in inbound if row.get("method") == "thread/started"]
    turn_completed = [row for row in inbound if row.get("method") == "turn/completed"]

    assert len(thread_starts) == 1
    assert len(thread_started) == 1
    assert len(turn_starts) == 2
    assert len(turn_completed) == 2
    native_thread_id = thread_started[0]["params"]["thread"]["id"]
    assert native_thread_id == trace["native_session_id"]
    assert {row["params"]["threadId"] for row in turn_starts} == {native_thread_id}
    assert {row["params"]["threadId"] for row in turn_completed} == {
        native_thread_id
    }
    assert all(row["params"]["turn"]["status"] == "completed" for row in turn_completed)
