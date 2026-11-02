from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import smoke_test
from event_normalizer import load_jsonl
from native_request_worker import run_request
from request_worker import _bwrap_binary, _working_bwrap
from run_workload import PINNED


OPENCODE_BINARY = Path(PINNED["opencode"]["binary"])


def _require_native_runtime() -> None:
    if not OPENCODE_BINARY.exists():
        pytest.skip(f"pinned OpenCode binary is missing: {OPENCODE_BINARY}")
    if not _working_bwrap():
        pytest.skip("bubblewrap is required for the native OpenCode integration gate")


def _run_opencode(
    root: Path,
    *,
    task: dict[str, Any],
    name: str,
    base_url: str,
) -> dict[str, Any]:
    return run_request(
        agent="opencode",
        agent_bin=OPENCODE_BINARY,
        task=task,
        task_sha256=f"{name}-sha256",
        request_id=name,
        mars_job_id=name,
        arrival_time_s=time.time(),
        workspace_root=root / "workspaces",
        local_inputs_dir=root / "localized_inputs",
        request_run_dir=root / "request",
        model=smoke_test.MODEL,
        base_url=base_url,
        api_key="EMPTY",
        context_window=32768,
        max_output_tokens=1024,
        max_iterations=5,
        temperature=0.0,
        timeout_s=120.0,
        require_bwrap=True,
    )


def _native_statuses(rows: list[dict[str, Any]], session_id: str) -> list[str]:
    statuses: list[str] = []
    for row in rows:
        if (
            row.get("event_type") != "session.status"
            or row.get("native_session_id") != session_id
        ):
            continue
        raw = row.get("raw")
        props = raw.get("properties") if isinstance(raw, dict) else None
        status = props.get("status") if isinstance(props, dict) else None
        kind = status.get("type") if isinstance(status, dict) else None
        if isinstance(kind, str):
            statuses.append(kind)
    return statuses


def test_real_opencode_worker_same_session_verification_and_live_tool(
    tmp_path: Path,
) -> None:
    """Exercise the full worker while observing a native tool in flight."""

    _require_native_runtime()
    task = {
        "schema": "mas_task_v1",
        "request_id": "native-opencode-same-session-verification",
        "benchmark": "repobench",
        "task_id": "native-opencode-same-session-verification-1",
        "payload": {
            "text": "Answer from the prompt first; inspect the workspace only if reminded."
        },
        "workspace_init": [
            {"path": "EVIDENCE.txt", "content": "same-session OpenCode gate\n"}
        ],
    }
    root = tmp_path / "verification"
    result_box: dict[str, Any] = {}
    errors: list[BaseException] = []

    # The first prompt deliberately receives a final answer without tools.
    # The worker must append a verification prompt to the same root session;
    # that prompt receives a two-second native bash call followed by a final
    # model round.
    with smoke_test.MockOpenAIServer(
        verification_nudge=True, shell_delay_s=2.0
    ) as server:

        def invoke() -> None:
            try:
                result_box["result"] = _run_opencode(
                    root,
                    task=task,
                    name="native_opencode_same_session_verification",
                    base_url=server.base_url,
                )
            except BaseException as exc:  # re-raised in the test thread
                errors.append(exc)

        worker = threading.Thread(target=invoke, name="native-opencode-worker-gate")
        worker.start()
        live_path = root / "request" / "live_events.jsonl"
        observed_inflight: list[dict[str, Any]] | None = None
        deadline = time.monotonic() + 90.0
        while worker.is_alive() and time.monotonic() < deadline:
            rows = load_jsonl(live_path)
            if any(row.get("event") == "tool_start" for row in rows) and not any(
                row.get("event") == "tool_end" for row in rows
            ):
                assert worker.is_alive()
                observed_inflight = rows
                break
            time.sleep(0.01)
        worker.join(timeout=150.0)

    assert not worker.is_alive(), "native OpenCode request hung"
    if errors:
        raise errors[0]
    assert observed_inflight is not None, "native tool_start was not visible in flight"
    assert [row["event"] for row in observed_inflight] == ["tool_start"]

    result = result_box["result"]
    trace = result["trace"]
    request_dir = root / "request"
    root_session_id = trace["native_session_id"]
    assert isinstance(root_session_id, str) and root_session_id.startswith("ses_")
    assert trace["status"] == "completed", trace.get("error")
    assert trace["native_terminal_status"] == "completed"
    assert trace["frontend_protocol"] == "opencode_legacy_server_sse"
    assert trace["backend_protocol"] == "chat_completions"
    assert trace["verification_nudge_count"] == 1
    assert trace["initial_verification_tool_count"] == 0
    assert trace["verification_tool_count"] == 1
    assert trace["verification_nudges"][0]["native_session_id"] == root_session_id

    # Initial answer = one logical step.  Verification = tool-calling step plus
    # final-answer step.  All three are native step-start/step-finish rounds.
    assert len(trace["llm_calls"]) == 3
    assert len(trace["backend_attempts"]) == 3
    assert [call["phase"] for call in trace["llm_calls"]] == [
        "initial",
        "verification",
        "verification",
    ]
    assert all(
        call["round_id"].startswith(root_session_id + ":")
        and call["api_kind"] == "chat_completions"
        and call["status"] == "completed"
        for call in trace["llm_calls"]
    )
    assert all(
        attempt["round_id"] in {call["round_id"] for call in trace["llm_calls"]}
        and attempt["protocol"] == "chat_completions"
        and attempt["status"] == "completed"
        for attempt in trace["backend_attempts"]
    )
    assert [len(round_row["backend_attempts"]) for round_row in trace["rounds"]] == [
        1,
        1,
        1,
    ]

    assert len(trace["tool_calls"]) == 1
    tool = trace["tool_calls"][0]
    assert tool["tool_name"] == "bash"
    assert tool["phase"] == "verification"
    assert tool["status"] == "completed"
    assert tool["parent_round_id"] == trace["llm_calls"][1]["round_id"]
    assert float(tool["wait_s"]) >= 0.0
    assert float(tool["duration_s"]) >= 1.5
    final_live_rows = load_jsonl(live_path)
    assert [row["event"] for row in final_live_rows] == [
        "tool_start",
        "tool_end",
    ]
    assert final_live_rows[0]["phase"] == final_live_rows[1]["phase"] == "verification"
    assert float(final_live_rows[1]["ts"]) > float(final_live_rows[0]["ts"])

    native_rows = load_jsonl(request_dir / "native_events.jsonl")
    event_types = [str(row.get("event_type") or "") for row in native_rows]
    assert event_types.index("server.connected") < event_types.index("session.created")
    created = [
        row["raw"]["properties"]["info"]
        for row in native_rows
        if row.get("event_type") == "session.created"
    ]
    assert len(created) == 1 and created[0]["id"] == root_session_id
    statuses = _native_statuses(native_rows, root_session_id)
    assert statuses.count("idle") == 2
    assert statuses.count("busy") >= 2
    assert (
        len(
            [
                row
                for row in native_rows
                if row.get("event_type") == "session.idle"
                and row.get("native_session_id") == root_session_id
            ]
        )
        == 2
    )
    assert {item["session_id"] for item in trace["agent_messages"]} == {root_session_id}
    assert trace["final_output"] == "mock task complete"
    assert trace["dynamic_graph"]["round_count"] == 3
    assert trace["dynamic_graph"]["tool_call_count"] == 1

    persisted = json.loads((request_dir / "result.json").read_text(encoding="utf-8"))
    assert persisted["schema"] == "mars_native_agent_request_result_v3"
    assert persisted["trace"]["status"] == "completed"
    assert persisted["trace"]["native_session_id"] == root_session_id
    command = json.loads((request_dir / "command.json").read_text(encoding="utf-8"))
    assert command[0] == str(_bwrap_binary())
    assert "--pure" in command and "serve" in command
    assert (request_dir / "backend_attempts.jsonl").is_file()
    assert (request_dir / "opencode_server.stdout.txt").is_file()
    assert (request_dir / "opencode_server.stderr.txt").is_file()

    # Pin the provider seam expected by the real Qwen/vLLM backend.  OpenCode
    # must own the tool loop and replay canonical OpenAI Chat messages; the
    # request-local MARS proxy may only add job_id and trace the transport.
    payloads = list(server.state.requests)
    assert len(payloads) == 3
    assert all(payload["model"] == smoke_test.MODEL for payload in payloads)
    assert all(payload["stream"] is True for payload in payloads)
    assert all(
        payload["stream_options"] == {"include_usage": True}
        for payload in payloads
    )
    assert all(payload["max_tokens"] == 1024 for payload in payloads)
    assert all(payload["temperature"] == 0 for payload in payloads)
    assert all(payload["tool_choice"] == "auto" for payload in payloads)
    assert all(
        payload["job_id"] == "native_opencode_same_session_verification"
        for payload in payloads
    )
    assert "top_p" not in payloads[0]
    assert "parallel_tool_calls" not in payloads[0]

    tool_specs = payloads[0]["tools"]
    assert isinstance(tool_specs, list) and tool_specs
    assert all(
        isinstance(spec, dict)
        and spec.get("type") == "function"
        and isinstance(spec.get("function"), dict)
        and isinstance(spec["function"].get("name"), str)
        and isinstance(spec["function"].get("parameters"), dict)
        for spec in tool_specs
    )
    assert "bash" in {spec["function"]["name"] for spec in tool_specs}

    replay = payloads[-1]["messages"]
    assistant_tool_calls = [
        tool_call
        for message in replay
        if message.get("role") == "assistant"
        for tool_call in message.get("tool_calls", [])
    ]
    assert len(assistant_tool_calls) == 1
    replayed_call = assistant_tool_calls[0]
    assert replayed_call["id"] == "call_mars_smoke"
    assert replayed_call["type"] == "function"
    assert replayed_call["function"]["name"] == "bash"
    assert isinstance(json.loads(replayed_call["function"]["arguments"]), dict)
    tool_results = [message for message in replay if message.get("role") == "tool"]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_call_id"] == replayed_call["id"]

    opencode_config = json.loads(
        (request_dir / "runtime" / "config" / "opencode.json").read_text(
            encoding="utf-8"
        )
    )
    provider = opencode_config["provider"]["mars"]
    assert opencode_config["model"] == f"mars/{smoke_test.MODEL}"
    assert opencode_config["small_model"] == f"mars/{smoke_test.MODEL}"
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["apiKey"] == "{env:MARS_AGENT_API_KEY}"
    assert provider["options"]["baseURL"].startswith("http://127.0.0.1:")
    assert provider["options"]["baseURL"].endswith("/v1")
    assert set(provider["models"]) == {smoke_test.MODEL}


class _RetryOnceChatServer:
    """Return one retryable 429, then a valid streaming final answer."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _send(
                self,
                status: int,
                content_type: str,
                body: bytes,
                *,
                retry_after: str | None = None,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                if retry_after is not None:
                    self.send_header("Retry-After", retry_after)
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

            def do_GET(self) -> None:  # noqa: N802
                if self.path.split("?", 1)[0].endswith("/models"):
                    body = json.dumps(
                        {
                            "object": "list",
                            "data": [{"id": smoke_test.MODEL, "object": "model"}],
                        }
                    ).encode("utf-8")
                    self._send(200, "application/json", body)
                    return
                self._send(404, "application/json", b'{"error":"not found"}')

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                with owner._lock:
                    owner.requests.append(payload)
                    request_index = len(owner.requests)
                if not self.path.split("?", 1)[0].endswith("/chat/completions"):
                    self._send(404, "application/json", b'{"error":"not found"}')
                    return
                if request_index == 1:
                    self._send(
                        429,
                        "application/json",
                        b'{"error":{"message":"retry once","type":"rate_limit_error","code":"rate_limit"}}',
                        retry_after="0",
                    )
                    return
                base = {
                    "id": "chatcmpl-opencode-retry-success",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": smoke_test.MODEL,
                }
                chunks = [
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": "retry complete",
                                },
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        **base,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "total_tokens": 12,
                        },
                    },
                ]
                body = (
                    b"".join(
                        b"data: "
                        + json.dumps(chunk, separators=(",", ":")).encode("utf-8")
                        + b"\n\n"
                        for chunk in chunks
                    )
                    + b"data: [DONE]\n\n"
                )
                self._send(200, "text/event-stream", body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self) -> "_RetryOnceChatServer":
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def test_real_opencode_worker_keeps_backend_retry_inside_one_logical_step(
    tmp_path: Path,
) -> None:
    """Physical retry attempts must not inflate the dynamic agent DAG."""

    _require_native_runtime()
    task = {
        "schema": "mas_task_v1",
        "request_id": "native-opencode-retry-separation",
        "benchmark": "smoke",
        "task_id": "native-opencode-retry-separation-1",
        "payload": {"text": "Answer briefly."},
        "workspace_init": [],
    }
    root = tmp_path / "retry"
    with _RetryOnceChatServer() as server:
        result = _run_opencode(
            root,
            task=task,
            name="native_opencode_retry_separation",
            base_url=server.base_url,
        )

    trace = result["trace"]
    assert trace["status"] == "completed", trace.get("error")
    assert trace["native_terminal_status"] == "completed"
    assert trace["final_output"] == "retry complete"
    assert len(server.requests) == 2
    assert all(
        payload["job_id"] == "native_opencode_retry_separation"
        for payload in server.requests
    )

    # One native step-start/step-finish pair owns both transport attempts.
    assert len(trace["llm_calls"]) == 1
    assert len(trace["backend_attempts"]) == 2
    logical_round = trace["llm_calls"][0]
    attempts = trace["backend_attempts"]
    assert logical_round["status"] == "completed"
    assert logical_round["native_response_id"] == "chatcmpl-opencode-retry-success"
    assert {attempt["round_id"] for attempt in attempts} == {logical_round["round_id"]}
    assert [attempt["http_status"] for attempt in attempts] == [429, 200]
    assert [attempt["status"] for attempt in attempts] == ["error", "completed"]
    assert len(trace["rounds"]) == 1
    assert [
        attempt["backend_attempt_id"]
        for attempt in trace["rounds"][0]["backend_attempts"]
    ] == ["backend_0001", "backend_0002"]
    assert trace["dynamic_graph"]["round_count"] == 1
    assert trace["dynamic_graph"]["tool_call_count"] == 0

    request_dir = root / "request"
    native_rows = load_jsonl(request_dir / "native_events.jsonl")
    root_session_id = trace["native_session_id"]
    statuses = _native_statuses(native_rows, root_session_id)
    assert "retry" in statuses
    assert statuses[-1] == "idle"
    assert any(
        row.get("event_type") == "session.idle"
        and row.get("native_session_id") == root_session_id
        for row in native_rows
    )
    persisted = json.loads((request_dir / "result.json").read_text(encoding="utf-8"))
    assert persisted["trace"]["dynamic_graph"]["round_count"] == 1
    assert len(persisted["trace"]["backend_attempts"]) == 2
