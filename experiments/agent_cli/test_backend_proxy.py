from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from openai.types.responses.tool import Tool
from pydantic import TypeAdapter

from backend_proxy import (
    BackendTraceProxy,
    _flatten_responses_namespace_tools,
    _restore_responses_namespace_calls,
)


def _namespace_request() -> dict[str, Any]:
    return {
        "model": "qwen",
        "input": [
            {"type": "message", "role": "user", "content": "delegate"},
            {
                "type": "function_call",
                "call_id": "call_previous",
                "namespace": "multi_agent_v1",
                "name": "spawn_agent",
                "arguments": '{"message":"inspect"}',
            },
        ],
        "stream": True,
        "tool_choice": {
            "type": "function",
            "namespace": "multi_agent_v1",
            "name": "spawn_agent",
        },
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Run a command.",
                "strict": False,
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "Tools for spawning and managing sub-agents.",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "description": "Spawn a sub-agent.",
                        "strict": False,
                        "defer_loading": False,
                        "parameters": {
                            "type": "object",
                            "properties": {"message": {"type": "string"}},
                            "required": ["message"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "type": "function",
                        "name": "wait_agent",
                        "description": "Wait for a sub-agent.",
                        "strict": False,
                        "parameters": {
                            "type": "object",
                            "properties": {"targets": {"type": "array"}},
                        },
                    },
                ],
            },
        ],
    }


def test_namespace_compat_flattens_history_and_restores_calls() -> None:
    payload = _namespace_request()
    reverse_names, namespace_count = _flatten_responses_namespace_tools(payload)

    assert namespace_count == 1
    assert reverse_names == {
        "multi_agent_v1__spawn_agent": ("multi_agent_v1", "spawn_agent"),
        "multi_agent_v1__wait_agent": ("multi_agent_v1", "wait_agent"),
    }
    assert [tool["name"] for tool in payload["tools"]] == [
        "exec_command",
        "multi_agent_v1__spawn_agent",
        "multi_agent_v1__wait_agent",
    ]
    assert all(tool["type"] == "function" for tool in payload["tools"])
    assert "defer_loading" not in payload["tools"][1]
    assert payload["tools"][1]["description"].startswith(
        "Tools for spawning and managing sub-agents."
    )
    assert payload["input"][1]["name"] == "multi_agent_v1__spawn_agent"
    assert "namespace" not in payload["input"][1]
    assert payload["tool_choice"]["name"] == "multi_agent_v1__spawn_agent"
    assert "namespace" not in payload["tool_choice"]
    # vLLM's ResponsesRequest uses this exact SDK Tool union.  The original
    # namespace object fails that union; the compatibility payload must pass.
    assert len(TypeAdapter(list[Tool]).validate_python(payload["tools"])) == 3

    response = {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "call_id": "call_new",
            "name": "multi_agent_v1__spawn_agent",
            "arguments": '{"message":"inspect"}',
        },
    }
    assert _restore_responses_namespace_calls(response, reverse_names) == 1
    assert response["item"]["name"] == "spawn_agent"
    assert response["item"]["namespace"] == "multi_agent_v1"


def test_proxy_is_protocol_transparent_and_records_physical_attempts(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    wire = (
        b'event: response.output_text.delta\n'
        b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
        b'event: response.completed\n'
        b'data: {"type":"response.completed","response":{"id":"resp_real"}}\n\n'
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            captured.append(json.loads(self.rfile.read(length)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    trace_path = tmp_path / "backend_attempts.jsonl"
    try:
        upstream = f"http://127.0.0.1:{server.server_address[1]}/v1"
        with BackendTraceProxy(
            upstream_base_url=upstream,
            trace_path=trace_path,
            request_id="req_1",
            mars_job_id="req_1",
            upstream_timeout_s=5,
            max_output_tokens=64,
            temperature=0.0,
            top_p=None,
        ) as proxy:
            body = json.dumps(
                {
                    "model": "qwen",
                    "input": "hello",
                    "stream": True,
                    "metadata": {"native": "unchanged"},
                    "top_p": 0.9,
                }
            ).encode()
            request = urllib.request.Request(
                proxy.base_url + "/responses",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.read() == wire
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert captured == [
        {
            "model": "qwen",
            "input": "hello",
            "stream": True,
            "metadata": {"native": "unchanged"},
            "job_id": "req_1",
            "max_output_tokens": 64,
            "temperature": 0.0,
        }
    ]
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "backend_attempt_started",
        "backend_attempt_first_token",
        "backend_attempt_ended",
    ]
    assert rows[0]["protocol"] == "responses"
    assert rows[1]["ttft_s"] is not None
    assert rows[2]["http_status"] == 200
    assert rows[2]["response_id"] == "resp_real"
    assert rows[2]["ttft_s"] is not None


def test_proxy_downgrades_namespace_tools_and_restores_stream(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    alias = "multi_agent_v1__spawn_agent"
    events = [
        {
            "type": "response.created",
            "response": {"id": "resp_namespace", "status": "in_progress"},
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_new",
                "name": alias,
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "item_id": "item_new",
            "name": alias,
            "arguments": '{"message":"inspect"}',
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_new",
                "name": alias,
                "arguments": '{"message":"inspect"}',
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_namespace",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_new",
                        "name": alias,
                        "arguments": '{"message":"inspect"}',
                    }
                ],
            },
        },
    ]
    wire = b"".join(
        b"event: "
        + event["type"].encode()
        + b"\n"
        + b"data: "
        + json.dumps(event, separators=(",", ":")).encode()
        + b"\n\n"
        for event in events
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            captured.append(json.loads(self.rfile.read(length)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    trace_path = tmp_path / "backend_attempts.jsonl"
    try:
        upstream = f"http://127.0.0.1:{server.server_address[1]}/v1"
        with BackendTraceProxy(
            upstream_base_url=upstream,
            trace_path=trace_path,
            request_id="req_namespace",
            mars_job_id="req_namespace",
            upstream_timeout_s=5,
            max_output_tokens=64,
            temperature=0.0,
            top_p=None,
            responses_namespace_tool_compat=True,
        ) as proxy:
            request = urllib.request.Request(
                proxy.base_url + "/responses",
                data=json.dumps(_namespace_request()).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                returned_wire = response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert len(captured) == 1
    forwarded = captured[0]
    assert forwarded["job_id"] == "req_namespace"
    assert [tool["name"] for tool in forwarded["tools"]] == [
        "exec_command",
        "multi_agent_v1__spawn_agent",
        "multi_agent_v1__wait_agent",
    ]
    assert all(tool["type"] == "function" for tool in forwarded["tools"])
    assert forwarded["input"][1]["name"] == alias
    assert "namespace" not in forwarded["input"][1]

    returned = []
    for line in returned_wire.splitlines():
        if line.startswith(b"data:"):
            returned.append(json.loads(line.split(b":", 1)[1]))
    namespaced_items: list[dict[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            if value.get("name") == "spawn_agent":
                namespaced_items.append(value)
            for child in value.values():
                collect(child)

    collect(returned)
    assert len(namespaced_items) == 4
    assert all(item.get("namespace") == "multi_agent_v1" for item in namespaced_items)

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert rows[0]["responses_namespace_tool_compat"] is True
    assert rows[0]["flattened_namespace_count"] == 1
    assert rows[0]["flattened_namespace_function_count"] == 2
    assert "responses_namespace_tools_flattened" in rows[0]["semantic_mutations"]
    assert rows[-1]["http_status"] == 200
    assert rows[-1]["response_id"] == "resp_namespace"
    assert rows[-1]["restored_namespace_call_count"] == 4
