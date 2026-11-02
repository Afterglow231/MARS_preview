#!/usr/bin/env python3
"""Run the pinned OpenCode and Codex CLIs against a two-round mock server.

This is an integration smoke test, not a model-quality test.  The mock emits a
real tool call on round one and a final answer on round two, so it exercises the
CLI, provider configuration, local tool execution, streaming trace proxy,
MARS ``job_id`` injection, event normalization, and dynamic-DAG construction.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from event_normalizer import load_jsonl
from request_worker import _native_session_id, run_request
from run_workload import PINNED
from trace_proxy import _responses_to_chat


MODEL = "mars-mock-model"


def _function_tools(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if isinstance(name, str):
            found.append((name, function))
    return found


def _select_tool(
    payload: dict[str, Any], *, shell_delay_s: float = 0.0
) -> tuple[str, dict[str, Any]]:
    tools = _function_tools(payload)
    by_name = {name: spec for name, spec in tools}
    shell_command = (
        f"sleep {float(shell_delay_s):.2f}; pwd"
        if shell_delay_s > 0
        else "pwd"
    )
    for name in ("exec_command", "shell_command", "bash"):
        if name in by_name:
            if name == "exec_command":
                return name, {
                    "cmd": shell_command,
                    "yield_time_ms": 1000,
                    "max_output_tokens": 1000,
                }
            if name == "shell_command":
                return name, {"command": shell_command}
            return name, {
                "command": shell_command,
                "description": "verify the mock workspace",
            }
    raise AssertionError(f"no supported shell tool in request; tools={[name for name, _ in tools]}")


def _has_tool_output(payload: dict[str, Any]) -> bool:
    messages = payload.get("messages")
    if isinstance(messages, list) and any(
        isinstance(item, dict) and item.get("role") == "tool" for item in messages
    ):
        return True
    inputs = payload.get("input")
    if isinstance(inputs, list) and any(
        isinstance(item, dict)
        and item.get("type") in {"function_call_output", "custom_tool_call_output"}
        for item in inputs
    ):
        return True
    return False


def _has_verification_nudge(payload: dict[str, Any]) -> bool:
    marker = "You have not yet performed the required tool-based verification step."

    def contains(value: Any) -> bool:
        if isinstance(value, str):
            return marker in value
        if isinstance(value, list):
            return any(contains(item) for item in value)
        if isinstance(value, dict):
            return any(contains(item) for item in value.values())
        return False

    return contains(payload.get("messages")) or contains(payload.get("input"))


def _sse(events: list[dict[str, Any]], *, include_done: bool = False) -> bytes:
    chunks: list[str] = []
    for event in events:
        kind = str(event.get("type") or "message")
        chunks.append(f"event: {kind}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n")
    if include_done:
        chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode("utf-8")


class _State:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def append(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.requests.append(payload)


class MockOpenAIServer:
    def __init__(
        self,
        *,
        malformed_tool_arguments: bool = False,
        verification_nudge: bool = False,
        shell_delay_s: float = 0.0,
    ) -> None:
        self.state = _State()
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _send(self, status: int, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

            def do_GET(self) -> None:  # noqa: N802
                if self.path.split("?", 1)[0].endswith("/models"):
                    body = json.dumps(
                        {"object": "list", "data": [{"id": MODEL, "object": "model"}]}
                    ).encode("utf-8")
                    self._send(200, "application/json", body)
                    return
                self._send(404, "application/json", b'{"error":"not found"}')

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                state.append(payload)
                path = self.path.split("?", 1)[0]
                try:
                    if path.endswith("/chat/completions"):
                        body = self._chat(payload)
                    elif path.endswith("/responses"):
                        body = self._responses(payload)
                    else:
                        self._send(404, "application/json", b'{"error":"not found"}')
                        return
                except Exception as exc:
                    body = json.dumps({"error": {"message": str(exc)}}).encode("utf-8")
                    self._send(400, "application/json", body)
                    return
                self._send(200, "text/event-stream", body)

            def _chat(self, payload: dict[str, Any]) -> bytes:
                if _has_tool_output(payload):
                    for message in payload.get("messages") or []:
                        if not isinstance(message, dict) or message.get("role") != "assistant":
                            continue
                        for tool_call in message.get("tool_calls") or []:
                            function = tool_call.get("function") if isinstance(tool_call, dict) else None
                            arguments = function.get("arguments") if isinstance(function, dict) else None
                            parsed = json.loads(arguments)
                            if not isinstance(parsed, dict):
                                raise AssertionError("replayed tool arguments must be a JSON object")
                base = {
                    "id": "chatcmpl-mars-smoke",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL,
                }
                should_call_tool = (
                    not _has_tool_output(payload)
                    and (not verification_nudge or _has_verification_nudge(payload))
                )
                if should_call_tool:
                    name, arguments = _select_tool(
                        payload, shell_delay_s=shell_delay_s
                    )
                    encoded_arguments = json.dumps(arguments, separators=(",", ":"))
                    if malformed_tool_arguments:
                        encoded_arguments = encoded_arguments[:-1]
                    chunks = [
                        {
                            **base,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call_mars_smoke",
                                                "type": "function",
                                                "function": {
                                                    "name": name,
                                                    "arguments": encoded_arguments,
                                                },
                                            }
                                        ],
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        },
                        {
                            **base,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                        },
                    ]
                else:
                    chunks = [
                        {
                            **base,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": "mock task complete"},
                                    "finish_reason": None,
                                }
                            ],
                        },
                        {
                            **base,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 120, "completion_tokens": 5, "total_tokens": 125},
                        },
                    ]
                return b"".join(
                    b"data: " + json.dumps(chunk, separators=(",", ":")).encode("utf-8") + b"\n\n"
                    for chunk in chunks
                ) + b"data: [DONE]\n\n"

            def _responses(self, payload: dict[str, Any]) -> bytes:
                should_call_tool = (
                    not _has_tool_output(payload)
                    and (not verification_nudge or _has_verification_nudge(payload))
                )
                if should_call_tool:
                    name, arguments = _select_tool(
                        payload, shell_delay_s=shell_delay_s
                    )
                    args = json.dumps(arguments, separators=(",", ":"))
                    events = [
                        {"type": "response.created", "response": {"id": "resp-mars-smoke-1"}},
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "type": "function_call",
                                "id": "item-mars-smoke",
                                "call_id": "call-mars-smoke",
                                "name": name,
                                "arguments": "",
                                "status": "in_progress",
                            },
                        },
                        {
                            "type": "response.function_call_arguments.delta",
                            "output_index": 0,
                            "item_id": "item-mars-smoke",
                            "delta": args,
                        },
                        {
                            "type": "response.function_call_arguments.done",
                            "output_index": 0,
                            "item_id": "item-mars-smoke",
                            "arguments": args,
                        },
                        {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": {
                                "type": "function_call",
                                "id": "item-mars-smoke",
                                "call_id": "call-mars-smoke",
                                "name": name,
                                "arguments": args,
                                "status": "completed",
                            },
                        },
                        {
                            "type": "response.completed",
                            "response": {
                                "id": "resp-mars-smoke-1",
                                "usage": {
                                    "input_tokens": 100,
                                    "input_tokens_details": {"cached_tokens": 0},
                                    "output_tokens": 10,
                                    "output_tokens_details": {"reasoning_tokens": 0},
                                    "total_tokens": 110,
                                },
                            },
                        },
                    ]
                else:
                    events = [
                        {"type": "response.created", "response": {"id": "resp-mars-smoke-2"}},
                        {
                            "type": "response.output_text.delta",
                            "output_index": 0,
                            "content_index": 0,
                            "item_id": "msg-mars-smoke",
                            "delta": "mock task complete",
                        },
                        {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": {
                                "type": "message",
                                "role": "assistant",
                                "id": "msg-mars-smoke",
                                "content": [{"type": "output_text", "text": "mock task complete"}],
                            },
                        },
                        {
                            "type": "response.completed",
                            "response": {
                                "id": "resp-mars-smoke-2",
                                "usage": {
                                    "input_tokens": 120,
                                    "input_tokens_details": {"cached_tokens": 0},
                                    "output_tokens": 5,
                                    "output_tokens_details": {"reasoning_tokens": 0},
                                    "total_tokens": 125,
                                },
                            },
                        },
                    ]
                return _sse(events)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self) -> "MockOpenAIServer":
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", action="append", choices=sorted(PINNED))
    parser.add_argument("--keep-output", default=None)
    parser.add_argument(
        "--full-runner",
        action="store_true",
        help="Also exercise run_workload.py, preflight, summary, and progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    agents = args.agent or sorted(PINNED)
    cpu_probe = str(min(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else None
    replay_chat, _, replay_sanitized = _responses_to_chat(
        {
            "model": MODEL,
            "input": [
                {
                    "type": "function_call",
                    "call_id": "malformed-history-call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"',
                }
            ],
        }
    )
    replay_arguments = replay_chat["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert replay_sanitized == 1 and isinstance(json.loads(replay_arguments), dict)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_output:
        root = Path(args.keep_output).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="mars_agent_cli_smoke_")
        root = Path(temporary.name)
    task = {
        "schema": "mas_task_v1",
        "request_id": "smoke-dynamic-tool-round",
        "benchmark": "smoke",
        "task_id": "smoke-1",
        "payload": {"text": "Inspect the workspace with a shell command, then finish."},
        "workspace_init": [{"path": "README.txt", "content": "MARS agent CLI smoke test\n"}],
    }
    results: dict[str, Any] = {}
    try:
        with MockOpenAIServer(shell_delay_s=0.40) as server:
            for agent in agents:
                agent_root = root / agent
                invocation_started = time.time()
                request_kwargs = dict(
                    agent=agent,
                    agent_bin=Path(PINNED[agent]["binary"]),
                    task=task,
                    task_sha256="smoke-task-sha256",
                    request_id=f"smoke_{agent}",
                    mars_job_id=f"smoke_{agent}",
                    arrival_time_s=time.time(),
                    workspace_root=agent_root / "workspaces",
                    local_inputs_dir=agent_root / "localized_inputs",
                    request_run_dir=agent_root / "request",
                    model=MODEL,
                    base_url=server.base_url,
                    api_key="EMPTY",
                    context_window=32768,
                    max_output_tokens=1024,
                    max_iterations=5,
                    temperature=0.0,
                    timeout_s=120.0,
                    require_bwrap=True,
                    cpu_affinity=cpu_probe,
                )
                saw_live_inflight = False
                if agent == "opencode":
                    result_box: dict[str, Any] = {}
                    errors: list[BaseException] = []

                    def run_opencode_request() -> None:
                        try:
                            result_box["result"] = run_request(**request_kwargs)
                        except BaseException as exc:  # surfaced in the main thread below
                            errors.append(exc)

                    request_thread = threading.Thread(target=run_opencode_request)
                    request_thread.start()
                    live_path = agent_root / "request" / "live_events.jsonl"
                    live_deadline = time.monotonic() + 90.0
                    while request_thread.is_alive() and time.monotonic() < live_deadline:
                        live_rows_during = load_jsonl(live_path)
                        configured_shell_rows = [
                            row
                            for row in live_rows_during
                            if row.get("timing_source")
                            == "opencode_configured_shell_wrapper"
                            and float(row.get("ts") or 0.0) >= invocation_started
                        ]
                        if any(
                            row.get("event") == "tool_start"
                            for row in configured_shell_rows
                        ):
                            assert not any(
                                row.get("event") == "tool_end"
                                for row in configured_shell_rows
                            ), configured_shell_rows
                            assert request_thread.is_alive()
                            saw_live_inflight = True
                            break
                        time.sleep(0.01)
                    request_thread.join(timeout=150)
                    assert not request_thread.is_alive(), "OpenCode smoke request hung"
                    if errors:
                        raise errors[0]
                    assert saw_live_inflight, (
                        "configured OpenCode bash start was not observable while "
                        "the 0.40s tool was still running"
                    )
                    result = result_box["result"]
                else:
                    result = run_request(**request_kwargs)
                trace = result["trace"]
                graph = trace["dynamic_graph"]
                assert trace["status"] == "completed", trace.get("error")
                assert graph["round_count"] == 2, graph
                assert graph["tool_call_count"] >= 1, graph
                assert all(
                    call.get("mars_metadata", {}).get("job_id") == f"smoke_{agent}"
                    for call in trace["llm_calls"]
                ), trace["llm_calls"]
                if agent == "codex":
                    assert all(
                        call.get("protocol") == "responses"
                        and call.get("upstream_protocol") == "chat_completions"
                        and call.get("responses_via_chat") is True
                        for call in trace["llm_calls"]
                    ), trace["llm_calls"]
                    command_tools = [
                        tool
                        for tool in trace["tool_calls"]
                        if tool.get("tool_name") == "command_execution"
                    ]
                    assert command_tools and all(
                        tool.get("timing_source") == "codex_bash_wrapper"
                        and float(tool.get("duration_s") or 0.0) > 0.0
                        for tool in command_tools
                    ), command_tools
                    shell_rows = [
                        json.loads(line)
                        for line in (agent_root / "request" / "codex_shell_events.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                        if line.strip()
                    ]
                    live_shell_rows = [
                        json.loads(line)
                        for line in (agent_root / "request" / "live_events.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                        if line.strip() and "codex_bash_wrapper" in line
                    ]
                    assert [row.get("event") for row in shell_rows] == [
                        "shell_start",
                        "shell_end",
                    ], shell_rows
                    assert [row.get("event") for row in live_shell_rows] == [
                        "tool_start",
                        "tool_end",
                    ], live_shell_rows
                elif agent == "opencode":
                    bash_tools = [
                        tool
                        for tool in trace["tool_calls"]
                        if tool.get("tool_name") == "bash"
                    ]
                    assert len(bash_tools) == 1, bash_tools
                    assert (
                        bash_tools[0].get("timing_source")
                        == "opencode_configured_shell_wrapper"
                        and bash_tools[0].get("timing_integrity") == "matched"
                        and float(bash_tools[0].get("duration_s") or 0.0) >= 0.30
                    ), bash_tools
                    assert trace["tool_timing_integrity"]["status"] == "passed", trace[
                        "tool_timing_integrity"
                    ]
                    request_dir = agent_root / "request"
                    shell_rows = load_jsonl(request_dir / "opencode_shell_events.jsonl")
                    live_bash_rows = [
                        row
                        for row in load_jsonl(request_dir / "live_events.jsonl")
                        if row.get("tool_name") == "bash"
                    ]
                    assert [row.get("event") for row in shell_rows] == [
                        "shell_start",
                        "shell_end",
                    ], shell_rows
                    assert [row.get("event") for row in live_bash_rows] == [
                        "tool_start",
                        "tool_end",
                    ], live_bash_rows
                    assert all(
                        row.get("timing_source")
                        == "opencode_configured_shell_wrapper"
                        for row in live_bash_rows
                    ), live_bash_rows
                    opencode_config = json.loads(
                        (request_dir / "runtime" / "config" / "opencode.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    assert opencode_config["shell"] == "/run/mars-opencode-shell/bash"
                    assert Path(opencode_config["shell"]).name == "bash"
                    launched_command = json.loads(
                        (request_dir / "command.initial.json").read_text(encoding="utf-8")
                    )
                    assert "--pure" in launched_command, launched_command
                results[agent] = {
                    "status": trace["status"],
                    "round_count": graph["round_count"],
                    "tool_call_count": graph["tool_call_count"],
                    "topology_signature": graph["topology_signature"],
                    "live_tool_inflight_observed": saw_live_inflight,
                }
            if args.full_runner:
                tasks_path = root / "tasks.jsonl"
                tasks_path.write_text(json.dumps(task, separators=(",", ":")) + "\n", encoding="utf-8")
                runner_path = Path(__file__).with_name("run_workload.py")
                for agent in agents:
                    out_dir = root / f"{agent}_full_runner"
                    command = [
                        sys.executable,
                        str(runner_path),
                        "--agent",
                        agent,
                        "--tasks",
                        str(tasks_path),
                        "--out-dir",
                        str(out_dir),
                        "--model",
                        MODEL,
                        "--base-url",
                        server.base_url,
                        "--rps",
                        "10",
                        "--max-workers",
                        "1",
                        "--max-requests",
                        "1",
                        "--context-window",
                        "32768",
                        "--max-output-tokens",
                        "1024",
                        "--max-iterations",
                        "5",
                        "--request-timeout-s",
                        "120",
                        "--scheduling-policy",
                        "mars",
                        "--no-localize-task-inputs",
                    ]
                    if cpu_probe is not None:
                        command.extend(["--cpu-affinity", cpu_probe])
                    subprocess.run(command, check=True, timeout=180)
                    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
                    progress = json.loads((out_dir / "progress.json").read_text(encoding="utf-8"))
                    dynamicity = json.loads(
                        (out_dir / "dynamicity_summary.json").read_text(encoding="utf-8")
                    )
                    full_traces = load_jsonl(out_dir / "traces.jsonl")
                    assert summary["run"]["completed_all_requests"] is True, summary
                    assert summary["status"] == {"completed": 1}, summary
                    assert dynamicity["dynamicity"]["round_count"]["min"] == 2, dynamicity
                    assert (
                        progress["completed_requests"] == 1
                        and progress["inflight_futures"] == 0
                    ), progress
                    if agent == "opencode":
                        assert len(full_traces) == 1
                        full_bash = [
                            tool
                            for tool in full_traces[0]["tool_calls"]
                            if tool.get("tool_name") == "bash"
                        ]
                        assert len(full_bash) == 1, full_bash
                        assert (
                            full_bash[0].get("status") == "completed"
                            and float(full_bash[0].get("duration_s") or 0.0) >= 0.30
                            and (full_bash[0].get("metadata") or {})
                            .get("native_state", {})
                            .get("status")
                            == "completed"
                            and str(
                                (full_bash[0].get("metadata") or {}).get(
                                    "linkage_quality", ""
                                )
                            ).startswith("native_")
                        ), full_bash
                    results[agent]["full_runner"] = "passed"

        verification_task = {
            "schema": "mas_task_v1",
            "request_id": "smoke-verification-nudge",
            "benchmark": "repobench",
            "task_id": "smoke-verification-1",
            "payload": {
                "text": "Answer from the prompt first; inspect the workspace only if explicitly reminded."
            },
            "workspace_init": [
                {"path": "EVIDENCE.txt", "content": "same-session verification evidence\n"}
            ],
        }
        with MockOpenAIServer(verification_nudge=True) as nudge_server:
            for agent in agents:
                agent_root = root / f"{agent}_verification_nudge"
                request_dir = agent_root / "request"
                request_offset = len(nudge_server.state.requests)
                nudge_result = run_request(
                    agent=agent,
                    agent_bin=Path(PINNED[agent]["binary"]),
                    task=verification_task,
                    task_sha256="smoke-verification-nudge-sha256",
                    request_id=f"smoke_{agent}_verification_nudge",
                    mars_job_id=f"smoke_{agent}_verification_nudge",
                    arrival_time_s=time.time(),
                    workspace_root=agent_root / "workspaces",
                    local_inputs_dir=agent_root / "localized_inputs",
                    request_run_dir=request_dir,
                    model=MODEL,
                    base_url=nudge_server.base_url,
                    api_key="EMPTY",
                    context_window=32768,
                    max_output_tokens=1024,
                    max_iterations=5,
                    temperature=0.0,
                    top_p=None,
                    timeout_s=120.0,
                    require_bwrap=True,
                    cpu_affinity=cpu_probe,
                )
                trace = nudge_result["trace"]
                phase_names = [item["phase"] for item in trace["phase_results"]]
                assert trace["status"] == "completed", trace.get("error")
                assert trace["verification_nudge_count"] == 1, trace
                assert trace["initial_verification_tool_count"] == 0, trace
                assert trace["verification_tool_count"] == 1, trace
                assert trace["agent_session_id"], trace
                assert phase_names == ["initial", "verification_nudge_1"], phase_names
                assert trace["phase_returncodes"] == {
                    "initial": 0,
                    "verification_nudge_1": 0,
                }, trace["phase_returncodes"]
                assert len(trace["llm_calls"]) == 3, trace["llm_calls"]
                assert [call["llm_call_id"] for call in trace["llm_calls"]] == [
                    "llm_0001",
                    "llm_0002",
                    "llm_0003",
                ], trace["llm_calls"]
                assert [call.get("phase") for call in trace["llm_calls"]] == [
                    "initial",
                    "verification_nudge_1",
                    "verification_nudge_1",
                ], trace["llm_calls"]
                assert (
                    trace["phase_results"][1]["timeout_budget_s"]
                    < trace["phase_results"][0]["timeout_budget_s"]
                ), trace["phase_results"]
                assert len(trace["tool_calls"]) == 1, trace["tool_calls"]
                assert trace["tool_calls"][0]["phase"] == "verification_nudge_1"
                assert len(
                    [item for item in nudge_result["events"] if item["event"] == "verification_nudge"]
                ) == 1, nudge_result["events"]

                initial_rows = load_jsonl(request_dir / "cli_events_timed.initial.jsonl")
                followup_rows = load_jsonl(
                    request_dir / "cli_events_timed.verification_nudge_1.jsonl"
                )
                assert _native_session_id(agent, initial_rows) == trace["agent_session_id"]
                followup_session = _native_session_id(agent, followup_rows)
                assert followup_session in {None, trace["agent_session_id"]}
                followup_command = json.loads(
                    (request_dir / "command.verification_nudge_1.json").read_text(
                        encoding="utf-8"
                    )
                )
                if agent == "codex":
                    resume_index = followup_command.index("resume")
                    assert followup_command[resume_index + 1] == trace["agent_session_id"]
                    for option in ("--sandbox", "--cd", "--color", "--model"):
                        assert followup_command.index(option) < resume_index
                else:
                    session_index = followup_command.index("--session")
                    assert followup_command[session_index + 1] == trace["agent_session_id"]

                canonical_rows = load_jsonl(request_dir / "cli_events_timed.jsonl")
                assert len(canonical_rows) == len(initial_rows) + len(followup_rows)
                assert all(
                    "top_p" not in payload
                    for payload in nudge_server.state.requests[request_offset:]
                ), nudge_server.state.requests[request_offset:]
                if agent == "opencode":
                    assert trace["llm_num_retries"] is None
                    assert trace["requested_llm_num_retries"] == 3
                    assert trace["llm_retry_policy"]["sdk_max_retries"] == 0
                    assert trace["llm_retry_policy"]["outer_max_retries"] == 5
                else:
                    assert trace["llm_num_retries"] == 3
                    assert trace["llm_retry_policy"]["request_max_retries"] == 3
                results[agent]["same_session_verification_nudge"] = "passed"

            if args.full_runner:
                tasks_path = root / "verification_nudge_tasks.jsonl"
                tasks_path.write_text(
                    json.dumps(verification_task, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                runner_path = Path(__file__).with_name("run_workload.py")
                for agent in agents:
                    out_dir = root / f"{agent}_verification_nudge_full_runner"
                    command = [
                        sys.executable,
                        str(runner_path),
                        "--agent",
                        agent,
                        "--tasks",
                        str(tasks_path),
                        "--out-dir",
                        str(out_dir),
                        "--model",
                        MODEL,
                        "--base-url",
                        nudge_server.base_url,
                        "--rps",
                        "10",
                        "--max-workers",
                        "1",
                        "--max-requests",
                        "1",
                        "--context-window",
                        "32768",
                        "--max-output-tokens",
                        "1024",
                        "--max-iterations",
                        "5",
                        "--request-timeout-s",
                        "120",
                        "--scheduling-policy",
                        "mars",
                        "--no-localize-task-inputs",
                    ]
                    if cpu_probe is not None:
                        command.extend(["--cpu-affinity", cpu_probe])
                    subprocess.run(command, check=True, timeout=180)
                    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
                    progress = json.loads((out_dir / "progress.json").read_text(encoding="utf-8"))
                    traces = load_jsonl(out_dir / "traces.jsonl")
                    run_config = json.loads(
                        (out_dir / "run_config.json").read_text(encoding="utf-8")
                    )["config"]
                    assert summary["run"]["completed_all_requests"] is True, summary
                    assert summary["status"] == {"completed": 1}, summary
                    assert progress["completed_requests"] == 1, progress
                    assert len(traces) == 1 and traces[0]["verification_nudge_count"] == 1
                    assert traces[0]["agent_session_id"]
                    assert run_config["top_p"] is None
                    if agent == "opencode":
                        assert run_config["llm_num_retries"] is None
                        assert run_config["llm_retry_policy"]["sdk_max_retries"] == 0
                        assert run_config["llm_retry_policy"]["outer_max_retries"] == 5
                    else:
                        assert run_config["llm_num_retries"] == 3
                    results[agent]["same_session_nudge_full_runner"] = "passed"
        if "codex" in agents:
            with MockOpenAIServer(malformed_tool_arguments=True) as malformed_server:
                malformed_root = root / "codex_malformed_tool_json"
                request_dir = malformed_root / "request"
                malformed_result = run_request(
                    agent="codex",
                    agent_bin=Path(PINNED["codex"]["binary"]),
                    task=task,
                    task_sha256="smoke-malformed-tool-json-sha256",
                    request_id="smoke_codex_malformed_tool_json",
                    mars_job_id="smoke_codex_malformed_tool_json",
                    arrival_time_s=time.time(),
                    workspace_root=malformed_root / "workspaces",
                    local_inputs_dir=malformed_root / "localized_inputs",
                    request_run_dir=request_dir,
                    model=MODEL,
                    base_url=malformed_server.base_url,
                    api_key="EMPTY",
                    context_window=32768,
                    max_output_tokens=1024,
                    max_iterations=5,
                    temperature=0.0,
                    timeout_s=120.0,
                    require_bwrap=True,
                    cpu_affinity=cpu_probe,
                )
                malformed_trace = malformed_result["trace"]
                assert malformed_trace["status"] == "completed", malformed_trace.get("error")
                assert sum(
                    int(call.get("output_tool_arguments_sanitized") or 0)
                    for call in malformed_trace["llm_calls"]
                ) == 1, malformed_trace["llm_calls"]
                proxy_events = [
                    json.loads(line)
                    for line in (request_dir / "llm_events.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                assert any(
                    event.get("event") == "tool_arguments_sanitized" for event in proxy_events
                ), proxy_events
                results["codex"]["malformed_tool_json_replay"] = "passed"
        print(json.dumps({"output": str(root), "results": results}, indent=2))
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
