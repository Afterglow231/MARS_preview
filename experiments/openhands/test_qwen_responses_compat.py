from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import qwen_responses_compat as compat
from qwen_responses_compat import (
    ResponsesCompatibilityError,
    chat_stream_to_responses,
    normalize_codex_responses_wire_body,
    responses_input_to_chat_messages,
    responses_to_chat_request,
)
from vllm.entrypoints.openai.protocol import ResponsesRequest


AGENT_CLI_DIR = Path(__file__).resolve().parents[1] / "agent_cli"
if str(AGENT_CLI_DIR) in sys.path:
    sys.path.remove(str(AGENT_CLI_DIR))
sys.path.insert(0, str(AGENT_CLI_DIR))


def _request(**updates: Any) -> ResponsesRequest:
    body: dict[str, Any] = {
        "model": "qwen-test",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "inspect"}],
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "description": "run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": True,
        "store": False,
        "max_output_tokens": 128,
        "temperature": 0.0,
        "job_id": "req_test",
    }
    body.update(updates)
    return ResponsesRequest.model_validate(body)


def test_history_preserves_parallel_calls_and_sanitizes_only_bad_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        compat._LOGGER,
        "warning",
        lambda message, *args, **kwargs: warnings.append(message % args),
    )
    request = _request(
        input=[
            {"type": "message", "role": "user", "content": "go"},
            {
                "type": "function_call",
                "call_id": "call_a",
                "name": "exec_command",
                "arguments": '{"cmd":"pwd"',
            },
            {
                "type": "function_call",
                "call_id": "call_b",
                "name": "exec_command",
                "arguments": '{"cmd":"ls"}',
            },
            {"type": "function_call_output", "call_id": "call_a", "output": "a"},
            {"type": "function_call_output", "call_id": "call_b", "output": "b"},
        ]
    )
    messages = responses_input_to_chat_messages(request)
    assistant = messages[1]
    assert [call["id"] for call in assistant["tool_calls"]] == ["call_a", "call_b"]
    malformed = assistant["tool_calls"][0]["function"]["arguments"]
    assert json.loads(malformed) == {
        "_mars_malformed_tool_arguments": '{"cmd":"pwd"'
    }
    assert assistant["tool_calls"][1]["function"]["arguments"] == '{"cmd":"ls"}'
    assert [message["tool_call_id"] for message in messages[2:]] == ["call_a", "call_b"]
    assert any("MARS_QWEN_HISTORY_TOOL_ARGUMENTS_SANITIZED" in item for item in warnings)

    # This is the exact vLLM history postprocessor that previously crashed
    # with JSONDecodeError before the request ever reached the model.
    from vllm.entrypoints.chat_utils import _postprocess_messages

    _postprocess_messages(messages)
    assert assistant["tool_calls"][0]["function"]["arguments"] == {
        "_mars_malformed_tool_arguments": '{"cmd":"pwd"'
    }
    assert assistant["tool_calls"][1]["function"]["arguments"] == {"cmd": "ls"}


def test_only_function_tools_are_accepted() -> None:
    with pytest.raises((ResponsesCompatibilityError, ValueError), match="only function tools"):
        responses_to_chat_request(
            _request(
                tools=[
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "description": "patch",
                        "format": {"type": "text"},
                    }
                ]
            )
        )


def test_codex_disabled_reasoning_include_is_an_explicit_noop() -> None:
    chat = responses_to_chat_request(
        _request(
            reasoning={},
            include=["reasoning.encrypted_content"],
        )
    )
    assert chat.model == "qwen-test"
    assert len(chat.tools or []) == 1


def test_codex_developer_message_is_preserved_as_qwen_system_message() -> None:
    chat = responses_to_chat_request(
        _request(
            input=[
                {"type": "message", "role": "developer", "content": "policy"},
                {"type": "message", "role": "user", "content": "task"},
            ]
        )
    )
    assert [message["role"] for message in chat.messages] == ["system", "user"]
    assert [message["content"] for message in chat.messages] == ["policy", "task"]


def test_nonstreaming_codex_wire_is_rejected_explicitly() -> None:
    with pytest.raises(ResponsesCompatibilityError, match="non-streaming Responses"):
        responses_to_chat_request(_request(stream=False))


def test_actual_codex_assistant_output_history_is_normalized_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        compat._LOGGER,
        "warning",
        lambda message, *args, **kwargs: warnings.append(message % args),
    )
    payload = _request().model_dump(mode="json", exclude_none=True)
    payload["input"] = [
        {"type": "message", "role": "user", "content": "inspect"},
        {
            "type": "message",
            "id": "msg_actual_codex",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "I will inspect the file.",
                }
            ],
        },
        {
            "type": "function_call",
            "id": "fc_actual_codex",
            "call_id": "call_actual_codex",
            "name": "exec_command",
            "arguments": '{"cmd":"cat EVIDENCE.txt"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_actual_codex",
            "output": "marker",
        },
    ]
    raw = json.dumps(payload).encode("utf-8")
    normalized, inserted = normalize_codex_responses_wire_body(raw)

    assert inserted == 2
    parsed = json.loads(normalized)
    assert parsed["input"][1]["status"] == "completed"
    assert parsed["input"][1]["content"][0]["annotations"] == []
    assert any("MARS_CODEX_RESPONSES_WIRE_NORMALIZED" in item for item in warnings)

    # This gate is intentionally before the adapter: it proves the exact
    # actual-Codex shape now reaches vLLM's ResponsesRequest successfully.
    request = ResponsesRequest.model_validate(parsed)
    chat = responses_to_chat_request(request)
    assert [message["role"] for message in chat.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert chat.messages[1]["content"] == "I will inspect the file."
    assert list(chat.messages[1]["tool_calls"])[0]["id"] == "call_actual_codex"


def test_active_reasoning_controls_are_rejected() -> None:
    with pytest.raises(ResponsesCompatibilityError, match="reasoning controls"):
        responses_to_chat_request(_request(reasoning={"summary": "auto"}))


def test_pinned_codex_real_wire_is_qwen_adapter_compatible(tmp_path: Path) -> None:
    """Gate the real Codex request shape, not a hand-authored approximation."""

    # The complete OpenHands test suite may have imported its worker module
    # under the same top-level name. Production puts agent_cli first; mirror
    # that import boundary explicitly for this cross-stack gate.
    loaded_worker = sys.modules.get("request_worker")
    if (
        loaded_worker is not None
        and Path(loaded_worker.__file__).resolve().parent != AGENT_CLI_DIR
    ):
        sys.modules.pop("request_worker", None)
        sys.modules.pop("native_request_worker", None)
    if str(AGENT_CLI_DIR) in sys.path:
        sys.path.remove(str(AGENT_CLI_DIR))
    sys.path.insert(0, str(AGENT_CLI_DIR))
    from native_request_worker import run_request
    from request_worker import _working_bwrap
    from run_workload import PINNED
    from smoke_test import MODEL, MockOpenAIServer

    codex_binary = Path(PINNED["codex"]["binary"])
    if not codex_binary.is_file() or not _working_bwrap():
        pytest.skip("pinned Codex and bubblewrap are required for the real-wire gate")
    task = {
        "schema": "mas_task_v1",
        "request_id": "codex-qwen-wire-gate",
        "benchmark": "smoke",
        "task_id": "codex-qwen-wire-gate",
        "payload": {
            "text": "Inspect README.txt with a shell command, then finish."
        },
        "workspace_init": [{"path": "README.txt", "content": "wire gate\n"}],
    }
    with MockOpenAIServer() as server:
        result = run_request(
            agent="codex",
            agent_bin=codex_binary,
            task=task,
            task_sha256="codex-qwen-wire-gate",
            request_id="codex-qwen-wire-gate",
            mars_job_id="codex-qwen-wire-gate",
            arrival_time_s=time.time(),
            workspace_root=tmp_path / "workspaces",
            local_inputs_dir=tmp_path / "inputs",
            request_run_dir=tmp_path / "request",
            model=MODEL,
            base_url=server.base_url,
            api_key="EMPTY",
            context_window=32768,
            max_output_tokens=1024,
            max_iterations=5,
            temperature=0.0,
            timeout_s=120.0,
            require_bwrap=True,
        )
        payloads = list(server.state.requests)

    assert result["trace"]["status"] == "completed"
    assert len(payloads) == 2
    converted = []
    for payload in payloads:
        assert {tool.get("type") for tool in payload["tools"]} == {"function"}
        assert payload.get("reasoning") in ({}, None)
        request = ResponsesRequest.model_validate(payload)
        converted.append(responses_to_chat_request(request))
        assert all(message["role"] != "developer" for message in converted[-1].messages)
    assert any(message["role"] == "tool" for message in converted[1].messages)


def test_stream_keeps_parallel_call_ids_and_emits_one_terminal_response() -> None:
    request = _request()

    async def chunks():
        rows = [
            {
                "id": "chat-1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "function": {
                                        "name": "exec_command",
                                        "arguments": '{"cmd":"pwd"}',
                                    },
                                },
                                {
                                    "index": 1,
                                    "id": "call_b",
                                    "function": {
                                        "name": "exec_command",
                                        "arguments": '{"cmd":"ls"}',
                                    },
                                },
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chat-1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            },
        ]
        for row in rows:
            yield f"data: {json.dumps(row)}\n\n"
        yield "data: [DONE]\n\n"

    async def collect() -> list[str]:
        return [chunk async for chunk in chat_stream_to_responses(request, chunks())]

    wire = asyncio.run(collect())
    events = [json.loads(line[5:]) for chunk in wire for line in chunk.splitlines() if line.startswith("data:")]
    done_items = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
    ]
    assert [item["call_id"] for item in done_items] == ["call_a", "call_b"]
    assert [item["arguments"] for item in done_items] == [
        '{"cmd":"pwd"}',
        '{"cmd":"ls"}',
    ]
    terminal = [event for event in events if event["type"] == "response.completed"]
    assert len(terminal) == 1
    assert terminal[0]["response"]["usage"]["input_tokens_details"]["cached_tokens"] == 4
    assert [event["sequence_number"] for event in events] == list(range(len(events)))


def test_length_finish_is_incomplete_even_with_partial_tool_arguments() -> None:
    request = _request()

    async def chunks():
        rows = [
            {
                "id": "chat-length",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_partial",
                                    "function": {
                                        "name": "exec_command",
                                        "arguments": '{"cmd":"pwd"',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chat-length",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "length"}
                ],
            },
        ]
        for row in rows:
            yield f"data: {json.dumps(row)}\n\n"
        yield "data: [DONE]\n\n"

    async def collect() -> list[str]:
        return [chunk async for chunk in chat_stream_to_responses(request, chunks())]

    wire = asyncio.run(collect())
    events = [
        json.loads(line[5:])
        for chunk in wire
        for line in chunk.splitlines()
        if line.startswith("data:")
    ]
    assert not any(event["type"] == "response.completed" for event in events)
    terminal = [event for event in events if event["type"] == "response.incomplete"]
    assert len(terminal) == 1
    assert terminal[0]["response"]["status"] == "incomplete"
    assert terminal[0]["response"]["incomplete_details"] == {
        "reason": "max_output_tokens"
    }
    done_items = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
    ]
    assert done_items and all(item["status"] == "incomplete" for item in done_items)


def test_missing_finish_reason_fails_closed() -> None:
    request = _request()

    async def chunks():
        yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': 'partial'}}]})}\n\n"
        yield "data: [DONE]\n\n"

    async def collect() -> list[str]:
        return [chunk async for chunk in chat_stream_to_responses(request, chunks())]

    wire = asyncio.run(collect())
    events = [
        json.loads(line[5:])
        for chunk in wire
        for line in chunk.splitlines()
        if line.startswith("data:")
    ]
    assert not any(event["type"] == "response.completed" for event in events)
    terminal = [event for event in events if event["type"] == "response.failed"]
    assert len(terminal) == 1
    assert terminal[0]["response"]["error"]["code"] == "backend_finish_reason_error"
