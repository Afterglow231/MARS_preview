#!/usr/bin/env python3
"""Strict Responses-API compatibility for non-Harmony tool-capable models.

Codex owns the agent loop and speaks the Responses API.  Qwen's supported
vLLM inference path owns prompt rendering and tool parsing through Chat
Completions.  This module bridges exactly one model inference at the server
boundary; it never stores a conversation, executes a tool, or decides whether
the agent should continue.

Unsupported Responses features fail explicitly.  In particular, custom,
freeform, namespace, hosted and image tools are never silently discarded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping
from http import HTTPStatus
from typing import Any

from fastapi import Request

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    ResponsesRequest,
    ResponsesResponse,
)
from vllm.entrypoints.openai.serving_responses import OpenAIServingResponses


_ORIGINAL_CREATE_RESPONSES = OpenAIServingResponses.create_responses
_INSTALLED = False
_LOGGER = logging.getLogger("vllm.entrypoints.openai.qwen_responses_compat")


class ResponsesCompatibilityError(ValueError):
    """A Responses feature cannot be represented by the Qwen chat backend."""


def normalize_codex_responses_wire_body(body: bytes) -> tuple[bytes, int]:
    """Normalize Codex history fields that vLLM rejects before route dispatch.

    Codex 0.149.1 replays assistant output messages without the response-only
    ``status`` and ``annotations`` fields.  OpenAI accepts that input shape,
    but the pinned Python SDK types used by vLLM require both fields and return
    a 422/400 before :class:`ResponsesRequest` or the Qwen adapter can run.

    This is a stateless wire-schema normalization at the inference server.  It
    does not add, remove, reorder, or retain history items, and the inserted
    fields are serialization metadata ignored by the Qwen chat template.
    """

    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return body, 0
    if not isinstance(payload, dict) or not isinstance(payload.get("input"), list):
        return body, 0

    changed = 0
    for raw_item in payload["input"]:
        if not isinstance(raw_item, dict):
            continue
        if raw_item.get("type") != "message" or raw_item.get("role") != "assistant":
            continue
        if "status" not in raw_item:
            raw_item["status"] = "completed"
            changed += 1
        content = raw_item.get("content")
        if not isinstance(content, list):
            continue
        for raw_part in content:
            if not isinstance(raw_part, dict) or raw_part.get("type") != "output_text":
                continue
            if "annotations" not in raw_part:
                raw_part["annotations"] = []
                changed += 1
    if not changed:
        return body, 0

    request_id = payload.get("request_id")
    job_id = payload.get("job_id")
    _LOGGER.warning(
        "MARS_CODEX_RESPONSES_WIRE_NORMALIZED "
        "request_id=%s job_id=%s inserted_fields=%d raw_sha256=%s",
        request_id,
        job_id,
        changed,
        hashlib.sha256(body).hexdigest(),
    )
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        changed,
    )


class CodexResponsesWireCompatibilityMiddleware:
    """ASGI middleware that runs before FastAPI's ResponsesRequest validation."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/v1/responses"
        ):
            await self.app(scope, receive, send)
            return

        headers = list(scope.get("headers") or [])
        header_map = {key.lower(): value for key, value in headers}
        if header_map.get(b"content-encoding", b"identity").lower() not in {
            b"",
            b"identity",
        }:
            await self.app(scope, receive, send)
            return

        body_parts: list[bytes] = []
        pending_messages: list[dict[str, Any]] = []
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                pending_messages.append(message)
                break
            body_parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        original_body = b"".join(body_parts)
        normalized_body, _ = normalize_codex_responses_wire_body(original_body)

        replayed = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {
                    "type": "http.request",
                    "body": normalized_body,
                    "more_body": False,
                }
            if pending_messages:
                return pending_messages.pop(0)
            return await receive()

        if normalized_body != original_body:
            scope = dict(scope)
            scope["headers"] = [
                (key, value) for key, value in headers if key.lower() != b"content-length"
            ] + [(b"content-length", str(len(normalized_body)).encode("ascii"))]
        await self.app(scope, replay_receive, send)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    return value


def _text_content(value: Any, *, where: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) or not isinstance(value, Iterable):
        raise ResponsesCompatibilityError(f"{where} content must be text")
    # Pydantic represents Iterable fields from OpenAI TypedDicts as a
    # single-use ValidatorIterator rather than list.  Materialize it exactly
    # once before applying the same strict text-part validation.
    value = list(value)
    parts: list[str] = []
    for index, raw_part in enumerate(value):
        part = _dump(raw_part)
        if not isinstance(part, dict):
            raise ResponsesCompatibilityError(f"{where} content[{index}] is invalid")
        kind = str(part.get("type") or "")
        if kind not in {"input_text", "output_text", "text"}:
            raise ResponsesCompatibilityError(
                f"{where} content type {kind!r} is unsupported by Qwen Responses compatibility"
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise ResponsesCompatibilityError(f"{where} content[{index}].text must be a string")
        parts.append(text)
    return "\n".join(parts)


def _flush_assistant_calls(
    messages: list[dict[str, Any]], pending: list[dict[str, Any]]
) -> None:
    if not pending:
        return
    # Responses represents assistant text and function calls as sibling output
    # items, while Chat Completions represents them in one assistant message.
    # Preserve that response grouping when the calls immediately follow an
    # assistant text item instead of manufacturing an extra assistant turn.
    if (
        messages
        and messages[-1].get("role") == "assistant"
        and "tool_calls" not in messages[-1]
    ):
        messages[-1]["tool_calls"] = list(pending)
    else:
        messages.append(
            {"role": "assistant", "content": None, "tool_calls": list(pending)}
        )
    pending.clear()


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _qwen_history_tool_arguments(
    arguments: str,
    *,
    request: ResponsesRequest,
    input_index: int,
    call_id: str,
) -> str:
    """Return a JSON-object string accepted by vLLM's Qwen chat template.

    Codex retains the model's exact argument string in its native history.  A
    malformed or non-object string cannot pass vLLM's ``json.loads`` prompt
    postprocessor, so only the server-local Chat request copy is wrapped in an
    auditable object.  The original bytes remain inside that object and the
    request-local proxy hash remains unchanged.
    """

    reason: str | None = None
    try:
        parsed = json.loads(
            arguments,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (TypeError, ValueError):
        parsed = None
        reason = "malformed_json"
        sanitized = {"_mars_malformed_tool_arguments": arguments}
    else:
        if isinstance(parsed, dict):
            return arguments
        reason = "non_object_json"
        sanitized = {"_mars_non_object_tool_arguments": parsed}
    _LOGGER.warning(
        "MARS_QWEN_HISTORY_TOOL_ARGUMENTS_SANITIZED "
        "request_id=%s job_id=%s input_index=%d call_id=%s reason=%s raw_sha256=%s",
        request.request_id,
        request.job_id,
        input_index,
        call_id,
        reason,
        hashlib.sha256(arguments.encode("utf-8", errors="surrogatepass")).hexdigest(),
    )
    return json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))


def responses_input_to_chat_messages(request: ResponsesRequest) -> list[dict[str, Any]]:
    """Convert the complete Responses input history without timestamp inference."""

    if request.previous_response_id is not None:
        raise ResponsesCompatibilityError(
            "previous_response_id is unsupported; Codex must send explicit response history"
        )
    messages: list[dict[str, Any]] = []
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})
    values: list[Any]
    if isinstance(request.input, str):
        values = [{"type": "message", "role": "user", "content": request.input}]
    else:
        values = list(request.input)

    pending_calls: list[dict[str, Any]] = []
    for index, raw_item in enumerate(values):
        item = _dump(raw_item)
        if not isinstance(item, dict):
            raise ResponsesCompatibilityError(f"input[{index}] is invalid")
        kind = str(item.get("type") or "")
        if kind == "function_call":
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                raise ResponsesCompatibilityError(
                    f"input[{index}].arguments must be an unmodified JSON string"
                )
            name = item.get("name")
            call_id = item.get("call_id") or item.get("id")
            if not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
                raise ResponsesCompatibilityError(f"input[{index}] has an invalid function call")
            arguments = _qwen_history_tool_arguments(
                arguments,
                request=request,
                input_index=index,
                call_id=call_id,
            )
            pending_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
            continue

        _flush_assistant_calls(messages, pending_calls)
        if kind == "function_call_output":
            call_id = item.get("call_id")
            output = item.get("output")
            if not isinstance(call_id, str) or not call_id:
                raise ResponsesCompatibilityError(f"input[{index}] has no function call_id")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
            messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
            continue
        if kind == "message":
            role = str(item.get("role") or "")
            if role not in {"user", "assistant", "system", "developer"}:
                raise ResponsesCompatibilityError(
                    f"input[{index}] message role {role!r} is unsupported"
                )
            phase = item.get("phase")
            if phase not in (None, ""):
                raise ResponsesCompatibilityError(
                    f"input[{index}] message phase is unsupported by the Qwen chat template"
                )
            messages.append(
                {
                    # The pinned Qwen3-Coder template has no developer branch;
                    # leaving this role unchanged would silently drop Codex's
                    # developer instructions.  System is the corresponding
                    # highest-priority role supported by this model template.
                    "role": "system" if role == "developer" else role,
                    "content": _text_content(item.get("content"), where=f"input[{index}]"),
                }
            )
            continue
        raise ResponsesCompatibilityError(
            f"input[{index}] type {kind!r} is unsupported by Qwen Responses compatibility"
        )
    _flush_assistant_calls(messages, pending_calls)
    return messages


def responses_tools_to_chat(request: ResponsesRequest) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw_tool in enumerate(request.tools):
        tool = _dump(raw_tool)
        if not isinstance(tool, dict) or tool.get("type") != "function":
            kind = tool.get("type") if isinstance(tool, dict) else type(tool).__name__
            raise ResponsesCompatibilityError(
                f"tools[{index}] type {kind!r} is unsupported; only function tools are allowed"
            )
        name = tool.get("name")
        parameters = tool.get("parameters")
        if not isinstance(name, str) or not name or not isinstance(parameters, dict):
            raise ResponsesCompatibilityError(f"tools[{index}] is not a valid function tool")
        function: dict[str, Any] = {"name": name, "parameters": parameters}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        if isinstance(tool.get("strict"), bool):
            function["strict"] = tool["strict"]
        result.append({"type": "function", "function": function})
    return result


def _chat_tool_choice(value: Any) -> Any:
    value = _dump(value)
    if value in {"auto", "none", "required"}:
        return value
    if isinstance(value, dict) and value.get("type") == "function":
        name = value.get("name")
        if isinstance(name, str) and name:
            return {"type": "function", "function": {"name": name}}
    raise ResponsesCompatibilityError(f"tool_choice {value!r} is unsupported")


def responses_to_chat_request(request: ResponsesRequest) -> ChatCompletionRequest:
    if request.background:
        raise ResponsesCompatibilityError("background Responses are unsupported")
    if request.prompt is not None:
        raise ResponsesCompatibilityError("prompt templates are unsupported")
    if not request.stream:
        raise ResponsesCompatibilityError(
            "non-streaming Responses are unsupported by the pinned Codex/Qwen bridge"
        )
    if request.reasoning is not None:
        reasoning = _dump(request.reasoning)
        if reasoning not in ({}, None):
            raise ResponsesCompatibilityError(
                "Responses reasoning controls are unsupported for this Qwen backend"
            )
    if request.include:
        # Codex asks for encrypted reasoning content even when
        # ``model_reasoning_summary = \"none\"`` produces no reasoning item.
        # In that disabled-reasoning configuration the include is a no-op, so
        # accepting it does not drop any model output or change the tool loop.
        unsupported = set(request.include) - {
            "message.output_text.logprobs",
            "reasoning.encrypted_content",
        }
        if unsupported:
            raise ResponsesCompatibilityError(
                f"unsupported include fields: {sorted(unsupported)!r}"
            )

    body: dict[str, Any] = {
        "model": request.model,
        "messages": responses_input_to_chat_messages(request),
        "stream": bool(request.stream),
        "stream_options": {"include_usage": True} if request.stream else None,
        "max_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "parallel_tool_calls": request.parallel_tool_calls,
        "tools": responses_tools_to_chat(request),
        "tool_choice": _chat_tool_choice(request.tool_choice),
        "job_id": request.job_id,
        "last_func_call": request.last_func_call,
        "is_last_step": request.is_last_step,
        "this_func_call": request.this_func_call,
        "request_id": request.request_id.removeprefix("resp_"),
        "priority": request.priority,
        "user": request.user,
    }
    if not body["tools"]:
        body["tools"] = None
        if body["tool_choice"] == "auto":
            body["tool_choice"] = "none"
    return ChatCompletionRequest.model_validate(
        {key: value for key, value in body.items() if value is not None}
    )


def _usage_from_chat(raw: Any) -> dict[str, Any] | None:
    value = _dump(raw)
    if not isinstance(value, dict):
        return None
    prompt = int(value.get("prompt_tokens") or 0)
    completion = int(value.get("completion_tokens") or 0)
    prompt_details = value.get("prompt_tokens_details") or {}
    completion_details = value.get("completion_tokens_details") or {}
    return {
        "input_tokens": prompt,
        "input_tokens_details": {
            "cached_tokens": int(prompt_details.get("cached_tokens") or 0)
        },
        "output_tokens": completion,
        "output_tokens_details": {
            "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0)
        },
        "total_tokens": int(value.get("total_tokens") or prompt + completion),
    }


def _base_response(
    request: ResponsesRequest,
    *,
    response_id: str,
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": time.time(),
        "completed_at": time.time() if status == "completed" else None,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": request.instructions,
        "metadata": request.metadata,
        "model": request.model or "",
        "output": output,
        "parallel_tool_calls": bool(request.parallel_tool_calls),
        "temperature": request.temperature,
        "tool_choice": _dump(request.tool_choice),
        "tools": [_dump(tool) for tool in request.tools],
        "top_p": request.top_p,
        "background": False,
        "max_output_tokens": request.max_output_tokens,
        "max_tool_calls": request.max_tool_calls,
        "previous_response_id": request.previous_response_id,
        "reasoning": _dump(request.reasoning),
        "service_tier": request.service_tier,
        "status": status,
        "text": _dump(request.text),
        "top_logprobs": request.top_logprobs,
        "truncation": request.truncation,
        "usage": usage,
        "user": request.user,
    }


def _sse(event: dict[str, Any]) -> str:
    kind = str(event["type"])
    return f"event: {kind}\ndata: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _iter_sse_json(chunk: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        value = line[5:].strip()
        if not value or value == "[DONE]":
            continue
        try:
            parsed = json.loads(value)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            result.append(parsed)
    return result


async def chat_stream_to_responses(
    request: ResponsesRequest,
    stream: AsyncIterator[str],
) -> AsyncGenerator[str, None]:
    """Convert Qwen parser output to a valid Responses event stream."""

    response_id = request.request_id
    if not response_id.startswith("resp_"):
        response_id = f"resp_{response_id}"
    sequence = 0
    output: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    message_id = f"msg_{uuid.uuid4().hex}"
    message_started = False
    message_text = ""
    tool_items: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None

    def event(kind: str, **payload: Any) -> str:
        nonlocal sequence
        row = {"type": kind, "sequence_number": sequence, **payload}
        sequence += 1
        return _sse(row)

    initial = _base_response(
        request, response_id=response_id, status="in_progress", output=[], usage=None
    )
    yield event("response.created", response=initial)
    yield event("response.in_progress", response=initial)
    try:
        async for raw_chunk in stream:
            for chunk in _iter_sse_json(raw_chunk):
                if chunk.get("usage") is not None:
                    usage = _usage_from_chat(chunk.get("usage"))
                choices = chunk.get("choices")
                if not isinstance(choices, list):
                    continue
                for choice in choices:
                    if not isinstance(choice, dict) or int(choice.get("index") or 0) != 0:
                        continue
                    raw_finish_reason = choice.get("finish_reason")
                    if raw_finish_reason is not None:
                        current_finish_reason = str(raw_finish_reason)
                        if (
                            finish_reason is not None
                            and finish_reason != current_finish_reason
                        ):
                            raise ResponsesCompatibilityError(
                                "Chat stream emitted conflicting finish reasons: "
                                f"{finish_reason!r} then {current_finish_reason!r}"
                            )
                        finish_reason = current_finish_reason
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    text = delta.get("content")
                    if isinstance(text, str) and text:
                        if not message_started:
                            message_started = True
                            yield event(
                                "response.output_item.added",
                                output_index=0,
                                item={
                                    "type": "message",
                                    "id": message_id,
                                    "role": "assistant",
                                    "content": [],
                                    "status": "in_progress",
                                },
                            )
                        message_text += text
                        yield event(
                            "response.output_text.delta",
                            output_index=0,
                            content_index=0,
                            item_id=message_id,
                            delta=text,
                            logprobs=[],
                        )
                    calls = delta.get("tool_calls")
                    if not isinstance(calls, list):
                        continue
                    for raw_call in calls:
                        if not isinstance(raw_call, dict):
                            continue
                        index = int(raw_call.get("index") or 0)
                        item = tool_items.get(index)
                        if item is None:
                            call_id = str(raw_call.get("id") or f"call_{uuid.uuid4().hex}")
                            item = {
                                "type": "function_call",
                                "id": f"fc_{uuid.uuid4().hex}",
                                "call_id": call_id,
                                "name": "",
                                "arguments": "",
                                "status": "in_progress",
                            }
                            tool_items[index] = item
                            yield event(
                                "response.output_item.added",
                                output_index=int(message_started) + index,
                                item=dict(item),
                            )
                        elif isinstance(raw_call.get("id"), str) and raw_call["id"]:
                            item["call_id"] = raw_call["id"]
                        function = raw_call.get("function")
                        if not isinstance(function, dict):
                            continue
                        if isinstance(function.get("name"), str):
                            item["name"] += function["name"]
                        arguments_delta = function.get("arguments")
                        if isinstance(arguments_delta, str) and arguments_delta:
                            item["arguments"] += arguments_delta
                            yield event(
                                "response.function_call_arguments.delta",
                                output_index=int(message_started) + index,
                                item_id=item["id"],
                                delta=arguments_delta,
                            )
    except Exception as exc:
        failed = _base_response(
            request, response_id=response_id, status="failed", output=output, usage=usage
        )
        failed["error"] = {"code": "backend_stream_error", "message": str(exc)}
        yield event("response.failed", response=failed)
        return

    terminal_error: str | None = None
    if finish_reason is None:
        terminal_error = "Chat stream ended without a finish_reason"
    elif finish_reason == "tool_calls" and not tool_items:
        terminal_error = "Chat stream finished with tool_calls but emitted no parsed tool call"
    elif finish_reason == "stop" and tool_items:
        terminal_error = "Chat stream emitted parsed tool calls but finished with stop"
    elif finish_reason not in {"stop", "tool_calls", "length", "content_filter"}:
        terminal_error = f"unsupported Chat finish_reason: {finish_reason!r}"
    if terminal_error is not None:
        failed = _base_response(
            request, response_id=response_id, status="failed", output=[], usage=usage
        )
        failed["error"] = {
            "code": "backend_finish_reason_error",
            "message": terminal_error,
        }
        yield event("response.failed", response=failed)
        return

    is_incomplete = finish_reason in {"length", "content_filter"}
    item_status = "incomplete" if is_incomplete else "completed"

    output_index = 0
    if message_started:
        message_item = {
            "type": "message",
            "id": message_id,
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": message_text, "annotations": [], "logprobs": []}
            ],
            "status": item_status,
        }
        output.append(message_item)
        yield event(
            "response.output_text.done",
            output_index=output_index,
            content_index=0,
            item_id=message_id,
            text=message_text,
            logprobs=[],
        )
        yield event("response.output_item.done", output_index=output_index, item=message_item)
        output_index += 1
    for index in sorted(tool_items):
        item = tool_items[index]
        item["status"] = item_status
        output.append(dict(item))
        yield event(
            "response.function_call_arguments.done",
            output_index=output_index,
            item_id=item["id"],
            name=item["name"],
            arguments=item["arguments"],
        )
        yield event("response.output_item.done", output_index=output_index, item=dict(item))
        output_index += 1
    if not output:
        message_item = {
            "type": "message",
            "id": message_id,
            "role": "assistant",
            "content": [{"type": "output_text", "text": "", "annotations": []}],
            "status": item_status,
        }
        output.append(message_item)
        yield event("response.output_item.done", output_index=0, item=message_item)

    terminal_status = "incomplete" if is_incomplete else "completed"
    completed = _base_response(
        request,
        response_id=response_id,
        status=terminal_status,
        output=output,
        usage=usage
        or {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
    )
    if is_incomplete:
        completed["incomplete_details"] = {
            "reason": (
                "max_output_tokens" if finish_reason == "length" else "content_filter"
            )
        }
        yield event("response.incomplete", response=completed)
    else:
        yield event("response.completed", response=completed)


def _full_chat_to_responses(
    request: ResponsesRequest, response: ChatCompletionResponse
) -> ResponsesResponse:
    if len(response.choices) != 1:
        raise ResponsesCompatibilityError("only one chat choice is supported")
    message = _dump(response.choices[0].message)
    output: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str):
        output.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
                "status": "completed",
            }
        )
    for raw_call in message.get("tool_calls") or []:
        call = _dump(raw_call)
        function = call.get("function") or {}
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex}",
                "call_id": str(call.get("id") or f"call_{uuid.uuid4().hex}"),
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or ""),
                "status": "completed",
            }
        )
    payload = _base_response(
        request,
        response_id=request.request_id,
        status="completed",
        output=output,
        usage=_usage_from_chat(response.usage),
    )
    # The vLLM route recognizes this concrete class and serializes it as JSON.
    return ResponsesResponse.model_validate(payload)


async def _create_responses_with_qwen_tools(
    self: OpenAIServingResponses,
    request: ResponsesRequest,
    raw_request: Request | None = None,
) -> AsyncGenerator[str, None] | ResponsesResponse | ErrorResponse:
    if self.use_harmony or not request.tools:
        return await _ORIGINAL_CREATE_RESPONSES(self, request, raw_request)
    try:
        chat_request = responses_to_chat_request(request)
    except (ResponsesCompatibilityError, ValueError, TypeError) as exc:
        return self.create_error_response(
            err_type="invalid_request_error",
            message=f"Qwen Responses compatibility rejected the request: {exc}",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    if raw_request is None:
        return self.create_error_response(
            err_type="server_error",
            message="raw request context is required for Qwen Responses compatibility",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    chat_handler = getattr(raw_request.app.state, "openai_serving_chat", None)
    if chat_handler is None:
        return self.create_error_response(
            err_type="server_error",
            message="Chat Completions is unavailable for Qwen Responses compatibility",
            status_code=HTTPStatus.NOT_IMPLEMENTED,
        )
    result = await chat_handler.create_chat_completion(chat_request, raw_request)
    if isinstance(result, ErrorResponse):
        return result
    if isinstance(result, ChatCompletionResponse):
        return _full_chat_to_responses(request, result)
    return chat_stream_to_responses(request, result)


def install_qwen_responses_compat() -> None:
    """Install the server-side adapter once, before vLLM creates handlers."""

    global _INSTALLED
    if _INSTALLED:
        return
    OpenAIServingResponses.create_responses = _create_responses_with_qwen_tools
    _INSTALLED = True
