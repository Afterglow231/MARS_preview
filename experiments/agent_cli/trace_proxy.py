#!/usr/bin/env python3
"""Small streaming reverse proxy that records exact LLM request boundaries.

The coding-agent CLIs intentionally remain unmodified.  Each worker points its
OpenAI-compatible provider at this loopback proxy.  It adds the same stable
per-agent ``job_id`` used by the OpenHands bridge, then forwards the request to
MARS/vLLM while recording request/first-token/end timestamps and token usage.
Prompts, tool arguments, and model output are never copied into the trace; only
counts, hashes, ids, and timings are retained.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import socket
import ssl
import threading
import time
import uuid
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit


_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _now() -> float:
    return time.time()


def _safe_json(raw: bytes | str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _usage(payload: dict[str, Any] | None) -> dict[str, int | None]:
    usage = (payload or {}).get("usage")
    if not isinstance(usage, dict):
        response = (payload or {}).get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "cached_input_tokens": None,
            "reasoning_tokens": None,
        }

    def integer(*names: str) -> int | None:
        for name in names:
            value = usage.get(name)
            if isinstance(value, (int, float)):
                return int(value)
        return None

    cached = None
    input_details = usage.get("input_tokens_details")
    if isinstance(input_details, dict):
        value = input_details.get("cached_tokens")
        if isinstance(value, (int, float)):
            cached = int(value)
    output_details = usage.get("output_tokens_details")
    reasoning = None
    if isinstance(output_details, dict):
        value = output_details.get("reasoning_tokens")
        if isinstance(value, (int, float)):
            reasoning = int(value)
    completion_details = usage.get("completion_tokens_details")
    if reasoning is None and isinstance(completion_details, dict):
        value = completion_details.get("reasoning_tokens")
        if isinstance(value, (int, float)):
            reasoning = int(value)
    return {
        "prompt_tokens": integer("prompt_tokens", "input_tokens"),
        "completion_tokens": integer("completion_tokens", "output_tokens"),
        "cached_input_tokens": cached,
        "reasoning_tokens": reasoning,
    }


def _response_id(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("id")
    if isinstance(value, str):
        return value
    response = payload.get("response")
    if isinstance(response, dict) and isinstance(response.get("id"), str):
        return str(response["id"])
    return None


def _contains_token_delta(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            if delta.get("content") not in (None, "") or delta.get("tool_calls"):
                return True
    event_type = str(payload.get("type") or "")
    if event_type.endswith(".delta"):
        return payload.get("delta") not in (None, "", [], {})
    return False


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"input_text", "output_text", "text"}:
            value = item.get("text")
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(parts)


def _sanitize_tool_arguments(arguments: Any) -> tuple[str, bool]:
    """Return a valid JSON object string for model output and replay history."""
    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    if isinstance(arguments, str):
        raw = arguments
    else:
        try:
            raw = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            raw = str(arguments)
    try:
        parsed = json.loads(raw, parse_constant=reject_nonstandard_constant)
    except (TypeError, ValueError):
        return (
            json.dumps(
                {"_mars_malformed_tool_arguments": raw},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            True,
        )
    if isinstance(parsed, dict):
        return raw, False
    return (
        json.dumps(
            {"_mars_non_object_tool_arguments": parsed},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        True,
    )


def _responses_to_chat(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, tuple[str | None, str]], int]:
    """Translate the Codex Responses request into a Qwen-compatible chat request."""
    system_parts: list[str] = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        system_parts.append(instructions)
    messages: list[dict[str, Any]] = []
    sanitized_history_arguments = 0
    inputs = payload.get("input")
    if isinstance(inputs, str):
        messages.append({"role": "user", "content": inputs})
    elif isinstance(inputs, list):
        for item in inputs:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "message":
                role = str(item.get("role") or "user")
                content = _text_content(item.get("content"))
                if role in {"system", "developer"}:
                    if content:
                        system_parts.append(content)
                else:
                    messages.append(
                        {
                            "role": role if role in {"user", "assistant", "tool"} else "user",
                            "content": content,
                        }
                    )
            elif kind in {"function_call", "custom_tool_call"}:
                call_id = str(item.get("call_id") or item.get("id") or "")
                name = str(item.get("name") or "")
                namespace = item.get("namespace")
                if isinstance(namespace, str) and namespace:
                    name = f"{namespace}__{name}"
                arguments, sanitized = _sanitize_tool_arguments(
                    item.get("arguments", item.get("input", "{}"))
                )
                sanitized_history_arguments += int(sanitized)
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                        ],
                    }
                )
            elif kind in {"function_call_output", "custom_tool_call_output"}:
                output = item.get("output")
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(item.get("call_id") or ""),
                        "content": output,
                    }
                )
    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

    chat_tools: list[dict[str, Any]] = []
    reverse_names: dict[str, tuple[str | None, str]] = {}

    def add_function(tool: dict[str, Any], namespace: str | None = None) -> None:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            return
        chat_name = f"{namespace}__{name}" if namespace else name
        reverse_names[chat_name] = (namespace, name)
        function = {
            "name": chat_name,
            "description": str(tool.get("description") or ""),
            "parameters": tool.get("parameters")
            if isinstance(tool.get("parameters"), dict)
            else {"type": "object", "properties": {}},
        }
        chat_tools.append({"type": "function", "function": function})

    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function":
            add_function(tool)
        elif tool.get("type") == "namespace" and isinstance(tool.get("name"), str):
            for child in tool.get("tools") or []:
                if isinstance(child, dict) and child.get("type") == "function":
                    add_function(child, namespace=str(tool["name"]))

    chat: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "job_id": payload.get("job_id"),
    }
    if chat_tools:
        chat["tools"] = chat_tools
        chat["tool_choice"] = "auto"
    if isinstance(payload.get("parallel_tool_calls"), bool):
        chat["parallel_tool_calls"] = payload["parallel_tool_calls"]
    if isinstance(payload.get("max_output_tokens"), int):
        chat["max_tokens"] = payload["max_output_tokens"]
    return chat, reverse_names, sanitized_history_arguments


def _responses_sse_event(payload: dict[str, Any]) -> bytes:
    event_type = str(payload.get("type") or "message")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {data}\n\n".encode("utf-8")


class _TraceLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, item: dict[str, Any]) -> None:
        line = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._fp.write(line + "\n")
            self._fp.flush()

    def close(self) -> None:
        with self._lock:
            self._fp.close()


class TraceProxy(AbstractContextManager["TraceProxy"]):
    """Loopback proxy for one agent request."""

    def __init__(
        self,
        *,
        upstream_base_url: str,
        trace_path: Path,
        request_id: str,
        mars_job_id: str,
        responses_via_chat: bool = False,
        max_llm_calls: int | None = None,
        upstream_timeout_s: float = 3000.0,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> None:
        upstream = urlsplit(upstream_base_url.rstrip("/"))
        if upstream.scheme not in {"http", "https"} or not upstream.hostname:
            raise ValueError(f"unsupported upstream base URL: {upstream_base_url}")
        self._upstream: SplitResult = upstream
        self._log = _TraceLog(trace_path)
        self._request_id = request_id
        self._mars_job_id = mars_job_id
        self._responses_via_chat = responses_via_chat
        self._max_llm_calls = max_llm_calls
        self._upstream_timeout_s = max(1.0, float(upstream_timeout_s))
        self._max_output_tokens = (
            max(1, int(max_output_tokens)) if max_output_tokens is not None else None
        )
        self._temperature = float(temperature) if temperature is not None else None
        self._top_p = float(top_p) if top_p is not None else None
        self._limit_exceeded = False
        self._counter = 0
        self._counter_lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        port = int(self._server.server_address[1])
        path = self._upstream.path.rstrip("/")
        return f"http://127.0.0.1:{port}{path}"

    def start(self) -> "TraceProxy":
        self._thread.start()
        return self

    @property
    def limit_exceeded(self) -> bool:
        return self._limit_exceeded

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        self._log.close()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _next_id(self) -> tuple[str, bool]:
        with self._counter_lock:
            self._counter += 1
            over_limit = bool(
                self._max_llm_calls is not None and self._counter > self._max_llm_calls
            )
            if over_limit:
                self._limit_exceeded = True
            return f"llm_{self._counter:04d}", over_limit

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                self._forward()

            def do_POST(self) -> None:  # noqa: N802
                self._forward()

            def do_DELETE(self) -> None:  # noqa: N802
                self._forward()

            def _forward(self) -> None:
                llm_call_id, over_limit = owner._next_id()
                started = _now()
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length > 0 else b""
                request_json = _safe_json(body)
                path = self.path
                configured_prefix = owner._upstream.path.rstrip("/")
                if configured_prefix and not path.startswith(configured_prefix + "/") and path != configured_prefix:
                    path = configured_prefix + (path if path.startswith("/") else "/" + path)

                protocol = (
                    "responses"
                    if path.split("?", 1)[0].endswith("/responses")
                    else "chat_completions"
                    if path.split("?", 1)[0].endswith("/chat/completions")
                    else "other"
                )
                if over_limit and protocol in {"responses", "chat_completions"}:
                    owner._log.write(
                        {
                            "schema": "mars_llm_proxy_event_v1",
                            "event": "agent_iteration_limit",
                            "ts": started,
                            "request_id": owner._request_id,
                            "mars_job_id": owner._mars_job_id,
                            "attempted_llm_call_id": llm_call_id,
                            "max_llm_calls": owner._max_llm_calls,
                        }
                    )
                    message = "Agent iteration budget exhausted; stop now and return your current result."
                    if protocol == "responses":
                        response_id = f"resp-mars-limit-{uuid.uuid4().hex}"
                        message_id = f"msg-mars-limit-{uuid.uuid4().hex}"
                        events = [
                            {
                                "type": "response.created",
                                "response": {"id": response_id, "status": "in_progress"},
                            },
                            {
                                "type": "response.output_text.delta",
                                "output_index": 0,
                                "content_index": 0,
                                "item_id": message_id,
                                "delta": message,
                            },
                            {
                                "type": "response.output_item.done",
                                "output_index": 0,
                                "item": {
                                    "type": "message",
                                    "role": "assistant",
                                    "id": message_id,
                                    "content": [{"type": "output_text", "text": message}],
                                    "status": "completed",
                                },
                            },
                            {
                                "type": "response.completed",
                                "response": {
                                    "id": response_id,
                                    "status": "completed",
                                    "usage": {
                                        "input_tokens": 0,
                                        "input_tokens_details": {"cached_tokens": 0},
                                        "output_tokens": 0,
                                        "output_tokens_details": {"reasoning_tokens": 0},
                                        "total_tokens": 0,
                                    },
                                },
                            },
                        ]
                        response_body = b"".join(_responses_sse_event(item) for item in events)
                    else:
                        chunk = {
                            "id": f"chatcmpl-mars-limit-{uuid.uuid4().hex}",
                            "object": "chat.completion.chunk",
                            "created": int(started),
                            "model": (request_json or {}).get("model"),
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": message},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        }
                        response_body = (
                            b"data: "
                            + json.dumps(chunk, separators=(",", ":")).encode("utf-8")
                            + b"\n\ndata: [DONE]\n\n"
                        )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(response_body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(response_body)
                    self.wfile.flush()
                    self.close_connection = True
                    return
                original_body = body
                mars_metadata: dict[str, str] = {}
                if request_json is not None and protocol in {"responses", "chat_completions"}:
                    # MARS groups all LLM rounds belonging to one dynamic agent
                    # execution by job_id.  Keep this server-facing so neither
                    # upstream CLI needs a fork or a custom provider package.
                    request_json["job_id"] = owner._mars_job_id
                    if owner._max_output_tokens is not None:
                        if protocol == "responses":
                            request_json["max_output_tokens"] = owner._max_output_tokens
                        else:
                            request_json["max_tokens"] = owner._max_output_tokens
                    if owner._temperature is not None:
                        request_json["temperature"] = owner._temperature
                    if owner._top_p is not None:
                        request_json["top_p"] = owner._top_p
                    else:
                        # OpenHands' maintained experiment uses top_p=null,
                        # which means the upstream request omits the field.
                        # Native frontends may inject their own provider
                        # default, so remove it explicitly for parity.
                        request_json.pop("top_p", None)
                    mars_metadata["job_id"] = owner._mars_job_id
                    body = json.dumps(
                        request_json,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                client_path = path
                responses_via_chat = bool(
                    owner._responses_via_chat
                    and protocol == "responses"
                    and request_json is not None
                    and self.command == "POST"
                )
                reverse_tool_names: dict[str, tuple[str | None, str]] = {}
                sanitized_history_arguments = 0
                if responses_via_chat:
                    (
                        chat_request,
                        reverse_tool_names,
                        sanitized_history_arguments,
                    ) = _responses_to_chat(request_json)
                    if owner._max_output_tokens is not None:
                        chat_request["max_tokens"] = owner._max_output_tokens
                    if owner._temperature is not None:
                        chat_request["temperature"] = owner._temperature
                    if owner._top_p is not None:
                        chat_request["top_p"] = owner._top_p
                    else:
                        chat_request.pop("top_p", None)
                    body = json.dumps(
                        chat_request,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    path = path.rsplit("/responses", 1)[0] + "/chat/completions"
                input_value = request_json.get("input") if request_json else None
                messages = request_json.get("messages") if request_json else None
                tools = request_json.get("tools") if request_json else None
                start_record = {
                    "schema": "mars_llm_proxy_event_v1",
                    "event": "llm_start",
                    "ts": started,
                    "request_id": owner._request_id,
                    "mars_job_id": owner._mars_job_id,
                    "llm_call_id": llm_call_id,
                    "method": self.command,
                    "path": client_path.split("?", 1)[0],
                    "upstream_path": path.split("?", 1)[0],
                    "protocol": protocol,
                    "upstream_protocol": "chat_completions" if responses_via_chat else protocol,
                    "responses_via_chat": responses_via_chat,
                    "history_tool_arguments_sanitized": sanitized_history_arguments,
                    "model": request_json.get("model") if request_json else None,
                    "stream": bool(request_json.get("stream")) if request_json else False,
                    "message_count": len(messages) if isinstance(messages, list) else None,
                    "input_item_count": len(input_value) if isinstance(input_value, list) else None,
                    "input_chars": len(input_value) if isinstance(input_value, str) else None,
                    "tool_count": len(tools) if isinstance(tools, list) else 0,
                    "max_output_tokens": owner._max_output_tokens,
                    "temperature": owner._temperature,
                    "top_p": owner._top_p,
                    "previous_response_id": request_json.get("previous_response_id") if request_json else None,
                    "mars_metadata": mars_metadata,
                    "original_body_bytes": len(original_body),
                    "original_body_sha256": hashlib.sha256(original_body).hexdigest(),
                    "forwarded_body_bytes": len(body),
                    "forwarded_body_sha256": hashlib.sha256(body).hexdigest(),
                }
                owner._log.write(start_record)

                conn: http.client.HTTPConnection | http.client.HTTPSConnection | None = None
                first_token_at: float | None = None
                last_payload: dict[str, Any] | None = None
                status: int | None = None
                try:
                    port = owner._upstream.port
                    if owner._upstream.scheme == "https":
                        conn = http.client.HTTPSConnection(
                            owner._upstream.hostname,
                            port or 443,
                            timeout=owner._upstream_timeout_s,
                            context=ssl.create_default_context(),
                        )
                    else:
                        conn = http.client.HTTPConnection(
                            owner._upstream.hostname,
                            port or 80,
                            timeout=owner._upstream_timeout_s,
                        )
                    headers = {
                        key: value
                        for key, value in self.headers.items()
                        if key.lower() not in _HOP_BY_HOP | {"host", "content-length"}
                    }
                    headers["Host"] = owner._upstream.netloc
                    headers["X-MARS-Agent-Request-ID"] = owner._request_id
                    headers["X-MARS-LLM-Call-ID"] = llm_call_id
                    conn.request(self.command, path, body=body or None, headers=headers)
                    response = conn.getresponse()
                    status = int(response.status)
                    self.send_response(status, response.reason)
                    content_type = response.getheader("Content-Type") or ""
                    for key, value in response.getheaders():
                        if key.lower() in _HOP_BY_HOP | {"content-length"}:
                            continue
                        self.send_header(key, value)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True

                    if "text/event-stream" in content_type.lower() and responses_via_chat:
                        response_id = f"resp-mars-{uuid.uuid4().hex}"
                        message_id = f"msg-mars-{uuid.uuid4().hex}"
                        message_started = False
                        message_text = ""
                        tool_deltas: dict[int, dict[str, str]] = {}
                        usage = {
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "cached_input_tokens": None,
                            "reasoning_tokens": None,
                        }

                        def send_event(event: dict[str, Any]) -> None:
                            self.wfile.write(_responses_sse_event(event))
                            self.wfile.flush()

                        send_event(
                            {
                                "type": "response.created",
                                "response": {"id": response_id, "status": "in_progress"},
                            }
                        )
                        while True:
                            line = response.readline()
                            if not line:
                                break
                            stripped = line.strip()
                            if not stripped.startswith(b"data:"):
                                continue
                            raw_data = stripped[5:].strip()
                            if raw_data == b"[DONE]":
                                continue
                            payload = _safe_json(raw_data)
                            if payload is None:
                                continue
                            parsed_usage = _usage(payload)
                            if any(value is not None for value in parsed_usage.values()):
                                usage = parsed_usage
                            if first_token_at is None and _contains_token_delta(payload):
                                first_token_at = _now()
                                owner._log.write(
                                    {
                                        "schema": "mars_llm_proxy_event_v1",
                                        "event": "llm_first_token",
                                        "ts": first_token_at,
                                        "request_id": owner._request_id,
                                        "llm_call_id": llm_call_id,
                                        "ttft_s": first_token_at - started,
                                    }
                                )
                            choices = payload.get("choices")
                            if not isinstance(choices, list):
                                continue
                            for choice in choices:
                                if not isinstance(choice, dict):
                                    continue
                                delta = choice.get("delta")
                                if not isinstance(delta, dict):
                                    continue
                                content = delta.get("content")
                                if isinstance(content, str) and content:
                                    if not message_started:
                                        message_started = True
                                        send_event(
                                            {
                                                "type": "response.output_item.added",
                                                "output_index": 0,
                                                "item": {
                                                    "type": "message",
                                                    "role": "assistant",
                                                    "id": message_id,
                                                    "content": [],
                                                    "status": "in_progress",
                                                },
                                            }
                                        )
                                    message_text += content
                                    send_event(
                                        {
                                            "type": "response.output_text.delta",
                                            "output_index": 0,
                                            "content_index": 0,
                                            "item_id": message_id,
                                            "delta": content,
                                        }
                                    )
                                calls = delta.get("tool_calls")
                                if not isinstance(calls, list):
                                    continue
                                for call in calls:
                                    if not isinstance(call, dict):
                                        continue
                                    index = int(call.get("index") or 0)
                                    accumulated = tool_deltas.setdefault(
                                        index,
                                        {"id": "", "name": "", "arguments": ""},
                                    )
                                    if isinstance(call.get("id"), str):
                                        accumulated["id"] = str(call["id"])
                                    function = call.get("function")
                                    if isinstance(function, dict):
                                        if isinstance(function.get("name"), str):
                                            accumulated["name"] += str(function["name"])
                                        if isinstance(function.get("arguments"), str):
                                            accumulated["arguments"] += str(function["arguments"])

                        output_index = 0
                        if message_started:
                            send_event(
                                {
                                    "type": "response.output_item.done",
                                    "output_index": output_index,
                                    "item": {
                                        "type": "message",
                                        "role": "assistant",
                                        "id": message_id,
                                        "content": [{"type": "output_text", "text": message_text}],
                                        "status": "completed",
                                    },
                                }
                            )
                            output_index += 1
                        for index in sorted(tool_deltas):
                            tool = tool_deltas[index]
                            safe_arguments, arguments_sanitized = _sanitize_tool_arguments(
                                tool["arguments"] or "{}"
                            )
                            if arguments_sanitized:
                                raw_arguments = tool["arguments"] or "{}"
                                owner._log.write(
                                    {
                                        "schema": "mars_llm_proxy_event_v1",
                                        "event": "tool_arguments_sanitized",
                                        "ts": _now(),
                                        "request_id": owner._request_id,
                                        "llm_call_id": llm_call_id,
                                        "tool_name": tool["name"],
                                        "raw_bytes": len(raw_arguments.encode("utf-8")),
                                        "raw_sha256": hashlib.sha256(
                                            raw_arguments.encode("utf-8")
                                        ).hexdigest(),
                                    }
                                )
                            namespace, original_name = reverse_tool_names.get(
                                tool["name"],
                                (None, tool["name"]),
                            )
                            item = {
                                "type": "function_call",
                                "id": f"item-mars-{uuid.uuid4().hex}",
                                "call_id": tool["id"] or f"call-mars-{uuid.uuid4().hex}",
                                "name": original_name,
                                "arguments": safe_arguments,
                                "status": "completed",
                            }
                            if namespace is not None:
                                item["namespace"] = namespace
                            send_event(
                                {
                                    "type": "response.output_item.done",
                                    "output_index": output_index,
                                    "item": item,
                                }
                            )
                            output_index += 1
                        if not message_started and not tool_deltas:
                            send_event(
                                {
                                    "type": "response.output_item.done",
                                    "output_index": 0,
                                    "item": {
                                        "type": "message",
                                        "role": "assistant",
                                        "id": message_id,
                                        "content": [{"type": "output_text", "text": ""}],
                                        "status": "completed",
                                    },
                                }
                            )
                        completed_usage = {
                            "input_tokens": int(usage["prompt_tokens"] or 0),
                            "input_tokens_details": {
                                "cached_tokens": int(usage["cached_input_tokens"] or 0)
                            },
                            "output_tokens": int(usage["completion_tokens"] or 0),
                            "output_tokens_details": {
                                "reasoning_tokens": int(usage["reasoning_tokens"] or 0)
                            },
                        }
                        completed_usage["total_tokens"] = (
                            completed_usage["input_tokens"] + completed_usage["output_tokens"]
                        )
                        last_payload = {
                            "type": "response.completed",
                            "response": {
                                "id": response_id,
                                "status": "completed",
                                "usage": completed_usage,
                            },
                        }
                        send_event(last_payload)
                    elif "text/event-stream" in content_type.lower():
                        while True:
                            line = response.readline()
                            if not line:
                                break
                            self.wfile.write(line)
                            self.wfile.flush()
                            stripped = line.strip()
                            if not stripped.startswith(b"data:"):
                                continue
                            raw_data = stripped[5:].strip()
                            if raw_data == b"[DONE]":
                                continue
                            payload = _safe_json(raw_data)
                            if payload is None:
                                continue
                            last_payload = payload
                            if first_token_at is None and _contains_token_delta(payload):
                                first_token_at = _now()
                                owner._log.write(
                                    {
                                        "schema": "mars_llm_proxy_event_v1",
                                        "event": "llm_first_token",
                                        "ts": first_token_at,
                                        "request_id": owner._request_id,
                                        "llm_call_id": llm_call_id,
                                        "ttft_s": first_token_at - started,
                                    }
                                )
                    else:
                        payload_bytes = response.read()
                        if payload_bytes:
                            self.wfile.write(payload_bytes)
                            self.wfile.flush()
                        last_payload = _safe_json(payload_bytes)
                        if payload_bytes:
                            first_token_at = _now()
                except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError, http.client.HTTPException) as exc:
                    owner._log.write(
                        {
                            "schema": "mars_llm_proxy_event_v1",
                            "event": "llm_proxy_error",
                            "ts": _now(),
                            "request_id": owner._request_id,
                            "llm_call_id": llm_call_id,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    if not self.wfile.closed:
                        try:
                            self.send_error(502, explain=str(exc))
                        except Exception:
                            pass
                finally:
                    ended = _now()
                    record = {
                        "schema": "mars_llm_proxy_event_v1",
                        "event": "llm_end",
                        "ts": ended,
                        "request_id": owner._request_id,
                        "llm_call_id": llm_call_id,
                        "protocol": protocol,
                        "http_status": status,
                        "response_id": _response_id(last_payload),
                        "duration_s": ended - started,
                        "ttft_s": (first_token_at - started) if first_token_at is not None else None,
                        **_usage(last_payload),
                    }
                    owner._log.write(record)
                    if conn is not None:
                        conn.close()

        return Handler
