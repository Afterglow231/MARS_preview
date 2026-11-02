#!/usr/bin/env python3
"""Request-scoped proxy between a native agent and MARS/vLLM.

The proxy normally forwards the native protocol unchanged apart from the
experiment controls that OpenHands also supplies at its LLM boundary
(``job_id`` and generation parameters).  Codex can additionally opt into a
schema-compatibility downgrade for Responses ``namespace`` tools: grouped
function tools are exposed to an older vLLM server as ordinary top-level
functions, then tool calls are restored to ``namespace`` + ``name`` before
they return to Codex.  Tool execution and subagent lifecycle remain owned by
the native frontend.  HTTP attempts are recorded separately from the
frontend's logical model rounds.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import socket
import ssl
import threading
import time
import urllib.parse
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable


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


def _json_object(raw: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _sse_payload(line: bytes) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith(b"data:"):
        return None
    data = stripped[5:].strip()
    if not data or data == b"[DONE]":
        return None
    return _json_object(data)


def _response_id(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    response = payload.get("response")
    if isinstance(response, dict) and response.get("id") is not None:
        return str(response["id"])
    if payload.get("id") is not None:
        return str(payload["id"])
    return None


def _has_delta(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    event_type = str(payload.get("type") or "")
    if event_type.endswith(".delta") and payload.get("delta") not in (None, "", [], {}):
        return True
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        if delta.get("content") not in (None, "") or delta.get("reasoning_content") not in (None, ""):
            return True
        if delta.get("tool_calls"):
            return True
    return False


_NAMESPACE_ALIAS_SEPARATOR = "__"


def _allocate_namespace_alias(
    namespace: str,
    name: str,
    *,
    occupied: set[str],
) -> str:
    """Return a deterministic function name that cannot shadow a real tool."""
    base = f"{namespace}{_NAMESPACE_ALIAS_SEPARATOR}{name}"
    candidate = base
    suffix = 2
    while candidate in occupied:
        candidate = f"{base}{_NAMESPACE_ALIAS_SEPARATOR}{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate


def _flatten_responses_namespace_tools(
    payload: dict[str, Any],
) -> tuple[dict[str, tuple[str, str]], int]:
    """Flatten Responses namespace function tools in-place for older vLLM.

    The returned mapping is scoped to one HTTP request and is used to restore
    model tool calls on the response path.  Historical function-call items and
    a namespaced function ``tool_choice`` are rewritten with the same aliases.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return {}, 0

    occupied = {
        str(tool["name"])
        for tool in tools
        if isinstance(tool, dict)
        and tool.get("type") != "namespace"
        and isinstance(tool.get("name"), str)
        and tool.get("name")
    }
    reverse_names: dict[str, tuple[str, str]] = {}
    aliases: dict[tuple[str, str], str] = {}
    flattened: list[dict[str, Any]] = []
    namespace_count = 0

    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "namespace":
            if isinstance(tool, dict):
                flattened.append(tool)
            continue
        namespace = tool.get("name")
        children = tool.get("tools")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("Responses namespace tool must have a non-empty name")
        if not isinstance(children, list):
            raise ValueError(f"Responses namespace {namespace!r} must contain a tools list")
        namespace_count += 1
        namespace_description = tool.get("description")
        for child in children:
            if not isinstance(child, dict):
                raise ValueError(f"Responses namespace {namespace!r} contains a non-object tool")
            child_type = child.get("type")
            if child_type != "function":
                raise ValueError(
                    f"Responses namespace compatibility only supports function children; "
                    f"{namespace!r} contains {child_type!r}"
                )
            child_name = child.get("name")
            if not isinstance(child_name, str) or not child_name:
                raise ValueError(
                    f"Responses namespace {namespace!r} contains a function without a name"
                )
            alias = _allocate_namespace_alias(
                namespace,
                child_name,
                occupied=occupied,
            )
            aliases[(namespace, child_name)] = alias
            reverse_names[alias] = (namespace, child_name)
            flat_child = dict(child)
            flat_child["name"] = alias
            # ``defer_loading`` is a namespace-child field in newer Responses
            # schemas and is not accepted by the older top-level FunctionTool.
            flat_child.pop("defer_loading", None)
            if isinstance(namespace_description, str) and namespace_description.strip():
                child_description = str(flat_child.get("description") or "").strip()
                flat_child["description"] = (
                    f"{namespace_description.strip()}\n\n{child_description}"
                    if child_description
                    else namespace_description.strip()
                )
            flattened.append(flat_child)

    if not reverse_names:
        return {}, 0
    payload["tools"] = flattened

    inputs = payload.get("input")
    if isinstance(inputs, list):
        for item in inputs:
            if not isinstance(item, dict) or item.get("type") not in {
                "function_call",
                "custom_tool_call",
            }:
                continue
            namespace = item.get("namespace")
            name = item.get("name")
            if not isinstance(namespace, str) or not namespace:
                continue
            if not isinstance(name, str) or not name:
                continue
            alias = aliases.get((namespace, name))
            if alias is None:
                alias = _allocate_namespace_alias(namespace, name, occupied=occupied)
                aliases[(namespace, name)] = alias
                reverse_names[alias] = (namespace, name)
            item["name"] = alias
            item.pop("namespace", None)

    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict):
        namespace = tool_choice.get("namespace")
        name = tool_choice.get("name")
        if (
            isinstance(namespace, str)
            and namespace
            and isinstance(name, str)
            and name
        ):
            alias = aliases.get((namespace, name))
            if alias is None:
                alias = _allocate_namespace_alias(namespace, name, occupied=occupied)
                aliases[(namespace, name)] = alias
                reverse_names[alias] = (namespace, name)
            tool_choice["name"] = alias
            tool_choice.pop("namespace", None)

    return reverse_names, namespace_count


def _restore_responses_namespace_calls(
    payload: dict[str, Any],
    reverse_names: dict[str, tuple[str, str]],
) -> int:
    """Restore flattened tool names in a Responses JSON object in-place."""
    if not reverse_names:
        return 0
    restored = 0

    def visit(value: Any) -> None:
        nonlocal restored
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        kind = str(value.get("type") or "")
        name = value.get("name")
        if (
            isinstance(name, str)
            and name in reverse_names
            and (
                kind in {"function_call", "custom_tool_call"}
                or "function_call" in kind
                or "custom_tool_call" in kind
            )
        ):
            namespace, original_name = reverse_names[name]
            value["name"] = original_name
            value["namespace"] = namespace
            restored += 1
        for child in value.values():
            visit(child)

    visit(payload)
    return restored


def _rewrite_sse_payload(
    line: bytes,
    reverse_names: dict[str, tuple[str, str]],
) -> tuple[bytes, dict[str, Any] | None, int]:
    """Restore one SSE data payload while retaining its line ending."""
    payload = _sse_payload(line)
    if payload is None:
        return line, None, 0
    restored = _restore_responses_namespace_calls(payload, reverse_names)
    if not restored:
        return line, payload, 0
    ending = b"\r\n" if line.endswith(b"\r\n") else b"\n"
    rewritten = b"data: " + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8") + ending
    return rewritten, payload, restored


class _JsonlLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def append(self, row: dict[str, Any]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()


class BackendTraceProxy(AbstractContextManager["BackendTraceProxy"]):
    """Forward native OpenAI traffic with optional schema compatibility."""

    def __init__(
        self,
        *,
        upstream_base_url: str,
        trace_path: Path,
        request_id: str,
        mars_job_id: str,
        upstream_timeout_s: float,
        max_output_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        responses_namespace_tool_compat: bool = False,
        admission_gate_path: Path | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        parsed = urllib.parse.urlsplit(upstream_base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"invalid upstream base URL: {upstream_base_url!r}")
        self._upstream = parsed
        self._base_path = parsed.path.rstrip("/")
        self._log = _JsonlLog(trace_path)
        self._request_id = request_id
        self._mars_job_id = mars_job_id
        self._timeout_s = float(upstream_timeout_s)
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._responses_namespace_tool_compat = bool(
            responses_namespace_tool_compat
        )
        self._admission_gate_path = admission_gate_path
        self._on_event = on_event
        self._counter = 0
        self._counter_lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._closing = threading.Event()

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("proxy is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}{self._base_path}"

    def _next_attempt_id(self) -> str:
        with self._counter_lock:
            self._counter += 1
            return f"backend_{self._counter:04d}"

    def _emit(self, row: dict[str, Any]) -> None:
        self._log.append(row)
        if self._on_event is not None:
            self._on_event(dict(row))

    def _wait_for_admission_gate(self) -> None:
        gate = self._admission_gate_path
        if gate is None:
            return
        while not gate.is_file():
            if self._closing.wait(timeout=0.05):
                raise ConnectionAbortedError("backend proxy closed while admission-paused")

    def start(self) -> "BackendTraceProxy":
        if self._server is not None:
            return self
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
                content_length = int(self.headers.get("Content-Length") or 0)
                original_body = self.rfile.read(content_length) if content_length else b""
                body = original_body
                client_path = self.path
                route_path = client_path.split("?", 1)[0]
                is_model_request = (
                    self.command == "POST"
                    and (route_path.endswith("/responses") or route_path.endswith("/chat/completions"))
                )
                if is_model_request:
                    owner._wait_for_admission_gate()
                request_json = _json_object(body) if body else None
                protocol = (
                    "responses"
                    if route_path.endswith("/responses")
                    else "chat_completions"
                    if route_path.endswith("/chat/completions")
                    else "other"
                )
                reverse_namespace_names: dict[str, tuple[str, str]] = {}
                flattened_namespace_count = 0
                if is_model_request and request_json is not None:
                    request_json["job_id"] = owner._mars_job_id
                    if owner._max_output_tokens is not None:
                        key = "max_output_tokens" if protocol == "responses" else "max_tokens"
                        request_json[key] = int(owner._max_output_tokens)
                    if owner._temperature is not None:
                        request_json["temperature"] = float(owner._temperature)
                    if owner._top_p is None:
                        request_json.pop("top_p", None)
                    else:
                        request_json["top_p"] = float(owner._top_p)
                    if (
                        owner._responses_namespace_tool_compat
                        and protocol == "responses"
                    ):
                        try:
                            (
                                reverse_namespace_names,
                                flattened_namespace_count,
                            ) = _flatten_responses_namespace_tools(request_json)
                        except ValueError as exc:
                            payload = json.dumps(
                                {
                                    "error": {
                                        "type": "invalid_request_error",
                                        "message": str(exc),
                                    }
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ).encode("utf-8")
                            self.send_response(400)
                            self.send_header("Content-Type", "application/json")
                            self.send_header("Content-Length", str(len(payload)))
                            self.send_header("Connection", "close")
                            self.end_headers()
                            self.wfile.write(payload)
                            self.wfile.flush()
                            self.close_connection = True
                            return
                    body = json.dumps(
                        request_json, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")

                attempt_id = owner._next_attempt_id() if is_model_request else None
                started_wall = time.time()
                started_mono = time.monotonic()
                if attempt_id is not None:
                    native_headers = {
                        key.lower(): value
                        for key, value in self.headers.items()
                        if key.lower()
                        in {
                            "x-session-affinity",
                            "x-session-id",
                            "x-parent-session-id",
                            "x-codex-turn-metadata",
                        }
                    }
                    owner._emit(
                        {
                            "schema": "mars_backend_attempt_v1",
                            "event": "backend_attempt_started",
                            "request_id": owner._request_id,
                            "mars_job_id": owner._mars_job_id,
                            "backend_attempt_id": attempt_id,
                            "protocol": protocol,
                            "ts": started_wall,
                            "monotonic_s": started_mono,
                            "model": request_json.get("model") if request_json else None,
                            "stream": bool(request_json.get("stream")) if request_json else False,
                            "frontend_correlation_headers": native_headers,
                            "input_sha256": hashlib.sha256(original_body).hexdigest(),
                            "forwarded_sha256": hashlib.sha256(body).hexdigest(),
                            "semantic_mutations": [
                                "job_id",
                                "max_output_tokens",
                                "temperature",
                                "top_p",
                            ]
                            + (
                                ["responses_namespace_tools_flattened"]
                                if reverse_namespace_names
                                else []
                            ),
                            "responses_namespace_tool_compat": bool(
                                owner._responses_namespace_tool_compat
                            ),
                            "flattened_namespace_count": flattened_namespace_count,
                            "flattened_namespace_function_count": len(
                                reverse_namespace_names
                            ),
                        }
                    )

                conn: http.client.HTTPConnection | http.client.HTTPSConnection | None = None
                status: int | None = None
                first_token_wall: float | None = None
                last_payload: dict[str, Any] | None = None
                restored_namespace_call_count = 0
                error: dict[str, str] | None = None
                try:
                    upstream_path = client_path
                    if owner._base_path and not route_path.startswith(owner._base_path + "/"):
                        upstream_path = owner._base_path + "/" + client_path.lstrip("/")
                    headers = {
                        key: value
                        for key, value in self.headers.items()
                        if key.lower() not in _HOP_BY_HOP | {"host", "content-length"}
                    }
                    headers["Host"] = owner._upstream.netloc
                    headers["X-MARS-Agent-Request-ID"] = owner._request_id
                    if attempt_id is not None:
                        headers["X-MARS-Backend-Attempt-ID"] = attempt_id
                    if owner._upstream.scheme == "https":
                        conn = http.client.HTTPSConnection(
                            owner._upstream.hostname,
                            owner._upstream.port or 443,
                            timeout=owner._timeout_s,
                            context=ssl.create_default_context(),
                        )
                    else:
                        conn = http.client.HTTPConnection(
                            owner._upstream.hostname,
                            owner._upstream.port or 80,
                            timeout=owner._timeout_s,
                        )
                    conn.request(self.command, upstream_path, body=body or None, headers=headers)
                    response = conn.getresponse()
                    status = int(response.status)
                    content_type = response.getheader("Content-Type") or ""
                    self.send_response(status, response.reason)
                    for key, value in response.getheaders():
                        if key.lower() in _HOP_BY_HOP | {"content-length", "connection"}:
                            continue
                        self.send_header(key, value)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    if "text/event-stream" in content_type.lower():
                        while True:
                            line = response.readline()
                            if not line:
                                break
                            (
                                forwarded_line,
                                payload,
                                restored_count,
                            ) = _rewrite_sse_payload(line, reverse_namespace_names)
                            restored_namespace_call_count += restored_count
                            self.wfile.write(forwarded_line)
                            self.wfile.flush()
                            if payload is None:
                                continue
                            last_payload = payload
                            if first_token_wall is None and _has_delta(payload):
                                first_token_wall = time.time()
                                if attempt_id is not None:
                                    owner._emit(
                                        {
                                            "schema": "mars_backend_attempt_v1",
                                            "event": "backend_attempt_first_token",
                                            "request_id": owner._request_id,
                                            "mars_job_id": owner._mars_job_id,
                                            "backend_attempt_id": attempt_id,
                                            "protocol": protocol,
                                            "ts": first_token_wall,
                                            "monotonic_s": time.monotonic(),
                                            "ttft_s": max(0.0, first_token_wall - started_wall),
                                        }
                                    )
                    else:
                        payload_bytes = response.read()
                        last_payload = _json_object(payload_bytes)
                        if last_payload is not None and reverse_namespace_names:
                            restored_namespace_call_count += (
                                _restore_responses_namespace_calls(
                                    last_payload,
                                    reverse_namespace_names,
                                )
                            )
                            payload_bytes = json.dumps(
                                last_payload,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        if payload_bytes:
                            self.wfile.write(payload_bytes)
                            self.wfile.flush()
                        if payload_bytes:
                            first_token_wall = time.time()
                except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError,
                        http.client.HTTPException) as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)}
                    if not self.headers_sent:
                        try:
                            payload = json.dumps({"error": error}).encode("utf-8")
                            self.send_response(502)
                            self.send_header("Content-Type", "application/json")
                            self.send_header("Content-Length", str(len(payload)))
                            self.send_header("Connection", "close")
                            self.end_headers()
                            self.wfile.write(payload)
                        except Exception:
                            pass
                finally:
                    if conn is not None:
                        conn.close()
                    if attempt_id is not None:
                        ended_wall = time.time()
                        owner._emit(
                            {
                                "schema": "mars_backend_attempt_v1",
                                "event": "backend_attempt_ended",
                                "request_id": owner._request_id,
                                "mars_job_id": owner._mars_job_id,
                                "backend_attempt_id": attempt_id,
                                "protocol": protocol,
                                "ts": ended_wall,
                                "monotonic_s": time.monotonic(),
                                "duration_s": max(0.0, time.monotonic() - started_mono),
                                "ttft_s": (
                                    max(0.0, first_token_wall - started_wall)
                                    if first_token_wall is not None
                                    else None
                                ),
                                "http_status": status,
                                "response_id": _response_id(last_payload),
                                "restored_namespace_call_count": (
                                    restored_namespace_call_count
                                ),
                                "error": error,
                            }
                        )

            @property
            def headers_sent(self) -> bool:
                return getattr(self, "_headers_buffer", None) == []

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._closing.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "BackendTraceProxy":
        return self.start()

    def __exit__(self, *_args: Any) -> None:
        self.close()
