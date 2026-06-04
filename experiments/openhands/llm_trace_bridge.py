#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import json
import threading
import time
from contextlib import contextmanager
from typing import Any


_PATCH_LOCK = threading.Lock()
_PATCHED = False
_REGISTRY: dict[int, Any] = {}


def _preview_text(value: Any, limit: int = 160) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[:limit] + "...(truncated)"
    return text


def _normalize_tool_call_arguments(raw_arguments: Any) -> tuple[str, bool]:
    if raw_arguments is None:
        return "{}", True
    if not isinstance(raw_arguments, str):
        try:
            parsed = raw_arguments if isinstance(raw_arguments, dict) else {"value": raw_arguments}
            return json.dumps(parsed, ensure_ascii=False), True
        except Exception:
            return "{}", True

    stripped = raw_arguments.strip()
    if not stripped:
        return "{}", True

    try:
        parsed = json.loads(stripped)
    except Exception:
        payload = {
            "_mars_invalid_tool_arguments": True,
            "raw_preview": _preview_text(stripped),
        }
        return json.dumps(payload, ensure_ascii=False), True

    if isinstance(parsed, dict):
        return stripped, False

    payload = {
        "_mars_invalid_tool_arguments": True,
        "raw_preview": _preview_text(stripped),
    }
    return json.dumps(payload, ensure_ascii=False), True


def _sanitize_history_messages(messages: list[Any], collector: Any | None = None) -> list[Any]:
    sanitized_messages: list[Any] = []

    for message_index, message in enumerate(messages):
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            sanitized_messages.append(message)
            continue

        cloned_message = copy.deepcopy(message)
        changed = False
        for tool_call_index, tool_call in enumerate(getattr(cloned_message, "tool_calls", []) or []):
            normalized, was_changed = _normalize_tool_call_arguments(
                getattr(tool_call, "arguments", None)
            )
            if was_changed:
                tool_call.arguments = normalized
                changed = True
                if collector is not None and hasattr(collector, "events"):
                    collector.events.write(
                        {
                            "ts": time.time(),
                            "event": "llm_history_tool_args_sanitized",
                            "request_id": getattr(collector, "request_id", None),
                            "message_index": int(message_index),
                            "tool_call_index": int(tool_call_index),
                            "tool_name": getattr(tool_call, "name", None),
                        }
                    )
        sanitized_messages.append(cloned_message if changed else message)

    return sanitized_messages


def _get_dynamic_extra_body(collector: Any) -> dict[str, Any]:
    return {
        # Use the MARS request id as the stable program/session id
        # This preserves per-session chaining semantics and avoids collisions when dataset ids are reused across runs.
        "job_id": str(getattr(collector, "request_id", "")),
    }


@contextmanager
def _temporary_extra_body(llm: Any, collector: Any):
    # Inject the per-request job_id only for the duration of one LLM call.
    original_extra_body = getattr(llm, "litellm_extra_body", None)
    merged_extra_body = dict(original_extra_body or {})
    merged_extra_body.update(_get_dynamic_extra_body(collector))
    llm.litellm_extra_body = merged_extra_body
    try:
        yield
    finally:
        llm.litellm_extra_body = original_extra_body


def register_llm_collector(llm: Any, collector: Any) -> None:
    with _PATCH_LOCK:
        _REGISTRY[id(llm)] = collector


def unregister_llm_collector(llm: Any) -> None:
    with _PATCH_LOCK:
        _REGISTRY.pop(id(llm), None)


def install_llm_trace_patch() -> None:
    global _PATCHED

    with _PATCH_LOCK:
        if _PATCHED:
            return

        import openhands.sdk.agent.agent as agent_module
        from openhands.sdk.llm.llm import LLM

        original_make_llm_completion = agent_module.make_llm_completion
        original_completion = LLM.completion
        original_responses = LLM.responses

        def traced_completion(self: Any, *args: Any, **kwargs: Any) -> Any:
            collector = _REGISTRY.get(id(self))
            if collector is None:
                return original_completion(self, *args, **kwargs)
            with _temporary_extra_body(self, collector):
                return original_completion(self, *args, **kwargs)

        def traced_responses(self: Any, *args: Any, **kwargs: Any) -> Any:
            collector = _REGISTRY.get(id(self))
            if collector is None:
                return original_responses(self, *args, **kwargs)
            with _temporary_extra_body(self, collector):
                return original_responses(self, *args, **kwargs)

        def traced_make_llm_completion(
            llm: Any,
            messages: list[Any],
            tools: list[Any] | None = None,
            on_token: Any | None = None,
        ) -> Any:
            collector = _REGISTRY.get(id(llm))
            if collector is None:
                return original_make_llm_completion(llm, messages, tools=tools, on_token=on_token)

            sanitized_messages = _sanitize_history_messages(messages, collector)

            call_id = collector.start_llm_call(
                model=str(getattr(llm, "model", "")),
                api_kind="responses" if bool(llm.uses_responses_api()) else "completion",
                message_count=len(sanitized_messages),
                tool_count=len(tools or []),
            )

            def wrapped_on_token(chunk: Any) -> None:
                collector.mark_llm_first_token(call_id)
                if on_token is not None:
                    on_token(chunk)

            callback = wrapped_on_token if (on_token is not None or bool(getattr(llm, "stream", False))) else None
            try:
                result = original_make_llm_completion(
                    llm,
                    sanitized_messages,
                    tools=tools,
                    on_token=callback,
                )
            except Exception as exc:
                collector.finish_llm_call(call_id, error=exc)
                raise

            collector.finish_llm_call(call_id, result=result)
            return result

        LLM.completion = traced_completion
        LLM.responses = traced_responses
        agent_module.make_llm_completion = traced_make_llm_completion
        _PATCHED = True
