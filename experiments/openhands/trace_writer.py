#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import threading
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from common import JSONLWriter, now_s


SCHEDULER_TOOL_NAMES = {
    "terminal",
    "file_editor",
    "task_tracker",
    "browser",
    "glob",
    "grep",
}
VERIFICATION_TOOL_NAMES = {
    "terminal",
    "file_editor",
    "browser",
    "glob",
    "grep",
}


def _preview_text(value: Any, limit: int = 400) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
    text = text.strip()
    if not text:
        return None
    if len(text) > limit:
        return text[:limit] + "...(truncated)"
    return text


def _extract_observation_preview(observation: Any) -> str | None:
    text = getattr(observation, "text", None)
    if isinstance(text, str) and text.strip():
        return _preview_text(text)
    try:
        dumped = observation.model_dump()
    except Exception:
        dumped = str(observation)
    return _preview_text(dumped)


def _extract_message_preview(message: Any) -> str | None:
    pieces: list[str] = []
    content = getattr(message, "content", None)
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                pieces.append(text)
    elif isinstance(content, str) and content.strip():
        pieces.append(content)
    if not pieces:
        return None
    return _preview_text("\n".join(pieces), limit=800)


def _coerce_ts(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return now_s()
    return now_s()


def _error_payload(error: Exception) -> dict[str, Any]:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(
                type(error),
                error,
                error.__traceback__,
            )
        ),
    }


def _synthesize_error_payload(
    *,
    status: str,
    conversation_errors: list[dict[str, Any]],
    agent_errors: list[dict[str, Any]],
    llm_calls: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if status in {"completed", "needs_confirmation"}:
        return None

    if conversation_errors:
        last = conversation_errors[-1]
        code = str(last.get("code") or "ConversationError")
        detail = str(last.get("detail") or code)
        return {
            "type": code,
            "message": detail,
            "traceback": None,
        }

    for item in reversed(agent_errors):
        message = _preview_text(item.get("error"), limit=2000)
        if message:
            return {
                "type": "AgentErrorEvent",
                "message": message,
                "traceback": None,
            }

    for item in reversed(llm_calls):
        message = _preview_text(item.get("error"), limit=2000)
        if message:
            return {
                "type": "LLMCallError",
                "message": message,
                "traceback": None,
            }

    for item in reversed(tool_calls):
        message = _preview_text(item.get("error"), limit=2000)
        if message:
            return {
                "type": "ToolCallError",
                "message": message,
                "traceback": None,
            }

    return None


def _extract_usage_fields(raw_response: Any) -> dict[str, int | None]:
    usage = None
    if isinstance(raw_response, dict):
        usage = raw_response.get("usage")
    else:
        usage = getattr(raw_response, "usage", None)
    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
        }

    def _get(name: str) -> int | None:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        try:
            return None if value is None else int(value)
        except Exception:
            return None

    reasoning_tokens = None
    completion_details = None
    if isinstance(usage, dict):
        completion_details = usage.get("completion_tokens_details")
    else:
        completion_details = getattr(usage, "completion_tokens_details", None)
    if completion_details is not None:
        if isinstance(completion_details, dict):
            value = completion_details.get("reasoning_tokens")
        else:
            value = getattr(completion_details, "reasoning_tokens", None)
        try:
            reasoning_tokens = None if value is None else int(value)
        except Exception:
            reasoning_tokens = None

    return {
        "prompt_tokens": _get("prompt_tokens") or _get("input_tokens"),
        "completion_tokens": _get("completion_tokens") or _get("output_tokens"),
        "reasoning_tokens": reasoning_tokens,
    }


@dataclass
class WorkerStartInfo:
    start_time_s: float
    queue_wait_s: float


class RequestTraceCollector:
    def __init__(
        self,
        *,
        events: JSONLWriter,
        request_id: str,
        dataset_request_id: str | None,
        benchmark: str | None,
        task_id: str | None,
        arrival_time_s: float,
        workspace_dir: Path,
    ) -> None:
        self.events = events
        self.request_id = request_id
        self.dataset_request_id = dataset_request_id
        self.benchmark = benchmark
        self.task_id = task_id
        self.arrival_time_s = float(arrival_time_s)
        self.workspace_dir = workspace_dir

        self._lock = threading.RLock()
        self._llm_counter = 0
        self._tool_counter = 0
        self._llm_calls: list[dict[str, Any]] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._llm_by_call_id: dict[str, dict[str, Any]] = {}
        self._llm_by_response_id: dict[str, dict[str, Any]] = {}
        self._tool_by_action_id: dict[str, dict[str, Any]] = {}
        self._tool_names_by_response_id: dict[str, list[str]] = {}
        self._pending_action_ids: deque[str] = deque()
        self._started_tool_stack: list[str] = []
        self._agent_messages: list[dict[str, Any]] = []
        self._agent_errors: list[dict[str, Any]] = []
        self._conversation_errors: list[dict[str, Any]] = []
        self._worker_info: WorkerStartInfo | None = None
        self._workspace_file_count = 0
        self._last_scheduler_tool_name: str | None = None
        self._verification_tool_count = 0

    def mark_worker_start(self) -> WorkerStartInfo:
        with self._lock:
            if self._worker_info is None:
                started = now_s()
                self._worker_info = WorkerStartInfo(
                    start_time_s=started,
                    queue_wait_s=float(max(0.0, started - self.arrival_time_s)),
                )
                self.events.write(
                    {
                        "ts": started,
                        "event": "request_start",
                        "request_id": self.request_id,
                        "dataset_request_id": self.dataset_request_id,
                        "benchmark": self.benchmark,
                        "task_id": self.task_id,
                        "queue_wait_s": self._worker_info.queue_wait_s,
                    }
                )
            return self._worker_info

    def set_workspace_materialized(self, file_count: int) -> None:
        with self._lock:
            self._workspace_file_count = int(file_count)
            self.events.write(
                {
                    "ts": now_s(),
                    "event": "workspace_ready",
                    "request_id": self.request_id,
                    "workspace_dir": str(self.workspace_dir),
                    "file_count": self._workspace_file_count,
                }
            )

    def on_token(self, _chunk: Any) -> None:
        return

    def on_event(self, event: Any) -> None:
        from openhands.sdk.event import ActionEvent, AgentErrorEvent, MessageEvent, ObservationEvent
        from openhands.sdk.event.conversation_error import ConversationErrorEvent

        with self._lock:
            if isinstance(event, ActionEvent):
                if event.action is None:
                    return
                self._tool_counter += 1
                tool_record = {
                    "tool_call_id": f"tool_{self._tool_counter:04d}",
                    "action_id": event.id,
                    "tool_name": event.tool_name,
                    "tool_kind": event.action.__class__.__name__,
                    "summary": getattr(event, "summary", None),
                    "llm_response_id": event.llm_response_id,
                    "native_tool_call_id": event.tool_call_id,
                    "enqueued_time_s": _coerce_ts(event.timestamp),
                    "start_time_s": None,
                    "end_time_s": None,
                    "wait_s": None,
                    "duration_s": None,
                    "status": "enqueued",
                    "observation_preview": None,
                    "error": None,
                }
                self._tool_calls.append(tool_record)
                self._tool_by_action_id[event.id] = tool_record
                if isinstance(event.llm_response_id, str) and event.llm_response_id:
                    self._tool_names_by_response_id.setdefault(event.llm_response_id, []).append(
                        str(event.tool_name)
                    )
                self._pending_action_ids.append(event.id)
                self.events.write(
                    {
                        "ts": _coerce_ts(event.timestamp),
                        "event": "tool_enqueued",
                        "request_id": self.request_id,
                        "tool_call_id": tool_record["tool_call_id"],
                        "action_id": event.id,
                        "tool_name": event.tool_name,
                        "tool_kind": tool_record["tool_kind"],
                        "summary": tool_record["summary"],
                        "llm_response_id": event.llm_response_id,
                    }
                )
                return

            if isinstance(event, ObservationEvent):
                tool_record = self._tool_by_action_id.get(event.action_id)
                if tool_record is not None:
                    tool_record["observation_preview"] = _extract_observation_preview(
                        event.observation
                    )
                return

            if isinstance(event, MessageEvent):
                if event.source == "agent":
                    preview = _extract_message_preview(event.llm_message)
                    self._agent_messages.append(
                        {
                            "ts": _coerce_ts(event.timestamp),
                            "preview": preview,
                            "llm_response_id": event.llm_response_id,
                        }
                    )
                return

            if isinstance(event, AgentErrorEvent):
                error_record = {
                    "ts": _coerce_ts(event.timestamp),
                    "tool_name": event.tool_name,
                    "tool_call_id": event.tool_call_id,
                    "error": event.error,
                }
                self._agent_errors.append(error_record)
                for tool_record in reversed(self._tool_calls):
                    if tool_record["native_tool_call_id"] == event.tool_call_id:
                        if tool_record["end_time_s"] is None:
                            tool_record["end_time_s"] = _coerce_ts(event.timestamp)
                            start_time_s = tool_record.get("start_time_s")
                            if isinstance(start_time_s, float):
                                tool_record["duration_s"] = float(
                                    max(0.0, tool_record["end_time_s"] - start_time_s)
                                )
                            tool_record["status"] = "agent_error"
                            tool_record["error"] = event.error
                        break
                return

            if isinstance(event, ConversationErrorEvent):
                self._conversation_errors.append(
                    {
                        "ts": _coerce_ts(event.timestamp),
                        "code": event.code,
                        "detail": event.detail,
                    }
                )
                return

    def start_llm_call(
        self,
        *,
        model: str,
        api_kind: str,
        message_count: int,
        tool_count: int,
    ) -> str:
        with self._lock:
            self._llm_counter += 1
            call_id = f"gpu_{self._llm_counter:04d}"
            started = now_s()
            record = {
                "llm_call_id": call_id,
                "model": model,
                "api_kind": api_kind,
                "message_count": int(message_count),
                "tool_count": int(tool_count),
                "submit_time_s": started,
                "first_token_time_s": None,
                "end_time_s": None,
                "ttft_s": None,
                "duration_s": None,
                "status": "submitted",
                "response_id": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "reasoning_tokens": None,
                "error": None,
            }
            self._llm_calls.append(record)
            self._llm_by_call_id[call_id] = record
            self.events.write(
                {
                    "ts": started,
                    "event": "gpu_submit",
                    "request_id": self.request_id,
                    "llm_call_id": call_id,
                    "model": model,
                    "api_kind": api_kind,
                    "message_count": int(message_count),
                    "tool_count": int(tool_count),
                }
            )
            return call_id

    def mark_llm_first_token(self, call_id: str) -> None:
        with self._lock:
            record = self._llm_by_call_id.get(call_id)
            if record is None or record["first_token_time_s"] is not None:
                return
            first_token_time_s = now_s()
            record["first_token_time_s"] = first_token_time_s
            record["ttft_s"] = float(max(0.0, first_token_time_s - record["submit_time_s"]))
            self.events.write(
                {
                    "ts": first_token_time_s,
                    "event": "gpu_first_token",
                    "request_id": self.request_id,
                    "llm_call_id": call_id,
                    "ttft_s": record["ttft_s"],
                }
            )

    def finish_llm_call(self, call_id: str, *, result: Any | None = None, error: Exception | None = None) -> None:
        with self._lock:
            record = self._llm_by_call_id.get(call_id)
            if record is None:
                return
            finished = now_s()
            record["end_time_s"] = finished
            record["duration_s"] = float(max(0.0, finished - record["submit_time_s"]))

            if result is not None:
                response_id = getattr(result, "id", None)
                record["response_id"] = response_id
                usage_fields = _extract_usage_fields(getattr(result, "raw_response", None))
                record.update(usage_fields)
                record["status"] = "ok"
                if isinstance(response_id, str) and response_id:
                    self._llm_by_response_id[response_id] = record
                self.events.write(
                    {
                        "ts": finished,
                        "event": "gpu_end",
                        "request_id": self.request_id,
                        "llm_call_id": call_id,
                        "response_id": response_id,
                        "duration_s": record["duration_s"],
                        "prompt_tokens": record["prompt_tokens"],
                        "completion_tokens": record["completion_tokens"],
                        "reasoning_tokens": record["reasoning_tokens"],
                    }
                )
                return

            record["status"] = "error"
            record["error"] = f"{type(error).__name__}: {error}" if error is not None else "unknown"
            self.events.write(
                {
                    "ts": finished,
                    "event": "gpu_error",
                    "request_id": self.request_id,
                    "llm_call_id": call_id,
                    "duration_s": record["duration_s"],
                    "error": record["error"],
                }
            )

    def on_tool_executor_start(self, tool_name: str, action: Any) -> str:
        with self._lock:
            action_id = None
            if self._pending_action_ids:
                action_id = self._pending_action_ids.popleft()
            if action_id is None or action_id not in self._tool_by_action_id:
                self._tool_counter += 1
                synthetic_action_id = f"synthetic_action_{self._tool_counter:04d}"
                tool_record = {
                    "tool_call_id": f"tool_{self._tool_counter:04d}",
                    "action_id": synthetic_action_id,
                    "tool_name": tool_name,
                    "tool_kind": action.__class__.__name__,
                    "summary": None,
                    "llm_response_id": None,
                    "native_tool_call_id": None,
                    "enqueued_time_s": now_s(),
                    "start_time_s": None,
                    "end_time_s": None,
                    "wait_s": None,
                    "duration_s": None,
                    "status": "enqueued",
                    "observation_preview": None,
                    "error": None,
                }
                self._tool_calls.append(tool_record)
                self._tool_by_action_id[synthetic_action_id] = tool_record
                action_id = synthetic_action_id

            tool_record = self._tool_by_action_id[action_id]
            started = now_s()
            tool_record["start_time_s"] = started
            tool_record["wait_s"] = float(max(0.0, started - tool_record["enqueued_time_s"]))
            tool_record["status"] = "running"
            self._started_tool_stack.append(action_id)
            self.events.write(
                {
                    "ts": started,
                    "event": "tool_start",
                    "request_id": self.request_id,
                    "tool_call_id": tool_record["tool_call_id"],
                    "action_id": action_id,
                    "tool_name": tool_record["tool_name"],
                    "wait_s": tool_record["wait_s"],
                }
            )
            return action_id

    def on_tool_executor_finish(
        self,
        action_id: str,
        *,
        observation: Any | None = None,
        error: Exception | None = None,
    ) -> None:
        with self._lock:
            tool_record = self._tool_by_action_id.get(action_id)
            if tool_record is None:
                return
            finished = now_s()
            tool_record["end_time_s"] = finished
            start_time_s = tool_record.get("start_time_s")
            if isinstance(start_time_s, float):
                tool_record["duration_s"] = float(max(0.0, finished - start_time_s))
            if observation is not None:
                tool_record["status"] = "ok"
                tool_record["observation_preview"] = _extract_observation_preview(observation)
            else:
                tool_record["status"] = "error"
                tool_record["error"] = f"{type(error).__name__}: {error}" if error is not None else "unknown"
            tool_name = str(tool_record.get("tool_name") or "")
            if tool_name in SCHEDULER_TOOL_NAMES:
                self._last_scheduler_tool_name = tool_name
            if tool_name in VERIFICATION_TOOL_NAMES:
                self._verification_tool_count += 1
            if self._started_tool_stack and self._started_tool_stack[-1] == action_id:
                self._started_tool_stack.pop()
            self.events.write(
                {
                    "ts": finished,
                    "event": "tool_end",
                    "request_id": self.request_id,
                    "tool_call_id": tool_record["tool_call_id"],
                    "action_id": action_id,
                    "tool_name": tool_record["tool_name"],
                    "duration_s": tool_record["duration_s"],
                    "status": tool_record["status"],
                    "error": tool_record["error"],
                }
            )

    def get_last_scheduler_tool_name(self) -> str | None:
        with self._lock:
            return self._last_scheduler_tool_name

    def get_verification_tool_count(self) -> int:
        with self._lock:
            return int(self._verification_tool_count)

    def has_verification_tool_use(self) -> bool:
        return self.get_verification_tool_count() > 0

    def finalize(
        self,
        *,
        status: str,
        conversation_status: str | None,
        error: Exception | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            end_time_s = now_s()
            worker_start = self._worker_info.start_time_s if self._worker_info else self.arrival_time_s
            queue_wait_s = self._worker_info.queue_wait_s if self._worker_info else 0.0

            gpu_total_s = sum(
                float(call["duration_s"] or 0.0)
                for call in self._llm_calls
                if call.get("duration_s") is not None
            )
            tool_total_s = sum(
                float(call["duration_s"] or 0.0)
                for call in self._tool_calls
                if call.get("duration_s") is not None
            )
            tool_wait_total_s = sum(
                float(call["wait_s"] or 0.0)
                for call in self._tool_calls
                if call.get("wait_s") is not None
            )

            llm_rounds = []
            llm_index_by_call_id: dict[str, int] = {}
            for idx, llm_call in enumerate(self._llm_calls):
                llm_rounds.append(
                    {
                        "round_index": idx,
                        "llm": dict(llm_call),
                        "tools": [],
                    }
                )
                llm_index_by_call_id[llm_call["llm_call_id"]] = idx

            orphan_tools: list[dict[str, Any]] = []
            for tool_call in self._tool_calls:
                response_id = tool_call.get("llm_response_id")
                llm_call = self._llm_by_response_id.get(response_id) if response_id else None
                if llm_call is None:
                    orphan_tools.append(dict(tool_call))
                    continue
                round_index = llm_index_by_call_id[llm_call["llm_call_id"]]
                llm_rounds[round_index]["tools"].append(dict(tool_call))

            trace = {
                "request_id": self.request_id,
                "dataset_request_id": self.dataset_request_id,
                "benchmark": self.benchmark,
                "task_id": self.task_id,
                "workspace_dir": str(self.workspace_dir),
                "workspace_file_count": self._workspace_file_count,
                "arrival_time_s": self.arrival_time_s,
                "start_time_s": worker_start,
                "end_time_s": end_time_s,
                "queue_wait_s": float(queue_wait_s),
                "e2e_latency_s": float(max(0.0, end_time_s - self.arrival_time_s)),
                "gpu_total_s": float(gpu_total_s),
                "tool_total_s": float(tool_total_s),
                "tool_wait_total_s": float(tool_wait_total_s),
                "wait_total_s": float(queue_wait_s + tool_wait_total_s),
                "verification_tool_count": int(self._verification_tool_count),
                "status": status,
                "conversation_status": conversation_status,
                "error": (
                    _error_payload(error)
                    if error is not None
                    else _synthesize_error_payload(
                        status=status,
                        conversation_errors=self._conversation_errors,
                        agent_errors=self._agent_errors,
                        llm_calls=self._llm_calls,
                        tool_calls=self._tool_calls,
                    )
                ),
                "llm_calls": [dict(call) for call in self._llm_calls],
                "tool_calls": [dict(call) for call in self._tool_calls],
                "rounds": llm_rounds,
                "orphan_tools": orphan_tools,
                "agent_messages": list(self._agent_messages),
                "agent_errors": list(self._agent_errors),
                "conversation_errors": list(self._conversation_errors),
            }

            self.events.write(
                {
                    "ts": end_time_s,
                    "event": "request_done",
                    "request_id": self.request_id,
                    "dataset_request_id": self.dataset_request_id,
                    "benchmark": self.benchmark,
                    "task_id": self.task_id,
                    "status": status,
                    "conversation_status": conversation_status,
                    "queue_wait_s": queue_wait_s,
                    "e2e_latency_s": trace["e2e_latency_s"],
                    "gpu_total_s": trace["gpu_total_s"],
                    "tool_total_s": trace["tool_total_s"],
                    "tool_wait_total_s": trace["tool_wait_total_s"],
                    "wait_total_s": trace["wait_total_s"],
                    "verification_tool_count": trace["verification_tool_count"],
                    "llm_call_count": len(self._llm_calls),
                    "tool_call_count": len(self._tool_calls),
                }
            )
            return trace
