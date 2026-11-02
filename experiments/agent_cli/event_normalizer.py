#!/usr/bin/env python3
"""Normalize OpenCode/Codex CLI events and construct a measured dynamic DAG."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _preview(value: Any, limit: int = 500) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = text.strip()
    if not text:
        return None
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def _captured(row: dict[str, Any], fallback: float) -> float:
    value = row.get("captured_at_s")
    if isinstance(value, (int, float)):
        return float(value)
    raw = row.get("raw")
    if isinstance(raw, dict):
        millis = raw.get("timestamp")
        if isinstance(millis, (int, float)):
            return float(millis) / 1000.0
    return fallback


def _raw(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw")
    return raw if isinstance(raw, dict) else row


def _opencode_exit_code(state: dict[str, Any]) -> int | None:
    metadata = state.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("exit")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _opencode_status(state: dict[str, Any]) -> str:
    """Map OpenCode's lifecycle status to the tool execution outcome.

    OpenCode marks a finished shell invocation as ``completed`` even when its
    process exits non-zero.  OpenHands records that observation as a failed
    tool call, so the exit code must take precedence over the lifecycle state.
    """

    status = str(state.get("status") or "unknown").lower()
    exit_code = _opencode_exit_code(state)
    if exit_code is not None and exit_code != 0:
        return "failed"
    return status


class OpenCodeTimingIntegrityError(RuntimeError):
    """Raised when configured-shell timings cannot be paired losslessly."""

    def __init__(self, details: dict[str, Any]) -> None:
        self.details = details
        super().__init__(
            "OpenCode configured-shell timing integrity failure: "
            + json.dumps(details, sort_keys=True, separators=(",", ":"))
        )


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _opencode_timing_key(
    state: dict[str, Any], workspace_dir: Path | None
) -> tuple[str, str] | None:
    params = state.get("input")
    if not isinstance(params, dict):
        return None
    command = params.get("command")
    if not isinstance(command, str) or workspace_dir is None:
        return None

    raw_workdir = params.get("workdir")
    if isinstance(raw_workdir, str) and raw_workdir:
        workdir = Path(raw_workdir)
        if not workdir.is_absolute():
            workdir = workspace_dir / workdir
    else:
        workdir = workspace_dir
    # OpenCode's resolvePath uses Node path.resolve (lexical normalization), so
    # do not resolve symlinks here.  The request workspace itself is an absolute
    # bind-mounted path in formal runs.
    resolved_cwd = os.path.abspath(os.path.normpath(str(workdir)))
    return _text_sha256(command), _text_sha256(resolved_cwd)


def _opencode_shell_spans(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Pair configured-shell rows while retaining integrity diagnostics."""

    spans: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    issues: list[dict[str, Any]] = []
    counts = {"shell_start_count": 0, "shell_end_count": 0}
    for index, row in enumerate(rows):
        event = _raw(row)
        kind = str(event.get("event") or "")
        if kind not in {"shell_start", "shell_end"}:
            continue
        counts[f"{kind}_count"] += 1
        shell_call_id = str(event.get("shell_call_id") or "")
        if not shell_call_id:
            issues.append({"kind": f"{kind}_missing_call_id", "row_index": index})
            continue
        span = spans.get(shell_call_id)
        if kind == "shell_start":
            if span is not None and "start_time_s" in span:
                issues.append(
                    {"kind": "duplicate_shell_start", "shell_call_id": shell_call_id}
                )
                continue
            if span is None:
                span = {"shell_call_id": shell_call_id, "row_index": index}
                spans[shell_call_id] = span
                order.append(shell_call_id)
            span.update(
                {
                    "start_time_s": event.get("ts"),
                    "command_sha256": event.get("command_sha256"),
                    "cwd_sha256": event.get("cwd_sha256"),
                }
            )
        else:
            if span is not None and "end_time_s" in span:
                issues.append(
                    {"kind": "duplicate_shell_end", "shell_call_id": shell_call_id}
                )
                continue
            if span is None:
                span = {"shell_call_id": shell_call_id, "row_index": index}
                spans[shell_call_id] = span
                order.append(shell_call_id)
            span.update(
                {
                    "end_time_s": event.get("ts"),
                    "duration_s": event.get("duration_s"),
                    "exit_code": event.get("exit_code"),
                    "status": event.get("status"),
                    "command_sha256": event.get("command_sha256")
                    or span.get("command_sha256"),
                    "cwd_sha256": event.get("cwd_sha256") or span.get("cwd_sha256"),
                }
            )

    valid: list[dict[str, Any]] = []
    for shell_call_id in order:
        span = spans[shell_call_id]
        missing: list[str] = []
        if not isinstance(span.get("start_time_s"), (int, float)):
            missing.append("start_time_s")
        if not isinstance(span.get("end_time_s"), (int, float)):
            missing.append("end_time_s")
        for field in ("command_sha256", "cwd_sha256"):
            value = span.get(field)
            if not isinstance(value, str) or len(value) != 64:
                missing.append(field)
        if missing:
            issues.append(
                {
                    "kind": "incomplete_shell_span",
                    "shell_call_id": shell_call_id,
                    "missing_or_invalid": missing,
                }
            )
            continue
        valid.append(span)
    return valid, issues, counts


def normalize_opencode(
    rows: list[dict[str, Any]],
    *,
    shell_rows: list[dict[str, Any]] | None = None,
    workspace_dir: Path | None = None,
    require_shell_timing: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tools_by_id: dict[str, dict[str, Any]] = {}
    tool_order: list[str] = []
    messages: list[dict[str, Any]] = []
    for index, wrapper in enumerate(rows):
        event = _raw(wrapper)
        observed = _captured(wrapper, float(index))
        kind = str(event.get("type") or "")
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        if kind == "tool_use":
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            timing = state.get("time") if isinstance(state.get("time"), dict) else {}
            start = timing.get("start")
            end = timing.get("end")
            tool_id = str(
                part.get("callID")
                or part.get("id")
                or f"tool_{len(tool_order) + 1:04d}"
            )
            start_s = float(start) / 1000.0 if isinstance(start, (int, float)) else observed
            end_s = float(end) / 1000.0 if isinstance(end, (int, float)) else None
            tool_name = str(part.get("tool") or "unknown")
            timing_key = (
                _opencode_timing_key(state, workspace_dir)
                if tool_name.lower() == "bash"
                else None
            )
            record = tools_by_id.get(tool_id)
            if record is None:
                record = {
                    "tool_call_id": tool_id,
                    "tool_name": tool_name,
                    "enqueued_time_s": start_s,
                    "start_time_s": start_s,
                    "end_time_s": end_s,
                    "wait_s": 0.0,
                    "duration_s": (
                        max(0.0, end_s - start_s) if end_s is not None else None
                    ),
                    "status": _opencode_status(state),
                    "exit_code": _opencode_exit_code(state),
                    "input_preview": _preview(state.get("input")),
                    "output_preview": _preview(state.get("output") or state.get("error")),
                    "raw_part_id": part.get("id"),
                    "_opencode_timing_key": timing_key,
                }
                tools_by_id[tool_id] = record
                tool_order.append(tool_id)
            else:
                record["start_time_s"] = min(float(record["start_time_s"]), start_s)
                record["enqueued_time_s"] = record["start_time_s"]
                if end_s is not None:
                    record["end_time_s"] = max(
                        float(record.get("end_time_s") or end_s), end_s
                    )
                record["status"] = _opencode_status(state)
                exit_code = _opencode_exit_code(state)
                if exit_code is not None:
                    record["exit_code"] = exit_code
                record["input_preview"] = _preview(state.get("input")) or record.get("input_preview")
                record["output_preview"] = _preview(
                    state.get("output") or state.get("error")
                ) or record.get("output_preview")
                if timing_key is not None:
                    record["_opencode_timing_key"] = timing_key
                if isinstance(record.get("end_time_s"), (int, float)):
                    record["duration_s"] = max(
                        0.0,
                        float(record["end_time_s"]) - float(record["start_time_s"]),
                    )
        elif kind == "text":
            messages.append(
                {
                    "message_id": str(part.get("id") or f"message_{len(messages) + 1:04d}"),
                    "time_s": observed,
                    "text_preview": _preview(part.get("text"), limit=1000),
                }
            )

    tools = [tools_by_id[tool_id] for tool_id in tool_order]
    native_bash = [tool for tool in tools if str(tool.get("tool_name") or "").lower() == "bash"]
    if shell_rows is None:
        for tool in native_bash:
            tool["timing_source"] = "opencode_native_part"
            tool["timing_integrity"] = "not_instrumented"
    else:
        spans, span_issues, counts = _opencode_shell_spans(shell_rows)
        span_queues: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
        span_occurrences: Counter[tuple[str, str]] = Counter()
        for span in sorted(
            spans,
            key=lambda item: (float(item["start_time_s"]), int(item.get("row_index") or 0)),
        ):
            key = (str(span["command_sha256"]), str(span["cwd_sha256"]))
            span["occurrence"] = span_occurrences[key]
            span_occurrences[key] += 1
            span_queues[key].append(span)

        native_occurrences: Counter[tuple[str, str]] = Counter()
        missing_native: list[dict[str, Any]] = []
        matched = 0
        for tool in native_bash:
            key = tool.get("_opencode_timing_key")
            occurrence: int | None = None
            span: dict[str, Any] | None = None
            if isinstance(key, tuple) and len(key) == 2:
                occurrence = native_occurrences[key]
                native_occurrences[key] += 1
                if span_queues[key]:
                    span = span_queues[key].popleft()
            if span is None:
                missing_native.append(
                    {
                        "tool_call_id": tool.get("tool_call_id"),
                        "command_sha256": key[0] if isinstance(key, tuple) else None,
                        "cwd_sha256": key[1] if isinstance(key, tuple) else None,
                        "occurrence": occurrence,
                    }
                )
                tool["timing_integrity"] = "mismatch"
                tool["timing_source"] = "opencode_native_part"
                continue

            start_s = float(span["start_time_s"])
            end_s = float(span["end_time_s"])
            exit_code = span.get("exit_code")
            status = str(span.get("status") or tool.get("status") or "completed")
            if isinstance(exit_code, (int, float)) and not isinstance(exit_code, bool):
                exit_code = int(exit_code)
                if exit_code != 0:
                    status = "failed"
            tool.update(
                {
                    "enqueued_time_s": start_s,
                    "start_time_s": start_s,
                    "end_time_s": end_s,
                    "wait_s": 0.0,
                    "duration_s": max(0.0, end_s - start_s),
                    "status": status,
                    "exit_code": exit_code,
                    "timing_source": "opencode_configured_shell_wrapper",
                    "timing_integrity": "matched",
                    "timing_match_occurrence": occurrence,
                    "shell_call_id": span.get("shell_call_id"),
                }
            )
            matched += 1

        unmatched_spans: list[dict[str, Any]] = []
        for queue in span_queues.values():
            for span in queue:
                unmatched_spans.append(
                    {
                        "shell_call_id": span.get("shell_call_id"),
                        "command_sha256": span.get("command_sha256"),
                        "cwd_sha256": span.get("cwd_sha256"),
                        "occurrence": span.get("occurrence"),
                    }
                )
        integrity = {
            "status": (
                "passed"
                if not span_issues and not missing_native and not unmatched_spans
                else "failed"
            ),
            "native_bash_count": len(native_bash),
            "matched_count": matched,
            "complete_shell_span_count": len(spans),
            **counts,
            "missing_native_matches": missing_native,
            "unmatched_shell_spans": unmatched_spans,
            "shell_span_issues": span_issues,
        }
        if integrity["status"] != "passed":
            for tool in native_bash:
                tool["timing_integrity"] = "mismatch"
            if require_shell_timing:
                raise OpenCodeTimingIntegrityError(integrity)

    for tool in tools:
        tool.pop("_opencode_timing_key", None)
    return tools, messages


_CODEX_INTERVAL_TOOL_TYPES = {
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "collab_tool_call",
    "web_search",
}


def _codex_tool_name(item_type: str, item: dict[str, Any]) -> str:
    if item_type == "mcp_tool_call":
        return f"mcp:{item.get('server')}:{item.get('tool')}"
    if item_type == "collab_tool_call":
        return f"collab:{item.get('tool')}"
    return item_type


def _codex_shell_spans(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair Bash-wrapper start/end rows in real execution-start order."""
    spans: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        event = _raw(row)
        shell_call_id = str(event.get("shell_call_id") or "")
        if not shell_call_id:
            continue
        kind = str(event.get("event") or "")
        if kind == "shell_start":
            if shell_call_id not in spans:
                spans[shell_call_id] = {
                    "shell_call_id": shell_call_id,
                    "start_time_s": event.get("ts"),
                    "argv_count": event.get("argv_count"),
                }
                order.append(shell_call_id)
        elif kind == "shell_end":
            span = spans.setdefault(shell_call_id, {"shell_call_id": shell_call_id})
            span.update(
                {
                    "end_time_s": event.get("ts"),
                    "duration_s": event.get("duration_s"),
                    "exit_code": event.get("exit_code"),
                    "status": event.get("status"),
                }
            )
    return [spans[shell_call_id] for shell_call_id in order]


def normalize_codex(
    rows: list[dict[str, Any]],
    *,
    shell_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    started: dict[str, float] = {}
    shell_spans = _codex_shell_spans(shell_rows or [])
    shell_span_by_item: dict[str, dict[str, Any]] = {}
    next_shell_span = 0
    todo_updates: Counter[str] = Counter()
    tools: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for index, wrapper in enumerate(rows):
        event = _raw(wrapper)
        observed = _captured(wrapper, float(index))
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or f"item_{index:04d}")
        item_type = str(item.get("type") or "")
        event_type = str(event.get("type") or "")
        if item_type == "todo_list" and event_type in {"item.started", "item.updated"}:
            # Codex keeps one todo_list item open for most of the thread, so the
            # item.started -> item.completed interval is a plan lifetime, not CPU
            # tool execution.  Treat each actual plan update as an instantaneous
            # task-tracker event instead of charging the whole conversation to
            # tool_total_s.
            update_index = todo_updates[item_id]
            todo_updates[item_id] += 1
            tools.append(
                {
                    "tool_call_id": f"{item_id}:update:{update_index:04d}",
                    "tool_name": "task_tracker",
                    "enqueued_time_s": observed,
                    "start_time_s": observed,
                    "end_time_s": observed,
                    "wait_s": 0.0,
                    "duration_s": 0.0,
                    "status": "completed",
                    "input_preview": _preview(item.get("items")),
                    "output_preview": None,
                    "raw_part_id": item_id,
                }
            )
            continue
        if event_type == "item.started":
            started[item_id] = observed
            if item_type == "command_execution" and next_shell_span < len(shell_spans):
                shell_span_by_item[item_id] = shell_spans[next_shell_span]
                next_shell_span += 1
            continue
        if event_type != "item.completed":
            continue
        if item_type in _CODEX_INTERVAL_TOOL_TYPES:
            name = _codex_tool_name(item_type, item)
            status = str(item.get("status") or "completed")
            exit_code = item.get("exit_code")
            start_s = started.get(item_id, observed)
            end_s = observed
            timing_source = "codex_json_capture"
            shell_span = shell_span_by_item.get(item_id)
            if item_type == "command_execution" and shell_span is None:
                if next_shell_span < len(shell_spans):
                    shell_span = shell_spans[next_shell_span]
                    next_shell_span += 1
            if shell_span is not None:
                shell_start = shell_span.get("start_time_s")
                shell_end = shell_span.get("end_time_s")
                if shell_span.get("exit_code") is not None:
                    exit_code = shell_span.get("exit_code")
                if isinstance(shell_start, (int, float)) and isinstance(
                    shell_end, (int, float)
                ):
                    start_s = float(shell_start)
                    end_s = float(shell_end)
                    timing_source = "codex_bash_wrapper"
            if isinstance(exit_code, (int, float)) and not isinstance(exit_code, bool):
                if int(exit_code) != 0:
                    status = "failed"
            tools.append(
                {
                    "tool_call_id": item_id,
                    "tool_name": name,
                    "enqueued_time_s": start_s,
                    "start_time_s": start_s,
                    "end_time_s": end_s,
                    "wait_s": 0.0,
                    "duration_s": max(0.0, end_s - start_s),
                    "status": status,
                    "exit_code": exit_code,
                    "timing_source": timing_source,
                    "shell_call_id": (
                        shell_span.get("shell_call_id") if shell_span is not None else None
                    ),
                    "input_preview": _preview(item.get("command") or item.get("arguments") or item.get("changes")),
                    "output_preview": _preview(
                        item.get("aggregated_output") or item.get("result") or item.get("error")
                    ),
                    "raw_part_id": item_id,
                }
            )
        elif item_type == "agent_message":
            messages.append(
                {
                    "message_id": item_id,
                    "time_s": observed,
                    "text_preview": _preview(item.get("text"), limit=1000),
                }
            )
    return tools, messages


class LiveToolEventNormalizer:
    """Convert streamed CLI JSON rows into OpenHands-style live tool events."""

    _TERMINAL_STATES = {"completed", "complete", "done", "failed", "error", "cancelled"}

    def __init__(
        self,
        *,
        agent: str,
        request_id: str,
        codex_shell_timing: bool = False,
        opencode_shell_timing: bool = False,
    ) -> None:
        self.agent = agent
        self.request_id = request_id
        self.codex_shell_timing = codex_shell_timing
        self.opencode_shell_timing = opencode_shell_timing
        self._active: dict[str, tuple[str, float]] = {}
        self._finished: set[str] = set()
        self._todo_updates: Counter[str] = Counter()

    def _start(self, tool_id: str, tool_name: str, ts: float) -> dict[str, Any] | None:
        if tool_id in self._active or tool_id in self._finished:
            return None
        self._active[tool_id] = (tool_name, ts)
        return {
            "ts": ts,
            "event": "tool_start",
            "request_id": self.request_id,
            "tool_call_id": tool_id,
            "tool_name": tool_name,
            "wait_s": 0.0,
        }

    def _end(self, tool_id: str, tool_name: str, ts: float, status: str) -> dict[str, Any] | None:
        if tool_id in self._finished:
            return None
        _name, start = self._active.pop(tool_id, (tool_name, ts))
        self._finished.add(tool_id)
        return {
            "ts": ts,
            "event": "tool_end",
            "request_id": self.request_id,
            "tool_call_id": tool_id,
            "tool_name": tool_name,
            "duration_s": max(0.0, ts - start),
            "status": status,
        }

    def feed(self, raw: dict[str, Any], observed: float) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self.agent == "codex":
            item = raw.get("item")
            if not isinstance(item, dict):
                return events
            item_id = str(item.get("id") or "")
            item_type = str(item.get("type") or "")
            event_type = str(raw.get("type") or "")
            if not item_id:
                return events
            if item_type == "todo_list" and event_type in {"item.started", "item.updated"}:
                update_index = self._todo_updates[item_id]
                self._todo_updates[item_id] += 1
                update_id = f"{item_id}:update:{update_index:04d}"
                start = self._start(update_id, "task_tracker", observed)
                end = self._end(update_id, "task_tracker", observed, "completed")
                return [item for item in (start, end) if item is not None]
            if item_type not in _CODEX_INTERVAL_TOOL_TYPES:
                return events
            if item_type == "command_execution" and self.codex_shell_timing:
                # The Bash interposer writes true wall-clock live events.  The
                # pinned Codex JSON stream flushes these lifecycle rows only
                # after the command has already finished.
                return events
            tool_name = _codex_tool_name(item_type, item)
            if event_type == "item.started":
                start = self._start(item_id, tool_name, observed)
                return [start] if start is not None else []
            if event_type == "item.completed":
                start = self._start(item_id, tool_name, observed)
                end = self._end(
                    item_id,
                    tool_name,
                    observed,
                    str(item.get("status") or "completed"),
                )
                return [item for item in (start, end) if item is not None]
            return events

        if self.agent == "opencode":
            if str(raw.get("type") or "") != "tool_use":
                return events
            part = raw.get("part")
            if not isinstance(part, dict):
                return events
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            timing = state.get("time") if isinstance(state.get("time"), dict) else {}
            tool_id = str(part.get("callID") or part.get("id") or "")
            if not tool_id:
                return events
            tool_name = str(part.get("tool") or "unknown")
            if tool_name.lower() == "bash" and self.opencode_shell_timing:
                # The pinned OpenCode JSON formatter emits bash lifecycle state
                # only after the command has completed, and its native start is
                # overwritten by later metadata updates.  The configured-shell
                # wrapper emits the true live start/end pair instead.
                return events
            raw_start = timing.get("start")
            raw_end = timing.get("end")
            start_s = float(raw_start) / 1000.0 if isinstance(raw_start, (int, float)) else observed
            end_s = float(raw_end) / 1000.0 if isinstance(raw_end, (int, float)) else observed
            start = self._start(tool_id, tool_name, start_s)
            if start is not None:
                events.append(start)
            status = _opencode_status(state)
            if isinstance(raw_end, (int, float)) or status in self._TERMINAL_STATES:
                end = self._end(tool_id, tool_name, end_s, status)
                if end is not None:
                    events.append(end)
            return events

        return events

    def finish(self, observed: float) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for tool_id, (tool_name, _start) in list(self._active.items()):
            end = self._end(tool_id, tool_name, observed, "aborted")
            if end is not None:
                events.append(end)
        return events


def normalize_llm_calls(proxy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    for event in proxy_rows:
        call_id = event.get("llm_call_id")
        if not isinstance(call_id, str):
            continue
        record = calls.setdefault(call_id, {"llm_call_id": call_id})
        kind = event.get("event")
        if kind == "llm_start":
            record.update(
                {
                    "start_time_s": event.get("ts"),
                    "protocol": event.get("protocol"),
                    "upstream_protocol": event.get("upstream_protocol"),
                    "responses_via_chat": event.get("responses_via_chat"),
                    "history_tool_arguments_sanitized": event.get(
                        "history_tool_arguments_sanitized", 0
                    ),
                    "model": event.get("model"),
                    "message_count": event.get("message_count"),
                    "input_item_count": event.get("input_item_count"),
                    "tool_count": event.get("tool_count"),
                    "previous_response_id": event.get("previous_response_id"),
                    "mars_metadata": event.get("mars_metadata"),
                    "original_request_body_bytes": event.get("original_body_bytes"),
                    "original_request_body_sha256": event.get("original_body_sha256"),
                    "forwarded_request_body_bytes": event.get("forwarded_body_bytes"),
                    "forwarded_request_body_sha256": event.get("forwarded_body_sha256"),
                }
            )
        elif kind == "llm_first_token":
            record["first_token_time_s"] = event.get("ts")
            record["ttft_s"] = event.get("ttft_s")
        elif kind == "tool_arguments_sanitized":
            record["output_tool_arguments_sanitized"] = int(
                record.get("output_tool_arguments_sanitized") or 0
            ) + 1
        elif kind == "llm_proxy_error":
            record["error"] = {
                "type": event.get("error_type"),
                "message": event.get("message"),
            }
        elif kind == "llm_end":
            record.update(
                {
                    "end_time_s": event.get("ts"),
                    "duration_s": event.get("duration_s"),
                    "ttft_s": event.get("ttft_s", record.get("ttft_s")),
                    "http_status": event.get("http_status"),
                    "response_id": event.get("response_id"),
                    "prompt_tokens": event.get("prompt_tokens"),
                    "completion_tokens": event.get("completion_tokens"),
                    "cached_input_tokens": event.get("cached_input_tokens"),
                    "reasoning_tokens": event.get("reasoning_tokens"),
                }
            )
    measured = [
        item
        for item in calls.values()
        if item.get("protocol") in {"chat_completions", "responses"}
    ]
    return sorted(
        measured,
        key=lambda item: (float(item.get("start_time_s") or math.inf), str(item["llm_call_id"])),
    )


def build_dynamic_graph(
    llm_calls: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a time-ordered DAG from measured LLM and tool intervals."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    for index, call in enumerate(llm_calls):
        node_id = str(call.get("llm_call_id") or f"llm_{index + 1:04d}")
        nodes.append(
            {
                "node_id": node_id,
                "kind": "llm",
                "round_index": index,
                "start_time_s": call.get("start_time_s"),
                "end_time_s": call.get("end_time_s"),
                "protocol": call.get("protocol"),
            }
        )

    assignments: dict[str, list[str]] = {str(call.get("llm_call_id")): [] for call in llm_calls}
    for index, tool in enumerate(sorted(tool_calls, key=lambda item: float(item.get("start_time_s") or 0.0))):
        tool_id = str(tool.get("tool_call_id") or f"tool_{index + 1:04d}")
        start = float(tool.get("start_time_s") or 0.0)
        end = float(tool.get("end_time_s") or start)
        before = [
            call
            for call in llm_calls
            if isinstance(call.get("end_time_s"), (int, float)) and float(call["end_time_s"]) <= start + 1e-6
        ]
        emitter = before[-1] if before else (llm_calls[min(index, len(llm_calls) - 1)] if llm_calls else None)
        after = [
            call
            for call in llm_calls
            if isinstance(call.get("start_time_s"), (int, float)) and float(call["start_time_s"]) >= end - 1e-6
        ]
        consumer = after[0] if after else None
        nodes.append(
            {
                "node_id": tool_id,
                "kind": "tool",
                "tool_name": tool.get("tool_name"),
                "start_time_s": tool.get("start_time_s"),
                "end_time_s": tool.get("end_time_s"),
                "status": tool.get("status"),
            }
        )
        if emitter is not None:
            emitter_id = str(emitter.get("llm_call_id"))
            edges.append({"source": emitter_id, "target": tool_id, "kind": "emits"})
            assignments.setdefault(emitter_id, []).append(tool_id)
        if consumer is not None:
            edges.append(
                {
                    "source": tool_id,
                    "target": str(consumer.get("llm_call_id")),
                    "kind": "observed_by",
                }
            )

    for left, right in zip(llm_calls, llm_calls[1:]):
        left_id = str(left.get("llm_call_id"))
        right_id = str(right.get("llm_call_id"))
        if not assignments.get(left_id):
            edges.append({"source": left_id, "target": right_id, "kind": "continuation"})

    signature_tokens: list[str] = []
    for call in llm_calls:
        call_id = str(call.get("llm_call_id"))
        names = sorted(
            str(node.get("tool_name") or "unknown")
            for node in nodes
            if node.get("node_id") in set(assignments.get(call_id, []))
        )
        signature_tokens.append("llm[" + ",".join(names) + "]")
    topology = "->".join(signature_tokens)
    return {
        "schema": "mars_dynamic_agent_graph_v1",
        "nodes": nodes,
        "edges": edges,
        "round_count": len(llm_calls),
        "tool_call_count": len(tool_calls),
        "max_tools_emitted_by_round": max((len(value) for value in assignments.values()), default=0),
        "topology_signature": topology,
        "topology_sha256": hashlib.sha256(topology.encode("utf-8")).hexdigest(),
    }


def summarize_dynamicity(traces: Iterable[dict[str, Any]]) -> dict[str, Any]:
    graphs = [item.get("dynamic_graph") for item in traces]
    graphs = [item for item in graphs if isinstance(item, dict)]
    rounds = [int(item.get("round_count") or 0) for item in graphs]
    tools = [int(item.get("tool_call_count") or 0) for item in graphs]
    signatures = Counter(str(item.get("topology_sha256") or "") for item in graphs)

    def stats(values: list[int]) -> dict[str, float | int | None]:
        if not values:
            return {"min": None, "mean": None, "p50": None, "p95": None, "max": None}
        ordered = sorted(values)

        def percentile(p: float) -> float:
            index = (len(ordered) - 1) * p / 100.0
            lo = int(math.floor(index))
            hi = int(math.ceil(index))
            if lo == hi:
                return float(ordered[lo])
            return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo))

        return {
            "min": min(values),
            "mean": sum(values) / len(values),
            "p50": percentile(50),
            "p95": percentile(95),
            "max": max(values),
        }

    entropy = 0.0
    if graphs:
        for count in signatures.values():
            probability = count / len(graphs)
            entropy -= probability * math.log2(probability)
    return {
        "schema": "mars_dynamicity_summary_v1",
        "request_count": len(graphs),
        "round_count": stats(rounds),
        "tool_call_count": stats(tools),
        "unique_round_counts": sorted(set(rounds)),
        "unique_topology_count": len(signatures),
        "topology_entropy_bits": entropy,
        "topology_frequencies": dict(signatures.most_common()),
    }
