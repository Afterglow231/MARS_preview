#!/usr/bin/env python3

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

try:
    from .frontend_contract import (
        AgentEventSink,
        AgentRequestContext,
        FrontendTerminalStatus,
    )
    from .lifecycle_collector import LifecycleCollector, LifecycleStateError
except ImportError:  # Supports direct execution from this directory.
    from frontend_contract import (  # type: ignore
        AgentEventSink,
        AgentRequestContext,
        FrontendTerminalStatus,
    )
    from lifecycle_collector import LifecycleCollector, LifecycleStateError  # type: ignore


class ManualClock:
    def __init__(self, *, wall: float = 1000.0, monotonic: float = 10.0) -> None:
        self.wall = wall
        self.monotonic_value = monotonic

    def wall_time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.monotonic_value

    def advance(self, seconds: float) -> None:
        self.wall += seconds
        self.monotonic_value += seconds


def _context(tmp_path: Path, *, arrival_monotonic_s: float | None = 0.0) -> AgentRequestContext:
    return AgentRequestContext(
        request_id="req-1",
        job_id="job-1",
        dataset_request_id="dataset-1",
        benchmark="terminalbench",
        task_id="task-1",
        prompt="solve",
        workspace_dir=tmp_path / "workspace",
        model="model",
        backend_base_url="http://127.0.0.1:8000/v1",
        arrival_time_s=990.0,
        arrival_monotonic_s=arrival_monotonic_s,
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_logical_round_is_distinct_from_backend_retry_attempts(tmp_path: Path) -> None:
    clock = ManualClock()
    collector = LifecycleCollector(
        _context(tmp_path),
        monotonic_clock=clock.monotonic,
        wall_clock=clock.wall_time,
    )
    assert isinstance(collector, AgentEventSink)
    collector.request_started()
    clock.advance(1.0)
    collector.frontend_started(native_session_id="session-1")
    clock.advance(1.0)
    collector.llm_round_started("round-1", api_kind="responses")
    collector.backend_attempt_started(
        "attempt-1", round_id="round-1", protocol="responses"
    )
    clock.advance(1.0)
    collector.backend_attempt_ended("attempt-1", status="retryable_error", http_status=500)
    collector.backend_attempt_started(
        "attempt-2", round_id="round-1", protocol="responses"
    )
    clock.advance(2.0)
    collector.backend_attempt_ended("attempt-2", status="ok", http_status=200)
    clock.advance(1.0)
    collector.llm_round_ended(
        "round-1",
        status="ok",
        native_response_id="response-1",
        usage={"prompt_tokens": 10, "completion_tokens": 3},
    )
    collector.frontend_terminal(FrontendTerminalStatus.COMPLETED)

    trace = collector.snapshot_openhands_trace()
    assert trace["gpu_total_s"] == pytest.approx(4.0)
    assert trace["backend_attempt_total_s"] == pytest.approx(3.0)
    assert trace["llm_call_count"] == 1
    assert trace["backend_attempt_count"] == 2
    assert len(trace["rounds"]) == 1
    assert [
        item["backend_attempt_id"] for item in trace["rounds"][0]["backend_attempts"]
    ] == ["attempt-1", "attempt-2"]


def test_backend_attempt_can_close_after_frontend_terminal(tmp_path: Path) -> None:
    clock = ManualClock()
    collector = LifecycleCollector(
        _context(tmp_path),
        monotonic_clock=clock.monotonic,
        wall_clock=clock.wall_time,
    )
    collector.request_started()
    collector.frontend_started(native_session_id="session-1")
    collector.llm_round_started("round-1", api_kind="responses")
    collector.backend_attempt_started(
        "attempt-1", round_id="round-1", protocol="responses"
    )
    clock.advance(2.0)
    collector.llm_round_ended(
        "round-1",
        status="completed",
        native_response_id="response-1",
    )
    collector.frontend_terminal(FrontendTerminalStatus.COMPLETED)
    terminal_end_time_s = collector.snapshot_openhands_trace()["end_time_s"]
    clock.advance(0.1)

    collector.backend_attempt_ended(
        "attempt-1",
        status="completed",
        http_status=200,
        metadata={"native_response_id": "response-1"},
    )

    trace = collector.snapshot_openhands_trace()
    attempt = trace["backend_attempts"][0]
    assert attempt["status"] == "completed"
    assert attempt["http_status"] == 200
    assert attempt["end_time_s"] is not None
    assert trace["end_time_s"] == terminal_end_time_s


def test_missing_queue_timestamp_keeps_tool_wait_unknown_and_live_is_flushed(
    tmp_path: Path,
) -> None:
    live_path = tmp_path / "live.jsonl"
    clock = ManualClock()
    collector = LifecycleCollector(
        _context(tmp_path, arrival_monotonic_s=None),
        live_events_path=live_path,
        monotonic_clock=clock.monotonic,
        wall_clock=clock.wall_time,
    )
    collector.request_started()
    collector.frontend_started()
    collector.tool_started("tool-1", tool_name="terminal")

    # The scheduler can observe the start before the tool finishes because the
    # compatibility JSONL writer flushes on every transition.
    live_rows = _rows(live_path)
    assert [row["event"] for row in live_rows] == ["tool_start"]
    assert live_rows[0]["wait_s"] is None

    clock.advance(2.0)
    collector.tool_ended("tool-1", status="ok")
    collector.frontend_terminal(FrontendTerminalStatus.COMPLETED)
    snapshot = collector.snapshot()
    tool = snapshot["tool_calls"][0]
    assert tool["wait_s"] is None
    assert tool["duration_s"] == pytest.approx(2.0)
    assert snapshot["metrics"]["queue_wait_s"] is None
    assert snapshot["metrics"]["tool_wait_total_s"] is None
    assert snapshot["metrics"]["wait_total_s"] is None
    assert [row["event"] for row in _rows(live_path)] == ["tool_start", "tool_end"]


def test_real_queue_event_produces_measured_wait(tmp_path: Path) -> None:
    clock = ManualClock()
    collector = LifecycleCollector(
        _context(tmp_path),
        monotonic_clock=clock.monotonic,
        wall_clock=clock.wall_time,
    )
    collector.tool_queued("tool-1", tool_name="file_editor", parent_round_id="round-1")
    clock.advance(0.75)
    collector.tool_started("tool-1")
    clock.advance(1.25)
    collector.tool_ended("tool-1", status="ok")
    collector.frontend_terminal(FrontendTerminalStatus.COMPLETED)

    tool = collector.snapshot()["tool_calls"][0]
    assert tool["wait_s"] == pytest.approx(0.75)
    assert tool["duration_s"] == pytest.approx(1.25)


def test_native_jsonl_is_append_only_and_payload_is_detached(tmp_path: Path) -> None:
    native_path = tmp_path / "native.jsonl"
    native_path.write_text('{"existing":true}\n', encoding="utf-8")
    clock = ManualClock()
    collector = LifecycleCollector(
        _context(tmp_path),
        native_events_path=native_path,
        monotonic_clock=clock.monotonic,
        wall_clock=clock.wall_time,
    )
    raw = {"type": "item.started", "item": {"values": [1]}}
    collector.record_native_event(raw, source="codex", event_type="item.started")
    raw["item"]["values"].append(2)

    rows = _rows(native_path)
    assert rows[0] == {"existing": True}
    assert rows[1]["raw"]["item"]["values"] == [1]
    assert rows[1]["native_timestamp_s"] is None
    assert collector.snapshot()["native_events"][0]["raw"]["item"]["values"] == [1]


def test_terminal_timestamp_does_not_move_during_cleanup_audit_events(
    tmp_path: Path,
) -> None:
    clock = ManualClock(monotonic=10.0, wall=1000.0)
    collector = LifecycleCollector(
        _context(tmp_path, arrival_monotonic_s=5.0),
        monotonic_clock=clock.monotonic,
        wall_clock=clock.wall_time,
    )
    collector.request_started()
    collector.frontend_started(native_session_id="session")
    clock.advance(4.0)
    outcome = collector.frontend_terminal(FrontendTerminalStatus.COMPLETED)
    before = collector.snapshot_openhands_trace()

    # Native process/cleanup records may be retained for provenance, but they
    # cannot redefine frontend completion or e2e latency.
    clock.advance(120.0)
    collector.record_native_event(
        {"type": "cleanup.finished"}, source="runtime", event_type="cleanup.finished"
    )
    after = collector.snapshot_openhands_trace()
    assert outcome.terminal_monotonic_s == 14.0
    assert before["end_time_s"] == after["end_time_s"] == 1004.0
    assert before["e2e_latency_s"] == after["e2e_latency_s"] == 9.0
    with pytest.raises(LifecycleStateError, match="already terminal"):
        collector.tool_started("late-tool", tool_name="terminal")


def test_parallel_tools_have_independent_live_lifetimes(tmp_path: Path) -> None:
    live_path = tmp_path / "live.jsonl"
    collector = LifecycleCollector(_context(tmp_path), live_events_path=live_path)
    collector.request_started()
    collector.frontend_started(native_session_id="parallel-session")
    count = 12
    all_started = threading.Barrier(count + 1)
    release = threading.Event()

    def run_tool(index: int) -> None:
        tool_id = f"tool-{index:02d}"
        collector.tool_queued(
            tool_id,
            tool_name="terminal",
            parent_round_id="round-parallel",
            native_tool_call_id=f"native-{index:02d}",
        )
        collector.tool_started(tool_id)
        all_started.wait(timeout=10.0)
        assert release.wait(timeout=10.0)
        collector.tool_ended(tool_id, status="ok")

    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = [executor.submit(run_tool, index) for index in range(count)]
        all_started.wait(timeout=10.0)
        running = collector.snapshot()
        assert running["metrics"]["active_tool_count"] == count
        assert len(running["active_tool_ids"]) == count
        assert len(_rows(live_path)) == count
        release.set()
        for future in futures:
            future.result(timeout=10.0)

    collector.frontend_terminal(FrontendTerminalStatus.COMPLETED)
    snapshot = collector.snapshot()
    assert snapshot["metrics"]["active_tool_count"] == 0
    assert snapshot["metrics"]["tool_call_count"] == count
    assert snapshot["metrics"]["tool_wait_unknown_count"] == 0
    live_rows = _rows(live_path)
    assert len(live_rows) == count * 2
    for index in range(count):
        tool_id = f"tool-{index:02d}"
        events = [row["event"] for row in live_rows if row["tool_call_id"] == tool_id]
        assert events == ["tool_start", "tool_end"]


def test_incomplete_spans_remain_unknown_in_derived_metrics(tmp_path: Path) -> None:
    collector = LifecycleCollector(_context(tmp_path))
    collector.llm_round_started("round-incomplete")
    collector.tool_queued("queued-only", tool_name="terminal")
    collector.frontend_terminal(FrontendTerminalStatus.ERROR, reason="native failure")

    trace = collector.snapshot_openhands_trace()
    assert trace["gpu_total_s"] is None
    assert trace["tool_total_s"] is None
    assert trace["tool_wait_total_s"] is None
    assert trace["llm_duration_unknown_count"] == 1
    assert trace["tool_duration_unknown_count"] == 1
    assert trace["error"] == {
        "type": "FrontendTerminal",
        "message": "native failure",
    }


def test_completed_frontend_rejects_unresolved_tool(tmp_path: Path) -> None:
    collector = LifecycleCollector(_context(tmp_path))
    collector.tool_queued("queued-only", tool_name="terminal")

    with pytest.raises(LifecycleStateError, match="unresolved tools"):
        collector.frontend_terminal(FrontendTerminalStatus.COMPLETED)
