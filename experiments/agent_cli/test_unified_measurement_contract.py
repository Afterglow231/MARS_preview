#!/usr/bin/env python3
"""Cross-frontend regression gates for the shared measurement contract."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest


AGENT_CLI_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = AGENT_CLI_DIR.parent
OPENHANDS_DIR = EXPERIMENTS_DIR / "openhands"
# Import the flat native bundle before adding the OpenHands flat bundle to the
# front of sys.path; both intentionally contain a request_worker.py.
import native_request_worker as native_worker  # noqa: E402
from frontend_contract import AgentRequestContext, FrontendTerminalStatus  # noqa: E402
from lifecycle_collector import LifecycleCollector  # noqa: E402

if str(OPENHANDS_DIR) not in sys.path:
    sys.path.insert(0, str(OPENHANDS_DIR))

import run_task_workload as openhands_workload  # noqa: E402
import trace_writer as openhands_trace  # noqa: E402


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


class EventBuffer:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def write(self, row: dict[str, Any]) -> None:
        self.rows.append(dict(row))


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in {path}")


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _call_lines(function: ast.FunctionDef, name: str) -> list[int]:
    return sorted(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _call_name(node) == name
    )


def _native_context(tmp_path: Path) -> AgentRequestContext:
    return AgentRequestContext(
        request_id="req-contract",
        job_id="req-contract",
        dataset_request_id="dataset-contract",
        benchmark="terminalbench",
        task_id="task-contract",
        prompt="solve",
        workspace_dir=tmp_path / "native-workspace",
        model="served-model",
        backend_base_url="http://127.0.0.1:8000/v1",
        arrival_time_s=990.0,
        arrival_monotonic_s=0.0,
        metadata={"public_latency_clock": "wall_openhands"},
    )


def test_both_frontends_capture_queue_start_before_materialization() -> None:
    openhands_runner = _function(OPENHANDS_DIR / "openhands_runner.py", "run_task_request")
    openhands_start = _call_lines(openhands_runner, "mark_worker_start")
    openhands_materialize = _call_lines(openhands_runner, "materialize_workspace_init")
    assert len(openhands_start) == len(openhands_materialize) == 1
    assert openhands_start[0] < openhands_materialize[0]

    native_run = _function(AGENT_CLI_DIR / "native_request_worker.py", "run_request")
    materialize_line = _call_lines(native_run, "materialize_workspace_init")[0]
    wall_capture_lines = [
        node.lineno
        for node in ast.walk(native_run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "time"
        and node.func.attr == "time"
    ]
    monotonic_capture_lines = [
        node.lineno
        for node in ast.walk(native_run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "time"
        and node.func.attr == "monotonic"
    ]
    assert min(wall_capture_lines) < materialize_line
    assert min(monotonic_capture_lines) < materialize_line

    request_started_calls = [
        node
        for node in ast.walk(native_run)
        if isinstance(node, ast.Call) and _call_name(node) == "request_started"
    ]
    assert len(request_started_calls) == 1
    observed_args = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in request_started_calls[0].keywords
    }
    assert observed_args == {
        "observed_time_s": "request_started_wall",
        "observed_monotonic_s": "request_started_mono",
    }


def test_queue_value_and_terminal_time_are_frozen_before_setup_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = ManualClock()
    monkeypatch.setattr(openhands_trace, "now_s", clock.wall_time)
    openhands_events = EventBuffer()
    openhands_collector = openhands_trace.RequestTraceCollector(
        events=openhands_events,
        request_id="req-contract",
        dataset_request_id="dataset-contract",
        benchmark="terminalbench",
        task_id="task-contract",
        arrival_time_s=990.0,
        workspace_dir=tmp_path / "openhands-workspace",
    )
    openhands_collector.mark_worker_start()
    clock.advance(20.0)  # workspace materialization/setup is not queue wait
    openhands_collector.set_workspace_materialized(3)
    clock.advance(5.0)
    openhands_collector.mark_frontend_terminal()
    clock.advance(100.0)  # Conversation/sandbox cleanup is not request latency
    openhands_result = openhands_collector.finalize(
        status="completed", conversation_status="finished"
    )
    assert openhands_result["queue_wait_s"] == pytest.approx(10.0)
    assert openhands_result["end_time_s"] == pytest.approx(1025.0)
    assert openhands_result["e2e_latency_s"] == pytest.approx(35.0)

    native_clock = ManualClock()
    native_collector = LifecycleCollector(
        _native_context(tmp_path),
        wall_clock=native_clock.wall_time,
        monotonic_clock=native_clock.monotonic,
    )
    native_collector.request_started(
        observed_time_s=native_clock.wall_time(),
        observed_monotonic_s=native_clock.monotonic(),
    )
    native_clock.advance(20.0)  # workspace/frontend setup
    native_collector.frontend_started(native_session_id="native-session")
    native_clock.advance(5.0)
    native_collector.frontend_terminal(FrontendTerminalStatus.COMPLETED)
    before_cleanup = native_collector.snapshot_openhands_trace()
    native_clock.advance(100.0)
    native_collector.record_native_event(
        {"type": "cleanup.finished"},
        source="runtime",
        event_type="cleanup.finished",
    )
    after_cleanup = native_collector.snapshot_openhands_trace()
    assert before_cleanup["queue_wait_s"] == pytest.approx(10.0)
    assert before_cleanup["end_time_s"] == after_cleanup["end_time_s"] == 1025.0
    assert before_cleanup["e2e_latency_s"] == after_cleanup["e2e_latency_s"] == 35.0


def test_call_sites_freeze_terminal_before_frontend_and_container_cleanup() -> None:
    openhands_runner = _function(OPENHANDS_DIR / "openhands_runner.py", "run_task_request")
    terminal_line = _call_lines(openhands_runner, "mark_frontend_terminal")[0]
    conversation_close_line = _call_lines(openhands_runner, "close")[0]
    assert terminal_line < conversation_close_line

    for helper_name in ("_run_codex", "_run_opencode"):
        helper = _function(AGENT_CLI_DIR / "native_request_worker.py", helper_name)
        terminal_calls = _call_lines(helper, "frontend_terminal")
        assert len(terminal_calls) == 1
        driver_contexts = [node for node in ast.walk(helper) if isinstance(node, ast.With)]
        assert any(
            any(
                isinstance(node, ast.Call) and _call_name(node) == "frontend_terminal"
                for node in ast.walk(context)
            )
            for context in driver_contexts
        ), f"{helper_name} must freeze terminal while its driver context is still alive"

    native_run = _function(AGENT_CLI_DIR / "native_request_worker.py", "run_request")
    docker_cleanup_line = _call_lines(native_run, "cleanup")[0]
    assert _call_lines(native_run, "_run_codex")[0] < docker_cleanup_line
    assert _call_lines(native_run, "_run_opencode")[0] < docker_cleanup_line


def test_every_summary_distribution_reports_coverage() -> None:
    traces = [
        {
            "arrival_time_s": 1.0,
            "end_time_s": 5.0,
            "e2e_latency_s": 4.0,
            "gpu_total_s": 2.0,
            "tool_total_s": 1.0,
            "tool_wait_total_s": 0.25,
            "queue_wait_s": 0.5,
            "wait_total_s": 0.75,
            "llm_calls": [{"round_id": "round-1"}],
            "tool_calls": [{"tool_call_id": "tool-1"}],
            "status": "completed",
        },
        {
            "arrival_time_s": 2.0,
            "end_time_s": 6.0,
            "e2e_latency_s": None,
            "gpu_total_s": None,
            "tool_total_s": None,
            "tool_wait_total_s": None,
            "queue_wait_s": 1.0,
            "wait_total_s": None,
            "llm_calls": [],
            "tool_calls": [],
            "status": "worker_error",
        },
    ]
    summary = openhands_workload._summary(traces, run={"requested_count": 2})
    distributions = (
        "latency_s",
        "gpu_total_s",
        "tool_total_s",
        "tool_wait_total_s",
        "queue_wait_s",
        "wait_total_s",
        "llm_call_count",
        "tool_call_count",
    )
    for name in distributions:
        assert {"sample_count", "unknown_count", "coverage"} <= set(summary[name])
        assert summary[name]["sample_count"] + summary[name]["unknown_count"] == 2
        assert summary[name]["coverage"] == pytest.approx(
            summary[name]["sample_count"] / 2.0
        )
    assert summary["latency_s"]["sample_count"] == 1
    assert summary["latency_s"]["unknown_count"] == 1
    assert summary["queue_wait_s"]["coverage"] == 1.0
    assert summary["llm_call_count"]["coverage"] == 1.0


def test_native_compatibility_events_use_logical_rounds_not_backend_attempts(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    collector = LifecycleCollector(
        _native_context(tmp_path),
        wall_clock=clock.wall_time,
        monotonic_clock=clock.monotonic,
    )
    collector.request_started(
        observed_time_s=clock.wall_time(), observed_monotonic_s=clock.monotonic()
    )
    collector.frontend_started(native_session_id="session-1")
    collector.llm_round_started("round-1", api_kind="responses")
    collector.backend_attempt_started(
        "attempt-1", round_id="round-1", protocol="responses"
    )
    clock.advance(1.0)
    collector.backend_attempt_ended(
        "attempt-1", status="retryable_error", http_status=429
    )
    collector.backend_attempt_started(
        "attempt-2", round_id="round-1", protocol="responses"
    )
    clock.advance(1.0)
    collector.llm_round_first_token("round-1")
    clock.advance(1.0)
    collector.backend_attempt_ended("attempt-2", status="completed", http_status=200)
    clock.advance(1.0)
    collector.llm_round_ended(
        "round-1", status="completed", native_response_id="response-1"
    )
    collector.frontend_terminal(FrontendTerminalStatus.COMPLETED)

    trace = collector.snapshot_openhands_trace()
    events = native_worker._request_events(
        collector,
        trace,
        task={
            "request_id": "dataset-contract",
            "benchmark": "terminalbench",
            "task_id": "task-contract",
        },
        verification_nudges=[],
    )
    event_names = [event["event"] for event in events]
    assert event_names.count("request_done") == 1
    assert event_names.count("gpu_submit") == 1
    assert event_names.count("gpu_first_token") == 1
    assert event_names.count("gpu_end") == 1

    gpu_events = [event for event in events if str(event["event"]).startswith("gpu_")]
    assert {event["projection_source"] for event in gpu_events} == {
        "native_logical_round"
    }
    assert {event["llm_call_id"] for event in gpu_events} == {"round-1"}
    assert all("attempt_id" not in event for event in gpu_events)
    assert all("backend_attempt_id" not in event for event in gpu_events)

    assert len(trace["llm_calls"]) == 1
    assert len(trace["rounds"]) == 1
    assert len(trace["backend_attempts"]) == 2
    assert [
        attempt["backend_attempt_id"]
        for attempt in trace["rounds"][0]["backend_attempts"]
    ] == ["attempt-1", "attempt-2"]
    done = next(event for event in events if event["event"] == "request_done")
    assert done["llm_call_count"] == 1
    assert done["backend_attempt_count"] == 2
