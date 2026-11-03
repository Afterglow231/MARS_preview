#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from event_normalizer import (
    LiveToolEventNormalizer,
    OpenCodeTimingIntegrityError,
    normalize_codex,
    normalize_opencode,
)


def _row(ts: float, raw: dict) -> dict:
    return {"captured_at_s": ts, "raw": raw}


def test_codex_todo_lifetime_is_not_tool_execution() -> None:
    rows = [
        _row(10.0, {"type": "item.started", "item": {"id": "todo", "type": "todo_list", "items": []}}),
        _row(20.0, {"type": "item.updated", "item": {"id": "todo", "type": "todo_list", "items": []}}),
        _row(1000.0, {"type": "item.completed", "item": {"id": "todo", "type": "todo_list", "items": []}}),
        _row(30.0, {"type": "item.started", "item": {"id": "cmd", "type": "command_execution"}}),
        _row(
            32.5,
            {
                "type": "item.completed",
                "item": {"id": "cmd", "type": "command_execution", "status": "completed"},
            },
        ),
    ]
    tools, _messages = normalize_codex(rows)

    todos = [item for item in tools if item["tool_name"] == "task_tracker"]
    command = next(item for item in tools if item["tool_name"] == "command_execution")
    assert len(todos) == 2
    assert sum(float(item["duration_s"]) for item in todos) == 0.0
    assert command["duration_s"] == 2.5


def test_opencode_updates_are_deduplicated() -> None:
    base = {"type": "tool_use", "part": {"id": "part", "callID": "call", "tool": "bash"}}
    running = {**base, "part": {**base["part"], "state": {"status": "running", "time": {"start": 1000}}}}
    done = {
        **base,
        "part": {
            **base["part"],
            "state": {"status": "completed", "time": {"start": 1000, "end": 3500}},
        },
    }
    tools, _messages = normalize_opencode([_row(1.1, running), _row(3.6, done)])
    assert len(tools) == 1
    assert tools[0]["duration_s"] == 2.5


def test_opencode_nonzero_shell_exit_is_a_tool_error() -> None:
    raw = {
        "type": "tool_use",
        "part": {
            "id": "part",
            "callID": "call",
            "tool": "bash",
            "state": {
                "status": "completed",
                "time": {"start": 1000, "end": 1250},
                "output": "python: command not found",
                "metadata": {"exit": 127},
            },
        },
    }
    tools, _messages = normalize_opencode([_row(1.3, raw)])
    assert tools[0]["status"] == "failed"
    assert tools[0]["exit_code"] == 127

    live = LiveToolEventNormalizer(agent="opencode", request_id="req").feed(raw, 1.3)
    assert [item["event"] for item in live] == ["tool_start", "tool_end"]
    assert live[-1]["status"] == "failed"


def test_opencode_bash_uses_hash_matched_configured_shell_wall_time(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workdir = workspace / "nested"
    workdir.mkdir(parents=True)
    command = "sleep 0.35; pwd"
    command_hash = hashlib.sha256(command.encode()).hexdigest()
    cwd_hash = hashlib.sha256(str(workdir.resolve()).encode()).hexdigest()
    native_rows = [
        _row(
            100.01,
            {
                "type": "tool_use",
                "part": {
                    "id": "part-1",
                    "callID": "call-1",
                    "tool": "bash",
                    "state": {
                        "status": "completed",
                        # Pinned OpenCode overwrites this start near completion.
                        "time": {"start": 100000, "end": 100007},
                        "input": {"command": command, "workdir": "nested"},
                        "metadata": {"exit": 0},
                    },
                },
            },
        )
    ]
    shell_rows = [
        {
            "event": "shell_start",
            "shell_call_id": "initial:shell-1",
            "ts": 10.0,
            "command_sha256": command_hash,
            "cwd_sha256": cwd_hash,
        },
        {
            "event": "shell_end",
            "shell_call_id": "initial:shell-1",
            "ts": 10.36,
            "duration_s": 0.36,
            "exit_code": 0,
            "status": "completed",
            "command_sha256": command_hash,
            "cwd_sha256": cwd_hash,
        },
    ]

    tools, _messages = normalize_opencode(
        native_rows,
        shell_rows=shell_rows,
        workspace_dir=workspace,
        require_shell_timing=True,
    )
    assert tools[0]["start_time_s"] == 10.0
    assert tools[0]["end_time_s"] == 10.36
    assert tools[0]["duration_s"] == pytest.approx(0.36)
    assert tools[0]["timing_source"] == "opencode_configured_shell_wrapper"
    assert tools[0]["timing_integrity"] == "matched"
    assert tools[0]["timing_match_occurrence"] == 0
    assert tools[0]["shell_call_id"] == "initial:shell-1"

    suppressed = LiveToolEventNormalizer(
        agent="opencode", request_id="req", opencode_shell_timing=True
    ).feed(native_rows[0]["raw"], 100.01)
    assert suppressed == []


def test_opencode_shell_timing_mismatch_is_an_explicit_failure(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw = {
        "type": "tool_use",
        "part": {
            "id": "part",
            "callID": "call",
            "tool": "bash",
            "state": {
                "status": "completed",
                "time": {"start": 1000, "end": 1001},
                "input": {"command": "pwd"},
            },
        },
    }
    with pytest.raises(OpenCodeTimingIntegrityError) as raised:
        normalize_opencode(
            [_row(1.1, raw)],
            shell_rows=[],
            workspace_dir=workspace,
            require_shell_timing=True,
        )
    assert raised.value.details["status"] == "failed"
    assert raised.value.details["native_bash_count"] == 1
    assert raised.value.details["matched_count"] == 0


def test_opencode_identical_hashes_are_paired_by_occurrence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "printf same"
    command_hash = hashlib.sha256(command.encode()).hexdigest()
    cwd_hash = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()

    def native(call_id: str, observed: float) -> dict:
        return _row(
            observed,
            {
                "type": "tool_use",
                "part": {
                    "id": f"part-{call_id}",
                    "callID": call_id,
                    "tool": "bash",
                    "state": {
                        "status": "completed",
                        "time": {"start": observed * 1000, "end": observed * 1000 + 1},
                        "input": {"command": command},
                    },
                },
            },
        )

    shell_rows = [
        {
            "event": "shell_start",
            "shell_call_id": "shell-0",
            "ts": 10.0,
            "command_sha256": command_hash,
            "cwd_sha256": cwd_hash,
        },
        {
            "event": "shell_start",
            "shell_call_id": "shell-1",
            "ts": 10.1,
            "command_sha256": command_hash,
            "cwd_sha256": cwd_hash,
        },
        {
            "event": "shell_end",
            "shell_call_id": "shell-1",
            "ts": 10.2,
            "command_sha256": command_hash,
            "cwd_sha256": cwd_hash,
            "exit_code": 0,
        },
        {
            "event": "shell_end",
            "shell_call_id": "shell-0",
            "ts": 10.4,
            "command_sha256": command_hash,
            "cwd_sha256": cwd_hash,
            "exit_code": 0,
        },
    ]
    tools, _messages = normalize_opencode(
        [native("call-0", 20.0), native("call-1", 20.1)],
        shell_rows=shell_rows,
        workspace_dir=workspace,
        require_shell_timing=True,
    )
    assert [tool["timing_match_occurrence"] for tool in tools] == [0, 1]
    assert [tool["shell_call_id"] for tool in tools] == ["shell-0", "shell-1"]
    assert [tool["duration_s"] for tool in tools] == pytest.approx([0.4, 0.1])


def test_live_codex_tools_match_openhands_event_names() -> None:
    normalizer = LiveToolEventNormalizer(agent="codex", request_id="req")
    start = normalizer.feed(
        {"type": "item.started", "item": {"id": "cmd", "type": "command_execution"}},
        10.0,
    )
    end = normalizer.feed(
        {
            "type": "item.completed",
            "item": {"id": "cmd", "type": "command_execution", "status": "completed"},
        },
        12.0,
    )
    assert [item["event"] for item in start + end] == ["tool_start", "tool_end"]
    assert start[0]["wait_s"] == 0.0
    assert end[0]["duration_s"] == 2.0


def test_codex_command_uses_bash_wrapper_wall_time() -> None:
    cli_rows = [
        _row(
            100.0,
            {
                "type": "item.started",
                "item": {"id": "cmd", "type": "command_execution", "command": "sleep 1"},
            },
        ),
        _row(
            100.001,
            {
                "type": "item.completed",
                "item": {
                    "id": "cmd",
                    "type": "command_execution",
                    "command": "sleep 1",
                    "exit_code": 0,
                    "status": "completed",
                },
            },
        ),
    ]
    shell_rows = [
        {"event": "shell_start", "shell_call_id": "shell-1", "ts": 10.0},
        {
            "event": "shell_end",
            "shell_call_id": "shell-1",
            "ts": 11.25,
            "duration_s": 1.25,
            "exit_code": 0,
            "status": "completed",
        },
    ]
    tools, _messages = normalize_codex(cli_rows, shell_rows=shell_rows)
    assert tools[0]["start_time_s"] == 10.0
    assert tools[0]["end_time_s"] == 11.25
    assert tools[0]["duration_s"] == 1.25
    assert tools[0]["timing_source"] == "codex_bash_wrapper"
    assert tools[0]["shell_call_id"] == "shell-1"

    live = LiveToolEventNormalizer(
        agent="codex", request_id="req", codex_shell_timing=True
    )
    assert live.feed(cli_rows[0]["raw"], 100.0) == []
    assert live.feed(cli_rows[1]["raw"], 100.001) == []


def test_codex_nonzero_shell_exit_is_a_tool_error() -> None:
    cli_rows = [
        _row(
            20.0,
            {
                "type": "item.started",
                "item": {"id": "cmd", "type": "command_execution"},
            },
        ),
        _row(
            20.1,
            {
                "type": "item.completed",
                "item": {
                    "id": "cmd",
                    "type": "command_execution",
                    "exit_code": 7,
                    "status": "completed",
                },
            },
        ),
    ]
    tools, _messages = normalize_codex(cli_rows)
    assert tools[0]["exit_code"] == 7
    assert tools[0]["status"] == "failed"
