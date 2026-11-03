#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from request_worker import (
    _agent_sandbox_command,
    _codex_command,
    _llm_retry_policy,
    _namespace_phase_records,
    _native_session_id,
    _openhands_tool_calls,
    _opencode_command,
    _remaining_timeout_s,
    _request_events,
    _run_and_capture,
    _verification_tool_count,
    _write_codex_config,
    _write_codex_shell_wrapper,
    _write_opencode_config,
    _write_opencode_shell_wrapper,
)
from sandbox_paths import task_readonly_paths


def test_agent_prompts_are_not_in_argv(tmp_path: Path) -> None:
    binary = tmp_path / "agent"
    workspace = tmp_path / "workspace"
    config = tmp_path / "config.json"
    binary.touch()
    workspace.mkdir()

    opencode = _opencode_command(
        binary=binary,
        config_path=config,
        workspace_dir=workspace,
        model="model",
    )
    codex = _codex_command(binary=binary, workspace_dir=workspace, model="model")

    assert opencode[-1] == str(workspace)
    assert codex[-1] == "-"
    assert codex[codex.index("--sandbox") + 1] == "danger-full-access"


def test_native_followups_resume_the_exact_session(tmp_path: Path) -> None:
    binary = tmp_path / "agent"
    workspace = tmp_path / "workspace"
    config = tmp_path / "config.json"
    binary.touch()
    workspace.mkdir()

    opencode = _opencode_command(
        binary=binary,
        config_path=config,
        workspace_dir=workspace,
        model="model",
        session_id="ses_open",
    )
    codex = _codex_command(
        binary=binary,
        workspace_dir=workspace,
        model="model",
        session_id="thread_codex",
        ephemeral=False,
    )

    assert opencode[-2:] == ["--session", "ses_open"]
    assert "--ephemeral" not in codex
    resume_index = codex.index("resume")
    assert codex[resume_index:] == ["resume", "thread_codex", "-"]
    for option in ("--json", "--skip-git-repo-check", "--color", "--sandbox", "--cd", "--model"):
        assert codex.index(option) < resume_index


def test_native_session_extraction_is_stable_and_conflicts_are_rejected() -> None:
    codex_rows = [
        {"raw": {"type": "thread.started", "thread_id": "thread-1"}},
        {"raw": {"type": "turn.completed"}},
    ]
    opencode_rows = [
        {"raw": {"type": "text", "sessionID": "session-1", "part": {}}},
        {"raw": {"type": "step_finish", "sessionID": "session-1", "part": {}}},
    ]
    assert _native_session_id("codex", codex_rows) == "thread-1"
    assert _native_session_id("opencode", opencode_rows) == "session-1"
    with pytest.raises(ValueError, match="conflicting opencode session ids"):
        _native_session_id(
            "opencode",
            opencode_rows + [{"raw": {"sessionID": "session-2"}}],
        )


def test_phase_namespacing_prevents_native_id_collisions() -> None:
    native = [{"tool_call_id": "tool-1", "raw_part_id": "part-1"}]
    initial = _namespace_phase_records(native, phase="initial", id_field="tool_call_id")
    nudge = _namespace_phase_records(
        native, phase="verification_nudge_1", id_field="tool_call_id"
    )
    assert initial[0]["tool_call_id"] == "initial:tool-1"
    assert nudge[0]["tool_call_id"] == "verification_nudge_1:tool-1"
    assert initial[0]["native_tool_call_id"] == nudge[0]["native_tool_call_id"] == "tool-1"
    assert initial[0]["native_raw_part_id"] == "part-1"


def test_remaining_timeout_uses_one_whole_request_deadline() -> None:
    now = time.monotonic()
    assert _remaining_timeout_s(now + 10.0, now=now + 4.0) == pytest.approx(6.0)
    assert _remaining_timeout_s(now + 10.0, now=now + 11.0) == 0.0
    assert _remaining_timeout_s(None, now=now) == 0.0


def test_verification_tools_match_openhands_equivalents_only() -> None:
    calls = [
        {"tool_name": "command_execution"},
        {"tool_name": "read"},
        {"tool_name": "webfetch"},
        {"tool_name": "glob"},
        {"tool_name": "grep"},
        {"tool_name": "task_tracker"},
        {"tool_name": "lsp"},
        {"tool_name": "skill"},
        {"tool_name": "finish"},
    ]
    assert _verification_tool_count(calls) == 5


def test_retry_provenance_does_not_claim_opencode_accepted_three() -> None:
    codex = _llm_retry_policy("codex", 3)
    opencode = _llm_retry_policy("opencode", 3)
    assert codex["request_max_retries"] == codex["stream_max_retries"] == 3
    assert codex["effective_matches_requested"] is True
    assert opencode["sdk_max_retries"] == 0
    assert opencode["outer_max_retries"] == 5
    assert opencode["max_total_attempts"] == 6
    assert opencode["effective_matches_requested"] is False


def test_request_start_carries_measured_queue_wait(tmp_path: Path) -> None:
    trace = {
        "benchmark": "repobench",
        "task_id": "task",
        "queue_wait_s": 1.25,
        "e2e_latency_s": 2.0,
        "gpu_total_s": 0.0,
        "tool_total_s": 0.0,
        "tool_wait_total_s": 0.0,
        "wait_total_s": 1.25,
        "dynamic_graph": {},
    }
    events = _request_events(
        agent="codex",
        request_id="req",
        mars_job_id="req",
        dataset_request_id="dataset",
        started=10.0,
        ended=12.0,
        workspace_dir=tmp_path,
        llm_calls=[],
        tool_calls=[],
        status="completed",
        trace=trace,
    )
    request_start = next(item for item in events if item["event"] == "request_start")
    assert request_start["queue_wait_s"] == 1.25


def test_large_prompt_is_streamed_through_stdin(tmp_path: Path) -> None:
    prompt = "x" * (3 * 1024 * 1024)
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    stdout_path = tmp_path / "stdout.jsonl"
    stderr_path = tmp_path / "stderr.txt"
    events_path = tmp_path / "events.jsonl"
    observed: list[tuple[dict[str, object], float]] = []
    script = (
        "import json,sys; data=sys.stdin.read(); "
        "print(json.dumps({'type':'stdin','bytes':len(data)}))"
    )

    returncode, timed_out = _run_and_capture(
        [sys.executable, "-c", script],
        env={},
        cwd=tmp_path,
        timeout_s=30,
        stdin_path=prompt_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timed_events_path=events_path,
        on_stdout_json=lambda raw, ts: observed.append((raw, ts)),
    )

    assert returncode == 0 and not timed_out
    assert observed[0][0] == {"type": "stdin", "bytes": len(prompt)}
    assert json.loads(stdout_path.read_text(encoding="utf-8"))["bytes"] == len(prompt)


def test_codex_llm_idle_timeout_is_independent(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    _write_codex_config(
        path,
        proxy_base_url="http://127.0.0.1:1234/v1",
        model="model",
        context_window=32768,
        llm_timeout_s=3000,
    )
    text = path.read_text(encoding="utf-8")
    assert 'sandbox_mode = "danger-full-access"' in text
    assert "stream_idle_timeout_ms = 3000000" in text
    assert "include_collaboration_mode_instructions = false" in text
    assert "model_reasoning_summary = \"none\"" in text
    assert "web_search = \"disabled\"" in text
    assert "[agents]\nenabled = false" in text
    assert "include_collaboration_mode =" not in text


def test_codex_bash_wrapper_records_real_wall_time_once(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    local_inputs = tmp_path / "localized_inputs"
    for path in (workspace, runtime / "home", local_inputs):
        path.mkdir(parents=True, exist_ok=True)
    wrapper = tmp_path / "host_only_wrapper.sh"
    shell_trace = tmp_path / "shell_events.jsonl"
    live_events = tmp_path / "live_events.jsonl"
    shell_trace.write_text("", encoding="utf-8")
    live_events.write_text("", encoding="utf-8")
    _write_codex_shell_wrapper(wrapper)
    command = _agent_sandbox_command(
        [
            "/bin/bash",
            "-c",
            'printf %s "$0" > argv0.txt; /bin/bash -c "sleep 0.20"',
        ],
        agent_bin=Path("/usr/bin/true"),
        workspace_dir=workspace,
        runtime_root=runtime,
        local_inputs_dir=local_inputs,
        require_bwrap=True,
        writable_paths=(shell_trace, live_events),
        codex_shell_wrapper=wrapper,
    )
    env = {
        "HOME": str(runtime / "home"),
        "PATH": "/usr/bin:/bin",
        "CODEX_THREAD_ID": "focused-thread",
        "CODEX_SESSION_ID": "focused-session",
        "MARS_CODEX_SHELL_TRACE": str(shell_trace),
        "MARS_CODEX_LIVE_EVENTS": str(live_events),
        "MARS_CODEX_REQUEST_ID_JSON": json.dumps("focused-request"),
    }
    result = subprocess.run(command, cwd=workspace, env=env, timeout=10, check=False)
    assert result.returncode == 0
    assert (workspace / "argv0.txt").read_text(encoding="utf-8") == "/bin/bash"
    shell_rows = [json.loads(line) for line in shell_trace.read_text().splitlines()]
    live_rows = [json.loads(line) for line in live_events.read_text().splitlines()]
    assert [row["event"] for row in shell_rows] == ["shell_start", "shell_end"]
    assert [row["event"] for row in live_rows] == ["tool_start", "tool_end"]
    assert float(shell_rows[-1]["duration_s"]) >= 0.18


def test_opencode_configured_shell_is_unique_live_and_measures_sleep(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    local_inputs = tmp_path / "localized_inputs"
    for path in (workspace, runtime / "home", local_inputs):
        path.mkdir(parents=True, exist_ok=True)
    wrapper = tmp_path / "host_only_opencode_wrapper.sh"
    shell_trace = tmp_path / "opencode_shell_events.jsonl"
    live_events = tmp_path / "live_events.jsonl"
    shell_trace.write_text("", encoding="utf-8")
    live_events.write_text("", encoding="utf-8")
    _write_opencode_shell_wrapper(wrapper)

    config_path = tmp_path / "opencode.json"
    _write_opencode_config(
        config_path,
        proxy_base_url="http://127.0.0.1:1234/v1",
        model="model",
        context_window=32768,
        max_output_tokens=1024,
        max_iterations=5,
        temperature=0.0,
        shell="/run/mars-opencode-shell/bash",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["shell"] == "/run/mars-opencode-shell/bash"
    assert Path(config["shell"]).name == "bash"

    # The outer system Bash represents CLI startup.  It must not be
    # interposed; only the explicitly configured ShellTool path is timed.
    command = _agent_sandbox_command(
        [
            "/bin/bash",
            "-c",
            "/run/mars-opencode-shell/bash -c 'sleep 0.35; pwd > tool_pwd.txt'",
        ],
        agent_bin=Path("/usr/bin/true"),
        workspace_dir=workspace,
        runtime_root=runtime,
        local_inputs_dir=local_inputs,
        require_bwrap=True,
        writable_paths=(shell_trace, live_events),
        opencode_shell_wrapper=wrapper,
    )
    env = {
        "HOME": str(runtime / "home"),
        "PATH": "/usr/bin:/bin",
        "MARS_OPENCODE_SHELL_TRACE": str(shell_trace),
        "MARS_OPENCODE_LIVE_EVENTS": str(live_events),
        "MARS_OPENCODE_REQUEST_ID_JSON": json.dumps("focused-request"),
        "MARS_AGENT_PHASE": "initial",
        "MARS_AGENT_PHASE_JSON": json.dumps("initial"),
    }
    proc = subprocess.Popen(
        command,
        cwd=workspace,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 5.0
    live_rows: list[dict] = []
    while time.monotonic() < deadline and proc.poll() is None:
        live_rows = [
            json.loads(line)
            for line in live_events.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(row.get("event") == "tool_start" for row in live_rows):
            break
        time.sleep(0.01)
    assert proc.poll() is None, "sleeping ShellTool completed before live-start assertion"
    assert [row["event"] for row in live_rows] == ["tool_start"], live_rows

    assert proc.wait(timeout=10) == 0
    shell_rows = [
        json.loads(line)
        for line in shell_trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    live_rows = [
        json.loads(line)
        for line in live_events.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["event"] for row in shell_rows] == ["shell_start", "shell_end"]
    assert [row["event"] for row in live_rows] == ["tool_start", "tool_end"]
    assert float(shell_rows[-1]["duration_s"]) >= 0.30
    assert shell_rows[0]["command_sha256"] == shell_rows[1]["command_sha256"]
    assert shell_rows[0]["cwd_sha256"] == shell_rows[1]["cwd_sha256"]
    assert (workspace / "tool_pwd.txt").read_text(encoding="utf-8").strip() == str(
        workspace
    )


def test_sandbox_exposes_only_this_requests_readonly_inputs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    localized = tmp_path / "_localized_inputs"
    benchmark = tmp_path / "benchmark"
    current_repo = localized / "paths" / "current-repo"
    foreign_repo = localized / "paths" / "foreign-repo"
    current_task = benchmark / "terminal-bench" / "tasks" / "current-task"
    foreign_task = benchmark / "terminal-bench" / "tasks" / "foreign-task"
    for path in (
        workspace,
        runtime,
        current_repo,
        foreign_repo,
        current_task,
        foreign_task,
    ):
        path.mkdir(parents=True, exist_ok=True)
    (current_repo / "visible.txt").write_text("current\n", encoding="utf-8")
    (foreign_repo / "secret.txt").write_text("foreign\n", encoding="utf-8")
    (current_task / "docker-compose.yaml").write_text("services: {}\n", encoding="utf-8")
    (foreign_task / "secret.txt").write_text("foreign\n", encoding="utf-8")

    task = {
        "payload": {"repo_cache_path": str(current_repo)},
        "workspace_init": [
            {
                "path": "tb_meta.json",
                "content": json.dumps({"task_dir": str(current_task)}),
            }
        ],
    }
    readonly_paths = task_readonly_paths(
        task,
        allowed_roots=(localized, benchmark),
    )
    command = _agent_sandbox_command(
        [
            "/bin/bash",
            "-c",
            'test -r "$1/visible.txt" && test -r "$2/docker-compose.yaml" '
            '&& test ! -e "$3" && test ! -e "$4"',
            "bash",
            str(current_repo),
            str(current_task),
            str(foreign_repo),
            str(foreign_task),
        ],
        agent_bin=Path("/usr/bin/true"),
        workspace_dir=workspace,
        runtime_root=runtime,
        local_inputs_dir=localized,
        readonly_paths=readonly_paths,
        require_bwrap=True,
    )

    result = subprocess.run(command, cwd=workspace, timeout=10, check=False)
    assert result.returncode == 0


def test_tool_is_not_orphaned_when_cli_emits_before_llm_stream_closes() -> None:
    llm_calls = [
        {
            "llm_call_id": "llm_0001",
            "response_id": "response-1",
            "start_time_s": 10.0,
            "end_time_s": 12.0,
        }
    ]
    tools = [
        {
            "tool_call_id": "tool-1",
            "tool_name": "bash",
            "start_time_s": 11.9,
            "end_time_s": 12.1,
            "status": "completed",
        }
    ]
    aligned, rounds, orphans = _openhands_tool_calls(tools, llm_calls)
    assert aligned[0]["emitter_llm_call_id"] == "llm_0001"
    assert [item["tool_call_id"] for item in rounds[0]["tools"]] == ["tool-1"]
    assert orphans == []
