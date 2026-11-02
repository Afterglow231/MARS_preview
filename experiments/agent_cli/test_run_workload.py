#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

import run_workload as workload
from run_workload import (
    AGENT_RUNTIME_BUNDLE_FILES,
    _endpoint,
    _normalize_openai_base_url,
    _openhands_argv,
    _parse_args,
    _prepare_agent_runtime_bundle,
    _validate_infrastructure_artifacts,
)
from security_policy import blocked_security_task_decision


def _args(
    monkeypatch,
    *extra: str,
    base_url: str = "http://127.0.0.1:8000/v1",
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_workload.py",
            "--agent",
            "opencode",
            "--tasks",
            "tasks.jsonl",
            "--out-dir",
            "out",
            "--model",
            "model",
            "--base-url",
            base_url,
            *extra,
        ],
    )
    return _parse_args()


def test_top_p_defaults_to_none_and_is_omitted_from_openhands_argv(
    monkeypatch, tmp_path: Path
) -> None:
    args = _args(monkeypatch)
    argv = _openhands_argv(args, workspace_root=tmp_path)
    assert args.top_p is None
    assert "--top-p" not in argv


def test_explicit_top_p_is_forwarded_to_openhands_argv(
    monkeypatch, tmp_path: Path
) -> None:
    args = _args(monkeypatch, "--top-p", "0.75")
    argv = _openhands_argv(args, workspace_root=tmp_path)
    index = argv.index("--top-p")
    assert argv[index + 1] == "0.75"


def test_input_localization_options_are_forwarded_to_openhands(
    monkeypatch, tmp_path: Path
) -> None:
    shared_inputs = tmp_path / "shared_inputs"
    args = _args(
        monkeypatch,
        "--local-inputs-dir",
        str(shared_inputs),
        "--no-localize-task-inputs",
    )

    argv = _openhands_argv(args, workspace_root=tmp_path / "workspaces")

    index = argv.index("--local-inputs-dir")
    assert argv[index + 1] == str(shared_inputs.resolve())
    assert "--no-localize-task-inputs" in argv


def test_security_policy_override_is_explicit_and_forwarded(
    monkeypatch, tmp_path: Path
) -> None:
    default_args = _args(monkeypatch)
    assert default_args.allow_blocked_security_tasks is False
    assert "--allow-blocked-security-tasks" not in _openhands_argv(
        default_args, workspace_root=tmp_path / "default"
    )

    override_args = _args(monkeypatch, "--allow-blocked-security-tasks")
    assert override_args.allow_blocked_security_tasks is True
    assert "--allow-blocked-security-tasks" in _openhands_argv(
        override_args, workspace_root=tmp_path / "override"
    )

    blocked_task = {
        "benchmark": "terminalbench",
        "payload": {"task_id": "SPRING-MESSAGING-VUL"},
    }
    assert blocked_security_task_decision(blocked_task) is not None
    assert (
        blocked_security_task_decision(
            blocked_task, allow_blocked_security_tasks=True
        )
        is None
    )
    assert (
        blocked_security_task_decision(
            {"benchmark": "swebench", "task_id": "spring-messaging-vul"}
        )
        is None
    )


def test_native_backend_rejects_blocked_task_before_starting_worker(
    monkeypatch, tmp_path: Path
) -> None:
    args = _args(monkeypatch)
    workload._BACKEND_CONFIG.clear()
    workload._BACKEND_CONFIG.update(
        {
            "agent": "opencode",
            "agent_bin": "/bin/true",
            "context_window": 32768,
            "local_inputs_dir": str(tmp_path / "localized_inputs"),
            "require_bwrap": True,
            "benchmark_root": None,
        }
    )
    monkeypatch.setattr(
        workload.subprocess,
        "Popen",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("blocked request must not start a worker")
        ),
    )
    request_id = "req_00000016"
    request_run_dir = tmp_path / "_request_runs" / request_id
    workspace_root = tmp_path / "workspaces"
    task = {
        "schema": "mas_task_v1",
        "request_id": "terminalbench_spring-messaging-vul",
        "benchmark": "terminalbench",
        "task_id": "spring-messaging-vul",
        "payload": {"category": "security"},
    }

    result = workload._agent_request_backend(
        args=args,
        task=task,
        request_id=request_id,
        arrival_time_s=workload.openhands_workload.now_s(),
        workspace_root=workspace_root,
        request_run_dir=request_run_dir,
        runtime_bundle_dir=tmp_path / "runtime_bundle",
    )

    trace = result["trace"]
    assert trace["status"] == "policy_rejected"
    assert trace["workspace_materialized"] is False
    assert not (workspace_root / request_id).exists()
    assert not (request_run_dir / "native_worker_config.json").exists()
    policy = json.loads((request_run_dir / "policy_decision.json").read_text())
    assert policy["decision"] == "reject"
    assert policy["task_id"] == "spring-messaging-vul"
    assert _validate_infrastructure_artifacts(
        [trace], out_dir=tmp_path, require_sandbox=True
    ) == []


def test_openhands_backend_uses_the_same_security_admission_policy(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        workload.openhands_workload.subprocess,
        "Popen",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("blocked request must not start OpenHands")
        ),
    )
    result = workload.openhands_workload._run_request_subprocess(
        args=Namespace(allow_blocked_security_tasks=False, model="model"),
        task={
            "request_id": "terminalbench_sql-injection-attack",
            "benchmark": "terminalbench",
            "task_id": "sql-injection-attack",
        },
        request_id="req_00000017",
        arrival_time_s=workload.openhands_workload.now_s(),
        workspace_root=tmp_path / "workspaces",
        request_run_dir=tmp_path / "request",
        runtime_bundle_dir=tmp_path / "runtime",
    )

    assert result["trace"]["status"] == "policy_rejected"
    assert result["trace"]["workspace_materialized"] is False


def test_worker_failure_does_not_report_unknown_durations_as_zero(
    monkeypatch, tmp_path: Path
) -> None:
    args = _args(monkeypatch)
    workload._BACKEND_CONFIG.clear()
    workload._BACKEND_CONFIG.update(
        {
            "agent": "opencode",
            "agent_bin": "/bin/true",
            "context_window": 32768,
            "local_inputs_dir": str(tmp_path / "localized_inputs"),
            "require_bwrap": True,
            "benchmark_root": None,
        }
    )

    class FailedProcess:
        returncode = 17

        def communicate(self, timeout=None):
            return "", "intentional worker failure"

    monkeypatch.setattr(workload.subprocess, "Popen", lambda *a, **k: FailedProcess())
    task = {
        "schema": "mas_task_v1",
        "request_id": "failure-contract",
        "benchmark": "smoke",
        "task_id": "failure-contract",
        "payload": {"text": "test"},
    }

    result = workload._agent_request_backend(
        args=args,
        task=task,
        request_id="request-failure-contract",
        arrival_time_s=workload.openhands_workload.now_s(),
        workspace_root=tmp_path / "workspaces",
        request_run_dir=tmp_path / "request_run",
        runtime_bundle_dir=tmp_path / "runtime_bundle",
    )

    trace = result["trace"]
    assert trace["status"] == "worker_error"
    for key in (
        "queue_wait_s",
        "gpu_total_s",
        "tool_total_s",
        "tool_wait_total_s",
        "wait_total_s",
    ):
        assert trace[key] is None


def test_infrastructure_validation_reconciles_raw_backend_attempts(
    tmp_path: Path,
) -> None:
    request_dir = tmp_path / "_request_runs" / "request-1"
    request_dir.mkdir(parents=True)
    trace = {
        "request_id": "request-1",
        "status": "completed",
        "native_session_id": "session-1",
        "outer_sandbox": "bubblewrap_pid_filesystem",
        "llm_calls": [
            {
                "round_id": "round-1",
                "status": "completed",
                "end_time_s": 3.0,
            }
        ],
        "backend_attempts": [
            {
                "backend_attempt_id": "backend_0001",
                "status": "completed",
                "http_status": 200,
                "end_time_s": 2.0,
            }
        ],
    }
    (request_dir / "result.json").write_text(
        '{"trace":{"request_id":"request-1"}}\n', encoding="utf-8"
    )
    raw_path = request_dir / "backend_attempts.jsonl"
    raw_path.write_text(
        '{"event":"backend_attempt_started","backend_attempt_id":"backend_0001"}\n'
        '{"event":"backend_attempt_ended","backend_attempt_id":"backend_0001",'
        '"http_status":200}\n',
        encoding="utf-8",
    )

    assert _validate_infrastructure_artifacts(
        [trace], out_dir=tmp_path, require_sandbox=True
    ) == []

    trace["backend_attempts"] = []
    errors = _validate_infrastructure_artifacts(
        [trace], out_dir=tmp_path, require_sandbox=True
    )
    assert any("raw/projected backend IDs differ" in error for error in errors)


def test_agent_runtime_bundle_is_a_hashed_frozen_copy(tmp_path: Path) -> None:
    bundle_dir, hashes = _prepare_agent_runtime_bundle(tmp_path)

    assert set(hashes) == set(AGENT_RUNTIME_BUNDLE_FILES)
    for name, expected_hash in hashes.items():
        assert (bundle_dir / name).is_file()
        assert len(expected_hash) == 64


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000/v1"),
        ("http://127.0.0.1:8000/", "http://127.0.0.1:8000/v1"),
        ("http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000/v1"),
        ("https://backend.example/prefix", "https://backend.example/prefix/v1"),
        ("https://backend.example/prefix/v1", "https://backend.example/prefix/v1"),
    ],
)
def test_openai_base_url_normalization(raw: str, normalized: str) -> None:
    assert _normalize_openai_base_url(raw) == normalized
    assert _endpoint(raw, "/models") == normalized + "/models"
    assert _endpoint(raw, "/chat/completions") == normalized + "/chat/completions"


def test_bare_base_url_is_forwarded_consistently_to_openhands_and_workers(
    monkeypatch, tmp_path: Path
) -> None:
    args = _args(monkeypatch, base_url="http://127.0.0.1:8000")
    argv = _openhands_argv(args, workspace_root=tmp_path)

    assert args.base_url == "http://127.0.0.1:8000/v1"
    base_index = argv.index("--base-url")
    assert argv[base_index + 1] == args.base_url


@pytest.mark.parametrize(
    "invalid",
    [
        "127.0.0.1:8000",
        "ftp://127.0.0.1:8000",
        "http://127.0.0.1:8000?route=v1",
        "http://127.0.0.1:8000/#fragment",
    ],
)
def test_invalid_openai_base_url_is_rejected(invalid: str) -> None:
    with pytest.raises(ValueError):
        _normalize_openai_base_url(invalid)


def test_existing_vllm_wrapper_normalizes_before_warmup_and_adapter(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[2]
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        '{"schema":"mas_task_v1","request_id":"url-gate",'
        '"benchmark":"smoke","task_id":"url-gate",'
        '"payload":{"text":"url gate"}}\n',
        encoding="utf-8",
    )
    available_cpus = os.sched_getaffinity(0)
    cpu = str(min(available_cpus))
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "USE_EXISTING_VLLM": "1",
        "BASE_URL": "http://127.0.0.1:8123",
        "AGENTS": "opencode",
        "TASKS_PATH": str(tasks),
        "RUN_DIR": str(tmp_path / "out"),
        "MARS_DIR_OVERRIDE": str(repo),
        "MARS_PYTHON": sys.executable,
        "CPU_SERVER": cpu,
        "CPU_AGENT": cpu,
        "CPU_GPU": cpu,
        "CPU_TOOL": cpu,
    }
    result = subprocess.run(
        ["bash", str(repo / "experiments/scripts/run_agent_cli_local.sh")],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "BASE_URL=http://127.0.0.1:8123/v1" in result.stdout
    assert "--base-url http://127.0.0.1:8123/v1" in result.stdout


def test_launcher_passes_mars_policy_to_engine(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[2]
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        '{"schema":"mas_task_v1","request_id":"policy-gate",'
        '"benchmark":"smoke","task_id":"policy-gate",'
        '"payload":{"text":"policy gate"}}\n',
        encoding="utf-8",
    )
    cpu = str(min(os.sched_getaffinity(0)))
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "AGENTS": "codex",
        "TASKS_PATH": str(tasks),
        "RUN_DIR": str(tmp_path / "out"),
        "MARS_DIR_OVERRIDE": str(repo),
        "MARS_PYTHON": sys.executable,
        "SCHEDULING_POLICY": "mars",
        "CPU_SERVER": cpu,
        "CPU_AGENT": cpu,
        "CPU_GPU": cpu,
        "CPU_TOOL": cpu,
        "REQUIRE_DISJOINT_SERVER_AGENT_CPU": "0",
    }

    result = subprocess.run(
        ["bash", str(repo / "experiments/scripts/run_agent_cli_local.sh")],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "--scheduling-policy mars --additional-config" in result.stdout
