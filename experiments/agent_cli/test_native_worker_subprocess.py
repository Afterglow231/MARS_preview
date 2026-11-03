#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import native_worker_entry
import native_request_worker
import run_workload as workload


def _args(*, timeout_s: float) -> argparse.Namespace:
    return argparse.Namespace(
        model="served-model",
        base_url="http://127.0.0.1:8000/v1",
        api_key="secret-not-for-json",
        max_output_tokens=128,
        max_iterations=4,
        temperature=0.0,
        top_p=None,
        gpu_timeout_s=timeout_s,
        llm_timeout_s=30.0,
        llm_num_retries=2,
        terminal_no_change_timeout_s=10,
        cpu_affinity=None,
    )


def _install_config(monkeypatch, tmp_path: Path) -> None:
    local_inputs = tmp_path / "local_inputs"
    local_inputs.mkdir()
    monkeypatch.setattr(
        workload,
        "_BACKEND_CONFIG",
        {
            "agent": "codex",
            "agent_bin": "/bin/true",
            "context_window": 4096,
            "local_inputs_dir": str(local_inputs),
            "benchmark_root": None,
            "require_bwrap": True,
        },
    )


def _invoke(tmp_path: Path, *, timeout_s: float) -> dict:
    return workload._agent_request_backend(
        args=_args(timeout_s=timeout_s),
        task={
            "request_id": "dataset-7",
            "benchmark": "terminalbench",
            "task_id": "task-7",
        },
        request_id="req_00000007",
        arrival_time_s=time.time(),
        workspace_root=tmp_path / "workspaces",
        request_run_dir=tmp_path / "request",
        runtime_bundle_dir=tmp_path / "unused_openhands_bundle",
    )


def test_request_runs_in_new_session_and_returns_json(
    monkeypatch, tmp_path: Path
) -> None:
    _install_config(monkeypatch, tmp_path)
    fake_entry = tmp_path / "fake_success.py"
    fake_entry.write_text(
        """
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--task-json', required=True)
parser.add_argument('--config-json', required=True)
parser.add_argument('--result-json', required=True)
args = parser.parse_args()
config = json.loads(Path(args.config_json).read_text())
result = {
    'trace': {
        'status': 'completed',
        'request_id': config['request_id'],
        'new_session': os.getpid() == os.getpgrp(),
        'api_key_received': os.environ.get('MARS_NATIVE_AGENT_API_KEY') == 'secret-not-for-json',
    },
    'events': [],
}
Path(args.result_json).write_text(json.dumps(result))
print('native worker stdout')
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(workload, "NATIVE_WORKER_ENTRY", fake_entry)

    result = _invoke(tmp_path, timeout_s=5.0)

    assert result["trace"]["status"] == "completed"
    assert result["trace"]["new_session"] is True
    assert result["trace"]["api_key_received"] is True
    request_dir = tmp_path / "request"
    assert "native worker stdout" in (request_dir / "stdout.txt").read_text()
    config_text = (request_dir / "native_worker_config.json").read_text()
    assert "secret-not-for-json" not in config_text
    assert "api_key" not in json.loads(config_text)


def test_native_entry_decodes_config_and_atomically_returns_result(
    monkeypatch, tmp_path: Path
) -> None:
    task_path = tmp_path / "task.json"
    config_path = tmp_path / "config.json"
    result_path = tmp_path / "returned.json"
    request_dir = tmp_path / "request"
    task = {"request_id": "dataset-8", "benchmark": "repobench"}
    task_path.write_text(json.dumps(task), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "agent": "opencode",
                "agent_bin": "/bin/true",
                "task_sha256": "abc",
                "request_id": "req_00000008",
                "mars_job_id": "req_00000008",
                "arrival_time_s": 123.5,
                "workspace_root": str(tmp_path / "workspaces"),
                "local_inputs_dir": str(tmp_path / "inputs"),
                "request_run_dir": str(request_dir),
                "model": "served-model",
                "base_url": "http://127.0.0.1:8000/v1",
                "context_window": 4096,
                "max_output_tokens": 128,
                "max_iterations": 4,
                "temperature": 0.0,
                "top_p": None,
                "timeout_s": 60.0,
                "llm_timeout_s": 30.0,
                "llm_num_retries": 2,
                "terminal_no_change_timeout_s": None,
                "require_bwrap": False,
                "cpu_affinity": "0-1",
                "benchmark_root": None,
            }
        ),
        encoding="utf-8",
    )
    captured: dict = {}

    def fake_run_request(**kwargs):
        captured.update(kwargs)
        return {"trace": {"status": "completed"}, "events": []}

    monkeypatch.setattr(native_worker_entry, "run_request", fake_run_request)
    monkeypatch.setenv("MARS_NATIVE_AGENT_API_KEY", "environment-only-key")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_worker_entry.py",
            "--task-json",
            str(task_path),
            "--config-json",
            str(config_path),
            "--result-json",
            str(result_path),
        ],
    )

    assert native_worker_entry.main() == 0
    assert captured["task"] == task
    assert captured["api_key"] == "environment-only-key"
    assert captured["agent_bin"] == Path("/bin/true")
    assert captured["request_run_dir"] == request_dir
    assert captured["terminal_no_change_timeout_s"] is None
    assert captured["cpu_affinity"] == "0-1"
    assert captured["allow_blocked_security_tasks"] is False
    assert json.loads(result_path.read_text())["trace"]["status"] == "completed"
    assert not result_path.with_name(result_path.name + ".entry.tmp").exists()


def test_native_worker_defense_rejects_blocked_task_without_workspace(
    tmp_path: Path,
) -> None:
    request_id = "req_00000017"
    workspace_root = tmp_path / "workspaces"
    request_run_dir = tmp_path / "request"
    result = native_request_worker.run_request(
        agent="opencode",
        agent_bin=tmp_path / "missing-opencode",
        task={
            "request_id": "terminalbench_sql-injection-attack",
            "benchmark": "terminalbench",
            "task_id": "sql-injection-attack",
        },
        task_sha256="blocked-task-sha",
        request_id=request_id,
        mars_job_id=request_id,
        arrival_time_s=time.time(),
        workspace_root=workspace_root,
        local_inputs_dir=tmp_path / "inputs",
        request_run_dir=request_run_dir,
        model="model",
        base_url="http://127.0.0.1:8000/v1",
        api_key="not-used",
        context_window=4096,
        max_output_tokens=128,
        max_iterations=4,
        temperature=0.0,
        timeout_s=30.0,
        require_bwrap=False,
    )

    assert result["trace"]["status"] == "policy_rejected"
    assert result["task_sha256"] == "blocked-task-sha"
    assert not (workspace_root / request_id).exists()
    assert (request_run_dir / "policy_decision.json").is_file()


def _process_is_running(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
    except FileNotFoundError:
        return False
    # A killed child can remain briefly as an init-owned zombie; it is no
    # longer executing and cannot retain a pipe or mutate benchmark state.
    return len(fields) < 3 or fields[2] != "Z"


def test_timeout_terminates_entire_worker_process_group(
    monkeypatch, tmp_path: Path
) -> None:
    _install_config(monkeypatch, tmp_path)
    fake_entry = tmp_path / "fake_hanging_worker.py"
    fake_entry.write_text(
        """
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

signal.signal(signal.SIGTERM, signal.SIG_IGN)
config_path = Path(sys.argv[sys.argv.index('--config-json') + 1])
config = json.loads(config_path.read_text())
child = subprocess.Popen([
    sys.executable,
    '-c',
    'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)',
])
Path(config['request_run_dir'], 'descendant_pid.txt').write_text(str(child.pid))
time.sleep(60)
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(workload, "NATIVE_WORKER_ENTRY", fake_entry)

    original_terminate = workload.openhands_workload._terminate_worker_process
    observed: dict[str, int | bool] = {}

    def terminate_quickly(proc) -> None:
        observed["pid"] = proc.pid
        observed["is_group_leader"] = os.getpgid(proc.pid) == proc.pid
        original_terminate(proc, terminate_grace_s=0.1)

    monkeypatch.setattr(
        workload.openhands_workload,
        "_terminate_worker_process",
        terminate_quickly,
    )

    result = _invoke(tmp_path, timeout_s=1.0)

    assert observed["is_group_leader"] is True
    assert result["trace"]["status"] == "worker_timeout"
    assert result["trace"]["error"]["type"] == "WorkerProcessTimeout"
    assert result["trace"]["error"]["timeout_s"] == 1.0
    descendant_pid = int((tmp_path / "request" / "descendant_pid.txt").read_text())
    deadline = time.monotonic() + 2.0
    while _process_is_running(descendant_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_is_running(descendant_pid)
