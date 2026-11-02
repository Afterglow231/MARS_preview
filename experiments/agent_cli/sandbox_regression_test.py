#!/usr/bin/env python3
"""CPU-only regression tests for agent process and filesystem containment."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from docker_proxy import (
    ScopedDockerProxy,
    _proc_start_time_ticks,
    cleanup_scoped_docker_project,
)
from request_worker import _agent_sandbox_command, _working_bwrap, _write_codex_rules
from run_workload import PINNED


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "forbidden"
    if isinstance(value, dict):
        return any(_contains_forbidden(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden(item) for item in value)
    return False


def _test_codex_rules(root: Path) -> None:
    codex_home = root / "codex_rules"
    rules_path = _write_codex_rules(codex_home)
    binary = Path(PINNED["codex"]["binary"])
    commands = (
        ["pkill", "-f", "conda"],
        ["/usr/bin/killall", "python"],
        ["/bin/kill", "1234"],
    )
    for command in commands:
        result = subprocess.run(
            [
                str(binary),
                "execpolicy",
                "check",
                "--rules",
                str(rules_path),
                "--",
                *command,
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=30,
        )
        payload = json.loads(result.stdout)
        assert _contains_forbidden(payload), (command, payload)


def _test_agent_namespace(root: Path, agent: str) -> None:
    unique = f"mars_conda_host_probe_{agent}_{uuid.uuid4().hex}"
    victim = subprocess.Popen(
        ["/bin/bash", "-c", f"exec -a {unique} /bin/sleep 120"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    agent_root = root / agent
    workspace = agent_root / "workspace"
    runtime = agent_root / "runtime"
    local_inputs = agent_root / "localized_inputs"
    home = runtime / "home"
    for path in (workspace, runtime, local_inputs, home):
        path.mkdir(parents=True, exist_ok=True)
    workspace_marker = workspace / "workspace_write_ok"
    child_marker = workspace / "own_child_kill_ok"
    outside_marker = agent_root / "outside_workspace_write"
    script = workspace / "probe.sh"
    script.write_text(
        "#!/bin/bash\n"
        "set -u\n"
        f"printf ok > {json.dumps(str(workspace_marker))}\n"
        "sleep 120 &\n"
        "child=$!\n"
        "kill \"${child}\"\n"
        "wait \"${child}\" 2>/dev/null || true\n"
        f"printf ok > {json.dumps(str(child_marker))}\n"
        "printf isolated > \"${HOME}/.bashrc\"\n"
        f"printf escaped > {json.dumps(str(outside_marker))} 2>/dev/null || true\n"
        f"/usr/bin/pkill -f {json.dumps(unique)} 2>/dev/null || true\n",
        encoding="utf-8",
    )
    env = {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
    }
    try:
        command = _agent_sandbox_command(
            ["/bin/bash", str(script)],
            agent_bin=Path(PINNED[agent]["binary"]),
            workspace_dir=workspace,
            runtime_root=runtime,
            local_inputs_dir=local_inputs,
            require_bwrap=True,
        )
        result = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        assert workspace_marker.read_text(encoding="utf-8") == "ok"
        assert child_marker.read_text(encoding="utf-8") == "ok"
        assert (home / ".bashrc").read_text(encoding="utf-8") == "isolated"
        assert not outside_marker.exists(), outside_marker
        assert victim.poll() is None, f"{agent} sandbox killed host PID {victim.pid}"
    finally:
        if victim.poll() is None:
            os.killpg(victim.pid, signal.SIGKILL)
        try:
            victim.wait(timeout=5)
        except subprocess.TimeoutExpired:
            victim.kill()


def _test_scoped_docker_proxy(root: Path) -> None:
    workspace = root / "docker_proxy" / "workspace"
    runtime = root / "docker_proxy" / "request" / "runtime"
    local_inputs = root / "docker_proxy" / "localized_inputs"
    task_dir = local_inputs / "proxy_task"
    for path in (workspace, runtime, local_inputs, task_dir):
        path.mkdir(parents=True, exist_ok=True)
    (task_dir / "docker-compose.yaml").write_text(
        "services:\n  client:\n    image: alpine:3.20\n",
        encoding="utf-8",
    )
    (workspace / "tb_meta.json").write_text(
        json.dumps({"task_id": "proxy-test", "task_dir": str(task_dir)}),
        encoding="utf-8",
    )
    proxy = ScopedDockerProxy(workspace_dir=workspace, runtime_root=runtime)
    env = {"HOME": str(runtime / "home"), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    original_path = env["PATH"]
    proxy.install_wrapper(env)
    assert env["PATH"].split(os.pathsep)[1 : 1 + len(original_path.split(os.pathsep))] == (
        original_path.split(os.pathsep)
    ), "Docker wrapper must preserve the pinned Node/runtime PATH"
    codex_command = _agent_sandbox_command(
        [str(PINNED["codex"]["binary"]), "--version"],
        agent_bin=Path(PINNED["codex"]["binary"]),
        workspace_dir=workspace,
        runtime_root=runtime,
        local_inputs_dir=local_inputs,
        readonly_paths=(task_dir,),
        require_bwrap=True,
    )
    codex_probe = subprocess.run(
        codex_command,
        cwd=workspace,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert codex_probe.returncode == 0, (
        codex_probe.returncode,
        codex_probe.stdout,
        codex_probe.stderr,
    )
    assert PINNED["codex"]["version"] in codex_probe.stdout
    script = (
        "docker compose version >/dev/null || exit 10; "
        "docker ps >/dev/null 2>&1; proxy_rc=$?; "
        "/usr/bin/docker ps >/dev/null 2>&1; direct_rc=$?; "
        "test \"${proxy_rc}\" -eq 126 -a \"${direct_rc}\" -ne 0"
    )
    command = _agent_sandbox_command(
        ["/bin/bash", "-lc", script],
        agent_bin=Path(PINNED["codex"]["binary"]),
        workspace_dir=workspace,
        runtime_root=runtime,
        local_inputs_dir=local_inputs,
        readonly_paths=(task_dir,),
        require_bwrap=True,
    )
    with proxy:
        result = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        foreign = subprocess.run(
            [str(runtime / "bin" / "docker"), "exec", "foreign", "bash", "-lc", "true"],
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )
        assert foreign.returncode == 126 and "blocked" in foreign.stderr.lower()
    cleanup = cleanup_scoped_docker_project(
        workspace_dir=workspace,
        runtime_root=runtime.parent / "parent_cleanup_runtime",
        cleanup_descriptor_path=runtime.parent / "docker_cleanup_descriptor.json",
    )
    assert cleanup["status"] == "completed" and cleanup["returncode"] == 0

    stray = subprocess.Popen(
        ["/usr/bin/docker", "events"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        descriptor_path = runtime.parent / "docker_cleanup_descriptor.json"
        descriptor_path.write_text(
            json.dumps(
                {
                    "schema": "mars_scoped_docker_cleanup_v1",
                    "workspace_dir": str(proxy.workspace_dir),
                    "task_dir": str(proxy.task_dir),
                    "compose_path": str(proxy.compose_path.resolve()),
                    "project": proxy.project,
                    "t_bench_env": {},
                    "active_processes": [
                        {
                            "pid": stray.pid,
                            "command": ["/usr/bin/docker", "events"],
                            "start_time_ticks": _proc_start_time_ticks(stray.pid),
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        cleanup = cleanup_scoped_docker_project(
            workspace_dir=workspace,
            runtime_root=runtime.parent / "parent_timeout_cleanup_runtime",
            cleanup_descriptor_path=descriptor_path,
        )
        assert stray.pid in cleanup["terminated_recorded_pids"]
    finally:
        if stray.poll() is None:
            os.killpg(stray.pid, signal.SIGKILL)
        stray.wait(timeout=5)


def main() -> None:
    node_bin = os.environ.get("MARS_NODE_BIN", "").strip()
    if node_bin:
        node_dir = str(Path(node_bin).expanduser().absolute().parent)
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if node_dir not in path_entries:
            os.environ["PATH"] = os.pathsep.join(
                [node_dir, *filter(None, path_entries)]
            )
    assert _working_bwrap(), "working bubblewrap is required (set MARS_BWRAP_BIN if needed)"
    with tempfile.TemporaryDirectory(prefix="mars_agent_sandbox_regression_") as temporary:
        root = Path(temporary)
        _test_codex_rules(root)
        for agent in sorted(PINNED):
            _test_agent_namespace(root, agent)
        _test_scoped_docker_proxy(root)
    print(
        "sandbox regression: Codex rules, PID/filesystem isolation, own-child kill, "
        "and scoped TerminalBench Docker proxy passed"
    )


if __name__ == "__main__":
    main()
