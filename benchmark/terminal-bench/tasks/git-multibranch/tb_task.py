#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Terminal-Bench task helper for MAS replay/baselines.
#
# This script provides a small, deterministic interface for running commands
# inside the task's Docker sandbox and for running the official task tests.
#
# All names are derived from the current working directory (per-request
# workspace), so concurrent requests do not collide.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _run(cmd: list[str], *, env: Optional[dict[str, str]] = None, timeout_s: Optional[float] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env=env, check=False, text=True, capture_output=True, timeout=timeout_s)


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _sanitize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_.-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "task"


def _read_meta() -> dict:
    p = Path("tb_meta.json")
    if not p.exists():
        raise SystemExit("tb_meta.json not found in current directory")
    obj = json.loads(p.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _project_name() -> str:
    # Per-request workspace directory is unique; derive stable docker compose project name from it.
    h = _sha(str(Path.cwd().resolve()))
    return f"mas-tb-{h[:12]}"


def _container_name(task_id: str) -> str:
    return f"{_project_name()}-{_sanitize(task_id)}-client"


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _compose_env(meta: dict) -> dict[str, str]:
    task_id = str(meta.get("task_id") or "task")
    task_dir = Path(str(meta.get("task_dir") or "")).expanduser().resolve()
    logs_dir = Path(".tb_logs").resolve()
    agent_logs_dir = Path(".tb_agent_logs").resolve()
    _ensure_dirs(logs_dir, agent_logs_dir)

    # Use a shared image tag per task_id to reuse docker build cache across requests.
    image_name = f"mas-terminal-bench/{_sanitize(task_id)}:latest"

    env = os.environ.copy()
    env["T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME"] = image_name
    env["T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME"] = _container_name(task_id)
    env["T_BENCH_TEST_DIR"] = "/tests"
    env["T_BENCH_CONTAINER_LOGS_PATH"] = "/logs"
    env["T_BENCH_CONTAINER_AGENT_LOGS_PATH"] = "/agent-logs"
    env["T_BENCH_TASK_LOGS_PATH"] = str(logs_dir)
    env["T_BENCH_TASK_AGENT_LOGS_PATH"] = str(agent_logs_dir)

    # Make compose run relative paths in task_dir consistent.
    env["T_BENCH_TASK_DIR"] = str(task_dir)
    return env


def _require_cmd(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise SystemExit(f"{name} not found. terminal-bench requires Docker + docker compose.")
    return p


def _compose_base_cmd() -> list[str]:
    _require_cmd("docker")
    # Prefer `docker compose` (Compose v2 plugin). Fallback to legacy docker-compose.
    try:
        r = _run(["docker", "compose", "version"], timeout_s=5.0)
        if r.returncode == 0:
            return ["docker", "compose"]
    except Exception:
        pass
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    raise SystemExit("docker compose not found. Install Docker Compose v2 plugin or docker-compose.")


def _compose(task_dir: Path, args: list[str], *, env: dict[str, str], timeout_s: Optional[float]) -> subprocess.CompletedProcess:
    return _run(
        [
            *_compose_base_cmd(),
            "-p",
            _project_name(),
            "-f",
            str((task_dir / "docker-compose.yaml").resolve()),
            *args,
        ],
        env=env,
        timeout_s=timeout_s,
    )


def _ensure_up(*, meta: dict, timeout_s: Optional[float], rebuild: bool) -> str:
    task_dir = Path(str(meta.get("task_dir") or "")).expanduser().resolve()
    if not task_dir.exists():
        raise SystemExit(f"task_dir not found: {task_dir}")
    compose_path = task_dir / "docker-compose.yaml"
    if not compose_path.exists():
        raise SystemExit(f"docker-compose.yaml not found: {compose_path}")

    env = _compose_env(meta)
    task_id = str(meta.get("task_id") or "task")
    cname = _container_name(task_id)

    if rebuild:
        r = _compose(task_dir, ["build"], env=env, timeout_s=timeout_s)
        if r.returncode != 0:
            sys.stderr.write(r.stdout)
            sys.stderr.write(r.stderr)
            raise SystemExit(r.returncode)

    r = _compose(task_dir, ["up", "-d"], env=env, timeout_s=timeout_s)
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        sys.stderr.write(r.stderr)
        raise SystemExit(r.returncode)

    # Copy tests into /tests inside the container (matching terminal-bench harness).
    # We always refresh /tests to avoid stale state across attempts.
    r = _run(["docker", "exec", cname, "bash", "-lc", "rm -rf /tests && mkdir -p /tests"], timeout_s=timeout_s)
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        sys.stderr.write(r.stderr)
        raise SystemExit(r.returncode)

    run_tests = task_dir / "run-tests.sh"
    tests_dir = task_dir / "tests"
    if run_tests.exists():
        r = _run(["docker", "cp", str(run_tests), f"{cname}:/tests/run-tests.sh"], timeout_s=timeout_s)
        if r.returncode != 0:
            sys.stderr.write(r.stdout)
            sys.stderr.write(r.stderr)
            raise SystemExit(r.returncode)
        _run(["docker", "exec", cname, "bash", "-lc", "chmod +x /tests/run-tests.sh"], timeout_s=timeout_s)
    if tests_dir.exists():
        # Copy *contents* of tests_dir into /tests
        r = _run(["docker", "cp", str(tests_dir) + "/.", f"{cname}:/tests/"], timeout_s=timeout_s)
        if r.returncode != 0:
            sys.stderr.write(r.stdout)
            sys.stderr.write(r.stderr)
            raise SystemExit(r.returncode)

    return cname


def _exec_in_container(*, cname: str, cmd: str, workdir: str, timeout_s: Optional[float]) -> int:
    p = subprocess.run(
        ["docker", "exec", "-w", workdir, cname, "bash", "-lc", cmd],
        check=False,
        text=False,
    )
    return int(p.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--workdir", type=str, default="/app", help="Working directory inside container for exec/test")
    ap.add_argument("--timeout-s", type=float, default=None, help="Timeout for docker compose setup (seconds)")
    ap.add_argument("--rebuild", action="store_true", help="Force docker compose build before up")

    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("up", help="Start container and copy tests into /tests")
    p_exec = sub.add_parser("exec", help="Execute a single command inside the container")
    p_exec.add_argument("command", type=str, help="Command string to run inside container (bash -lc)")
    sub.add_parser("test", help="Run the official task tests (/tests/run-tests.sh) inside the container")
    sub.add_parser("down", help="docker compose down (stop and remove container)")
    args = ap.parse_args()

    _require_cmd("docker")

    meta = _read_meta()
    task_dir = Path(str(meta.get("task_dir") or "")).expanduser().resolve()
    env = _compose_env(meta)
    timeout_s = float(args.timeout_s) if args.timeout_s is not None else None

    if args.cmd == "up":
        _ensure_up(meta=meta, timeout_s=timeout_s, rebuild=bool(args.rebuild))
        return

    if args.cmd == "down":
        r = _compose(task_dir, ["down"], env=env, timeout_s=timeout_s)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        raise SystemExit(r.returncode)

    cname = _ensure_up(meta=meta, timeout_s=timeout_s, rebuild=bool(args.rebuild))

    if args.cmd == "exec":
        rc = _exec_in_container(cname=cname, cmd=str(args.command), workdir=str(args.workdir), timeout_s=None)
        raise SystemExit(rc)

    if args.cmd == "test":
        rc = _exec_in_container(cname=cname, cmd="bash /tests/run-tests.sh", workdir=str(args.workdir), timeout_s=None)
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
