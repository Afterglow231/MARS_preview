#!/usr/bin/env python3
"""Request-scoped Docker proxy for TerminalBench helpers.

The agent sandbox never receives the host Docker socket.  Instead, a tiny
``docker`` shim talks to this proxy over a request-local Unix socket.  The
proxy accepts only the exact compose/exec/cp shapes emitted by ``tb_task.py``
and restricts them to the request's compose project and container.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import signal
import socket
import socketserver
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


_WRAPPER = r'''#!/usr/bin/env python3
import base64
import json
import os
import socket
import sys

sock_path = os.environ.get("MARS_DOCKER_PROXY_SOCKET")
if not sock_path:
    sys.stderr.write("Docker is unavailable: MARS_DOCKER_PROXY_SOCKET is not set.\n")
    raise SystemExit(125)
payload = {
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "env": {k: v for k, v in os.environ.items() if k.startswith("T_BENCH_")},
}
wire = json.dumps(payload, separators=(",", ":")).encode("utf-8")
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect(sock_path)
    client.sendall(wire)
    client.shutdown(socket.SHUT_WR)
    chunks = []
    while True:
        chunk = client.recv(1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
reply = json.loads(b"".join(chunks))
sys.stdout.buffer.write(base64.b64decode(reply.get("stdout_b64") or ""))
sys.stderr.buffer.write(base64.b64decode(reply.get("stderr_b64") or ""))
raise SystemExit(int(reply.get("returncode", 125)))
'''


def _inside(path: Path, root: Path) -> bool:
    path = path.resolve(strict=False)
    root = root.resolve(strict=False)
    return path == root or root in path.parents


def _proc_start_time_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _sanitize(value: str) -> str:
    value = re.sub(r"[^a-z0-9_.-]+", "-", value.strip().lower())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "task"


def _read_terminalbench_meta(workspace_dir: Path) -> tuple[str, Path]:
    path = workspace_dir / "tb_meta.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_id = str(payload.get("task_id") or "").strip()
    task_dir_raw = str(payload.get("task_dir") or "").strip()
    if not task_id or not task_dir_raw:
        raise RuntimeError(f"invalid TerminalBench metadata: {path}")
    task_dir = Path(task_dir_raw).expanduser().resolve()
    compose = task_dir / "docker-compose.yaml"
    if not compose.is_file():
        compose = task_dir / "docker-compose.yml"
    if not compose.is_file():
        raise RuntimeError(f"TerminalBench compose file is missing: {task_dir}")
    return task_id, task_dir


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class ScopedDockerProxy:
    """Whitelist Docker operations for one TerminalBench request."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        runtime_root: Path,
        cleanup_descriptor_path: Path | None = None,
    ) -> None:
        self.workspace_dir = workspace_dir.resolve()
        self.runtime_root = runtime_root.resolve()
        self.task_id, self.task_dir = _read_terminalbench_meta(self.workspace_dir)
        self.compose_path = self.task_dir / "docker-compose.yaml"
        if not self.compose_path.is_file():
            self.compose_path = self.task_dir / "docker-compose.yml"
        digest = hashlib.sha1(str(self.workspace_dir).encode("utf-8", errors="ignore")).hexdigest()
        self.project = f"mas-tb-{digest[:12]}"
        self.container = f"{self.project}-{_sanitize(self.task_id)}-client"
        self.socket_path = self.runtime_root / "docker_proxy.sock"
        self.audit_path = self.runtime_root.parent / "docker_proxy_audit.jsonl"
        self.docker_config = self.runtime_root / "docker_config"
        self.cleanup_descriptor_path = (
            cleanup_descriptor_path.resolve()
            if cleanup_descriptor_path is not None
            else self.runtime_root.parent / "docker_cleanup_descriptor.json"
        )
        self._server: _ThreadingUnixServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._active: set[subprocess.Popen[bytes]] = set()
        self._active_commands: dict[int, dict[str, Any]] = {}
        self._last_compose_env: dict[str, str] = {}

    def install_wrapper(self, env: dict[str, str]) -> None:
        bin_dir = self.runtime_root / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        wrapper = bin_dir / "docker"
        wrapper.write_text(_WRAPPER, encoding="utf-8")
        wrapper.chmod(0o755)
        self.docker_config.mkdir(parents=True, exist_ok=True)
        env["MARS_DOCKER_PROXY_SOCKET"] = str(self.socket_path)
        safe_system_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        existing_path = env.get("PATH", "")
        preserved = [item for item in existing_path.split(os.pathsep) if item]
        for item in safe_system_path.split(os.pathsep):
            if item not in preserved:
                preserved.append(item)
        env["PATH"] = os.pathsep.join([str(bin_dir), *preserved])

    def __enter__(self) -> "ScopedDockerProxy":
        if not Path("/usr/bin/docker").is_file():
            raise RuntimeError("/usr/bin/docker is required for TerminalBench")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        outer = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                chunks: list[bytes] = []
                while True:
                    chunk = self.request.recv(1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if sum(map(len, chunks)) > 4 * 1024 * 1024:
                        outer._send_error(self.request, "Docker proxy request is too large")
                        return
                try:
                    payload = json.loads(b"".join(chunks))
                    reply = outer._handle(payload)
                except Exception as exc:
                    reply = outer._reply(126, b"", f"Docker command blocked: {exc}\n".encode())
                    outer._audit([], "blocked", str(exc))
                self.request.sendall(json.dumps(reply, separators=(",", ":")).encode("utf-8"))

        self._server = _ThreadingUnixServer(str(self.socket_path), Handler)
        os.chmod(self.socket_path, 0o600)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def stop(self) -> None:
        with self._lock:
            active = list(self._active)
        for proc in active:
            self._terminate(proc)
        cleanup_error: Exception | None = None
        try:
            self._cleanup_compose()
        except Exception as exc:
            cleanup_error = exc
        finally:
            if self._server is not None:
                self._server.shutdown()
                self._server.server_close()
                self._server = None
            if self._thread is not None:
                self._thread.join(timeout=5)
                self._thread = None
            self.socket_path.unlink(missing_ok=True)
        if cleanup_error is not None:
            raise cleanup_error

    @staticmethod
    def _reply(returncode: int, stdout: bytes, stderr: bytes) -> dict[str, Any]:
        return {
            "returncode": int(returncode),
            "stdout_b64": base64.b64encode(stdout).decode("ascii"),
            "stderr_b64": base64.b64encode(stderr).decode("ascii"),
        }

    def _send_error(self, sock: socket.socket, message: str) -> None:
        sock.sendall(json.dumps(self._reply(126, b"", (message + "\n").encode())).encode())

    def _audit(self, argv: list[str], decision: str, detail: str = "") -> None:
        row = {
            "ts": time.time(),
            "decision": decision,
            "argv": argv,
            "detail": detail,
            "project": self.project,
            "container": self.container,
        }
        with self._lock:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _validate(self, argv: list[str], cwd: str) -> str:
        if Path(cwd).resolve() != self.workspace_dir:
            raise ValueError(f"cwd must be request workspace: {cwd}")
        if argv == ["compose", "version"]:
            return "version"
        compose_prefix = [
            "compose", "-p", self.project, "-f", str(self.compose_path.resolve())
        ]
        if argv[: len(compose_prefix)] == compose_prefix:
            tail = argv[len(compose_prefix) :]
            if tail in (["build"], ["up", "-d"], ["down"]):
                return f"compose_{tail[0]}"
            raise ValueError(f"compose action is not allowed: {tail!r}")
        if argv and argv[0] == "exec":
            rest = argv[1:]
            if rest[:1] == ["-w"] and len(rest) >= 3:
                rest = rest[2:]
            if len(rest) == 4 and rest[0] == self.container and rest[1:3] == ["bash", "-lc"]:
                return "exec"
            raise ValueError("exec is restricted to bash -lc in the request container")
        if len(argv) == 3 and argv[0] == "cp":
            source = Path(argv[1].removesuffix("/.")).resolve(strict=False)
            destination = argv[2]
            if _inside(source, self.task_dir) and destination.startswith(f"{self.container}:/tests"):
                return "cp"
            raise ValueError("cp is restricted to task assets -> request container /tests")
        raise ValueError(f"Docker operation is not allowed: {argv[:4]!r}")

    def _base_env(self, forwarded: dict[str, str]) -> dict[str, str]:
        env = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": str(self.runtime_root / "home"),
            "DOCKER_CONFIG": str(self.docker_config),
        }
        env.update({str(k): str(v) for k, v in forwarded.items() if str(k).startswith("T_BENCH_")})
        return env

    def _validate_compose_security(self, env: dict[str, str]) -> None:
        command = [
            "/usr/bin/docker", "compose", "-p", self.project, "-f", str(self.compose_path),
            "config", "--format", "json",
        ]
        result = subprocess.run(
            command,
            cwd=self.workspace_dir,
            env=env,
            check=False,
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise ValueError(f"cannot validate compose file: {result.stderr.decode(errors='replace')[-500:]}")
        config = json.loads(result.stdout)
        for name, service in (config.get("services") or {}).items():
            if service.get("privileged"):
                raise ValueError(f"service {name} requests privileged mode")
            for key in ("pid", "ipc", "network_mode", "cgroup"):
                if str(service.get(key) or "").lower() == "host":
                    raise ValueError(f"service {name} requests host {key}")
            if service.get("devices"):
                raise ValueError(f"service {name} requests host devices")
            security_opts = [str(x).lower() for x in service.get("security_opt") or []]
            if any("unconfined" in value for value in security_opts):
                raise ValueError(f"service {name} disables confinement")
            capabilities = {str(x).upper() for x in service.get("cap_add") or []}
            if capabilities & {"ALL", "SYS_ADMIN", "SYS_PTRACE", "SYS_MODULE"}:
                raise ValueError(f"service {name} requests dangerous capabilities")
            for volume in service.get("volumes") or []:
                if not isinstance(volume, dict) or volume.get("type") != "bind":
                    continue
                source = Path(str(volume.get("source") or "")).resolve(strict=False)
                if not (_inside(source, self.workspace_dir) or _inside(source, self.task_dir)):
                    raise ValueError(f"service {name} bind mount escapes task/workspace: {source}")
                if source == Path("/run/docker.sock") or source == Path("/var/run/docker.sock"):
                    raise ValueError("Docker socket bind is forbidden")
            build = service.get("build")
            if isinstance(build, dict) and build.get("context"):
                context = Path(str(build["context"])).resolve(strict=False)
                if not _inside(context, self.task_dir):
                    raise ValueError(f"service {name} build context escapes task: {context}")

    def _handle(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("request must be a JSON object")
        argv = payload.get("argv")
        cwd = payload.get("cwd")
        forwarded = payload.get("env") or {}
        if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
            raise ValueError("argv must be a string list")
        if not isinstance(cwd, str) or not isinstance(forwarded, dict):
            raise ValueError("invalid cwd/env")
        operation = self._validate(argv, cwd)
        env = self._base_env(forwarded)
        if operation.startswith("compose_") and operation != "compose_down":
            self._validate_compose_security(env)
            with self._lock:
                self._last_compose_env = env
                self._write_cleanup_descriptor(env)
        command = ["/usr/bin/docker", *argv]
        self._audit(argv, "allowed", operation)
        proc = subprocess.Popen(
            command,
            cwd=self.workspace_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        with self._lock:
            self._active.add(proc)
            self._active_commands[proc.pid] = {
                "command": list(command),
                "start_time_ticks": _proc_start_time_ticks(proc.pid),
            }
            self._write_cleanup_descriptor(self._last_compose_env or env)
        try:
            stdout, stderr = proc.communicate()
        finally:
            with self._lock:
                self._active.discard(proc)
                self._active_commands.pop(proc.pid, None)
                self._write_cleanup_descriptor(self._last_compose_env or env)
        return self._reply(int(proc.returncode), stdout, stderr)

    @staticmethod
    def _terminate(proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _write_cleanup_descriptor(self, env: dict[str, str]) -> None:
        payload = {
            "schema": "mars_scoped_docker_cleanup_v1",
            "workspace_dir": str(self.workspace_dir),
            "task_dir": str(self.task_dir),
            "compose_path": str(self.compose_path.resolve()),
            "project": self.project,
            "t_bench_env": {
                key: value for key, value in env.items() if key.startswith("T_BENCH_")
            },
            "active_processes": [
                {"pid": pid, **details}
                for pid, details in sorted(self._active_commands.items())
            ],
        }
        self.cleanup_descriptor_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cleanup_descriptor_path.with_suffix(
            self.cleanup_descriptor_path.suffix + ".tmp"
        )
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(self.cleanup_descriptor_path)

    def _cleanup_compose(self, *, force: bool = False) -> dict[str, Any]:
        if not self._last_compose_env and not force:
            return {"status": "not_started", "returncode": None}
        env = self._last_compose_env or self._base_env({})
        command = [
            "/usr/bin/docker", "compose", "-p", self.project, "-f", str(self.compose_path), "down"
        ]
        try:
            result = subprocess.run(
                command,
                cwd=self.workspace_dir,
                env=env,
                check=False,
                capture_output=True,
                timeout=120,
            )
        except Exception as exc:
            payload = {
                "status": "error",
                "returncode": None,
                "error": f"{type(exc).__name__}: {exc}",
                "project": self.project,
            }
        else:
            payload = {
                "status": "completed" if result.returncode == 0 else "error",
                "returncode": int(result.returncode),
                "stdout": result.stdout.decode("utf-8", errors="replace")[-2000:],
                "stderr": result.stderr.decode("utf-8", errors="replace")[-2000:],
                "project": self.project,
            }
        cleanup_path = self.cleanup_descriptor_path.with_name("docker_cleanup.json")
        cleanup_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        if payload["status"] != "completed":
            raise RuntimeError(f"scoped Docker cleanup failed: {payload}")
        return payload


def cleanup_scoped_docker_project(
    *,
    workspace_dir: Path,
    runtime_root: Path,
    cleanup_descriptor_path: Path,
) -> dict[str, Any]:
    """Idempotently clean one request project from the parent process."""

    proxy = ScopedDockerProxy(
        workspace_dir=workspace_dir,
        runtime_root=runtime_root,
        cleanup_descriptor_path=cleanup_descriptor_path,
    )
    forwarded: dict[str, str] = {}
    active_processes: list[dict[str, Any]] = []
    if cleanup_descriptor_path.is_file():
        descriptor = json.loads(cleanup_descriptor_path.read_text(encoding="utf-8"))
        expected = {
            "workspace_dir": str(proxy.workspace_dir),
            "task_dir": str(proxy.task_dir),
            "compose_path": str(proxy.compose_path.resolve()),
            "project": proxy.project,
        }
        for key, value in expected.items():
            if descriptor.get(key) != value:
                raise RuntimeError(
                    f"Docker cleanup descriptor mismatch for {key}: "
                    f"{descriptor.get(key)!r} != {value!r}"
                )
        forwarded = {
            str(key): str(value)
            for key, value in (descriptor.get("t_bench_env") or {}).items()
            if str(key).startswith("T_BENCH_")
        }
        active_processes = list(descriptor.get("active_processes") or [])
    terminated_pids: list[int] = []
    termination_errors: list[str] = []
    for item in active_processes:
        try:
            pid = int(item.get("pid"))
            expected_command = [str(value) for value in item.get("command") or []]
            expected_start_time = int(item.get("start_time_ticks"))
        except (AttributeError, TypeError, ValueError) as exc:
            termination_errors.append(f"invalid active process descriptor: {item!r}: {exc}")
            continue
        if pid <= 1 or not expected_command or expected_command[0] != "/usr/bin/docker":
            termination_errors.append(f"unsafe active process descriptor: {item!r}")
            continue
        proc_path = Path(f"/proc/{pid}")
        if not proc_path.exists():
            continue
        try:
            actual_command = [
                part.decode("utf-8", errors="replace")
                for part in (proc_path / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except OSError as exc:
            termination_errors.append(f"cannot inspect Docker pid {pid}: {exc}")
            continue
        if not actual_command:
            continue
        if _proc_start_time_ticks(pid) != expected_start_time:
            termination_errors.append(
                f"Docker pid {pid} start time changed; refusing process-group signal"
            )
            continue
        if actual_command != expected_command:
            termination_errors.append(
                f"Docker pid {pid} command changed: {actual_command!r} != "
                f"{expected_command!r}"
            )
            continue
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        deadline = time.monotonic() + 5.0
        process_running = True
        while process_running and time.monotonic() < deadline:
            if not proc_path.exists():
                process_running = False
                break
            try:
                process_running = (proc_path / "stat").read_text().split()[2] != "Z"
            except (OSError, IndexError):
                process_running = False
            if not process_running:
                break
            time.sleep(0.05)
        if process_running:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        terminated_pids.append(pid)
    proxy.docker_config.mkdir(parents=True, exist_ok=True)
    proxy._last_compose_env = proxy._base_env(forwarded)
    result = proxy._cleanup_compose(force=True)
    result["terminated_recorded_pids"] = terminated_pids
    result["termination_errors"] = termination_errors
    cleanup_path = cleanup_descriptor_path.with_name("docker_cleanup.json")
    cleanup_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if termination_errors:
        raise RuntimeError(
            "scoped Docker process-group cleanup was incomplete: "
            + "; ".join(termination_errors)
        )
    return result
