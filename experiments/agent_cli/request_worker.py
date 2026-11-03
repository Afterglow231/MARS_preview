#!/usr/bin/env python3
"""Run one mas_task_v1 request through OpenCode or Codex."""

from __future__ import annotations

import json
import os
import selectors
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from docker_proxy import ScopedDockerProxy
from event_normalizer import (
    LiveToolEventNormalizer,
    OpenCodeTimingIntegrityError,
    build_dynamic_graph,
    load_jsonl,
    normalize_codex,
    normalize_llm_calls,
    normalize_opencode,
)
from sandbox_paths import task_readonly_paths
from trace_proxy import TraceProxy


_BWRAP_CHECK_LOCK = threading.Lock()
_BWRAP_AVAILABLE: bool | None = None
_BWRAP_CHECKED_PATH: str | None = None


OPENHANDS_DIR = Path(
    os.environ.get(
        "MARS_OPENHANDS_DIR",
        str(Path(__file__).resolve().parents[1] / "openhands"),
    )
).expanduser().resolve()
if str(OPENHANDS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(OPENHANDS_DIR))

from request_bootstrap import (  # noqa: E402
    build_user_request,
    build_verification_nudge,
    materialize_workspace_init,
    requires_tool_verification,
)


_VERIFICATION_TOOL_EQUIVALENTS = {
    # OpenHands ``terminal``.
    "command_execution": "terminal",
    "bash": "terminal",
    "shell": "terminal",
    "terminal": "terminal",
    # OpenHands ``file_editor``.
    "file_change": "file_editor",
    "apply_patch": "file_editor",
    "edit": "file_editor",
    "write": "file_editor",
    "read": "file_editor",
    "patch": "file_editor",
    # OpenHands exposes these as their own verification-capable tools.
    "browser": "browser",
    "web_search": "browser",
    "websearch": "browser",
    "webfetch": "browser",
    "glob": "glob",
    "grep": "grep",
}


def _llm_retry_policy(agent: str, requested_retries: int) -> dict[str, Any]:
    """Describe the retry policy the pinned native frontend really applies."""

    requested = max(0, int(requested_retries))
    if agent == "codex":
        return {
            "frontend": "codex",
            "requested_retries": requested,
            "request_max_retries": requested,
            "stream_max_retries": requested,
            "effective_matches_requested": True,
            "source": "request_local_codex_config",
        }
    if agent == "opencode":
        return {
            "frontend": "opencode",
            "requested_retries": requested,
            "sdk_max_retries": 0,
            "outer_max_retries": 5,
            "max_total_attempts": 6,
            "effective_matches_requested": requested == 5,
            "source": "stock_opencode_v1.18.23_fixed_policy",
        }
    raise ValueError(f"unsupported agent: {agent}")


def _verification_tool_kind(tool_name: Any) -> str | None:
    return _VERIFICATION_TOOL_EQUIVALENTS.get(str(tool_name or "").strip().lower())


def _verification_tool_count(tool_calls: list[dict[str, Any]]) -> int:
    return sum(1 for item in tool_calls if _verification_tool_kind(item.get("tool_name")))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _build_prompt(task: dict[str, Any], *, agent: str, max_iterations: int) -> str:
    # Keep the user-visible request byte-for-byte aligned with the maintained
    # OpenHands runner. Iteration enforcement belongs in TraceProxy rather than
    # in an agent-specific prompt suffix.
    del agent, max_iterations
    return build_user_request(task)


def _runtime_env(request_run_dir: Path, workspace_dir: Path, api_key: str) -> dict[str, str]:
    runtime_root = request_run_dir / "runtime"
    paths = {
        "home": runtime_root / "home",
        "tmp": runtime_root / "tmp",
        "cache": runtime_root / "cache",
        "config": runtime_root / "config",
        "data": runtime_root / "data",
        "state": runtime_root / "state",
        "codex": runtime_root / "codex_home",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    allowed = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "USER", "LOGNAME", "SHELL")
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    node_binary = _node_binary()
    if node_binary is not None:
        path_entries = env.get("PATH", "").split(os.pathsep)
        node_dir = str(node_binary.parent)
        if node_dir not in path_entries:
            env["PATH"] = os.pathsep.join([node_dir, *filter(None, path_entries)])
    env.update(
        {
            "HOME": str(paths["home"]),
            "TMPDIR": str(paths["tmp"]),
            "TMP": str(paths["tmp"]),
            "TEMP": str(paths["tmp"]),
            "XDG_CACHE_HOME": str(paths["cache"]),
            "XDG_CONFIG_HOME": str(paths["config"]),
            "XDG_DATA_HOME": str(paths["data"]),
            "XDG_STATE_HOME": str(paths["state"]),
            "CODEX_HOME": str(paths["codex"]),
            "MARS_AGENT_API_KEY": api_key,
            "NO_PROXY": "127.0.0.1,localhost",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
            "OPENCODE_DISABLE_AUTOCOMPACT": "1",
            "PWD": str(workspace_dir),
        }
    )
    return env


def _opencode_command(
    *,
    binary: Path,
    config_path: Path,
    workspace_dir: Path,
    model: str,
    session_id: str | None = None,
) -> list[str]:
    del config_path
    command = [
        str(binary),
        "run",
        "--pure",
        "--format",
        "json",
        "--auto",
        "--agent",
        "build",
        "--model",
        f"mars/{model}",
        "--title",
        workspace_dir.name,
        "--dir",
        str(workspace_dir),
    ]
    if session_id:
        command.extend(["--session", session_id])
    return command


def _write_opencode_config(
    path: Path,
    *,
    proxy_base_url: str,
    model: str,
    context_window: int,
    max_output_tokens: int,
    max_iterations: int,
    temperature: float,
    shell: str | None = None,
) -> None:
    payload = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"mars/{model}",
        "small_model": f"mars/{model}",
        "autoupdate": False,
        "share": "disabled",
        "provider": {
            "mars": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "MARS vLLM",
                "options": {
                    "baseURL": proxy_base_url,
                    "apiKey": "{env:MARS_AGENT_API_KEY}",
                },
                "models": {
                    model: {
                        "name": model,
                        "limit": {"context": context_window, "output": max_output_tokens},
                    }
                },
            }
        },
        "agent": {
            "build": {
                "steps": max_iterations,
                "temperature": temperature,
            }
        },
    }
    if shell is not None:
        payload["shell"] = shell
    _write_json(path, payload)


def _write_codex_config(
    path: Path,
    *,
    proxy_base_url: str,
    model: str,
    context_window: int,
    llm_timeout_s: float,
    llm_num_retries: int = 3,
    sandbox_mode: str = "danger-full-access",
    enable_subagents: bool = False,
) -> None:
    idle_ms = max(300_000, int(llm_timeout_s * 1000))
    content = f'''model = {json.dumps(model)}
model_provider = "mars"
model_context_window = {int(context_window)}
model_auto_compact_token_limit = {max(1024, int(context_window * 0.90))}
approval_policy = "never"
sandbox_mode = {json.dumps(sandbox_mode)}
include_apps_instructions = false
include_collaboration_mode_instructions = {str(bool(enable_subagents)).lower()}
include_permissions_instructions = true
model_reasoning_summary = "none"
web_search = "disabled"

[agents]
enabled = {str(bool(enable_subagents)).lower()}

[model_providers.mars]
name = "MARS vLLM"
base_url = {json.dumps(proxy_base_url)}
env_key = "MARS_AGENT_API_KEY"
wire_api = "responses"
requires_openai_auth = false
request_max_retries = {max(0, int(llm_num_retries))}
stream_max_retries = {max(0, int(llm_num_retries))}
stream_idle_timeout_ms = {idle_ms}
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_codex_rules(codex_home: Path) -> Path:
    """Install defense-in-depth rules for host-wide process-control commands."""
    rules_path = codex_home / "rules" / "default.rules"
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    commands = (
        "pkill",
        "/usr/bin/pkill",
        "killall",
        "/usr/bin/killall",
        "kill",
        "/bin/kill",
        "/usr/bin/kill",
    )
    rules = "\n".join(
        "prefix_rule("
        f"pattern=[{json.dumps(command)}], "
        'decision="forbidden", '
        'justification="Process control is confined to the request sandbox; terminate only a directly spawned child."'
        ")"
        for command in commands
    )
    rules_path.write_text(rules + "\n", encoding="utf-8")
    return rules_path


def _write_codex_shell_wrapper(path: Path) -> None:
    """Write the Bash interposer used to time Codex command execution.

    Codex buffers its JSON ``item.started``/``item.completed`` events, so their
    host capture timestamps do not describe command wall time.  The outer
    bubblewrap namespace overlays this script on both Bash paths and exposes the
    real binary at ``/run/mars-agent-shell/bash.real``.  Codex injects
    ``CODEX_THREAD_ID`` and ``CODEX_SESSION_ID`` only for model-reachable tool
    processes; startup shell-snapshot probes are therefore passed through.
    """
    content = r'''#!/run/mars-agent-shell/bash.real
real_bash=/run/mars-agent-shell/bash.real

passthrough() {
  exec -a "$0" "${real_bash}" "$@"
}

if [[ "${MARS_CODEX_SHELL_WRAPPER_ACTIVE:-}" == 1 \
      || -z "${CODEX_THREAD_ID:-}" \
      || -z "${CODEX_SESSION_ID:-}" \
      || -z "${MARS_CODEX_SHELL_TRACE:-}" \
      || -z "${MARS_CODEX_LIVE_EVENTS:-}" ]]; then
  passthrough "$@"
fi

export MARS_CODEX_SHELL_WRAPPER_ACTIVE=1
start_ns="$(/usr/bin/date +%s%N 2>/dev/null)" || passthrough "$@"
if [[ ! "${start_ns}" =~ ^[0-9]+$ ]]; then
  passthrough "$@"
fi
native_shell_call_id="codex_shell_${BASHPID}_${RANDOM}_${start_ns}"
agent_phase="${MARS_AGENT_PHASE:-initial}"
shell_call_id="${agent_phase}:${native_shell_call_id}"
start_ts="$((start_ns / 1000000000)).$(printf '%09d' "$((start_ns % 1000000000))")"
printf '{"event":"shell_start","ts":%s,"phase":%s,"shell_call_id":"%s","native_shell_call_id":"%s","argv_count":%d}\n' \
  "${start_ts}" "${MARS_AGENT_PHASE_JSON:-\"initial\"}" "${shell_call_id}" "${native_shell_call_id}" "$#" >>"${MARS_CODEX_SHELL_TRACE}" 2>/dev/null || true
printf '{"ts":%s,"event":"tool_start","request_id":%s,"phase":%s,"tool_call_id":"%s","native_tool_call_id":"%s","tool_name":"command_execution","wait_s":0.0,"timing_source":"codex_bash_wrapper"}\n' \
  "${start_ts}" "${MARS_CODEX_REQUEST_ID_JSON}" "${MARS_AGENT_PHASE_JSON:-\"initial\"}" "${shell_call_id}" "${native_shell_call_id}" >>"${MARS_CODEX_LIVE_EVENTS}" 2>/dev/null || true

# For the normal Codex ``bash -c SCRIPT`` form, supplying the original wrapper
# path as Bash's command-name argument preserves the original value of ``$0``.
if [[ "$#" == 2 && ( "$1" == -c || "$1" == -lc ) ]]; then
  "${real_bash}" "$1" "$2" "$0"
else
  "${real_bash}" "$@"
fi
exit_code=$?
end_ns="$(/usr/bin/date +%s%N 2>/dev/null)"
if [[ ! "${end_ns}" =~ ^[0-9]+$ ]]; then
  exit "${exit_code}"
fi
end_ts="$((end_ns / 1000000000)).$(printf '%09d' "$((end_ns % 1000000000))")"
duration_ns=$((end_ns - start_ns))
duration_s="$((duration_ns / 1000000000)).$(printf '%09d' "$((duration_ns % 1000000000))")"
status=completed
if [[ "${exit_code}" != 0 ]]; then
  status=failed
fi
printf '{"event":"shell_end","ts":%s,"phase":%s,"shell_call_id":"%s","native_shell_call_id":"%s","duration_s":%s,"exit_code":%d,"status":"%s"}\n' \
  "${end_ts}" "${MARS_AGENT_PHASE_JSON:-\"initial\"}" "${shell_call_id}" "${native_shell_call_id}" "${duration_s}" "${exit_code}" "${status}" >>"${MARS_CODEX_SHELL_TRACE}" 2>/dev/null || true
printf '{"ts":%s,"event":"tool_end","request_id":%s,"phase":%s,"tool_call_id":"%s","native_tool_call_id":"%s","tool_name":"command_execution","duration_s":%s,"status":"%s","exit_code":%d,"timing_source":"codex_bash_wrapper"}\n' \
  "${end_ts}" "${MARS_CODEX_REQUEST_ID_JSON}" "${MARS_AGENT_PHASE_JSON:-\"initial\"}" "${shell_call_id}" "${native_shell_call_id}" "${duration_s}" "${status}" "${exit_code}" >>"${MARS_CODEX_LIVE_EVENTS}" 2>/dev/null || true
exit "${exit_code}"
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o500)


def _write_opencode_shell_wrapper(path: Path) -> None:
    """Write the configured OpenCode shell that emits true bash wall time.

    Unlike the Codex interposer, this wrapper is mounted only at the unique
    configured-shell path ``/run/mars-opencode-shell/bash``.  OpenCode startup
    keeps using the system shell, while ShellTool alone invokes this basename-
    preserving path.  Command and resolved-cwd hashes allow the delayed native
    JSON part to be matched without recording command text in the timing log.
    """

    content = r'''#!/run/mars-opencode-shell/bash.real
real_bash=/run/mars-opencode-shell/bash.real

passthrough() {
  exec -a "$0" "${real_bash}" "$@"
}

if [[ "${MARS_OPENCODE_SHELL_WRAPPER_ACTIVE:-}" == 1 \
      || -z "${MARS_OPENCODE_SHELL_TRACE:-}" \
      || -z "${MARS_OPENCODE_LIVE_EVENTS:-}" \
      || -z "${MARS_OPENCODE_REQUEST_ID_JSON:-}" \
      || "$#" != 2 \
      || "$1" != -c ]]; then
  passthrough "$@"
fi

command_sha256="$(printf '%s' "$2" | /usr/bin/sha256sum 2>/dev/null)" || passthrough "$@"
command_sha256="${command_sha256%% *}"
cwd_text="$(/usr/bin/pwd -P 2>/dev/null)" || passthrough "$@"
cwd_sha256="$(printf '%s' "${cwd_text}" | /usr/bin/sha256sum 2>/dev/null)" || passthrough "$@"
cwd_sha256="${cwd_sha256%% *}"
if [[ ! "${command_sha256}" =~ ^[0-9a-f]{64}$ \
      || ! "${cwd_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  passthrough "$@"
fi

export MARS_OPENCODE_SHELL_WRAPPER_ACTIVE=1
start_ns="$(/usr/bin/date +%s%N 2>/dev/null)" || passthrough "$@"
if [[ ! "${start_ns}" =~ ^[0-9]+$ ]]; then
  passthrough "$@"
fi
native_shell_call_id="opencode_shell_${BASHPID}_${RANDOM}_${start_ns}"
agent_phase="${MARS_AGENT_PHASE:-initial}"
shell_call_id="${agent_phase}:${native_shell_call_id}"
start_ts="$((start_ns / 1000000000)).$(printf '%09d' "$((start_ns % 1000000000))")"
printf '{"event":"shell_start","ts":%s,"phase":%s,"shell_call_id":"%s","native_shell_call_id":"%s","command_sha256":"%s","cwd_sha256":"%s"}\n' \
  "${start_ts}" "${MARS_AGENT_PHASE_JSON:-\"initial\"}" "${shell_call_id}" "${native_shell_call_id}" "${command_sha256}" "${cwd_sha256}" >>"${MARS_OPENCODE_SHELL_TRACE}" 2>/dev/null || true
printf '{"ts":%s,"event":"tool_start","request_id":%s,"phase":%s,"tool_call_id":"%s","native_tool_call_id":"%s","tool_name":"bash","wait_s":0.0,"timing_source":"opencode_configured_shell_wrapper","command_sha256":"%s","cwd_sha256":"%s"}\n' \
  "${start_ts}" "${MARS_OPENCODE_REQUEST_ID_JSON}" "${MARS_AGENT_PHASE_JSON:-\"initial\"}" "${shell_call_id}" "${native_shell_call_id}" "${command_sha256}" "${cwd_sha256}" >>"${MARS_OPENCODE_LIVE_EVENTS}" 2>/dev/null || true

# Node invokes the configured shell as ``bash -c COMMAND``.  Supplying the
# wrapper path as command-name preserves the shell's normal $0 semantics.
"${real_bash}" -c "$2" "$0"
exit_code=$?
end_ns="$(/usr/bin/date +%s%N 2>/dev/null)"
if [[ ! "${end_ns}" =~ ^[0-9]+$ ]]; then
  exit "${exit_code}"
fi
end_ts="$((end_ns / 1000000000)).$(printf '%09d' "$((end_ns % 1000000000))")"
duration_ns=$((end_ns - start_ns))
duration_s="$((duration_ns / 1000000000)).$(printf '%09d' "$((duration_ns % 1000000000))")"
status=completed
if [[ "${exit_code}" != 0 ]]; then
  status=failed
fi
printf '{"event":"shell_end","ts":%s,"phase":%s,"shell_call_id":"%s","native_shell_call_id":"%s","command_sha256":"%s","cwd_sha256":"%s","duration_s":%s,"exit_code":%d,"status":"%s"}\n' \
  "${end_ts}" "${MARS_AGENT_PHASE_JSON:-\"initial\"}" "${shell_call_id}" "${native_shell_call_id}" "${command_sha256}" "${cwd_sha256}" "${duration_s}" "${exit_code}" "${status}" >>"${MARS_OPENCODE_SHELL_TRACE}" 2>/dev/null || true
printf '{"ts":%s,"event":"tool_end","request_id":%s,"phase":%s,"tool_call_id":"%s","native_tool_call_id":"%s","tool_name":"bash","duration_s":%s,"status":"%s","exit_code":%d,"timing_source":"opencode_configured_shell_wrapper","command_sha256":"%s","cwd_sha256":"%s"}\n' \
  "${end_ts}" "${MARS_OPENCODE_REQUEST_ID_JSON}" "${MARS_AGENT_PHASE_JSON:-\"initial\"}" "${shell_call_id}" "${native_shell_call_id}" "${duration_s}" "${status}" "${exit_code}" "${command_sha256}" "${cwd_sha256}" >>"${MARS_OPENCODE_LIVE_EVENTS}" 2>/dev/null || true
exit "${exit_code}"
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o500)


def _codex_command(
    *,
    binary: Path,
    workspace_dir: Path,
    model: str,
    sandbox_mode: str = "danger-full-access",
    session_id: str | None = None,
    ephemeral: bool = True,
) -> list[str]:
    command = [
        str(binary),
        "exec",
        "--json",
    ]
    if ephemeral and session_id is None:
        command.append("--ephemeral")
    command.extend(
        [
            "--skip-git-repo-check",
            "--color",
            "never",
            "--sandbox",
            # The outer bubblewrap namespace is the security boundary. Running
            # a second Codex OS sandbox here is redundant and unsupported on
            # several benchmark hosts.
            sandbox_mode,
            "--cd",
            str(workspace_dir),
            "--model",
            model,
        ]
    )
    if session_id is not None:
        # All exec-level options must precede ``resume``.  The pinned CLI's
        # resume subcommand accepts only its session id and follow-up prompt.
        command.extend(["resume", session_id])
    # A dash makes both the initial and follow-up prompt come from stdin,
    # avoiding ARG_MAX for InfiniteBench/RepoBench requests.
    command.append("-")
    return command


def _terminate_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return
    deadline = time.monotonic() + 5.0
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _bwrap_binary() -> Path:
    """Resolve bubblewrap without assuming a system-wide installation."""

    configured = os.environ.get("MARS_BWRAP_BIN", "").strip()
    if configured:
        return Path(configured).expanduser().absolute()
    discovered = shutil.which("bwrap")
    if discovered:
        return Path(discovered).absolute()
    return Path("/usr/bin/bwrap")


def _node_binary() -> Path | None:
    configured = os.environ.get("MARS_NODE_BIN", "").strip()
    if configured:
        candidate = Path(configured).expanduser().absolute()
        return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None
    discovered = shutil.which("node")
    return Path(discovered).absolute() if discovered else None


def _is_bwrap_command(command: list[str] | tuple[str, ...]) -> bool:
    if not command:
        return False
    try:
        return Path(command[0]).absolute() == _bwrap_binary()
    except (OSError, ValueError):
        return False


def _working_bwrap() -> bool:
    global _BWRAP_AVAILABLE, _BWRAP_CHECKED_PATH
    with _BWRAP_CHECK_LOCK:
        binary = _bwrap_binary()
        binary_key = str(binary)
        if _BWRAP_AVAILABLE is None or _BWRAP_CHECKED_PATH != binary_key:
            _BWRAP_CHECKED_PATH = binary_key
            if not binary.is_file() or not os.access(binary, os.X_OK):
                _BWRAP_AVAILABLE = False
            else:
                result = subprocess.run(
                    [
                        str(binary),
                        "--ro-bind",
                        "/usr",
                        "/usr",
                        "--ro-bind",
                        "/bin",
                        "/bin",
                        "--ro-bind",
                        "/lib",
                        "/lib",
                        "--ro-bind",
                        "/lib64",
                        "/lib64",
                        "--proc",
                        "/proc",
                        "/bin/true",
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
                _BWRAP_AVAILABLE = result.returncode == 0
        return bool(_BWRAP_AVAILABLE)


def _agent_sandbox_command(
    command: list[str],
    *,
    agent_bin: Path,
    workspace_dir: Path,
    runtime_root: Path,
    local_inputs_dir: Path,
    require_bwrap: bool,
    readonly_paths: tuple[Path, ...] = (),
    writable_paths: tuple[Path, ...] = (),
    codex_shell_wrapper: Path | None = None,
    opencode_shell_wrapper: Path | None = None,
) -> list[str]:
    if not _working_bwrap():
        if require_bwrap:
            raise RuntimeError(
                f"Agent CLI requires working bubblewrap isolation ({_bwrap_binary()}); "
                "pass --allow-unsandboxed-agent only for explicit debugging"
            )
        return command

    # Bind the smallest tree containing both the launcher symlink and its
    # resolved package. This is node_modules for both pinned npm runtimes.
    agent_runtime_root = Path(
        os.path.commonpath((str(agent_bin.absolute()), str(agent_bin.resolve())))
    )
    if agent_runtime_root.is_file() or agent_runtime_root == agent_bin.absolute():
        agent_runtime_root = agent_runtime_root.parent
    bind_paths = [
        workspace_dir,
        runtime_root,
        agent_runtime_root,
        *readonly_paths,
        *writable_paths,
    ]
    node_binary = _node_binary()
    if node_binary is not None:
        bind_paths.append(node_binary)
    parents: set[Path] = set()
    for bind_path in bind_paths:
        current = bind_path.parent
        while current != Path("/"):
            parents.add(current)
            current = current.parent
    bwrap = [
        str(_bwrap_binary()),
        "--die-with-parent",
        "--dev",
        "/dev",
        "--unshare-pid",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/var",
        "--dir",
        "/var/tmp",
        "--dir",
        "/run",
    ]
    for parent in sorted(parents, key=lambda item: (len(item.parts), str(item))):
        if parent not in {
            Path("/usr"),
            Path("/bin"),
            Path("/sbin"),
            Path("/lib"),
            Path("/lib64"),
            Path("/etc"),
            Path("/tmp"),
            Path("/var"),
            Path("/var/tmp"),
            Path("/run"),
        }:
            bwrap.extend(["--dir", str(parent)])
    for system_path in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc"):
        if Path(system_path).exists():
            bwrap.extend(["--ro-bind", system_path, system_path])
    if node_binary is not None:
        bwrap.extend(["--ro-bind", str(node_binary), str(node_binary)])
    bwrap.extend(["--bind", str(workspace_dir), str(workspace_dir)])
    bwrap.extend(["--bind", str(runtime_root), str(runtime_root)])
    bwrap.extend(["--ro-bind", str(agent_runtime_root), str(agent_runtime_root)])
    for readonly_path in readonly_paths:
        if readonly_path.exists():
            bwrap.extend(["--ro-bind", str(readonly_path), str(readonly_path)])
    for writable_path in writable_paths:
        if writable_path.exists():
            bwrap.extend(["--bind", str(writable_path), str(writable_path)])
    if codex_shell_wrapper is not None:
        bwrap.extend(["--dir", "/run/mars-agent-shell"])
        # Bind the real executable before overlaying both paths.  /bin is a
        # separate mount in this sandbox even on merged-/usr hosts.
        bwrap.extend(["--ro-bind", "/usr/bin/bash", "/run/mars-agent-shell/bash.real"])
        bwrap.extend(["--ro-bind", str(codex_shell_wrapper), "/usr/bin/bash"])
        bwrap.extend(["--ro-bind", str(codex_shell_wrapper), "/bin/bash"])
    if opencode_shell_wrapper is not None:
        # Only OpenCode's ConfigV1.shell target is interposed.  System Bash
        # remains untouched, so CLI startup and dependency probes cannot create
        # false tool spans.
        bwrap.extend(["--dir", "/run/mars-opencode-shell"])
        bwrap.extend(
            ["--ro-bind", "/usr/bin/bash", "/run/mars-opencode-shell/bash.real"]
        )
        bwrap.extend(
            [
                "--ro-bind",
                str(opencode_shell_wrapper),
                "/run/mars-opencode-shell/bash",
            ]
        )
    bwrap.extend(["--chdir", str(workspace_dir), "--"])
    bwrap.extend(command)
    return bwrap


def _cpu_affinity_command(command: list[str], cpu_affinity: str | None) -> list[str]:
    if not cpu_affinity:
        return command
    taskset = Path("/usr/bin/taskset")
    if not taskset.is_file():
        raise RuntimeError("--cpu-affinity requires /usr/bin/taskset")
    return [str(taskset), "-c", cpu_affinity, *command]


def _run_and_capture(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout_s: float,
    stdin_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    timed_events_path: Path,
    on_stdout_json: Any | None = None,
    phase: str = "initial",
) -> tuple[int, bool]:
    # Both CLIs consume a piped prompt.  A regular file descriptor avoids both
    # argv's ARG_MAX and pipe-buffer deadlocks for multi-megabyte prompts.
    with stdin_path.open("rb") as stdin_fp:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdin=stdin_fp,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    assert proc.stdout is not None and proc.stderr is not None
    selector = selectors.DefaultSelector()
    streams = {proc.stdout.fileno(): ("stdout", proc.stdout), proc.stderr.fileno(): ("stderr", proc.stderr)}
    for fd, (_, stream) in streams.items():
        os.set_blocking(fd, False)
        selector.register(stream, selectors.EVENT_READ, data=fd)
    buffers: dict[int, bytes] = {fd: b"" for fd in streams}
    started = time.monotonic()
    timed_out = False

    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp, timed_events_path.open(
        "w", encoding="utf-8"
    ) as timed_fp:
        def record_stdout_json(raw: dict[str, Any]) -> None:
            captured_at_s = time.time()
            timed_fp.write(
                json.dumps(
                    {"captured_at_s": captured_at_s, "phase": phase, "raw": raw},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            timed_fp.flush()
            if on_stdout_json is not None:
                try:
                    on_stdout_json(raw, captured_at_s)
                except Exception:
                    # Trace instrumentation must not take down the agent.
                    pass

        while selector.get_map():
            if timeout_s > 0 and time.monotonic() - started > timeout_s and proc.poll() is None:
                timed_out = True
                _terminate_group(proc)
            ready = selector.select(timeout=0.2)
            if not ready and proc.poll() is not None:
                ready = [(key, selectors.EVENT_READ) for key in list(selector.get_map().values())]
            for key, _ in ready:
                fd = int(key.data)
                label, stream = streams[fd]
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    if buffers[fd]:
                        remainder = buffers[fd]
                        buffers[fd] = b""
                        if label == "stdout":
                            line = remainder.decode("utf-8", errors="replace")
                            try:
                                raw = json.loads(line)
                            except Exception:
                                raw = None
                            if isinstance(raw, dict):
                                record_stdout_json(raw)
                    continue
                target = stdout_fp if label == "stdout" else stderr_fp
                target.write(chunk)
                target.flush()
                if label != "stdout":
                    continue
                data = buffers[fd] + chunk
                lines = data.split(b"\n")
                buffers[fd] = lines.pop()
                for raw_line in lines:
                    try:
                        raw = json.loads(raw_line.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    if not isinstance(raw, dict):
                        continue
                    record_stdout_json(raw)
        proc.wait()
    return int(proc.returncode), timed_out


def _remaining_timeout_s(deadline_monotonic: float | None, *, now: float | None = None) -> float:
    """Return the remaining whole-request budget; zero means it is exhausted.

    ``None`` represents the existing unlimited-timeout contract.
    """

    if deadline_monotonic is None:
        return 0.0
    current = time.monotonic() if now is None else float(now)
    return max(0.0, float(deadline_monotonic) - current)


def _native_session_id(agent: str, rows: list[dict[str, Any]]) -> str | None:
    """Extract one stable native session id and reject conflicting events."""

    found: set[str] = set()
    for wrapper in rows:
        raw = wrapper.get("raw") if isinstance(wrapper.get("raw"), dict) else wrapper
        if not isinstance(raw, dict):
            continue
        value: Any = None
        if agent == "codex" and str(raw.get("type") or "") == "thread.started":
            value = raw.get("thread_id")
        elif agent == "opencode":
            value = raw.get("sessionID")
        if isinstance(value, str) and value.strip():
            found.add(value.strip())
    if len(found) > 1:
        raise ValueError(f"conflicting {agent} session ids: {sorted(found)}")
    return next(iter(found), None)


def _namespace_phase_records(
    records: list[dict[str, Any]],
    *,
    phase: str,
    id_field: str,
) -> list[dict[str, Any]]:
    """Make process-local native ids unique without discarding provenance."""

    namespaced: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        item = dict(record)
        native = str(item.get(id_field) or f"item_{index:04d}")
        item[f"native_{id_field}"] = native
        item[id_field] = f"{phase}:{native}"
        raw_part_id = item.get("raw_part_id")
        if raw_part_id is not None:
            item["native_raw_part_id"] = raw_part_id
            item["raw_part_id"] = f"{phase}:{raw_part_id}"
        item["phase"] = phase
        namespaced.append(item)
    return namespaced


def _combine_phase_files(sources: list[Path], destination: Path) -> None:
    """Concatenate phase artifacts while retaining each source file verbatim."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out:
        needs_separator = False
        for source in sources:
            if not source.is_file():
                continue
            data = source.read_bytes()
            if not data:
                continue
            if needs_separator and not data.startswith(b"\n"):
                out.write(b"\n")
            out.write(data)
            needs_separator = not data.endswith(b"\n")


def _sum_duration(records: list[dict[str, Any]]) -> float:
    total = 0.0
    for item in records:
        duration = item.get("duration_s")
        if isinstance(duration, (int, float)):
            total += float(duration)
    return total


def _openhands_llm_calls(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aligned: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["submit_time_s"] = item.get("start_time_s")
        item["api_kind"] = "completion"
        http_status = item.get("http_status")
        item["status"] = (
            "ok"
            if isinstance(http_status, int) and 200 <= http_status < 300 and not item.get("error")
            else "error"
        )
        item.setdefault("error", None)
        aligned.append(item)
    return aligned


def _tool_kind(tool_name: str) -> str:
    lowered = tool_name.lower()
    if lowered in {"command_execution", "bash", "shell", "terminal"}:
        return "TerminalAction"
    if lowered in {"file_change", "edit", "write", "read", "patch"}:
        return "FileEditorAction"
    if lowered in {"task_tracker", "todo_list", "todowrite", "todoread"}:
        return "TaskTrackerAction"
    return "AgentToolAction"


def _openhands_tool_calls(
    records: list[dict[str, Any]],
    llm_calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aligned: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        tool_id = str(item.get("tool_call_id") or "")
        tool_name = str(item.get("tool_name") or "unknown")
        raw_status = str(item.get("status") or "unknown").lower()
        item["action_id"] = item.get("raw_part_id") or tool_id
        item["tool_kind"] = _tool_kind(tool_name)
        item["verification_tool_kind"] = _verification_tool_kind(tool_name)
        item["summary"] = item.get("input_preview")
        item["native_tool_call_id"] = (
            item.get("native_tool_call_id")
            or item.get("native_raw_part_id")
            or item.get("raw_part_id")
            or tool_id
        )
        item["observation_preview"] = item.get("output_preview")
        item["status"] = (
            "ok"
            if raw_status in {"ok", "completed", "complete", "done", "success"}
            else "error"
            if raw_status in {"failed", "failure", "error", "cancelled", "aborted"}
            else raw_status
        )
        item["error"] = (
            {"type": "AgentToolError", "message": item.get("output_preview")}
            if item["status"] == "error"
            else None
        )
        tool_start = float(item.get("start_time_s") or 0.0)
        # Associate a tool with the latest model round that had already begun.
        # Some CLIs emit the tool event before the proxy observes the final SSE
        # bytes for that same response, so requiring ``llm.end <= tool.start``
        # incorrectly turns the first tool into an orphan.
        emitters = [
            call
            for call in llm_calls
            if isinstance(call.get("start_time_s"), (int, float))
            and float(call["start_time_s"]) <= tool_start + 1e-6
        ]
        emitter = emitters[-1] if emitters else None
        item["llm_response_id"] = emitter.get("response_id") if emitter else None
        item["emitter_llm_call_id"] = emitter.get("llm_call_id") if emitter else None
        aligned.append(item)

    rounds: list[dict[str, Any]] = []
    assigned_ids: set[str] = set()
    for index, call in enumerate(llm_calls):
        call_id = call.get("llm_call_id")
        tools = [item for item in aligned if item.get("emitter_llm_call_id") == call_id]
        assigned_ids.update(str(item.get("tool_call_id") or "") for item in tools)
        rounds.append({"round_index": index, "llm": call, "tools": tools})
    orphan_tools = [
        item
        for item in aligned
        if str(item.get("tool_call_id") or "") not in assigned_ids
    ]
    return aligned, rounds, orphan_tools


def _request_events(
    *,
    agent: str,
    request_id: str,
    mars_job_id: str,
    dataset_request_id: Any,
    started: float,
    ended: float,
    workspace_dir: Path,
    llm_calls: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    status: str,
    trace: dict[str, Any],
    verification_nudges: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    common = {
        "agent": agent,
        "request_id": request_id,
        "mars_job_id": mars_job_id,
        "dataset_request_id": dataset_request_id,
        "benchmark": trace.get("benchmark"),
        "task_id": trace.get("task_id"),
    }
    events: list[dict[str, Any]] = [
        {
            "ts": started,
            "event": "request_start",
            **common,
            "queue_wait_s": trace["queue_wait_s"],
        },
        {
            "ts": started,
            "event": "workspace_ready",
            **common,
            "workspace_dir": str(workspace_dir),
            "file_count": trace.get("workspace_file_count"),
        },
        {
            "ts": started,
            "event": "session_init",
            **common,
            "frontend": agent,
            "agent_session_id": trace.get("agent_session_id"),
            "workspace_dir": str(workspace_dir),
            "runtime_dir": str(workspace_dir / ".mars_runtime"),
            "artifact_dir": str(workspace_dir / ".mars_runtime" / "artifacts"),
        },
    ]
    for nudge in verification_nudges or []:
        events.append(
            {
                "ts": float(nudge["ts"]),
                "event": "verification_nudge",
                **common,
                "attempt": int(nudge.get("attempt") or 1),
                "phase": nudge.get("phase"),
                "agent_session_id": trace.get("agent_session_id"),
            }
        )
    for call in llm_calls:
        call_common = {**common, "llm_call_id": call.get("llm_call_id")}
        submit = call.get("submit_time_s")
        first = call.get("first_token_time_s")
        end = call.get("end_time_s")
        if isinstance(submit, (int, float)):
            events.append(
                {
                    "ts": float(submit),
                    "event": "gpu_submit",
                    **call_common,
                    "model": call.get("model"),
                    "api_kind": call.get("api_kind"),
                    "message_count": call.get("message_count"),
                    "tool_count": call.get("tool_count"),
                }
            )
        if isinstance(first, (int, float)):
            events.append(
                {
                    "ts": float(first),
                    "event": "gpu_first_token",
                    **call_common,
                    "ttft_s": call.get("ttft_s"),
                }
            )
        if isinstance(end, (int, float)):
            events.append(
                {
                    "ts": float(end),
                    "event": "gpu_end",
                    **call_common,
                    "duration_s": call.get("duration_s"),
                    "status": call.get("status"),
                    "response_id": call.get("response_id"),
                    "prompt_tokens": call.get("prompt_tokens"),
                    "completion_tokens": call.get("completion_tokens"),
                    "cached_input_tokens": call.get("cached_input_tokens"),
                    "reasoning_tokens": call.get("reasoning_tokens"),
                    "error": call.get("error"),
                }
            )
    for tool in tool_calls:
        tool_common = {
            **common,
            "tool_call_id": tool.get("tool_call_id"),
            "tool_name": tool.get("tool_name"),
            "action_id": tool.get("action_id"),
            "tool_kind": tool.get("tool_kind"),
            "summary": tool.get("summary"),
            "llm_response_id": tool.get("llm_response_id"),
            "native_tool_call_id": tool.get("native_tool_call_id"),
        }
        for event_name, field in (
            ("tool_enqueued", "enqueued_time_s"),
            ("tool_start", "start_time_s"),
            ("tool_end", "end_time_s"),
        ):
            ts = tool.get(field)
            if isinstance(ts, (int, float)):
                event = {"ts": float(ts), "event": event_name, **tool_common}
                if event_name == "tool_start":
                    event["wait_s"] = tool.get("wait_s")
                if event_name == "tool_end":
                    event.update(
                        duration_s=tool.get("duration_s"),
                        status=tool.get("status"),
                        error=tool.get("error"),
                    )
                events.append(event)
    if status == "verification_missing":
        events.append(
            {
                "ts": ended,
                "event": "verification_missing",
                **common,
                "verification_tool_count": trace.get("verification_tool_count", 0),
            }
        )
    events.append(
        {
            "ts": ended,
            "event": "request_done",
            **common,
            "status": status,
            "conversation_status": trace.get("conversation_status"),
            "queue_wait_s": trace["queue_wait_s"],
            "e2e_latency_s": trace["e2e_latency_s"],
            "gpu_total_s": trace["gpu_total_s"],
            "tool_total_s": trace["tool_total_s"],
            "tool_wait_total_s": trace["tool_wait_total_s"],
            "wait_total_s": trace["wait_total_s"],
            "verification_tool_count": trace.get("verification_tool_count", 0),
            "verification_nudge_count": trace.get("verification_nudge_count", 0),
            "llm_call_count": len(llm_calls),
            "tool_call_count": len(tool_calls),
            "topology_sha256": (trace.get("dynamic_graph") or {}).get("topology_sha256"),
        }
    )
    return sorted(events, key=lambda event: (float(event.get("ts") or 0.0), str(event.get("event") or "")))


def run_request(
    *,
    agent: str,
    agent_bin: Path,
    task: dict[str, Any],
    task_sha256: str,
    request_id: str,
    mars_job_id: str,
    arrival_time_s: float,
    workspace_root: Path,
    local_inputs_dir: Path,
    request_run_dir: Path,
    model: str,
    base_url: str,
    api_key: str,
    context_window: int,
    max_output_tokens: int,
    max_iterations: int,
    temperature: float,
    top_p: float | None = None,
    timeout_s: float,
    require_bwrap: bool,
    llm_timeout_s: float = 3000.0,
    llm_num_retries: int = 3,
    terminal_no_change_timeout_s: int | None = 30,
    benchmark_root: Path | None = None,
    cpu_affinity: str | None = None,
) -> dict[str, Any]:
    # Native CLIs do not expose OpenHands' terminal no-change knob. Retain it
    # in provenance while enforcing one whole-request wall-clock deadline over
    # both the initial run and the optional same-session verification follow-up.
    request_run_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = (workspace_root / request_id).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    _write_json(request_run_dir / "task.json", task)
    prompt_path = request_run_dir / "prompt.txt"
    prompt_path.write_text(
        _build_prompt(task, agent=agent, max_iterations=max_iterations) + "\n",
        encoding="utf-8",
    )

    started_monotonic = time.monotonic()
    started = time.time()
    deadline_monotonic = (
        started_monotonic + float(timeout_s) if float(timeout_s) > 0 else None
    )
    queue_wait_s = max(0.0, started - arrival_time_s)
    verification_required = requires_tool_verification(task)
    workspace_file_count = materialize_workspace_init(task, workspace_dir)
    (workspace_dir / ".mars_runtime" / "artifacts").mkdir(parents=True, exist_ok=True)

    llm_events_path = request_run_dir / "llm_events.jsonl"
    live_events_path = request_run_dir / "live_events.jsonl"
    live_events_path.write_text("", encoding="utf-8")
    env = _runtime_env(request_run_dir, workspace_dir, api_key)
    runtime_root = request_run_dir / "runtime"
    codex_shell_timing = agent == "codex" and _working_bwrap()
    opencode_shell_timing = agent == "opencode" and _working_bwrap()
    codex_shell_wrapper_path: Path | None = None
    opencode_shell_wrapper_path: Path | None = None
    if codex_shell_timing:
        codex_shell_wrapper_path = request_run_dir / "codex_bash_wrapper.sh"
        _write_codex_shell_wrapper(codex_shell_wrapper_path)
        env.update(
            {
                "MARS_CODEX_LIVE_EVENTS": str(live_events_path),
                "MARS_CODEX_REQUEST_ID_JSON": json.dumps(request_id, ensure_ascii=False),
            }
        )
    if opencode_shell_timing:
        opencode_shell_wrapper_path = request_run_dir / "opencode_bash_wrapper.sh"
        _write_opencode_shell_wrapper(opencode_shell_wrapper_path)
        env.update(
            {
                "MARS_OPENCODE_LIVE_EVENTS": str(live_events_path),
                "MARS_OPENCODE_REQUEST_ID_JSON": json.dumps(
                    request_id, ensure_ascii=False
                ),
            }
        )

    phase_artifacts: dict[str, dict[str, Path]] = {}

    def artifacts_for(phase: str) -> dict[str, Path]:
        artifacts = phase_artifacts.get(phase)
        if artifacts is None:
            shell_stem = (
                "codex_shell_events"
                if agent == "codex"
                else "opencode_shell_events"
            )
            artifacts = {
                "stdout": request_run_dir / f"cli_stdout.{phase}.jsonl",
                "stderr": request_run_dir / f"cli_stderr.{phase}.txt",
                "timed": request_run_dir / f"cli_events_timed.{phase}.jsonl",
                "command": request_run_dir / f"command.{phase}.json",
                "shell": request_run_dir / f"{shell_stem}.{phase}.jsonl",
            }
            phase_artifacts[phase] = artifacts
        return artifacts

    docker_proxy: ScopedDockerProxy | None = None
    docker_tempdir: tempfile.TemporaryDirectory[str] | None = None
    docker_access = "none"
    allowed_input_roots = [local_inputs_dir]
    if benchmark_root is not None:
        allowed_input_roots.append(benchmark_root)
    readonly_paths = task_readonly_paths(
        task,
        allowed_roots=allowed_input_roots,
    )
    if str(task.get("benchmark") or "").strip().lower() == "terminalbench":
        try:
            # Keep the AF_UNIX path short even when formal output paths are
            # descriptive, and never expose the host Docker socket directly.
            docker_tempdir = tempfile.TemporaryDirectory(
                prefix=f"mars-dp-{os.getuid()}-{request_id[-8:]}-",
                dir="/tmp",
            )
            docker_runtime_root = Path(docker_tempdir.name) / "runtime"
            docker_runtime_root.mkdir(parents=True, exist_ok=True)
            docker_proxy = ScopedDockerProxy(
                workspace_dir=workspace_dir,
                runtime_root=docker_runtime_root,
            )
            docker_proxy.install_wrapper(env)
            readonly_paths = tuple(
                sorted(
                    {*readonly_paths, docker_proxy.task_dir, docker_runtime_root},
                    key=str,
                )
            )
            docker_access = "scoped_terminalbench_proxy"
        except Exception as exc:
            docker_access = "unavailable_legacy_missing_asset"
            (request_run_dir / "docker_proxy_unavailable.txt").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )

    phase_order: list[str] = []
    phase_results: list[dict[str, Any]] = []
    verification_nudges: list[dict[str, Any]] = []
    agent_session_id: str | None = None
    control_error: dict[str, Any] | None = None
    opencode_timing_integrity_by_phase: dict[str, dict[str, Any]] = {}
    outer_sandbox_enabled = bool(require_bwrap and _working_bwrap())
    iteration_limited = False
    config_path: Path | None = None
    codex_sandbox_mode = "danger-full-access" if require_bwrap else "workspace-write"
    live_fp = live_events_path.open("a", encoding="utf-8")

    def execute_phase(
        *,
        phase: str,
        stdin_path: Path,
        session_id: str | None,
    ) -> tuple[int | None, bool]:
        nonlocal outer_sandbox_enabled
        artifacts = artifacts_for(phase)
        phase_order.append(phase)
        remaining = (
            None
            if deadline_monotonic is None
            else _remaining_timeout_s(deadline_monotonic)
        )
        phase_started = time.time()
        if remaining is not None and remaining <= 0:
            for key in ("stdout", "stderr", "timed"):
                artifacts[key].write_text("", encoding="utf-8")
            if codex_shell_timing or opencode_shell_timing:
                artifacts["shell"].write_text("", encoding="utf-8")
            phase_results.append(
                {
                    "phase": phase,
                    "start_time_s": phase_started,
                    "end_time_s": phase_started,
                    "timeout_budget_s": 0.0,
                    "returncode": None,
                    "timed_out": True,
                    "started": False,
                    "not_started_reason": "whole_request_timeout_exhausted",
                }
            )
            return None, True

        if agent == "opencode":
            assert config_path is not None
            native_command = _opencode_command(
                binary=agent_bin,
                config_path=config_path,
                workspace_dir=workspace_dir,
                model=model,
                session_id=session_id,
            )
        elif agent == "codex":
            native_command = _codex_command(
                binary=agent_bin,
                workspace_dir=workspace_dir,
                model=model,
                sandbox_mode=codex_sandbox_mode,
                session_id=session_id,
                # Verification-required requests must persist their initial
                # thread so the single nudge can resume that exact session.
                ephemeral=not verification_required,
            )
        else:
            raise ValueError(f"unsupported agent: {agent}")

        phase_env = dict(env)
        phase_env["MARS_AGENT_PHASE"] = phase
        phase_env["MARS_AGENT_PHASE_JSON"] = json.dumps(phase, ensure_ascii=False)
        writable_paths: tuple[Path, ...] = ()
        if codex_shell_timing:
            artifacts["shell"].write_text("", encoding="utf-8")
            phase_env["MARS_CODEX_SHELL_TRACE"] = str(artifacts["shell"])
            writable_paths = (artifacts["shell"], live_events_path)
        elif opencode_shell_timing:
            artifacts["shell"].write_text("", encoding="utf-8")
            phase_env["MARS_OPENCODE_SHELL_TRACE"] = str(artifacts["shell"])
            writable_paths = (artifacts["shell"], live_events_path)

        command = _agent_sandbox_command(
            native_command,
            agent_bin=agent_bin,
            workspace_dir=workspace_dir,
            runtime_root=runtime_root,
            local_inputs_dir=local_inputs_dir,
            readonly_paths=readonly_paths,
            writable_paths=writable_paths,
            codex_shell_wrapper=codex_shell_wrapper_path,
            opencode_shell_wrapper=opencode_shell_wrapper_path,
            require_bwrap=require_bwrap,
        )
        sandboxed = _is_bwrap_command(command)
        outer_sandbox_enabled = outer_sandbox_enabled or sandboxed
        command = _cpu_affinity_command(command, cpu_affinity)
        _write_json(artifacts["command"], command)
        if phase == "initial":
            _write_json(request_run_dir / "command.json", command)

        live_normalizer = LiveToolEventNormalizer(
            agent=agent,
            request_id=request_id,
            codex_shell_timing=codex_shell_timing,
            opencode_shell_timing=opencode_shell_timing,
        )

        def emit_live(raw: dict[str, Any], captured_at_s: float) -> None:
            for event in live_normalizer.feed(raw, captured_at_s):
                item = dict(event)
                native_id = str(item.get("tool_call_id") or "unknown")
                item["native_tool_call_id"] = native_id
                item["tool_call_id"] = f"{phase}:{native_id}"
                item["phase"] = phase
                live_fp.write(
                    json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            live_fp.flush()

        try:
            returncode, timed_out = _run_and_capture(
                command,
                env=phase_env,
                cwd=workspace_dir,
                timeout_s=0.0 if remaining is None else remaining,
                stdin_path=stdin_path,
                stdout_path=artifacts["stdout"],
                stderr_path=artifacts["stderr"],
                timed_events_path=artifacts["timed"],
                on_stdout_json=emit_live,
                phase=phase,
            )
        finally:
            phase_ended = time.time()
            for event in live_normalizer.finish(phase_ended):
                item = dict(event)
                native_id = str(item.get("tool_call_id") or "unknown")
                item["native_tool_call_id"] = native_id
                item["tool_call_id"] = f"{phase}:{native_id}"
                item["phase"] = phase
                live_fp.write(
                    json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            live_fp.flush()
            if "returncode" in locals():
                phase_results.append(
                    {
                        "phase": phase,
                        "start_time_s": phase_started,
                        "end_time_s": phase_ended,
                        "timeout_budget_s": remaining,
                        "returncode": returncode,
                        "timed_out": timed_out,
                        "started": True,
                        "command_path": artifacts["command"].name,
                        "stdout_path": artifacts["stdout"].name,
                        "stderr_path": artifacts["stderr"].name,
                        "timed_events_path": artifacts["timed"].name,
                        "shell_events_path": (
                            artifacts["shell"].name
                            if codex_shell_timing or opencode_shell_timing
                            else None
                        ),
                    }
                )
        return returncode, timed_out

    def normalize_phase(phase: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nonlocal control_error
        artifacts = artifacts_for(phase)
        rows = load_jsonl(artifacts["timed"])
        if agent == "opencode":
            if opencode_shell_timing:
                try:
                    tools, messages = normalize_opencode(
                        rows,
                        shell_rows=load_jsonl(artifacts["shell"]),
                        workspace_dir=workspace_dir,
                        require_shell_timing=True,
                    )
                except OpenCodeTimingIntegrityError as exc:
                    details = {"status": "failed", **exc.details}
                    opencode_timing_integrity_by_phase[phase] = details
                    if control_error is None:
                        control_error = {
                            "type": "OpenCodeTimingIntegrityError",
                            "message": str(exc),
                            "phase": phase,
                            "timing_integrity": details,
                        }
                    # Preserve the native conversation for diagnostics, but the
                    # explicit control error prevents publication as a valid
                    # completed measurement.
                    tools, messages = normalize_opencode(rows)
                else:
                    bash_tools = [
                        item
                        for item in tools
                        if str(item.get("tool_name") or "").lower() == "bash"
                    ]
                    opencode_timing_integrity_by_phase[phase] = {
                        "status": "passed",
                        "native_bash_count": len(bash_tools),
                        "matched_count": sum(
                            item.get("timing_integrity") == "matched"
                            for item in bash_tools
                        ),
                    }
            else:
                tools, messages = normalize_opencode(rows)
        else:
            tools, messages = normalize_codex(
                rows,
                shell_rows=load_jsonl(artifacts["shell"]),
            )
        return (
            _namespace_phase_records(tools, phase=phase, id_field="tool_call_id"),
            _namespace_phase_records(messages, phase=phase, id_field="message_id"),
        )

    try:
        with (
            docker_proxy if docker_proxy is not None else nullcontext(),
            TraceProxy(
                upstream_base_url=base_url,
                trace_path=llm_events_path,
                request_id=request_id,
                mars_job_id=mars_job_id,
                responses_via_chat=(agent == "codex"),
                max_llm_calls=max_iterations,
                upstream_timeout_s=llm_timeout_s,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
            ).start() as proxy,
        ):
            if agent == "opencode":
                config_path = Path(env["XDG_CONFIG_HOME"]) / "opencode.json"
                _write_opencode_config(
                    config_path,
                    proxy_base_url=proxy.base_url,
                    model=model,
                    context_window=context_window,
                    max_output_tokens=max_output_tokens,
                    max_iterations=max_iterations,
                    temperature=temperature,
                    shell=(
                        "/run/mars-opencode-shell/bash"
                        if opencode_shell_timing
                        else None
                    ),
                )
                env["OPENCODE_CONFIG"] = str(config_path)
            elif agent == "codex":
                config_path = Path(env["CODEX_HOME"]) / "config.toml"
                _write_codex_config(
                    config_path,
                    proxy_base_url=proxy.base_url,
                    model=model,
                    context_window=context_window,
                    llm_timeout_s=llm_timeout_s,
                    llm_num_retries=llm_num_retries,
                    sandbox_mode=codex_sandbox_mode,
                )
                _write_codex_rules(Path(env["CODEX_HOME"]))
            else:
                raise ValueError(f"unsupported agent: {agent}")

            initial_returncode, initial_timed_out = execute_phase(
                phase="initial",
                stdin_path=prompt_path,
                session_id=None,
            )
            initial_tools, _initial_messages = normalize_phase("initial")
            initial_verification_tool_count = _verification_tool_count(initial_tools)
            initial_rows = load_jsonl(artifacts_for("initial")["timed"])
            try:
                agent_session_id = _native_session_id(agent, initial_rows)
            except ValueError as exc:
                control_error = {
                    "type": "NativeSessionIdConflict",
                    "message": str(exc),
                }

            should_nudge = bool(
                verification_required
                and initial_returncode == 0
                and not initial_timed_out
                and not proxy.limit_exceeded
                and control_error is None
                and initial_verification_tool_count <= 0
            )
            if should_nudge:
                if agent_session_id is None:
                    control_error = {
                        "type": "NativeSessionIdMissing",
                        "message": (
                            f"{agent} did not emit a native session id; refusing to start "
                            "the required follow-up in a different session"
                        ),
                    }
                else:
                    nudge_phase = "verification_nudge_1"
                    nudge_path = request_run_dir / "verification_nudge.txt"
                    nudge_path.write_text(build_verification_nudge(task) + "\n", encoding="utf-8")
                    verification_nudges.append(
                        {
                            "ts": time.time(),
                            "attempt": 1,
                            "phase": nudge_phase,
                            "agent_session_id": agent_session_id,
                        }
                    )
                    execute_phase(
                        phase=nudge_phase,
                        stdin_path=nudge_path,
                        session_id=agent_session_id,
                    )
                    followup_rows = load_jsonl(artifacts_for(nudge_phase)["timed"])
                    try:
                        followup_session_id = _native_session_id(agent, followup_rows)
                    except ValueError as exc:
                        control_error = {
                            "type": "NativeSessionIdConflict",
                            "message": str(exc),
                        }
                    else:
                        if (
                            followup_session_id is not None
                            and followup_session_id != agent_session_id
                        ):
                            control_error = {
                                "type": "NativeSessionResumeMismatch",
                                "message": (
                                    f"requested native session {agent_session_id!r}, but "
                                    f"{agent} emitted {followup_session_id!r}"
                                ),
                            }
            iteration_limited = bool(proxy.limit_exceeded)
    finally:
        live_fp.close()
        if docker_tempdir is not None:
            audit_source = Path(docker_tempdir.name) / "docker_proxy_audit.jsonl"
            if audit_source.is_file():
                shutil.copy2(audit_source, request_run_dir / "docker_proxy_audit.jsonl")
            docker_tempdir.cleanup()

    # Retain canonical one-file artifacts for existing analysis while keeping
    # immutable per-phase files so the follow-up can never overwrite phase one.
    _combine_phase_files(
        [phase_artifacts[name]["stdout"] for name in phase_order],
        request_run_dir / "cli_stdout.jsonl",
    )
    _combine_phase_files(
        [phase_artifacts[name]["stderr"] for name in phase_order],
        request_run_dir / "cli_stderr.txt",
    )
    _combine_phase_files(
        [phase_artifacts[name]["timed"] for name in phase_order],
        request_run_dir / "cli_events_timed.jsonl",
    )
    if codex_shell_timing:
        _combine_phase_files(
            [phase_artifacts[name]["shell"] for name in phase_order],
            request_run_dir / "codex_shell_events.jsonl",
        )
    if opencode_shell_timing:
        _combine_phase_files(
            [phase_artifacts[name]["shell"] for name in phase_order],
            request_run_dir / "opencode_shell_events.jsonl",
        )

    ended = time.time()
    proxy_rows = load_jsonl(llm_events_path)
    llm_calls = _openhands_llm_calls(normalize_llm_calls(proxy_rows))
    for call in llm_calls:
        call_start = call.get("start_time_s")
        if not isinstance(call_start, (int, float)):
            continue
        for phase_result in reversed(phase_results):
            phase_start = phase_result.get("start_time_s")
            phase_end = phase_result.get("end_time_s")
            if (
                isinstance(phase_start, (int, float))
                and isinstance(phase_end, (int, float))
                and float(phase_start) - 1e-6 <= float(call_start) <= float(phase_end) + 1e-6
            ):
                call["phase"] = phase_result.get("phase")
                break

    raw_tool_calls: list[dict[str, Any]] = []
    agent_messages: list[dict[str, Any]] = []
    for phase in phase_order:
        phase_tools, phase_messages = normalize_phase(phase)
        raw_tool_calls.extend(phase_tools)
        agent_messages.extend(phase_messages)
    tool_calls, rounds, orphan_tools = _openhands_tool_calls(raw_tool_calls, llm_calls)
    dynamic_graph = build_dynamic_graph(llm_calls, tool_calls)
    verification_tool_count = _verification_tool_count(tool_calls)
    initial_verification_tool_count = _verification_tool_count(
        [item for item in tool_calls if item.get("phase") == "initial"]
    )

    stderr_path = request_run_dir / "cli_stderr.txt"
    stderr_preview = (
        stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        if stderr_path.is_file()
        else ""
    )
    any_timed_out = any(bool(item.get("timed_out")) for item in phase_results)
    failed_phase = next(
        (
            item
            for item in phase_results
            if isinstance(item.get("returncode"), int) and int(item["returncode"]) != 0
        ),
        None,
    )
    returncode = phase_results[-1].get("returncode") if phase_results else None
    status = (
        "worker_timeout"
        if any_timed_out
        else "error"
        if iteration_limited
        else "worker_error"
        if control_error is not None or failed_phase is not None
        else "verification_missing"
        if verification_required and verification_tool_count <= 0
        else "completed"
    )
    error: dict[str, Any] | None = None
    if status in {"worker_timeout", "worker_error", "error"}:
        if control_error is not None:
            error = {**control_error, "stderr_preview": stderr_preview}
        elif any_timed_out:
            error = {
                "type": "WorkerProcessTimeout",
                "message": f"whole request exceeded timeout after {timeout_s:.1f}s",
                "stderr_preview": stderr_preview,
            }
        elif iteration_limited:
            error = {
                "type": "MaxIterationsReached",
                "message": f"Agent reached maximum iterations limit ({max_iterations}).",
                "stderr_preview": stderr_preview,
            }
        else:
            assert failed_phase is not None
            error = {
                "type": "WorkerProcessError",
                "message": (
                    f"{failed_phase.get('phase')} exited with code "
                    f"{failed_phase.get('returncode')}"
                ),
                "stderr_preview": stderr_preview,
            }

    gpu_total_s = _sum_duration(llm_calls)
    tool_total_s = _sum_duration(tool_calls)
    tool_wait_total_s = sum(
        float(item.get("wait_s") or 0.0)
        for item in tool_calls
        if isinstance(item.get("wait_s"), (int, float))
    )
    retry_policy = _llm_retry_policy(agent, llm_num_retries)
    phase_returncodes = {
        str(item.get("phase")): item.get("returncode") for item in phase_results
    }
    tool_timing_integrity: dict[str, Any] | None = None
    if agent == "opencode":
        if opencode_shell_timing:
            timing_failed = any(
                item.get("status") != "passed"
                for item in opencode_timing_integrity_by_phase.values()
            )
            tool_timing_integrity = {
                "status": "failed" if timing_failed else "passed",
                "source": "opencode_configured_shell_wrapper",
                "phases": opencode_timing_integrity_by_phase,
            }
        else:
            tool_timing_integrity = {
                "status": "not_instrumented",
                "source": "opencode_native_part",
                "phases": {},
            }
    trace = {
        "schema": "mars_agent_cli_trace_v1",
        "agent": agent,
        "request_id": request_id,
        "mars_job_id": mars_job_id,
        "dataset_request_id": task.get("request_id"),
        "benchmark": task.get("benchmark"),
        "task_id": task.get("task_id"),
        "workspace_dir": str(workspace_dir),
        "workspace_file_count": workspace_file_count,
        "outer_sandbox": (
            "bubblewrap_pid_filesystem" if outer_sandbox_enabled else "disabled_debug_only"
        ),
        "docker_access": docker_access,
        "sandbox_readonly_paths": [str(path) for path in readonly_paths],
        "cpu_affinity": cpu_affinity,
        "llm_timeout_s": llm_timeout_s,
        # OpenCode 1.18.23 fixes SDK retries at 0 and outer retries at 5;
        # retaining the user request under a separate key avoids pretending it
        # accepted the OpenHands/Codex retry knob.
        "llm_num_retries": llm_num_retries if agent == "codex" else None,
        "requested_llm_num_retries": llm_num_retries,
        "llm_retry_policy": retry_policy,
        "top_p": top_p,
        "terminal_no_change_timeout_s": terminal_no_change_timeout_s,
        "tool_timing_integrity": tool_timing_integrity,
        "arrival_time_s": arrival_time_s,
        "start_time_s": started,
        "end_time_s": ended,
        "queue_wait_s": queue_wait_s,
        "e2e_latency_s": max(0.0, ended - arrival_time_s),
        "gpu_total_s": gpu_total_s,
        "tool_total_s": tool_total_s,
        "tool_wait_total_s": tool_wait_total_s,
        "wait_total_s": queue_wait_s + tool_wait_total_s,
        "verification_required": verification_required,
        "initial_verification_tool_count": initial_verification_tool_count,
        "verification_tool_count": verification_tool_count,
        "verification_nudge_count": len(verification_nudges),
        "verification_nudges": verification_nudges,
        "agent_session_id": agent_session_id,
        "phase_results": phase_results,
        "phase_returncodes": phase_returncodes,
        "status": status,
        "conversation_status": (
            "finished"
            if status in {"completed", "verification_missing"}
            else "error"
            if status in {"error", "worker_error", "worker_timeout"}
            else None
        ),
        "returncode": returncode,
        "error": error,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "rounds": rounds,
        "orphan_tools": orphan_tools,
        "agent_messages": agent_messages,
        "agent_errors": [],
        "conversation_errors": [],
        "dynamic_graph": dynamic_graph,
    }
    events = _request_events(
        agent=agent,
        request_id=request_id,
        mars_job_id=mars_job_id,
        dataset_request_id=task.get("request_id"),
        started=started,
        ended=ended,
        workspace_dir=workspace_dir,
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        status=status,
        trace=trace,
        verification_nudges=verification_nudges,
    )
    result = {
        "schema": "mars_agent_cli_request_result_v1",
        "task_sha256": task_sha256,
        "agent": agent,
        "model": model,
        "trace": trace,
        "events": events,
    }
    _write_json(request_run_dir / "result.json", result)
    return result
