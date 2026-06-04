#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import multiprocessing as mp
import os
import json
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from lib import atomic_write_text, ensure_removed, safe_join


@dataclass(frozen=True)
class ToolTaskSpec:
    request_id: str
    round_id: int
    keep_venv: bool = False
    action: Optional[dict[str, Any]] = None
    workspace_dir: Optional[str] = None


@dataclass(frozen=True)
class ToolPoolConfig:
    cpu_cores: list[int]
    venv_root: Path
    tool_env_mode: str = "per_invocation_venv" 
    thread_pool: bool = False 
    sleep_s: float = 3.0
    venv_copies: bool = True
    with_pip: bool = True
    venv_timeout_s: Optional[float] = None
    command_timeout_s: Optional[float] = None
    max_output_tail_chars: int = 4000


def _set_affinity(core: int) -> None:
    try:
        os.sched_setaffinity(0, {core})  
    except Exception:
        pass


def _clamp_threads_single() -> None:
    # HuggingFace tokenizers uses internal thread pools and warns on fork-after-use.
    # For reproducibility and to avoid potential deadlocks, disable it explicitly.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for k in (
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(k, "1")


def _read_tail_text(path: Path, *, max_chars: int) -> Optional[str]:
    try:
        max_bytes = max(256, int(max_chars) * 4)
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            n = f.tell()
            f.seek(max(0, n - max_bytes))
            b = f.read()
        s = b.decode("utf-8", errors="ignore")
        return s[-max_chars:] if len(s) > max_chars else s
    except Exception:
        return None


def _is_hidden(rel: Path) -> bool:
    return any(part.startswith(".") for part in rel.parts)


def _read_file_slice(path: Path, *, offset: int, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    max_bytes = max(256, int(max_chars) * 4)
    with path.open("rb") as f:
        if offset > 0:
            f.seek(int(offset))
        data = f.read(max_bytes)
    text = data.decode("utf-8", errors="ignore")
    return text[:max_chars] if len(text) > max_chars else text


def _list_dir_entries(
    root: Path,
    *,
    depth: int,
    include_hidden: bool,
    max_entries: int,
) -> list[str]:
    if depth < 0:
        depth = 0
    out: list[str] = []
    if root.is_file():
        return [root.name]
    for cur, dirs, files in os.walk(root):
        rel = Path(cur).relative_to(root)
        cur_depth = 0 if rel == Path(".") else len(rel.parts)
        if cur_depth > depth:
            continue
        if cur_depth == depth:
            dirs[:] = []
        dirs.sort()
        files.sort()
        for d in list(dirs):
            rel_path = (rel / d) if rel != Path(".") else Path(d)
            if not include_hidden and _is_hidden(rel_path):
                dirs.remove(d)
                continue
            out.append(f"{rel_path.as_posix()}/")
            if len(out) >= max_entries:
                return out
        for f in files:
            rel_path = (rel / f) if rel != Path(".") else Path(f)
            if not include_hidden and _is_hidden(rel_path):
                continue
            out.append(rel_path.as_posix())
            if len(out) >= max_entries:
                return out
    return out


def _search_files(
    root: Path,
    *,
    pattern: str,
    glob: Optional[str],
    case_sensitive: bool,
    include_hidden: bool,
    max_results: int,
) -> list[str]:
    import re

    results: list[str] = []
    if max_results <= 0:
        return results
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags=flags)
    except re.error:
        regex = None

    def _match(line: str) -> bool:
        if regex is not None:
            return regex.search(line) is not None
        if case_sensitive:
            return pattern in line
        return pattern.lower() in line.lower()

    if root.is_file():
        files = [root]
        base = root.parent
    else:
        base = root
        files = list(root.rglob(glob)) if glob else list(root.rglob("*"))
        files = [p for p in files if p.is_file()]
        files.sort()

    for p in files:
        rel = p.relative_to(base) if base.is_dir() else Path(p.name)
        if not include_hidden and _is_hidden(rel):
            continue
        try:
            data = p.read_bytes()[: 512 * 1024]
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            continue
        for idx, line in enumerate(text.splitlines(), 1):
            if _match(line):
                results.append(f"{rel.as_posix()}:{idx}:{line[:200]}")
                if len(results) >= max_results:
                    return results
    return results


def _tool_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    # Reduce variability from implicit multi-thread libs.
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for k in (
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.setdefault(k, "1")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PIP_REQUIRE_VIRTUALENV", "1")
    env.setdefault("PYTHONNOUSERSITE", "1")
    return env


def _venv_env(base_env: dict[str, str], venv_dir: Path) -> dict[str, str]:
    env = dict(base_env)
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = str(venv_dir / "bin") + os.pathsep + env.get("PATH", "")
    return env


def _check_workspace_venv(base_env: dict[str, str], workspace_dir: Path, *, venv_dir_name: str = ".venv") -> dict[str, str]:
    venv_dir = workspace_dir / venv_dir_name
    if (venv_dir / "bin").is_dir():
        return _venv_env(base_env, venv_dir)
    return dict(base_env)


def _build_install_commands(
    *,
    packages: list[str],
    requirements: Optional[str],
    constraints: Optional[str],
    editable: bool,
    pip_args: list[str],
) -> list[str]:
    cmd: list[str] = ["python", "-m", "pip", "install"]
    if constraints:
        cmd += ["-c", constraints]
    if editable:
        cmd += ["-e"]
    if requirements:
        cmd += ["-r", requirements]
    else:
        cmd += packages
    cmd += pip_args
    return [" ".join(shlex.quote(x) for x in cmd)]


def _commands_for_action(action: dict[str, Any]) -> list[str]:
    tool = str(action.get("tool") or "")
    cmds = action.get("commands")
    if isinstance(cmds, str):
        return [cmds]
    if isinstance(cmds, list):
        return [str(x) for x in cmds]

    if tool == "compile":
        return ["python -m compileall -q ."]
    if tool == "test":
        return ["python -m pytest -q --maxfail=1"]
    if tool == "install":
        packages: list[str] = []
        req = action.get("requirements")
        requirements = req.strip() if isinstance(req, str) and req.strip() else None
        pk = action.get("packages")
        if isinstance(pk, str) and pk.strip():
            packages = [pk.strip()]
        elif isinstance(pk, list):
            packages = [str(x).strip() for x in pk if str(x).strip()]
        constraints = action.get("constraints")
        constraints_s = constraints.strip() if isinstance(constraints, str) and constraints.strip() else None
        editable = bool(action.get("editable", False))
        pip_args = action.get("pip_args")
        if isinstance(pip_args, str) and pip_args.strip():
            pip_args_list = [pip_args.strip()]
        elif isinstance(pip_args, list):
            pip_args_list = [str(x).strip() for x in pip_args if str(x).strip()]
        else:
            pip_args_list = []
        if not requirements and not packages:
            return []
        return _build_install_commands(
            packages=packages,
            requirements=requirements,
            constraints=constraints_s,
            editable=editable,
            pip_args=pip_args_list,
        )
    return []


def _split_shell_segments(cmd: str) -> list[str]:
    parts = re.split(r"(?:&&|\|\||;|\|)", cmd)
    return [p.strip() for p in parts if p.strip()]


def _is_within_path(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _expand_path_token(token: str) -> Optional[Path]:
    t = str(token).strip()
    if not t:
        return None
    if t.startswith("${HOME}"):
        t = t.replace("${HOME}", os.path.expanduser("~"), 1)
    if not (t.startswith("/") or t.startswith("~/") or t == "~" or t.startswith("$HOME/") or t == "$HOME"):
        return None
    expanded = os.path.expandvars(os.path.expanduser(t))
    try:
        return Path(expanded).resolve(strict=False)
    except Exception:
        return Path(expanded)


def _blocked_delete_target_reason(target: str, workspace_dir: Path) -> Optional[str]:
    expanded = _expand_path_token(target)
    if expanded is None:
        return None

    try:
        ws = workspace_dir.resolve(strict=False)
    except Exception:
        ws = workspace_dir
    if _is_within_path(expanded, ws):
        return None

    home = Path.home().resolve(strict=False)
    if expanded == home or _is_within_path(expanded, home):
        return f"deleting home path outside workspace: {target}"

    system_roots = (
        Path("/usr"),
        Path("/etc"),
        Path("/var"),
        Path("/opt"),
        Path("/bin"),
        Path("/sbin"),
        Path("/lib"),
        Path("/lib64"),
        Path("/boot"),
        Path("/dev"),
        Path("/proc"),
        Path("/sys"),
        Path("/home"),
    )
    for root in system_roots:
        if expanded == root or _is_within_path(expanded, root):
            return f"deleting protected system path: {target}"

    parts_lower = {p.lower() for p in expanded.parts}
    if "miniconda3" in parts_lower or "anaconda3" in parts_lower or ".conda" in parts_lower:
        return f"deleting conda-managed path outside workspace: {target}"

    return None


def _blocked_run_command_reason(cmd: str, workspace_dir: Path) -> Optional[str]:
    if str(os.environ.get("MARS_ALLOW_UNSAFE_RUN_CMD", "")).strip() == "1":
        return None

    segments = _split_shell_segments(cmd)
    for seg in segments:
        try:
            tokens = shlex.split(seg, posix=True)
        except Exception:
            continue
        if not tokens:
            continue

        exe = tokens[0].lower()
        seg_l = seg.lower()

        if exe in {"sudo", "doas"} or seg_l.startswith("sudo "):
            return "privilege escalation command is not allowed"

        if exe in {"conda", "mamba", "micromamba"}:
            t = [x.lower() for x in tokens[1:]]
            if "remove" in t and ("env" in t or "-n" in t or "--name" in t):
                return "conda environment removal is blocked"

        if "miniconda3" in seg_l and re.search(r"\s-p\s+(\$HOME|\$\{HOME\}|~|/home/|/opt/)", seg):
            return "installing Miniconda to global/home path is blocked"

        if exe == "rm":
            targets = [x for x in tokens[1:] if not x.startswith("-")]
            for target in targets:
                if target in {"/", "/*", "~", "$HOME", "${HOME}"}:
                    return f"dangerous rm target is blocked: {target}"
                reason = _blocked_delete_target_reason(target, workspace_dir)
                if reason is not None:
                    return reason

        if exe in {"rmdir", "unlink"}:
            targets = [x for x in tokens[1:] if not x.startswith("-")]
            for target in targets:
                reason = _blocked_delete_target_reason(target, workspace_dir)
                if reason is not None:
                    return reason

    return None


def _tool_worker(
    core_id: int,
    task_q: mp.Queue,
    result_q: mp.Queue,
    ready_q: mp.Queue,
    cfg: ToolPoolConfig,
):
    _set_affinity(core_id)
    _clamp_threads_single()
    cfg.venv_root.mkdir(parents=True, exist_ok=True)
    worker_pid = os.getpid()
    try:
        worker_affinity = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        worker_affinity = None
    try:
        ready_q.put({
            "cpu_core": core_id,
            "pid": worker_pid,
            "worker_affinity": worker_affinity,
        })
    except Exception:
        pass

    def _resolve_cwd(workspace_dir: Path, cwd_value: Any) -> tuple[Path, Optional[str], Optional[str]]:
        # Resolve the working directory for tools.
        if not isinstance(cwd_value, str) or not cwd_value.strip():
            return workspace_dir, None, None

        raw = str(cwd_value).strip()
        if raw == "/workspace":
            return workspace_dir, raw, None

        p = Path(raw)
        if p.is_absolute():
            try:
                ws = workspace_dir.resolve(strict=False)
                abs_p = p.expanduser().resolve(strict=False)
                if abs_p == ws or ws in abs_p.parents:
                    return abs_p, None, None
            except Exception as e:
                return workspace_dir, None, f"cwd_error: {type(e).__name__}: {e}"
            return workspace_dir, None, f"cwd_error: absolute cwd outside workspace: {raw}"

        try:
            return safe_join(workspace_dir, raw), None, None
        except Exception as e:
            return workspace_dir, None, f"cwd_error: {type(e).__name__}: {e}"

    while True:
        # The worker owns one CPU slot and processes tasks until it receives the shutdown sentinel.
        task = task_q.get()
        if task is None:
            break

        task_id: str = str(task.get("task_id") or "")
        spec: dict[str, Any] = task.get("spec") if isinstance(task.get("spec"), dict) else {}
        request_id = str(spec.get("request_id") or "")
        round_id = int(spec.get("round_id") or 0)
        keep_venv = bool(spec.get("keep_venv", False))
        enqueued_time_s = spec.get("enqueued_time_s")
        workspace_dir_s = spec.get("workspace_dir")
        workspace_dir = Path(str(workspace_dir_s)) if workspace_dir_s else Path(".")
        try:
            workspace_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            workspace_dir = Path(".")

        action = spec.get("action") if isinstance(spec.get("action"), dict) else None
        
        # Prepare the execution environment
        if action is None:
            tool_type = str(spec.get("tool_type", "venv_sleep"))
            sleep_s_req = spec.get("sleep_s")
            sleep_s = float(sleep_s_req) if isinstance(sleep_s_req, (int, float)) else float(cfg.sleep_s)
            action = {"tool": tool_type, "sleep_s": sleep_s}
        tool = str((action or {}).get("tool") or "venv_sleep")

        t_start = time.time()
        mode = str(getattr(cfg, "tool_env_mode", "per_invocation_venv") or "per_invocation_venv").strip()
        base_env = _tool_subprocess_env()

        def _preexec():
            try:
                os.sched_setaffinity(0, {core_id}) 
            except Exception:
                pass

        t_venv_s: Optional[float] = None
        t_venv_e: Optional[float] = None
        venv_dir: Optional[Path] = None
        venv_rc = 0
        venv_err: Optional[str] = None

        # Run the tool action.
        tool_rc: Optional[int] = None
        tool_err: Optional[str] = None
        t_sleep_s: Optional[float] = None
        t_sleep_e: Optional[float] = None
        command_results: list[dict[str, Any]] = []
        write_file: Optional[dict[str, Any]] = None
        cwd_used: Optional[str] = None
        cwd_sanitized_from: Optional[str] = None

        run_like_tools = ("run_cmd", "run", "compile", "test", "install", "lint", "format", "typecheck")

        try:
            # Dispatch the normalized action.
            if mode == "workspace_venv":
                env = _check_workspace_venv(base_env, workspace_dir)
            else:
                if mode == "per_request_venv":
                    venv_dir = cfg.venv_root / f"venv_{request_id}"
                else:
                    mode = "per_invocation_venv"
                    venv_dir = cfg.venv_root / f"venv_{request_id}_r{round_id}_{task_id}"
                    ensure_removed(venv_dir)

                cmd = [os.sys.executable, "-m", "venv"]
                if cfg.venv_copies:
                    cmd.append("--copies")
                if not cfg.with_pip:
                    cmd.append("--without-pip")
                cmd.append(str(venv_dir))

                assert venv_dir is not None
                if mode == "per_invocation_venv" or (not venv_dir.exists()):
                    t_venv_s = time.time()
                    try:
                        p = subprocess.run(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=cfg.venv_timeout_s,
                            check=False,
                            preexec_fn=_preexec,
                        )
                        venv_rc = int(p.returncode)
                        if p.stderr:
                            venv_err = p.stderr[-4000:]
                    except subprocess.TimeoutExpired as e:
                        venv_rc = 124
                        venv_err = f"TimeoutExpired: {e}"
                    except Exception as e:
                        venv_rc = 125
                        venv_err = f"{type(e).__name__}: {e}"
                    t_venv_e = time.time()

                env = _venv_env(base_env, venv_dir)

            if tool == "venv_sleep":
                sleep_s_req = action.get("sleep_s")
                sleep_s = float(sleep_s_req) if isinstance(sleep_s_req, (int, float)) else float(cfg.sleep_s)
                t_sleep_s = time.time()
                time.sleep(max(0.0, float(sleep_s)))
                t_sleep_e = time.time()
                tool_rc = 0
            elif tool == "venv":
                check_pip = bool(action.get("check_pip", False))
                tool_rc = 0
                if check_pip:
                    try:
                        if mode == "workspace_venv":
                            vpy = workspace_dir / ".venv" / "bin" / "python"
                        else:
                            assert venv_dir is not None
                            vpy = venv_dir / "bin" / "python"
                        p = subprocess.run(
                            [str(vpy), "-m", "pip", "--version"],
                            cwd=str(workspace_dir),
                            env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=10.0,
                            check=False,
                            preexec_fn=_preexec,
                        )
                        tool_rc = int(p.returncode)
                        if p.stderr:
                            tool_err = p.stderr[-4000:]
                    except Exception as e:
                        tool_rc = 125
                        tool_err = f"venv_check_error: {type(e).__name__}: {e}"
            elif tool == "write_file":
                rel = str(action.get("path") or "artifact.txt")
                content = str(action.get("content") or "")
                try:
                    out_path = safe_join(workspace_dir, rel)
                    atomic_write_text(out_path, content)
                    write_file = {"path": rel, "bytes": len(content.encode("utf-8", errors="ignore"))}
                    tool_rc = 0
                except Exception as e:
                    tool_rc = 125
                    tool_err = f"write_file_error: {type(e).__name__}: {e}"
            elif tool in ("read_file", "list_files", "list_dir", "search"):
                rc = 0
                stdout = ""
                stderr = ""
                t0 = time.time()
                try:
                    if tool == "read_file":
                        rel = str(action.get("path") or "").strip()
                        if not rel:
                            raise ValueError("read_file requires path")
                        offset = int(action.get("offset") or 0)
                        max_chars = int(action.get("max_chars") or int(cfg.max_output_tail_chars))
                        max_chars = max(1, min(max_chars, int(cfg.max_output_tail_chars)))
                        p = safe_join(workspace_dir, rel)
                        stdout = _read_file_slice(p, offset=max(0, offset), max_chars=max_chars)
                    elif tool in ("list_files", "list_dir"):
                        rel = str(action.get("path") or "").strip()
                        depth = int(action.get("depth") or (10 if bool(action.get("recursive", False)) else 2))
                        max_entries = int(action.get("max_entries") or 200)
                        include_hidden = bool(action.get("include_hidden", False))
                        root = safe_join(workspace_dir, rel) if rel else workspace_dir
                        entries = _list_dir_entries(
                            root,
                            depth=max(0, depth),
                            include_hidden=include_hidden,
                            max_entries=max(1, max_entries),
                        )
                        stdout = "\n".join(entries)
                    else:
                        pattern = str(action.get("pattern") or "").strip()
                        if not pattern:
                            raise ValueError("search requires pattern")
                        rel = str(action.get("path") or "").strip()
                        glob = action.get("glob")
                        max_results = int(action.get("max_results") or 50)
                        include_hidden = bool(action.get("include_hidden", False))
                        case_sensitive = bool(action.get("case_sensitive", True))
                        root = safe_join(workspace_dir, rel) if rel else workspace_dir
                        matches = _search_files(
                            root,
                            pattern=pattern,
                            glob=str(glob) if isinstance(glob, str) and glob.strip() else None,
                            case_sensitive=case_sensitive,
                            include_hidden=include_hidden,
                            max_results=max(1, max_results),
                        )
                        stdout = "\n".join(matches)
                except Exception as e:
                    rc = 125
                    stderr = f"{tool}_error: {type(e).__name__}: {e}"
                t1 = time.time()
                out_tail = (
                    stdout[-int(cfg.max_output_tail_chars):]
                    if len(stdout) > int(cfg.max_output_tail_chars)
                    else stdout
                )
                err_tail = (
                    stderr[-int(cfg.max_output_tail_chars):]
                    if len(stderr) > int(cfg.max_output_tail_chars)
                    else stderr
                )
                command_results.append({
                    "idx": 0,
                    "cmd": tool,
                    "t_start_s": t0,
                    "t_end_s": t1,
                    "duration_s": float(t1 - t0),
                    "rc": rc,
                    "timed_out": False,
                    "stdout_tail": out_tail,
                    "stderr_tail": err_tail,
                })
                tool_rc = rc
                if rc != 0:
                    tool_err = err_tail
            elif tool in run_like_tools:
                # Shell based tools
                commands = _commands_for_action(action)
                cwd_value = action.get("cwd")
                cwd, cwd_sanitized_from, cwd_err = _resolve_cwd(workspace_dir, cwd_value)
                cwd_used = str(cwd)

                if cwd_err is not None:
                    tool_rc = 125
                    tool_err = cwd_err
                else:
                    timeout_s = action.get("timeout_s")
                    timeout = (
                        float(timeout_s)
                        if isinstance(timeout_s, (int, float))
                        else (float(cfg.command_timeout_s) if cfg.command_timeout_s is not None else None)
                    )
                    log_dir = workspace_dir / ".tool_logs"
                    log_dir.mkdir(parents=True, exist_ok=True)

                    if not commands:
                        tool_rc = 2
                        tool_err = f"{tool}_missing_commands"
                    else:
                        tool_rc = 0
                        for idx, cmd_str in enumerate(commands):
                            cmd_str = str(cmd_str)
                            t0 = time.time()
                            out_p = log_dir / f"{task_id}_{idx}.stdout"
                            err_p = log_dir / f"{task_id}_{idx}.stderr"
                            rc = 0
                            timed_out = False
                            blocked_reason = _blocked_run_command_reason(cmd_str, workspace_dir)
                            if blocked_reason is not None:
                                t1 = time.time()
                                err_tail = f"command_blocked: {blocked_reason}"
                                command_results.append({
                                    "idx": idx,
                                    "cmd": cmd_str,
                                    "t_start_s": t0,
                                    "t_end_s": t1,
                                    "duration_s": float(t1 - t0),
                                    "rc": 126,
                                    "timed_out": False,
                                    "stdout_tail": "",
                                    "stderr_tail": err_tail,
                                })
                                tool_rc = 126
                                tool_err = err_tail
                                break
                            try:
                                with out_p.open("wb") as out_f, err_p.open("wb") as err_f:
                                    p = subprocess.run(
                                        ["bash", "-lc", cmd_str],
                                        cwd=str(cwd),
                                        env=env,
                                        stdout=out_f,
                                        stderr=err_f,
                                        text=False,
                                        timeout=timeout,
                                        check=False,
                                        preexec_fn=_preexec,
                                    )
                                rc = int(p.returncode)
                            except subprocess.TimeoutExpired:
                                timed_out = True
                                rc = 124
                            except Exception as e:
                                rc = 125
                                tool_err = f"run_error: {type(e).__name__}: {e}"
                            t1 = time.time()

                            out_tail = _read_tail_text(out_p, max_chars=int(cfg.max_output_tail_chars))
                            err_tail = _read_tail_text(err_p, max_chars=int(cfg.max_output_tail_chars))
                            try:
                                out_p.unlink(missing_ok=True)
                                err_p.unlink(missing_ok=True)
                            except Exception:
                                pass

                            command_results.append({
                                "idx": idx,
                                "cmd": cmd_str,
                                "t_start_s": t0,
                                "t_end_s": t1,
                                "duration_s": float(t1 - t0),
                                "rc": rc,
                                "timed_out": bool(timed_out),
                                "stdout_tail": out_tail,
                                "stderr_tail": err_tail,
                            })
                            if rc != 0:
                                tool_rc = rc
                                break
            else:
                tool_rc = 2
                tool_err = f"unknown_tool: {tool}"
        except Exception as e:
            tool_rc = 125
            tool_err = f"tool_worker_error: {type(e).__name__}: {e}"
        finally:
            if mode == "per_invocation_venv" and (not keep_venv) and venv_dir is not None:
                shutil.rmtree(venv_dir, ignore_errors=True)

            t_end = time.time()
            res: dict[str, Any] = {
                "task_id": task_id,
                "request_id": request_id,
                "round_id": round_id,
                "tool": tool,
                "tool_env_mode": mode,
                "cpu_core": core_id,
                "worker_pid": worker_pid,
                "worker_affinity": worker_affinity,
                "keep_venv": keep_venv,
                "enqueued_time_s": (float(enqueued_time_s) if isinstance(enqueued_time_s, (int, float)) else None),
                "venv_dir": (str(venv_dir) if (venv_dir is not None and (keep_venv or mode == "per_request_venv")) else None),
                "t_start_s": t_start,
                "t_end_s": t_end,
                "t_venv_start_s": t_venv_s,
                "t_venv_end_s": t_venv_e,
                "t_sleep_start_s": t_sleep_s,
                "t_sleep_end_s": t_sleep_e,
                "venv_rc": venv_rc,
                "venv_err_tail": venv_err,
                "tool_rc": tool_rc,
                "tool_err_tail": tool_err,
                "command_results": (
                    command_results
                    if tool in run_like_tools or tool in ("read_file", "list_files", "list_dir", "search")
                    else None
                ),
                "write_file": write_file,
            }
            if tool in run_like_tools:
                res["cwd"] = cwd_used
                res["cwd_sanitized_from"] = cwd_sanitized_from
            try:
                result_q.put(res)
            except Exception:
                pass


class ToolPool:
    def __init__(self, cfg: ToolPoolConfig):
        self._cfg = cfg
        self._thread_pool = bool(getattr(cfg, "thread_pool", False))

        self._ctx: Optional[mp.context.BaseContext] = None
        self._task_q: Optional[mp.Queue] = None
        self._ready_q: Optional[mp.Queue] = None
        self._result_q: Any = None
        self._workers: list[tuple[int, mp.Process]] = []
        self._workers_ready: list[dict[str, Any]] = []

        self._futures: dict[str, "queue.Queue[dict[str, Any]]"] = {}
        self._futures_lock = threading.Lock()

        self._procs: dict[str, subprocess.Popen] = {}
        self._procs_lock = threading.Lock()
        self._waiters: dict[str, threading.Thread] = {}

        self._stats_lock = threading.Lock()
        self._submitted = 0
        self._completed = 0
        self._ema_queue_wait_s: Optional[float] = None
        self._ema_task_s: Optional[float] = None
        self._ema_tool_s: Optional[float] = None
        self._last_result_time_s: Optional[float] = None

        self._collector_stop = threading.Event()
        self._collector = threading.Thread(target=self._collect_results, name="tool-result-collector", daemon=True)

        if not self._thread_pool:
            self._ctx = mp.get_context("spawn")
            self._task_q = self._ctx.Queue()
            self._result_q = self._ctx.Queue()
            self._ready_q = self._ctx.Queue()

            for core in cfg.cpu_cores:
                p = self._ctx.Process(
                    target=_tool_worker,
                    args=(core, self._task_q, self._result_q, self._ready_q, cfg),
                    daemon=True,
                )
                p.start()
                self._workers.append((core, p))

            # Wait workers ready
            expected = len(self._workers)
            ready: list[dict[str, Any]] = []
            deadline = time.time() + 10.0
            while len(ready) < expected and time.time() < deadline:
                try:
                    msg = self._ready_q.get(timeout=0.2)  
                except Exception:
                    continue
                if isinstance(msg, dict):
                    ready.append(msg)
            self._workers_ready = ready
        else:
            self._result_q = queue.Queue()

        self._collector.start()

    def submit(self, spec: ToolTaskSpec) -> tuple[str, float, "queue.Queue[dict[str, Any]]"]:
        task_id = uuid.uuid4().hex
        enq_t = time.time()
        q: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=1)
        with self._futures_lock:
            self._futures[task_id] = q
        with self._stats_lock:
            self._submitted += 1

        if not self._thread_pool:
            assert self._task_q is not None
            self._task_q.put({
                "task_id": task_id,
                "spec": {
                    "request_id": spec.request_id,
                    "round_id": spec.round_id,
                    "keep_venv": spec.keep_venv,
                    "enqueued_time_s": float(enq_t),
                    "action": (spec.action if isinstance(spec.action, dict) else None),
                    "workspace_dir": spec.workspace_dir,
                },
            })
        else:
            payload = {
                "task_id": task_id,
                "spec": {
                    "request_id": spec.request_id,
                    "round_id": spec.round_id,
                    "keep_venv": spec.keep_venv,
                    "enqueued_time_s": float(enq_t),
                    "action": (spec.action if isinstance(spec.action, dict) else None),
                    "workspace_dir": spec.workspace_dir,
                },
                "cfg": {
                    "cpu_cores": list(self._cfg.cpu_cores),
                    "venv_root": str(self._cfg.venv_root),
                    "tool_env_mode": str(getattr(self._cfg, "tool_env_mode", "per_invocation_venv")),
                    "sleep_s": float(getattr(self._cfg, "sleep_s", 3.0)),
                    "venv_copies": bool(getattr(self._cfg, "venv_copies", True)),
                    "with_pip": bool(getattr(self._cfg, "with_pip", True)),
                    "venv_timeout_s": (float(self._cfg.venv_timeout_s) if self._cfg.venv_timeout_s is not None else None),
                    "command_timeout_s": (float(self._cfg.command_timeout_s) if self._cfg.command_timeout_s is not None else None),
                    "max_output_tail_chars": int(getattr(self._cfg, "max_output_tail_chars", 4000)),
                },
            }
            self._spawn_thread_pool_task(task_id, payload)
        return task_id, enq_t, q

    def stats(self) -> dict[str, Any]:
        with self._futures_lock:
            inflight = len(self._futures)
        with self._stats_lock:
            num_workers = len(self._cfg.cpu_cores) if self._thread_pool else len(self._workers)
            out = {
                "num_workers": int(num_workers),
                "inflight": int(inflight),
                "submitted": int(self._submitted),
                "completed": int(self._completed),
                "ema_queue_wait_s": self._ema_queue_wait_s,
                "ema_task_s": self._ema_task_s,
                "ema_tool_s": self._ema_tool_s,
                "last_result_time_s": self._last_result_time_s,
            }
        return out

    def worker_info(self) -> list[dict[str, Any]]:
        if self._thread_pool:
            return [{
                "mode": "thread_pool",
                "cpu_cores": list(self._cfg.cpu_cores),
            }]
        if self._workers_ready:
            return sorted(self._workers_ready, key=lambda x: int(x.get("cpu_core", 0)))
        out: list[dict[str, Any]] = []
        for core, p in self._workers:
            out.append({"cpu_core": core, "pid": p.pid})
        return out

    def shutdown(self, timeout_s: float = 10.0) -> None:
        if self._thread_pool:
            with self._procs_lock:
                procs = list(self._procs.items())
            t0 = time.time()
            for _, p in procs:
                try:
                    p.terminate()
                except Exception:
                    pass
            for _, p in procs:
                try:
                    p.wait(timeout=max(0.0, timeout_s - (time.time() - t0)))
                except Exception:
                    pass
            for _, p in procs:
                try:
                    if p.poll() is None:
                        p.kill()
                except Exception:
                    pass
            for _, p in procs:
                try:
                    p.wait(timeout=1.0)
                except Exception:
                    pass

            self._collector_stop.set()
            self._collector.join(timeout=2.0)
            return

        # Stop collector after workers finish flushing results
        for _ in self._workers:
            assert self._task_q is not None
            self._task_q.put(None)
        t0 = time.time()
        for _, p in self._workers:
            p.join(timeout=max(0.0, timeout_s - (time.time() - t0)))
        for _, p in self._workers:
            if p.is_alive():
                p.terminate()
        for _, p in self._workers:
            p.join(timeout=1.0)

        self._collector_stop.set()
        self._collector.join(timeout=2.0)

    def _spawn_thread_pool_task(self, task_id: str, payload: dict[str, Any]) -> None:
        worker_script = Path(__file__).with_name("tool_worker_subprocess.py")
        cmd = [os.sys.executable, str(worker_script)]
        input_text = json.dumps(payload, ensure_ascii=False)

        def _preexec():
            try:
                os.sched_setaffinity(0, set(self._cfg.cpu_cores))
            except Exception:
                pass

        try:
            p = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=_preexec,
            )
        except Exception as e:
            err_res = {
                "task_id": task_id,
                "request_id": str(payload.get("spec", {}).get("request_id") or ""),
                "round_id": int(payload.get("spec", {}).get("round_id") or 0),
                "tool": str((payload.get("spec", {}).get("action") or {}).get("tool") or "tool"),
                "cpu_core": None,
                "worker_pid": None,
                "worker_affinity": None,
                "keep_venv": bool(payload.get("spec", {}).get("keep_venv", False)),
                "enqueued_time_s": float(payload.get("spec", {}).get("enqueued_time_s") or 0.0),
                "venv_dir": None,
                "t_start_s": None,
                "t_end_s": None,
                "t_venv_start_s": None,
                "t_venv_end_s": None,
                "t_sleep_start_s": None,
                "t_sleep_end_s": None,
                "venv_rc": 125,
                "venv_err_tail": f"thread_pool_spawn_error: {type(e).__name__}: {e}",
                "tool_rc": 125,
                "tool_err_tail": f"thread_pool_spawn_error: {type(e).__name__}: {e}",
            }
            try:
                self._result_q.put(err_res)
            except Exception:
                pass
            return

        with self._procs_lock:
            self._procs[task_id] = p

        t = threading.Thread(
            target=self._wait_thread_pool_task,
            name=f"tool-thread-pool-wait-{task_id[:8]}",
            args=(task_id, p, input_text),
            daemon=True,
        )
        with self._procs_lock:
            self._waiters[task_id] = t
        t.start()

    def _wait_thread_pool_task(self, task_id: str, p: subprocess.Popen, input_text: str) -> None:
        def _parse_stdout(stdout: str) -> Optional[dict[str, Any]]:
            s = stdout.strip()
            if not s:
                return None
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass
            for line in reversed(s.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    continue
            return None

        stdout = ""
        stderr = ""
        try:
            stdout, stderr = p.communicate(input=input_text)
        except Exception as e:
            try:
                p.kill()
            except Exception:
                pass
            res = {
                "task_id": task_id,
                "cpu_core": None,
                "worker_pid": getattr(p, "pid", None),
                "worker_affinity": None,
                "t_start_s": None,
                "t_end_s": None,
                "venv_rc": 126,
                "venv_err_tail": f"thread_pool_wait_error: {type(e).__name__}: {e}",
                "tool_rc": 126,
                "tool_err_tail": f"thread_pool_wait_error: {type(e).__name__}: {e}",
                "worker_stderr_tail": (stderr[-4000:] if isinstance(stderr, str) else None),
            }
        else:
            parsed = _parse_stdout(stdout)
            if parsed is None:
                res = {
                    "task_id": task_id,
                    "cpu_core": None,
                    "worker_pid": getattr(p, "pid", None),
                    "worker_affinity": None,
                    "t_start_s": None,
                    "t_end_s": None,
                    "venv_rc": 126,
                    "venv_err_tail": "thread_pool_parse_error: invalid_json",
                    "tool_rc": int(p.returncode) if p.returncode is not None else 126,
                    "tool_err_tail": "thread_pool_parse_error: invalid_json",
                    "worker_stdout_tail": (stdout[-4000:] if isinstance(stdout, str) else None),
                    "worker_stderr_tail": (stderr[-4000:] if isinstance(stderr, str) else None),
                }
            else:
                res = parsed
                if res.get("cpu_core") == -1:
                    res["cpu_core"] = None

        try:
            self._result_q.put(res)
        except Exception:
            pass
        finally:
            with self._procs_lock:
                self._procs.pop(task_id, None)
                self._waiters.pop(task_id, None)

    def _collect_results(self) -> None:
        def _update_ema(prev: Optional[float], x: Optional[float], *, alpha: float = 0.2) -> Optional[float]:
            if x is None:
                return prev
            if prev is None:
                return float(x)
            return float(prev) * (1.0 - float(alpha)) + float(x) * float(alpha)

        while not self._collector_stop.is_set():
            try:
                res = self._result_q.get(timeout=0.2)
            except Exception:
                continue
            task_id = str(res.get("task_id", ""))
            if not task_id:
                continue
            q: Optional["queue.Queue[dict[str, Any]]"] = None
            with self._futures_lock:
                q = self._futures.pop(task_id, None)
            if q is not None:
                try:
                    q.put(res, timeout=0.1)
                except Exception:
                    pass

            try:
                enq = float(res.get("enqueued_time_s"))  
                t_task_s = float(res.get("t_task_start_s"))  
                t_task_e = float(res.get("t_task_end_s")) 
            except Exception:
                enq = t_task_s = t_task_e = 0.0
            try:
                t_tool_s = float(res.get("t_tool_start_s")) 
                t_tool_e = float(res.get("t_tool_end_s")) 
            except Exception:
                t_tool_s = t_tool_e = 0.0

            queue_wait_s: Optional[float] = None
            task_s: Optional[float] = None
            tool_s: Optional[float] = None
            if enq > 0 and t_task_s > 0:
                queue_wait_s = max(0.0, float(t_task_s - enq))
            if t_task_e > 0 and t_task_s > 0:
                task_s = max(0.0, float(t_task_e - t_task_s))
            if t_tool_e > 0 and t_tool_s > 0:
                tool_s = max(0.0, float(t_tool_e - t_tool_s))

            with self._stats_lock:
                self._completed += 1
                self._ema_queue_wait_s = _update_ema(self._ema_queue_wait_s, queue_wait_s)
                self._ema_task_s = _update_ema(self._ema_task_s, task_s)
                self._ema_tool_s = _update_ema(self._ema_tool_s, tool_s)
                self._last_result_time_s = time.time()
