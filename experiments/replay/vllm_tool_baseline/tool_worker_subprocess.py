#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from tool_pool import ToolPoolConfig, _tool_worker


def _to_optional_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _to_optional_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return int(x)
    try:
        return int(str(x).strip())
    except Exception:
        return None


def _fallback_result(*, task_id: str, spec: dict[str, Any], err: str) -> dict[str, Any]:
    action = spec.get("action") if isinstance(spec.get("action"), dict) else {}
    tool_name = str((action or {}).get("tool") or "tool")
    return {
        "task_id": task_id,
        "request_id": str(spec.get("request_id") or ""),
        "round_id": int(spec.get("round_id") or 0),
        "tool": tool_name,
        "tool_env_mode": None,
        "cpu_core": None,
        "worker_pid": os.getpid(),
        "worker_affinity": None,
        "keep_venv": bool(spec.get("keep_venv", False)),
        "enqueued_time_s": _to_optional_float(spec.get("enqueued_time_s")),
        "venv_dir": None,
        "t_start_s": None,
        "t_end_s": None,
        "t_venv_start_s": None,
        "t_venv_end_s": None,
        "t_sleep_start_s": None,
        "t_sleep_end_s": None,
        "venv_rc": 125,
        "venv_err_tail": err,
        "tool_rc": 125,
        "tool_err_tail": err,
    }


def main() -> None:
    # Main tool-call process
    t_invocation_start_s = time.time()
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except Exception as e:
        res = _fallback_result(task_id="", spec={}, err=f"payload_json_error: {type(e).__name__}: {e}")
        res["t_start_s"] = float(t_invocation_start_s)
        res["t_end_s"] = float(time.time())
        print(json.dumps(res, ensure_ascii=False), flush=True)
        return

    task_id = str(payload.get("task_id") or "")
    spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else {}
    cfg_d = payload.get("cfg") if isinstance(payload.get("cfg"), dict) else {}

    cpu_cores_raw = cfg_d.get("cpu_cores")
    cpu_cores: list[int] = []
    if isinstance(cpu_cores_raw, list):
        for x in cpu_cores_raw:
            xi = _to_optional_int(x)
            if xi is not None and xi >= 0:
                cpu_cores.append(int(xi))

    venv_root = Path(str(cfg_d.get("venv_root") or "tool_venvs"))
    tool_env_mode = str(cfg_d.get("tool_env_mode") or "per_invocation_venv")
    sleep_s = float(_to_optional_float(cfg_d.get("sleep_s")) or 3.0)
    venv_copies = bool(cfg_d.get("venv_copies", True))
    with_pip = bool(cfg_d.get("with_pip", True))
    venv_timeout_s = _to_optional_float(cfg_d.get("venv_timeout_s"))
    command_timeout_s = _to_optional_float(cfg_d.get("command_timeout_s"))
    max_output_tail_chars = int(_to_optional_int(cfg_d.get("max_output_tail_chars")) or 4000)

    cfg = ToolPoolConfig(
        cpu_cores=cpu_cores,
        venv_root=venv_root,
        tool_env_mode=tool_env_mode,
        thread_pool=True,
        sleep_s=sleep_s,
        venv_copies=venv_copies,
        with_pip=with_pip,
        venv_timeout_s=venv_timeout_s,
        command_timeout_s=command_timeout_s,
        max_output_tail_chars=max_output_tail_chars,
    )

    task_q: mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue()
    ready_q: mp.Queue = mp.Queue()
    task_q.put({"task_id": task_id, "spec": spec})
    task_q.put(None)

    try:
        # Child subprocesses inherit the CPU set applied by the parent process
        _tool_worker(-1, task_q, result_q, ready_q, cfg)
    except Exception as e:
        res = _fallback_result(
            task_id=task_id,
            spec=spec,
            err=f"tool_worker_crash: {type(e).__name__}: {e}",
        )
        res["t_start_s"] = float(t_invocation_start_s)
        res["t_end_s"] = float(time.time())
        print(json.dumps(res, ensure_ascii=False), flush=True)
        return

    try:
        res = result_q.get(timeout=5.0)
    except Exception as e:
        res = _fallback_result(
            task_id=task_id,
            spec=spec,
            err=f"tool_worker_no_result: {type(e).__name__}: {e}",
        )

    if isinstance(res, dict) and res.get("cpu_core") == -1:
        res["cpu_core"] = None
    if isinstance(res, dict) and isinstance(res.get("t_start_s"), (int, float)):
        if float(res["t_start_s"]) > float(t_invocation_start_s):
            res["t_start_s"] = float(t_invocation_start_s)
    elif isinstance(res, dict):
        res["t_start_s"] = float(t_invocation_start_s)
    if isinstance(res, dict) and (not isinstance(res.get("t_end_s"), (int, float))):
        res["t_end_s"] = float(time.time())

    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
