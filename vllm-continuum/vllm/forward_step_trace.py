"""Low-overhead per-forward-step tracing for MARS/vLLM experiments."""

from __future__ import annotations

from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator


_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_STEP_COUNTER = itertools.count()
_WRITE_LOCK = threading.Lock()
_CUDA_PROFILER_LOCK = threading.Lock()
_CUDA_PROFILER_THREAD_STARTED = False


def _env_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE_VALUES


def enabled() -> bool:
    return _env_true("STAMP_FORWARD_STEP_TRACE")


def nvtx_enabled() -> bool:
    return enabled() and _env_true("STAMP_FORWARD_STEP_NVTX", "1")


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def next_step_id() -> int:
    return next(_STEP_COUNTER)


def _trace_path() -> Path | None:
    raw = os.environ.get("STAMP_FORWARD_STEP_TRACE_PATH")
    if not raw:
        return None
    return Path(raw).expanduser()


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    try:
        return int(value)
    except Exception:
        return str(value)


def write_record(record: dict[str, Any]) -> None:
    if not enabled():
        return
    path = _trace_path()
    if path is None:
        return

    item = dict(record)
    item.setdefault("ts", time.time())
    item.setdefault("pid", os.getpid())
    line = json.dumps(
        item,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )

    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(line)
            fp.write("\n")


def maybe_start_cuda_profiler_window() -> None:
    """Start an optional cudaProfilerApi capture window inside this CUDA process."""
    if not _env_true("STAMP_CUDA_PROFILER_CAPTURE"):
        return

    global _CUDA_PROFILER_THREAD_STARTED
    with _CUDA_PROFILER_LOCK:
        if _CUDA_PROFILER_THREAD_STARTED:
            return
        _CUDA_PROFILER_THREAD_STARTED = True

    delay_s = max(0.0, _float_env("STAMP_CUDA_PROFILER_DELAY_S", 0.0))
    duration_s = max(0.0, _float_env("STAMP_CUDA_PROFILER_DURATION_S", 0.0))

    def _worker() -> None:
        try:
            if delay_s:
                time.sleep(delay_s)
            import torch

            cudart = torch.cuda.cudart()
            cudart.cudaProfilerStart()
            write_record(
                {
                    "event": "cuda_profiler_start",
                    "delay_s": delay_s,
                    "duration_s": duration_s,
                }
            )
            if duration_s:
                time.sleep(duration_s)
                cudart.cudaProfilerStop()
                write_record(
                    {
                        "event": "cuda_profiler_stop",
                        "delay_s": delay_s,
                        "duration_s": duration_s,
                    }
                )
        except Exception as exc:
            write_record(
                {
                    "event": "cuda_profiler_error",
                    "error": repr(exc),
                    "delay_s": delay_s,
                    "duration_s": duration_s,
                }
            )

    threading.Thread(
        target=_worker,
        name="stamp-cuda-profiler-window",
        daemon=True,
    ).start()


@contextmanager
def nvtx_range(message: str | None) -> Iterator[None]:
    if message:
        maybe_start_cuda_profiler_window()

    if not message or not nvtx_enabled():
        yield
        return

    pushed = False
    try:
        import torch

        torch.cuda.nvtx.range_push(str(message))
        pushed = True
    except Exception:
        pushed = False

    try:
        yield
    finally:
        if pushed:
            try:
                import torch

                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
