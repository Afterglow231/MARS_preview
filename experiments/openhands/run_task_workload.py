#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import concurrent.futures
import errno
import faulthandler
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from common import (
    JSONLWriter,
    KVEventsCollector,
    atomic_write_json,
    load_tasks_jsonl,
    mean,
    now_s,
    parse_cpu_list,
    percentile,
    set_self_affinity,
)
from request_bootstrap import build_user_request

LOCAL_RUNTIME_BUNDLE_FILES = (
    "command_guard.py",
    "common.py",
    "llm_trace_bridge.py",
    "openhands_runner.py",
    "request_bootstrap.py",
    "request_worker.py",
    "sandbox_shell.sh",
    "tool_trace_bridge.py",
    "trace_writer.py",
)
LOCAL_INPUT_PATH_KEYS = {
    "groundtruth_path",
    "repo_cache_path",
    "src_path",
}


def _safe_print(*args: Any, **kwargs: Any) -> None:
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        try:
            sys.stdout = open(os.devnull, "w")
        except Exception:
            pass
    except OSError as exc:
        if getattr(exc, "errno", None) != errno.EPIPE:
            raise
        try:
            sys.stdout = open(os.devnull, "w")
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run hybrid mas_task_v1 workloads with OpenHands native tools and a vLLM backend."
    )
    ap.add_argument("--tasks", type=str, required=True, help="Path to mas_task_v1 JSONL workload.")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for run artifacts.")
    ap.add_argument(
        "--workspace-root",
        type=str,
        default=None,
        help="Per-request workspaces root. Defaults to <out-dir>/workspaces.",
    )
    ap.add_argument(
        "--allow-workspace-outside-out-dir",
        action="store_true",
        help="Allow --workspace-root outside --out-dir. Disabled by default for strict local isolation.",
    )
    ap.add_argument(
        "--no-localize-task-inputs",
        action="store_true",
        help="Do not copy local task input paths into <out-dir>/_localized_inputs.",
    )
    ap.add_argument("--model", type=str, required=True, help="Model name served by the vLLM endpoint.")
    ap.add_argument("--base-url", type=str, required=True, help="OpenAI-compatible vLLM base URL.")
    ap.add_argument("--api-key", type=str, default="EMPTY", help="API key for the OpenAI-compatible endpoint.")
    ap.add_argument(
        "--openhands-python",
        type=str,
        default=os.environ.get("OPENHANDS_PYTHON") or sys.executable,
        help="Python executable for the OpenHands worker environment.",
    )
    ap.add_argument("--rps", type=float, default=1.0, help="Fixed request arrival rate.")
    ap.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Maximum concurrent OpenHands sessions. Use 0 to disable the outer admission cap.",
    )
    ap.add_argument("--max-requests", type=int, default=None, help="Optional cap on number of tasks to run.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle tasks before dispatch.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed used for shuffling.")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    ap.add_argument("--top-p", type=float, default=None, help="LLM top-p.")
    ap.add_argument("--max-output-tokens", type=int, default=2048, help="Maximum output tokens per LLM call.")
    ap.add_argument("--llm-timeout-s", type=int, default=3000, help="LLM HTTP timeout in seconds.")
    ap.add_argument("--llm-num-retries", type=int, default=3, help="OpenHands/LiteLLM retry count per LLM call.")
    ap.add_argument("--max-iterations", type=int, default=40, help="Maximum OpenHands iterations per request.")
    ap.add_argument(
        "--terminal-no-change-timeout-s",
        type=int,
        default=30,
        help="Terminal no-change timeout passed to OpenHands TerminalTool.",
    )
    ap.add_argument(
        "--terminal-type",
        type=str,
        choices=["tmux", "subprocess"],
        default=None,
        help="Optional terminal backend override for OpenHands TerminalTool.",
    )
    ap.add_argument(
        "--cpu-affinity",
        type=str,
        default=None,
        help="Optional CPU affinity for tool pool.",
    )
    ap.add_argument("--emit-mode", type=str, default="rps", help="Emission mode metadata for parity with baseline.")
    ap.add_argument(
        "--gpu-timeout-s",
        type=float,
        default=60000.0,
        help="Hard per-request worker timeout in seconds. Use 0 to disable.",
    )
    ap.add_argument("--scheduling-policy", type=str, default="fcfs", help="vLLM scheduling policy metadata.")
    ap.add_argument("--kv-events", action="store_true", help="Collect vLLM KV events via ZMQ and dump kv_events.jsonl.")
    ap.add_argument(
        "--kv-events-endpoint",
        type=str,
        default=None,
        help="ZMQ subscriber endpoint for KV events, e.g. tcp://127.0.0.1:5557.",
    )
    ap.add_argument("--thread-pool", type=str, default="on", help="Baseline parity metadata.")
    ap.add_argument("--cpu-gpu", type=str, default=None, help="Baseline parity metadata.")
    ap.add_argument("--cpu-tool", type=str, default=None, help="Baseline parity metadata.")
    ap.add_argument(
        "--autellix-tail-fcfs-after-finished",
        "--autellix-fcfs-after-finished",
        dest="autellix_tail_fcfs_after_finished",
        type=int,
        default=0,
        help="Autellix baseline parity metadata.",
    )
    ap.add_argument("--mars-active-pool-size", type=int, default=8, help="Baseline parity metadata.")
    ap.add_argument(
        "--mars-long-prefill-tokens",
        type=int,
        default=50000,
        help=(
            "OpenHands MARS admission threshold. Requests with estimated "
            "prefill tokens above this value are treated as long requests."
        ),
    )
    ap.add_argument(
        "--mars-cpu-backlog-low",
        type=float,
        default=1.0,
        help="CPU pressure clear threshold; ratio is active OpenHands tool calls divided by tool workers.",
    )
    ap.add_argument(
        "--mars-cpu-backlog-high",
        type=float,
        default=4.0,
        help="CPU pressure trigger threshold; ratio is active OpenHands tool calls divided by tool workers.",
    )
    ap.add_argument("--mars-kv-target-ratio", type=float, default=0.9, help="OpenHands mars control.")
    ap.add_argument("--mars-kv-stat-interval-s", type=float, default=0.2, help="OpenHands mars control.")
    ap.add_argument("--mars-window-min", type=int, default=2, help="OpenHands mars control.")
    ap.add_argument("--mars-window-init", type=int, default=8, help="OpenHands mars control.")
    ap.add_argument("--mars-window-inc", type=int, default=1, help="OpenHands mars control.")
    ap.add_argument("--mars-window-dec-factor", type=float, default=0.7, help="OpenHands mars control.")
    ap.add_argument("--mars-control-interval-s", type=float, default=0.2, help="OpenHands mars control.")
    ap.add_argument("--mars-cpu-queue-wait-high-s", type=float, default=5.0, help="OpenHands mars control.")
    ap.add_argument("--mars-cpu-queue-wait-low-s", type=float, default=2.0, help="OpenHands mars control.")
    ap.add_argument("--mars-kv-max-stale-s", type=float, default=2.0, help="OpenHands mars control.")
    ap.add_argument("--mars-long-max-inflight", type=int, default=1, help="OpenHands mars control.")
    ap.add_argument("--mars-tail-max-inflight", type=int, default=3, help="OpenHands mars tail control.")
    ap.add_argument("--mars-tail-kv-budget-ratio", type=float, default=1.0, help="OpenHands mars tail control.")
    ap.add_argument("--mars-no-kv-active-limit", type=int, default=2, help="OpenHands mars control.")
    ap.add_argument("--prefill-tokenizer", type=str, default=None, help="Optional local tokenizer path for prompt length estimation.")
    return ap.parse_args()


def _prepare_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = load_tasks_jsonl(Path(args.tasks).expanduser().resolve())
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(tasks)
    if args.max_requests is not None:
        tasks = tasks[: args.max_requests]
    return tasks


def _is_within_path(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _require_path_under(path: Path, root: Path, *, label: str) -> None:
    path_r = path.resolve(strict=False)
    root_r = root.resolve(strict=False)
    if path_r == root_r or _is_within_path(path_r, root_r):
        return
    raise SystemExit(
        f"{label} must be inside out_dir for strict OpenHands isolation: "
        f"{path_r} is outside {root_r}. "
        "Use --allow-workspace-outside-out-dir only for explicit debugging."
    )


def _looks_like_url(text: str) -> bool:
    stripped = text.strip()
    return (
        "://" in stripped
        or stripped.startswith("git@")
        or stripped.startswith("ssh:")
    )


def _expand_existing_local_path(value: str) -> Path | None:
    text = str(value).strip()
    if not text or _looks_like_url(text):
        return None
    expanded = os.path.expandvars(os.path.expanduser(text))
    path = Path(expanded)
    if not path.is_absolute():
        return None
    try:
        resolved = path.resolve(strict=False)
    except Exception:
        resolved = path
    if not resolved.exists():
        return None
    return resolved


def _localized_input_name(src: Path) -> str:
    digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:16]
    name = src.name or "input"
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return f"{digest}_{safe_name}"


def _assert_no_symlink_escape(root: Path) -> None:
    if not root.exists():
        return
    root_r = root.resolve(strict=False)
    paths = [root]
    if root.is_dir():
        paths.extend(root.rglob("*"))
    for path in paths:
        if not path.is_symlink():
            continue
        try:
            target = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"Localized input contains dangling symlink: {path}") from exc
        if not _is_within_path(target, root_r):
            raise ValueError(
                f"Localized input symlink escapes its copied root: {path} -> {target}"
            )


def _copy_local_input(src: Path, local_inputs_dir: Path, path_map: dict[str, str]) -> str:
    src_r = src.resolve(strict=True)
    key = str(src_r)
    existing = path_map.get(key)
    if existing:
        return existing

    local_inputs_r = local_inputs_dir.resolve(strict=False)
    if src_r == local_inputs_r or _is_within_path(src_r, local_inputs_r):
        localized = str(src_r)
        path_map[key] = localized
        return localized

    dst = local_inputs_dir / "paths" / _localized_input_name(src_r)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists():
            if tmp.is_dir() and not tmp.is_symlink():
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                tmp.unlink(missing_ok=True)
        try:
            if src_r.is_dir():
                shutil.copytree(src_r, tmp, symlinks=True)
            else:
                tmp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_r, tmp)
            _assert_no_symlink_escape(tmp)
            tmp.replace(dst)
        except Exception:
            if tmp.exists():
                if tmp.is_dir() and not tmp.is_symlink():
                    shutil.rmtree(tmp, ignore_errors=True)
                else:
                    tmp.unlink(missing_ok=True)
            raise
    _assert_no_symlink_escape(dst)
    localized = str(dst.resolve(strict=False))
    path_map[key] = localized
    return localized


def _rewrite_local_input_paths(
    obj: Any,
    *,
    local_inputs_dir: Path,
    path_map: dict[str, str],
    stats: dict[str, int],
) -> Any:
    if isinstance(obj, list):
        return [
            _rewrite_local_input_paths(
                item,
                local_inputs_dir=local_inputs_dir,
                path_map=path_map,
                stats=stats,
            )
            for item in obj
        ]
    if not isinstance(obj, dict):
        return obj

    rewritten: dict[str, Any] = {}
    for key, value in obj.items():
        if key in LOCAL_INPUT_PATH_KEYS and isinstance(value, str):
            src = _expand_existing_local_path(value)
            if src is not None:
                localized = _copy_local_input(src, local_inputs_dir, path_map)
                if localized != value:
                    stats["rewritten_paths"] = int(stats.get("rewritten_paths", 0)) + 1
                rewritten[key] = localized
                continue
        rewritten[key] = _rewrite_local_input_paths(
            value,
            local_inputs_dir=local_inputs_dir,
            path_map=path_map,
            stats=stats,
        )
    return rewritten


def _rewrite_workspace_init_content(
    content: str,
    *,
    local_inputs_dir: Path,
    path_map: dict[str, str],
    stats: dict[str, int],
) -> str:
    try:
        parsed = json.loads(content)
    except Exception:
        return content
    if not isinstance(parsed, (dict, list)):
        return content
    rewritten = _rewrite_local_input_paths(
        parsed,
        local_inputs_dir=local_inputs_dir,
        path_map=path_map,
        stats=stats,
    )
    if rewritten == parsed:
        return content
    stats["rewritten_workspace_init_files"] = (
        int(stats.get("rewritten_workspace_init_files", 0)) + 1
    )
    return json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n"


def _localize_task_inputs(
    tasks: list[dict[str, Any]],
    *,
    local_inputs_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    path_map: dict[str, str] = {}
    stats: dict[str, int] = {
        "rewritten_paths": 0,
        "rewritten_workspace_init_files": 0,
        "unique_copied_paths": 0,
    }
    localized_tasks: list[dict[str, Any]] = []

    for task in tasks:
        rewritten_task = _rewrite_local_input_paths(
            task,
            local_inputs_dir=local_inputs_dir,
            path_map=path_map,
            stats=stats,
        )
        workspace_init = rewritten_task.get("workspace_init")
        if isinstance(workspace_init, list):
            new_workspace_init: list[Any] = []
            for item in workspace_init:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("content"), str)
                ):
                    new_item = dict(item)
                    new_item["content"] = _rewrite_workspace_init_content(
                        str(item["content"]),
                        local_inputs_dir=local_inputs_dir,
                        path_map=path_map,
                        stats=stats,
                    )
                    new_workspace_init.append(new_item)
                else:
                    new_workspace_init.append(item)
            rewritten_task = dict(rewritten_task)
            rewritten_task["workspace_init"] = new_workspace_init
        localized_tasks.append(rewritten_task)

    stats["unique_copied_paths"] = len(path_map)
    return localized_tasks, stats


@dataclass(frozen=True)
class QueuedTask:
    req_i: int
    task: dict[str, Any]
    request_id: str
    arrival_time_s: float
    prefill_tokens_est: int
    seq: int


class PrefillTokenEstimator:
    def __init__(self, tokenizer_name_or_path: str | None = None) -> None:
        self._tokenizer = None
        if tokenizer_name_or_path:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name_or_path,
                    trust_remote_code=True,
                )
            except Exception:
                self._tokenizer = None

    # Estimate prompt prefill length
    def estimate(self, task: dict[str, Any]) -> int:
        prompt = build_user_request(task)
        if self._tokenizer is not None:
            try:
                return int(len(self._tokenizer.encode(prompt)))
            except Exception:
                pass
        return int(max(1, math.ceil(len(prompt) / 4.0)))


class LiveToolStats:
    def __init__(self, *, num_workers: int, ema_alpha: float = 0.2) -> None:
        self.num_workers = max(1, int(num_workers))
        self.ema_alpha = max(0.01, min(1.0, float(ema_alpha)))
        self._paths: dict[str, Path] = {}
        self._offsets: dict[str, int] = {}
        self._active_tool_ids_by_request: dict[str, set[str]] = {}
        self._inflight = 0
        self._ema_queue_wait_s: float | None = None

    def register(self, request_id: str, live_events_path: Path) -> None:
        self._paths[request_id] = live_events_path
        self._offsets.setdefault(request_id, 0)
        self._active_tool_ids_by_request.setdefault(request_id, set())

    def unregister(self, request_id: str) -> None:
        active_ids = self._active_tool_ids_by_request.pop(request_id, set())
        if active_ids:
            self._inflight = max(0, int(self._inflight) - len(active_ids))
        self._paths.pop(request_id, None)
        self._offsets.pop(request_id, None)

    def poll(self) -> None:
        for request_id, path in list(self._paths.items()):
            if not path.exists():
                continue
            offset = int(self._offsets.get(request_id, 0))
            try:
                with path.open("r", encoding="utf-8") as handle:
                    handle.seek(offset)
                    chunk = handle.read()
                    self._offsets[request_id] = handle.tell()
            except FileNotFoundError:
                continue
            if not chunk:
                continue
            for raw_line in chunk.splitlines():
                text = raw_line.strip()
                if not text:
                    continue
                try:
                    event = json.loads(text)
                except Exception:
                    continue
                self._apply_event(request_id, event)

    def _apply_event(self, request_id: str, event: dict[str, Any]) -> None:
        name = str(event.get("event") or "")
        tool_call_id = str(event.get("tool_call_id") or "")
        active_ids = self._active_tool_ids_by_request.setdefault(request_id, set())
        if name == "tool_start":
            if tool_call_id and tool_call_id not in active_ids:
                active_ids.add(tool_call_id)
                self._inflight += 1
            wait_s = event.get("wait_s")
            # Track recent tool dispatch pressure. The EMA fuction smooths short spikes before MARS uses it for CPU-pressure admission control.
            if isinstance(wait_s, (int, float)):
                wait = float(wait_s)
                if self._ema_queue_wait_s is None:
                    self._ema_queue_wait_s = wait
                else:
                    alpha = float(self.ema_alpha)
                    self._ema_queue_wait_s = (
                        (alpha * wait) + ((1.0 - alpha) * float(self._ema_queue_wait_s))
                    )
            return
        if name == "tool_end":
            if tool_call_id and tool_call_id in active_ids:
                active_ids.remove(tool_call_id)
                self._inflight = max(0, int(self._inflight) - 1)

    def stats(self) -> dict[str, Any]:
        return {
            "num_workers": int(self.num_workers),
            "inflight": int(self._inflight),
            "ema_queue_wait_s": self._ema_queue_wait_s,
        }


def _normalize_base_url_root(base_url: str) -> str:
    parts = urlsplit(str(base_url).strip())
    scheme = parts.scheme or "http"
    netloc = parts.netloc or parts.path
    if not netloc:
        raise ValueError(f"Invalid base_url: {base_url!r}")
    return f"{scheme}://{netloc}"


def _http_get_json_with_status(
    url: str, *, timeout_s: float = 3.0
) -> tuple[dict[str, Any] | None, int | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            payload = response.read().decode("utf-8")
            status = int(getattr(response, "status", 200) or 200)
    except urllib.error.HTTPError as exc:
        return None, int(exc.code)
    except Exception:
        return None, None
    try:
        data = json.loads(payload)
    except Exception:
        return None, status
    return (data if isinstance(data, dict) else None), status


def _http_get_text(url: str, *, timeout_s: float = 3.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


_KV_CACHE_STATS_UNSUPPORTED_ROOTS: set[str] = set()


def _fetch_kv_cache_stats(base_url: str, *, timeout_s: float = 3.0) -> dict[str, Any] | None:
    # Query the vLLM KV-cache stats endpoint used by MARS admission control.
    # Some servers do not expose KV-cache stats info. Do not repeatedly probe unsupported backends during serving.
    root = _normalize_base_url_root(base_url)
    if root in _KV_CACHE_STATS_UNSUPPORTED_ROOTS:
        return None
    stats, status = _http_get_json_with_status(f"{root}/v1/kv_cache_stats", timeout_s=timeout_s)
    if status in {404, 501}:
        _KV_CACHE_STATS_UNSUPPORTED_ROOTS.add(root)
        return None
    return stats


def _fetch_metrics_gpu_cache_usage(base_url: str, *, timeout_s: float = 3.0) -> float | None:
    # Fallback KV-pressure signal for MARS when the structured /v1/kv_cache_stats endpoint is unavailable.
    root = _normalize_base_url_root(base_url)
    payload = _http_get_text(f"{root}/metrics", timeout_s=timeout_s)
    if not payload:
        return None
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("vllm:gpu_cache_usage_perc"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = float(parts[-1])
        except Exception:
            continue
        return max(0.0, min(1.0, value / 100.0))
    return None


def _kv_blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    if block_size <= 0:
        return 0
    return int(math.ceil(float(max(0, int(num_tokens))) / float(block_size)))


def _summary(traces: list[dict[str, Any]], *, run: dict[str, Any]) -> dict[str, Any]:
    latencies = [float(t["e2e_latency_s"]) for t in traces if t.get("e2e_latency_s") is not None]
    gpu_totals = [float(t["gpu_total_s"]) for t in traces if t.get("gpu_total_s") is not None]
    tool_totals = [float(t["tool_total_s"]) for t in traces if t.get("tool_total_s") is not None]
    tool_waits = [float(t["tool_wait_total_s"]) for t in traces if t.get("tool_wait_total_s") is not None]
    queue_waits = [float(t["queue_wait_s"]) for t in traces if t.get("queue_wait_s") is not None]
    waits = [float(t["wait_total_s"]) for t in traces if t.get("wait_total_s") is not None]
    llm_call_counts = [len(t.get("llm_calls") or []) for t in traces]
    tool_call_counts = [len(t.get("tool_calls") or []) for t in traces]

    statuses: dict[str, int] = {}
    for trace in traces:
        status = str(trace.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1

    start_ts = [float(t["arrival_time_s"]) for t in traces if t.get("arrival_time_s") is not None]
    end_ts = [float(t["end_time_s"]) for t in traces if t.get("end_time_s") is not None]
    throughput = None
    if start_ts and end_ts and max(end_ts) > min(start_ts):
        throughput = float(len(end_ts) / (max(end_ts) - min(start_ts)))

    return {
        "run": run,
        "count": len(traces),
        "status": statuses,
        "latency_s": {
            "mean": mean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "gpu_total_s": {
            "mean": mean(gpu_totals),
            "p50": percentile(gpu_totals, 50),
            "p95": percentile(gpu_totals, 95),
        },
        "tool_total_s": {
            "mean": mean(tool_totals),
            "p50": percentile(tool_totals, 50),
            "p95": percentile(tool_totals, 95),
        },
        "tool_wait_total_s": {
            "mean": mean(tool_waits),
            "p50": percentile(tool_waits, 50),
            "p95": percentile(tool_waits, 95),
        },
        "queue_wait_s": {
            "mean": mean(queue_waits),
            "p50": percentile(queue_waits, 50),
            "p95": percentile(queue_waits, 95),
        },
        "wait_total_s": {
            "mean": mean(waits),
            "p50": percentile(waits, 50),
            "p95": percentile(waits, 95),
        },
        "llm_call_count": {
            "mean": mean([float(x) for x in llm_call_counts]),
            "p50": percentile([float(x) for x in llm_call_counts], 50),
            "p95": percentile([float(x) for x in llm_call_counts], 95),
        },
        "tool_call_count": {
            "mean": mean([float(x) for x in tool_call_counts]),
            "p50": percentile([float(x) for x in tool_call_counts], 50),
            "p95": percentile([float(x) for x in tool_call_counts], 95),
        },
        "throughput_rps": throughput,
    }


def _write_task_json(path: Path, task: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prepare_local_runtime_bundle(out_dir: Path) -> Path:
    source_dir = Path(__file__).resolve().parent
    missing = [name for name in LOCAL_RUNTIME_BUNDLE_FILES if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing local OpenHands runtime files: "
            + ", ".join(str(source_dir / name) for name in missing)
        )

    bundle_dir = (out_dir / "_runtime_bundle").resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for name in LOCAL_RUNTIME_BUNDLE_FILES:
        shutil.copy2(source_dir / name, bundle_dir / name)

    for name in ("request_worker.py", "sandbox_shell.sh"):
        path = bundle_dir / name
        path.chmod(path.stat().st_mode | 0o111)

    return bundle_dir


def _terminate_worker_process(
    proc: subprocess.Popen[str],
    *,
    terminate_grace_s: float = 5.0,
) -> None:
    if proc.poll() is not None:
        return

    def _signal_group(sig: signal.Signals) -> bool:
        if not hasattr(os, "killpg"):
            return False
        try:
            os.killpg(proc.pid, sig)
            return True
        except ProcessLookupError:
            return True
        except Exception:
            return False

    if not _signal_group(signal.SIGTERM):
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        except Exception:
            pass

    deadline = time.monotonic() + max(0.1, float(terminate_grace_s))
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)

    if proc.poll() is not None:
        return

    if not _signal_group(signal.SIGKILL):
        try:
            proc.kill()
        except ProcessLookupError:
            return
        except Exception:
            return


def _run_request_subprocess(
    *,
    args: argparse.Namespace,
    task: dict[str, Any],
    request_id: str,
    arrival_time_s: float,
    workspace_root: Path,
    request_run_dir: Path,
    runtime_bundle_dir: Path,
) -> dict[str, Any]:
    # Run one OpenHands request in an isolated worker process.
    # The main workload runner keeps scheduling/MARS state, while the worker owns the child processes it spawned as one process group.
    task_json = request_run_dir / "task.json"
    result_json = request_run_dir / "result.json"
    stdout_path = request_run_dir / "stdout.txt"
    stderr_path = request_run_dir / "stderr.txt"
    live_events_path = request_run_dir / "live_events.jsonl"
    _write_task_json(task_json, task)

    worker_script = (runtime_bundle_dir / "request_worker.py").resolve()
    worker_python = Path(args.openhands_python).expanduser()
    cmd = [
        str(worker_python),
        str(worker_script),
        "--task-json",
        str(task_json),
        "--result-json",
        str(result_json),
        "--request-id",
        request_id,
        "--arrival-time-s",
        str(arrival_time_s),
        "--workspace-root",
        str(workspace_root),
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--api-key",
        args.api_key,
        "--temperature",
        str(args.temperature),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--llm-timeout-s",
        str(args.llm_timeout_s),
        "--llm-num-retries",
        str(args.llm_num_retries),
        "--max-iterations",
        str(args.max_iterations),
        "--terminal-no-change-timeout-s",
        str(args.terminal_no_change_timeout_s),
        "--live-events-path",
        str(live_events_path),
    ]
    if args.top_p is not None:
        cmd.extend(["--top-p", str(args.top_p)])
    if args.terminal_type:
        cmd.extend(["--terminal-type", str(args.terminal_type)])

    env = os.environ.copy()
    host_home = env.get("MARS_HOST_HOME") or str(Path.home().resolve())
    env["MARS_HOST_HOME"] = host_home
    env.setdefault("MARS_HOST_CONDA_ROOT", str(Path(host_home) / "miniconda3"))
    env.setdefault("MARS_HOST_VENVS_ROOT", str(Path(host_home) / ".venvs"))
    env.setdefault("MARS_HOST_MODELS_ROOT", str(Path(host_home) / "models"))
    if getattr(args, "local_inputs_dir", None):
        env["MARS_LOCAL_INPUTS_DIR"] = str(args.local_inputs_dir)
    env.setdefault("MARS_REQUIRE_BWRAP", "1")
    env.setdefault("MARS_ALLOW_HOST_MODELS_READ", "0")
    runtime_pythonpath = str(runtime_bundle_dir)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = runtime_pythonpath + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = runtime_pythonpath

    def _failed_result(
        *,
        status: str,
        error_type: str,
        message: str,
        stdout_text: str,
        stderr_text: str,
        returncode: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        end_time_s = now_s()
        error_payload: dict[str, Any] = {
            "type": error_type,
            "message": message,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout_preview": (stdout_text or "")[:2000],
            "stderr_preview": (stderr_text or "")[:2000],
        }
        event_payload: dict[str, Any] = {
            "ts": end_time_s,
            "event": status,
            "request_id": request_id,
            "dataset_request_id": task.get("request_id"),
            "benchmark": task.get("benchmark"),
            "task_id": task.get("task_id"),
        }
        if returncode is not None:
            error_payload["returncode"] = int(returncode)
            event_payload["returncode"] = int(returncode)
        if timeout_s is not None:
            error_payload["timeout_s"] = float(timeout_s)
            event_payload["timeout_s"] = float(timeout_s)
        return {
            "trace": {
                "request_id": request_id,
                "dataset_request_id": task.get("request_id"),
                "benchmark": task.get("benchmark"),
                "task_id": task.get("task_id"),
                "workspace_dir": str((workspace_root / request_id).resolve()),
                "workspace_file_count": 0,
                "arrival_time_s": arrival_time_s,
                "start_time_s": arrival_time_s,
                "end_time_s": end_time_s,
                "queue_wait_s": 0.0,
                "e2e_latency_s": float(max(0.0, end_time_s - arrival_time_s)),
                "gpu_total_s": 0.0,
                "tool_total_s": 0.0,
                "tool_wait_total_s": 0.0,
                "wait_total_s": 0.0,
                "status": status,
                "conversation_status": None,
                "error": error_payload,
                "llm_calls": [],
                "tool_calls": [],
                "rounds": [],
                "orphan_tools": [],
                "agent_messages": [],
                "agent_errors": [],
                "conversation_errors": [],
            },
            "events": [event_payload],
        }

    timeout_s = float(args.gpu_timeout_s)
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(request_run_dir),
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout_text, stderr_text = proc.communicate(timeout=(timeout_s if timeout_s > 0 else None))
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_worker_process(proc)
        stdout_text, stderr_text = proc.communicate()

    stdout_path.write_text(stdout_text or "", encoding="utf-8")
    stderr_path.write_text(stderr_text or "", encoding="utf-8")

    if timed_out:
        return _failed_result(
            status="worker_timeout",
            error_type="WorkerProcessTimeout",
            message=f"worker exceeded timeout after {timeout_s:.1f}s",
            stdout_text=stdout_text or "",
            stderr_text=stderr_text or "",
            returncode=proc.returncode,
            timeout_s=timeout_s,
        )

    if proc.returncode != 0:
        return _failed_result(
            status="worker_error",
            error_type="WorkerProcessError",
            message=f"worker exited with code {proc.returncode}",
            stdout_text=stdout_text or "",
            stderr_text=stderr_text or "",
            returncode=proc.returncode,
        )

    result = json.loads(result_json.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Invalid worker result for {request_id}: expected JSON object")
    return result


def _result_from_future_exception(meta: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    # Wrapper for exceptions raised by the executor future rather than by the worker process.
    # Keep the workload run alive and emit a trace record for the failed request.
    request_id = str(meta.get("request_id") or "")
    arrival_time_s = float(meta.get("arrival_time_s") or now_s())
    end_time_s = now_s()
    request_run_dir = Path(str(meta.get("request_run_dir") or "")).expanduser()
    stdout_path = request_run_dir / "stdout.txt"
    stderr_path = request_run_dir / "stderr.txt"
    err_type = type(exc).__name__
    err_msg = str(exc)

    trace = {
        "request_id": request_id,
        "dataset_request_id": meta.get("dataset_request_id"),
        "benchmark": meta.get("benchmark"),
        "task_id": meta.get("task_id"),
        "workspace_dir": meta.get("workspace_dir"),
        "workspace_file_count": 0,
        "arrival_time_s": arrival_time_s,
        "start_time_s": arrival_time_s,
        "end_time_s": end_time_s,
        "queue_wait_s": 0.0,
        "e2e_latency_s": float(max(0.0, end_time_s - arrival_time_s)),
        "gpu_total_s": 0.0,
        "tool_total_s": 0.0,
        "tool_wait_total_s": 0.0,
        "wait_total_s": 0.0,
        "status": "worker_future_exception",
        "conversation_status": None,
        "error": {
            "type": err_type,
            "message": err_msg,
            "stdout_path": str(stdout_path) if stdout_path.exists() else None,
            "stderr_path": str(stderr_path) if stderr_path.exists() else None,
        },
        "llm_calls": [],
        "tool_calls": [],
        "rounds": [],
        "orphan_tools": [],
        "agent_messages": [],
        "agent_errors": [],
        "conversation_errors": [],
    }
    return {
        "trace": trace,
        "events": [
            {
                "ts": end_time_s,
                "event": "worker_future_exception",
                "request_id": request_id,
                "dataset_request_id": meta.get("dataset_request_id"),
                "benchmark": meta.get("benchmark"),
                "task_id": meta.get("task_id"),
                "exception_type": err_type,
                "message": err_msg,
            }
        ],
    }


def _collect_completed_requests(
    futures: set[concurrent.futures.Future[dict[str, Any]]],
    *,
    traces: list[dict[str, Any]],
    traces_writer: JSONLWriter,
    events_writer: JSONLWriter,
    future_meta: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] | None = None,
    live_tool_stats: LiveToolStats | None = None,
    pending_kv_blocks_by_request_id: dict[str, int] | None = None,
) -> set[concurrent.futures.Future[dict[str, Any]]]:
    # Non-blockingly collect completed request futures, persist their trace and event records.
    if not futures:
        return futures
    done, not_done = concurrent.futures.wait(
        futures,
        timeout=0,
        return_when=concurrent.futures.FIRST_COMPLETED,
    )
    if not done:
        return futures
    for future in done:
        meta = future_meta.pop(future, None) if future_meta is not None else None
        meta = meta or {}
        try:
            result = future.result()
        except concurrent.futures.CancelledError as exc:
            result = _result_from_future_exception(meta, exc)
        except BaseException as exc:
            result = _result_from_future_exception(meta, exc)
        trace = result["trace"]
        request_id = str(trace.get("request_id") or "") or str(meta.get("request_id") or "")
        if live_tool_stats is not None and request_id:
            live_tool_stats.unregister(request_id)
        if pending_kv_blocks_by_request_id is not None and request_id:
            pending_kv_blocks_by_request_id.pop(request_id, None)
        traces.append(trace)
        traces_writer.write(trace)
        for event in sorted(result.get("events") or [], key=lambda item: float(item.get("ts", 0.0))):
            events_writer.write(event)
        _safe_print(
            f"[done] request_id={trace.get('request_id')} status={trace.get('status')} "
            f"e2e_latency_s={trace.get('e2e_latency_s')}",
            flush=True,
        )
    return set(not_done)


def main() -> None:
    args = _parse_args()
    if args.cpu_affinity:
        set_self_affinity(parse_cpu_list(args.cpu_affinity))

    tasks_path = Path(args.tasks).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    workspace_root = (
        Path(args.workspace_root).expanduser().resolve()
        if args.workspace_root
        else (out_dir / "workspaces").resolve()
    )
    if not args.allow_workspace_outside_out_dir:
        _require_path_under(workspace_root, out_dir, label="workspace_root")
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    local_inputs_dir = (out_dir / "_localized_inputs").resolve()
    local_inputs_dir.mkdir(parents=True, exist_ok=True)
    setattr(args, "local_inputs_dir", str(local_inputs_dir))
    os.environ["MARS_LOCAL_INPUTS_DIR"] = str(local_inputs_dir)
    request_runs_root = (out_dir / "_request_runs").resolve()
    request_runs_root.mkdir(parents=True, exist_ok=True)
    runtime_bundle_dir = _prepare_local_runtime_bundle(out_dir)

    tasks = _prepare_tasks(args)
    input_localization_stats = {
        "rewritten_paths": 0,
        "rewritten_workspace_init_files": 0,
        "unique_copied_paths": 0,
    }
    if not args.no_localize_task_inputs:
        tasks, input_localization_stats = _localize_task_inputs(
            tasks,
            local_inputs_dir=local_inputs_dir,
        )
    _safe_print(
        "[input_localization] "
        f"enabled={not bool(args.no_localize_task_inputs)} "
        f"dir={local_inputs_dir} "
        f"unique_copied_paths={input_localization_stats['unique_copied_paths']} "
        f"rewritten_paths={input_localization_stats['rewritten_paths']} "
        f"rewritten_workspace_init_files={input_localization_stats['rewritten_workspace_init_files']}",
        flush=True,
    )
    effective_max_workers = max(
        1,
        len(tasks) if args.max_workers <= 0 else min(args.max_workers, len(tasks)),
    )

    atomic_write_json(
        out_dir / "run_config.json",
        {
            "tasks": str(tasks_path),
            "out_dir": str(out_dir),
            "workspace_root": str(workspace_root),
            "local_inputs_dir": str(local_inputs_dir),
            "input_localization": {
                "enabled": not bool(args.no_localize_task_inputs),
                **input_localization_stats,
            },
            "config": {
                "model": args.model,
                "base_url": args.base_url,
                "rps": args.rps,
                "emit_mode": args.emit_mode,
                "max_workers": args.max_workers,
                "effective_max_workers": effective_max_workers,
                "runtime_bundle_dir": str(runtime_bundle_dir),
                "max_requests": args.max_requests,
                "shuffle": bool(args.shuffle),
                "seed": args.seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_output_tokens": args.max_output_tokens,
                "llm_timeout_s": args.llm_timeout_s,
                "llm_num_retries": args.llm_num_retries,
                "max_iterations": args.max_iterations,
                "terminal_no_change_timeout_s": args.terminal_no_change_timeout_s,
                "terminal_type": args.terminal_type,
                "cpu_affinity": args.cpu_affinity,
                "gpu_timeout_s": args.gpu_timeout_s,
                "scheduling_policy": args.scheduling_policy,
                "kv_events": bool(args.kv_events),
                "kv_events_endpoint": args.kv_events_endpoint,
                "thread_pool": args.thread_pool,
                "cpu_gpu": args.cpu_gpu,
                "cpu_tool": args.cpu_tool,
                "autellix_tail_fcfs_after_finished": args.autellix_tail_fcfs_after_finished,
                "mars_active_pool_size": args.mars_active_pool_size,
                "mars_long_prefill_tokens": args.mars_long_prefill_tokens,
                "mars_cpu_backlog_low": args.mars_cpu_backlog_low,
                "mars_cpu_backlog_high": args.mars_cpu_backlog_high,
                "mars_kv_target_ratio": args.mars_kv_target_ratio,
                "mars_kv_stat_interval_s": args.mars_kv_stat_interval_s,
                "mars_window_min": args.mars_window_min,
                "mars_window_init": args.mars_window_init,
                "mars_window_inc": args.mars_window_inc,
                "mars_window_dec_factor": args.mars_window_dec_factor,
                "mars_control_interval_s": args.mars_control_interval_s,
                "mars_cpu_queue_wait_high_s": args.mars_cpu_queue_wait_high_s,
                "mars_cpu_queue_wait_low_s": args.mars_cpu_queue_wait_low_s,
                "mars_kv_max_stale_s": args.mars_kv_max_stale_s,
                "mars_long_max_inflight": args.mars_long_max_inflight,
                "mars_tail_max_inflight": args.mars_tail_max_inflight,
                "mars_tail_kv_budget_ratio": args.mars_tail_kv_budget_ratio,
                "mars_no_kv_active_limit": args.mars_no_kv_active_limit,
                "prefill_tokenizer": args.prefill_tokenizer,
                "openhands_python": str(Path(args.openhands_python).expanduser()),
                "allow_workspace_outside_out_dir": bool(args.allow_workspace_outside_out_dir),
                "localize_task_inputs": not bool(args.no_localize_task_inputs),
                "browser_enabled": False,
                "wait_semantics": "request_queue_wait + OpenHands native tool dispatch wait",
            },
        },
    )

    start_wall = now_s()
    events_path = out_dir / "events.jsonl"
    traces_path = out_dir / "traces.jsonl"
    events_path.write_text("", encoding="utf-8")
    traces_path.write_text("", encoding="utf-8")
    events_writer = JSONLWriter(events_path)
    traces_writer = JSONLWriter(traces_path)
    kv_collector = None
    if args.kv_events:
        if not args.kv_events_endpoint:
            raise SystemExit("--kv-events requires --kv-events-endpoint")
        kv_collector = KVEventsCollector(
            connect_endpoint=args.kv_events_endpoint,
            out_path=(out_dir / "kv_events.jsonl"),
        )
        kv_collector.start()

    run_start_event = {
        "ts": start_wall,
        "event": "run_start",
        "tasks": str(tasks_path),
        "count": len(tasks),
        "rps": args.rps,
        "max_workers": args.max_workers,
        "effective_max_workers": effective_max_workers,
        "cpu_affinity": args.cpu_affinity,
        "out_dir": str(out_dir),
        "workspace_root": str(workspace_root),
        "local_inputs_dir": str(local_inputs_dir),
        "input_localization": {
            "enabled": not bool(args.no_localize_task_inputs),
            **input_localization_stats,
        },
        "scheduling_policy": args.scheduling_policy,
        "runtime_bundle_dir": str(runtime_bundle_dir),
        "kv_events": bool(args.kv_events),
        "kv_events_endpoint": args.kv_events_endpoint,
    }
    events_writer.write(run_start_event)
    if args.kv_events:
        events_writer.write(
            {
                "ts": now_s(),
                "event": "kv_events_started",
                "endpoint": args.kv_events_endpoint,
                "path": str(out_dir / "kv_events.jsonl"),
            }
        )
    _safe_print(
        f"[run_start] tasks={tasks_path} count={len(tasks)} rps={args.rps} "
        f"max_workers={args.max_workers} effective_max_workers={effective_max_workers}",
        flush=True,
    )

    progress_path = out_dir / "progress.json"
    progress_interval_s = 30.0

    scheduling_policy = str(args.scheduling_policy or "fcfs").lower()
    use_mars = scheduling_policy == "mars" and int(args.mars_active_pool_size) > 0
    prefill_estimator = PrefillTokenEstimator(args.prefill_tokenizer)
    if args.cpu_tool:
        tool_cores = parse_cpu_list(args.cpu_tool)
    elif args.cpu_affinity:
        tool_cores = parse_cpu_list(args.cpu_affinity)
    else:
        tool_cores = list(range(max(1, min(8, effective_max_workers))))
    live_tool_stats = LiveToolStats(num_workers=len(tool_cores))

    traces: list[dict[str, Any]] = []
    futures: set[concurrent.futures.Future[dict[str, Any]]] = set()
    future_meta: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
    pending_kv_blocks_by_request_id: dict[str, int] = {}
    backlog: list[QueuedTask] = []
    next_index = 0
    executor: concurrent.futures.ThreadPoolExecutor | None = None
    fatal_exc: BaseException | None = None
    run_completed = False
    stop_logged = False
    termination_reason = "completed"
    _poll_runtime_state_fn = None
    signal_state: dict[str, Any] = {"requested": False, "signal": None}
    prev_signal_handlers: dict[int, Any] = {}
    last_progress_log_s = start_wall
    last_emit_wall_s = start_wall
    last_done_wall_s = start_wall
    sigusr1_registered = False

    def _progress_snapshot(now_ts: float) -> dict[str, Any]:
        # Build a progress snapshot for logging and shutdown diagnostics.
        oldest_inflight_s = None
        oldest_inflight_request_ids: list[str] = []
        if future_meta:
            oldest = sorted(
                future_meta.values(),
                key=lambda meta: float(meta.get("arrival_time_s") or now_ts),
            )
            oldest_inflight_request_ids = [
                str(meta.get("request_id") or "") for meta in oldest[:5] if meta.get("request_id")
            ]
            oldest_arrival = float(oldest[0].get("arrival_time_s") or now_ts)
            oldest_inflight_s = float(max(0.0, now_ts - oldest_arrival))

        return {
            "ts": float(now_ts),
            "requested_count": int(len(tasks)),
            "emitted_requests": int(next_index),
            "completed_requests": int(len(traces)),
            "inflight_futures": int(len(futures)),
            "pending_backlog": int(len(backlog)),
            "last_emit_age_s": float(max(0.0, now_ts - last_emit_wall_s)),
            "last_done_age_s": float(max(0.0, now_ts - last_done_wall_s)),
            "oldest_inflight_age_s": oldest_inflight_s,
            "oldest_inflight_request_ids": oldest_inflight_request_ids,
            "signal_requested": bool(signal_state["requested"]),
            "termination_reason": termination_reason,
        }

    def _periodic_log_progress(*, force: bool = False) -> None:
        nonlocal last_progress_log_s
        now_ts = now_s()
        if not force and (now_ts - last_progress_log_s) < progress_interval_s:
            return
        snapshot = _progress_snapshot(now_ts)
        atomic_write_json(progress_path, snapshot)
        try:
            events_writer.write({"event": "run_progress", **snapshot})
        except Exception:
            pass
        oldest_inflight_age_s = snapshot["oldest_inflight_age_s"]
        oldest_inflight_text = (
            f"{float(oldest_inflight_age_s):.1f}"
            if isinstance(oldest_inflight_age_s, (int, float))
            else "None"
        )
        _safe_print(
            "[progress] "
            f"emitted={snapshot['emitted_requests']}/{snapshot['requested_count']} "
            f"completed={snapshot['completed_requests']} "
            f"inflight={snapshot['inflight_futures']} "
            f"backlog={snapshot['pending_backlog']} "
            f"last_emit_age_s={snapshot['last_emit_age_s']:.1f} "
            f"last_done_age_s={snapshot['last_done_age_s']:.1f} "
            f"oldest_inflight_age_s={oldest_inflight_text}",
            flush=True,
        )
        last_progress_log_s = now_ts

    def _handle_stop_signal(signum: int, _frame: Any) -> None:
        signal_state["requested"] = True
        try:
            signal_state["signal"] = signal.Signals(signum).name
        except Exception:
            signal_state["signal"] = f"SIG{signum}"

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        prev_signal_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _handle_stop_signal)
    if hasattr(signal, "SIGUSR1"):
        try:
            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
            sigusr1_registered = True
        except Exception:
            sigusr1_registered = False

    try:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=effective_max_workers)

        mars_window_min = max(
            1,
            min(int(args.mars_window_min), int(max(1, args.mars_active_pool_size))),
        )
        mars_active_limit = max(
            int(mars_window_min),
            min(int(args.mars_window_init), int(max(1, args.mars_active_pool_size))),
        )
        mars_cpu_overloaded = False
        mars_kv_overloaded = False
        mars_last_control_mono = 0.0
        mars_kv_cached: dict[str, Any] | None = None
        mars_kv_cached_mono = 0.0
        mars_kv_last_full: dict[str, Any] | None = None
        mars_kv_last_full_mono = 0.0
        mars_kv_reliable_prev: bool | None = None
        mars_kv_source_prev: str | None = None

        def _poll_runtime_state() -> None:
            nonlocal futures
            nonlocal last_done_wall_s
            live_tool_stats.poll()
            prev_done_count = len(traces)
            futures = _collect_completed_requests(
                futures,
                traces=traces,
                traces_writer=traces_writer,
                events_writer=events_writer,
                future_meta=future_meta,
                live_tool_stats=live_tool_stats,
                pending_kv_blocks_by_request_id=pending_kv_blocks_by_request_id,
            )
            if len(traces) > prev_done_count:
                last_done_wall_s = now_s()

        _poll_runtime_state_fn = _poll_runtime_state

        def _submit_admitted_request(
            entry: QueuedTask,
            *,
            kv_blocks_cost: int | None = None,
            is_long_task: bool = False,
            admission_meta: dict[str, Any] | None = None,
        ) -> None:
            assert executor is not None
            request_run_dir = (request_runs_root / entry.request_id).resolve()
            live_events_path = request_run_dir / "live_events.jsonl"
            future = executor.submit(
                _run_request_subprocess,
                args=args,
                task=entry.task,
                request_id=entry.request_id,
                arrival_time_s=entry.arrival_time_s,
                workspace_root=workspace_root,
                request_run_dir=request_run_dir,
                runtime_bundle_dir=runtime_bundle_dir,
            )
            futures.add(future)
            future_meta[future] = {
                "request_id": entry.request_id,
                "dataset_request_id": entry.task.get("request_id"),
                "benchmark": entry.task.get("benchmark"),
                "task_id": entry.task.get("task_id"),
                "arrival_time_s": entry.arrival_time_s,
                "workspace_dir": str((workspace_root / entry.request_id).resolve()),
                "request_run_dir": str(request_run_dir),
                "is_long": bool(is_long_task),
            }
            live_tool_stats.register(entry.request_id, live_events_path)
            if kv_blocks_cost is not None and kv_blocks_cost > 0:
                pending_kv_blocks_by_request_id[entry.request_id] = int(kv_blocks_cost)
            if admission_meta is not None:
                payload = {
                    "ts": now_s(),
                    "event": "mars_admit",
                    "policy": scheduling_policy,
                    "request_id": entry.request_id,
                    "dataset_request_id": entry.task.get("request_id"),
                    "benchmark": entry.task.get("benchmark"),
                    "task_id": entry.task.get("task_id"),
                    "prefill_tokens_est": int(entry.prefill_tokens_est),
                    "is_long": bool(is_long_task),
                    "kv_blocks_est": int(kv_blocks_cost) if kv_blocks_cost is not None else None,
                }
                payload.update(admission_meta)
                events_writer.write(payload)

        def _process_mars_submissions() -> None:
            # External Admission Controller of MARS. Each call polls CPU/tool pressure and GPU KV-cache pressure, updates the external
            # admission window, selects backlog requests that fit the current budget, and submits them to the worker pool.
            nonlocal mars_active_limit
            nonlocal mars_cpu_overloaded
            nonlocal mars_kv_overloaded
            nonlocal mars_last_control_mono
            nonlocal mars_kv_cached
            nonlocal mars_kv_cached_mono
            nonlocal mars_kv_last_full
            nonlocal mars_kv_last_full_mono
            nonlocal mars_kv_reliable_prev
            nonlocal mars_kv_source_prev

            if not use_mars or not backlog:
                return

            # Use live tool inflight count and smoothed tool queue waiting time to decide whether the tool side is overloaded.
            cpu_stats = live_tool_stats.stats()
            num_workers = int(cpu_stats.get("num_workers") or 0) or max(1, len(tool_cores))
            inflight = int(cpu_stats.get("inflight") or 0)
            backlog_ratio = float(inflight) / float(max(1, num_workers))
            ema_wait = cpu_stats.get("ema_queue_wait_s")
            ema_wait_s = float(ema_wait) if isinstance(ema_wait, (int, float)) else None

            cpu_healthy_for_growth = backlog_ratio <= float(args.mars_cpu_backlog_low) and (
                ema_wait_s is None or ema_wait_s <= float(args.mars_cpu_queue_wait_low_s)
            )
            cpu_cap_ratio = max(
                0.1,
                (
                    float(args.mars_cpu_backlog_low)
                    + float(args.mars_cpu_backlog_high)
                )
                / 2.0,
            )
            cpu_window_cap = int(math.ceil(float(num_workers) * float(cpu_cap_ratio)))
            cpu_window_cap = max(
                int(mars_window_min),
                min(int(args.mars_active_pool_size), int(cpu_window_cap)),
            )

            prev_cpu_overloaded = bool(mars_cpu_overloaded)
            cpu_overloaded = prev_cpu_overloaded
            if prev_cpu_overloaded:
                if backlog_ratio <= float(args.mars_cpu_backlog_low) and (
                    ema_wait_s is None or ema_wait_s <= float(args.mars_cpu_queue_wait_low_s)
                ):
                    cpu_overloaded = False
            else:
                if backlog_ratio >= float(args.mars_cpu_backlog_high) or (
                    ema_wait_s is not None
                    and ema_wait_s >= float(args.mars_cpu_queue_wait_high_s)
                ):
                    cpu_overloaded = True
            if cpu_overloaded != prev_cpu_overloaded:
                mars_cpu_overloaded = bool(cpu_overloaded)
                events_writer.write(
                    {
                        "ts": now_s(),
                        "event": "mars_cpu_pressure",
                        "overloaded": bool(cpu_overloaded),
                        "backlog_ratio": float(backlog_ratio),
                        "inflight": int(inflight),
                        "num_workers": int(num_workers),
                        "ema_queue_wait_s": ema_wait_s,
                        "high_ratio": float(args.mars_cpu_backlog_high),
                        "low_ratio": float(args.mars_cpu_backlog_low),
                        "high_wait_s": float(args.mars_cpu_queue_wait_high_s),
                        "low_wait_s": float(args.mars_cpu_queue_wait_low_s),
                    }
                )

            # KV telemetry collection
            now_mono = time.perf_counter()
            if mars_kv_cached is None or (
                (now_mono - mars_kv_cached_mono) >= float(args.mars_kv_stat_interval_s)
            ):
                kv_source = "none"
                fresh_kv = _fetch_kv_cache_stats(args.base_url) or None
                if fresh_kv is not None:
                    mars_kv_cached = fresh_kv
                    kv_source = "kv_cache_stats"
                else:
                    gpu_cache_usage = _fetch_metrics_gpu_cache_usage(args.base_url)
                    if gpu_cache_usage is not None:
                        mars_kv_cached = {"usage": float(gpu_cache_usage)}
                        kv_source = "metrics_usage"
                    else:
                        mars_kv_cached = None
                if isinstance(mars_kv_cached, dict):
                    try:
                        block_size_candidate = int(mars_kv_cached.get("block_size"))
                        num_gpu_blocks_candidate = int(mars_kv_cached.get("num_gpu_blocks"))
                        num_free_blocks_candidate = int(mars_kv_cached.get("num_free_blocks"))
                    except Exception:
                        block_size_candidate = None
                        num_gpu_blocks_candidate = None
                        num_free_blocks_candidate = None
                    if (
                        isinstance(block_size_candidate, int)
                        and block_size_candidate > 0
                        and isinstance(num_gpu_blocks_candidate, int)
                        and num_gpu_blocks_candidate > 0
                        and isinstance(num_free_blocks_candidate, int)
                        and num_free_blocks_candidate >= 0
                    ):
                        mars_kv_last_full = dict(mars_kv_cached)
                        mars_kv_last_full_mono = now_mono
                if mars_kv_cached is None and mars_kv_last_full is not None:
                    last_full_age_s = float(now_mono - mars_kv_last_full_mono)
                    if last_full_age_s <= float(args.mars_kv_max_stale_s):
                        mars_kv_cached = dict(mars_kv_last_full)
                        mars_kv_cached["_kv_source"] = "stale_last_full"
                        kv_source = "stale_last_full"
                elif (
                    isinstance(mars_kv_cached, dict)
                    and "usage" in mars_kv_cached
                    and mars_kv_last_full is not None
                ):
                    last_full_age_s = float(now_mono - mars_kv_last_full_mono)
                    if last_full_age_s <= float(args.mars_kv_max_stale_s):
                        merged_stats = dict(mars_kv_last_full)
                        merged_stats["usage"] = mars_kv_cached["usage"]
                        merged_stats["_kv_source"] = "metrics_usage+last_full"
                        mars_kv_cached = merged_stats
                        kv_source = "metrics_usage+last_full"
                if isinstance(mars_kv_cached, dict) and "_kv_source" not in mars_kv_cached:
                    mars_kv_cached["_kv_source"] = kv_source
                mars_kv_cached_mono = now_mono

            kv_stats = mars_kv_cached
            block_size = None
            budget_blocks = None
            kv_usage = None
            num_free_blocks = None
            num_gpu_blocks = None
            kv_source = "none"
            if isinstance(kv_stats, dict):
                kv_source = str(kv_stats.get("_kv_source") or "unknown")
                try:
                    if kv_stats.get("block_size") is not None:
                        block_size = int(kv_stats.get("block_size"))
                except Exception:
                    block_size = None
                try:
                    if kv_stats.get("num_gpu_blocks") is not None:
                        num_gpu_blocks = int(kv_stats.get("num_gpu_blocks"))
                except Exception:
                    num_gpu_blocks = None
                try:
                    if kv_stats.get("num_free_blocks") is not None:
                        num_free_blocks = int(kv_stats.get("num_free_blocks"))
                except Exception:
                    num_free_blocks = None
                try:
                    if isinstance(kv_stats.get("usage"), (int, float)):
                        kv_usage = float(kv_stats.get("usage"))
                except Exception:
                    kv_usage = None
                if kv_usage is None and num_gpu_blocks and num_free_blocks is not None and num_gpu_blocks > 0:
                    kv_usage = max(
                        0.0,
                        min(
                            1.0,
                            1.0 - (float(num_free_blocks) / float(num_gpu_blocks)),
                        ),
                    )
                if num_gpu_blocks is not None and num_free_blocks is not None:
                    kv_target = max(0.0, min(0.999, float(args.mars_kv_target_ratio)))
                    reserve_blocks = int(math.ceil(float(num_gpu_blocks) * (1.0 - kv_target)))
                    budget_blocks = max(0, int(num_free_blocks) - int(reserve_blocks))
            kv_telemetry_reliable = bool(
                block_size is not None
                and block_size > 0
                and num_gpu_blocks is not None
                and num_gpu_blocks > 0
                and num_free_blocks is not None
                and num_free_blocks >= 0
            )

            # Prepare KV usage info for the adaptive window.
            kv_usage_high = max(0.1, min(0.999, float(args.mars_kv_target_ratio)))
            kv_usage_low = max(0.0, min(float(kv_usage_high) - 0.01, float(kv_usage_high) - 0.05))
            kv_window_cap = int(args.mars_active_pool_size)
            if not kv_telemetry_reliable:
                kv_window_cap = max(
                    1,
                    min(int(args.mars_no_kv_active_limit), int(args.mars_active_pool_size)),
                )
            elif kv_usage is not None:
                if kv_usage <= float(kv_usage_low):
                    kv_window_cap = int(args.mars_active_pool_size)
                elif kv_usage >= float(kv_usage_high):
                    kv_window_cap = int(mars_window_min)
                else:
                    span = max(1e-6, float(kv_usage_high) - float(kv_usage_low))
                    frac = max(0.0, min(1.0, (float(kv_usage) - float(kv_usage_low)) / span))
                    kv_window_cap_f = float(args.mars_active_pool_size) - frac * float(
                        int(args.mars_active_pool_size) - int(mars_window_min)
                    )
                    kv_window_cap = int(round(kv_window_cap_f))
                    kv_window_cap = max(
                        int(mars_window_min),
                        min(int(args.mars_active_pool_size), int(kv_window_cap)),
                    )

            prev_kv_overloaded = bool(mars_kv_overloaded)
            kv_overloaded = (not kv_telemetry_reliable) or prev_kv_overloaded
            if kv_telemetry_reliable and kv_usage is not None:
                if prev_kv_overloaded:
                    if float(kv_usage) <= float(kv_usage_low):
                        kv_overloaded = False
                else:
                    if float(kv_usage) >= float(kv_usage_high):
                        kv_overloaded = True
            elif not kv_telemetry_reliable:
                kv_overloaded = True
            if (
                kv_telemetry_reliable != mars_kv_reliable_prev
                or kv_source != mars_kv_source_prev
            ):
                mars_kv_reliable_prev = bool(kv_telemetry_reliable)
                mars_kv_source_prev = str(kv_source)
                events_writer.write(
                    {
                        "ts": now_s(),
                        "event": "mars_kv_telemetry",
                        "reliable": bool(kv_telemetry_reliable),
                        "source": str(kv_source),
                        "kv_usage": float(kv_usage) if kv_usage is not None else None,
                        "kv_num_free_blocks": num_free_blocks,
                        "kv_num_gpu_blocks": num_gpu_blocks,
                        "kv_block_size": block_size,
                    }
                )
            if kv_overloaded != prev_kv_overloaded:
                mars_kv_overloaded = bool(kv_overloaded)
                events_writer.write(
                    {
                        "ts": now_s(),
                        "event": "mars_kv_pressure",
                        "overloaded": bool(kv_overloaded),
                        "kv_telemetry_reliable": bool(kv_telemetry_reliable),
                        "kv_source": str(kv_source),
                        "kv_usage": float(kv_usage) if kv_usage is not None else None,
                        "kv_usage_high": float(kv_usage_high),
                        "kv_usage_low": float(kv_usage_low),
                        "kv_window_cap": int(kv_window_cap),
                        "kv_budget_blocks": int(budget_blocks) if budget_blocks is not None else None,
                        "kv_num_free_blocks": num_free_blocks,
                        "kv_num_gpu_blocks": num_gpu_blocks,
                    }
                )


            # Adaptive window control. Shrink multiplicatively under CPU/KV pressure and grow additively only when both sides look healthy.
            if mars_last_control_mono <= 0.0 or (
                (now_mono - mars_last_control_mono) >= float(args.mars_control_interval_s)
            ):
                prev_limit = int(mars_active_limit)
                new_limit = prev_limit
                admission_overloaded = bool(cpu_overloaded) or bool(kv_overloaded)
                if admission_overloaded:
                    new_limit = max(
                        int(mars_window_min),
                        int(
                            math.floor(
                                float(prev_limit) * float(args.mars_window_dec_factor)
                            )
                        ),
                    )
                else:
                    if (
                        cpu_healthy_for_growth
                        and not kv_overloaded
                        and kv_telemetry_reliable
                        and (kv_usage is None or float(kv_usage) <= float(kv_usage_low))
                        and (budget_blocks is None or int(budget_blocks) > 0)
                    ):
                        new_limit = min(
                            int(args.mars_active_pool_size),
                            int(prev_limit + int(args.mars_window_inc)),
                        )
                new_limit = max(
                    int(mars_window_min),
                    min(int(args.mars_active_pool_size), int(new_limit)),
                )
                new_limit = max(int(mars_window_min), min(int(cpu_window_cap), int(new_limit)))
                new_limit = max(int(mars_window_min), min(int(kv_window_cap), int(new_limit)))
                if new_limit != prev_limit:
                    mars_active_limit = int(new_limit)
                    events_writer.write(
                        {
                            "ts": now_s(),
                            "event": "mars_window_update",
                            "prev": int(prev_limit),
                            "new": int(new_limit),
                            "cpu_window_cap": int(cpu_window_cap),
                            "kv_window_cap": int(kv_window_cap),
                            "cpu_healthy_for_growth": bool(cpu_healthy_for_growth),
                            "overloaded": bool(admission_overloaded),
                            "cpu_overloaded": bool(cpu_overloaded),
                            "kv_overloaded": bool(kv_overloaded),
                            "backlog_ratio": float(backlog_ratio),
                            "inflight": int(inflight),
                            "num_workers": int(num_workers),
                            "ema_queue_wait_s": ema_wait_s,
                            "kv_usage": float(kv_usage) if kv_usage is not None else None,
                            "kv_budget_blocks": int(budget_blocks) if budget_blocks is not None else None,
                        }
                    )
                mars_last_control_mono = float(now_mono)

            active_limit = min(int(mars_active_limit), int(cpu_window_cap), int(kv_window_cap))
            slots = max(0, int(active_limit) - int(len(futures)))
            if slots <= 0:
                return

            long_threshold = int(args.mars_long_prefill_tokens or 0)
            entries = list(backlog)
            long_entries = [
                entry for entry in entries if long_threshold > 0 and int(entry.prefill_tokens_est) > long_threshold
            ]
            short_entries = [
                entry for entry in entries if not (long_threshold > 0 and int(entry.prefill_tokens_est) > long_threshold)
            ]
            backlog_total = len(entries)
            backlog_long = len(long_entries)
            backlog_short = len(short_entries)
            pending_kv_blocks_total = sum(int(v) for v in pending_kv_blocks_by_request_id.values())
            pending_long_inflight = sum(
                1 for meta in future_meta.values() if bool(meta.get("is_long"))
            )
            tracked_inflight = sum(
                1
                for meta in future_meta.values()
                if str(meta.get("request_id") or "") in pending_kv_blocks_by_request_id
            )
            untracked_inflight = max(0, int(len(futures)) - int(tracked_inflight))
            max_long_inflight = max(0, int(args.mars_long_max_inflight))

            tail_mode = bool(
                long_threshold > 0 and backlog_long > 0 and backlog_short == 0 and kv_telemetry_reliable
            )
            if tail_mode:
                max_long_inflight = max(0, int(args.mars_tail_max_inflight))

            admission_overloaded = bool(cpu_overloaded) or bool(kv_overloaded)
            low_pressure_fcfs = bool(
                (not tail_mode)
                and (not admission_overloaded)
                and bool(cpu_healthy_for_growth)
                and bool(kv_telemetry_reliable)
                and (kv_usage is None or float(kv_usage) <= float(kv_usage_low))
                and (budget_blocks is None or int(budget_blocks) > 0)
                and (int(backlog_total) <= int(slots))
            )

            # Choose the request ordering policy for this admission round based on telemetry reliability, tail state, and current pressure.
            if not kv_telemetry_reliable:
                selection_mode = "no_kv_short_only"
            elif tail_mode:
                selection_mode = "tail_kv_fit"
            elif low_pressure_fcfs:
                selection_mode = "low_pressure_fcfs"
            else:
                selection_mode = "cpu_overloaded_long_first" if bool(cpu_overloaded) else "short_first"

            budget_remaining = None
            tail_budget_blocks = None
            if tail_mode and num_gpu_blocks is not None:
                tail_budget_blocks = max(
                    0,
                    int(
                        math.floor(
                            float(num_gpu_blocks)
                            * max(0.0, min(1.0, float(args.mars_tail_kv_budget_ratio)))
                        )
                    ),
                )
                budget_remaining = max(0, int(tail_budget_blocks) - int(pending_kv_blocks_total))
            elif budget_blocks is not None:
                budget_remaining = max(0, int(budget_blocks) - int(pending_kv_blocks_total))

            if tail_mode:
                slots = min(slots, max(0, int(max_long_inflight) - int(pending_long_inflight)))
                if int(untracked_inflight) > 0:
                    slots = min(slots, 1)
                if slots <= 0:
                    return

            if tail_mode:
                candidates = sorted(long_entries, key=lambda entry: int(entry.seq))
            elif low_pressure_fcfs:
                candidates = sorted(entries, key=lambda entry: int(entry.seq))
            elif cpu_overloaded:
                candidates = sorted(
                    entries,
                    key=lambda entry: (-int(entry.prefill_tokens_est), int(entry.seq)),
                )
            else:
                candidates = sorted(
                    entries,
                    key=lambda entry: (
                        int(long_threshold > 0 and int(entry.prefill_tokens_est) > long_threshold),
                        int(entry.prefill_tokens_est),
                        int(entry.seq),
                    ),
                )

            selected: list[tuple[QueuedTask, bool, int | None, int | None]] = []
            budget_remaining_sel = int(budget_remaining) if budget_remaining is not None else None
            selected_long_count = 0
            selected_request_ids: set[str] = set()
            for entry in candidates:
                if len(selected) >= int(slots):
                    break
                is_long = bool(long_threshold > 0 and int(entry.prefill_tokens_est) > long_threshold)
                if is_long:
                    if not kv_telemetry_reliable:
                        continue
                    if int(pending_long_inflight + selected_long_count) >= int(max_long_inflight):
                        continue
                kv_blocks_cost = (
                    _kv_blocks_for_tokens(int(entry.prefill_tokens_est), int(block_size))
                    if (is_long and block_size is not None)
                    else None
                )
                if budget_remaining_sel is not None and kv_blocks_cost is not None:
                    if kv_blocks_cost > int(budget_remaining_sel):
                        continue
                    budget_remaining_sel = max(0, int(budget_remaining_sel) - int(kv_blocks_cost))
                selected.append(
                    (
                        entry,
                        bool(is_long),
                        int(kv_blocks_cost) if kv_blocks_cost is not None else None,
                        int(budget_remaining_sel) if budget_remaining_sel is not None else None,
                    )
                )
                selected_request_ids.add(entry.request_id)
                if is_long:
                    selected_long_count += 1

            if (
                tail_mode
                and int(len(selected)) < int(slots)
                and budget_remaining_sel is not None
                and int(budget_remaining_sel) > 0
                and block_size is not None
            ):
                tail_fill_candidates = sorted(
                    (
                        (
                            _kv_blocks_for_tokens(int(entry.prefill_tokens_est), int(block_size)),
                            int(entry.seq),
                            entry,
                        )
                        for entry in candidates
                        if entry.request_id not in selected_request_ids
                    ),
                    key=lambda item: (int(item[0]), int(item[1])),
                )
                for kv_blocks_cost, _seq, entry in tail_fill_candidates:
                    if len(selected) >= int(slots):
                        break
                    if int(pending_long_inflight + selected_long_count) >= int(max_long_inflight):
                        break
                    if int(kv_blocks_cost) > int(budget_remaining_sel):
                        continue
                    budget_remaining_sel = max(0, int(budget_remaining_sel) - int(kv_blocks_cost))
                    selected.append(
                        (
                            entry,
                            True,
                            int(kv_blocks_cost),
                            int(budget_remaining_sel),
                        )
                    )
                    selected_request_ids.add(entry.request_id)
                    selected_long_count += 1

            if not selected:
                return

            # Commit the selected requests, submit to workers.
            backlog[:] = [entry for entry in backlog if entry.request_id not in selected_request_ids]

            for entry, is_long, kv_blocks_cost, budget_remaining_after in selected:
                _submit_admitted_request(
                    entry,
                    kv_blocks_cost=kv_blocks_cost,
                    is_long_task=is_long,
                    admission_meta={
                        "active_limit": int(active_limit),
                        "cpu_window_cap": int(cpu_window_cap),
                        "kv_window_cap": int(kv_window_cap),
                        "selection_mode": str(selection_mode),
                        "backlog_total": int(backlog_total),
                        "backlog_short": int(backlog_short),
                        "backlog_long": int(backlog_long),
                        "pending_long_inflight": int(pending_long_inflight),
                        "cpu_backlog_ratio": float(backlog_ratio),
                        "cpu_inflight": int(inflight),
                        "cpu_num_workers": int(num_workers),
                        "cpu_ema_queue_wait_s": ema_wait_s,
                        "kv_telemetry_reliable": bool(kv_telemetry_reliable),
                        "kv_source": str(kv_source),
                        "kv_usage": float(kv_usage) if kv_usage is not None else None,
                        "kv_num_free_blocks": num_free_blocks,
                        "kv_num_gpu_blocks": num_gpu_blocks,
                        "pending_kv_blocks_total": int(pending_kv_blocks_total),
                        "kv_budget_blocks": int(budget_blocks) if budget_blocks is not None else None,
                        "tail_kv_budget_blocks": int(tail_budget_blocks) if tail_budget_blocks is not None else None,
                        "kv_budget_remaining": budget_remaining_after,
                        "tail_max_inflight": int(max_long_inflight) if tail_mode else None,
                        "tail_untracked_inflight": int(untracked_inflight) if tail_mode else None,
                        "tail_mode": bool(tail_mode),
                    },
                )

        while True:
            # Loop to refresh live worker states 
            _poll_runtime_state()
            _periodic_log_progress()

            # Stop accepting new work after a termination signal
            if signal_state["requested"]:
                if not stop_logged:
                    stop_logged = True
                    termination_reason = f"signal:{signal_state.get('signal') or 'SIGTERM'}"
                    events_writer.write(
                        {
                            "ts": now_s(),
                            "event": "run_stop_requested",
                            "signal": signal_state.get("signal"),
                            "emitted_requests": int(next_index),
                            "completed_requests": int(len(traces)),
                            "inflight_futures": int(len(futures)),
                            "backlog_requests": int(len(backlog)),
                        }
                    )
                    _safe_print(
                        f"[stop] signal={signal_state.get('signal')} emitted={next_index} "
                        f"completed={len(traces)} inflight={len(futures)} backlog={len(backlog)}",
                        flush=True,
                    )
                break

            # Emit all requests whose scheduled arrival time has passed.
            now = now_s()
            emitted_any = False
            while next_index < len(tasks):
                target_ts = start_wall + (next_index / args.rps if args.rps > 0 else 0.0)
                if now < target_ts:
                    break
                task = tasks[next_index]
                request_id = f"req_{next_index:08d}"
                arrival_time_s = now_s()
                entry = QueuedTask(
                    req_i=next_index,
                    task=task,
                    request_id=request_id,
                    arrival_time_s=arrival_time_s,
                    prefill_tokens_est=prefill_estimator.estimate(task),
                    seq=next_index,
                )
                events_writer.write(
                    {
                        "ts": arrival_time_s,
                        "event": "request_emit",
                        "request_id": request_id,
                        "dataset_request_id": task.get("request_id"),
                        "benchmark": task.get("benchmark"),
                        "task_id": task.get("task_id"),
                        "prefill_tokens_est": int(entry.prefill_tokens_est),
                    }
                )
                _safe_print(
                    f"[emit] request_id={request_id} dataset_request_id={task.get('request_id')} "
                    f"benchmark={task.get('benchmark')} task_id={task.get('task_id')}",
                    flush=True,
                )
                if use_mars:
                    backlog.append(entry)
                else:
                    _submit_admitted_request(entry)
                next_index += 1
                emitted_any = True
                last_emit_wall_s = now
                now = now_s()

            if use_mars:
                _process_mars_submissions()

            if next_index >= len(tasks) and not backlog and not futures:
                run_completed = True
                break

            if emitted_any:
                continue

            # Process future incoming requests
            if futures:
                done, not_done = concurrent.futures.wait(
                    futures,
                    timeout=0.25,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if done:
                    futures = _collect_completed_requests(
                        done,
                        traces=traces,
                        traces_writer=traces_writer,
                        events_writer=events_writer,
                        future_meta=future_meta,
                        live_tool_stats=live_tool_stats,
                        pending_kv_blocks_by_request_id=pending_kv_blocks_by_request_id,
                    ) | set(not_done)
                    continue

            # Sleep to avoid busy-waiting
            next_target_ts = None
            if next_index < len(tasks):
                next_target_ts = start_wall + (next_index / args.rps if args.rps > 0 else 0.0)
            sleep_s = 0.25
            if next_target_ts is not None:
                sleep_s = max(0.01, min(0.25, float(next_target_ts - now_s())))
            time.sleep(sleep_s)
    except BaseException as exc:
        # Record fatal failures from the main scheduling loop
        fatal_exc = exc
        termination_reason = f"exception:{type(exc).__name__}"
        try:
            events_writer.write(
                {
                    "ts": now_s(),
                    "event": "run_exception",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "emitted_requests": int(next_index),
                    "completed_requests": int(len(traces)),
                    "inflight_futures": int(len(futures)),
                    "backlog_requests": int(len(backlog)),
                }
            )
        except Exception:
            pass
    finally:
        # Finalize traces/events record
        if callable(_poll_runtime_state_fn):
            try:
                _poll_runtime_state_fn()
            except Exception as exc:
                if fatal_exc is None:
                    fatal_exc = exc
                    termination_reason = f"exception:{type(exc).__name__}"
                try:
                    events_writer.write(
                        {
                            "ts": now_s(),
                            "event": "run_poll_exception",
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                except Exception:
                    pass
        if executor is not None:
            if signal_state["requested"] or fatal_exc is not None:
                for future in list(futures):
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True, cancel_futures=False)
        if kv_collector is not None:
            kv_collector.stop()

        completed_all_requests = bool(
            run_completed
            and not signal_state["requested"]
            and fatal_exc is None
            and next_index >= len(tasks)
            and not backlog
            and not futures
        )
        if not completed_all_requests and termination_reason == "completed":
            termination_reason = "partial"

        end_wall = now_s()
        run = {
            "start_time_s": start_wall,
            "end_time_s": end_wall,
            "wall_time_s": float(end_wall - start_wall),
            "out_dir": str(out_dir),
            "requested_count": int(len(tasks)),
            "emitted_requests": int(next_index),
            "completed_requests": int(len(traces)),
            "pending_backlog": int(len(backlog)),
            "inflight_futures": int(len(futures)),
            "completed_all_requests": bool(completed_all_requests),
            "termination_reason": termination_reason,
            "signal": signal_state.get("signal"),
        }
        summary = _summary(traces, run=run)
        atomic_write_json(out_dir / "summary.json", summary)
        try:
            _periodic_log_progress(force=True)
        except Exception:
            pass
        try:
            events_writer.write(
                {
                    "ts": end_wall,
                    "event": "run_end",
                    "summary": summary,
                }
            )
        except Exception:
            pass
        events_writer.close()
        traces_writer.close()
        for sig, handler in prev_signal_handlers.items():
            signal.signal(sig, handler)
        if sigusr1_registered and hasattr(signal, "SIGUSR1"):
            try:
                faulthandler.unregister(signal.SIGUSR1)
            except Exception:
                pass

    _safe_print(f"[OK] logs -> {out_dir}")
    _safe_print(json.dumps(summary, ensure_ascii=False, indent=2))

    if signal_state["requested"]:
        sig_name = str(signal_state.get("signal") or "SIGTERM")
        sig_num = getattr(signal, sig_name, signal.SIGTERM)
        raise SystemExit(128 + int(sig_num))
    if fatal_exc is not None:
        if isinstance(fatal_exc, SystemExit):
            raise fatal_exc
        raise fatal_exc


if __name__ == "__main__":
    main()
