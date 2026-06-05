#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from common import JSONLWriter


class BufferedEventsWriter:
    def __init__(self, *, live_path: Path | None = None) -> None:
        self.items: list[dict[str, Any]] = []
        self._live_writer = JSONLWriter(live_path) if live_path is not None else None

    def write(self, obj: dict[str, Any]) -> None:
        self.items.append(obj)
        if self._live_writer is not None:
            self._live_writer.write(obj)

    def close(self) -> None:
        if self._live_writer is not None:
            self._live_writer.close()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Single-request OpenHands worker.")
    ap.add_argument("--task-json", type=str, required=True)
    ap.add_argument("--result-json", type=str, required=True)
    ap.add_argument("--request-id", type=str, required=True)
    ap.add_argument("--arrival-time-s", type=float, required=True)
    ap.add_argument("--workspace-root", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--base-url", type=str, required=True)
    ap.add_argument("--api-key", type=str, default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-output-tokens", type=int, default=2048)
    ap.add_argument("--llm-timeout-s", type=int, default=3000)
    ap.add_argument("--llm-num-retries", type=int, default=3)
    ap.add_argument("--max-iterations", type=int, default=40)
    ap.add_argument("--terminal-no-change-timeout-s", type=int, default=30)
    ap.add_argument(
        "--terminal-type",
        type=str,
        choices=["tmux", "subprocess"],
        default=None,
    )
    ap.add_argument("--live-events-path", type=str, default=None)
    return ap.parse_args()


def _prepare_early_worker_env(workspace_root: Path, request_id: str) -> None:
    workspace_dir = (workspace_root / request_id).resolve()
    runtime_root = workspace_dir / ".mars_runtime" / "worker"
    dirs = {
        "home": runtime_root / "home",
        "tmp": runtime_root / "tmp",
        "cache": runtime_root / "cache",
        "config": runtime_root / "config",
        "data": runtime_root / "data",
        "state": runtime_root / "state",
        "pycache": runtime_root / "pycache",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    host_home = os.environ.get("MARS_HOST_HOME") or str(Path.home().resolve())
    os.environ["MARS_HOST_HOME"] = host_home
    os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
    os.environ.update(
        {
            "HOME": str(dirs["home"]),
            "TMPDIR": str(dirs["tmp"]),
            "TMP": str(dirs["tmp"]),
            "TEMP": str(dirs["tmp"]),
            "XDG_CACHE_HOME": str(dirs["cache"]),
            "XDG_CONFIG_HOME": str(dirs["config"]),
            "XDG_DATA_HOME": str(dirs["data"]),
            "XDG_STATE_HOME": str(dirs["state"]),
            "PYTHONPYCACHEPREFIX": str(dirs["pycache"]),
            "PYTHONNOUSERSITE": "1",
        }
    )


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    _prepare_early_worker_env(workspace_root, args.request_id)

    from openhands_runner import OpenHandsConfig, run_task_request

    task = json.loads(Path(args.task_json).read_text(encoding="utf-8"))
    live_events_path = (
        Path(args.live_events_path).expanduser().resolve()
        if args.live_events_path
        else None
    )
    writer = BufferedEventsWriter(live_path=live_events_path)
    config = OpenHandsConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        llm_timeout_s=args.llm_timeout_s,
        llm_num_retries=args.llm_num_retries,
        max_iterations=args.max_iterations,
        terminal_no_change_timeout_s=args.terminal_no_change_timeout_s,
        terminal_type=args.terminal_type,
    )
    trace = run_task_request(
        config=config,
        task=task,
        request_id=args.request_id,
        arrival_time_s=args.arrival_time_s,
        workspace_root=workspace_root,
        events=writer,
    )
    result_path = Path(args.result_json).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result_path.write_text(
            json.dumps({"trace": trace, "events": writer.items}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
