#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import JSONLWriter
from openhands_runner import OpenHandsConfig, run_task_request


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


def main() -> None:
    args = _parse_args()
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
        workspace_root=Path(args.workspace_root).expanduser().resolve(),
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
