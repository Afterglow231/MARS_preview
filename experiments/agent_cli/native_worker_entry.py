#!/usr/bin/env python3
"""JSON boundary for one native Codex/OpenCode request worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from native_request_worker import run_request


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--result-json", required=True)
    return parser.parse_args()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".entry.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = _parse_args()
    task = _load_object(Path(args.task_json), label="task")
    config = _load_object(Path(args.config_json), label="worker config")
    api_key = os.environ["MARS_NATIVE_AGENT_API_KEY"]

    benchmark_root_raw = config.get("benchmark_root")
    result = run_request(
        agent=str(config["agent"]),
        agent_bin=Path(config["agent_bin"]),
        task=task,
        task_sha256=str(config["task_sha256"]),
        request_id=str(config["request_id"]),
        mars_job_id=str(config["mars_job_id"]),
        arrival_time_s=float(config["arrival_time_s"]),
        workspace_root=Path(config["workspace_root"]),
        local_inputs_dir=Path(config["local_inputs_dir"]),
        request_run_dir=Path(config["request_run_dir"]),
        model=str(config["model"]),
        base_url=str(config["base_url"]),
        api_key=api_key,
        context_window=int(config["context_window"]),
        max_output_tokens=int(config["max_output_tokens"]),
        max_iterations=int(config["max_iterations"]),
        temperature=float(config["temperature"]),
        top_p=(float(config["top_p"]) if config.get("top_p") is not None else None),
        timeout_s=float(config["timeout_s"]),
        llm_timeout_s=float(config["llm_timeout_s"]),
        llm_num_retries=int(config["llm_num_retries"]),
        terminal_no_change_timeout_s=(
            int(config["terminal_no_change_timeout_s"])
            if config.get("terminal_no_change_timeout_s") is not None
            else None
        ),
        require_bwrap=bool(config["require_bwrap"]),
        cpu_affinity=(
            str(config["cpu_affinity"])
            if config.get("cpu_affinity") not in (None, "")
            else None
        ),
        benchmark_root=(Path(benchmark_root_raw) if benchmark_root_raw else None),
        admission_gate_path=(
            Path(str(config["admission_gate_path"]))
            if config.get("admission_gate_path")
            else None
        ),
        allow_blocked_security_tasks=bool(
            config.get("allow_blocked_security_tasks", False)
        ),
        codex_enable_subagents=bool(config.get("codex_enable_subagents", False)),
    )
    if not isinstance(result, dict):
        raise TypeError("native request worker returned a non-object result")
    _write_json_atomic(Path(args.result_json), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
