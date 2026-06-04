#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import atomic_write_text, safe_join


PRIMARY_TEXT_FIELDS = (
    "text",
    "instruction",
    "problem_statement",
    "task_description",
)
BENCHMARKS_REQUIRING_VERIFICATION = {
    "repobench",
    "infinitebench",
}


def requires_tool_verification(task: dict[str, Any]) -> bool:
    benchmark = str(task.get("benchmark") or "").strip().lower()
    return benchmark in BENCHMARKS_REQUIRING_VERIFICATION


def extract_request_text(task: dict[str, Any]) -> str:
    payload = task.get("payload")
    if isinstance(payload, dict):
        for key in PRIMARY_TEXT_FIELDS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return json.dumps(task, ensure_ascii=False, indent=2)


def build_user_request(task: dict[str, Any]) -> str:
    request_text = extract_request_text(task)
    text = (
        "You are working in a local workspace.\n"
        "Use your standard OpenHands tools to help with the user's request.\n\n"
        "User request:\n"
        f"{request_text}"
    )
    text += (
        "\n\nKeep all temporary, generated, cache, and helper files inside the current workspace only. "
        "Use the '.mars_runtime/artifacts' directory for scratch outputs when possible. "
        "Do not write files outside the workspace. "
        "Do not install packages, language toolchains, or mutate conda/venv/python/runtime environments. "
        "Do not use broad process-control commands such as pkill, killall, or kill against all processes or process groups."
    )
    if requires_tool_verification(task):
        text += (
            "\n\nBefore you finish, you must perform at least one concrete tool-based verification step. "
            "Use a real OpenHands tool such as terminal or file_editor to inspect or validate relevant "
            "evidence instead of answering only from the prompt context. Do not finish immediately without "
            "at least one non-think, non-finish tool action."
        )
    return text


def build_verification_nudge(task: dict[str, Any]) -> str:
    request_text = extract_request_text(task)
    return (
        "You have not yet performed the required tool-based verification step.\n"
        "Before finishing, use at least one concrete tool action such as terminal or file_editor to "
        "inspect or validate relevant evidence for this task.\n\n"
        "Original user request:\n"
        f"{request_text}"
    )


def materialize_workspace_init(task: dict[str, Any], workspace_dir: Path) -> int:
    count = 0
    workspace_init = task.get("workspace_init") or []
    if not isinstance(workspace_init, list):
        return count
    for item in workspace_init:
        if not isinstance(item, dict):
            continue
        rel = item.get("path")
        content = item.get("content")
        if not isinstance(rel, str) or not isinstance(content, str):
            continue
        out_path = safe_join(workspace_dir, rel)
        atomic_write_text(out_path, content)
        count += 1
    return count
