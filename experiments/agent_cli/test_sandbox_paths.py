#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from sandbox_paths import task_readonly_paths


def test_task_readonly_paths_selects_only_current_localized_inputs(tmp_path: Path) -> None:
    localized = tmp_path / "_localized_inputs"
    benchmark = tmp_path / "benchmark"
    current_repo = localized / "paths" / "current-repo"
    foreign_repo = localized / "paths" / "foreign-repo"
    task_dir = benchmark / "terminal-bench" / "tasks" / "current-task"
    foreign_task = benchmark / "terminal-bench" / "tasks" / "foreign-task"
    for path in (current_repo, foreign_repo, task_dir, foreign_task):
        path.mkdir(parents=True)

    task = {
        "payload": {"repo_cache_path": str(current_repo)},
        "workspace_init": [
            {
                "path": "tb_meta.json",
                "content": json.dumps({"task_dir": str(task_dir)}),
            }
        ],
    }

    assert task_readonly_paths(
        task,
        allowed_roots=(localized, benchmark),
    ) == (current_repo.resolve(), task_dir.resolve())


def test_task_readonly_paths_rejects_roots_outside_and_symlink_escape(
    tmp_path: Path,
) -> None:
    localized = tmp_path / "_localized_inputs"
    benchmark = tmp_path / "benchmark"
    outside = tmp_path / "outside"
    localized.mkdir()
    benchmark.mkdir()
    outside.mkdir()
    escape = localized / "escape"
    escape.symlink_to(outside, target_is_directory=True)

    task = {
        "repo_cache_path": str(localized),
        "src_path": str(outside),
        "groundtruth_path": str(escape),
        "task_dir": str(benchmark),
    }

    assert task_readonly_paths(task, allowed_roots=(localized, benchmark)) == ()
