#!/usr/bin/env python3
"""Resolve the minimal read-only input paths needed by one agent request."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


TASK_INPUT_PATH_KEYS = frozenset(
    {
        "groundtruth_path",
        "repo_cache_path",
        "src_path",
        "task_dir",
    }
)


def _is_strict_descendant(path: Path, root: Path) -> bool:
    return path != root and root in path.parents


def _json_container(value: str) -> dict[str, Any] | list[Any] | None:
    stripped = value.lstrip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, (dict, list)) else None


def task_readonly_paths(
    task: dict[str, Any],
    *,
    allowed_roots: Iterable[Path],
) -> tuple[Path, ...]:
    """Return existing per-task inputs below explicit roots.

    OpenHands localizes repository/ground-truth inputs into a shared artifact
    tree and also rewrites JSON stored in ``workspace_init[].content``.  This
    walker handles both representations without exposing the whole shared tree
    to a request sandbox.  Paths outside the approved roots, the roots
    themselves, missing paths, and symlink escapes are deliberately ignored.
    """

    roots = tuple(
        sorted(
            {Path(root).expanduser().resolve(strict=False) for root in allowed_roots},
            key=str,
        )
    )
    selected: set[Path] = set()

    def add_path(raw: str) -> None:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            return
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError):
            return
        if any(_is_strict_descendant(resolved, root) for root in roots):
            selected.add(resolved)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in TASK_INPUT_PATH_KEYS and isinstance(child, str):
                    add_path(child)
                walk(child)
            return
        if isinstance(value, list):
            for child in value:
                walk(child)
            return
        if isinstance(value, str):
            decoded = _json_container(value)
            if decoded is not None:
                walk(decoded)

    walk(task)
    return tuple(sorted(selected, key=str))
