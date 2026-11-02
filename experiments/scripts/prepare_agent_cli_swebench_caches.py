#!/usr/bin/env python3
"""Prepare exact SWE-Bench repository snapshots referenced by a workload."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any


REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command!r}\n"
            f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
        )
    return result


def _requirements(tasks: Path, max_requests: int) -> list[dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    count = 0
    with tasks.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if max_requests > 0 and count >= max_requests:
                break
            count += 1
            task = json.loads(line)
            if str(task.get("benchmark") or "").lower() != "swebench":
                continue
            payload = task.get("payload") or {}
            repo = str(payload.get("repo") or "").strip()
            commit = str(payload.get("base_commit") or "").strip().lower()
            path = str(payload.get("repo_cache_path") or "").strip()
            if not REPO_RE.fullmatch(repo) or not COMMIT_RE.fullmatch(commit) or not path:
                raise ValueError(f"invalid SWE-Bench cache metadata at {tasks}:{line_number}")
            item = {"repo": repo, "commit": commit, "path": path}
            prior = found.get(path)
            if prior is not None and prior != item:
                raise ValueError(f"conflicting cache metadata for {path}")
            found[path] = item
    return sorted(found.values(), key=lambda item: (item["repo"], item["commit"]))


def _prepare(item: dict[str, str], cache_root: Path) -> dict[str, str]:
    repo = item["repo"]
    commit = item["commit"]
    # Workloads store portable cache paths such as
    # ${HOME}/models/swebench_repo_cache/<name>; the runtime maps that cache
    # directory onto MARS_SWEBENCH_CACHE_ROOT, so place each checkout by name.
    name = Path(item["path"]).name
    if name in ("", ".", ".."):
        raise ValueError(f"invalid cache path: {item['path']}")
    path = (cache_root / name).resolve(strict=False)

    if path.exists() and not path.is_dir():
        raise RuntimeError(f"cache path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    git_dir = path / ".git"
    url = f"https://github.com/{repo}.git"
    if not git_dir.exists():
        existing = [entry for entry in path.iterdir()]
        if existing:
            raise RuntimeError(f"refusing non-empty non-git cache: {path}")
        _run(["git", "init", str(path)])
        _run(["git", "-C", str(path), "remote", "add", "origin", url])
    else:
        origin = _run(["git", "-C", str(path), "remote", "get-url", "origin"]).stdout.strip()
        if origin != url:
            raise RuntimeError(f"unexpected origin for {path}: {origin!r} != {url!r}")

    probe = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--verify", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    current = probe.stdout.strip() if probe.returncode == 0 else ""
    if current != commit:
        _run(["git", "-C", str(path), "fetch", "--depth=1", "origin", commit])
        _run(["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"])
    actual = _run(["git", "-C", str(path), "rev-parse", "HEAD"]).stdout.strip()
    if actual != commit:
        raise RuntimeError(f"cache HEAD mismatch for {path}: {actual} != {commit}")
    if not any(entry.name != ".git" for entry in path.iterdir()):
        raise RuntimeError(f"cache checkout is empty: {path}")
    return {**item, "path": str(path), "head": actual, "status": "ready"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            os.environ.get(
                "MARS_SWEBENCH_CACHE_ROOT",
                str(Path.home() / "models" / "swebench_repo_cache"),
            )
        ),
    )
    parser.add_argument("--max-requests", type=int, default=400)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    tasks = args.tasks.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    if args.manifest is None:
        args.manifest = cache_root / "manifest.json"
    cache_root.mkdir(parents=True, exist_ok=True)
    requirements = _requirements(tasks, args.max_requests)
    print(f"Preparing {len(requirements)} exact SWE-Bench caches with jobs={args.jobs}", flush=True)
    rows: list[dict[str, str]] = []
    print_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {executor.submit(_prepare, item, cache_root): item for item in requirements}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            with print_lock:
                print(f"[ready] {row['repo']} {row['commit']} -> {row['path']}", flush=True)
    rows.sort(key=lambda item: (item["repo"], item["commit"]))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "schema": "mars_swebench_cache_manifest_v1",
                "tasks": str(tasks),
                "count": len(rows),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(args.manifest)
    print(f"Prepared {len(rows)} SWE-Bench caches; manifest={args.manifest}", flush=True)


if __name__ == "__main__":
    main()
