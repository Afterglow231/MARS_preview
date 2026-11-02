#!/usr/bin/env python3
"""Preflight local assets needed by an agent-CLI workload."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


PATH_KEYS = {"groundtruth_path", "repo_cache_path", "src_path", "task_dir"}
_MARS_BENCHMARK_PREFIX_RE = re.compile(r"/[^\"'\s]*/MARS/benchmark/")
_SWEBENCH_CACHE_PREFIX_RE = re.compile(
    r"/[^\"'\s]*/models/swebench_repo_cache(?=/|$)"
)
_ABSOLUTE_GIT_TASK_BENCH_RE = re.compile(r"/[^\"'\s]*/GitTaskBench(?=/|$)")
_GIT_TASK_BENCH_ALIAS_RE = re.compile(r"(?<!benchmark)/GitTaskBench(?=/|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=0)
    return parser.parse_args()


def expand_path(
    value: str,
    benchmark_root: Path,
    swebench_cache_root: Path | None = None,
) -> Path | None:
    if swebench_cache_root is None:
        swebench_cache_root = Path(
            os.environ.get(
                "MARS_SWEBENCH_CACHE_ROOT",
                str(Path.home() / "models" / "swebench_repo_cache"),
            )
        ).expanduser().resolve(strict=False)
    value = value.replace("${MARS_BENCHMARK_ROOT}", str(benchmark_root))
    value = value.replace("$MARS_BENCHMARK_ROOT", str(benchmark_root))
    value = value.replace("${MARS_SWEBENCH_CACHE_ROOT}", str(swebench_cache_root))
    value = value.replace("$MARS_SWEBENCH_CACHE_ROOT", str(swebench_cache_root))
    value = value.replace("${HOME}", str(Path.home()))
    value = value.replace("$HOME", str(Path.home()))
    value = _MARS_BENCHMARK_PREFIX_RE.sub(f"{benchmark_root}/", value)
    value = _SWEBENCH_CACHE_PREFIX_RE.sub(str(swebench_cache_root), value)
    value = _ABSOLUTE_GIT_TASK_BENCH_RE.sub(
        str(benchmark_root / "GitTaskBench"), value
    )
    value = _GIT_TASK_BENCH_ALIAS_RE.sub(
        str(benchmark_root / "GitTaskBench"), value
    )
    expanded = Path(os.path.expandvars(value)).expanduser()
    if not expanded.is_absolute():
        return None
    return expanded.resolve(strict=False)


def collect_paths(obj: Any, benchmark_root: Path, out: list[tuple[str, Path]]) -> None:
    if isinstance(obj, list):
        for item in obj:
            collect_paths(item, benchmark_root, out)
        return
    if not isinstance(obj, dict):
        return
    for key, value in obj.items():
        if key in PATH_KEYS and isinstance(value, str):
            path = expand_path(value, benchmark_root)
            if path is not None:
                out.append((key, path))
        if key == "content" and isinstance(value, str):
            try:
                embedded = json.loads(value)
            except Exception:
                embedded = None
            if embedded is not None:
                collect_paths(embedded, benchmark_root, out)
        elif isinstance(value, (dict, list)):
            collect_paths(value, benchmark_root, out)


def main() -> None:
    args = parse_args()
    tasks_path = args.tasks.expanduser().resolve()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    if not tasks_path.is_file():
        raise SystemExit(f"workload not found: {tasks_path}")
    if not benchmark_root.is_dir():
        raise SystemExit(f"benchmark root not found: {benchmark_root}")

    counts: collections.Counter[str] = collections.Counter()
    missing: collections.Counter[tuple[str, str]] = collections.Counter()
    unique_paths: set[tuple[str, Path]] = set()
    checked_repo_paths: set[tuple[Path, str]] = set()
    total = 0
    with tasks_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if args.max_requests > 0 and total >= args.max_requests:
                break
            task = json.loads(line)
            if not isinstance(task, dict) or task.get("schema") != "mas_task_v1":
                raise SystemExit(f"invalid mas_task_v1 record: {tasks_path}:{line_number}")
            total += 1
            benchmark = str(task.get("benchmark") or "unknown").lower()
            counts[benchmark] += 1
            paths: list[tuple[str, Path]] = []
            collect_paths(task, benchmark_root, paths)
            payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
            expected_repo_commit = str(payload.get("base_commit") or "").strip()
            for key, path in paths:
                unique_paths.add((key, path))
                if not path.exists():
                    missing[(key, str(path))] += 1
                elif key == "repo_cache_path":
                    check_key = (path, expected_repo_commit)
                    if check_key in checked_repo_paths:
                        continue
                    checked_repo_paths.add(check_key)
                    if not path.is_dir() or not (path / ".git").exists():
                        missing[("repo-cache-git", str(path))] += 1
                        continue
                    probe = subprocess.run(
                        ["git", "-C", str(path), "rev-parse", "HEAD"],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    actual = probe.stdout.strip() if probe.returncode == 0 else ""
                    if not expected_repo_commit or actual != expected_repo_commit:
                        missing[
                            (
                                "repo-cache-head",
                                f"{path} (actual={actual or 'unreadable'}, expected={expected_repo_commit or 'missing'})",
                            )
                        ] += 1
                elif key == "task_dir" and not (
                    (path / "docker-compose.yaml").is_file()
                    or (path / "docker-compose.yml").is_file()
                ):
                    missing[("docker-compose", str(path))] += 1

    terminal_count = counts.get("terminalbench", 0)
    docker_error = ""
    if terminal_count:
        docker = shutil.which("docker")
        if not docker:
            docker_error = "docker CLI is missing"
        elif not (Path("/run/docker.sock").exists() or Path("/var/run/docker.sock").exists()):
            docker_error = "Docker socket is missing"
        else:
            probe = subprocess.run(
                [docker, "version", "--format", "{{.Server.Version}}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if probe.returncode != 0:
                docker_error = (probe.stderr or probe.stdout).strip()[-500:]

    print(f"tasks={total} benchmark_counts={dict(sorted(counts.items()))}")
    print(f"benchmark_root={benchmark_root} unique_local_inputs={len(unique_paths)}")
    if missing:
        print(f"missing_asset_references={sum(missing.values())} unique_missing={len(missing)}")
        for (key, path), references in list(sorted(missing.items()))[:20]:
            print(f"  {key}: {path} (referenced {references}x)")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more unique missing paths")
        raise SystemExit(2)
    if docker_error:
        raise SystemExit(f"TerminalBench Docker preflight failed: {docker_error}")
    if terminal_count:
        print(
            "terminalbench_docker=scoped_proxy "
            "(agent receives no Docker socket; compose/exec/cp are request-whitelisted)"
        )
    print("asset_preflight=ok")


if __name__ == "__main__":
    main()
