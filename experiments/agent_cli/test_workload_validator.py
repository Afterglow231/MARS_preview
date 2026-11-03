from __future__ import annotations

import importlib.util
from pathlib import Path


def _validator_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_agent_cli_workload.py"
    spec = importlib.util.spec_from_file_location("validate_agent_cli_workload", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expand_path_relocates_foreign_mars_benchmark_root(tmp_path: Path) -> None:
    validator = _validator_module()
    benchmark_root = tmp_path / "benchmark"

    relocated = validator.expand_path(
        "/old/host/checkout/MARS/benchmark/terminal-bench/tasks/demo",
        benchmark_root,
    )

    assert relocated == (
        benchmark_root / "terminal-bench" / "tasks" / "demo"
    ).resolve(strict=False)


def test_expand_path_relocates_git_task_bench_alias(tmp_path: Path) -> None:
    validator = _validator_module()
    benchmark_root = tmp_path / "benchmark"

    relocated = validator.expand_path(
        "/old/host/GitTaskBench/dataset/repo",
        benchmark_root,
    )

    assert relocated == (
        benchmark_root / "GitTaskBench" / "dataset" / "repo"
    ).resolve(strict=False)


def test_expand_path_relocates_foreign_swebench_cache(tmp_path: Path) -> None:
    validator = _validator_module()
    benchmark_root = tmp_path / "benchmark"
    cache_root = tmp_path / "models" / "swebench_repo_cache"

    relocated = validator.expand_path(
        "/home/old-user/models/swebench_repo_cache/django__django",
        benchmark_root,
        cache_root,
    )

    assert relocated == (cache_root / "django__django").resolve(strict=False)
