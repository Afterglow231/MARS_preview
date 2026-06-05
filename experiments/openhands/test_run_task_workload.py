from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

import run_task_workload as workload


def test_safe_print_tolerates_broken_pipe() -> None:
    repo_root = REPO_ROOT
    script = f"""
import os
import sys
from pathlib import Path

repo_root = Path({str(repo_root)!r})
sys.path.insert(0, str(repo_root / "experiments" / "openhands"))

import run_task_workload as workload

read_fd, write_fd = os.pipe()
os.close(read_fd)
broken_stdout = os.fdopen(write_fd, "w", buffering=1)
orig_stdout = sys.stdout
orig_stderr = sys.stderr
sys.stdout = broken_stdout
sys.stderr = open(os.devnull, "w")
try:
    workload._safe_print("first", flush=True)
    workload._safe_print("second", flush=True)
finally:
    try:
        broken_stdout.close()
    except Exception:
        pass
    try:
        sys.stderr.close()
    except Exception:
        pass
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr

print("ok")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_localize_task_inputs_rewrites_payload_and_workspace_meta(tmp_path: Path) -> None:
    source_repo = tmp_path / "host_cache" / "repo"
    source_repo.mkdir(parents=True)
    (source_repo / "README.md").write_text("cached repo\n", encoding="utf-8")
    task = {
        "schema": "mas_task_v1",
        "benchmark": "swebench",
        "task_id": "demo",
        "request_id": "demo",
        "workspace_init": [
            {
                "path": "meta.json",
                "content": json.dumps({"repo_cache_path": str(source_repo)}),
            }
        ],
        "payload": {"repo_cache_path": str(source_repo)},
    }

    localized, stats = workload._localize_task_inputs(
        [task],
        local_inputs_dir=tmp_path / "out" / "_localized_inputs",
    )

    payload_path = Path(localized[0]["payload"]["repo_cache_path"])
    meta = json.loads(localized[0]["workspace_init"][0]["content"])
    meta_path = Path(meta["repo_cache_path"])
    assert payload_path == meta_path
    assert payload_path.is_dir()
    assert payload_path.is_relative_to(tmp_path / "out" / "_localized_inputs")
    assert (payload_path / "README.md").read_text(encoding="utf-8") == "cached repo\n"
    assert stats["unique_copied_paths"] == 1
    assert stats["rewritten_paths"] >= 2


def test_require_path_under_rejects_workspace_outside_out_dir(tmp_path: Path) -> None:
    try:
        workload._require_path_under(
            tmp_path / "other" / "workspaces",
            tmp_path / "out",
            label="workspace_root",
        )
    except SystemExit:
        return
    raise AssertionError("expected SystemExit for workspace_root outside out_dir")
