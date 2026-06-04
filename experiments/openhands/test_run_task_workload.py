from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_safe_print_tolerates_broken_pipe() -> None:
    repo_root = Path(__file__).resolve().parents[2]
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
