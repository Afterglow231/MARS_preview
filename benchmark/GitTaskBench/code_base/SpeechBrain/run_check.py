#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Run GitTaskBench test script in the current workspace.
# This mirrors GitTaskBench evaluation behavior for output discovery.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _find_output(output_dir: Path, *, multi_output: bool) -> Path | None:
    if not output_dir.exists():
        return None
    # Look for output.* files
    output_files = [p for p in output_dir.glob("output.*") if p.is_file()]
    if not output_files:
        output_files = [p for p in output_dir.glob("output*") if p.is_file()]
    if not output_files:
        output_files = [p for p in output_dir.glob("output_*") if p.is_file()]
    if not output_files:
        for subdir in output_dir.glob("output*"):
            if subdir.is_dir():
                output_files.extend([p for p in subdir.glob("*") if p.is_file()])
    if not output_files:
        return None
    if multi_output:
        return output_dir
    # Choose a stable file ordering
    output_files = sorted(output_files, key=lambda p: p.name)
    return output_files[0]


def main() -> int:
    root = Path(__file__).resolve().parent
    meta_path = root / "meta.json"
    if not meta_path.exists():
        print("meta.json missing")
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    test_script = root / str(meta.get("test_script") or "test_script.py")
    if not test_script.exists():
        print(f"test_script missing: {test_script}")
        return 2

    output_dir = root / str(meta.get("output_dir") or "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = root / str(meta.get("result_path") or "results/results.jsonl")
    result_path.parent.mkdir(parents=True, exist_ok=True)

    groundtruth_path = None
    gt = meta.get("groundtruth")
    if isinstance(gt, dict):
        gt_path = root / str(gt.get("dst_path") or "")
        if gt_path.exists():
            groundtruth_path = gt_path

    multi_output = bool(meta.get("multi_output"))
    output_path = _find_output(output_dir, multi_output=multi_output)
    if output_path is None:
        print(f"output not found under: {output_dir}")
        return 2

    # Detect which CLI args the test script supports.
    script_text = test_script.read_text(encoding="utf-8", errors="ignore")
    has_output = "--output" in script_text
    has_groundtruth = "--groundtruth" in script_text
    has_result = "--result" in script_text

    cmd = [sys.executable, str(test_script)]
    if has_output:
        cmd += ["--output", str(output_path)]
    if has_groundtruth and groundtruth_path is not None:
        cmd += ["--groundtruth", str(groundtruth_path)]
    if has_result:
        cmd += ["--result", str(result_path)]

    p = subprocess.run(cmd)
    rc = int(p.returncode)
    if rc != 0:
        return rc

    # If a result file exists, use it to decide pass/fail.
    if result_path.exists():
        try:
            lines = result_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            last = ""
            for line in reversed(lines):
                if line.strip():
                    last = line
                    break
            if last:
                obj = json.loads(last)
                if isinstance(obj, dict) and "Result" in obj:
                    ok = bool(obj.get("Result"))
                    if "Process" in obj:
                        ok = ok and bool(obj.get("Process"))
                    return 0 if ok else 2
        except Exception:
            # If parsing fails, fall back to rc==0.
            return rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
