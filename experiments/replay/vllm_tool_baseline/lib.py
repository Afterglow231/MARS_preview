#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional


def now_s() -> float:
    return time.time()


def safe_join(root: Path, rel: str) -> Path:
    # Resolve a user-provided relative path under root. Absolute paths and traversal outside root are rejected. 
    rel = str(rel).strip()
    if rel in ("", "."):
        return root
    p = Path(rel)
    if p.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {rel}")
    root_res = root.resolve(strict=False)
    cand = (root / p).resolve(strict=False)
    if cand == root_res or root_res in cand.parents:
        return cand
    raise ValueError(f"Path escapes root: root={root_res} rel={rel} -> {cand}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def parse_cpu_list(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty cpu list spec")
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a.strip())
            end = int(b.strip())
            if end < start:
                raise ValueError(f"Invalid cpu range: {part}")
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _sched_setaffinity(pid: int, cores: Iterable[int]) -> None:
    try:
        os.sched_setaffinity(pid, set(cores)) 
    except AttributeError:
        return


def set_self_affinity(cores: list[int]) -> None:
    if not cores:
        raise ValueError("Empty cores")
    _sched_setaffinity(0, cores)


def validate_cores_subset(cores: list[int], *, allowed_min: int, allowed_max: int, name: str) -> None:
    bad = [c for c in cores if c < allowed_min or c > allowed_max]
    if bad:
        raise ValueError(f"{name} contains cores out of [{allowed_min},{allowed_max}]: {bad[:10]}")


class JSONLWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, obj: dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False, sort_keys=False)
        with self._lock:
            self._fp.write(line + "\n")
            self._fp.flush()

    def close(self) -> None:
        with self._lock:
            self._fp.close()


def percentile(values: list[float], p: float) -> Optional[float]:
    # Experiment result analysis.
    if not values:
        return None
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)


def mean(values: list[float]) -> Optional[float]:
    # Experiment result analysis.
    if not values:
        return None
    return float(sum(values) / len(values))


def format_ts(ts_s: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_s))


def ensure_removed(path: Path) -> None:
    # Cleanup for temporary files or directories.
    try:
        if path.is_dir():
            import shutil
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)  # py>=3.8
    except Exception:
        pass


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
