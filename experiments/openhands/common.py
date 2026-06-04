#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

MARS_REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = MARS_REPO_ROOT / "benchmark"
_MARS_BENCHMARK_PREFIX_RE = re.compile(r"/[^\"'\s]*/MARS/benchmark/")
_GIT_TASK_BENCH_ALIAS_RE = re.compile(r"(?<!benchmark)/GitTaskBench(?=/|$)")
_BENCHMARK_ROOT_PLACEHOLDERS = (
    "${MARS_BENCHMARK_ROOT}/",
    "$MARS_BENCHMARK_ROOT/",
)
_REPO_ROOT_PLACEHOLDERS = (
    "${MARS_REPO_ROOT}/",
    "$MARS_REPO_ROOT/",
)
_HOME_PLACEHOLDERS = (
    "${HOME}/",
    "$HOME/",
)


def now_s() -> float:
    return time.time()


def safe_join(root: Path, rel: str) -> Path:
    rel = str(rel).strip()
    if rel in ("", "."):
        return root
    path = Path(rel)
    if path.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {rel}")
    root_resolved = root.resolve(strict=False)
    candidate = (root / path).resolve(strict=False)
    if candidate == root_resolved or root_resolved in candidate.parents:
        return candidate
    raise ValueError(f"Path escapes root: root={root_resolved} rel={rel} -> {candidate}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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


def set_self_affinity(cores: list[int]) -> None:
    if not cores:
        raise ValueError("Empty cores")
    try:
        os.sched_setaffinity(0, set(cores))  # type: ignore[attr-defined]
    except AttributeError:
        return


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


def load_tasks_jsonl(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"Invalid JSONL line (not object): {path}:{lineno}")
            schema = obj.get("schema")
            if schema != "mas_task_v1":
                raise ValueError(
                    f"Unexpected schema: {path}:{lineno}: {schema!r} "
                    "(expected 'mas_task_v1')"
                )
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid task line (missing payload dict): {path}:{lineno}")
            tasks.append(_relocate_embedded_paths(obj))
    if not tasks:
        raise ValueError(f"Empty tasks file: {path}")
    return tasks


def _relocate_string_paths(text: str) -> str:
    rewritten = text
    for placeholder in _BENCHMARK_ROOT_PLACEHOLDERS:
        rewritten = rewritten.replace(placeholder, f"{BENCHMARK_ROOT}/")
    for placeholder in _REPO_ROOT_PLACEHOLDERS:
        rewritten = rewritten.replace(placeholder, f"{MARS_REPO_ROOT}/")
    home_root = Path.home()
    for placeholder in _HOME_PLACEHOLDERS:
        rewritten = rewritten.replace(placeholder, f"{home_root}/")
    rewritten = _MARS_BENCHMARK_PREFIX_RE.sub(f"{BENCHMARK_ROOT}/", rewritten)
    git_task_bench_root = str(BENCHMARK_ROOT / "GitTaskBench")
    return _GIT_TASK_BENCH_ALIAS_RE.sub(git_task_bench_root, rewritten)


def _relocate_embedded_paths(obj: Any) -> Any:
    if isinstance(obj, str):
        return _relocate_string_paths(obj)
    if isinstance(obj, list):
        return [_relocate_embedded_paths(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _relocate_embedded_paths(value) for key, value in obj.items()}
    return obj


def percentile(values: list[float], p: float) -> float | None:
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


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return {"__bytes_b64__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    return obj


def _decode_kv_event(ev: Any) -> Any:
    if not isinstance(ev, list) or not ev or not isinstance(ev[0], str):
        return ev
    typ = ev[0]
    if typ == "BlockStored":
        return {
            "type": typ,
            "block_hashes": ev[1] if len(ev) > 1 else None,
            "parent_block_hash": ev[2] if len(ev) > 2 else None,
            "token_ids": ev[3] if len(ev) > 3 else None,
            "block_size": ev[4] if len(ev) > 4 else None,
            "lora_id": ev[5] if len(ev) > 5 else None,
            "medium": ev[6] if len(ev) > 6 else None,
        }
    if typ == "BlockRemoved":
        return {
            "type": typ,
            "block_hashes": ev[1] if len(ev) > 1 else None,
            "medium": ev[2] if len(ev) > 2 else None,
        }
    if typ == "AllBlocksCleared":
        return {"type": typ}
    return {"type": typ, "data": ev[1:]}


def _decode_kv_event_batch(decoded: Any) -> Any:
    if not isinstance(decoded, list) or len(decoded) < 2:
        return decoded
    ts = decoded[0]
    events_raw = decoded[1]
    dp_rank = decoded[2] if len(decoded) > 2 else None
    events: Any = events_raw
    if isinstance(events_raw, list):
        events = [_decode_kv_event(event) for event in events_raw]
    return {"ts": ts, "data_parallel_rank": dp_rank, "events": events}


class KVEventsCollector:
    def __init__(self, *, connect_endpoint: str, out_path: str | Path) -> None:
        self.connect_endpoint = connect_endpoint
        self.out_path = Path(out_path)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("KV events collector already started")

        try:
            import msgspec  # type: ignore[import-not-found]
            import zmq  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(
                f"KV events requires msgspec+pyzmq available in the runner environment: {exc}"
            ) from exc

        def _run() -> None:
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.SUBSCRIBE, b"")
            sock.connect(self.connect_endpoint)

            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            fp = self.out_path.open("a", encoding="utf-8")
            n = 0
            try:
                while not self._stop.is_set():
                    if not sock.poll(200):
                        continue
                    frames = sock.recv_multipart()
                    if len(frames) != 3:
                        continue
                    topic_b, seq_b, payload_b = frames
                    seq = int.from_bytes(seq_b, "big", signed=False)
                    try:
                        decoded = msgspec.msgpack.decode(payload_b)
                        batch = _decode_kv_event_batch(decoded)
                        payload_b64 = None
                    except Exception:
                        batch = None
                        payload_b64 = base64.b64encode(payload_b).decode("ascii")
                    rec = {
                        "recv_time_s": now_s(),
                        "seq": seq,
                        "topic": topic_b.decode("utf-8", errors="ignore"),
                        "batch": _jsonify(batch),
                        "payload_b64": payload_b64,
                    }
                    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    if n % 200 == 0:
                        fp.flush()
            finally:
                try:
                    fp.flush()
                    fp.close()
                except Exception:
                    pass
                try:
                    sock.close(linger=0)
                except Exception:
                    pass

        self._thread = threading.Thread(target=_run, name="kv-events", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        self._thread = None
