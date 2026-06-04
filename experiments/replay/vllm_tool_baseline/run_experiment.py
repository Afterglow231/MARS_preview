#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vLLM (in-process AsyncLLM) + serverless tool baseline (MARS tasks)

行为（按你的需求改造）：
- 直接读取 `mas_task_v1` task JSONL（例如 `MARS/experiments/replay/tasks_mixed_sampled.jsonl`）
- 进程内创建 `AsyncLLM` engine（不再启动 HTTP vLLM server / 不再使用 replay dataset）
- Request 以固定 RPS 发射；每个 request 在内部多轮串行：GPU(action JSON) -> Tool -> GPU -> ...
- GPU 侧通过 `RequestOutputCollector` 送/收（多轮 prompt append-only，保留 KV/prefix cache 复用机会）
- Tool 调用仍使用外置 `ToolPool`（绑定 CPU 核）

输出：
- events.jsonl: timeline 事件
- traces.jsonl: 每个 request 的完整 trace（含每轮 GPU/Tool 指标）
- summary.json: 汇总（mean/p50/p95/throughput）
- kv_events.jsonl: vLLM KV cache 事件（仅当 --kv-events，使用 ZMQ subscriber）
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
from dataclasses import dataclass
import json
import os
import math
import random
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

# Avoid noisy fork warnings (and potential deadlocks) from HuggingFace tokenizers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import AsyncEngineArgs, SamplingParams
from vllm.config.kv_events import KVEventsConfig
from vllm.sampling_params import GuidedDecodingParams
from vllm.v1.engine.async_llm import AsyncLLM

from lib import (
    JSONLWriter,
    atomic_write_json,
    atomic_write_text,
    mean,
    now_s,
    parse_cpu_list,
    percentile,
    safe_join,
    set_self_affinity,
    validate_cores_subset,
)
from tool_pool import ToolPool, ToolPoolConfig, ToolTaskSpec


def load_tasks_jsonl(path: Path) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError(f"Invalid JSONL line (not object): {path}:{ln}")
            schema = obj.get("schema")
            if schema != "mas_task_v1":
                raise ValueError(f"Unexpected schema: {path}:{ln}: {schema!r} (expected 'mas_task_v1')")
            if "payload" not in obj or not isinstance(obj.get("payload"), dict):
                raise ValueError(f"Invalid task line (missing payload dict): {path}:{ln}")
            data.append(obj)
    if not data:
        raise ValueError(f"Empty tasks file: {path}")
    return data


def _proc_cpus_allowed_list(pid: int) -> Optional[str]:
    try:
        txt = Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    for line in txt.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return None


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _default_cpu_gpu() -> str:
    """Pick a reasonable default CPU set for vLLM processes.

    decode 性能会显著依赖 CPU：vLLM 的 scheduler/worker + CUDA launch 都需要 CPU。
    默认只给极少 CPU（例如 1 个物理核）会导致 decode 极慢（tok/s 级别）。
    """
    cpu_count = os.cpu_count() or 0
    if cpu_count <= 0:
        return "0"

    # Prefer the last 16 logical CPUs, but avoid overlapping the default tool
    # core range (0-7) when possible.
    span = min(16, cpu_count)
    start = max(0, cpu_count - span)
    if start <= 7 and cpu_count > 8:
        start = 8
    end = cpu_count - 1
    if start >= end:
        return str(end)
    return f"{start}-{end}"


def _default_prefill_tokens_csv() -> Optional[str]:
    repo_root = Path(__file__).resolve().parents[3]
    cand = repo_root / "experiments" / "replay" / "prefill_token_report" / "qwen3_round1_full_5" / "prefill_tokens.csv"
    if cand.is_file():
        return str(cand)
    return None


def load_prefill_tokens_csv(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not isinstance(row, dict):
                continue
            req_id = str(row.get("request_id") or "").strip()
            if not req_id:
                continue
            val = row.get("prefill_tokens")
            if val is None:
                continue
            try:
                n = int(str(val).strip())
            except Exception:
                continue
            if n > 0:
                out[req_id] = n
    return out


def _parse_cuda_graph_sizes(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    values: list[int] = []
    if text.startswith("[") and text.endswith("]"):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("--cuda-graph-sizes JSON form must be a list of integers.")
        for item in parsed:
            n = int(item)
            if n <= 0:
                raise ValueError("--cuda-graph-sizes values must be positive.")
            values.append(n)
    else:
        for part in text.replace(",", " ").split():
            n = int(part)
            if n <= 0:
                raise ValueError("--cuda-graph-sizes values must be positive.")
            values.append(n)

    if not values:
        return None

    deduped: list[int] = []
    seen: set[int] = set()
    for n in values:
        if n in seen:
            continue
        seen.add(n)
        deduped.append(n)
    return deduped


def _is_gpt_oss_120b_model(*, model_path: Optional[str], served_model_name: Optional[str]) -> bool:
    parts = [str(model_path or ""), str(served_model_name or "")]
    joined = " ".join(parts).lower()
    compact = "".join(ch for ch in joined if ch.isalnum())
    if "gpt-oss" in joined and "120b" in joined:
        return True
    return "gptoss120b" in compact


def _is_gpt_oss_model(*, model_path: Optional[str], served_model_name: Optional[str]) -> bool:
    parts = [str(model_path or ""), str(served_model_name or "")]
    joined = " ".join(parts).lower()
    compact = "".join(ch for ch in joined if ch.isalnum())
    return ("gpt-oss" in joined) or ("gptoss" in compact)


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return {"__bytes_b64__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    return obj


def _decode_kv_event(ev: Any) -> Any:
    # ZMQ payload is msgpack-encoded msgspec.Struct; when decoded without type
    # info, it becomes nested lists. KV events are array-like tagged structs
    # where the first element is the type tag string.
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
    # EventBatch is array-like: [ts, events, data_parallel_rank?]
    if not isinstance(decoded, list) or len(decoded) < 2:
        return decoded
    ts = decoded[0]
    events_raw = decoded[1]
    dp_rank = decoded[2] if len(decoded) > 2 else None
    events: Any = events_raw
    if isinstance(events_raw, list):
        events = [_decode_kv_event(e) for e in events_raw]
    return {"ts": ts, "data_parallel_rank": dp_rank, "events": events}


class _KVEventsCollector:
    def __init__(self, *, connect_endpoint: str, out_path: Path) -> None:
        self.connect_endpoint = connect_endpoint
        self.out_path = out_path
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("KV events collector already started")

        try:
            import msgspec  # type: ignore[import-not-found]
            import zmq  # type: ignore[import-not-found]
        except Exception as e:
            raise RuntimeError(
                f"KV events requires msgspec+pyzmq available in the runner environment: {e}"
            ) from e

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
                    if sock.poll(200):  # ms
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

        t = threading.Thread(target=_run, name="kv-events", daemon=True)
        t.start()
        self._thread = t

    def stop(self, *, timeout_s: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        self._thread = None


def _build_system_prompt(
    *,
    run_max_commands: int,
    run_max_command_chars: int,
    write_file_max_chars: int,
    system_prompt_file: Optional[str],
) -> str:
    # Keep identical semantics with record_replay.py (system prompt must not change).
    system_prompt = (
        "You are an autonomous, rigorous Principal Software Engineer Agent running in a standard Linux environment.\n"
        "Your goal is to deliver a VERIFIED, executable, and robust solution (tests must pass).\n"
        "Style: respond in professional and concise language. Do NOT respond your analysis about the problem.\n"
        "\n"
        "CORE PHILOSOPHY:\n"
        "1) Distrust static analysis: never assume code works; always run checks/tests.\n"
        "2) Environment ownership: you must set up an isolated runtime environment.\n"
        "3) Test-driven: reproduce failures or write tests before/while implementing fixes.\n"
        "\n"
        "WORKFLOW LOOP:\n"
        "1) Environment: create .venv, then install deps in .venv as needed.\n"
        "2) Explore: inspect files and understand the existing code.\n"
        "3) Reproduce: ensure failing tests/checks exist.\n"
        "4) Implement: make minimal changes.\n"
        "5) Verify: run compile checks + tests; if failing, apply smallest fix and re-run.\n"
        "\n"
        "AVAILABLE TOOLS (OUTPUT EXACTLY ONE JSON OBJECT; no markdown; no extra text):\n"
        "- list_files: list directory entries. {\"action\":\"tool\",\"tool\":\"list_files\",\"path\":\".\",\"recursive\":false,\"max_entries\":200}\n"
        "- read_file: read a file slice. {\"action\":\"tool\",\"tool\":\"read_file\",\"path\":\"README.md\",\"offset\":0,\"max_chars\":1200}\n"
        "- write_file: create/overwrite a file. {\"action\":\"tool\",\"tool\":\"write_file\",\"path\":\"solution.py\",\"content\":\"...\"}\n"
        "- run_cmd: run shell commands (bash -lc). {\"action\":\"tool\",\"tool\":\"run_cmd\",\"commands\":[\"python -m compileall -q .\",\"python -m pytest -q --maxfail=1\"]}\n"
        "- Optional semantic aliases (execute commands): compile/test/install/lint/format/typecheck.\n"
        "\n"
        "ENVIRONMENT:\n"
        "- Workspace filesystem persists across tool invocations (including .venv).\n"
        "- Shell session state is non-persistent across tool calls. ALWAYS use full-path executables (./.venv/bin/python, ./.venv/bin/pip, etc.).\n"
        "- Create an isolated venv immediately at the start: run_cmd [\"python3 -m venv .venv\"].\n"
        "- Do NOT install packages globally. Use ./.venv/bin/pip only.\n"
        "\n"
        "DEPENDENCIES:\n"
        "- Prefer zero-dependency solutions.\n"
        "- If third-party packages are required, write pinned deps to requirements.txt or pyproject.toml, then install via ./.venv/bin/pip (use constraints when possible).\n"
        "\n"
        "COMMAND RULES:\n"
        f"- For run_cmd/compile/test/install/lint/format/typecheck: at most {int(run_max_commands)} commands; each <= {int(run_max_command_chars)} chars.\n"
        "- Commands must be idempotent (safe to rerun) and use relative paths only.\n"
        f"- Keep write_file.content short (recommended <= {int(write_file_max_chars)} chars).\n"
        "\n"
        "FINALIZE:\n"
        "- Only after tests pass, output: {\"action\":\"final\"}.\n"
    )
    if system_prompt_file:
        sp = Path(str(system_prompt_file)).expanduser().resolve()
        system_prompt = sp.read_text(encoding="utf-8")
        if not system_prompt.endswith("\n"):
            system_prompt += "\n"
    return system_prompt


def _default_test_command_for_benchmark(bench: str) -> str:
    bench = str(bench or "").lower()
    if bench == "swebench":
        return "python run_check.py"
    if bench == "gittaskbench":
        return "python prepare_repo.py && python prepare_data.py && python run_check.py"
    if bench == "terminalbench":
        return "python tb_task.py test"
    return "python -m pytest -q --maxfail=1"


def _default_compile_command_for_benchmark(bench: str) -> str:
    bench = str(bench or "").lower()
    if bench == "swebench":
        return "python run_check.py --mode compileall"
    if bench == "terminalbench":
        return "python tb_task.py exec \"python -m compileall -q .\""
    return "python -m compileall -q ."


def _build_install_commands(
    *,
    packages: list[str],
    requirements: Optional[str],
    constraints: Optional[str],
    editable: bool,
    pip_args: list[str],
) -> list[str]:
    import shlex

    cmd: list[str] = ["python", "-m", "pip", "install"]
    if constraints:
        cmd += ["-c", constraints]
    if editable:
        cmd += ["-e"]
    if requirements:
        cmd += ["-r", requirements]
    else:
        cmd += packages
    cmd += pip_args
    return [" ".join(shlex.quote(x) for x in cmd)]


def _extract_json_obj(text: str) -> dict[str, Any]:
    import ast

    s = (text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(s) if ch == "{"][:50]
    for start in starts:
        try:
            obj, _end = decoder.raw_decode(s[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    i = s.find("{")
    j = s.rfind("}")
    if i < 0 or j <= i:
        if i >= 0 and j < 0:
            raise ValueError(f"Likely truncated JSON (no closing brace). output_prefix={s[:200]}")
        raise ValueError(f"No JSON object found in output: {s[:200]}")
    cand = s[i : j + 1]
    try:
        obj = json.loads(cand)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        obj = ast.literal_eval(cand)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    raise ValueError(f"Failed to parse JSON object from output: {s[:200]}")


def _normalize_action(obj: dict[str, Any], *, bench: str, infer_tool_intent: bool = False) -> dict[str, Any]:
    import shlex

    out = dict(obj)
    if "action" not in out:
        if "tool" in out:
            out["action"] = "tool"
        elif infer_tool_intent and ("cmd" in out or "command" in out):
            cmd_obj = out.get("commands")
            if cmd_obj is None:
                cmd_obj = out.get("cmd", out.get("command"))
            commands: list[str] = []
            if isinstance(cmd_obj, str) and cmd_obj.strip():
                commands = [cmd_obj.strip()]
            elif isinstance(cmd_obj, list):
                vals = [str(x) for x in cmd_obj if str(x).strip()]
                if vals:
                    if len(vals) >= 3 and vals[0] in ("bash", "sh", "zsh") and vals[1] in ("-lc", "-c"):
                        commands = [vals[2]]
                    else:
                        commands = [" ".join(shlex.quote(x) for x in vals)]
            if commands:
                out["action"] = "tool"
                out["tool"] = "run_cmd"
                out["commands"] = commands
            else:
                out["action"] = "final"
        elif infer_tool_intent and isinstance(out.get("path"), str) and ("line_start" in out or "line_end" in out):
            path = str(out.get("path") or "").strip()
            if path:
                start = int(out.get("line_start") or 1)
                end = int(out.get("line_end") or start)
                if start < 1:
                    start = 1
                if end < start:
                    end = start
                out["action"] = "tool"
                out["tool"] = "run_cmd"
                out["commands"] = [f"sed -n '{start},{end}p' -- {shlex.quote(path)}"]
            else:
                out["action"] = "final"
        elif infer_tool_intent and isinstance(out.get("path"), str) and (
            "recursive" in out or "depth" in out or "max_entries" in out
        ):
            out["action"] = "tool"
            out["tool"] = "list_files"
            out["path"] = str(out.get("path") or ".") or "."
            if "recursive" not in out:
                out["recursive"] = bool(int(out.get("depth") or 0) > 1)
            if "max_entries" not in out:
                out["max_entries"] = 200
        elif infer_tool_intent and isinstance(out.get("path"), str) and "content" in out:
            out["action"] = "tool"
            out["tool"] = "write_file"
        else:
            out["action"] = "final"
    if infer_tool_intent:
        out["action"] = str(out["action"]).strip().lower()
        if out["action"] in (
            "run_cmd",
            "run",
            "write_file",
            "compile",
            "test",
            "install",
            "venv",
            "lint",
            "format",
            "typecheck",
            "read_file",
            "list_files",
            "list_dir",
            "search",
        ):
            out["tool"] = out["action"]
            out["action"] = "tool"
    else:
        out["action"] = str(out["action"])
    if out["action"] == "tool":
        out["tool"] = str(out.get("tool") or "")
        if out["tool"] not in (
            "run_cmd",
            "run",
            "write_file",
            "compile",
            "test",
            "install",
            "venv",
            "lint",
            "format",
            "typecheck",
            "read_file",
            "list_files",
            "list_dir",
            "search",
        ):
            raise ValueError(f"Unsupported tool: {out['tool']}")

        if out["tool"] in ("run_cmd", "run", "compile", "test", "install", "lint", "format", "typecheck"):
            cmds = out.get("commands")
            if isinstance(cmds, str):
                out["commands"] = [cmds]
            elif isinstance(cmds, list):
                out["commands"] = [str(x) for x in cmds]
            elif cmds is None:
                out["commands"] = []
            else:
                raise ValueError("tool requires commands (str or list[str])")

            if out["tool"] == "compile" and not out["commands"]:
                out["commands"] = [_default_compile_command_for_benchmark(bench)]
            if out["tool"] == "test" and not out["commands"]:
                out["commands"] = [_default_test_command_for_benchmark(bench)]
            if out["tool"] == "install" and not out["commands"]:
                packages: list[str] = []
                req = out.get("requirements")
                requirements = req.strip() if isinstance(req, str) and req.strip() else None
                pk = out.get("packages")
                if isinstance(pk, str) and pk.strip():
                    packages = [pk.strip()]
                elif isinstance(pk, list):
                    packages = [str(x).strip() for x in pk if str(x).strip()]
                constraints = out.get("constraints")
                constraints_s = str(constraints).strip() if isinstance(constraints, str) and constraints.strip() else None
                editable = bool(out.get("editable", False))
                pip_args = out.get("pip_args")
                if isinstance(pip_args, str) and pip_args.strip():
                    pip_args_list = [pip_args.strip()]
                elif isinstance(pip_args, list):
                    pip_args_list = [str(x).strip() for x in pip_args if str(x).strip()]
                else:
                    pip_args_list = []
                if not requirements and not packages:
                    raise ValueError("install tool requires packages/requirements or commands")
                out["commands"] = _build_install_commands(
                    packages=packages,
                    requirements=requirements,
                    constraints=constraints_s,
                    editable=editable,
                    pip_args=pip_args_list,
                )
            if out["tool"] in ("lint", "format", "typecheck") and not out["commands"]:
                raise ValueError(f"{out['tool']} tool requires commands")

        if out["tool"] == "read_file":
            if not isinstance(out.get("path"), str) or not str(out.get("path") or "").strip():
                raise ValueError("read_file requires path")
            offset = int(out.get("offset") or 0)
            max_chars = int(out.get("max_chars") or 1200)
            out["offset"] = max(0, offset)
            out["max_chars"] = max(1, max_chars)
        if out["tool"] == "write_file":
            if not isinstance(out.get("path"), str):
                raise ValueError("write_file requires path")
            if not isinstance(out.get("content"), str):
                raise ValueError("write_file requires content")
        if out["tool"] == "venv":
            if "check_pip" in out:
                out["check_pip"] = bool(out.get("check_pip"))
    else:
        out["action"] = "final"
    return out


def _build_action_schema(*, run_max_commands: int, run_max_command_chars: int, write_file_max_chars: int) -> dict[str, Any]:
    # NOTE: xgrammar prints a warning for JSONSchema `allOf` with multiple
    # clauses and may not fully enforce it. We avoid `allOf` here to keep
    # guided decoding stable; runtime validation is handled in _normalize_action.
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["tool", "final"]},
            "tool": {
                "type": "string",
                "enum": [
                    "run_cmd",
                    "run",
                    "write_file",
                    "compile",
                    "test",
                    "install",
                    "venv",
                    "lint",
                    "format",
                    "typecheck",
                    "read_file",
                    "list_files",
                    "list_dir",
                    "search",
                ],
            },
            "path": {"type": "string", "minLength": 1, "maxLength": 256},
            "content": {"type": "string", "maxLength": int(write_file_max_chars)},
            "commands": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": int(run_max_command_chars)},
                "minItems": 1,
                "maxItems": int(run_max_commands),
            },
            "packages": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 256},
                "minItems": 1,
                "maxItems": 20,
            },
            "requirements": {"type": "string", "minLength": 1, "maxLength": 256},
            "constraints": {"type": "string", "minLength": 1, "maxLength": 256},
            "editable": {"type": "boolean"},
            "pip_args": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 256},
                "minItems": 1,
                "maxItems": 10,
            },
            "check_pip": {"type": "boolean"},
            "cwd": {"type": "string", "maxLength": 256},
            "timeout_s": {"type": "number"},
            "pattern": {"type": "string", "minLength": 1, "maxLength": 256},
            "glob": {"type": "string", "minLength": 1, "maxLength": 256},
            "offset": {"type": "integer", "minimum": 0, "maximum": 10000000},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": 20000},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 1000},
            "depth": {"type": "integer", "minimum": 0, "maximum": 10},
            "max_entries": {"type": "integer", "minimum": 1, "maximum": 5000},
            "recursive": {"type": "boolean"},
            "include_hidden": {"type": "boolean"},
            "case_sensitive": {"type": "boolean"},
        },
        "required": ["action"],
    }


def _build_task_user_prompt(task: dict[str, Any]) -> str:
    bench = str(task.get("benchmark") or "unknown")
    task_id = str(task.get("task_id") or task.get("request_id") or "")
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}

    parts: list[str] = []
    parts.append(f"[TASK] benchmark={bench} task_id={task_id}")
    parts.append("")
    parts.append("[WORKSPACE]")
    parts.append("- You operate in a per-request working directory. All paths must be relative.")
    parts.append("- Tool invocations share the same workspace filesystem (including .venv).")
    parts.append("- IMPORTANT: shell state is non-persistent across tool calls.")
    parts.append("")

    if bench == "swebench":
        problem = str(payload.get("problem_statement") or "")
        parts.append("[SWEBENCH]")
        parts.append("- Files: problem_statement.txt, patch.diff, test_patch.diff, meta.json, run_check.py")
        parts.append("- run_check.py will copy the cached repo snapshot into ./repo and run checks.")
        parts.append("- Typical loop: run_check.py -> inspect ./repo -> minimal fix -> rerun run_check.py -> final.")
        parts.append("")
        parts.append("[PROBLEM_STATEMENT]")
        parts.append(problem)
    elif bench == "gittaskbench":
        desc = str(payload.get("task_description") or "")
        working_subdir = str(payload.get("working_subdir") or "")
        out_dir = str(payload.get("output_dir") or "")
        parts.append("[GITTASKBENCH]")
        parts.append("- Files: meta.json, problem_statement.txt, query.json, prepare_repo.py, prepare_data.py, run_check.py, test_script.py")
        if working_subdir:
            parts.append(f"- working_subdir: {working_subdir}")
        if out_dir:
            parts.append(f"- output_dir: {out_dir}")
        parts.append("- Suggested: run `python prepare_repo.py`, then `python prepare_data.py`, then work in repo/ or working_subdir, then `python run_check.py`.")
        parts.append("")
        parts.append("[PROBLEM_STATEMENT]")
        parts.append(desc)
    elif bench == "terminalbench":
        instr = str(payload.get("instruction") or "")
        hint = str(payload.get("hint") or "")
        parts.append("[TERMINAL-BENCH]")
        parts.append("- Files: tb_meta.json, tb_task.py, problem_statement.txt")
        parts.append("- Use `python tb_task.py exec \"<cmd>\"` to run commands inside the task container.")
        parts.append("- Use `python tb_task.py test` to run official tests.")
        if hint:
            parts.append(f"- Hint: {hint}")
        parts.append("")
        parts.append("[INSTRUCTION]")
        parts.append(instr)
    else:
        parts.append("[PROBLEM]")
        parts.append(str(payload.get("text") or ""))

    parts.append("")
    parts.append("[NOTE] Output exactly ONE JSON object for each step (no extra text).")
    return "\n".join(parts).strip() + "\n"


async def _generate_one_action(
    *,
    engine: AsyncLLM,
    engine_request_id: str,
    prompt: str,
    sampling_params: SamplingParams,
    timeout_s: float,
    stop_on_first_json_object: bool = False,
) -> dict[str, Any]:
    t_submit = time.time()
    collector = await engine.add_request(engine_request_id, prompt, sampling_params, arrival_time=t_submit)

    first_token_time: Optional[float] = None
    latest_text = ""
    last_out: Optional[Any] = None

    async def _drain() -> None:
        nonlocal first_token_time, latest_text, last_out
        # When guided decoding is disabled, models sometimes keep generating
        # after emitting a valid JSON object (until max_tokens). We can stop
        # early as soon as we observe a complete top-level JSON object.
        json_prefix: Optional[str] = None
        aborted = False
        scan_pos = 0
        in_string = False
        escape = False
        depth = 0
        obj_start: Optional[int] = None

        while True:
            out = await collector.get()
            last_out = out
            if out and out.outputs:
                txt = out.outputs[0].text
                if isinstance(txt, str) and txt:
                    if first_token_time is None:
                        first_token_time = time.time()
                    if not aborted:
                        latest_text = txt
                    # Stream-scan for the first complete JSON object.
                    if stop_on_first_json_object and not aborted:
                        n = len(latest_text)
                        i = scan_pos
                        while i < n:
                            ch = latest_text[i]
                            if in_string:
                                if escape:
                                    escape = False
                                elif ch == "\\":
                                    escape = True
                                elif ch == "\"":
                                    in_string = False
                            else:
                                if ch == "\"":
                                    in_string = True
                                elif ch == "{":
                                    if obj_start is None:
                                        obj_start = i
                                    depth += 1
                                elif ch == "}":
                                    if depth > 0:
                                        depth -= 1
                                        if depth == 0 and obj_start is not None:
                                            json_prefix = latest_text[obj_start : i + 1]
                                            latest_text = json_prefix
                                            try:
                                                await engine.abort(engine_request_id)
                                            except Exception:
                                                pass
                                            aborted = True
                                            break
                            i += 1
                        scan_pos = n
                    elif aborted and json_prefix is not None:
                        latest_text = json_prefix
            if getattr(out, "finished", False):
                break

    try:
        await asyncio.wait_for(_drain(), timeout=timeout_s)
    except asyncio.TimeoutError:
        try:
            await engine.abort(engine_request_id)
        except Exception:
            pass
        raise

    t_end = time.time()
    if first_token_time is None:
        first_token_time = t_end

    prompt_tokens = None
    completion_tokens = None
    num_cached_tokens = None
    finish_reason = None
    try:
        if last_out is not None:
            if getattr(last_out, "prompt_token_ids", None) is not None:
                prompt_tokens = len(last_out.prompt_token_ids or [])
            if getattr(last_out, "num_cached_tokens", None) is not None:
                num_cached_tokens = int(last_out.num_cached_tokens or 0)
            if getattr(last_out, "outputs", None):
                out0 = (last_out.outputs or [None])[0]
                if out0 is not None and getattr(out0, "token_ids", None) is not None:
                    completion_tokens = len(out0.token_ids or [])
                if out0 is not None and getattr(out0, "finish_reason", None) is not None:
                    finish_reason = str(out0.finish_reason)
    except Exception:
        pass

    return {
        "submit_time_s": t_submit,
        "first_token_time_s": first_token_time,
        "end_time_s": t_end,
        "ttft_s": float(first_token_time - t_submit),
        "decode_time_s": float(t_end - first_token_time),
        "latency_s": float(t_end - t_submit),
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "num_cached_tokens": num_cached_tokens,
        "text": latest_text,
    }


async def run_one_request_async(
    *,
    request_id: str,
    arrival_time_s: float,
    req_spec: dict[str, Any],
    workspace_root: Path,
    seed: int,
    engine: AsyncLLM,
    tool_pool: ToolPool,
    events: JSONLWriter,
    gpu_timeout_s: float,
    tool_keep_venv: bool,
    max_rounds: int,
    action_max_tokens: int,
    action_json_retries: int,
    infer_tool_intent: bool,
    parse_feedback_max_retries: int,
    use_guided_decoding: bool,
    guided_json_schema: dict[str, Any],
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    trace_rounds: list[dict[str, Any]] = []
    req_error: Optional[dict[str, Any]] = None
    real_tool_calls_seen = 0

    workspace_dir = (workspace_root / request_id).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    ws_init = req_spec.get("workspace_init") or []
    if isinstance(ws_init, list):
        for item in ws_init:
            if not isinstance(item, dict):
                continue
            rel = item.get("path")
            content = item.get("content")
            if isinstance(rel, str) and isinstance(content, str):
                out_path = safe_join(workspace_dir, rel)
                atomic_write_text(out_path, content)

    events.write({
        "ts": arrival_time_s,
        "event": "request_emit",
        "request_id": request_id,
        "dataset_request_id": req_spec.get("request_id"),
        "benchmark": req_spec.get("benchmark"),
    })

    gpu_total = 0.0
    tool_total = 0.0
    tool_wait_total = 0.0

    def _sanitize_action(act: Any) -> Optional[dict[str, Any]]:
        if not isinstance(act, dict):
            return None
        out = dict(act)
        if out.get("tool") == "write_file" and "content" in out:
            c = str(out.get("content") or "")
            out.pop("content", None)
            out["content_len"] = len(c)
        if out.get("tool") in ("run_cmd", "run", "compile", "test", "install", "lint", "format", "typecheck"):
            cmds = out.get("commands")
            if isinstance(cmds, list):
                out["commands"] = [str(x)[:500] for x in cmds]
            elif isinstance(cmds, str):
                out["commands"] = [cmds[:500]]
        return out

    bench = str(req_spec.get("benchmark") or "unknown")
    system_prompt = str(req_spec.get("__system_prompt") or "")
    user_prompt = str(req_spec.get("__user_prompt") or "")
    # Append-only prompt buffer (so next round can reuse KV prefix cache on prompt+previous outputs)
    prompt_buf = system_prompt + "\n\n" + user_prompt
    parse_feedback_used = 0

    for i in range(1, int(max_rounds) + 1):
        round_header = f"\n\n[ROUND {i}] Now output exactly ONE JSON object for the next action.\n"
        prompt_buf += round_header
        round_prefix = prompt_buf

        retries = max(1, int(action_json_retries))
        selected_gpu: Optional[dict[str, Any]] = None
        selected_act: Optional[dict[str, Any]] = None
        selected_attempt = 0
        selected_raw: Optional[str] = None
        gpu_attempts: list[dict[str, Any]] = []

        for attempt in range(1, retries + 1):
            engine_rid = f"{request_id}__r{i:03d}"
            if attempt > 1:
                engine_rid = f"{engine_rid}__a{attempt:02d}"

            prompt_for_attempt = round_prefix
            if attempt > 1:
                prompt_for_attempt += (
                    "\n\n[FORMAT_ERROR] The previous output was invalid or truncated. "
                    "Output again: EXACTLY ONE JSON object, no extra text. Keep it short.\n"
                )

            sp = SamplingParams(
                max_tokens=int(action_max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                seed=int(seed),
                guided_decoding=(
                    GuidedDecodingParams(json=guided_json_schema) if use_guided_decoding else None
                ),
                extra_args={
                    "job_id": str(request_id),
                    "round_id": int(i),
                    "attempt": int(attempt),
                    # Continuum scheduling metadata (best-effort): mark the
                    # final allowed round as the last step so the scheduler
                    # won't pin KV for a job we will not resume.
                    "is_last_step": bool(i >= int(max_rounds)),
                },
            )

            events.write({
                "ts": now_s(),
                "event": "gpu_submit",
                "request_id": request_id,
                "round_id": i,
                "attempt": attempt,
                "max_tokens": int(action_max_tokens),
                "guided_decoding": bool(use_guided_decoding),
            })

            try:
                    gpu = await _generate_one_action(
                        engine=engine,
                        engine_request_id=engine_rid,
                        prompt=prompt_for_attempt,
                        sampling_params=sp,
                        timeout_s=float(gpu_timeout_s),
                        stop_on_first_json_object=(not use_guided_decoding),
                    )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                gpu_attempts.append({"attempt": attempt, "error": err})
                events.write({
                    "ts": now_s(),
                    "event": "gpu_error",
                    "request_id": request_id,
                    "round_id": i,
                    "attempt": attempt,
                    "error": err,
                })
                continue

            events.write({
                "ts": gpu["first_token_time_s"],
                "event": "gpu_first_token",
                "request_id": request_id,
                "round_id": i,
                "attempt": attempt,
                "ttft_s": gpu["ttft_s"],
            })
            events.write({
                "ts": gpu["end_time_s"],
                "event": "gpu_end",
                "request_id": request_id,
                "round_id": i,
                "attempt": attempt,
                "latency_s": gpu["latency_s"],
                "prompt_tokens": gpu.get("prompt_tokens"),
                "completion_tokens": gpu.get("completion_tokens"),
                "requested_max_tokens": int(action_max_tokens),
                "finish_reason": gpu.get("finish_reason"),
                "num_cached_tokens": gpu.get("num_cached_tokens"),
            })
            gpu_total += float(gpu["latency_s"])

            raw = str(gpu.get("text") or "")
            try:
                act = _normalize_action(
                    _extract_json_obj(raw),
                    bench=bench,
                    infer_tool_intent=bool(infer_tool_intent),
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                g = dict(gpu)
                g["attempt"] = int(attempt)
                g["action_parse_error"] = err
                gpu_attempts.append(g)
                events.write({
                    "ts": now_s(),
                    "event": "action_parse_error",
                    "request_id": request_id,
                    "round_id": i,
                    "attempt": attempt,
                    "error": err,
                    "finish_reason": gpu.get("finish_reason"),
                    "completion_tokens": gpu.get("completion_tokens"),
                })
                continue

            selected_gpu = gpu
            selected_act = act
            selected_attempt = attempt
            selected_raw = raw
            if attempt > 1:
                g = dict(gpu)
                g["attempt"] = int(attempt)
                gpu_attempts.append(g)
            if attempt > 1:
                prompt_buf = round_prefix + "\n\n[FORMAT_ERROR] Previous output invalid; corrected output follows.\n" + raw
            else:
                prompt_buf = round_prefix + raw
            break

        if selected_gpu is None or selected_act is None:
            if bool(infer_tool_intent) and int(parse_feedback_used) < int(parse_feedback_max_retries):
                parse_feedback_used += 1
                fb_ts = now_s()
                parse_errors = [
                    str(x.get("action_parse_error") or x.get("error") or "")
                    for x in gpu_attempts
                    if isinstance(x, dict)
                ]
                parse_errors = [x for x in parse_errors if x]
                synthetic_tool_res = {
                    "task_id": None,
                    "request_id": request_id,
                    "round_id": i,
                    "tool": "parse_feedback",
                    "cpu_core": None,
                    "keep_venv": tool_keep_venv,
                    "venv_dir": None,
                    "t_start_s": fb_ts,
                    "t_end_s": fb_ts,
                    "venv_rc": 126,
                    "tool_rc": 126,
                    "tool_err_tail": (
                        "action_parse_failed_after_retries; "
                        "return exactly one JSON object and choose action=tool when external tools are needed"
                    ),
                    "parse_feedback_retry_index": int(parse_feedback_used),
                    "parse_feedback_retry_max": int(parse_feedback_max_retries),
                    "parse_errors": parse_errors[:5],
                    "synthetic_feedback": True,
                }
                events.write({
                    "ts": fb_ts,
                    "event": "tool_enqueued",
                    "request_id": request_id,
                    "round_id": i,
                    "tool_type": "parse_feedback",
                })
                events.write({
                    "ts": fb_ts,
                    "event": "tool_start",
                    "request_id": request_id,
                    "round_id": i,
                    "cpu_core": None,
                    "tool_type": "parse_feedback",
                })
                events.write({
                    "ts": fb_ts,
                    "event": "tool_end",
                    "request_id": request_id,
                    "round_id": i,
                    "cpu_core": None,
                    "tool_type": "parse_feedback",
                    "venv_rc": 126,
                    "tool_rc": 126,
                })
                events.write({
                    "ts": fb_ts,
                    "event": "parse_feedback_retry",
                    "request_id": request_id,
                    "round_id": i,
                    "retry_index": int(parse_feedback_used),
                    "max_retry": int(parse_feedback_max_retries),
                })
                trace_rounds.append({
                    "round_id": i,
                    "gpu_attempts": gpu_attempts,
                    "action": {"action": "tool", "tool": "parse_feedback"},
                    "tool": {
                        "task_id": None,
                        "enqueued_time_s": fb_ts,
                        "start_time_s": fb_ts,
                        "end_time_s": fb_ts,
                        "wait_s": 0.0,
                        "duration_s": 0.0,
                        **synthetic_tool_res,
                    },
                })
                try:
                    prompt_buf = (
                        round_prefix
                        + "\n\n[TOOL_RESULT]\n"
                        + json.dumps(_jsonify(synthetic_tool_res), ensure_ascii=False)
                        + "\n"
                        + "[SYSTEM_HINT] Your previous action output could not be parsed. "
                        + "Output exactly one valid JSON object. If work is unfinished, choose action=tool.\n"
                    )
                except Exception:
                    prompt_buf = (
                        round_prefix
                        + "\n\n[TOOL_RESULT]\n"
                        + str(synthetic_tool_res)
                        + "\n[SYSTEM_HINT] output exactly one valid JSON object.\n"
                    )
                continue

            if bool(infer_tool_intent) and int(parse_feedback_max_retries) > 0:
                events.write({
                    "ts": now_s(),
                    "event": "parse_feedback_exhausted",
                    "request_id": request_id,
                    "round_id": i,
                    "used_retry": int(parse_feedback_used),
                    "max_retry": int(parse_feedback_max_retries),
                })

            req_error = {
                "stage": "action_parse",
                "round_id": i,
                "attempts": gpu_attempts,
            }
            trace_rounds.append({
                "round_id": i,
                "gpu_attempts": gpu_attempts,
            })
            break

        round_trace: dict[str, Any] = {
            "round_id": i,
            "gpu": selected_gpu,
            "gpu_attempt": selected_attempt,
            "action": _sanitize_action(selected_act),
        }
        if gpu_attempts and selected_attempt > 1:
            round_trace["gpu_attempts"] = gpu_attempts

        if selected_act.get("action") != "tool":
            if (
                bool(infer_tool_intent)
                and int(real_tool_calls_seen) == 0
                and int(parse_feedback_used) < int(parse_feedback_max_retries)
            ):
                parse_feedback_used += 1
                fb_ts = now_s()
                synthetic_tool_res = {
                    "task_id": None,
                    "request_id": request_id,
                    "round_id": i,
                    "tool": "parse_feedback",
                    "cpu_core": None,
                    "keep_venv": tool_keep_venv,
                    "venv_dir": None,
                    "t_start_s": fb_ts,
                    "t_end_s": fb_ts,
                    "venv_rc": 126,
                    "tool_rc": 126,
                    "tool_err_tail": (
                        "action_final_without_tool_before_any_tool; "
                        "continue with action=tool to perform required steps before final answer"
                    ),
                    "parse_feedback_retry_index": int(parse_feedback_used),
                    "parse_feedback_retry_max": int(parse_feedback_max_retries),
                    "parse_feedback_reason": "final_without_tool",
                    "synthetic_feedback": True,
                }
                events.write({
                    "ts": fb_ts,
                    "event": "tool_enqueued",
                    "request_id": request_id,
                    "round_id": i,
                    "tool_type": "parse_feedback",
                })
                events.write({
                    "ts": fb_ts,
                    "event": "tool_start",
                    "request_id": request_id,
                    "round_id": i,
                    "cpu_core": None,
                    "tool_type": "parse_feedback",
                })
                events.write({
                    "ts": fb_ts,
                    "event": "tool_end",
                    "request_id": request_id,
                    "round_id": i,
                    "cpu_core": None,
                    "tool_type": "parse_feedback",
                    "venv_rc": 126,
                    "tool_rc": 126,
                })
                events.write({
                    "ts": fb_ts,
                    "event": "parse_feedback_retry",
                    "request_id": request_id,
                    "round_id": i,
                    "retry_index": int(parse_feedback_used),
                    "max_retry": int(parse_feedback_max_retries),
                    "reason": "final_without_tool",
                })
                round_trace["tool"] = {
                    "task_id": None,
                    "enqueued_time_s": fb_ts,
                    "start_time_s": fb_ts,
                    "end_time_s": fb_ts,
                    "wait_s": 0.0,
                    "duration_s": 0.0,
                    **synthetic_tool_res,
                }
                trace_rounds.append(round_trace)
                try:
                    prompt_buf = (
                        round_prefix
                        + (selected_raw or "")
                        + "\n\n[TOOL_RESULT]\n"
                        + json.dumps(_jsonify(synthetic_tool_res), ensure_ascii=False)
                        + "\n"
                        + "[SYSTEM_HINT] You finished too early without any tool use. "
                        + "Output exactly one valid JSON object and continue with action=tool.\n"
                    )
                except Exception:
                    prompt_buf = (
                        round_prefix
                        + (selected_raw or "")
                        + "\n\n[TOOL_RESULT]\n"
                        + str(synthetic_tool_res)
                        + "\n[SYSTEM_HINT] continue with action=tool.\n"
                    )
                continue

            trace_rounds.append(round_trace)
            break

        tool_name = str(selected_act.get("tool") or "")
        real_tool_calls_seen += 1
        enq_ts = now_s()
        enq_event: dict[str, Any] = {
            "ts": enq_ts,
            "event": "tool_enqueued",
            "request_id": request_id,
            "round_id": i,
            "tool_type": tool_name,
        }
        if tool_name in ("run_cmd", "run", "compile", "test", "install", "lint", "format", "typecheck"):
            cmds = selected_act.get("commands")
            if isinstance(cmds, list):
                enq_event["num_commands"] = len(cmds)
            elif isinstance(cmds, str):
                enq_event["num_commands"] = 1
        events.write(enq_event)

        task_id, tool_enqueued_time_s, res_q = tool_pool.submit(
            ToolTaskSpec(
                request_id=request_id,
                round_id=i,
                keep_venv=tool_keep_venv,
                action=selected_act,
                workspace_dir=str(workspace_dir),
            )
        )
        try:
            res = await asyncio.to_thread(res_q.get, timeout=3600.0)
        except Exception as e:
            res = {
                "task_id": task_id,
                "request_id": request_id,
                "round_id": i,
                "tool": tool_name,
                "cpu_core": None,
                "keep_venv": tool_keep_venv,
                "venv_dir": None,
                "t_start_s": None,
                "t_end_s": None,
                "t_venv_start_s": None,
                "t_venv_end_s": None,
                "t_sleep_start_s": None,
                "t_sleep_end_s": None,
                "venv_rc": 126,
                "venv_err_tail": f"tool_result_timeout: {type(e).__name__}: {e}",
                "tool_rc": 126,
                "tool_err_tail": f"tool_result_timeout: {type(e).__name__}: {e}",
            }

        tool_start = res.get("t_start_s")
        tool_end = res.get("t_end_s")
        if isinstance(tool_start, (int, float)):
            events.write({
                "ts": float(tool_start),
                "event": "tool_start",
                "request_id": request_id,
                "round_id": i,
                "cpu_core": res.get("cpu_core"),
                "tool_type": tool_name,
            })
        if isinstance(tool_end, (int, float)):
            events.write({
                "ts": float(tool_end),
                "event": "tool_end",
                "request_id": request_id,
                "round_id": i,
                "cpu_core": res.get("cpu_core"),
                "tool_type": tool_name,
                "venv_rc": res.get("venv_rc"),
                "tool_rc": res.get("tool_rc"),
            })

        # Queueing delay (waiting for a free CPU core)
        if isinstance(tool_start, (int, float)):
            tool_wait_total += float(tool_start - tool_enqueued_time_s)
        if isinstance(tool_start, (int, float)) and isinstance(tool_end, (int, float)):
            tool_total += float(tool_end - tool_start)

        round_trace["tool"] = {
            "task_id": task_id,
            "enqueued_time_s": tool_enqueued_time_s,
            "start_time_s": tool_start,
            "end_time_s": tool_end,
            "wait_s": (float(tool_start - tool_enqueued_time_s) if isinstance(tool_start, (int, float)) else None),
            "duration_s": (float(tool_end - tool_start) if isinstance(tool_start, (int, float)) and isinstance(tool_end, (int, float)) else None),
            **res,
        }

        trace_rounds.append(round_trace)

        # Append tool result to prompt buffer (append-only).
        try:
            prompt_buf += "\n\n[TOOL_RESULT]\n" + json.dumps(_jsonify(res), ensure_ascii=False) + "\n"
        except Exception:
            prompt_buf += "\n\n[TOOL_RESULT]\n" + str(res) + "\n"

    end_time_s = now_s()
    e2e_latency_s = float(end_time_s - arrival_time_s)
    trace_obj = {
        "request_id": request_id,
        "dataset_request_id": req_spec.get("request_id"),
        "benchmark": req_spec.get("benchmark"),
        "workspace_dir": str(workspace_dir),
        "arrival_time_s": arrival_time_s,
        "end_time_s": end_time_s,
        "e2e_latency_s": e2e_latency_s,
        "gpu_total_s": gpu_total,
        "tool_total_s": tool_total,
        "tool_wait_total_s": tool_wait_total,
        "rounds": trace_rounds,
        "error": req_error,
    }
    if req_error is None:
        events.write({
            "ts": end_time_s,
            "event": "request_done",
            "request_id": request_id,
            "e2e_latency_s": e2e_latency_s,
        })
    else:
        try:
            err_s = json.dumps(req_error, ensure_ascii=False)
        except Exception:
            err_s = str(req_error)
        events.write({
            "ts": end_time_s,
            "event": "request_error",
            "request_id": request_id,
            "error": err_s,
            "error_detail": req_error,
        })
    return trace_obj


async def main_async(args: argparse.Namespace) -> None:
    cpu_gpu = parse_cpu_list(args.cpu_gpu)
    cpu_tool = parse_cpu_list(args.cpu_tool)
    cpu_count = os.cpu_count() or 0
    cpu_max = max(0, cpu_count - 1) if cpu_count else 127
    validate_cores_subset(cpu_tool, allowed_min=0, allowed_max=cpu_max, name="--cpu-tool")
    validate_cores_subset(cpu_gpu, allowed_min=0, allowed_max=cpu_max, name="--cpu-gpu")
    if set(cpu_gpu) & set(cpu_tool):
        raise SystemExit("--cpu-gpu and --cpu-tool must not overlap")
    if len(cpu_gpu) < 8:
        print(
            f"[WARN] --cpu-gpu has only {len(cpu_gpu)} CPU(s); vLLM decode may be extremely slow. "
            "Recommend >=16 logical CPUs (or >=8 physical cores).",
            file=sys.stderr,
        )

    scheduler_pid = os.getpid()
    # IMPORTANT: child processes inherit the parent's CPU affinity.
    # We first allow both CPU sets so ToolPool workers can pin themselves onto
    # --cpu-tool cores, then pin this (scheduler) process + vLLM EngineCore onto
    # --cpu-gpu cores.
    parent_affinity = sorted(set(cpu_gpu) | set(cpu_tool))
    set_self_affinity(parent_affinity)
    scheduler_cpus_allowed_parent = _proc_cpus_allowed_list(scheduler_pid)

    tasks_path = Path(args.tasks).resolve()
    tasks = load_tasks_jsonl(tasks_path)

    run_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    outputs_root = (Path(__file__).parent / "outputs").resolve()
    if args.output_dir is None:
        group_dir = outputs_root
    else:
        p = Path(args.output_dir).expanduser()
        if not p.is_absolute() and p.parts and p.parts[0] == "outputs":
            group_dir = (Path(__file__).parent / p).resolve()
        else:
            group_dir = p.resolve()
    group_dir.mkdir(parents=True, exist_ok=True)

    out_dir = (group_dir / run_ts).resolve()
    if out_dir.exists():
        i = 1
        while True:
            cand = (group_dir / f"{run_ts}_{i:02d}").resolve()
            if not cand.exists():
                out_dir = cand
                break
            i += 1
    out_dir.mkdir(parents=True, exist_ok=False)

    os.environ.setdefault("RUN_OUTPUT_DIR", str(out_dir))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    workspace_root = out_dir / "workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)

    events = JSONLWriter(out_dir / "events.jsonl")
    traces = JSONLWriter(out_dir / "traces.jsonl")

    kv_collector: Optional[_KVEventsCollector] = None
    kv_events_endpoint: Optional[str] = None
    kv_events_cfg: Optional[KVEventsConfig] = None
    if args.kv_events:
        kv_port = _pick_free_port()
        kv_bind_endpoint = f"tcp://*:{kv_port}"
        kv_events_endpoint = f"tcp://127.0.0.1:{kv_port}"
        kv_collector = _KVEventsCollector(
            connect_endpoint=kv_events_endpoint,
            out_path=(out_dir / "kv_events.jsonl"),
        )
        kv_events_cfg = KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=kv_bind_endpoint,
        )

    system_prompt = _build_system_prompt(
        run_max_commands=int(args.run_max_commands),
        run_max_command_chars=int(args.run_max_command_chars),
        write_file_max_chars=int(args.write_file_max_chars),
        system_prompt_file=(str(args.system_prompt_file) if args.system_prompt_file else None),
    )
    action_schema = _build_action_schema(
        run_max_commands=int(args.run_max_commands),
        run_max_command_chars=int(args.run_max_command_chars),
        write_file_max_chars=int(args.write_file_max_chars),
    )

    tool_cfg = ToolPoolConfig(
        cpu_cores=cpu_tool,
        venv_root=(out_dir / "tool_venvs"),
        tool_env_mode=str(args.tool_env_mode),
        thread_pool=(str(getattr(args, "thread_pool", "off")).lower() == "on"),
        sleep_s=float(args.tool_sleep_s),
        venv_copies=True,
        with_pip=True,
        venv_timeout_s=(float(args.tool_venv_timeout_s) if args.tool_venv_timeout_s is not None else None),
        command_timeout_s=(None if float(args.tool_command_timeout_s) <= 0 else float(args.tool_command_timeout_s)),
    )
    tool_pool = ToolPool(tool_cfg)
    tool_workers = tool_pool.worker_info()
    for w in tool_workers:
        pid = w.get("pid")
        if isinstance(pid, int) and pid > 0:
            w["cpus_allowed_list"] = _proc_cpus_allowed_list(pid)

    # Now pin this (scheduler) process and vLLM EngineCore children to --cpu-gpu.
    set_self_affinity(cpu_gpu)
    scheduler_cpus_allowed = _proc_cpus_allowed_list(scheduler_pid)

    # Create in-process AsyncLLM engine.
    engine_args = AsyncEngineArgs(
        model=str(args.model_path),
        served_model_name=(str(args.served_model_name) if args.served_model_name else None),
        trust_remote_code=True,
        dtype=str(args.dtype),
        enforce_eager=bool(args.enforce_eager),
        max_seq_len_to_capture=int(args.max_seq_len_to_capture),
        cuda_graph_sizes=(list(args.cuda_graph_sizes) if args.cuda_graph_sizes else []),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=(int(args.max_model_len) if args.max_model_len is not None else None),
        scheduling_policy=str(args.scheduling_policy),
        autellix_tail_fcfs_after_finished=int(
            args.autellix_tail_fcfs_after_finished),
        disable_log_stats=(not bool(args.log_stats)),
        guided_decoding_backend=str(args.guided_decoding_backend),
        guided_decoding_disable_fallback=bool(args.guided_decoding_disable_fallback),
        guided_decoding_disable_any_whitespace=bool(args.guided_decoding_disable_any_whitespace),
        guided_decoding_disable_additional_properties=bool(args.guided_decoding_disable_additional_properties),
        kv_events_config=kv_events_cfg,
        enable_log_requests=bool(args.log_requests),
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    prefill_tokens_csv: Optional[Path] = None
    prefill_tokens_by_request_id: dict[str, int] = {}
    if args.prefill_tokens_csv:
        try:
            prefill_tokens_csv = Path(str(args.prefill_tokens_csv)).expanduser().resolve()
        except Exception:
            prefill_tokens_csv = None
    if prefill_tokens_csv is not None and prefill_tokens_csv.is_file():
        try:
            prefill_tokens_by_request_id = load_prefill_tokens_csv(prefill_tokens_csv)
        except Exception as e:
            print(f"[WARN] Failed to load --prefill-tokens-csv={prefill_tokens_csv}: {type(e).__name__}: {e}", file=sys.stderr)
            prefill_tokens_by_request_id = {}

    enable_tool_intent_inference = _is_gpt_oss_model(
        model_path=(str(args.model_path) if args.model_path else None),
        served_model_name=(str(args.served_model_name) if args.served_model_name else None),
    )
    request_retry_max = max(0, int(getattr(args, "request_retry_max", 3) or 0))
    enable_parse_feedback = bool(
        _is_gpt_oss_120b_model(
            model_path=(str(args.model_path) if args.model_path else None),
            served_model_name=(str(args.served_model_name) if args.served_model_name else None),
        ) and request_retry_max > 0
    )

    start_wall = now_s()
    events.write({
        "ts": start_wall,
        "event": "run_start",
        "out_dir": str(out_dir),
        "emit_mode": str(getattr(args, "emit_mode", "rps")),
        "rps": args.rps,
        "num_requests": args.num_requests,
        "duration_s": args.duration_s,
        "cpu_gpu": args.cpu_gpu,
        "cpu_tool": args.cpu_tool,
        "tasks": str(tasks_path),
        "model_path": args.model_path,
        "served_model_name": args.served_model_name,
        "tool_intent_inference_enabled": bool(enable_tool_intent_inference),
        "tool0_request_retry_enabled": False,
        "tool0_request_retry_max": 0,
        "parse_feedback_enabled": bool(enable_parse_feedback),
        "parse_feedback_max_retry": int(request_retry_max),
        "scheduling_policy": str(args.scheduling_policy),
        "mars_active_pool_size": int(getattr(args, "mars_active_pool_size", 0) or 0),
        "mars_long_prefill_tokens": int(getattr(args, "mars_long_prefill_tokens", 0) or 0),
        "mars_long_kv_ratio": float(getattr(args, "mars_long_kv_ratio", 0.0) or 0.0),
        "prefill_tokens_csv": (str(prefill_tokens_csv) if prefill_tokens_csv is not None else None),
        "prefill_tokens_csv_count": int(len(prefill_tokens_by_request_id)),
        "mars_kv_target_ratio": float(getattr(args, "mars_kv_target_ratio", 0.0) or 0.0),
        "mars_kv_stat_interval_s": float(getattr(args, "mars_kv_stat_interval_s", 0.0) or 0.0),
        "mars_window_min": int(getattr(args, "mars_window_min", 0) or 0),
        "mars_window_init": int(getattr(args, "mars_window_init", 0) or 0),
        "mars_window_inc": int(getattr(args, "mars_window_inc", 0) or 0),
        "mars_window_dec_factor": float(getattr(args, "mars_window_dec_factor", 0.0) or 0.0),
        "mars_control_interval_s": float(getattr(args, "mars_control_interval_s", 0.0) or 0.0),
        "mars_cpu_backlog_high": float(getattr(args, "mars_cpu_backlog_high", 0.0) or 0.0),
        "mars_cpu_backlog_low": float(getattr(args, "mars_cpu_backlog_low", 0.0) or 0.0),
        "mars_cpu_queue_wait_high_s": float(getattr(args, "mars_cpu_queue_wait_high_s", 0.0) or 0.0),
        "mars_cpu_queue_wait_low_s": float(getattr(args, "mars_cpu_queue_wait_low_s", 0.0) or 0.0),
        "gpu_timeout_s": float(args.gpu_timeout_s),
        "log_stats": bool(args.log_stats),
        "kv_events": bool(args.kv_events),
        "kv_events_endpoint": kv_events_endpoint,
        "guided_decoding_backend": str(args.guided_decoding_backend),
        "guided_decoding_disable_fallback": bool(args.guided_decoding_disable_fallback),
        "guided_decoding_disable_any_whitespace": bool(args.guided_decoding_disable_any_whitespace),
        "guided_decoding_disable_additional_properties": bool(args.guided_decoding_disable_additional_properties),
        "action_guided_decoding": bool(args.action_guided_decoding),
        "tool_keep_venv": bool(args.tool_keep_venv),
        "tool_env_mode": str(args.tool_env_mode),
        "thread_pool": str(getattr(args, "thread_pool", "off")),
        "tool_venv_timeout_s": args.tool_venv_timeout_s,
        "tool_command_timeout_s": args.tool_command_timeout_s,
        "scheduler_pid": scheduler_pid,
        "scheduler_cpus_allowed_list": scheduler_cpus_allowed,
        "scheduler_cpus_allowed_list_parent": scheduler_cpus_allowed_parent,
        "tool_workers": tool_workers,
    })

    traces_list: list[dict[str, Any]] = []
    traces_lock = asyncio.Lock()

    try:
        if kv_collector is not None:
            kv_collector.start()
            events.write({
                "ts": now_s(),
                "event": "kv_events_started",
                "endpoint": kv_events_endpoint,
                "path": str(out_dir / "kv_events.jsonl"),
            })

        pending: set[asyncio.Task] = set()
        pending_long: set[asyncio.Task] = set()
        pending_kv_blocks_by_task: dict[asyncio.Task, int] = {}
        pending_kv_blocks_total = 0
        scheduling_policy = str(args.scheduling_policy)
        mars_active_pool_size = int(getattr(args, "mars_active_pool_size", 0) or 0)
        use_mars_admission = scheduling_policy == "mars" and mars_active_pool_size > 0
        use_mars = use_mars_admission
        mars_long_prefill_tokens = int(getattr(args, "mars_long_prefill_tokens", 0) or 0)
        mars_long_kv_ratio = float(getattr(args, "mars_long_kv_ratio", 0.0) or 0.0)
        mars_kv_target_ratio = float(getattr(args, "mars_kv_target_ratio", 0.0) or 0.0) or 0.9
        mars_kv_stat_interval_s = float(getattr(args, "mars_kv_stat_interval_s", 0.0) or 0.0) or 0.2
        mars_window_min = int(getattr(args, "mars_window_min", 0) or 0) or 4
        mars_window_init = int(getattr(args, "mars_window_init", 0) or 0) or 30
        mars_window_inc = int(getattr(args, "mars_window_inc", 0) or 0) or 1
        mars_window_dec_factor = float(getattr(args, "mars_window_dec_factor", 0.0) or 0.0) or 0.7
        mars_control_interval_s = float(getattr(args, "mars_control_interval_s", 0.0) or 0.0) or 0.2
        mars_cpu_backlog_high = float(getattr(args, "mars_cpu_backlog_high", 0.0) or 0.0) or 2.0
        mars_cpu_backlog_low = float(getattr(args, "mars_cpu_backlog_low", 0.0) or 0.0) or 1.0
        mars_cpu_queue_wait_high_s = float(getattr(args, "mars_cpu_queue_wait_high_s", 0.0) or 0.0) or 5.0
        mars_cpu_queue_wait_low_s = float(getattr(args, "mars_cpu_queue_wait_low_s", 0.0) or 0.0) or 2.0
        tokenizer = None
        need_tokenizer = bool(use_mars_admission and mars_long_prefill_tokens > 0 and not prefill_tokens_by_request_id)
        if use_mars and not prefill_tokens_by_request_id:
            need_tokenizer = True
        if need_tokenizer:
            try:
                tokenizer = await engine.get_tokenizer()
            except Exception:
                tokenizer = None
        start_mono = time.perf_counter()
        i = 0
        emit_done = False
        emit_mode = str(getattr(args, "emit_mode", "rps"))
        if emit_mode == "poison":
            emit_mode = "poisson"
        poisson_rng: Optional[random.Random] = None
        poisson_next_mono: Optional[float] = None
        if emit_mode == "poisson":
            poisson_rng = random.Random(int(args.seed) + 1)
            poisson_next_mono = start_mono

        def _prepare_spec(spec: dict[str, Any]) -> dict[str, Any]:
            spec2 = dict(spec)
            spec2["__system_prompt"] = system_prompt
            spec2["__user_prompt"] = _build_task_user_prompt(spec2)
            return spec2

        def _estimate_round1_prompt_tokens(spec2: dict[str, Any]) -> Optional[int]:
            rid = str(spec2.get("request_id") or "")
            if rid and rid in prefill_tokens_by_request_id:
                return int(prefill_tokens_by_request_id[rid])
            if tokenizer is None:
                return None
            prompt = (
                system_prompt
                + "\n\n"
                + str(spec2.get("__user_prompt") or "")
                + "\n\n[ROUND 1] Now output exactly ONE JSON object for the next action.\n"
            )
            try:
                return len(tokenizer.encode(prompt))
            except Exception:
                return None

        async def _get_kv_cache_stats() -> Optional[dict[str, Any]]:
            call = getattr(getattr(engine, "engine_core", None), "call_utility_async", None)
            if call is None:
                return None
            try:
                out = await call("get_kv_cache_stats")
            except Exception:
                return None
            return out if isinstance(out, dict) else None

        async def _drain_done() -> None:
            nonlocal pending_kv_blocks_total
            done = {t for t in pending if t.done()}
            for t in done:
                pending.discard(t)
                pending_long.discard(t)
                if t in pending_kv_blocks_by_task:
                    pending_kv_blocks_total -= pending_kv_blocks_by_task.pop(t, 0)
                try:
                    t.result()
                except Exception as e:
                    events.write({
                        "ts": now_s(),
                        "event": "request_task_error",
                        "task_name": str(getattr(t, "get_name", lambda: "")() or ""),
                        "error": f"{type(e).__name__}: {e}",
                    })

        async def _launch(
            req_i: int,
            spec2: dict[str, Any],
            arrival_ts: float,
            *,
            kv_blocks_cost: Optional[int] = None,
            is_long_task: bool = False,
        ) -> None:
            nonlocal pending_kv_blocks_total
            rid = f"req_{req_i:08d}"
            spec2 = dict(spec2)

            async def _run() -> None:
                tr: Optional[dict[str, Any]] = None
                try:
                    tr = await run_one_request_async(
                        request_id=rid,
                        arrival_time_s=arrival_ts,
                        req_spec=spec2,
                        workspace_root=workspace_root,
                        seed=int(args.seed),
                        engine=engine,
                        tool_pool=tool_pool,
                        events=events,
                        gpu_timeout_s=float(args.gpu_timeout_s),
                        tool_keep_venv=bool(args.tool_keep_venv),
                        max_rounds=int(args.max_rounds),
                        action_max_tokens=int(args.action_max_tokens),
                        action_json_retries=int(args.action_json_retries),
                        infer_tool_intent=bool(enable_tool_intent_inference),
                        parse_feedback_max_retries=(int(request_retry_max) if enable_parse_feedback else 0),
                        use_guided_decoding=bool(args.action_guided_decoding),
                        guided_json_schema=action_schema,
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                    )
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    events.write({
                        "ts": now_s(),
                        "event": "request_error",
                        "request_id": rid,
                        "error": err,
                    })
                    end_ts = now_s()
                    tr = {
                        "request_id": rid,
                        "dataset_request_id": spec2.get("request_id"),
                        "benchmark": spec2.get("benchmark"),
                        "workspace_dir": str((workspace_root / rid).resolve()),
                        "arrival_time_s": float(arrival_ts),
                        "end_time_s": float(end_ts),
                        "e2e_latency_s": float(end_ts - float(arrival_ts)),
                        "gpu_total_s": None,
                        "tool_total_s": None,
                        "tool_wait_total_s": None,
                        "rounds": [],
                        "error": {"stage": "exception", "error": err},
                    }
                if tr is not None:
                    async with traces_lock:
                        try:
                            traces.write(tr)
                        except Exception as e:
                            events.write({
                                "ts": now_s(),
                                "event": "trace_write_error",
                                "request_id": rid,
                                "error": f"{type(e).__name__}: {e}",
                            })
                        traces_list.append(tr)

            t = asyncio.create_task(_run(), name=f"req-{rid}")
            pending.add(t)
            if is_long_task:
                pending_long.add(t)
            if kv_blocks_cost is not None and kv_blocks_cost > 0:
                pending_kv_blocks_by_task[t] = int(kv_blocks_cost)
                pending_kv_blocks_total += int(kv_blocks_cost)

        @dataclass
        class _MarsQueuedReq:
            req_i: int
            spec: dict[str, Any]
            arrival_ts: float
            prefill_tokens_est: int
            seq: int

        mars_backlog: dict[int, _MarsQueuedReq] = {}
        mars_seq = 0
        mars_kv_cached: Optional[dict[str, Any]] = None
        mars_kv_cached_mono = 0.0

        mars_active_limit = int(mars_active_pool_size)
        mars_cpu_overloaded = False
        mars_kv_overloaded = False
        mars_last_control_mono = 0.0
        mars_short_empty_since_mono: Optional[float] = None
        mars_tail_phase = False
        mars_tail_kv_budget_blocks_cached: Optional[int] = None
        if use_mars:
            mars_window_min = max(1, min(int(mars_window_min), int(mars_active_pool_size)))
            mars_window_init = max(
                int(mars_window_min),
                min(int(mars_window_init), int(mars_active_pool_size)),
            )
            mars_active_limit = int(mars_window_init)
            mars_window_inc = max(0, int(mars_window_inc))
            mars_window_dec_factor = max(0.1, min(0.99, float(mars_window_dec_factor)))
            mars_control_interval_s = max(0.01, float(mars_control_interval_s))

        def _kv_blocks_for_tokens(tokens: int, block_size: int) -> int:
            if tokens <= 0 or block_size <= 0:
                return 0
            return int(math.ceil(int(tokens) / float(block_size)))

        def _mars_estimate_prefill_tokens(spec: dict[str, Any]) -> Optional[int]:
            pt = spec.get("prefill_tokens")
            if isinstance(pt, int) and pt > 0:
                return int(pt)
            rid0 = str(spec.get("request_id") or "").strip()
            if rid0 and rid0 in prefill_tokens_by_request_id:
                return int(prefill_tokens_by_request_id[rid0])
            if tokenizer is None:
                return None
            try:
                return _estimate_round1_prompt_tokens(_prepare_spec(spec))
            except Exception:
                return None

        def _mars_enqueue(req_i: int, spec: dict[str, Any], arrival_ts: float) -> None:
            nonlocal mars_seq
            mars_seq += 1
            est = _mars_estimate_prefill_tokens(spec)
            est_for_sort = int(est) if est is not None else 10**12
            # Large requests stay in the normal backlog; admission is handled
            # by MARS's CPU/KV-aware policy.
            ent = _MarsQueuedReq(
                req_i=int(req_i),
                spec=spec,
                arrival_ts=float(arrival_ts),
                prefill_tokens_est=int(est_for_sort),
                seq=int(mars_seq),
            )
            mars_backlog[int(req_i)] = ent

        async def _mars_get_kv_budget() -> tuple[Optional[int], Optional[int], Optional[dict[str, Any]]]:
            nonlocal mars_kv_cached, mars_kv_cached_mono
            if not use_mars:
                return None, None, None
            now = time.perf_counter()
            if mars_kv_cached is None or (now - mars_kv_cached_mono) >= float(mars_kv_stat_interval_s):
                mars_kv_cached = await _get_kv_cache_stats()
                mars_kv_cached_mono = now

            kv_stats = mars_kv_cached
            if kv_stats is None:
                return None, None, None
            try:
                block_size = int(kv_stats.get("block_size") or 0) or None
                num_gpu_blocks = int(kv_stats.get("num_gpu_blocks") or 0) or None
                num_free_blocks = int(kv_stats.get("num_free_blocks") or 0) or None
            except Exception:
                return None, None, kv_stats
            if block_size is None or num_gpu_blocks is None or num_free_blocks is None:
                return block_size, None, kv_stats
            kv_target = max(0.0, min(0.999, float(mars_kv_target_ratio)))
            reserve_blocks = int(math.ceil(float(num_gpu_blocks) * (1.0 - kv_target)))
            budget_blocks = max(0, int(num_free_blocks) - int(reserve_blocks))
            return block_size, int(budget_blocks), kv_stats

        async def _fill_active_mars() -> None:
            nonlocal mars_active_limit, mars_cpu_overloaded, mars_kv_overloaded
            nonlocal mars_last_control_mono
            nonlocal mars_short_empty_since_mono
            nonlocal mars_tail_phase, mars_tail_kv_budget_blocks_cached
            if not use_mars:
                return

            cpu_stats = tool_pool.stats()
            num_workers = int(cpu_stats.get("num_workers") or 0) or max(1, len(cpu_tool))
            inflight = int(cpu_stats.get("inflight") or 0)
            backlog_ratio = float(inflight) / float(max(1, num_workers))
            ema_wait = cpu_stats.get("ema_queue_wait_s")
            ema_wait_s = float(ema_wait) if isinstance(ema_wait, (int, float)) else None
            cpu_healthy_for_growth = backlog_ratio <= float(mars_cpu_backlog_low) and (
                ema_wait_s is None or ema_wait_s <= float(mars_cpu_queue_wait_low_s)
            )

            cpu_cap_ratio = max(
                0.1,
                (float(mars_cpu_backlog_low) + float(mars_cpu_backlog_high)) / 2.0,
            )
            cpu_window_cap = int(math.ceil(float(num_workers) * float(cpu_cap_ratio)))
            cpu_window_cap = max(
                int(mars_window_min),
                min(int(mars_active_pool_size), int(cpu_window_cap)),
            )

            prev_cpu_overloaded = bool(mars_cpu_overloaded)
            cpu_overloaded = prev_cpu_overloaded
            if prev_cpu_overloaded:
                if backlog_ratio <= float(mars_cpu_backlog_low) and (
                    ema_wait_s is None or ema_wait_s <= float(mars_cpu_queue_wait_low_s)
                ):
                    cpu_overloaded = False
            else:
                if backlog_ratio >= float(mars_cpu_backlog_high) or (
                    ema_wait_s is not None and ema_wait_s >= float(mars_cpu_queue_wait_high_s)
                ):
                    cpu_overloaded = True
            if cpu_overloaded != prev_cpu_overloaded:
                mars_cpu_overloaded = bool(cpu_overloaded)
                events.write({
                    "ts": now_s(),
                    "event": "mars_cpu_pressure",
                    "overloaded": bool(cpu_overloaded),
                    "backlog_ratio": float(backlog_ratio),
                    "inflight": int(inflight),
                    "num_workers": int(num_workers),
                    "ema_queue_wait_s": ema_wait_s,
                    "high_ratio": float(mars_cpu_backlog_high),
                    "low_ratio": float(mars_cpu_backlog_low),
                    "high_wait_s": float(mars_cpu_queue_wait_high_s),
                    "low_wait_s": float(mars_cpu_queue_wait_low_s),
                })

            block_size, budget_blocks, kv_stats = await _mars_get_kv_budget()
            kv_usage: Optional[float] = None
            kv_num_free_blocks: Optional[int] = None
            kv_num_gpu_blocks: Optional[int] = None
            if isinstance(kv_stats, dict):
                try:
                    if isinstance(kv_stats.get("usage"), (int, float)):
                        kv_usage = float(kv_stats.get("usage"))
                    if kv_usage is None:
                        kv_num_free_blocks = int(kv_stats.get("num_free_blocks") or 0)
                        kv_num_gpu_blocks = int(kv_stats.get("num_gpu_blocks") or 0)
                        if kv_num_gpu_blocks > 0:
                            kv_usage = max(
                                0.0,
                                min(
                                    1.0,
                                    1.0 - (float(kv_num_free_blocks) / float(kv_num_gpu_blocks)),
                                ),
                            )
                except Exception:
                    kv_usage = None

            # KV co-scheduling:
            # - track KV pressure with hysteresis
            # - derive a KV-based window cap that shrinks smoothly as usage increases
            kv_usage_high = max(0.1, min(0.999, float(mars_kv_target_ratio)))
            kv_usage_low = max(0.0, min(float(kv_usage_high) - 0.01, float(kv_usage_high) - 0.05))
            kv_window_cap = int(mars_active_pool_size)
            if kv_usage is not None:
                if kv_usage <= float(kv_usage_low):
                    kv_window_cap = int(mars_active_pool_size)
                elif kv_usage >= float(kv_usage_high):
                    kv_window_cap = int(mars_window_min)
                else:
                    span = max(1e-6, float(kv_usage_high) - float(kv_usage_low))
                    frac = max(0.0, min(1.0, (float(kv_usage) - float(kv_usage_low)) / span))
                    kv_window_cap_f = float(mars_active_pool_size) - frac * float(
                        int(mars_active_pool_size) - int(mars_window_min)
                    )
                    kv_window_cap = int(round(kv_window_cap_f))
                    kv_window_cap = max(
                        int(mars_window_min),
                        min(int(mars_active_pool_size), int(kv_window_cap)),
                    )

            prev_kv_overloaded = bool(mars_kv_overloaded)
            kv_overloaded = prev_kv_overloaded
            if kv_usage is not None:
                if prev_kv_overloaded:
                    if float(kv_usage) <= float(kv_usage_low):
                        kv_overloaded = False
                else:
                    if float(kv_usage) >= float(kv_usage_high):
                        kv_overloaded = True
            if kv_overloaded != prev_kv_overloaded:
                mars_kv_overloaded = bool(kv_overloaded)
                events.write({
                    "ts": now_s(),
                    "event": "mars_kv_pressure",
                    "overloaded": bool(kv_overloaded),
                    "kv_usage": float(kv_usage) if kv_usage is not None else None,
                    "kv_usage_high": float(kv_usage_high),
                    "kv_usage_low": float(kv_usage_low),
                    "kv_window_cap": int(kv_window_cap),
                    "kv_budget_blocks": (int(budget_blocks) if budget_blocks is not None else None),
                    "kv_num_free_blocks": (kv_stats.get("num_free_blocks") if isinstance(kv_stats, dict) else None),
                    "kv_num_gpu_blocks": (kv_stats.get("num_gpu_blocks") if isinstance(kv_stats, dict) else None),
                })

            now_mono = time.perf_counter()
            if mars_last_control_mono <= 0.0 or (
                (now_mono - mars_last_control_mono) >= float(mars_control_interval_s)
            ):
                prev_limit = int(mars_active_limit)
                new_limit = prev_limit
                admission_overloaded = bool(cpu_overloaded) or bool(kv_overloaded)
                if admission_overloaded:
                    new_limit = max(
                        int(mars_window_min),
                        int(math.floor(float(prev_limit) * float(mars_window_dec_factor))),
                    )
                else:
                    if (
                        cpu_healthy_for_growth
                        and not kv_overloaded
                        and (kv_usage is None or float(kv_usage) <= float(kv_usage_low))
                        and (budget_blocks is None or int(budget_blocks) > 0)
                    ):
                        new_limit = min(
                            int(mars_active_pool_size),
                            int(prev_limit + int(mars_window_inc)),
                        )
                new_limit = max(
                    int(mars_window_min),
                    min(int(mars_active_pool_size), int(new_limit)),
                )
                new_limit = max(int(mars_window_min), min(int(cpu_window_cap), int(new_limit)))
                new_limit = max(int(mars_window_min), min(int(kv_window_cap), int(new_limit)))
                if new_limit != prev_limit:
                    mars_active_limit = int(new_limit)
                    events.write({
                        "ts": now_s(),
                        "event": "mars_window_update",
                        "prev": int(prev_limit),
                        "new": int(new_limit),
                        "cpu_window_cap": int(cpu_window_cap),
                        "kv_window_cap": int(kv_window_cap),
                        "cpu_healthy_for_growth": bool(cpu_healthy_for_growth),
                        "overloaded": bool(admission_overloaded),
                        "cpu_overloaded": bool(cpu_overloaded),
                        "kv_overloaded": bool(kv_overloaded),
                        "backlog_ratio": float(backlog_ratio),
                        "inflight": int(inflight),
                        "num_workers": int(num_workers),
                        "ema_queue_wait_s": ema_wait_s,
                        "kv_usage": float(kv_usage) if kv_usage is not None else None,
                        "kv_usage_high": float(kv_usage_high),
                        "kv_usage_low": float(kv_usage_low),
                        "kv_budget_blocks": (int(budget_blocks) if budget_blocks is not None else None),
                    })
                mars_last_control_mono = float(now_mono)

            active_limit = min(int(mars_active_limit), int(cpu_window_cap), int(kv_window_cap))
            prefer_long = bool(mars_cpu_overloaded)

            if not mars_backlog:
                return

            # MARS uses this as a long-request threshold for admission
            # decisions; long requests still stay in the normal backlog.
            long_prefill_tokens = int(mars_long_prefill_tokens)
            long_prefill_tokens = int(long_prefill_tokens) if long_prefill_tokens > 0 else 0

            budget_remaining = int(budget_blocks) if budget_blocks is not None else None
            bs = int(block_size) if block_size is not None else None

            ents_all = list(mars_backlog.values())
            short_ents = ents_all
            long_ents: list[_MarsQueuedReq] = []
            if long_prefill_tokens > 0:
                short_ents = [e for e in ents_all if int(e.prefill_tokens_est) <= int(long_prefill_tokens)]
                long_ents = [e for e in ents_all if int(e.prefill_tokens_est) > int(long_prefill_tokens)]

            backlog_total = int(len(ents_all))
            backlog_long = int(len(long_ents)) if long_prefill_tokens > 0 else 0
            backlog_short = int(backlog_total - backlog_long) if long_prefill_tokens > 0 else int(backlog_total)

            tail_mode = bool(
                emit_done and long_prefill_tokens > 0 and backlog_long > 0 and backlog_short == 0
            )
            # Tail mode: only long requests remain. We enter tail
            # phase immediately, but we may "soft start" tail draining with a
            # small incremental concurrency when there are still untracked
            # in-flight requests (likely previously admitted short requests).
            if tail_mode and not mars_tail_phase:
                tail_bs = bs
                tail_num_free_blocks: Optional[int] = None
                if isinstance(kv_stats, dict):
                    try:
                        tail_num_free_blocks = int(kv_stats.get("num_free_blocks") or 0) or None
                    except Exception:
                        tail_num_free_blocks = None

                tail_kv_ratio: Optional[float] = None
                tail_kv_budget_blocks: Optional[int] = None
                if (
                    tail_bs is not None
                    and tail_num_free_blocks is not None
                    and tail_num_free_blocks > 0
                    and float(mars_long_kv_ratio) > 0.0
                ):
                    tail_kv_ratio = max(0.0, min(1.0, float(mars_long_kv_ratio)))
                    tail_kv_budget_blocks = max(
                        1,
                        int(float(tail_num_free_blocks) * float(tail_kv_ratio)),
                    )

                mars_tail_kv_budget_blocks_cached = (
                    int(tail_kv_budget_blocks) if tail_kv_budget_blocks is not None else None
                )
                mars_tail_phase = True
                events.write({
                    "ts": now_s(),
                    "event": "mars_tail_phase_start",
                    "policy": scheduling_policy,
                    "count": int(backlog_long),
                    "long_prefill_tokens": int(long_prefill_tokens),
                    "mars_long_kv_ratio": float(mars_long_kv_ratio),
                })
                if mars_tail_kv_budget_blocks_cached is not None:
                    events.write({
                        "ts": now_s(),
                        "event": "mars_tail_kv_budget",
                        "policy": scheduling_policy,
                        "ratio": float(tail_kv_ratio) if tail_kv_ratio is not None else None,
                        "block_size": int(tail_bs) if tail_bs is not None else None,
                        "num_free_blocks": int(tail_num_free_blocks)
                        if tail_num_free_blocks is not None
                        else None,
                        "budget_blocks": int(mars_tail_kv_budget_blocks_cached),
                        "budget_tokens": int(mars_tail_kv_budget_blocks_cached) * int(tail_bs)
                        if tail_bs is not None
                        else None,
                        "kv_stats": kv_stats,
                    })

            # Tail draining: ignore the adaptive window and use the fixed admission
            # cap (mars_active_pool_size), then let the KV budget determine how
            # many long requests can run concurrently.
            if tail_mode:
                active_limit = int(mars_active_pool_size)

            slots = max(0, int(active_limit) - int(len(pending)))
            if slots <= 0:
                return
            tail_soft_start = False
            if tail_mode:
                # Soft-start tail draining: if there are still untracked in-flight
                # requests (likely short), admit at most one tail request at a
                # time to avoid sudden KV interference.
                untracked_inflight = max(0, int(len(pending)) - int(len(pending_kv_blocks_by_task)))
                if untracked_inflight > 0:
                    slots = min(int(slots), 1)
                    tail_soft_start = True

            # Low-pressure fast path: when CPU/KV are both healthy and there is
            # enough slack to admit the entire backlog in this tick, degrade to
            # FCFS (arrival order). This avoids unnecessary short-first
            # reordering in the lightly-loaded regime (e.g., rps=0.05 CPU-heavy),
            # while keeping the high-pressure behavior unchanged.
            admission_overloaded = bool(cpu_overloaded) or bool(kv_overloaded)
            low_pressure_fcfs = bool(
                (not tail_mode)
                and (not admission_overloaded)
                and bool(cpu_healthy_for_growth)
                and (kv_usage is None or float(kv_usage) <= float(kv_usage_low))
                and (budget_blocks is None or int(budget_blocks) > 0)
                and (int(backlog_total) <= int(slots))
            )

            # Goal: minimize mean e2e latency (approx) by defaulting to short-first
            # admission; when CPU is overloaded, switch to long-first to reduce
            # tool-call pressure. When only long requests remain (tail), switch to
            # KV-budgeted tail draining.
            if tail_mode:
                selection_mode = "tail_kv_drain_fit_softstart" if tail_soft_start else "tail_kv_drain_fit"
            else:
                if low_pressure_fcfs:
                    selection_mode = "low_pressure_fcfs"
                else:
                    selection_mode = "cpu_overloaded_long_first" if prefer_long else "short_first"

            # "Needle insertion" (non-tail long admission): allow long requests
            # to start early only when there are no queued short requests for a
            # sustained period, KV is not overloaded, and long concurrency is
            # small. This protects p50 by preventing long requests from occupying
            # the window while short requests are still arriving.
            if backlog_short == 0:
                if mars_short_empty_since_mono is None:
                    mars_short_empty_since_mono = float(now_mono)
            else:
                mars_short_empty_since_mono = None

            short_empty_s = (
                float(now_mono - float(mars_short_empty_since_mono))
                if mars_short_empty_since_mono is not None
                else 0.0
            )
            short_empty_required_s = max(float(mars_control_interval_s), 1.0)
            if emit_mode == "rps":
                try:
                    rps_now = float(args.rps)
                    if rps_now > 0:
                        short_empty_required_s = max(
                            float(mars_control_interval_s),
                            min(5.0, 1.0 / float(rps_now)),
                        )
                except Exception:
                    pass

            non_tail_long_cap = 2
            non_tail_long_inflight = int(len(pending_long))
            non_tail_long_slots = max(0, int(non_tail_long_cap) - int(non_tail_long_inflight))
            allow_non_tail_long_insert = bool(
                (not tail_mode)
                and (long_prefill_tokens > 0)
                and (backlog_long > 0)
                and (backlog_short == 0)
                and (not kv_overloaded)
                and (non_tail_long_slots > 0)
                and ((len(pending) == 0) or (float(short_empty_s) >= float(short_empty_required_s)))
            )

            tail_kv_budget_blocks_used: Optional[int] = (
                int(mars_tail_kv_budget_blocks_cached)
                if (tail_mode and mars_tail_kv_budget_blocks_cached is not None)
                else None
            )
            if tail_mode and float(mars_long_kv_ratio) > 0.0 and isinstance(kv_stats, dict):
                # Recompute tail KV budget from *current* free blocks so tail
                # concurrency can ramp up as KV pressure drops.
                try:
                    tail_num_free_blocks_now = int(kv_stats.get("num_free_blocks") or 0)
                    if tail_num_free_blocks_now > 0:
                        tail_kv_ratio_now = max(0.0, min(1.0, float(mars_long_kv_ratio)))
                        tail_kv_budget_blocks_used = max(
                            1,
                            int(float(tail_num_free_blocks_now) * float(tail_kv_ratio_now)),
                        )
                except Exception:
                    pass
            budget_remaining_sel = int(budget_remaining) if budget_remaining is not None else None
            selected: list[tuple[_MarsQueuedReq, Optional[int], Optional[int]]] = []

            def _try_select(ent: _MarsQueuedReq) -> None:
                nonlocal budget_remaining_sel
                if len(selected) >= int(slots):
                    return
                kv_blocks: Optional[int] = None
                if bs is not None and budget_remaining_sel is not None:
                    kv_blocks = _kv_blocks_for_tokens(int(ent.prefill_tokens_est), int(bs))
                    if kv_blocks > int(budget_remaining_sel):
                        return
                    budget_remaining_sel = max(0, int(budget_remaining_sel) - int(kv_blocks))
                selected.append((ent, kv_blocks, int(budget_remaining_sel) if budget_remaining_sel is not None else None))

            if tail_mode:
                # Tail-only: drain remaining long requests using the KV budget.
                #
                # Unlike strict FCFS, we pack a *batch* of long requests that fit
                # within the remaining KV budget. This avoids head-of-line
                # blocking by an oversized request when other long requests could
                # still run safely without blowing GPU memory.
                candidates_long = sorted(
                    long_ents,
                    key=lambda e: (int(e.seq), int(e.req_i)),
                )

                tail_bs = bs
                tail_num_free_blocks: Optional[int] = None
                if isinstance(kv_stats, dict):
                    try:
                        tail_num_free_blocks = int(kv_stats.get("num_free_blocks") or 0)
                    except Exception:
                        tail_num_free_blocks = None

                budget_remaining_tail: Optional[int] = None
                if tail_bs is not None and tail_kv_budget_blocks_used is not None:
                    budget_remaining_tail = max(
                        0,
                        int(tail_kv_budget_blocks_used) - int(pending_kv_blocks_total),
                    )

                if tail_bs is None or budget_remaining_tail is None:
                    # No KV budget info: admit at most one long request at a
                    # time. Allow overlap with untracked in-flight requests
                    # (likely short), but avoid long-long interference.
                    if int(pending_kv_blocks_total) <= 0 and candidates_long:
                        # Pick the smallest-long as a safer default than strict
                        # FCFS when we cannot reason about KV fit.
                        ent0 = min(
                            candidates_long,
                            key=lambda e: (int(e.prefill_tokens_est), int(e.seq), int(e.req_i)),
                        )
                        kv_blocks_cost = (
                            _kv_blocks_for_tokens(int(ent0.prefill_tokens_est), int(tail_bs))
                            if tail_bs is not None
                            else None
                        )
                        selected.append(
                            (ent0, int(kv_blocks_cost) if kv_blocks_cost is not None else None, None)
                        )
                else:
                    budget_remaining_tail_sel = int(budget_remaining_tail)
                    # KV-fit scan (FCFS order with skipping): select a batch of
                    # requests that fit in the remaining KV budget.
                    for ent in candidates_long:
                        if len(selected) >= int(slots) or int(budget_remaining_tail_sel) <= 0:
                            break
                        kv_blocks_cost = _kv_blocks_for_tokens(int(ent.prefill_tokens_est), int(tail_bs))
                        if kv_blocks_cost > int(budget_remaining_tail_sel):
                            continue
                        budget_remaining_tail_sel = int(budget_remaining_tail_sel) - int(kv_blocks_cost)
                        selected.append((ent, int(kv_blocks_cost), int(budget_remaining_tail_sel)))

                    if not selected and int(pending_kv_blocks_total) <= 0 and candidates_long:
                        # Nothing fits but there is no inflight KV footprint:
                        # admit the oldest long request even if it exceeds the
                        # budget ratio to avoid deadlock.
                        ent0 = candidates_long[0]
                        kv_blocks_cost = _kv_blocks_for_tokens(int(ent0.prefill_tokens_est), int(tail_bs))
                        if kv_blocks_cost > int(budget_remaining_tail):
                            tail_kv_ratio = max(0.0, min(1.0, float(mars_long_kv_ratio)))
                            events.write({
                                "ts": now_s(),
                                "event": "mars_tail_budget_exceeded_single",
                                "policy": scheduling_policy,
                                "request_id": f"req_{ent0.req_i:08d}",
                                "prefill_tokens_est": int(ent0.prefill_tokens_est),
                                "block_size": int(tail_bs),
                                "req_blocks": int(kv_blocks_cost),
                                "budget_blocks": int(tail_kv_budget_blocks_used) if tail_kv_budget_blocks_used is not None else None,
                                "ratio": float(tail_kv_ratio),
                                "num_free_blocks": int(tail_num_free_blocks)
                                if tail_num_free_blocks is not None
                                else None,
                            })
                        selected.append((ent0, int(kv_blocks_cost), 0))
            elif low_pressure_fcfs:
                # FCFS admission in the low-pressure regime: preserve arrival
                # order across short/long requests.
                candidates = sorted(
                    ents_all,
                    key=lambda e: (int(e.seq), int(e.req_i)),
                )
                long_slots = int(non_tail_long_slots)
                for ent in candidates:
                    if len(selected) >= int(slots):
                        break
                    is_long_ent = bool(
                        long_prefill_tokens > 0 and int(ent.prefill_tokens_est) > int(long_prefill_tokens)
                    )
                    if is_long_ent:
                        if (not allow_non_tail_long_insert) or long_slots <= 0:
                            continue
                    before = len(selected)
                    _try_select(ent)
                    if is_long_ent and len(selected) > before:
                        long_slots = max(0, int(long_slots) - 1)
            elif prefer_long:
                # CPU overloaded: long-first (best-effort KV fit).
                candidates = sorted(
                    ents_all,
                    key=lambda e: (int(e.prefill_tokens_est), -int(e.seq)),
                    reverse=True,
                )
                for ent in candidates:
                    if len(selected) >= int(slots):
                        break
                    _try_select(ent)
            else:
                # CPU healthy: short-first; only admit long requests when the
                # window has spare capacity beyond available short requests.
                candidates_short = sorted(
                    short_ents,
                    key=lambda e: (int(e.prefill_tokens_est), int(e.seq)),
                )
                for ent in candidates_short:
                    if len(selected) >= int(slots):
                        break
                    _try_select(ent)

                if allow_non_tail_long_insert and len(selected) < int(slots) and long_ents:
                    candidates_long = sorted(
                        long_ents,
                        key=lambda e: (int(e.prefill_tokens_est), int(e.seq)),
                    )
                    long_slots = int(non_tail_long_slots)
                    for ent in candidates_long:
                        if len(selected) >= int(slots) or long_slots <= 0:
                            break
                        before = len(selected)
                        _try_select(ent)
                        if len(selected) > before:
                            long_slots = max(0, int(long_slots) - 1)

            if not selected:
                return

            for ent, kv_blocks, budget_remaining_after in selected:
                ent2 = mars_backlog.pop(int(ent.req_i), None)
                if ent2 is None:
                    continue

                is_long = bool(long_prefill_tokens > 0 and int(ent2.prefill_tokens_est) > int(long_prefill_tokens))
                kv_blocks_cost: Optional[int] = None
                if is_long and long_prefill_tokens > 0:
                    if kv_blocks is not None:
                        kv_blocks_cost = int(kv_blocks)
                    elif bs is not None:
                        kv_blocks_cost = _kv_blocks_for_tokens(int(ent2.prefill_tokens_est), int(bs))

                await _launch(
                    ent2.req_i,
                    _prepare_spec(ent2.spec),
                    ent2.arrival_ts,
                    kv_blocks_cost=kv_blocks_cost,
                    is_long_task=bool(is_long),
                )
                events.write({
                    "ts": now_s(),
                    "event": "mars_admit",
                    "policy": scheduling_policy,
                    "request_id": f"req_{ent2.req_i:08d}",
                    "dataset_request_id": str(ent2.spec.get("request_id") or ""),
                    "benchmark": str(ent2.spec.get("benchmark") or ""),
                    "prefill_tokens_est": int(ent2.prefill_tokens_est),
                    "is_long": bool(is_long) if long_prefill_tokens > 0 else None,
                    "long_prefill_tokens": int(long_prefill_tokens) if long_prefill_tokens > 0 else None,
                    "kv_blocks_est": (int(kv_blocks) if kv_blocks is not None else None),
                    "kv_budget_blocks": (
                        int(tail_kv_budget_blocks_used)
                        if (tail_mode and tail_kv_budget_blocks_used is not None)
                        else (int(budget_blocks) if budget_blocks is not None else None)
                    ),
                    "kv_budget_remaining": (int(budget_remaining_after) if budget_remaining_after is not None else None),
                    "kv_budget_mode": (
                        (
                            (
                                "tail_ratio"
                                if tail_kv_budget_blocks_used is not None
                                else "tail_singleton"
                            )
                            if tail_mode
                            else "target_ratio"
                        )
                        if long_prefill_tokens > 0
                        else None
                    ),
                    "tail_mode": bool(tail_mode) if long_prefill_tokens > 0 else None,
                    "tail_kv_budget_blocks": int(tail_kv_budget_blocks_used)
                    if (tail_mode and tail_kv_budget_blocks_used is not None)
                    else None,
                    "active_limit": int(active_limit),
                    "cpu_window_cap": int(cpu_window_cap),
                    "prefer_long": bool(prefer_long),
                    "selection_mode": str(selection_mode),
                    "backlog_total": int(backlog_total),
                    "backlog_short": int(backlog_short),
                    "backlog_long": int(backlog_long),
                    "short_empty_s": float(short_empty_s),
                    "short_empty_required_s": float(short_empty_required_s),
                    "non_tail_long_inflight": int(non_tail_long_inflight),
                    "non_tail_long_cap": int(non_tail_long_cap),
                    "allow_non_tail_long_insert": bool(allow_non_tail_long_insert),
                    "cpu_backlog_ratio": float(backlog_ratio),
                    "cpu_inflight": int(inflight),
                    "cpu_num_workers": int(num_workers),
                    "cpu_ema_queue_wait_s": ema_wait_s,
                    "kv_usage": (kv_stats.get("usage") if isinstance(kv_stats, dict) else None),
                    "kv_num_free_blocks": (kv_stats.get("num_free_blocks") if isinstance(kv_stats, dict) else None),
                    "kv_num_gpu_blocks": (kv_stats.get("num_gpu_blocks") if isinstance(kv_stats, dict) else None),
                })

        if (args.num_requests is None) == (args.duration_s is None):
            raise SystemExit("Must set exactly one of --num-requests or --duration-s")

        while True:
            elapsed = time.perf_counter() - start_mono
            if args.num_requests is not None:
                if i >= int(args.num_requests):
                    break
            else:
                if elapsed >= float(args.duration_s):
                    break

            if emit_mode == "rps":
                target = start_mono + (i / float(args.rps))
                sleep_s = target - time.perf_counter()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
            elif emit_mode == "poisson":
                if poisson_rng is None or poisson_next_mono is None:
                    raise RuntimeError("poisson emitter not initialized")
                if i == 0:
                    target = poisson_next_mono
                else:
                    poisson_next_mono += float(poisson_rng.expovariate(float(args.rps)))
                    target = poisson_next_mono
                sleep_s = target - time.perf_counter()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
            else:
                # Avoid starving the event loop in burst mode.
                if i % 50 == 0:
                    await asyncio.sleep(0)

            arrival_ts = now_s()
            spec = tasks[i % len(tasks)]
            if use_mars:
                _mars_enqueue(i, spec, arrival_ts)
                await _drain_done()
                await _fill_active_mars()
            else:
                await _launch(i, _prepare_spec(spec), arrival_ts)
            i += 1

        emit_done = True

        if not use_mars:
            if pending:
                await asyncio.gather(*pending)
        else:
            while mars_backlog or pending:
                await _drain_done()
                await _fill_active_mars()
                active_limit = int(mars_active_limit)
                if pending and (len(pending) >= active_limit or not mars_backlog):
                    await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                else:
                    await asyncio.sleep(0)

    finally:
        end_wall = now_s()
        try:
            tool_pool.shutdown()
        except Exception:
            pass
        try:
            engine.shutdown()
        except Exception:
            pass
        try:
            if kv_collector is not None:
                kv_collector.stop(timeout_s=5.0)
        except Exception:
            pass

        lat = [float(x["e2e_latency_s"]) for x in traces_list if "e2e_latency_s" in x]
        done_ts = [float(x["end_time_s"]) for x in traces_list if "end_time_s" in x]
        start_ts = [float(x["arrival_time_s"]) for x in traces_list if "arrival_time_s" in x]
        throughput = None
        if start_ts and done_ts and max(done_ts) > min(start_ts):
            throughput = float(len(done_ts) / (max(done_ts) - min(start_ts)))

        summary = {
            "run": {
                "start_time_s": start_wall,
                "end_time_s": end_wall,
                "wall_time_s": float(end_wall - start_wall),
                "out_dir": str(out_dir),
            },
            "count": len(traces_list),
            "latency_s": {"mean": mean(lat), "p50": percentile(lat, 50), "p95": percentile(lat, 95)},
            "throughput_rps": throughput,
        }
        atomic_write_json(out_dir / "summary.json", summary)
        events.write({"ts": end_wall, "event": "run_end", "summary": summary})
        events.close()
        traces.close()

        print(f"[OK] logs -> {out_dir}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="Optional JSON config file")
    pre_args, _ = pre.parse_known_args()

    cfg_defaults: dict[str, Any] = {}
    if pre_args.config:
        cfg_path = Path(pre_args.config).expanduser().resolve()
        cfg_defaults = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfg_defaults, dict):
            raise SystemExit(f"--config must be a JSON object, got: {type(cfg_defaults).__name__}")

    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[pre],
    )
    ap.add_argument("--tasks", type=str, default=None, help="mas_task_v1 task JSONL")
    ap.add_argument(
        "--model-path",
        type=str,
        default=str(Path.home() / "models" / "Qwen3-Coder-30B-A3B-Instruct"),
    )
    ap.add_argument("--served-model-name", type=str, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--max-model-len", type=int, default=260000)
    ap.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forwarded to vLLM engine (--enforce-eager / --no-enforce-eager).",
    )
    ap.add_argument(
        "--max-seq-len-to-capture",
        type=int,
        default=8192,
        help=(
            "Forwarded to vLLM engine. Sequences longer than this fall back "
            "to eager mode instead of using CUDA graph."
        ),
    )
    ap.add_argument(
        "--cuda-graph-sizes",
        type=str,
        default=None,
        help=(
            "Forwarded to vLLM engine. Capture batch-size list for CUDA graph. "
            "Accepts comma/space separated integers or JSON list, e.g. "
            "'1,2,4,8,16,24,32' or '[1, 2, 4, 8]'."
        ),
    )
    ap.add_argument(
        "--scheduling-policy",
        type=str,
        choices=[
            "fcfs",
            "priority",
            "continuum",
            "continuum_dy",
            "autellix",
            "infercept",
            "mars",
        ],
        default="fcfs",
        help="Forwarded to vLLM engine (--scheduling-policy).",
    )
    ap.add_argument(
        "--autellix-tail-fcfs-after-finished",
        "--autellix-fcfs-after-finished",
        dest="autellix_tail_fcfs_after_finished",
        type=int,
        default=0,
        help=(
            "When --scheduling-policy=autellix, switch the internal scheduler "
            "to FCFS after this many finished jobs (job_id). Set <=0 to "
            "disable."
        ),
    )
    ap.add_argument(
        "--mars-active-pool-size",
        type=int,
        default=30,
        help=(
            "When --scheduling-policy=mars, cap concurrent in-flight MAS requests (external admission control). "
            "Extra requests are kept outside vLLM and only submitted when there is an available slot. Set <=0 to disable."
        ),
    )
    ap.add_argument(
        "--mars-long-prefill-tokens",
        type=int,
        default=50000,
        help=(
            "For --scheduling-policy=mars: use this initial round-1 prefill-token threshold "
            "to identify long requests for admission heuristics. Set <=0 to disable."
        ),
    )
    ap.add_argument(
        "--mars-long-kv-ratio",
        type=float,
        default=0.9,
        help=(
            "In MARS tail mode, limit the sum KV footprint of concurrently running long requests. "
            "We approximate each request's KV footprint as ceil(prefill_tokens_est / block_size) blocks and ensure "
            "the total does not exceed (current free KV blocks * ratio). Set <=0 to disable."
        ),
    )
    ap.add_argument(
        "--prefill-tokens-csv",
        type=str,
        default=_default_prefill_tokens_csv(),
        help=(
            "Optional CSV mapping request_id -> prefill_tokens (round-1). "
            "Used by mars long-request admission to avoid expensive tokenization."
        ),
    )
    ap.add_argument(
        "--mars-kv-target-ratio",
        type=float,
        default=0.9,
        help="mars only: admit requests to target this KV cache usage ratio (approx).",
    )
    ap.add_argument(
        "--mars-kv-stat-interval-s",
        type=float,
        default=0.2,
        help="mars only: minimum interval between KV stat queries.",
    )
    ap.add_argument(
        "--mars-window-min",
        type=int,
        default=4,
        help="mars only: minimum active pool size (co-schedule admission).",
    )
    ap.add_argument(
        "--mars-window-init",
        type=int,
        default=30,
        help="mars only: initial active pool size (clamped into [window-min, --mars-active-pool-size]).",
    )
    ap.add_argument(
        "--mars-window-inc",
        type=int,
        default=1,
        help="mars only: additive increase step when CPU is healthy and KV has headroom.",
    )
    ap.add_argument(
        "--mars-window-dec-factor",
        type=float,
        default=0.7,
        help="mars only: multiplicative decrease factor when CPU or KV is overloaded.",
    )
    ap.add_argument(
        "--mars-control-interval-s",
        type=float,
        default=0.2,
        help="mars only: minimum interval between window updates (seconds).",
    )
    ap.add_argument(
        "--mars-cpu-backlog-high",
        type=float,
        default=2.0,
        help="mars only: treat CPU as overloaded when tool inflight/worker >= this threshold.",
    )
    ap.add_argument(
        "--mars-cpu-backlog-low",
        type=float,
        default=1.0,
        help="mars only: switch back to CPU-healthy when tool inflight/worker <= this threshold.",
    )
    ap.add_argument(
        "--mars-cpu-queue-wait-high-s",
        type=float,
        default=5.0,
        help="mars only: treat CPU as overloaded when EMA tool queue wait >= this threshold (seconds).",
    )
    ap.add_argument(
        "--mars-cpu-queue-wait-low-s",
        type=float,
        default=2.0,
        help="mars only: switch back to CPU-healthy when EMA tool queue wait <= this threshold (seconds).",
    )
    ap.add_argument(
        "--guided-decoding-backend",
        type=str,
        choices=["auto", "xgrammar", "guidance", "outlines", "lm-format-enforcer"],
        default="xgrammar",
        help="Forwarded to vLLM engine (guided decoding backend).",
    )
    ap.add_argument(
        "--guided-decoding-disable-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forwarded to vLLM engine. If true, do not fall back to another guided decoding backend on error.",
    )
    ap.add_argument(
        "--guided-decoding-disable-any-whitespace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to vLLM engine. If true, disallow arbitrary whitespace between JSON tokens to avoid whitespace-token loops.",
    )
    ap.add_argument(
        "--guided-decoding-disable-additional-properties",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forwarded to vLLM engine. If true, do not enforce JSONSchema additionalProperties (use runtime validation instead).",
    )
    ap.add_argument(
        "--log-stats",
        action="store_true",
        help="Enable vLLM periodic stats logs (includes KV cache usage).",
    )
    ap.add_argument("--log-requests", action="store_true", help="Enable vLLM request logs (debug).")
    ap.add_argument(
        "--kv-events",
        action="store_true",
        help="Enable vLLM KV cache events (via ZMQ) and dump to kv_events.jsonl in the run output dir.",
    )

    ap.add_argument(
        "--emit-mode",
        type=str,
        choices=["rps", "poisson", "poison", "burst"],
        default="rps",
        help=(
            "Request emission mode. rps uses fixed-rate arrivals; "
            "poisson uses exponential inter-arrival times with rate --rps; "
            "burst submits as fast as possible (ignores --rps)."
        ),
    )
    ap.add_argument("--rps", type=float, default=1, help="Fixed request emission rate")
    ap.add_argument("--num-requests", type=int, default=None)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-timeout-s", type=float, default=60000.0)
    # Agent loop knobs
    ap.add_argument("--max-rounds", type=int, default=8, help="Max tool/action rounds per request")
    ap.add_argument("--action-max-tokens", type=int, default=8192, help="Max tokens for each JSON action generation")
    ap.add_argument("--action-json-retries", type=int, default=2, help="Retries when JSON action is invalid/truncated")
    ap.add_argument(
        "--request-retry-max",
        type=int,
        default=3,
        help=(
            "Max in-request parse-feedback retries after repeated action JSON parse failures. "
            "Only enabled for GPT-OSS 120B. 0 disables parse-feedback retry."
        ),
    )
    ap.add_argument(
        "--action-guided-decoding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use vLLM guided decoding (JSON schema) for actions. Slower but can reduce format errors.",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--run-max-commands", type=int, default=6)
    ap.add_argument("--run-max-command-chars", type=int, default=800)
    ap.add_argument("--write-file-max-chars", type=int, default=2000)
    ap.add_argument("--system-prompt-file", type=str, default=None, help="Override system prompt from a UTF-8 file")

    ap.add_argument("--tool-sleep-s", type=float, default=3.0)
    ap.add_argument("--tool-keep-venv", action="store_true")
    ap.add_argument("--tool-venv-timeout-s", type=float, default=None)
    ap.add_argument(
        "--tool-command-timeout-s",
        type=float,
        default=300.0,
        help="Default timeout for each tool 'run' command (seconds). Set <=0 to disable.",
    )
    ap.add_argument(
        "--tool-env-mode",
        type=str,
        choices=["per_invocation_venv", "per_request_venv", "workspace_venv"],
        default="per_request_venv",
        help="Tool execution environment semantics. workspace_venv means the agent creates/uses ./ .venv in the request workspace.",
    )
    ap.add_argument(
        "--thread_pool",
        type=str,
        choices=["on", "off"],
        default="off",
        help=(
            "Tool execution mode. "
            "off: one tool worker per CPU core (tool calls queue for free cores). "
            "on: spawn one subprocess per tool call pinned to --cpu-tool (no core-queueing; OS time-slices)."
        ),
    )

    ap.add_argument("--cpu-gpu", type=str, default=_default_cpu_gpu())
    ap.add_argument("--cpu-tool", type=str, default="0-7")
    ap.add_argument("--output-dir", type=str, default=None)

    ap.set_defaults(**cfg_defaults)
    args = ap.parse_args()
    args.cuda_graph_sizes = _parse_cuda_graph_sizes(args.cuda_graph_sizes)

    if args.tasks is None:
        raise SystemExit("Must set --tasks (or provide it in --config)")
    if (args.num_requests is None) == (args.duration_s is None):
        raise SystemExit("Must set exactly one of --num-requests or --duration-s")
    if str(args.emit_mode) in ("rps", "poisson", "poison"):
        if args.rps is None:
            raise SystemExit("Must set --rps (or provide it in --config) when --emit-mode=rps/poisson")
        if float(args.rps) <= 0:
            raise SystemExit("--rps must be > 0 when --emit-mode=rps/poisson")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
