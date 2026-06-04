#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional

import requests


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _set_affinity(cores: list[int]) -> None:
    try:
        os.sched_setaffinity(0, set(cores))  
    except Exception:
        pass


@dataclass(frozen=True)
class VLLMServerConfig:
    python_bin: str
    model_path: str
    host: str = "127.0.0.1"
    port: int = 0
    served_model_name: Optional[str] = None
    gpu_cores: Optional[list[int]] = None
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    trust_remote_code: bool = True
    disable_log_stats: bool = True
    disable_uvicorn_access_log: bool = True
    extra_args: Optional[list[str]] = None
    log_file: Optional[Path] = None
    env_offline: bool = True


class VLLMServer:
    def __init__(self, cfg: VLLMServerConfig):
        self.cfg = cfg
        self.port = cfg.port if cfg.port != 0 else _pick_free_port()
        self.base_url = f"http://{cfg.host}:{self.port}"
        self._proc: Optional[subprocess.Popen] = None
        self._log_fp: Optional[IO[str]] = None
        self._cmd: Optional[list[str]] = None

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("Server already started")

        cmd = [
            self.cfg.python_bin,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            self.cfg.host,
            "--port",
            str(self.port),
            "--model",
            self.cfg.model_path,
            "--dtype",
            self.cfg.dtype,
            "--tensor-parallel-size",
            str(self.cfg.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.cfg.gpu_memory_utilization),
            "--enable-force-include-usage",
        ]
        if self.cfg.trust_remote_code:
            cmd.append("--trust-remote-code")
        else:
            cmd.append("--no-trust-remote-code")

        if self.cfg.disable_log_stats:
            cmd.append("--disable-log-stats")
        if self.cfg.disable_uvicorn_access_log:
            cmd.append("--disable-uvicorn-access-log")
        if self.cfg.max_model_len is not None:
            cmd += ["--max-model-len", str(self.cfg.max_model_len)]
        if self.cfg.served_model_name:
            cmd += ["--served-model-name", self.cfg.served_model_name]
        if self.cfg.extra_args:
            cmd += list(self.cfg.extra_args)

        env = os.environ.copy()
        if self.cfg.env_offline:
            env.setdefault("HF_HUB_OFFLINE", "1")
            env.setdefault("TRANSFORMERS_OFFLINE", "1")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
        if self.cfg.log_file is not None:
            env.setdefault("RUN_OUTPUT_DIR", str(self.cfg.log_file.parent))

        stdout = None
        stderr = None
        if self.cfg.log_file is not None:
            self.cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_fp = self.cfg.log_file.open("a", encoding="utf-8")
            stdout = self._log_fp
            stderr = self._log_fp

        def _preexec():
            if self.cfg.gpu_cores:
                _set_affinity(self.cfg.gpu_cores)

        try:
            self._cmd = cmd
            self._proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=stdout,
                stderr=stderr,
                preexec_fn=_preexec if self.cfg.gpu_cores else None,
            )
        except Exception:
            if self._log_fp is not None:
                try:
                    self._log_fp.close()
                except Exception:
                    pass
                self._log_fp = None
            raise

    def _read_log_tail(self, *, max_lines: int = 120, max_chars: int = 12000) -> Optional[str]:
        p = self.cfg.log_file
        if p is None or not p.exists():
            return None
        try:
            max_bytes = max(1024, int(max_chars) * 4)
            with p.open("rb") as f:
                f.seek(0, os.SEEK_END)
                n = f.tell()
                f.seek(max(0, n - max_bytes))
                b = f.read()
            s = b.decode("utf-8", errors="ignore")
            lines = s.splitlines()[-max_lines:]
            out = "\n".join(lines).strip()
            return out if out else None
        except Exception:
            return None

    def wait_ready(self, timeout_s: float = 300.0) -> None:
        deadline = time.time() + timeout_s
        last_err: Optional[str] = None
        while time.time() < deadline:
            if self._proc is None:
                raise RuntimeError("Server not started")
            if self._proc.poll() is not None:
                tail = self._read_log_tail()
                cmd = " ".join(self._cmd or [])
                hint = None
                if tail:
                    m = re.search(r"estimated maximum model length is (\\d+)", tail)
                    if m:
                        hint = f"Hint: try setting --max-model-len <= {m.group(1)} (or increase --gpu-memory-utilization)."
                msg = f"vLLM server exited early (code={self._proc.returncode}): {last_err or ''}"
                if cmd:
                    msg += f"\ncmd: {cmd}"
                if hint:
                    msg += f"\n{hint}"
                if tail:
                    msg += f"\n--- vllm_server.log tail ---\n{tail}"
                raise RuntimeError(msg)
            try:
                r = requests.get(f"{self.base_url}/v1/models", timeout=2.0)
                if r.status_code == 200:
                    return
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(0.5)
        raise TimeoutError(f"Timed out waiting for vLLM server: {last_err or ''}")

    def get_first_model_id(self) -> str:
        r = requests.get(f"{self.base_url}/v1/models", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        models = data.get("data") or []
        if not models:
            raise RuntimeError(f"Unexpected /v1/models response: {data}")
        mid = models[0].get("id")
        if not mid:
            raise RuntimeError(f"Unexpected /v1/models response: {data}")
        return str(mid)

    def stop(self, timeout_s: float = 15.0) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=timeout_s)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None
            if self._log_fp is not None:
                try:
                    self._log_fp.close()
                except Exception:
                    pass
                self._log_fp = None
