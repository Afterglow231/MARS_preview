#!/usr/bin/env python3
"""Write a reproducible backend manifest without importing either server."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable


MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
)


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _model_fingerprint(model_path: Path, mode: str) -> dict[str, Any]:
    if not model_path.is_dir():
        raise SystemExit(f"Model directory not found: {model_path}")

    selected: list[Path] = []
    for name in MODEL_METADATA_FILES:
        path = model_path / name
        if path.is_file():
            selected.append(path)
    weight_files = sorted(
        path
        for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth")
        for path in model_path.glob(pattern)
        if path.is_file()
    )

    entries: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for path in sorted(selected) + weight_files:
        relative = str(path.relative_to(model_path))
        size = path.stat().st_size
        file_hash = _sha256_file(path) if (mode == "full" or path in selected) else None
        entry = {"path": relative, "size": size, "sha256": file_hash}
        entries.append(entry)
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(size).encode("ascii"))
        combined.update(b"\0")
        combined.update((file_hash or "metadata-only").encode("ascii"))
        combined.update(b"\n")

    return {
        "mode": mode,
        "digest": combined.hexdigest(),
        "files": entries,
        "weight_file_count": len(weight_files),
        "weight_bytes": sum(path.stat().st_size for path in weight_files),
    }


def _run_text(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_info(source_dir: str | None) -> dict[str, Any]:
    if not source_dir:
        return {"source_dir": None, "actual_commit": None, "dirty": None}
    source = Path(source_dir).expanduser().resolve()
    actual = _run_text(["git", "-C", str(source), "rev-parse", "HEAD"])
    status = _run_text(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"]
    )
    return {
        "source_dir": str(source),
        "actual_commit": actual,
        "dirty": None if status is None else bool(status),
    }


def _python_info(python_bin: str | None, *, include_packages: bool) -> dict[str, Any]:
    if not python_bin:
        return {
            "python": None,
            "python_realpath": None,
            "python_version": None,
            "python_prefix": None,
            "packages": [],
        }

    # Do not resolve the executable symlink before invoking it. A venv's
    # ``bin/python`` commonly points at the system interpreter, while Python
    # discovers the venv from the path used to start it. Executing the resolved
    # target would silently fingerprint the system environment instead.
    requested_path = Path(python_bin).expanduser().absolute()
    python_path = str(requested_path)
    if not requested_path.is_file() or not os.access(requested_path, os.X_OK):
        raise SystemExit(f"Python interpreter is not executable: {python_path}")
    python_realpath = str(requested_path.resolve())
    version = _run_text(
        [
            python_path,
            "-c",
            "import platform; print(platform.python_version())",
        ]
    )
    if version is None:
        raise SystemExit(f"Cannot inspect Python interpreter: {python_path}")
    python_prefix = _run_text(
        [python_path, "-c", "import sys; print(sys.prefix)"]
    )
    if python_prefix is None:
        raise SystemExit(f"Cannot inspect Python prefix: {python_path}")
    packages: list[str] = []
    if include_packages:
        freeze = _run_text([python_path, "-m", "pip", "freeze", "--all"])
        if freeze is None:
            raise SystemExit(f"Cannot inspect Python packages: {python_path}")
        if freeze:
            packages = sorted(line for line in freeze.splitlines() if line.strip())
    return {
        "python": python_path,
        "python_realpath": python_realpath,
        "python_version": version,
        "python_prefix": python_prefix,
        "packages": packages,
    }


def _client_source_fingerprint(specs: Iterable[str]) -> dict[str, Any]:
    """Hash the shared client implementation by stable logical labels."""

    entries: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for spec in specs:
        label, separator, raw_path = spec.partition("=")
        label = label.strip()
        if not separator or not label or not raw_path.strip():
            raise SystemExit(
                "--client-source must use a stable label=/path/to/file form: "
                f"{spec!r}"
            )
        if label in seen_labels:
            raise SystemExit(f"Duplicate --client-source label: {label}")
        seen_labels.add(label)
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Client source file not found: {path}")
        entries.append(
            {
                "label": label,
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )

    entries.sort(key=lambda entry: entry["label"])
    combined = hashlib.sha256()
    for entry in entries:
        combined.update(entry["label"].encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(entry["size"]).encode("ascii"))
        combined.update(b"\0")
        combined.update(entry["sha256"].encode("ascii"))
        combined.update(b"\n")
    return {"digest": combined.hexdigest(), "files": entries}


def _gpu_info(cuda_visible_devices: str) -> list[dict[str, Any]]:
    devices = cuda_visible_devices.strip()
    command = ["nvidia-smi"]
    if devices and devices.lower() != "all":
        command.extend(["-i", devices])
    command.extend(
        [
            "--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    output = _run_text(command)
    if not output:
        return []
    keys = ("index", "uuid", "name", "memory_total_mib", "driver_version", "pci_bus_id")
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        values = [part.strip() for part in line.split(",", maxsplit=5)]
        if len(values) == len(keys):
            rows.append(dict(zip(keys, values)))
    return rows


def _bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "y"}:
        return True
    if normalized in {"0", "false", "no", "off", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def _optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend-name", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--image-digest", default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--include-packages", type=_bool, default=False)
    parser.add_argument("--orchestrator-python", required=True)
    parser.add_argument("--openhands-python", required=True)
    parser.add_argument(
        "--client-source",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Shared workload client source file to hash; repeat as needed.",
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--nccl-port", type=int, required=True)
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument(
        "--model-hash-mode",
        choices=("metadata", "full"),
        default="full",
    )
    parser.add_argument("--tasks-path", required=True)
    parser.add_argument("--rps", type=float, required=True)
    parser.add_argument("--max-requests", type=_optional_int, default=None)
    parser.add_argument("--shuffle", type=_bool, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--llm-timeout-s", type=int, required=True)
    parser.add_argument("--llm-num-retries", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--emit-mode", required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--cpu-gpu", required=True)
    parser.add_argument("--cpu-tool", required=True)
    parser.add_argument("--cpu-cores", required=True)
    parser.add_argument("--warmup-enabled", type=_bool, required=True)
    parser.add_argument("--warmup-requests", type=int, required=True)
    parser.add_argument("--warmup-max-tokens", type=int, required=True)
    parser.add_argument("--warmup-artifact", required=True)
    parser.add_argument("--tool-call-parser", required=True)
    parser.add_argument("--auto-tool-choice-enabled", type=_bool, required=True)
    parser.add_argument("--kv-events", type=_bool, required=True)
    parser.add_argument("--triton-autotune-level", type=int, required=True)
    parser.add_argument("--load-worker", type=int, required=True)
    parser.add_argument("--upstream-patched", type=_bool, default=False)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    tasks_path = Path(args.tasks_path).expanduser().resolve()
    if not tasks_path.is_file():
        raise SystemExit(f"Tasks file not found: {tasks_path}")

    git_info = _git_info(args.source_dir)
    python_info = _python_info(args.python, include_packages=args.include_packages)
    orchestrator_python_info = _python_info(
        args.orchestrator_python, include_packages=True)
    openhands_python_info = _python_info(
        args.openhands_python, include_packages=True)
    client_source = _client_source_fingerprint(args.client_source)
    manifest = {
        "schema": "mars_backend_provenance_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "backend": {
            "name": args.backend_name,
            "repo_url": args.repo_url,
            "ref": args.ref,
            "expected_commit": args.expected_commit,
            **git_info,
            "runtime": args.runtime,
            "image": args.image,
            "image_digest": args.image_digest,
            **python_info,
        },
        "client": {
            "environment": {
                "orchestrator": orchestrator_python_info,
                "openhands": openhands_python_info,
            },
            "source": client_source,
        },
        "server": {
            "command": list(args.command),
            "host": args.host,
            "port": args.port,
            "nccl_port": args.nccl_port,
            "log_path": str(Path(args.server_log).expanduser().resolve()),
        },
        "model": {
            "path": str(model_path),
            "served_name": args.served_model_name,
            "fingerprint": _model_fingerprint(model_path, args.model_hash_mode),
        },
        "workload": {
            "tasks_path": str(tasks_path),
            "tasks_sha256": _sha256_file(tasks_path),
            "rps": args.rps,
            "max_requests": args.max_requests,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_output_tokens": args.max_output_tokens,
            "llm_timeout_s": args.llm_timeout_s,
            "llm_num_retries": args.llm_num_retries,
            "max_iterations": args.max_iterations,
            "emit_mode": args.emit_mode,
            "max_workers": args.max_workers,
        },
        "resources": {
            "cuda_visible_devices": args.cuda_visible_devices,
            "tp": args.tp,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "cpu_gpu": args.cpu_gpu,
            "cpu_tool": args.cpu_tool,
            "cpu_cores": args.cpu_cores,
            "gpus": _gpu_info(args.cuda_visible_devices),
        },
        "warmup": {
            "enabled": args.warmup_enabled,
            "requests": args.warmup_requests,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.warmup_max_tokens,
            "artifact": str(Path(args.warmup_artifact).expanduser().resolve()),
        },
        "integration": {
            "tool_call_parser": args.tool_call_parser,
            "auto_tool_choice_enabled": args.auto_tool_choice_enabled,
            "kv_events": args.kv_events,
            "upstream_patched": args.upstream_patched,
            "triton_autotune_level": args.triton_autotune_level,
            "load_worker": args.load_worker,
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    temp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
