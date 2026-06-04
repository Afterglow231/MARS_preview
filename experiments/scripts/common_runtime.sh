#!/usr/bin/env bash

# Shared helpers for experiment launchers. This file is intended to be sourced.

mars_bool_true() {
  case "${1:-}" in
    1|on|true|yes|y|TRUE|ON|True|YES|Y) return 0 ;;
    *) return 1 ;;
  esac
}

mars_default_python() {
  local candidate="$1"
  if [[ -n "${candidate}" && -x "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  printf '%s\n' "${candidate}"
}

mars_print_vllm_build_help() {
  local mars_dir="$1"
  local python_bin="${2:-python}"

  cat >&2 <<EOF
Build the local MARS vLLM tree before running GPU experiments:

  cd ${mars_dir}/vllm-continuum
  MODE=source bash scripts/create_continuum_vllm_conda_env.sh
  conda activate mars

If the conda environment already exists, rebuild this source tree in it:

  conda activate mars
  cd ${mars_dir}/vllm-continuum
  rm -f vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so
  mkdir -p vllm/vllm_flash_attn
  python -m pip install -e . --no-build-isolation --no-deps

Verify the install:

  PYTHONPATH=${mars_dir}:${mars_dir}/vllm-continuum ${python_bin} - <<'PY'
import importlib, vllm
print(vllm.__file__)
for name in ("vllm.vllm_flash_attn._vllm_fa2_C", "vllm.vllm_flash_attn._vllm_fa3_C"):
    print(name, importlib.import_module(name).__file__)
PY
EOF
}

mars_ensure_vllm_flash_attn_extensions() {
  local mars_dir="$1"
  local python_bin="${2:-python3}"

  if mars_bool_true "${VLLM_SKIP_FLASH_ATTN_EXT_CHECK:-0}"; then
    return 0
  fi

  local dst="${mars_dir}/vllm-continuum/vllm/vllm_flash_attn"
  local fa2="${dst}/_vllm_fa2_C.abi3.so"
  local fa3="${dst}/_vllm_fa3_C.abi3.so"
  local missing=()

  [[ -e "${fa2}" ]] || missing+=("${fa2}")
  [[ -e "${fa3}" ]] || missing+=("${fa3}")

  if (( ${#missing[@]} > 0 )); then
    echo "Missing vLLM FlashAttention extension module(s):" >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo "" >&2
    echo "This usually means vllm-continuum has not been built from source in the active Python environment." >&2
    echo "Without these modules, vLLM may fail later with: ValueError: Unsupported FA version: None" >&2
    echo "" >&2
    mars_print_vllm_build_help "${mars_dir}" "${python_bin}"
    exit 1
  fi

  if [[ -n "${python_bin}" && -x "${python_bin}" ]]; then
    if ! PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="${mars_dir}:${mars_dir}/vllm-continuum${PYTHONPATH:+:${PYTHONPATH}}" \
      "${python_bin}" - <<'PY' >/dev/null 2>&1
import importlib
for name in ("vllm.vllm_flash_attn._vllm_fa2_C", "vllm.vllm_flash_attn._vllm_fa3_C"):
    importlib.import_module(name)
PY
    then
      echo "Found vLLM FlashAttention extension files, but ${python_bin} could not import them." >&2
      echo "The extensions may have been built for a different Python, PyTorch, or CUDA ABI." >&2
      echo "" >&2
      mars_print_vllm_build_help "${mars_dir}" "${python_bin}"
      exit 1
    fi
  fi
}

mars_pick_free_port() {
  local python_bin="$1"
  "${python_bin}" - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
}

mars_infer_tp_from_devices() {
  local devices="${1// /}"
  if [[ -z "${devices}" || "${devices}" == "all" ]]; then
    echo 1
    return 0
  fi
  IFS=',' read -r -a _mars_gpus <<< "${devices}"
  if [[ "${#_mars_gpus[@]}" -lt 1 ]]; then
    echo 1
  else
    echo "${#_mars_gpus[@]}"
  fi
}

mars_merge_additional_config() {
  local python_bin="$1"
  local existing_json="$2"
  local mars_active_pool_size="$3"
  local mars_cpu_backlog_low="$4"
  local mars_cpu_backlog_high="$5"
  "${python_bin}" - "${existing_json}" "${mars_active_pool_size}" "${mars_cpu_backlog_low}" "${mars_cpu_backlog_high}" <<'PY'
import json
import sys

raw = sys.argv[1]
data = json.loads(raw) if raw.strip() else {}
if not isinstance(data, dict):
    raise SystemExit("--additional-config must be a JSON object")
data.setdefault("mars_active_pool_size", int(sys.argv[2]))
data.setdefault("mars_cpu_backlog_low", float(sys.argv[3]))
data.setdefault("mars_cpu_backlog_high", float(sys.argv[4]))
print(json.dumps(data, separators=(",", ":")))
PY
}

mars_wait_for_vllm() {
  local python_bin="$1"
  local base_url="$2"
  local timeout_s="${3:-300}"
  local server_pid="${4:-}"
  local log_path="${5:-}"
  local require_kv_stats="${6:-0}"
  "${python_bin}" - "${base_url}" "${timeout_s}" "${server_pid}" "${log_path}" "${require_kv_stats}" <<'PY'
import os
from pathlib import Path
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

base_url = sys.argv[1].rstrip("/")
timeout_s = float(sys.argv[2])
server_pid = int(sys.argv[3]) if sys.argv[3].strip() else None
log_path = Path(sys.argv[4]) if sys.argv[4].strip() else None
require_kv_stats = sys.argv[5].strip().lower() in {"1", "on", "true", "yes", "y"}
models_url = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"
kv_stats_url = f"{base_url}/kv_cache_stats" if base_url.endswith("/v1") else f"{base_url}/v1/kv_cache_stats"
deadline = time.time() + timeout_s
last_err = None

def alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True

def log_tail() -> str:
    if log_path is None or not log_path.exists():
        return ""
    try:
        return "\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:])
    except Exception:
        return ""

def http_ok(url: str) -> bool:
    try:
        with urlopen(url, timeout=2.0) as resp:
            return 200 <= resp.status < 300 and bool(resp.read(4096))
    except URLError as exc:
        global last_err
        last_err = f"{type(exc).__name__}: {exc}"
        return False

while time.time() < deadline:
    if not alive(server_pid):
        detail = f"vLLM server pid {server_pid} exited before readiness probe succeeded."
        tail = log_tail()
        raise SystemExit(detail + (f"\nLast log lines:\n{tail}" if tail else ""))
    if http_ok(models_url) and (not require_kv_stats or http_ok(kv_stats_url)):
        raise SystemExit(0)
    time.sleep(0.5)
raise SystemExit(f"Timed out waiting for vLLM server at {models_url}: {last_err}")
PY
}

mars_assert_local_port_available() {
  local python_bin="$1"
  local host="$2"
  local port="$3"
  local label="$4"
  "${python_bin}" - "${host}" "${port}" "${label}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
label = sys.argv[3]
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
except OSError as exc:
    raise SystemExit(f"{label} port {port} on {host} is not available: {exc}")
finally:
    sock.close()
PY
}

mars_assert_local_port_range_available() {
  local python_bin="$1"
  local host="$2"
  local start_port="$3"
  local count="$4"
  local label="$5"
  "${python_bin}" - "${host}" "${start_port}" "${count}" "${label}" <<'PY'
import socket
import sys

host = sys.argv[1]
start_port = int(sys.argv[2])
count = int(sys.argv[3])
label = sys.argv[4]
for port in range(start_port, start_port + count):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    except OSError as exc:
        raise SystemExit(
            f"{label} port range {start_port}-{start_port + count - 1} on "
            f"{host} is not available because port {port} failed: {exc}"
        )
    finally:
        sock.close()
PY
}
