#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-}"
TP="${TP:-1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-continuum}" # fcfs|priority|continuum
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-./continuum_exp}"

# Optional: recording / monitoring
# When RECORD=1, write:
#   ${RUN_OUTPUT_DIR}/vllm_record/<run_id>/vllm_serve.log
#   ${RUN_OUTPUT_DIR}/vllm_record/<run_id>/metrics_gpu.jsonl
RECORD="${RECORD:-0}" # 1 to enable recording
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1}"
LOG_STATS_INTERVAL="${LOG_STATS_INTERVAL:-1}"
ENABLE_LOG_REQUESTS="${ENABLE_LOG_REQUESTS:-}"

# Optional: CPU KV offload via LMCache
LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-}"
KV_TRANSFER_CONFIG_JSON="${KV_TRANSFER_CONFIG_JSON:-}"
if [[ -n "${LMCACHE_MAX_LOCAL_CPU_SIZE}" && -z "${KV_TRANSFER_CONFIG_JSON}" ]]; then
  KV_TRANSFER_CONFIG_JSON='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
fi

if [[ -z "${MODEL}" ]]; then
  echo "MODEL is required. Example:" >&2
  echo "  MODEL=meta-llama/Llama-3.1-70B-Instruct TP=4 bash scripts/start_vllm_service.sh" >&2
  exit 1
fi

if command -v python >/dev/null 2>&1; then
  GPU_COUNT="$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "")"
  if [[ -n "${GPU_COUNT}" ]] && [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]]; then
    if (( TP > GPU_COUNT )); then
      echo "TP=${TP} but only ${GPU_COUNT} CUDA device(s) visible." >&2
      echo "Fix: set TP=${GPU_COUNT} (or adjust CUDA_VISIBLE_DEVICES / request more GPUs)." >&2
      exit 2
    fi
  fi
fi

mkdir -p "${RUN_OUTPUT_DIR}"

args=(
  serve
  "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP}"
  --scheduling-policy "${SCHEDULING_POLICY}"
)

if [[ -n "${MAX_MODEL_LEN}" ]]; then
  args+=(--max-model-len "${MAX_MODEL_LEN}")
fi

if [[ -n "${KV_TRANSFER_CONFIG_JSON}" ]]; then
  args+=(--kv-transfer-config "${KV_TRANSFER_CONFIG_JSON}")
fi

if [[ "${RECORD}" == "1" && -z "${ENABLE_LOG_REQUESTS}" ]]; then
  ENABLE_LOG_REQUESTS="1"
fi
if [[ "${ENABLE_LOG_REQUESTS}" == "1" ]]; then
  args+=(--enable-log-requests)
fi

echo "RUN_OUTPUT_DIR=${RUN_OUTPUT_DIR}"
echo "Starting vLLM: vllm ${args[*]}"

if [[ "${RECORD}" != "1" ]]; then
  exec env RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR}" \
    ${LMCACHE_MAX_LOCAL_CPU_SIZE:+LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE}"} \
    vllm "${args[@]}"
fi

run_id="$(date +%Y%m%d_%H%M%S)"
record_dir="${RUN_OUTPUT_DIR%/}/vllm_record/${run_id}"
mkdir -p "${record_dir}"

cat >"${record_dir}/run_info.txt" <<EOF
run_id=${run_id}
date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cwd=$(pwd)
MODEL=${MODEL}
HOST=${HOST}
PORT=${PORT}
TP=${TP}
SCHEDULING_POLICY=${SCHEDULING_POLICY}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
RUN_OUTPUT_DIR=${RUN_OUTPUT_DIR}
INTERVAL_SECONDS=${INTERVAL_SECONDS}
LOG_STATS_INTERVAL=${LOG_STATS_INTERVAL}
ENABLE_LOG_REQUESTS=${ENABLE_LOG_REQUESTS:-0}
LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_MAX_LOCAL_CPU_SIZE}
KV_TRANSFER_CONFIG_JSON=${KV_TRANSFER_CONFIG_JSON}
EOF

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >"${record_dir}/nvidia_smi_start.txt" 2>/dev/null || true
fi

# Determine metrics URL (0.0.0.0 is not routable for clients).
metrics_host="${HOST}"
if [[ "${metrics_host}" == "0.0.0.0" ]]; then
  metrics_host="127.0.0.1"
fi
METRICS_URL="${METRICS_URL:-http://${metrics_host}:${PORT}/metrics}"

monitor_pid=""
cleanup() {
  if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" >/dev/null 2>&1; then
    kill "${monitor_pid}" >/dev/null 2>&1 || true
    wait "${monitor_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Recording enabled. Logs: ${record_dir}"
echo "Monitoring ${METRICS_URL} + nvidia-smi every ${INTERVAL_SECONDS}s -> ${record_dir}/metrics_gpu.jsonl"

# Start a lightweight monitor in background (JSONL).
python -u - "${METRICS_URL}" "${INTERVAL_SECONDS}" "${record_dir}/metrics_gpu.jsonl" >"${record_dir}/monitor.log" 2>&1 <<'PY' &
import datetime as dt
import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

metrics_url = sys.argv[1]
interval_s = max(0.1, float(sys.argv[2]))
out_path = sys.argv[3]

keep = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:gpu_prefix_cache_queries_total",
    "vllm:gpu_prefix_cache_hits_total",
}
line_re = re.compile(
    r"^(?P<name>[^ {]+)(?:\\{(?P<labels>[^}]*)\\})?\\s+(?P<value>[-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?)$"
)

def iso_utc(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat()

def safe_float(s: str) -> Optional[float]:
    try:
        v = float(s)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v

def fetch_metrics(url: str, timeout_s: float = 2.0) -> Tuple[Dict[str, float], Optional[str]]:
    req = urllib.request.Request(url, headers={"Accept": "text/plain; version=0.0.4"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return {}, f"metrics fetch failed: {e}"
    vals: Dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = line_re.match(line)
        if not m:
            continue
        name = m.group("name")
        if name not in keep:
            continue
        v = safe_float(m.group("value"))
        if v is None:
            continue
        vals[name] = vals.get(name, 0.0) + v
    return vals, None

def run_nvidia_smi() -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[str]]:
    try:
        gpu_out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="replace")
    except FileNotFoundError:
        return None, None, "nvidia-smi not found"
    except subprocess.CalledProcessError as e:
        return None, None, f"nvidia-smi gpu query failed: {e.output!r}"

    gpus: List[Dict[str, Any]] = []
    for line in gpu_out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "util_gpu_perc": safe_float(parts[3]),
                "util_mem_perc": safe_float(parts[4]),
                "mem_total_mb": safe_float(parts[5]),
                "mem_used_mb": safe_float(parts[6]),
                "temp_c": safe_float(parts[7]),
                "power_w": safe_float(parts[8]),
            }
        )

    procs: List[Dict[str, Any]] = []
    try:
        proc_out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="replace")
        for line in proc_out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            procs.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": int(parts[1]) if parts[1].isdigit() else parts[1],
                    "process_name": parts[2],
                    "used_memory_mb": safe_float(parts[3]),
                }
            )
    except subprocess.CalledProcessError as e:
        return {"gpus": gpus}, None, f"nvidia-smi proc query failed: {e.output!r}"

    return {"gpus": gpus}, procs, None

def compute_derived(current: Dict[str, float], prev: Optional[Dict[str, float]], dt_s: Optional[float]) -> Dict[str, Any]:
    if prev is None or dt_s is None or dt_s <= 0:
        return {}

    def delta(name: str) -> Optional[float]:
        if name not in current or name not in prev:
            return None
        d = current[name] - prev[name]
        if not math.isfinite(d) or d < 0:
            return None
        return d

    out: Dict[str, Any] = {}
    prompt_delta = delta("vllm:prompt_tokens_total")
    gen_delta = delta("vllm:generation_tokens_total")
    if prompt_delta is not None:
        out["prompt_tokens_per_s"] = prompt_delta / dt_s
    if gen_delta is not None:
        out["generation_tokens_per_s"] = gen_delta / dt_s

    hit_delta = delta("vllm:prefix_cache_hits_total")
    query_delta = delta("vllm:prefix_cache_queries_total")
    if hit_delta is not None and query_delta is not None and query_delta > 0:
        out["prefix_cache_hit_rate"] = hit_delta / query_delta

    total_hits = current.get("vllm:prefix_cache_hits_total")
    total_queries = current.get("vllm:prefix_cache_queries_total")
    if (
        total_hits is not None
        and total_queries is not None
        and math.isfinite(total_hits)
        and math.isfinite(total_queries)
        and total_queries > 0
    ):
        out["prefix_cache_hit_rate_cumulative"] = total_hits / total_queries
    return out

prev_metrics: Optional[Dict[str, float]] = None
prev_monotonic: Optional[float] = None

with open(out_path, "a", encoding="utf-8") as f:
    while True:
        ts = time.time()
        mono = time.monotonic()
        dt_s: Optional[float] = None
        if prev_monotonic is not None:
            dt_s = mono - prev_monotonic

        metrics, metrics_err = fetch_metrics(metrics_url)
        gpu, procs, gpu_err = run_nvidia_smi()
        derived = compute_derived(metrics, prev_metrics, dt_s)

        rec: Dict[str, Any] = {
            "ts_unix": ts,
            "ts_utc": iso_utc(ts),
            "monotonic_s": mono,
            "interval_s": dt_s,
            "metrics_url": metrics_url,
            "metrics": metrics,
            "derived": derived,
            "gpu": gpu,
            "gpu_processes": procs,
        }
        errs: List[str] = []
        if metrics_err:
            errs.append(metrics_err)
        if gpu_err:
            errs.append(gpu_err)
        if errs:
            rec["errors"] = errs

        f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        f.flush()

        prev_metrics = metrics
        prev_monotonic = mono

        time.sleep(interval_s)
PY
monitor_pid="$!"

env RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR}" \
  VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-${LOG_STATS_INTERVAL}}" \
  ${LMCACHE_MAX_LOCAL_CPU_SIZE:+LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE}"} \
  vllm "${args[@]}" 2>&1 | tee "${record_dir}/vllm_serve.log"
