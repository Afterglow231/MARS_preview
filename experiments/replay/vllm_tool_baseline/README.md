# vLLM + Serverless Tool Baseline

This runner executes `mas_task_v1` workloads with an in-process vLLM engine and a CPU tool pool. Each request follows a serial GPU/tool pattern and writes timeline events, per-request traces, and a summary.

Run from the repository root:

```bash
cd /path/to/MARS
export MODEL_PATH=${MODEL_PATH:-$HOME/models/Qwen3-Coder-30B-A3B-Instruct}
```

## Quick Run

```bash
CUDA_VISIBLE_DEVICES=3 \
python3 experiments/replay/vllm_tool_baseline/run_experiment.py \
  --tasks benchmark/hybrid_workloads/test_workload.jsonl \
  --model-path "$MODEL_PATH" \
  --scheduling-policy mars \
  --rps 0.2 --num-requests 10 \
  --output-dir outputs/testbed_smoke
```

The wrapper script provides the same path-relative defaults:

```bash
CUDA_VISIBLE_DEVICES=3 \
MODEL_PATH="$MODEL_PATH" \
TASKS_PATH=$PWD/benchmark/hybrid_workloads/test_workload.jsonl \
NUM_REQUESTS=10 RPS=0.2 SCHEDULING_POLICY=mars \
bash experiments/scripts/run_baseline_continuum.sh
```

## InferCept + LMCache Swap

Use `infercept_swap` to let InferCept choose among evict, preserve-on-GPU, and
LMCache-backed swap for tool-call KV:

```bash
CUDA_VISIBLE_DEVICES=3 \
python3 experiments/replay/vllm_tool_baseline/run_experiment.py \
  --tasks benchmark/hybrid_workloads/test_workload.jsonl \
  --model-path "$MODEL_PATH" \
  --scheduling-policy infercept_swap \
  --rps 0.2 --num-requests 10 \
  --output-dir outputs/infercept_swap
```

When no `--lmcache-config-file` is supplied, the runner writes
`lmcache_cpu.yaml` in the run output directory and uses
`SelectiveLMCacheConnectorV1` with a local CPU LMCache backend.

For validation runs where you want to observe actual CPU-to-GPU LMCache loads
instead of vLLM GPU prefix-cache hits, add `--no-enable-prefix-caching`.

InferCept decisions are written to `infercept_kv_decisions.jsonl` in the run
output directory. To count the selected KV actions:

```bash
jq -r '.action' outputs/infercept_swap/<run>/infercept_kv_decisions.jsonl | sort | uniq -c
```

To keep the full vLLM/LMCache stdout and stderr for later grepping, wrap the
run with `tee`:

```bash
python3 experiments/replay/vllm_tool_baseline/run_experiment.py ... \
  2>&1 | tee outputs/infercept_swap/run.log
```

## Outputs

Each run writes:

- `events.jsonl`: timeline events such as request emit, GPU submit/end, tool enqueue/start/end, and request completion.
- `traces.jsonl`: one complete trace per request.
- `summary.json`: aggregate latency, throughput, and completion statistics.
- `kv_events.jsonl`: optional vLLM KV-cache events when `--kv-events` is enabled.

Run outputs are ignored by Git. The clean tree intentionally omits analysis and visualization helpers.
