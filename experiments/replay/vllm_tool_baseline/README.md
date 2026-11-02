# vLLM + Serverless Tool Baseline

This runner executes `mas_task_v1` workloads with an in-process vLLM engine and a CPU tool pool. Each request follows a serial GPU/tool pattern and writes timeline events, per-request traces, and a summary.

Run from the repository root:

```bash
cd /path/to/MARS
export MODEL_PATH=${MODEL_PATH:-$HOME/models/Qwen3-Coder-30B-A3B-Instruct}
```

## Quick Run

```bash
CUDA_VISIBLE_DEVICES=0 \
python3 experiments/replay/vllm_tool_baseline/run_experiment.py \
  --tasks benchmark/hybrid_workloads/test_workload.jsonl \
  --model-path "$MODEL_PATH" \
  --scheduling-policy mars \
  --rps 0.2 --num-requests 10 \
  --output-dir outputs/testbed_smoke
```

The wrapper script provides the same path-relative defaults:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH="$MODEL_PATH" \
TASKS_PATH=$PWD/benchmark/hybrid_workloads/test_workload.jsonl \
NUM_REQUESTS=10 RPS=0.2 SCHEDULING_POLICY=mars \
bash experiments/scripts/run_baseline_continuum.sh
```

Supported in-process policies include `fcfs`, `priority`, `continuum`,
`continuum_dy`, `autellix`, `infercept`, `infercept_swap`, `lightllm`, and
`mars`.

## LightLLM Policy

Use `lightllm` to compare the ported LightLLM scheduling policy with MARS and
the other policies while retaining the same vLLM execution engine and tool
simulator:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH="$MODEL_PATH" \
TASKS_PATH=$PWD/benchmark/hybrid_workloads/test_workload.jsonl \
NUM_REQUESTS=400 RPS=0.2 SCHEDULING_POLICY=lightllm \
  bash experiments/scripts/run_baseline_continuum.sh
```

The port follows upstream LightLLM commit
[`5d59e4907dc9b5338426b56723bdee42d8af9309`](https://github.com/ModelTC/LightLLM/commit/5d59e4907dc9b5338426b56723bdee42d8af9309)
(Apache-2.0) for strict FCFS admission, peak KV-token estimation,
output-length EMA, prefix-aware last-token classification, and aggressive
prefill/decode packing. Unsupported parallel/data-plane modes fail fast rather
than silently changing the policy. It is not the native
LightLLM server: vLLM still supplies request execution, paged KV allocation,
preemption, caching, CUDA graphs, and kernels. Use the separate
[`lightllm_native` OpenHands path](../../baselines/lightllm/README.md) for a
full-system LightLLM comparison, and do not attribute results across the two
tracks as scheduler-only effects.

## InferCept + LMCache Swap

Use `infercept_swap` to let InferCept choose among evict, preserve-on-GPU, and
LMCache-backed swap for tool-call KV:

```bash
CUDA_VISIBLE_DEVICES=0 \
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
