# LightLLM Baselines

MARS exposes two deliberately separate LightLLM baselines. They answer
different experimental questions and must not be reported as if they were the
same system.

| Name | Execution path | Intended comparison | What is held constant |
| --- | --- | --- | --- |
| `lightllm_native` | Unmodified upstream LightLLM server plus the shared OpenHands client | End-to-end OpenHands performance | Workload, model, sampling, hardware allocation, warmup, and client |
| `--scheduling-policy lightllm` | Ported LightLLM scheduling policy inside the MARS-vLLM engine | Scheduler comparison in `vllm+tool` replay | vLLM execution engine, KV-cache implementation, kernels, tool simulator, and workload |

The upstream source is [ModelTC/LightLLM](https://github.com/ModelTC/LightLLM),
licensed under
[Apache-2.0](https://github.com/ModelTC/LightLLM/blob/5d59e4907dc9b5338426b56723bdee42d8af9309/LICENSE).
Both baselines are derived from the full upstream commit
[`5d59e4907dc9b5338426b56723bdee42d8af9309`](https://github.com/ModelTC/LightLLM/commit/5d59e4907dc9b5338426b56723bdee42d8af9309).
The immutable identities are recorded in `upstream.env`:

```text
commit: 5d59e4907dc9b5338426b56723bdee42d8af9309
linux/amd64 image: ghcr.io/modeltc/lightllm@sha256:a94be53d781f2b022a98953f9177df18174bb20d9da00169fbe1050aee3300f5
OCI index: sha256:aaff499c8e79fab060fa61b4f49e0762c609a70bfe9f227a7e9c66c4c1ecf42b
```

## 1. Native LightLLM with OpenHands

`lightllm_native` keeps LightLLM as a complete, external serving system. MARS
does not vendor or patch its source, replace its scheduler, or synthesize
vLLM-private telemetry. The integration layer only starts the pinned server,
checks OpenAI streaming/tool-call compatibility, records provenance, and sends
the same OpenHands workload used for the MARS-vLLM runs.
`lightllm_native` is an experiment/backend label, not a vLLM scheduling-policy
value.

The Qwen3-Coder workload is text-only. The shared OpenHands client fixes the
SDK's public `force_string_serializer=True` control for every compared backend:
text parts are joined in order with newlines, while native `tool_calls`,
`tool_call_id`, and tool names remain unchanged. This is necessary because the
pinned LightLLM Qwen template accepts OpenAI string content but not a list of
text content parts; vLLM already performs the same newline join for this
string-format template. This is a common client representation choice, not a
LightLLM patch. It must not be reused for a multimodal workload, because the
string serializer intentionally retains only text content.

### Setup

The official OCI image is the reference runtime. It requires Docker access,
the NVIDIA container runtime, and a `linux/amd64` host:

```bash
cd /path/to/MARS
bash experiments/scripts/setup_lightllm_baseline.sh --runtime docker
bash experiments/scripts/setup_lightllm_baseline.sh --runtime docker --check
```

The setup command pulls by immutable digest and verifies that the image's OCI
revision label equals the pinned commit. Inspect the exact launch command
without starting a server with:

```bash
bash experiments/scripts/run_openhands_lightllm.sh --dry-run
```

If Docker is unavailable, an external source checkout is supported as a
fallback:

```bash
bash experiments/scripts/setup_lightllm_baseline.sh --runtime source
LIGHTLLM_RUNTIME=source \
  bash experiments/scripts/run_openhands_lightllm.sh
```

The source and conda environment are installed under
`${MARS_EXTERNAL_BASELINES_ROOT:-$HOME/.cache/mars/baselines}`, never inside
this repository. The Docker image remains the reference for reported results:
the source fallback resolves Python dependencies at setup time and therefore
is not guaranteed to reproduce the image's complete package stack.

### Run a matched OpenHands comparison

Run the two servers sequentially on the same idle GPU. The following example
uses explicit common controls and disables vLLM KV-event instrumentation,
which has no native LightLLM equivalent:

```bash
cd /path/to/MARS
export MODEL_PATH=$HOME/models/Qwen3-Coder-30B-A3B-Instruct
export MODEL_NAME=Qwen3-Coder-30B-A3B-Instruct
export TASKS_PATH=$PWD/benchmark/hybrid_workloads/test_workload.jsonl
export CUDA_VISIBLE_DEVICES=0 TP=1
export CPU_GPU=0-1 CPU_TOOL=16-23 CPU_CORES=0-1,16-23
export DTYPE=bfloat16 GPU_MEMORY_UTILIZATION=0.90 MAX_MODEL_LEN=262144
export MAX_REQUESTS=400 RPS=0.2 MAX_WORKERS=0 SHUFFLE=0 SEED=0
export TEMPERATURE=0.0 TOP_P=1.0 MAX_OUTPUT_TOKENS=2048
export LLM_TIMEOUT_S=3000 LLM_NUM_RETRIES=3 MAX_ITERATIONS=40
export WARMUP_REQUESTS=1 WARMUP_MAX_TOKENS=128
export KV_EVENTS=0
RUN_ID=$(date +%Y%m%d_%H%M%S)

SCHEDULING_POLICY=mars OPENAI_COMPAT_WARMUP=1 \
OUT_DIR="$PWD/outputs/openhands/mars/matched/$RUN_ID" \
  bash experiments/scripts/run_openhands_qwen3_local.sh

LIGHTLLM_RUNTIME=docker WARMUP_ENABLED=1 \
OUT_DIR="$PWD/outputs/openhands/lightllm_native/matched/$RUN_ID" \
  bash experiments/scripts/run_openhands_lightllm.sh
```

Do not run the two measurements concurrently. Keep the GPU model and count,
tensor-parallel size, CPU affinity, model files, task file and ordering,
sampling limits, OpenHands environment, warmup, and server/client timeouts
identical. Do not introduce backend-specific tuning flags for only one run.
The launcher keeps LightLLM's official
`LIGHTLLM_TRITON_AUTOTUNE_LEVEL=0` default. Setting it to `1` triggers a
hardware-specific tuning pass and therefore defines a different experiment;
pre-tune consistently, exclude that startup phase, and retain the recorded
override in `provenance.json` if you intentionally use it.
The native launcher requires the upstream `qwen3_coder` tool parser for this
Qwen3-Coder workload and rejects a modified source checkout.

Each output directory contains the normal OpenHands `run_config.json`,
`events.jsonl`, `traces.jsonl`, and `summary.json`, plus `provenance.json` and
warmup/server logs. Provenance defaults to a full content hash of the model
weights; `MODEL_HASH_MODE=metadata` is a faster diagnostic mode and is not
strong enough to establish model identity.

## 2. LightLLM Policy in vLLM + Tool Replay

The `lightllm` scheduling policy ports the pinned upstream policy's strict
FCFS/head-of-line admission, peak KV-token estimate, output-length EMA,
pre-lookup batch admission, post-prefix last-token decode classification, and
aggressive prefill/decode packing into the MARS-modified vLLM scheduler. Run it
through the same in-process replay path as the other scheduler baselines:

```bash
cd /path/to/MARS
conda activate mars
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=$HOME/models/Qwen3-Coder-30B-A3B-Instruct \
TASKS_PATH=$PWD/benchmark/hybrid_workloads/test_workload.jsonl \
NUM_REQUESTS=400 RPS=0.2 SCHEDULING_POLICY=lightllm \
  bash experiments/scripts/run_baseline_continuum.sh
```

For a scheduler comparison, repeat the command with `SCHEDULING_POLICY=mars`
or another supported in-process policy while keeping every other variable and
the GPU state unchanged.

This track is a scheduler-isolation port, not native LightLLM. Request
execution, token batching, paged KV allocation, prefix caching, preemption,
CUDA graphs, kernels, and model workers remain vLLM's implementations. Where
native LightLLM couples scheduling to its router, shared-memory request state,
radix cache, token allocator, and pause/recovery path, the port uses the
closest scheduler-level behavior available without changing vLLM's data plane.
Unsupported execution modes (DP/PP/DCP, speculative or async scheduling,
KV connectors, LoRA, multimodal/encoder-decoder models, and nonstandard vLLM
prefill caps) fail fast instead of silently changing the named policy.

## Interpretation Boundary

- `lightllm_native` versus MARS-vLLM on OpenHands is a **full-system** result.
  Differences may come from scheduling, kernels, cache management, batching,
  runtime dependencies, or any other engine component.
- `lightllm` versus `mars` in `vllm+tool` is a **policy-isolation** result on a
  common vLLM engine. It does not measure native LightLLM throughput.
- Do not compare `lightllm_native` numbers directly with `lightllm` replay
  numbers. They use different execution engines and different workload paths
  (real OpenHands execution versus simulated tool replay).
- Do not attribute the native full-system gap solely to LightLLM's scheduler,
  and do not present the policy port as the performance of the upstream
  LightLLM server.
