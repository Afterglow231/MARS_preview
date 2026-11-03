# Codex and OpenCode Integration

This directory serves the native [Codex CLI](https://github.com/openai/codex)
and [OpenCode](https://github.com/anomalyco/opencode) agents from a local
MARS/vLLM server. Both frontends share the outer workload controller used by
the OpenHands integration, while the native agent framework owns every inner
decision:

- MARS/vLLM is only the model server and scheduler.
- Codex owns its thread, turns, model/tool loop, local tools, verification turn,
  interruption, and terminal state through `codex app-server`.
- OpenCode owns its root/child sessions, model/tool loop, permissions, local
  tools, verification prompt, retry state, and terminal state through
  `opencode --pure serve` plus its authenticated HTTP/SSE API.

There is no predeclared agent graph and no reconstruction from
`codex exec --json` or `opencode run --format json`. The graph is projected
from native turn/session, message, step, and tool-call IDs as it unfolds, and
MARS admits each root workflow (including any child sessions it spawns) as one
unit.

## Architecture

The maintained OpenHands workload harness still owns fixed-RPS emission, MARS
admission/window control, progress accounting, global `events.jsonl`,
`traces.jsonl`, and `summary.json`. It invokes a fresh request worker process
for each Codex/OpenCode task, just like the OpenHands request backend.

Within one request:

1. A request-local transparent proxy forwards model HTTP traffic to MARS/vLLM.
   It injects only benchmark controls (`job_id`, token limit, temperature, and
   optional `top_p`) and records physical backend attempts. It never stores
   conversation history, executes tools, or chooses a terminal state.
2. The native frontend runs in a request-local bubblewrap filesystem and PID
   namespace with isolated HOME/XDG/Codex state. A command such as
   `pkill -f conda` can affect only that namespace, not the runner or vLLM.
3. Native frontend events define logical LLM rounds and tool spans in real
   time. HTTP retries remain child `backend_attempts`; they do not become fake
   agent rounds.
4. The frontend's native completed/error/timeout/interrupt event freezes the
   latency boundary. Reconciliation, server shutdown, Docker cleanup, and
   process teardown are excluded. OpenHands uses the same terminal boundary.

Codex speaks the Responses API. For Qwen/non-Harmony models, the custom vLLM
entrypoint installs a strict, server-side, single-inference Responses-to-Chat
adapter. vLLM's Qwen chat template and tool parser remain authoritative. The
adapter retains no session state and supports function tools only; unsupported
Responses built-ins fail with an explicit 400. OpenCode talks directly to Chat
Completions. Thus protocol translation is a model-server concern, not a second
agent loop in front of Codex.

The pinned Codex client also replays assistant output items without two
response-only serialization fields required by vLLM's pinned OpenAI Pydantic
types. A stateless ASGI middleware at `/v1/responses` supplies only those
missing fields before route validation; it does not retain, infer, or reorder
history. The real-GPU gate requires every resulting backend attempt to finish
with HTTP 200, so a later-round schema mismatch cannot be reported as a
successful agent request.

TerminalBench never receives the host Docker socket. Its request-local `docker`
shim reaches a scoped proxy restricted to that request's project and container.
The whole request also has a parent process-group timeout, so hung descendants
are terminated without touching other experiments.

## Installation

The pinned frontend versions live in [`upstream.env`](upstream.env):
Codex `0.149.1`, OpenCode `1.18.23`, Node `24.18.1`, and npm `11.16.0`.
Upstream sources and npm runtimes are installed under
`${MARS_AGENT_BASELINES_ROOT:-$HOME/external/mars_agent_baselines}`, never
inside this repository.

System requirements in addition to the `mars` conda environment: a working
[bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`), `flock`,
`taskset`, and Docker for TerminalBench tasks. On Ubuntu:

```bash
sudo apt-get install -y bubblewrap util-linux
```

Install the pinned Node runtime, then run the idempotent setup. It installs
both frontends, keeps commit-pinned source checkouts for provenance, and runs
the CPU-side integration gates:

```bash
export MARS_AGENT_BASELINES_ROOT="$HOME/external/mars_agent_baselines"
mkdir -p "$MARS_AGENT_BASELINES_ROOT/toolchain"
curl -fL https://nodejs.org/dist/v24.18.1/node-v24.18.1-linux-x64.tar.xz \
  | tar -xJ -C "$MARS_AGENT_BASELINES_ROOT/toolchain"
export MARS_NODE_BIN="$MARS_AGENT_BASELINES_ROOT/toolchain/node-v24.18.1-linux-x64/bin/node"
export PATH="$(dirname "$MARS_NODE_BIN"):$PATH"
npm install --global npm@11.16.0

conda activate mars
cd /path/to/MARS
PYTHON_BIN="$(command -v python)" \
  bash experiments/scripts/setup_agent_cli.sh
```

If bubblewrap is not installed system-wide, point `MARS_BWRAP_BIN` at a working
user-local binary. `PYTEST_PYTHON` (default: `PYTHON_BIN`) must provide pytest
and this checkout's vLLM dependencies, because setup also runs the Qwen
Responses compatibility tests.

The gates cover real pinned-binary handshakes, same-session verification,
OpenCode SSE retry/idle semantics, logical-round versus backend-attempt
separation, live long-running tool spans, parallel/root-child tools,
whole-request descendant termination, and hostile `pkill`/outside-workspace
writes.

## Usage

### 1. GPU smoke test

Run one tool-using request per frontend through a real Qwen3-Coder/vLLM server
before larger runs. The validator requires native terminal success, a real
session ID, at least two logical model rounds, one completed tool call, and
HTTP 200 for every physical backend attempt:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=$HOME/models/Qwen3-Coder-30B-A3B-Instruct \
AGENTS=codex,opencode \
  bash experiments/scripts/run_agent_cli_smoke.sh
```

### 2. Workload runs

`run_agent_cli_local.sh` starts one MARS/vLLM server and runs the selected
frontend against the task stream. The workload references SWE-bench
repositories by commit; prepare them once before the first run, then list any
benchmark inputs that are still missing locally:

```bash
python experiments/scripts/prepare_agent_cli_swebench_caches.py \
  --tasks $PWD/benchmark/hybrid_workloads/test_workload.jsonl
python experiments/scripts/validate_agent_cli_workload.py \
  --tasks $PWD/benchmark/hybrid_workloads/test_workload.jsonl \
  --benchmark-root $PWD/benchmark
```

This preview ships only part of the benchmark task assets referenced by
`test_workload.jsonl` (for example, most TerminalBench tasks and GitTaskBench
code bases are not bundled). Requests with missing inputs still exercise the
serving path, but they are not meaningful task-quality results.

Then run Codex (or `AGENTS=opencode`) with a server policy:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=$HOME/models/Qwen3-Coder-30B-A3B-Instruct \
TASKS_PATH=$PWD/benchmark/hybrid_workloads/test_workload.jsonl \
AGENTS=codex MAX_REQUESTS=20 RPS=0.2 SCHEDULING_POLICY=mars \
  bash experiments/scripts/run_agent_cli_local.sh
```

`SCHEDULING_POLICY` selects the engine policy (`mars`, `fcfs`, `continuum`,
`continuum_dy`, `autellix`, `infercept`, or `lightllm`). For the
Continuum + CONCUR baselines, keep the engine policy and add CONCUR's
client-side admission controller with `WORKLOAD_SCHEDULING_POLICY`:

```bash
SCHEDULING_POLICY=continuum_dy WORKLOAD_SCHEDULING_POLICY=continuum_dy_con \
  bash experiments/scripts/run_agent_cli_local.sh
```

Useful controls:

- `DRY_RUN=1` prints the server and agent commands without starting anything.
- `USE_EXISTING_VLLM=1 BASE_URL=http://host:port/v1` reuses a running server.
- `CPU_SERVER`/`CPU_AGENT` (defaults `0-1`/`2-9`) pin the server and the agent
  sandboxes to disjoint cores; adjust them to your CPU layout.
- `GPU_MEMORY_UTILIZATION` defaults to `0.95`, which retains the 262,144-token
  context for the BF16 model on a 96 GB H100.
- A per-user `flock` and a busy-GPU check prevent two runs from sharing a GPU;
  set `ALLOW_CONCURRENT_RUNS=1` only on purpose.
- Formal runs are immutable: use a new `RUN_DIR`/`TIMESTAMP` after an
  interruption rather than mixing two server lifetimes with `RESUME=1`.

A native terminal `completed` status means the frontend finished its loop;
benchmark answer correctness still requires a separate evaluator.

## Output layout and metric meaning

Results are written to `outputs/agent_cli/<policy>/<run_label>/<timestamp>/`,
the same hierarchy as OpenHands. Every run contains:

- `summary.json`: OpenHands-shaped latency, queue, logical-GPU, and tool
  distributions. Every distribution includes `sample_count`, `unknown_count`,
  and `coverage`; an unobserved duration is never silently replaced by zero.
- `progress.json`: emitted, terminal, inflight, and termination counts.
- `traces.jsonl`: one detailed record per request, including logical
  `llm_calls`, separate `backend_attempts`, native tools, root/child messages,
  terminal truth, and the dynamic graph.
- `events.jsonl`: shared scheduler/admission events plus compatible
  `gpu_submit`, `gpu_first_token`, `gpu_end`, tool, and `request_done` events.
- `dynamicity_summary.json`: round-depth and topology diversity aggregates.
- `_request_runs/<id>/result.json`: the normalized per-request result.
- `_request_runs/<id>/native_events.jsonl`: append-only raw native lifecycle.
- `_request_runs/<id>/lifecycle_events.jsonl`: normalized logical lifecycle.
- `_request_runs/<id>/backend_attempts.jsonl`: physical vLLM HTTP attempts.
- `_request_runs/<id>/live_events.jsonl`: flushed tool start/end events used by
  the shared CPU-pressure controller.
- Codex requests additionally retain `codex_app_server.raw.jsonl`,
  `codex_app_server.wire.jsonl`, and app-server stderr.
- OpenCode requests additionally retain server stdout/stderr and authoritative
  reconciled session messages inside `result.json`.

`gpu_total_s` is the sum of complete logical model-round envelopes, matching
the OpenHands LLM-call definition. `backend_attempt_total_s` is separate and
may contain multiple attempts for one logical round. Tool queue delay remains
`null` when the frontend did not expose a usable timing boundary. For Codex
in-process tools such as `update_plan`, the native function-call/output pair is
the complete exposed execution envelope; enqueue is explicitly labeled as the
assumed start because Codex publishes no separate queue/start event. A native
turn cannot publish success while any tool remains queued or running.

For the dynamic-DAG claim, report task success alongside `round_count`,
`unique_round_counts`, `unique_topology_count`, `topology_entropy_bits`, and the
topology-frequency distribution. Graph diversity alone is not a quality
metric.
