# MARS: Efficient, Adaptive Co-Scheduling for Heterogeneous Agentic Systems

**MARS** is a serving-system prototype for heterogeneous agentic workloads. It
extends vLLM with scheduling policies for workloads that interleave GPU
inference with CPU-side tool execution, and provides an OpenHands-based
integration path for end-to-end agent evaluation.

This repository is a preview release for the
[MARS paper](https://arxiv.org/abs/2604.26963). It includes the code required
to run the experiments, the MARS-modified vLLM source code, the
OpenHands workload runner, and a compact hybrid workload.

Supported scheduling policies:

- `mars`: adaptive co-scheduling for heterogeneous agentic workloads.
- `fcfs`: first-come, first-served request scheduling.
- `continuum` and `continuum_dy`: baselines built from the official
  [vLLM-Continuum](https://github.com/Hanchenli/vllm-continuum) implementation;
  `continuum_dy` extends it with dynamically computed TTL values.
- `autellix`: an Autellix baseline implemented with the PLAS scheduling policy
  proposed in the [Autellix paper](https://arxiv.org/abs/2502.13965).
- `infercept`: an InferCept baseline implemented following the
  [InferCept paper](https://arxiv.org/abs/2402.01869)'s min-waste principle
  across tool-call interceptions.

---

## How It Works

### MARS-Modified vLLM

`vllm-continuum/` contains the local vLLM source code used by the experiments.
It adds the scheduler implementations and runtime hooks needed to compare
MARS with the bundled baselines.

### Agentic Workloads

This preview release includes a compact test workload at
`benchmark/hybrid_workloads/test_workload.jsonl`. This file is the default
workload consumed by the vLLM replay and OpenHands launch scripts.

### Evaluation Paths

The repository provides two execution paths:

- **vLLM replay**: runs the workload through an in-process vLLM engine with
  simulated tool-side execution.
- **OpenHands integration**: starts a local OpenAI-compatible vLLM server and
  runs OpenHands workers against it for end-to-end agent execution.

---

## Project Structure

```text
MARS/
|-- vllm-continuum/                  # MARS-modified vLLM source tree
|-- experiments/
|   |-- replay/vllm_tool_baseline/   # vLLM replay runtime
|   |-- openhands/                   # OpenHands integration and tests
|   `-- scripts/                     # Maintained launch scripts
|-- benchmark/
|   |-- hybrid_workloads/test_workload.jsonl # Test workload
|   |-- GitTaskBench/                # Referenced task assets
|   `-- terminal-bench/              # Referenced task assets
`-- LICENSE.txt
```

---

## Installation

**Requirements:** `conda`, `git`, `cmake`, `ninja`, `python>=3.12`,
`torch==2.8.0+cu128`; Datacenter-level NVIDIA GPU. Tested on H100 and H200.

```bash
# Clone the repository and enter the repo root.
git clone <repo-url> MARS
cd MARS

# Build the MARS-modified vLLM environment.
cd vllm-continuum
MODE=source bash scripts/create_continuum_vllm_conda_env.sh
conda activate mars
cd ..

# If CUDA auto-detection fails, rebuild with explicit CUDA settings:
# cd vllm-continuum
# MODE=source CUDA_VERSION_OVERRIDE=12.8 TORCH_CUDA_TAG=cu128 \
#   bash scripts/create_continuum_vllm_conda_env.sh
# conda activate mars
# cd ..

# Verify that Python imports vLLM from this repository and sees the local
# FlashAttention extension modules.
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/vllm-continuum" \
python - <<'PY'
import importlib
import vllm

print("vllm:", vllm.__file__)
for name in ("vllm.vllm_flash_attn._vllm_fa2_C", "vllm.vllm_flash_attn._vllm_fa3_C"):
    mod = importlib.import_module(name)
    print(name, mod.__file__)
PY

# Create the OpenHands worker environment.
# The vLLM service runs in conda env `mars`; OpenHands workers use this venv.
# OpenHands 1.14 requires Python >=3.12. These commands use python3.12 because
# python3.13 is not installed by default on many servers; override as needed.
OPENHANDS_VENV="${OPENHANDS_VENV:-$HOME/.venvs/openhands312}"
OPENHANDS_VENV_PYTHON="${OPENHANDS_VENV_PYTHON:-python3.12}"
"$OPENHANDS_VENV_PYTHON" -m venv "$OPENHANDS_VENV"
"$OPENHANDS_VENV/bin/python" -m pip install --upgrade pip
"$OPENHANDS_VENV/bin/python" -m pip install \
  openhands-sdk==1.14.0 \
  openhands-tools==1.14.0 \
  litellm==1.82.3
export OPENHANDS_PYTHON="$OPENHANDS_VENV/bin/python"

# Verify the OpenHands runtime.
OPENHANDS_SUPPRESS_BANNER=1 "$OPENHANDS_PYTHON" - <<'PY'
from openhands.sdk import Agent, Conversation, LLM, Tool
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool
print("OpenHands runtime OK")
PY
```

---

## Usage

Set `MODEL_PATH` to a local model directory before running GPU experiments. The
scripts default to `$HOME/models/Qwen3-Coder-30B-A3B-Instruct`.

### 1. vLLM Replay Experiment

Runs the test workload through the vllm replay baseline with the selected
scheduling policy:

```bash
conda activate mars
cd /path/to/MARS
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=$HOME/models/Qwen3-Coder-30B-A3B-Instruct \
TASKS_PATH=$PWD/benchmark/hybrid_workloads/test_workload.jsonl \
NUM_REQUESTS=100 RPS=0.2 SCHEDULING_POLICY=mars \
bash experiments/scripts/run_baseline_continuum.sh
```

### 2. OpenHands Integration Experiment

Starts a local vLLM OpenAI-compatible server and runs the OpenHands workload
runner:

```bash
conda activate mars
cd /path/to/MARS
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=$HOME/models/Qwen3-Coder-30B-A3B-Instruct \
TASKS_PATH=$PWD/benchmark/hybrid_workloads/test_workload.jsonl \
MAX_REQUESTS=20 RPS=0.2 \
bash experiments/scripts/run_openhands_qwen3_local.sh
```


### 3. Scheduling Policy Smoke Test

Runs all the scheduling policies on one GPU. By default, the smoke test uses the
first 200 requests for the vLLM replay target and the first 50 requests for the
OpenHands target.

```bash
conda activate mars
cd /path/to/MARS
bash experiments/scripts/run_policy_smoke_gpu1.sh
```


---

## End-to-End Walkthrough

The following commands run a compact verification pipeline on the bundled
workload:

```bash
conda activate mars
cd /path/to/MARS
export MODEL_PATH=$HOME/models/Qwen3-Coder-30B-A3B-Instruct

# Step 1: run a small vLLM replay job
CUDA_VISIBLE_DEVICES=0 NUM_REQUESTS=100 RPS=0.2 SCHEDULING_POLICY=mars \
  bash experiments/scripts/run_baseline_continuum.sh

# Step 2: run a small OpenHands job
CUDA_VISIBLE_DEVICES=0 MAX_REQUESTS=20 RPS=0.2 SCHEDULING_POLICY=mars \
  bash experiments/scripts/run_openhands_qwen3_local.sh

# Step 3: compare selected policies on GPU 0
GPU=0 POLICIES="fcfs mars" TARGETS="vllm openhands" \
  VLLM_REQUESTS=200 OPENHANDS_REQUESTS=50 \
  bash experiments/scripts/run_policy_smoke_gpu1.sh
```

---

## Reproducibility Notes

- All launch scripts are path-relative to the repository root.
- Override `MODEL_PATH`, `TASKS_PATH`, `PYTHON_BIN`, `MARS_PYTHON`,
  `OPENHANDS_PYTHON`, and output variables as needed for your machine.
- CPU binding defaults are `CPU_GPU=0-1`, `CPU_TOOL=16-23`, and
  `CPU_CORES=0-1,16-23`; override them if your node has a different CPU layout.

---

## Troubleshooting

`ValueError: Unsupported FA version: None`

: The local `vllm-continuum` tree was not built correctly, or Python is
  importing a source tree without `_vllm_fa2_C.abi3.so` and
  `_vllm_fa3_C.abi3.so`. Rebuild `vllm-continuum` in the active `mars`
  environment and rerun the installation verification.

`OPENHANDS_PYTHON not found`

: Create the OpenHands environment above or pass
  `OPENHANDS_PYTHON=/path/to/openhands/python`.

`Model path not found`

: Download or mount the model locally and set `MODEL_PATH=/path/to/model`.

---

## Coming Soon

- **Full experiment reproduction** — end-to-end scripts to reproduce the
  complete experiment results reported in the MARS paper.
- **Analysis and visualization utilities** — scripts for result analysis,
  plotting, figure generation, and workload construction.

---

## Acknowledgements

This repository builds on the following open-source projects:

- [vLLM](https://github.com/vllm-project/vllm)
- [vLLM-continuum](https://github.com/Hanchenli/vllm-continuum)
- [OpenHands](https://github.com/All-Hands-AI/OpenHands)
- [GitTaskBench](https://github.com/git-task-bench/GitTaskBench)
- [Terminal-Bench](https://github.com/laude-institute/terminal-bench)

We thank the authors and contributors of these projects for their valuable work.
