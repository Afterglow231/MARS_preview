#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mars}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

MODE="${MODE:-source}" # source | precompiled (experimental)
CUDA_VERSION_OVERRIDE="${CUDA_VERSION_OVERRIDE:-}" # e.g. 12.4 / 12.1 / 11.8
TORCH_CUDA_TAG_OVERRIDE="${TORCH_CUDA_TAG:-}" # e.g. cu128 (recommended for this repo)

usage() {
  cat <<'EOF'
Create a conda env `mars` for this repo, selecting the right CUDA wheel set.

Environment variables:
  ENV_NAME=mars                      conda env name
  PYTHON_VERSION=3.10                python version tested with this vLLM fork
  MODE=source|precompiled            source: builds vLLM C++/CUDA extensions locally (recommended)
  CUDA_VERSION_OVERRIDE=12.4|12.1... override autodetected CUDA (driver/toolkit)
  TORCH_CUDA_TAG=cu128               override PyTorch CUDA wheel tag (default: inferred; this repo expects cu128)

Examples:
  MODE=source bash scripts/create_continuum_vllm_conda_env.sh
  MODE=source CUDA_VERSION_OVERRIDE=12.7 bash scripts/create_continuum_vllm_conda_env.sh
  TORCH_CUDA_TAG=cu128 MODE=source bash scripts/create_continuum_vllm_conda_env.sh

Notes:
  - This script needs internet access (conda/pip downloads, and precompiled wheels).
  - For MODE=source, this script installs a conda CUDA toolkit (nvcc) and builds vLLM from source.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd conda

python_cpython_spec() {
  local ver="$1"
  # Pin to CPython builds to avoid accidentally solving to GraalPy/PyPy, which
  # breaks PyTorch wheels.
  if [[ "${ver}" == *.*.* ]]; then
    echo "python=${ver}=*cpython*"
  else
    echo "python=${ver}.*=*cpython*"
  fi
}

PYTHON_SPEC="$(python_cpython_spec "${PYTHON_VERSION}")"

detect_cuda_version() {
  if [[ -n "${CUDA_VERSION_OVERRIDE}" ]]; then
    echo "${CUDA_VERSION_OVERRIDE}"
    return 0
  fi

  # Prefer nvidia-smi (reports max CUDA supported by driver)
  if command -v nvidia-smi >/dev/null 2>&1; then
    local smi
    smi="$(nvidia-smi 2>/dev/null || true)"
    # Example: "CUDA Version: 12.4"
    local ver
    ver="$(printf "%s" "${smi}" | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' | head -n 1 | awk '{print $3}' || true)"
    if [[ -n "${ver}" ]]; then
      echo "${ver}"
      return 0
    fi
  fi

  # Fallback to nvcc (reports toolkit version)
  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_out
    nvcc_out="$(nvcc --version 2>/dev/null || true)"
    # Example: "release 12.4,"
    local ver
    ver="$(printf "%s" "${nvcc_out}" | grep -Eo 'release [0-9]+\.[0-9]+' | head -n 1 | awk '{print $2}' || true)"
    if [[ -n "${ver}" ]]; then
      echo "${ver}"
      return 0
    fi
  fi

  # Fallback to /usr/local/cuda/version.txt
  if [[ -f /usr/local/cuda/version.txt ]]; then
    local ver
    ver="$(grep -Eo '[0-9]+\.[0-9]+' /usr/local/cuda/version.txt | head -n 1 || true)"
    if [[ -n "${ver}" ]]; then
      echo "${ver}"
      return 0
    fi
  fi

  echo ""
}

cuda_to_torch_tag() {
  # Map driver/toolkit CUDA version -> PyTorch wheel tag.
  # This repo pins torch==2.8.0 which is distributed as cu128 wheels.
  local cuda_ver="$1"
  local major="${cuda_ver%%.*}"
  local minor="${cuda_ver#*.}"
  minor="${minor%%.*}"

  if [[ -z "${cuda_ver}" || -z "${major}" || -z "${minor}" ]]; then
    echo ""
    return 0
  fi

  if (( major > 12 )) || (( major == 12 && minor >= 7 )); then
    echo "cu128"
    return 0
  fi

  echo ""
}

CUDA_VER="$(detect_cuda_version)"
if [[ -z "${CUDA_VER}" ]]; then
  echo "Could not detect CUDA version. Set CUDA_VERSION_OVERRIDE (e.g. 12.4) and retry." >&2
  exit 1
fi

TORCH_CUDA_TAG="$(cuda_to_torch_tag "${CUDA_VER}")"
# Allow explicit override from env var TORCH_CUDA_TAG.
if [[ -n "${TORCH_CUDA_TAG_OVERRIDE}" ]]; then
  TORCH_CUDA_TAG="${TORCH_CUDA_TAG_OVERRIDE}"
fi

if [[ -z "${TORCH_CUDA_TAG}" ]]; then
  echo "CUDA '${CUDA_VER}' does not map to a supported PyTorch wheel tag for this repo." >&2
  echo "This repo pins torch==2.8.0 (cu128). Please use a node/driver that supports CUDA >= 12.7, or set TORCH_CUDA_TAG=cu128 to try anyway." >&2
  exit 1
fi

if [[ "${MODE}" == "precompiled" ]]; then
  echo "WARNING: MODE=precompiled is experimental for this fork and may crash due to ABI mismatch." >&2
elif [[ "${MODE}" != "source" ]]; then
  echo "Unknown MODE='${MODE}'. Use MODE=precompiled or MODE=source." >&2
  exit 1
fi

echo "[1/5] Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION})..."
conda create -n "${ENV_NAME}" -y -c conda-forge -c defaults "${PYTHON_SPEC}" pip

echo "[2/5] Installing base build tools (cmake/ninja/git)..."
conda install -n "${ENV_NAME}" -y -c conda-forge cmake ninja git pkg-config "${PYTHON_SPEC}"

# Conda's activate/deactivate hooks are not compatible with `set -u` (nounset):
# they may reference unset variables like CONDA_BACKUP_CXX.
NOUNSET_WAS_SET=0
if [[ "${-}" == *u* ]]; then
  NOUNSET_WAS_SET=1
  set +u
fi
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
if [[ "${NOUNSET_WAS_SET}" -eq 1 ]]; then
  set -u
fi

if ! python -c 'import platform,sys; sys.exit(0 if platform.python_implementation()=="CPython" else 1)'; then
  echo "ERROR: conda solved '${ENV_NAME}' to a non-CPython runtime (e.g. GraalPy/PyPy), which cannot run PyTorch wheels." >&2
  echo "Fix: conda install -n '${ENV_NAME}' -c conda-forge '${PYTHON_SPEC}'" >&2
  exit 1
fi

echo "[3/5] Installing CUDA PyTorch stack via pip (${TORCH_CUDA_TAG})..."
python -m pip install --upgrade pip
python -m pip install --isolated --no-cache-dir \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}" \
  "torch==2.8.0+${TORCH_CUDA_TAG}" \
  "torchvision==0.23.0+${TORCH_CUDA_TAG}" \
  "torchaudio==2.8.0+${TORCH_CUDA_TAG}"

# Install the rest of the CUDA requirements (from PyPI).
python -m pip install --isolated --no-cache-dir \
  --index-url "https://pypi.org/simple" \
  --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}" \
  -r requirements/cuda.txt

echo "[4/5] Installing repo + experiment deps..."
python -m pip install lmcache==0.4.4 hf_transfer requests pytest
if [[ -d mini-swe-agent ]]; then
  python -m pip install -e mini-swe-agent
  python -m pip install datasets
else
  echo "mini-swe-agent/ is not present in this clean tree; skipping SWE-bench helper install."
fi

if [[ "${MODE}" == "precompiled" ]]; then
  echo "[5/5] Installing this repo (editable) with precompiled vLLM extensions..."
  VLLM_USE_PRECOMPILED=1 python -m pip install -e . --no-build-isolation --no-deps
else
  echo "[5/5] Installing CUDA toolkit (nvcc) via conda and building vLLM from source (this can take a while)..."
  # `conda install` may trigger an automatic `conda activate --reactivate`,
  # which sources deactivate scripts that are not nounset-safe.
  if [[ "${-}" == *u* ]]; then
    set +u
    conda install -y -c conda-forge cuda-toolkit=12.8.1 "${PYTHON_SPEC}"
    set -u
  else
    conda install -y -c conda-forge cuda-toolkit=12.8.1 "${PYTHON_SPEC}"
  fi

  # conda-forge CUDA puts headers/libs under targets/<arch>-linux.
  # Help CMake/Torch find the toolkit root consistently across nodes.
  CUDA_TARGETS_DIR=""
  case "$(uname -m)" in
    x86_64) CUDA_TARGETS_DIR="${CONDA_PREFIX}/targets/x86_64-linux" ;;
    aarch64) CUDA_TARGETS_DIR="${CONDA_PREFIX}/targets/sbsa-linux" ;;
    ppc64le) CUDA_TARGETS_DIR="${CONDA_PREFIX}/targets/ppc64le-linux" ;;
  esac
  if [[ -n "${CUDA_TARGETS_DIR}" && -d "${CUDA_TARGETS_DIR}" ]]; then
    export CUDAToolkit_ROOT="${CUDA_TARGETS_DIR}"
    export CUDA_TOOLKIT_ROOT_DIR="${CUDA_TARGETS_DIR}"
    echo "Using CUDA toolkit root: ${CUDA_TARGETS_DIR}"
  fi

  python -m pip install -e . --no-build-isolation
fi

echo ""
echo "Done."
echo "Activate:  conda activate ${ENV_NAME}"
echo "Serve:     MODEL=<MODEL_NAME> TP=<NUM_GPUS> bash scripts/start_vllm_service.sh"
echo "Verify:    python -c 'import importlib; importlib.import_module(\"vllm.vllm_flash_attn._vllm_fa2_C\"); importlib.import_module(\"vllm.vllm_flash_attn._vllm_fa3_C\")'"
