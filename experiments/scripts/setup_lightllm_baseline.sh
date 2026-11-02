#!/usr/bin/env bash
set -euo pipefail

# Prepare an unmodified, commit-pinned LightLLM runtime outside this repo.
# The official OCI image is the reference runtime.  A source environment is
# available for machines without Docker access, but is intentionally external.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCK_FILE="${MARS_DIR}/experiments/baselines/lightllm/upstream.env"
source "${LOCK_FILE}"

RUNTIME="${LIGHTLLM_RUNTIME:-docker}"
DRY_RUN=0
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage: setup_lightllm_baseline.sh [--runtime docker|source] [--check] [--dry-run]

The Docker runtime pulls the official image by immutable linux/amd64 digest.
The source runtime checks out the pinned commit and installs it into a dedicated
conda prefix below $MARS_EXTERNAL_BASELINES_ROOT (or ~/.cache/mars/baselines).

Environment overrides:
  LIGHTLLM_RUNTIME                 docker (reference) or source
  MARS_EXTERNAL_BASELINES_ROOT     external cache root
  LIGHTLLM_SOURCE_DIR              external source checkout (source mode)
  LIGHTLLM_ENV_DIR                 external conda prefix (source mode)
  CONDA_BIN                        conda executable (source mode)
  LIGHTLLM_DOCKER_BIN              docker executable (docker mode)
EOF
}

while (($#)); do
  case "$1" in
    --runtime)
      [[ $# -ge 2 ]] || { echo "--runtime requires a value" >&2; exit 2; }
      RUNTIME="$2"
      shift 2
      ;;
    --check)
      CHECK_ONLY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

die() {
  echo "$*" >&2
  exit 1
}

prepare_docker() {
  local docker_bin="${LIGHTLLM_DOCKER_BIN:-docker}"
  command -v "${docker_bin}" >/dev/null 2>&1 || die "Docker is required for LIGHTLLM_RUNTIME=docker."

  if [[ "${DRY_RUN}" == "1" ]]; then
    run "${docker_bin}" pull --platform "${LIGHTLLM_PLATFORM}" "${LIGHTLLM_IMAGE}"
    run "${docker_bin}" image inspect "${LIGHTLLM_IMAGE}"
    return 0
  fi

  if ! "${docker_bin}" info >/dev/null 2>&1; then
    die "Cannot access the Docker daemon. Add this user to the Docker group, or use --runtime source."
  fi

  if [[ "${CHECK_ONLY}" != "1" ]]; then
    run "${docker_bin}" pull --platform "${LIGHTLLM_PLATFORM}" "${LIGHTLLM_IMAGE}"
  fi

  local revision
  revision="$(
    "${docker_bin}" image inspect \
      --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' \
      "${LIGHTLLM_IMAGE}" 2>/dev/null || true
  )"
  [[ -n "${revision}" ]] || die "Pinned LightLLM image is not present; run without --check first."
  [[ "${revision}" == "${LIGHTLLM_COMMIT}" ]] || die \
    "OCI revision mismatch: expected ${LIGHTLLM_COMMIT}, image reports ${revision}"

  echo "LightLLM Docker runtime ready."
  echo "  image=${LIGHTLLM_IMAGE}"
  echo "  revision=${revision}"
}

prepare_source() {
  local external_root_default="${XDG_CACHE_HOME:-${HOME}/.cache}/mars/baselines"
  local external_root="${MARS_EXTERNAL_BASELINES_ROOT:-${external_root_default}}"
  local install_root="${external_root}/lightllm/${LIGHTLLM_COMMIT}"
  local source_dir="${LIGHTLLM_SOURCE_DIR:-${install_root}/source}"
  local env_dir="${LIGHTLLM_ENV_DIR:-${install_root}/env}"
  local conda_bin="${CONDA_BIN:-${HOME}/miniconda3/bin/conda}"
  local python_bin

  command -v realpath >/dev/null 2>&1 || die "realpath is required for source setup."
  # Resolve relative paths and symlinked parents before enforcing the external
  # checkout boundary.  A lexical check alone can be bypassed with `.` or a
  # symlink whose target is inside this repository.
  external_root="$(realpath -m -- "${external_root}")"
  install_root="$(realpath -m -- "${install_root}")"
  source_dir="$(realpath -m -- "${source_dir}")"
  env_dir="$(realpath -m -- "${env_dir}")"
  python_bin="${env_dir}/bin/python"

  case "${source_dir}" in
    "${MARS_DIR}"|"${MARS_DIR}"/*)
      die "LIGHTLLM_SOURCE_DIR must be outside the MARS repository: ${source_dir}"
      ;;
  esac
  case "${env_dir}" in
    "${MARS_DIR}"|"${MARS_DIR}"/*)
      die "LIGHTLLM_ENV_DIR must be outside the MARS repository: ${env_dir}"
      ;;
  esac

  command -v git >/dev/null 2>&1 || die "git is required for source setup."
  if [[ "${CHECK_ONLY}" != "1" ]]; then
    [[ -x "${conda_bin}" ]] || die "Conda executable not found: ${conda_bin}"
  fi

  if [[ "${CHECK_ONLY}" == "1" && "${DRY_RUN}" != "1" ]]; then
    [[ -d "${source_dir}/.git" ]] || die \
      "External LightLLM checkout is not installed: ${source_dir}"
    [[ -x "${python_bin}" ]] || die "LightLLM Python not found: ${python_bin}"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    run mkdir -p "${install_root}"
    run git clone --filter=blob:none --no-checkout "${LIGHTLLM_REPO_URL}" "${source_dir}"
    run git -C "${source_dir}" fetch --depth 1 origin "${LIGHTLLM_COMMIT}"
    run git -C "${source_dir}" checkout --detach "${LIGHTLLM_COMMIT}"
    run "${conda_bin}" create -y -p "${env_dir}" python=3.10
    run "${python_bin}" -m pip install --no-cache-dir -r "${source_dir}/requirements.txt"
    run "${python_bin}" -m pip install --no-cache-dir --no-deps -e "${source_dir}"
    return 0
  fi

  mkdir -p "${install_root}"
  if [[ ! -d "${source_dir}/.git" ]]; then
    [[ ! -e "${source_dir}" ]] || die "Non-git path already exists: ${source_dir}"
    run git clone --filter=blob:none --no-checkout "${LIGHTLLM_REPO_URL}" "${source_dir}"
    # A --no-checkout clone reports every tracked path as deleted. Populate
    # the pinned tree before applying the ordinary dirty-tree guard below.
    run git -C "${source_dir}" fetch --depth 1 origin "${LIGHTLLM_COMMIT}"
    run git -C "${source_dir}" checkout --detach "${LIGHTLLM_COMMIT}"
  fi

  local actual_commit dirty
  actual_commit="$(git -C "${source_dir}" rev-parse HEAD 2>/dev/null || true)"
  if [[ "${actual_commit}" != "${LIGHTLLM_COMMIT}" ]]; then
    dirty="$(git -C "${source_dir}" status --porcelain --untracked-files=all)"
    [[ -z "${dirty}" ]] || die "External LightLLM checkout is modified: ${source_dir}"
    [[ "${CHECK_ONLY}" != "1" ]] || die \
      "External checkout is at ${actual_commit:-unknown}, expected ${LIGHTLLM_COMMIT}"
    run git -C "${source_dir}" fetch --depth 1 origin "${LIGHTLLM_COMMIT}"
    run git -C "${source_dir}" checkout --detach "${LIGHTLLM_COMMIT}"
  fi

  actual_commit="$(git -C "${source_dir}" rev-parse HEAD)"
  [[ "${actual_commit}" == "${LIGHTLLM_COMMIT}" ]] || die "Failed to select pinned commit."
  dirty="$(git -C "${source_dir}" status --porcelain --untracked-files=all)"
  [[ -z "${dirty}" ]] || die "External LightLLM checkout is modified: ${source_dir}"

  if [[ "${CHECK_ONLY}" != "1" ]]; then
    if [[ ! -x "${python_bin}" ]]; then
      run "${conda_bin}" create -y -p "${env_dir}" python=3.10
    fi
    run "${python_bin}" -m pip install --no-cache-dir -r "${source_dir}/requirements.txt"
    run "${python_bin}" -m pip install --no-cache-dir --no-deps -e "${source_dir}"
  fi

  [[ -x "${python_bin}" ]] || die "LightLLM Python not found: ${python_bin}"
  [[ -x "${env_dir}/bin/hypercorn" ]] || die \
    "LightLLM HTTP entrypoint not found: ${env_dir}/bin/hypercorn"
  PYTHONDONTWRITEBYTECODE=1 "${python_bin}" - "${source_dir}" <<'PY'
from pathlib import Path
import sys
import lightllm

source_dir = Path(sys.argv[1]).resolve()
module_path = Path(lightllm.__file__).resolve()
if source_dir not in module_path.parents:
    raise SystemExit(f"lightllm imports from {module_path}, expected under {source_dir}")
print(f"lightllm={module_path}")
PY

  echo "LightLLM source runtime ready."
  echo "  source=${source_dir}"
  echo "  python=${python_bin}"
  echo "  commit=${actual_commit}"
}

case "${RUNTIME}" in
  docker) prepare_docker ;;
  source) prepare_source ;;
  *) die "Unsupported runtime: ${RUNTIME} (expected docker or source)" ;;
esac
