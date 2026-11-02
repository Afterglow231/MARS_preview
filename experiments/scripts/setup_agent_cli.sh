#!/usr/bin/env bash
set -euo pipefail

# Install exact OpenCode/Codex releases outside the MARS repository, retain
# commit-pinned source checkouts for provenance, and run the native lifecycle
# gates. The script is idempotent and refuses to replace dirty source checkouts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PINS_FILE="${MARS_DIR}/experiments/agent_cli/upstream.env"
source "${PINS_FILE}"

HOST_ROOT="${HOST_ROOT:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-${MARS_AGENT_BASELINES_ROOT:-${HOST_ROOT}/external/mars_agent_baselines}}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${EXTERNAL_ROOT}/runtime}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
PYTEST_PYTHON="${PYTEST_PYTHON:-${PYTHON_BIN}}"
MARS_BWRAP_BIN="${MARS_BWRAP_BIN:-$(command -v bwrap || true)}"
if [[ -z "${MARS_BWRAP_BIN}" && -x "${EXTERNAL_ROOT}/toolchain/bubblewrap/usr/bin/bwrap" ]]; then
  MARS_BWRAP_BIN="${EXTERNAL_ROOT}/toolchain/bubblewrap/usr/bin/bwrap"
fi
MARS_BWRAP_BIN="${MARS_BWRAP_BIN:-/usr/bin/bwrap}"
export MARS_BWRAP_BIN
DEFAULT_NODE_BIN="${EXTERNAL_ROOT}/toolchain/node-v${NODE_VERSION}-linux-x64/bin/node"
MARS_NODE_BIN="${MARS_NODE_BIN:-$(command -v node || true)}"
MARS_NODE_BIN="${MARS_NODE_BIN:-${DEFAULT_NODE_BIN}}"
export MARS_NODE_BIN
if [[ -x "${MARS_NODE_BIN}" ]]; then
  PATH="$(dirname "${MARS_NODE_BIN}"):${PATH}"
  export PATH
fi

die() {
  echo "$*" >&2
  exit 1
}

bool_true() {
  case "${1:-}" in
    1|true|TRUE|on|ON|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command is unavailable: $1"
}

ensure_checkout() {
  local name="$1"
  local repo_url="$2"
  local tag="$3"
  local expected_commit="$4"
  local dst="${EXTERNAL_ROOT}/${name}"

  if [[ -e "${dst}" && ! -d "${dst}/.git" ]]; then
    die "Refusing to replace non-git path: ${dst}"
  fi
  if [[ ! -d "${dst}/.git" ]]; then
    mkdir -p "${dst}"
    git -C "${dst}" init
    git -C "${dst}" remote add origin "${repo_url}"
  fi
  local actual_origin
  actual_origin="$(git -C "${dst}" remote get-url origin)"
  [[ "${actual_origin}" == "${repo_url}" ]] || die \
    "Unexpected ${name} origin: ${actual_origin}; expected ${repo_url}"
  if [[ -n "$(git -C "${dst}" status --porcelain --untracked-files=normal)" ]]; then
    die "Refusing to change dirty checkout: ${dst}"
  fi

  local current=""
  current="$(git -C "${dst}" rev-parse HEAD 2>/dev/null || true)"
  if [[ "${current}" != "${expected_commit}" ]]; then
    git -C "${dst}" fetch --depth=1 origin "refs/tags/${tag}:refs/tags/${tag}"
    local fetched
    fetched="$(git -C "${dst}" rev-list -n 1 "${tag}")"
    [[ "${fetched}" == "${expected_commit}" ]] || die \
      "${name} tag ${tag} resolved to ${fetched}; expected ${expected_commit}"
    git -C "${dst}" checkout --detach "${expected_commit}"
  fi
  [[ "$(git -C "${dst}" rev-parse HEAD)" == "${expected_commit}" ]] || die \
    "${name} checkout verification failed"
  echo "${name}: source=${dst} commit=${expected_commit}"
}

ensure_npm_runtime() {
  local name="$1"
  local package="$2"
  local version="$3"
  local binary_name="$4"
  local dst="${RUNTIME_ROOT}/${name}"
  local binary="${dst}/node_modules/.bin/${binary_name}"
  local installed=""
  if [[ -x "${binary}" ]]; then
    installed="$(${binary} --version 2>/dev/null || true)"
  fi
  if [[ "${installed}" != *"${version}"* ]]; then
    mkdir -p "${dst}"
    if [[ ! -f "${dst}/package.json" ]]; then
      npm --prefix "${dst}" init -y >/dev/null
    fi
    npm --prefix "${dst}" install --save-exact "${package}@${version}"
  fi
  [[ -x "${binary}" ]] || die "Runtime binary was not installed: ${binary}"
  installed="$(${binary} --version)"
  [[ "${installed}" == *"${version}"* ]] || die \
    "Unexpected ${name} runtime version: ${installed}; expected ${version}"
  echo "${name}: binary=${binary} version=${installed}"
}

main() {
  require_command git
  [[ -x "${MARS_NODE_BIN}" ]] || die "Required Node binary is unavailable: ${MARS_NODE_BIN}"
  [[ "$("${MARS_NODE_BIN}" --version)" == "v${NODE_VERSION}" ]] || die \
    "Unexpected Node version: $("${MARS_NODE_BIN}" --version); expected v${NODE_VERSION}"
  require_command npm
  [[ "$(npm --version)" == "${NPM_VERSION}" ]] || die \
    "Unexpected npm version: $(npm --version); expected ${NPM_VERSION}"
  require_command taskset
  require_command flock
  [[ -x "${MARS_BWRAP_BIN}" ]] || die \
    "Required bubblewrap binary is unavailable: ${MARS_BWRAP_BIN}"
  [[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || die "python3/python is required"
  [[ -x "${PYTEST_PYTHON}" ]] || PYTEST_PYTHON="${PYTHON_BIN}"
  mkdir -p "${EXTERNAL_ROOT}" "${RUNTIME_ROOT}"

  ensure_checkout opencode "${OPENCODE_REPO_URL}" "${OPENCODE_TAG}" "${OPENCODE_COMMIT}"
  ensure_checkout codex "${CODEX_REPO_URL}" "${CODEX_TAG}" "${CODEX_COMMIT}"
  ensure_npm_runtime opencode opencode-ai "${OPENCODE_VERSION}" opencode
  ensure_npm_runtime codex @openai/codex "${CODEX_VERSION}" codex

  if bool_true "${RUN_SMOKE_TEST}"; then
    "${PYTHON_BIN}" "${MARS_DIR}/experiments/agent_cli/sandbox_regression_test.py"
    "${PYTHON_BIN}" "${MARS_DIR}/experiments/agent_cli/smoke_test.py" \
      --full-runner
    PYTHONPATH="${MARS_DIR}/experiments/agent_cli:${MARS_DIR}/experiments/openhands:${MARS_DIR}:${MARS_DIR}/vllm-continuum${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTEST_PYTHON}" -m pytest -q \
      "${MARS_DIR}"/experiments/agent_cli/test_*.py \
      "${MARS_DIR}/experiments/openhands/test_qwen_responses_compat.py" \
      "${MARS_DIR}/experiments/openhands/test_vllm_api_server_mars.py"
    PYTHONPATH="${MARS_DIR}/experiments/openhands:${MARS_DIR}/experiments/agent_cli:${MARS_DIR}:${MARS_DIR}/vllm-continuum${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTEST_PYTHON}" -m pytest -q \
      "${MARS_DIR}"/experiments/openhands/test_*.py
  fi
}

main "$@"
