#!/usr/bin/env bash
set -euo pipefail

workspace="${MARS_WORKSPACE_DIR:?MARS_WORKSPACE_DIR is required}"
runtime="${MARS_RUNTIME_DIR:?MARS_RUNTIME_DIR is required}"
home_dir="${HOME:?HOME is required}"
host_home="${MARS_HOST_HOME:-}"
host_conda_root="${MARS_HOST_CONDA_ROOT:-}"
host_venvs_root="${MARS_HOST_VENVS_ROOT:-}"
host_models_root="${MARS_HOST_MODELS_ROOT:-}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
guard_source_py="${script_dir}/command_guard.py"
guard_runtime_py="${runtime}/command_guard.py"
guard_runtime_rc="${runtime}/bash_guard.rc"
tmp_dir="${TMPDIR:-$runtime/tmp}"
cache_dir="${XDG_CACHE_HOME:-$runtime/cache}"
config_dir="${XDG_CONFIG_HOME:-$runtime/config}"
data_dir="${XDG_DATA_HOME:-$runtime/data}"
state_dir="${XDG_STATE_HOME:-$runtime/state}"

mkdir -p \
  "$workspace" \
  "$runtime" \
  "$home_dir" \
  "$tmp_dir" \
  "$cache_dir" \
  "$config_dir" \
  "$data_dir" \
  "$state_dir"

if [[ -f "$guard_source_py" ]]; then
  cp "$guard_source_py" "$guard_runtime_py"
fi

cat >"$guard_runtime_rc" <<EOF
export MARS_GUARD_COMMAND_PY="$guard_runtime_py"

__mars_guard_preexec() {
  trap - DEBUG
  local cmd="\${BASH_COMMAND:-}"
  local reason=""
  local rc=0
  local cmd_lc="\${cmd,,}"

  if [[ -n "\$cmd" ]]; then
    case "\$cmd_lc" in
      *pkill*|*killall*|*kill*)
        if [[ -f "\$MARS_GUARD_COMMAND_PY" ]]; then
          reason="\$(MARS_GUARD_ACTIVE=1 /usr/bin/env python3 "\$MARS_GUARD_COMMAND_PY" check-terminal --workspace-dir "\$MARS_WORKSPACE_DIR" --command "\$cmd" 2>/dev/null || true)"
          if [[ -n "\$reason" ]]; then
            printf 'Command blocked by MARS runtime guard.\nReason: %s\nCommand: %s\n' "\$reason" "\$cmd" >&2
            rc=1
          fi
        fi
        ;;
    esac
  fi

  trap '__mars_guard_preexec' DEBUG
  return "\$rc"
}

shopt -s extdebug
trap '__mars_guard_preexec' DEBUG
EOF

declare -A seen_dirs=()
parent_dirs=()

add_dir() {
  local dir="$1"
  [[ -n "$dir" ]] || return 0
  [[ "$dir" == "/" || "$dir" == "." ]] && return 0
  if [[ -n "${seen_dirs[$dir]:-}" ]]; then
    return 0
  fi
  seen_dirs["$dir"]=1
  parent_dirs+=("$dir")
}

collect_parent_dirs() {
  local path="$1"
  local current
  current="$(dirname "$path")"
  while [[ "$current" != "/" && "$current" != "." ]]; do
    add_dir "$current"
    current="$(dirname "$current")"
  done
}

collect_parent_dirs "$workspace"
if [[ -n "$host_home" ]]; then
  collect_parent_dirs "$host_home"
  add_dir "$host_home"
fi
if [[ -n "$host_conda_root" ]]; then
  collect_parent_dirs "$host_conda_root"
fi
if [[ -n "$host_venvs_root" ]]; then
  collect_parent_dirs "$host_venvs_root"
fi
if [[ -n "$host_models_root" ]]; then
  collect_parent_dirs "$host_models_root"
fi

bwrap_args=(
  --die-with-parent
  --dev-bind /dev /dev
  --proc /proc
  --tmpfs /tmp
  --dir /var
  --dir /var/tmp
  --dir /run
)

for (( idx=${#parent_dirs[@]}-1 ; idx>=0 ; idx-- )); do
  bwrap_args+=(--dir "${parent_dirs[$idx]}")
done

bwrap_args+=(
  --ro-bind /usr /usr
  --ro-bind /bin /bin
  --ro-bind /sbin /sbin
  --ro-bind /lib /lib
  --ro-bind /lib64 /lib64
  --ro-bind /etc /etc
  --bind "$workspace" "$workspace"
  --chdir "$workspace"
  --setenv HOME "$home_dir"
  --setenv TMPDIR "$tmp_dir"
  --setenv TMP "$tmp_dir"
  --setenv TEMP "$tmp_dir"
  --setenv XDG_CACHE_HOME "$cache_dir"
  --setenv XDG_CONFIG_HOME "$config_dir"
  --setenv XDG_DATA_HOME "$data_dir"
  --setenv XDG_STATE_HOME "$state_dir"
  --setenv MARS_WORKSPACE_DIR "$workspace"
  --setenv MARS_RUNTIME_DIR "$runtime"
)

if [[ -n "$host_conda_root" && -d "$host_conda_root" ]]; then
  bwrap_args+=(--ro-bind "$host_conda_root" "$host_conda_root")
fi
if [[ -n "$host_venvs_root" && -d "$host_venvs_root" ]]; then
  bwrap_args+=(--ro-bind "$host_venvs_root" "$host_venvs_root")
fi
if [[ -n "$host_models_root" && -d "$host_models_root" ]]; then
  bwrap_args+=(--ro-bind "$host_models_root" "$host_models_root")
fi

if /usr/bin/bwrap \
  --ro-bind /usr /usr \
  --ro-bind /bin /bin \
  --ro-bind /lib /lib \
  --ro-bind /lib64 /lib64 \
  --proc /proc \
  /bin/true >/dev/null 2>&1; then
  exec /usr/bin/bwrap "${bwrap_args[@]}" /usr/bin/env BASH_ENV="$guard_runtime_rc" /bin/bash --noprofile --rcfile "$guard_runtime_rc" "$@"
fi

exec /usr/bin/env BASH_ENV="$guard_runtime_rc" /bin/bash --noprofile --rcfile "$guard_runtime_rc" "$@"
