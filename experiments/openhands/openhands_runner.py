#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from common import now_s
from llm_trace_bridge import install_llm_trace_patch, register_llm_collector, unregister_llm_collector
from request_bootstrap import (
    build_user_request,
    build_verification_nudge,
    materialize_workspace_init,
    requires_tool_verification,
)
from tool_trace_bridge import wrap_agent_tools
from trace_writer import RequestTraceCollector


MAX_VERIFICATION_NUDGES = 1
RUNTIME_DIR_NAME = ".mars_runtime"
SANDBOX_SHELL_PATH = Path(__file__).with_name("sandbox_shell.sh")
HOST_HOME = Path(os.environ.get("MARS_HOST_HOME") or Path.home()).expanduser().resolve()


def _load_openhands_sdk() -> dict[str, Any]:
    try:
        from pydantic import SecretStr

        from openhands.sdk import Agent, Conversation, LLM, Tool
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.preset.default import register_default_tools
        from openhands.tools.task_tracker import TaskTrackerTool
        from openhands.tools.terminal import TerminalTool
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenHands SDK is not installed in the current Python environment. "
            "Use the dedicated OpenHands Python environment to run this runner."
        ) from exc
    return {
        "Agent": Agent,
        "Conversation": Conversation,
        "FileEditorTool": FileEditorTool,
        "LLM": LLM,
        "SecretStr": SecretStr,
        "TaskTrackerTool": TaskTrackerTool,
        "TerminalTool": TerminalTool,
        "Tool": Tool,
        "register_default_tools": register_default_tools,
    }


# Runtime configuration used to construct one OpenHands conversation.
@dataclass(frozen=True)
class OpenHandsConfig:
    model: str
    base_url: str
    api_key: str
    temperature: float | None
    top_p: float | None
    max_output_tokens: int | None
    llm_timeout_s: int | None
    llm_num_retries: int
    max_iterations: int
    terminal_no_change_timeout_s: int | None
    terminal_type: str | None


def _normalize_openai_compatible_model(model: str) -> str:
    text = str(model).strip()
    if not text:
        raise ValueError("model must be non-empty")
    if "/" in text:
        return text
    return f"openai/{text}"


def _build_tools(config: OpenHandsConfig, sdk: dict[str, Any]) -> list[Any]:
    Tool = sdk["Tool"]
    TerminalTool = sdk["TerminalTool"]
    FileEditorTool = sdk["FileEditorTool"]
    TaskTrackerTool = sdk["TaskTrackerTool"]

    terminal_params: dict[str, Any] = {}
    if config.terminal_no_change_timeout_s is not None:
        terminal_params["no_change_timeout_seconds"] = int(config.terminal_no_change_timeout_s)
    # Force subprocess mode so we can route all terminal commands through the
    # workspace-scoped sandbox shell instead of a host-level tmux session.
    terminal_params["terminal_type"] = "subprocess"
    terminal_params["shell_path"] = str(SANDBOX_SHELL_PATH)

    return [
        Tool(name=TerminalTool.name, params=terminal_params),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ]


def _prepare_workspace_runtime_dirs(workspace_dir: Path) -> dict[str, Path]:
    runtime_root = (workspace_dir / RUNTIME_DIR_NAME).resolve()
    dirs = {
        "runtime_root": runtime_root,
        "home": runtime_root / "home",
        "tmp": runtime_root / "tmp",
        "cache": runtime_root / "cache",
        "config": runtime_root / "config",
        "data": runtime_root / "data",
        "state": runtime_root / "state",
        "artifacts": runtime_root / "artifacts",
        "pycache": runtime_root / "pycache",
        "pip_cache": runtime_root / "pip-cache",
        "uv_cache": runtime_root / "uv-cache",
        "npm_cache": runtime_root / "npm-cache",
        "cargo_home": runtime_root / "cargo",
        "rustup_home": runtime_root / "rustup",
        "nvm_home": runtime_root / "nvm",
        "deno_home": runtime_root / "deno",
        "elan_home": runtime_root / "elan",
        "opam_root": runtime_root / "opam",
        "maven_home": runtime_root / "m2",
        "keras_home": runtime_root / "keras",
        "gradle_home": runtime_root / "gradle",
        "hf_home": runtime_root / "hf",
        "torch_home": runtime_root / "torch",
        "cuda_cache": runtime_root / "cuda-cache",
        "jupyter_config": runtime_root / "jupyter-config",
        "jupyter_data": runtime_root / "jupyter-data",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _apply_workspace_runtime_env(workspace_dir: Path) -> dict[str, str]:
    dirs = _prepare_workspace_runtime_dirs(workspace_dir)
    host_conda_root = os.environ.get("MARS_HOST_CONDA_ROOT") or str(HOST_HOME / "miniconda3")
    host_venvs_root = os.environ.get("MARS_HOST_VENVS_ROOT") or str(HOST_HOME / ".venvs")
    host_models_root = os.environ.get("MARS_HOST_MODELS_ROOT") or str(HOST_HOME / "models")
    env_updates = {
        "HOME": str(dirs["home"]),
        "TMPDIR": str(dirs["tmp"]),
        "TMP": str(dirs["tmp"]),
        "TEMP": str(dirs["tmp"]),
        "XDG_CACHE_HOME": str(dirs["cache"]),
        "XDG_CONFIG_HOME": str(dirs["config"]),
        "XDG_DATA_HOME": str(dirs["data"]),
        "XDG_STATE_HOME": str(dirs["state"]),
        "PIP_CACHE_DIR": str(dirs["pip_cache"]),
        "UV_CACHE_DIR": str(dirs["uv_cache"]),
        "NPM_CONFIG_CACHE": str(dirs["npm_cache"]),
        "YARN_CACHE_FOLDER": str(dirs["npm_cache"]),
        "PNPM_HOME": str(dirs["npm_cache"]),
        "CARGO_HOME": str(dirs["cargo_home"]),
        "RUSTUP_HOME": str(dirs["rustup_home"]),
        "NVM_DIR": str(dirs["nvm_home"]),
        "DENO_DIR": str(dirs["deno_home"]),
        "DENO_INSTALL": str(dirs["deno_home"]),
        "ELAN_HOME": str(dirs["elan_home"]),
        "OPAMROOT": str(dirs["opam_root"]),
        "MAVEN_CONFIG": str(dirs["maven_home"]),
        "KERAS_HOME": str(dirs["keras_home"]),
        "GRADLE_USER_HOME": str(dirs["gradle_home"]),
        "HF_HOME": str(dirs["hf_home"]),
        "TRANSFORMERS_CACHE": str(dirs["hf_home"]),
        "TORCH_HOME": str(dirs["torch_home"]),
        "CUDA_CACHE_PATH": str(dirs["cuda_cache"]),
        "PYTHONPYCACHEPREFIX": str(dirs["pycache"]),
        "IPYTHONDIR": str(dirs["config"] / "ipython"),
        "JUPYTER_CONFIG_DIR": str(dirs["jupyter_config"]),
        "JUPYTER_DATA_DIR": str(dirs["jupyter_data"]),
        "JUPYTER_RUNTIME_DIR": str(dirs["runtime_root"] / "jupyter-runtime"),
        "PYTHONUSERBASE": str(dirs["home"] / ".local"),
        "PIP_CONFIG_FILE": "/dev/null",
        "PIP_REQUIRE_VIRTUALENV": "1",
        "PIP_NO_INPUT": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "",
        "VIRTUAL_ENV": "",
        "CONDA_PREFIX": "",
        "CONDA_DEFAULT_ENV": "",
        "CONDA_EXE": "",
        "CONDA_PYTHON_EXE": "",
        "_CE_CONDA": "",
        "_CE_M": "",
        "MARS_WORKSPACE_DIR": str(workspace_dir),
        "MARS_RUNTIME_DIR": str(dirs["runtime_root"]),
        "MARS_ARTIFACT_DIR": str(dirs["artifacts"]),
        "MARS_HOST_HOME": str(HOST_HOME),
        "MARS_HOST_CONDA_ROOT": host_conda_root,
        "MARS_HOST_VENVS_ROOT": host_venvs_root,
        "MARS_HOST_MODELS_ROOT": host_models_root,
        "MARS_LOCAL_INPUTS_DIR": os.environ.get("MARS_LOCAL_INPUTS_DIR", ""),
    }
    for key, value in env_updates.items():
        os.environ[key] = value
    local_bin = dirs["home"] / ".local" / "bin"
    local_bin.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")
    env_updates["PATH"] = os.environ["PATH"]
    return {key: value for key, value in env_updates.items()}


def run_task_request(
    *,
    config: OpenHandsConfig,
    task: dict[str, Any],
    request_id: str,
    arrival_time_s: float,
    workspace_root: Path,
    events: Any,
) -> dict[str, Any]:
    # Run OpenHands requests with full tool chain & trace record
    workspace_dir = (workspace_root / request_id).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    runtime_env = _apply_workspace_runtime_env(workspace_dir)
    sdk = _load_openhands_sdk()
    Agent = sdk["Agent"]
    Conversation = sdk["Conversation"]
    LLM = sdk["LLM"]
    SecretStr = sdk["SecretStr"]
    register_default_tools = sdk["register_default_tools"]

    install_llm_trace_patch()
    register_default_tools(enable_browser=False)

    dataset_request_id = task.get("request_id")
    benchmark = task.get("benchmark")
    task_id = task.get("task_id")

    collector = RequestTraceCollector(
        events=events,
        request_id=request_id,
        dataset_request_id=str(dataset_request_id) if dataset_request_id is not None else None,
        benchmark=str(benchmark) if benchmark is not None else None,
        task_id=str(task_id) if task_id is not None else None,
        arrival_time_s=arrival_time_s,
        workspace_dir=workspace_dir,
    )
    collector.mark_worker_start()
    file_count = materialize_workspace_init(task, workspace_dir)
    collector.set_workspace_materialized(file_count)

    llm = LLM(
        model=_normalize_openai_compatible_model(config.model),
        api_key=SecretStr(config.api_key),
        base_url=config.base_url,
        stream=True,
        timeout=config.llm_timeout_s,
        num_retries=config.llm_num_retries,
        temperature=config.temperature,
        top_p=config.top_p,
        max_output_tokens=config.max_output_tokens,
    )
    register_llm_collector(llm, collector)

    conversation = None
    error: Exception | None = None
    status = "unknown"
    conversation_status: str | None = None
    verification_required = requires_tool_verification(task)
    started_at = now_s()
    events.write(
        {
            "ts": started_at,
            "event": "session_init",
            "request_id": request_id,
            "dataset_request_id": dataset_request_id,
            "benchmark": benchmark,
            "task_id": task_id,
            "workspace_dir": str(workspace_dir),
            "runtime_dir": runtime_env.get("MARS_RUNTIME_DIR"),
            "artifact_dir": runtime_env.get("MARS_ARTIFACT_DIR"),
        }
    )

    try:
        agent = Agent(
            llm=llm,
            tools=_build_tools(config, sdk),
            system_prompt_kwargs={"cli_mode": True},
            condenser=None,
        )
        conversation = Conversation(
            agent=agent,
            workspace=workspace_dir,
            callbacks=[collector.on_event],
            token_callbacks=[collector.on_token],
            max_iteration_per_run=config.max_iterations,
            visualizer=None,
            delete_on_close=False,
        )
        conversation._ensure_agent_ready()
        wrap_agent_tools(conversation.agent, collector)
        conversation.send_message(build_user_request(task))
        conversation.run()
        if verification_required:
            for attempt in range(MAX_VERIFICATION_NUDGES):
                if collector.has_verification_tool_use():
                    break
                events.write(
                    {
                        "ts": now_s(),
                        "event": "verification_nudge",
                        "request_id": request_id,
                        "dataset_request_id": dataset_request_id,
                        "benchmark": benchmark,
                        "task_id": task_id,
                        "attempt": int(attempt + 1),
                    }
                )
                conversation.send_message(build_verification_nudge(task))
                conversation.run()
                if collector.has_verification_tool_use():
                    break
        conversation_status = str(conversation.state.execution_status.value)
        status = conversation_status
        if verification_required and not collector.has_verification_tool_use():
            events.write(
                {
                    "ts": now_s(),
                    "event": "verification_missing",
                    "request_id": request_id,
                    "dataset_request_id": dataset_request_id,
                    "benchmark": benchmark,
                    "task_id": task_id,
                }
            )
            status = "verification_missing"
    except Exception as exc:
        error = exc
        status = "exception"
        conversation_status = (
            str(conversation.state.execution_status.value)
            if conversation is not None
            else None
        )
    finally:
        unregister_llm_collector(llm)
        if conversation is not None:
            try:
                conversation.close()
            except Exception:
                pass

    if status == "finished":
        status = "completed"
    elif status == "waiting_for_confirmation":
        status = "needs_confirmation"

    return collector.finalize(
        status=status,
        conversation_status=conversation_status,
        error=error,
    )
