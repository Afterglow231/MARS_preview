#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import inspect
import textwrap
from typing import Any

from command_guard import blocked_file_editor_path_reason, blocked_terminal_command_reason
from common import now_s

DEFAULT_TERMINAL_COMMAND_TIMEOUT_S = 1000.0
_TIMEOUT_PATCH_MARKER_ATTR = "_mars_nochange_timeout_patch"
_INJECTED_TIMEOUT_ATTR = "_mars_timeout_injected"


def _mark_timeout_injected(action: Any) -> Any:
    try:
        setattr(action, _INJECTED_TIMEOUT_ATTR, True)
        return action
    except Exception:
        pass
    try:
        object.__setattr__(action, _INJECTED_TIMEOUT_ATTR, True)
    except Exception:
        return action
    return action


def _make_blocked_terminal_observation(action: Any, reason: str) -> Any:
    from openhands.tools.terminal.definition import TerminalObservation

    command = str(getattr(action, "command", "") or "")
    return TerminalObservation.from_text(
        text=(
            "Command blocked by MARS safety hook.\n"
            f"Reason: {reason}\n"
            f"Command: {command}"
        ),
        is_error=True,
        command=command,
        exit_code=126,
        timeout=False,
    )


def _with_default_terminal_timeout(action: Any) -> Any:
    command = str(getattr(action, "command", "") or "")
    is_input = bool(getattr(action, "is_input", False))
    timeout = getattr(action, "timeout", None)
    if not command.strip() or is_input or timeout is not None:
        return action

    if hasattr(action, "model_copy"):
        updated = action.model_copy(
            update={"timeout": DEFAULT_TERMINAL_COMMAND_TIMEOUT_S}
        )
        return _mark_timeout_injected(updated)

    try:
        setattr(action, "timeout", DEFAULT_TERMINAL_COMMAND_TIMEOUT_S)
    except Exception:
        return action
    return _mark_timeout_injected(action)


def _patch_terminal_session_execute(terminal_session_module: Any) -> bool:
    session_cls = getattr(terminal_session_module, "TerminalSession", None)
    execute = getattr(session_cls, "execute", None)
    if session_cls is None or execute is None:
        return False
    if getattr(execute, _TIMEOUT_PATCH_MARKER_ATTR, False):
        return True

    source = textwrap.dedent(inspect.getsource(execute))
    original = "is_blocking = action.timeout is not None"
    replacement = (
        "is_blocking = action.timeout is not None and "
        f"not getattr(action, \"{_INJECTED_TIMEOUT_ATTR}\", False)"
    )
    if original not in source:
        raise RuntimeError("Unable to patch TerminalSession.execute: blocking guard not found")

    namespace = dict(vars(terminal_session_module))
    exec(source.replace(original, replacement, 1), namespace)
    patched_execute = namespace["execute"]
    setattr(patched_execute, _TIMEOUT_PATCH_MARKER_ATTR, True)
    setattr(session_cls, "execute", patched_execute)
    return True


def install_terminal_timeout_patch() -> bool:
    try:
        from openhands.tools.terminal.terminal import terminal_session as terminal_session_module
    except ModuleNotFoundError:
        return False
    return _patch_terminal_session_execute(terminal_session_module)


def _make_blocked_file_editor_observation(action: Any, reason: str) -> Any:
    from openhands.tools.file_editor.definition import FileEditorObservation

    command = str(getattr(action, "command", "") or "")
    path = str(getattr(action, "path", "") or "")
    return FileEditorObservation.from_text(
        text=(
            "File edit blocked by MARS safety hook.\n"
            f"Reason: {reason}\n"
            f"Path: {path}"
        ),
        is_error=True,
        command=command or "view",
        path=path,
    )


class TracingToolExecutor:
    def __init__(self, *, tool_name: str, executor: Any, collector: Any):
        self.tool_name = tool_name
        self.executor = executor
        self.collector = collector
        self._mars_tracing_wrapped = True

    def __call__(self, action: Any, conversation: Any | None = None) -> Any:
        action_id = self.collector.on_tool_executor_start(self.tool_name, action)
        try:
            if str(self.tool_name).lower() in {"terminal", "terminaltool"}:
                action = _with_default_terminal_timeout(action)
                command = str(getattr(action, "command", "") or "")
                is_input = bool(getattr(action, "is_input", False))
                if command.strip() and not is_input:
                    reason = blocked_terminal_command_reason(
                        command,
                        self.collector.workspace_dir,
                    )
                    if reason is not None:
                        if hasattr(self.collector, "events"):
                            self.collector.events.write(
                                {
                                    "ts": now_s(),
                                    "event": "tool_blocked",
                                    "request_id": self.collector.request_id,
                                    "tool_name": self.tool_name,
                                    "command": command,
                                    "reason": reason,
                                }
                            )
                        blocked = _make_blocked_terminal_observation(action, reason)
                        self.collector.on_tool_executor_finish(action_id, observation=blocked)
                        return blocked
            if str(self.tool_name).lower() in {"file_editor", "fileeditortool"}:
                path = str(getattr(action, "path", "") or "")
                if path.strip():
                    reason = blocked_file_editor_path_reason(
                        path,
                        self.collector.workspace_dir,
                    )
                    if reason is not None:
                        if hasattr(self.collector, "events"):
                            self.collector.events.write(
                                {
                                    "ts": now_s(),
                                    "event": "tool_blocked",
                                    "request_id": self.collector.request_id,
                                    "tool_name": self.tool_name,
                                    "path": path,
                                    "reason": reason,
                                }
                            )
                        blocked = _make_blocked_file_editor_observation(action, reason)
                        self.collector.on_tool_executor_finish(action_id, observation=blocked)
                        return blocked
            observation = self.executor(action, conversation)
        except Exception as exc:
            self.collector.on_tool_executor_finish(action_id, error=exc)
            raise
        self.collector.on_tool_executor_finish(action_id, observation=observation)
        return observation

    def close(self) -> None:
        close = getattr(self.executor, "close", None)
        if callable(close):
            close()


def wrap_agent_tools(agent: Any, collector: Any) -> None:
    install_terminal_timeout_patch()
    for tool_name, tool in list(agent.tools_map.items()):
        executor = getattr(tool, "executor", None)
        if executor is None:
            continue
        if getattr(executor, "_mars_tracing_wrapped", False):
            continue
        agent.tools_map[tool_name] = tool.set_executor(
            TracingToolExecutor(
                tool_name=tool.name,
                executor=executor,
                collector=collector,
            )
        )
