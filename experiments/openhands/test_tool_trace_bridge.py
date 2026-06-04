from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
import types
import unittest


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import tool_trace_bridge


@dataclass(frozen=True)
class DummyAction:
    command: str
    is_input: bool = False
    timeout: float | None = None

    def model_copy(self, *, update: dict[str, object] | None = None) -> "DummyAction":
        return replace(self, **(update or {}))


class DummyCollector:
    def __init__(self) -> None:
        self.workspace_dir = Path("/tmp")
        self.request_id = "req"
        self.events = None
        self.started: list[tuple[str, DummyAction]] = []
        self.finished: list[tuple[str, object | None, Exception | None]] = []

    def on_tool_executor_start(self, tool_name: str, action: DummyAction) -> str:
        self.started.append((tool_name, action))
        return "tool-call"

    def on_tool_executor_finish(
        self,
        action_id: str,
        observation: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.finished.append((action_id, observation, error))


class ToolTraceBridgeTest(unittest.TestCase):
    def test_terminal_wrapper_applies_default_timeout(self) -> None:
        collector = DummyCollector()
        seen: list[DummyAction] = []

        def executor(action: DummyAction, conversation: object | None = None) -> object:
            seen.append(action)
            return {"ok": True}

        wrapped = tool_trace_bridge.TracingToolExecutor(
            tool_name="terminal",
            executor=executor,
            collector=collector,
        )
        wrapped(DummyAction(command="python3 simple_bpe.py"))

        self.assertEqual(len(seen), 1)
        self.assertEqual(
            seen[0].timeout,
            tool_trace_bridge.DEFAULT_TERMINAL_COMMAND_TIMEOUT_S,
        )
        self.assertTrue(getattr(seen[0], "_mars_timeout_injected", False))

    def test_terminal_wrapper_preserves_explicit_timeout(self) -> None:
        collector = DummyCollector()
        seen: list[DummyAction] = []

        def executor(action: DummyAction, conversation: object | None = None) -> object:
            seen.append(action)
            return {"ok": True}

        wrapped = tool_trace_bridge.TracingToolExecutor(
            tool_name="terminal",
            executor=executor,
            collector=collector,
        )
        wrapped(DummyAction(command="sleep 5", timeout=12.0))

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].timeout, 12.0)
        self.assertFalse(getattr(seen[0], "_mars_timeout_injected", False))

    def test_patch_allows_no_change_timeout_for_injected_timeout(self) -> None:
        @dataclass(frozen=True)
        class FakeAction:
            timeout: float | None = None
            _mars_timeout_injected: bool = False

        class FakeTerminalSession:
            def execute(self, action: FakeAction) -> bool:
                is_blocking = action.timeout is not None
                return is_blocking

        fake_module = types.SimpleNamespace(TerminalSession=FakeTerminalSession)

        patched = tool_trace_bridge._patch_terminal_session_execute(fake_module)

        self.assertTrue(patched)
        self.assertTrue(FakeTerminalSession().execute(FakeAction(timeout=12.0)))
        self.assertFalse(
            FakeTerminalSession().execute(
                FakeAction(timeout=12.0, _mars_timeout_injected=True)
            )
        )


if __name__ == "__main__":
    unittest.main()
