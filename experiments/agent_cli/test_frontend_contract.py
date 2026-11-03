#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

try:
    from .frontend_contract import (
        AgentFrontend,
        AgentRequestContext,
        FrontendCapabilities,
        FrontendOutcome,
        FrontendTerminalStatus,
    )
except ImportError:  # Supports direct execution from this directory.
    from frontend_contract import (  # type: ignore
        AgentFrontend,
        AgentRequestContext,
        FrontendCapabilities,
        FrontendOutcome,
        FrontendTerminalStatus,
    )


def test_request_context_detaches_mutable_configuration(tmp_path: Path) -> None:
    runtime_env = {"TOKEN": "before"}
    metadata = {"nested": {"items": [1, 2]}}
    context = AgentRequestContext(
        request_id="req-1",
        job_id="job-1",
        prompt="solve it",
        workspace_dir=tmp_path,
        model="model",
        backend_base_url="http://127.0.0.1:8000/v1",
        readonly_paths=[tmp_path / "input"],
        runtime_env=runtime_env,
        metadata=metadata,
    )

    runtime_env["TOKEN"] = "after"
    metadata["nested"]["items"].append(3)
    assert context.runtime_env["TOKEN"] == "before"
    assert context.metadata["nested"]["items"] == (1, 2)
    assert context.readonly_paths == (tmp_path / "input",)
    with pytest.raises(TypeError):
        context.runtime_env["NEW"] = "forbidden"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        context.request_id = "changed"  # type: ignore[misc]


def test_terminal_outcome_is_typed_and_requires_real_finite_times() -> None:
    outcome = FrontendOutcome(
        request_id="req",
        status="completed",  # type: ignore[arg-type]
        terminal_monotonic_s=12.0,
        terminal_time_s=1012.0,
    )
    assert outcome.status is FrontendTerminalStatus.COMPLETED

    with pytest.raises(ValueError, match="finite"):
        FrontendOutcome(
            request_id="req",
            status=FrontendTerminalStatus.ERROR,
            terminal_monotonic_s=float("nan"),
            terminal_time_s=1012.0,
        )


def test_agent_frontend_protocol_accepts_a_native_lifecycle_owner() -> None:
    class Frontend:
        capabilities = FrontendCapabilities(name="test")

        def run(self, request, events):  # type: ignore[no-untyped-def]
            return events.frontend_terminal(FrontendTerminalStatus.COMPLETED)

        def cancel(self, reason: str) -> None:
            del reason

        def close(self) -> None:
            return

    assert isinstance(Frontend(), AgentFrontend)
