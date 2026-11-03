#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from experiments.agent_cli.codex_app_server import (
    CodexAppServerDriver,
    CodexEventCallbackError,
    CodexProcessExited,
    build_codex_app_server_command,
)


PINNED_CODEX = Path(
    os.environ.get(
        "MARS_AGENT_BASELINES_ROOT",
        str(Path.home() / "external" / "mars_agent_baselines"),
    )
) / "runtime/codex/node_modules/.bin/codex"


MOCK_APP_SERVER = r'''
import json
import os
import sys
import time


capture_path = os.environ["MOCK_CODEX_CAPTURE"]
capture = open(capture_path, "w", encoding="utf-8", buffering=1)
initialized = False
thread_id = "thread-native-001"
turn_index = 0
active_turn_id = None
approval_turn_id = None


def emit(message):
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def turn(turn_id, status, *, error=None, text=None):
    items = []
    if text is not None:
        items.append({"type": "agentMessage", "id": "message-" + turn_id, "text": text})
    return {
        "id": turn_id,
        "items": items,
        "status": status,
        "error": error,
        "startedAt": 1,
        "completedAt": 2 if status != "inProgress" else None,
        "durationMs": 1000 if status != "inProgress" else None,
    }


for raw_line in sys.stdin:
    message = json.loads(raw_line)
    capture.write(json.dumps(message, separators=(",", ":")) + "\n")
    method = message.get("method")
    request_id = message.get("id")

    if method == "initialize":
        emit({
            "id": request_id,
            "result": {
                "userAgent": "mock-codex/0.149.1",
                "codexHome": os.environ.get("CODEX_HOME", "/tmp/mock-codex"),
                "platformFamily": "unix",
                "platformOs": "linux",
            },
        })
        continue

    if method == "initialized":
        initialized = True
        continue

    if method == "thread/start":
        if not initialized:
            emit({"id": request_id, "error": {"code": -32000, "message": "Not initialized"}})
            continue
        # Exercise notification-before-response ordering.
        emit({
            "method": "thread/started",
            "params": {"thread": {"id": thread_id, "ephemeral": True}},
            "emittedAtMs": 1000,
        })
        emit({
            "id": request_id,
            "result": {
                "thread": {"id": thread_id, "ephemeral": True},
                "model": message["params"].get("model") or "mock-model",
                "modelProvider": message["params"].get("modelProvider") or "mars",
                "cwd": message["params"].get("cwd") or "/tmp",
            },
        })
        continue

    if method == "turn/start":
        if message["params"].get("threadId") != thread_id:
            emit({"id": request_id, "error": {"code": -32602, "message": "wrong thread"}})
            continue
        turn_index += 1
        active_turn_id = "turn-%03d" % turn_index
        text = message["params"]["input"][0].get("text", "")
        emit({"id": request_id, "result": {"turn": turn(active_turn_id, "inProgress")}})
        emit({
            "method": "turn/started",
            "params": {"threadId": thread_id, "turn": turn(active_turn_id, "inProgress")},
            "emittedAtMs": 2000 + turn_index,
        })

        if text == "exit-before-terminal":
            capture.flush()
            sys.stdout.flush()
            sys.exit(23)
        if text == "hang":
            continue
        if text == "iteration-limit":
            emit({
                "method": "rawResponse/completed",
                "params": {
                    "threadId": thread_id,
                    "turnId": active_turn_id,
                    "responseId": "response-" + active_turn_id,
                    "usage": {
                        "inputTokens": 10,
                        "cachedInputTokens": 0,
                        "outputTokens": 2,
                    },
                },
                "emittedAtMs": 2050,
            })
            # A real Codex turn would now begin its next model round.  Hold the
            # turn open so the client must enforce its native round budget via
            # turn/interrupt.
            continue
        if text == "server-request":
            approval_turn_id = active_turn_id
            emit({
                "method": "item/commandExecution/requestApproval",
                "id": "approval-1",
                "params": {
                    "threadId": thread_id,
                    "turnId": active_turn_id,
                    "itemId": "command-approval",
                    "command": "false",
                    "cwd": "/tmp",
                },
            })
            continue

        # A terminal notification for another turn must never end this turn.
        if turn_index == 1:
            emit({
                "method": "turn/completed",
                "params": {
                    "threadId": thread_id,
                    "turn": turn("turn-unrelated", "completed", text="wrong terminal"),
                },
                "emittedAtMs": 2100,
            })

        item_id = "command-" + active_turn_id
        emit({
            "method": "item/started",
            "params": {
                "threadId": thread_id,
                "turnId": active_turn_id,
                "startedAtMs": 3000,
                "item": {
                    "type": "commandExecution",
                    "id": item_id,
                    "command": "printf ok",
                    "cwd": "/tmp",
                    "status": "inProgress",
                },
            },
            "emittedAtMs": 3001,
        })
        emit({
            "method": "rawResponseItem/completed",
            "params": {
                "threadId": thread_id,
                "turnId": active_turn_id,
                "item": {
                    "type": "function_call",
                    "id": "fc-" + active_turn_id,
                    "call_id": item_id,
                    "name": "shell_command",
                    "arguments": "{\"command\":\"printf ok\"}",
                },
            },
            "emittedAtMs": 3100,
        })
        emit({
            "method": "rawResponse/completed",
            "params": {
                "threadId": thread_id,
                "turnId": active_turn_id,
                "responseId": "response-" + active_turn_id,
                "usage": {"inputTokens": 10, "cachedInputTokens": 2, "outputTokens": 3},
            },
            "emittedAtMs": 3200,
        })
        emit({
            "method": "item/completed",
            "params": {
                "threadId": thread_id,
                "turnId": active_turn_id,
                "completedAtMs": 3500,
                "item": {
                    "type": "commandExecution",
                    "id": item_id,
                    "command": "printf ok",
                    "cwd": "/tmp",
                    "status": "completed",
                    "exitCode": 0,
                    "durationMs": 500,
                },
            },
            "emittedAtMs": 3501,
        })
        emit({
            "method": "turn/completed",
            "params": {
                "threadId": thread_id,
                "turn": turn(active_turn_id, "completed", text="done " + text),
            },
            "emittedAtMs": 4000 + turn_index,
        })
        continue

    if method == "turn/interrupt":
        emit({"id": request_id, "result": {}})
        # Again prove that the driver keys completion by both thread and turn.
        emit({
            "method": "turn/completed",
            "params": {
                "threadId": thread_id,
                "turn": turn("turn-not-active", "completed", text="wrong terminal"),
            },
            "emittedAtMs": 5000,
        })
        time.sleep(0.01)
        requested_turn_id = message["params"].get("turnId")
        emit({
            "method": "turn/completed",
            "params": {
                "threadId": thread_id,
                "turn": turn(requested_turn_id, "interrupted"),
            },
            "emittedAtMs": 5001,
        })
        active_turn_id = None
        continue

    # This is the client's fail-closed response to our server-initiated request.
    if method is None and request_id == "approval-1":
        status = "failed" if "error" in message else "completed"
        error = (
            {"message": "approval client rejected request"}
            if status == "failed"
            else None
        )
        emit({
            "method": "turn/completed",
            "params": {
                "threadId": thread_id,
                "turn": turn(approval_turn_id, status, error=error),
            },
            "emittedAtMs": 6000,
        })
        approval_turn_id = None
        active_turn_id = None
        continue

    if method is not None and "id" in message:
        emit({"id": request_id, "error": {"code": -32601, "message": "unknown method"}})
'''


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _mock_driver(
    tmp_path: Path,
    *,
    callback=None,
) -> tuple[CodexAppServerDriver, Path, Path, Path]:
    script = tmp_path / "mock_codex_app_server.py"
    script.write_text(MOCK_APP_SERVER, encoding="utf-8")
    capture = tmp_path / "client_messages.jsonl"
    raw = tmp_path / "app_server_raw.jsonl"
    wire = tmp_path / "app_server_wire.jsonl"
    stderr = tmp_path / "app_server_stderr.txt"
    env = dict(os.environ)
    env["MOCK_CODEX_CAPTURE"] = str(capture)
    driver = CodexAppServerDriver(
        command=[sys.executable, str(script)],
        raw_jsonl_path=raw,
        wire_audit_path=wire,
        stderr_path=stderr,
        env=env,
        event_callback=callback,
        rpc_timeout_s=1.0,
        interrupt_grace_s=0.5,
        shutdown_timeout_s=0.5,
    )
    return driver, capture, raw, wire


def _start_mock_thread(driver: CodexAppServerDriver, tmp_path: Path) -> None:
    result = driver.start_thread(
        model="Qwen3-Coder-30B-A3B-Instruct-FP8",
        model_provider="mars",
        cwd=tmp_path,
        approval_policy="never",
        sandbox="dangerFullAccess",
        ephemeral=True,
        experimental_raw_events=True,
    )
    assert result["thread"]["id"] == "thread-native-001"


def test_native_lifecycle_raw_events_and_same_thread_verification(tmp_path: Path) -> None:
    callbacks: list[tuple[str, float]] = []
    driver, capture, raw, wire = _mock_driver(
        tmp_path,
        callback=lambda message, ts: callbacks.append((message["method"], ts)),
    )

    with driver:
        _start_mock_thread(driver, tmp_path)
        initial = driver.run_turn(
            "initial task",
            phase="initial",
            responsesapi_client_metadata={"mars_job_id": "req-001"},
            client_user_message_id="client-initial",
            timeout_s=2.0,
        )
        verification = driver.run_verification_turn(
            "verification nudge",
            responsesapi_client_metadata={"mars_job_id": "req-001"},
            client_user_message_id="client-verification",
            timeout_s=2.0,
        )

        assert initial.thread_id == verification.thread_id == driver.thread_id
        assert initial.turn_id == "turn-001"
        assert verification.turn_id == "turn-002"
        assert initial.status == verification.status == "completed"
        assert initial.raw_response_count == verification.raw_response_count == 1
        assert initial.timed_out is False
        assert initial.iteration_limited is False
        assert initial.interrupt_requested is False
        assert all(
            event.message.get("params", {}).get("turn", {}).get("id")
            != "turn-unrelated"
            for event in initial.events
        )
        assert [event.method for event in initial.events] == [
            "turn/started",
            "item/started",
            "rawResponseItem/completed",
            "rawResponse/completed",
            "item/completed",
            "turn/completed",
        ]

    sent = _jsonl(capture)
    assert sent[0]["method"] == "initialize"
    assert "jsonrpc" not in sent[0]
    assert sent[1] == {"method": "initialized"}
    thread_start = next(row for row in sent if row.get("method") == "thread/start")
    assert thread_start["params"]["experimentalRawEvents"] is True
    assert thread_start["params"]["ephemeral"] is True
    turns = [row for row in sent if row.get("method") == "turn/start"]
    assert len(turns) == 2
    assert {row["params"]["threadId"] for row in turns} == {"thread-native-001"}
    assert turns[0]["params"]["responsesapiClientMetadata"] == {
        "mars_job_id": "req-001"
    }
    assert turns[0]["params"]["clientUserMessageId"] == "client-initial"
    assert turns[1]["params"]["clientUserMessageId"] == "client-verification"

    # Raw JSONL is the exact server stdout protocol, not a normalized trace.
    received = _jsonl(raw)
    assert any(row.get("method") == "rawResponseItem/completed" for row in received)
    assert any(row.get("method") == "rawResponse/completed" for row in received)
    assert any("emittedAtMs" in row for row in received if "method" in row)

    audit = _jsonl(wire)
    assert {row["direction"] for row in audit} == {"to_server", "from_server"}
    assert all(row["schema"] == "codex_app_server_wire_v1" for row in audit)
    callback_methods = [method for method, _ts in callbacks]
    assert "item/started" in callback_methods
    assert callback_methods.count("turn/completed") == 3  # includes unrelated terminal


def test_timeout_interrupts_and_waits_for_matching_native_terminal(tmp_path: Path) -> None:
    driver, capture, raw, _wire = _mock_driver(tmp_path)
    with driver:
        _start_mock_thread(driver, tmp_path)
        started = time.monotonic()
        result = driver.run_turn(
            "hang",
            timeout_s=0.05,
            interrupt_grace_s=0.5,
            responsesapi_client_metadata={"mars_job_id": "req-timeout"},
        )
        elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert result.status == "interrupted"
    assert result.timed_out is True
    assert result.iteration_limited is False
    assert result.interrupt_requested is True
    assert result.turn_id == "turn-001"
    assert result.events[-1].method == "turn/completed"
    assert result.events[-1].message["params"]["turn"]["id"] == "turn-001"
    assert all(
        event.message.get("params", {}).get("turn", {}).get("id")
        != "turn-not-active"
        for event in result.events
    )
    sent = _jsonl(capture)
    interrupt = next(row for row in sent if row.get("method") == "turn/interrupt")
    assert interrupt["params"] == {
        "threadId": "thread-native-001",
        "turnId": "turn-001",
    }
    assert any(
        row.get("method") == "turn/completed"
        and row.get("params", {}).get("turn", {}).get("status") == "interrupted"
        for row in _jsonl(raw)
    )


def test_max_raw_responses_interrupts_without_fabricating_success(tmp_path: Path) -> None:
    driver, capture, _raw, _wire = _mock_driver(tmp_path)
    with driver:
        _start_mock_thread(driver, tmp_path)
        started = time.monotonic()
        result = driver.run_turn(
            "iteration-limit",
            timeout_s=2.0,
            max_raw_responses=1,
            interrupt_grace_s=0.5,
        )
        elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert result.status == "interrupted"
    assert result.iteration_limited is True
    assert result.timed_out is False
    assert result.interrupt_requested is True
    assert result.raw_response_count == 1
    assert result.turn["status"] == "interrupted"
    sent = _jsonl(capture)
    interrupt = next(row for row in sent if row.get("method") == "turn/interrupt")
    assert interrupt["params"]["turnId"] == result.turn_id


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_max_raw_responses_must_be_a_positive_integer(
    tmp_path: Path, value: object
) -> None:
    driver, _capture, _raw, _wire = _mock_driver(tmp_path)
    with driver:
        _start_mock_thread(driver, tmp_path)
        with pytest.raises(ValueError, match="max_raw_responses"):
            driver.run_turn("initial task", timeout_s=1.0, max_raw_responses=value)  # type: ignore[arg-type]


def test_process_exit_never_becomes_synthetic_completion(tmp_path: Path) -> None:
    driver, _capture, _raw, _wire = _mock_driver(tmp_path)
    with driver:
        _start_mock_thread(driver, tmp_path)
        with pytest.raises(CodexProcessExited) as raised:
            driver.run_turn("exit-before-terminal", timeout_s=1.0)
    assert raised.value.returncode == 23


def test_server_initiated_request_is_rejected_without_deadlock(tmp_path: Path) -> None:
    driver, capture, _raw, _wire = _mock_driver(tmp_path)
    with driver:
        _start_mock_thread(driver, tmp_path)
        result = driver.run_turn("server-request", timeout_s=1.0)
        unexpected = driver.unexpected_server_requests

    assert result.status == "failed"
    assert result.error == {"message": "approval client rejected request"}
    assert len(unexpected) == 1
    assert unexpected[0]["method"] == "item/commandExecution/requestApproval"
    sent = _jsonl(capture)
    rejection = next(row for row in sent if row.get("id") == "approval-1")
    assert rejection["error"]["code"] == -32601
    assert "Unsupported server request" in rejection["error"]["message"]


def test_live_callback_failure_is_explicit_and_raw_stream_survives(tmp_path: Path) -> None:
    def broken_callback(message: dict, _ts: float) -> None:
        if message.get("method") == "item/started":
            raise RuntimeError("collector failed")

    driver, _capture, raw, _wire = _mock_driver(tmp_path, callback=broken_callback)
    with driver:
        _start_mock_thread(driver, tmp_path)
        with pytest.raises(CodexEventCallbackError, match="collector failed"):
            driver.run_turn("initial task", timeout_s=1.0)
    assert any(row.get("method") == "turn/completed" for row in _jsonl(raw))


@pytest.mark.skipif(not PINNED_CODEX.is_file(), reason="pinned Codex runtime is absent")
def test_real_pinned_codex_app_server_initialize_handshake(tmp_path: Path) -> None:
    """No model request: verify the installed 0.149.1 binary's real wire handshake."""

    codex_home = tmp_path / "codex-home"
    home = tmp_path / "home"
    codex_home.mkdir()
    home.mkdir()
    env = dict(os.environ)
    env.update(
        {
            "CODEX_HOME": str(codex_home),
            "HOME": str(home),
            "RUST_LOG": "error",
        }
    )
    raw = tmp_path / "real_raw.jsonl"
    wire = tmp_path / "real_wire.jsonl"
    stderr = tmp_path / "real_stderr.txt"
    with CodexAppServerDriver(
        command=build_codex_app_server_command(PINNED_CODEX),
        raw_jsonl_path=raw,
        wire_audit_path=wire,
        stderr_path=stderr,
        env=env,
        rpc_timeout_s=15.0,
        shutdown_timeout_s=1.0,
    ) as driver:
        assert driver.initialize_result is not None
        assert Path(driver.initialize_result["codexHome"]) == codex_home
        assert driver.initialize_result["platformOs"] == "linux"

    sent = [
        row["message"]
        for row in _jsonl(wire)
        if row["direction"] == "to_server"
    ]
    assert sent[0]["method"] == "initialize"
    assert sent[1] == {"method": "initialized"}
    assert all("jsonrpc" not in message for message in sent)
    assert _jsonl(raw)[0]["id"] == sent[0]["id"]
