from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from experiments.baselines.lightllm.openai_compat_probe import (
    PROBE_TOKEN,
    stream_tool_probe,
    wait_for_models,
)


class _FakeResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines
        self.headers = {"Content-Type": "text/event-stream; charset=utf-8"}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self._lines)


def _data(payload: object) -> bytes:
    if payload == "[DONE]":
        return b"data: [DONE]\n\n"
    return ("data: " + json.dumps(payload) + "\n\n").encode("utf-8")


def _valid_lines() -> list[bytes]:
    return [
        _data(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "baseline_probe",
                                        "arguments": json.dumps({"token": PROBE_TOKEN}),
                                    }
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _data(
            {
                "choices": [
                    {"delta": {}, "finish_reason": "tool_calls"}
                ]
            }
        ),
        _data(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                },
            }
        ),
        _data("[DONE]"),
    ]


class StreamingProbeTests(unittest.TestCase):
    def _run(self, lines: list[bytes]) -> dict:
        with patch(
            "experiments.baselines.lightllm.openai_compat_probe.urllib.request.urlopen",
            return_value=_FakeResponse(lines),
        ):
            return stream_tool_probe(
                "http://127.0.0.1:8000/v1",
                model="test-model",
                api_key="EMPTY",
                timeout_s=1.0,
                max_tokens=32,
            )

    def test_requires_complete_openai_stream_contract(self) -> None:
        record = self._run(_valid_lines())
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["usage"]["total_tokens"], 28)
        self.assertIn("tool_calls", record["finish_reasons"])

    def test_rejects_missing_usage_done_or_tool_finish(self) -> None:
        cases = {
            "usage": _valid_lines()[:2] + _valid_lines()[3:],
            "done": _valid_lines()[:-1],
            "finish_reason": [
                _valid_lines()[0],
                _data({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                *_valid_lines()[2:],
            ],
        }
        for label, lines in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(RuntimeError):
                    self._run(lines)

    def test_readiness_requires_the_expected_model_id(self) -> None:
        responses = [
            {"object": "list", "data": []},
            {"object": "list", "data": [{"id": "test-model"}]},
        ]
        with patch(
            "experiments.baselines.lightllm.openai_compat_probe._get_json",
            side_effect=responses,
        ) as get_json:
            result = wait_for_models(
                "http://127.0.0.1:8000/v1",
                model="test-model",
                api_key="EMPTY",
                timeout_s=1.0,
                poll_interval_s=0.001,
            )
        self.assertEqual(result["data"][0]["id"], "test-model")
        self.assertEqual(get_json.call_count, 2)


if __name__ == "__main__":
    unittest.main()
