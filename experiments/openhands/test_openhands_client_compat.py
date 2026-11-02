from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
import unittest


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common import OPENHANDS_FORCE_STRING_SERIALIZER
from openhands_runner import OpenHandsConfig, _build_llm


class _FakeLLM:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class OpenHandsClientCompatibilityTests(unittest.TestCase):

    def test_shared_client_forces_text_string_serialization(self) -> None:
        config = OpenHandsConfig(
            model="Qwen3-Coder-30B-A3B-Instruct",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=128,
            llm_timeout_s=30,
            llm_num_retries=1,
            max_iterations=2,
            terminal_no_change_timeout_s=30,
            terminal_type="subprocess",
        )
        llm = _build_llm(
            config,
            {
                "LLM": _FakeLLM,
                "SecretStr": lambda value: value,
            },
        )

        self.assertTrue(OPENHANDS_FORCE_STRING_SERIALIZER)
        self.assertIs(llm.kwargs["force_string_serializer"], True)
        self.assertEqual(
            llm.kwargs["model"],
            "openai/Qwen3-Coder-30B-A3B-Instruct",
        )
        self.assertIs(llm.kwargs["stream"], True)

    def test_sdk_string_serializer_preserves_tool_threading(self) -> None:
        try:
            from openhands.sdk.llm import Message, TextContent
            from openhands.sdk.llm.message import MessageToolCall
        except ModuleNotFoundError:
            self.skipTest("OpenHands SDK is installed only in the worker venv")

        format_kwargs = {
            "cache_enabled": False,
            "vision_enabled": False,
            "function_calling_enabled": True,
            "force_string_serializer": True,
            "send_reasoning_content": False,
        }
        user = Message(
            role="user",
            content=[TextContent(text="one"), TextContent(text="two")],
        ).to_chat_dict(**format_kwargs)
        assistant = Message(
            role="assistant",
            content=[],
            tool_calls=[
                MessageToolCall(
                    id="call_1",
                    name="terminal",
                    arguments='{"command":"pwd"}',
                    origin="completion",
                )
            ],
        ).to_chat_dict(**format_kwargs)
        tool = Message(
            role="tool",
            content=[TextContent(text="ok")],
            tool_call_id="call_1",
            name="terminal",
        ).to_chat_dict(**format_kwargs)

        self.assertEqual(user["content"], "one\ntwo")
        self.assertNotIn("content", assistant)
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_1")
        self.assertEqual(tool["content"], "ok")
        self.assertEqual(tool["tool_call_id"], "call_1")
        self.assertEqual(tool["name"], "terminal")


if __name__ == "__main__":
    unittest.main()
