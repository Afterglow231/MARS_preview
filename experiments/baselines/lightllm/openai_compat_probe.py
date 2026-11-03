#!/usr/bin/env python3
"""Wait for an OpenAI endpoint and exercise streaming tool calling.

The generated requests are synthetic warmups.  They are written separately
from benchmark traces and must be run identically for every compared backend.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request


PROBE_TOKEN = "mars-openai-tool-probe-v1"


def _validated_usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(
            "Streaming response did not include the requested final usage object"
        )

    counts: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        count = value.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise RuntimeError(f"Streaming usage has invalid {key}: {count!r}")
        counts[key] = count
    if counts["prompt_tokens"] <= 0 or counts["completion_tokens"] <= 0:
        raise RuntimeError(f"Streaming usage contains no token work: {counts!r}")
    if counts["total_tokens"] != counts["prompt_tokens"] + counts["completion_tokens"]:
        raise RuntimeError(f"Streaming usage token totals are inconsistent: {counts!r}")
    return value


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _get_json(url: str, *, api_key: str, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return data


def wait_for_models(
    base_url: str,
    *,
    model: str,
    api_key: str,
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            payload = _get_json(_models_url(base_url), api_key=api_key, timeout_s=3.0)
            model_ids = [
                str(item.get("id"))
                for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id") is not None
            ]
            if model not in model_ids:
                raise RuntimeError(
                    f"Endpoint serves {model_ids}, expected model {model!r}"
                )
            return payload
        except Exception as exc:  # readiness errors are expected while loading
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(max(0.05, poll_interval_s))
    raise TimeoutError(
        f"Timed out after {timeout_s:.1f}s waiting for {_models_url(base_url)}; "
        f"last error: {last_error}"
    )


def _probe_payload(model: str, *, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "This is an API protocol check. Call baseline_probe exactly once "
                    f"with token={PROBE_TOKEN!r}. Do not answer in plain text."
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "baseline_probe",
                    "description": "Return the supplied protocol-check token.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token": {"type": "string", "enum": [PROBE_TOKEN]}
                        },
                        "required": ["token"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "baseline_probe"},
        },
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        # MARS sends this extension on real OpenHands calls.  LightLLM should
        # ignore it through its upstream OpenAI request model, not through a proxy.
        "job_id": "mars_protocol_probe",
    }


def stream_tool_probe(
    base_url: str,
    *,
    model: str,
    api_key: str,
    timeout_s: float,
    max_tokens: int,
) -> dict[str, Any]:
    body = json.dumps(_probe_payload(model, max_tokens=max_tokens)).encode("utf-8")
    request = urllib.request.Request(
        _chat_url(base_url),
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    started = time.time()
    first_chunk_s: float | None = None
    names: list[str] = []
    argument_fragments: list[str] = []
    finish_reasons: list[str] = []
    usage: dict[str, Any] | None = None
    chunks = 0
    saw_done = False

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type.lower():
                raise RuntimeError(f"Expected SSE response, got Content-Type={content_type!r}")
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data_text = line[5:].strip()
                if data_text == "[DONE]":
                    saw_done = True
                    break
                chunk = json.loads(data_text)
                if first_chunk_s is None:
                    first_chunk_s = time.time() - started
                chunks += 1
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    if not isinstance(choice, dict):
                        continue
                    reason = choice.get("finish_reason")
                    if reason:
                        finish_reasons.append(str(reason))
                    delta = choice.get("delta") or {}
                    for tool_call in delta.get("tool_calls") or []:
                        if not isinstance(tool_call, dict):
                            continue
                        function = tool_call.get("function") or {}
                        name = function.get("name")
                        if name:
                            names.append(str(name))
                        arguments = function.get("arguments")
                        if arguments is not None:
                            argument_fragments.append(str(arguments))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Tool probe HTTP {exc.code}: {detail[:2000]}") from exc

    arguments_text = "".join(argument_fragments)
    if "baseline_probe" not in names:
        raise RuntimeError(
            "Streaming response did not contain a baseline_probe tool call "
            f"(names={names}, finish_reasons={finish_reasons}, chunks={chunks})"
        )
    try:
        arguments = json.loads(arguments_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Tool arguments are not valid JSON: {arguments_text!r}") from exc
    if not isinstance(arguments, dict) or arguments.get("token") != PROBE_TOKEN:
        raise RuntimeError(f"Unexpected tool arguments: {arguments!r}")
    if "tool_calls" not in finish_reasons:
        raise RuntimeError(
            "Streaming tool call did not finish with finish_reason='tool_calls' "
            f"(finish_reasons={finish_reasons})"
        )
    if not saw_done:
        raise RuntimeError("Streaming response ended without the required [DONE] sentinel")
    usage = _validated_usage(usage)

    return {
        "status": "ok",
        "started_at_s": started,
        "duration_s": time.time() - started,
        "first_chunk_s": first_chunk_s,
        "chunks": chunks,
        "tool_names": names,
        "tool_arguments": arguments,
        "finish_reasons": finish_reasons,
        "usage": usage,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--wait-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.5)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.requests < 0:
        raise SystemExit("--requests must be >= 0")
    wait_for_models(
        args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout_s=args.wait_timeout_s,
        poll_interval_s=args.poll_interval_s,
    )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index in range(args.requests):
            record = stream_tool_probe(
                args.base_url,
                model=args.model,
                api_key=args.api_key,
                timeout_s=args.request_timeout_s,
                max_tokens=args.max_tokens,
            )
            record["warmup_index"] = index
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    print(output)


if __name__ == "__main__":
    main()
