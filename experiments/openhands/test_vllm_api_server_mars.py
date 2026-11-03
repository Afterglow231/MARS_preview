from __future__ import annotations

import asyncio

from vllm_api_server_mars import _engine_utility_rpc


def test_engine_utility_rpc_supports_v1_engine_core() -> None:
    class EngineCore:
        async def call_utility_async(self, method: str):
            return {"method": method}

    class AsyncEngine:
        engine_core = EngineCore()

    call = _engine_utility_rpc(AsyncEngine())

    assert call is not None
    assert asyncio.run(call("get_kv_cache_stats")) == {
        "method": "get_kv_cache_stats"
    }


def test_engine_utility_rpc_preserves_direct_v0_interface() -> None:
    class LegacyEngine:
        async def call_utility_async(self, method: str):
            return method

    call = _engine_utility_rpc(LegacyEngine())

    assert call is not None
    assert asyncio.run(call("get_kv_cache_stats")) == "get_kv_cache_stats"
