#!/usr/bin/env python3
from __future__ import annotations

from http import HTTPStatus

import uvloop
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai import api_server as api
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils import FlexibleArgumentParser


def _ensure_kv_cache_stats_route(app) -> None:
    for route in app.routes:
        if getattr(route, "path", None) == "/v1/kv_cache_stats":
            return

    async def kv_cache_stats(raw_request: Request) -> JSONResponse:
        client = api.engine_client(raw_request)
        call = getattr(client, "call_utility_async", None)
        if not callable(call):
            raise HTTPException(
                status_code=HTTPStatus.NOT_IMPLEMENTED,
                detail="Engine client does not support utility RPCs.",
            )
        try:
            stats = await call("get_kv_cache_stats")
        except Exception as exc:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch kv cache stats: {type(exc).__name__}: {exc}",
            ) from exc
        if not isinstance(stats, dict):
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Unexpected kv cache stats payload.",
            )
        return JSONResponse(content=stats)

    app.add_api_route("/v1/kv_cache_stats", kv_cache_stats, methods=["GET"])


_orig_build_app = api.build_app


def _build_app_with_kv_route(args):
    app = _orig_build_app(args)
    _ensure_kv_cache_stats_route(app)
    return app


api.build_app = _build_app_with_kv_route


if __name__ == "__main__":
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server with mars KV stats route."
    )
    parser = api.make_arg_parser(parser)
    args = parser.parse_args()
    api.validate_parsed_serve_args(args)
    uvloop.run(api.run_server(args))
