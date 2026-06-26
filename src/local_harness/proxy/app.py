"""Starlette app: the proxy's HTTP surface.

POST /v1/chat/completions  — OpenAI clients (opencode, aider, Continue, ...)
POST /v1/messages          — Anthropic clients (Claude Code, ...)
GET  /v1/models            — forwarded upstream
GET  /health               — capability report of the upstream
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .anthropic import anthropic_sse, anthropic_to_openai, openai_sse, openai_to_anthropic
from .engine import ProxyEngine


def create_app(engine: ProxyEngine) -> Starlette:
    async def chat_completions(request: Request):
        body = await request.json()
        stream = bool(body.pop("stream", False))
        body.pop("stream_options", None)
        resp = await engine.handle_chat(body)
        if stream:
            return StreamingResponse(openai_sse(resp), media_type="text/event-stream")
        return JSONResponse(resp)

    async def messages(request: Request):
        abody = await request.json()
        stream = bool(abody.pop("stream", False))
        requested_model = abody.get("model", "")
        obody = anthropic_to_openai(abody)
        oresp = await engine.handle_chat(obody)
        aresp = openai_to_anthropic(oresp, requested_model)
        if stream:
            return StreamingResponse(anthropic_sse(aresp), media_type="text/event-stream")
        return JSONResponse(aresp)

    async def models(request: Request):
        return JSONResponse(await engine.forward_models())

    async def health(request: Request):
        return JSONResponse({
            "status": "ok",
            "upstream": engine.cfg.upstream_url,
            "model": engine.client.model,
            "capabilities": engine.caps.to_dict(),
        })

    @asynccontextmanager
    async def lifespan(app):
        await engine.start()
        yield

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/messages", messages, methods=["POST"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
