"""Starlette HTTP surface for the session server.

  GET    /                                           -> the web client (single-page app)
  POST   /session                 {task, preset?}   -> {run_id}
  POST   /session/plan             {task, n?}        -> {candidates: [{text, score}]}
  POST   /session/{id}/message     {content, preset?}-> {run_id}
  POST   /session/{id}/interrupt                     -> {interrupted: bool}
  POST   /session/{id}/permission  {request_id, approved} -> {resolved: bool}
  GET    /session/{id}/events      [?replay=0]       -> SSE stream of events
  GET    /sessions                                   -> [{run_id, task, status, ...}]
  DELETE /session/{id}                               -> {deleted: true}
  GET    /health                                     -> capability report

The events stream is the OpenCode-style bus: every client that opens it observes
the same live session (catch-up from seq 0, then live), and the SSE `event:` field
carries the harness event type so a thin client can render without guessing.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from ..events.bus import TERMINAL
from .sessions import SessionManager
from .webui import index_html


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def create_server_app(
    manager: SessionManager, *,
    health: dict | Callable[[], dict] | None = None,
    on_startup: Callable[[], Awaitable[None]] | None = None,
) -> Starlette:
    async def start_session(request: Request):
        body = await request.json()
        task = body.get("task", "")
        if not task:
            return JSONResponse({"error": "missing task"}, status_code=400)
        return JSONResponse({"run_id": manager.start(
            task, preset=body.get("preset"), code_mode=body.get("code_mode"))})

    async def send_message(request: Request):
        run_id = request.path_params["run_id"]
        body = await request.json()
        try:
            manager.send(run_id, body.get("content", ""), preset=body.get("preset"),
                         code_mode=body.get("code_mode"))
        except KeyError:
            return JSONResponse({"error": "unknown run"}, status_code=404)
        return JSONResponse({"run_id": run_id})

    async def plan_route(request: Request):
        body = await request.json()
        task = body.get("task", "")
        if not task:
            return JSONResponse({"error": "missing task"}, status_code=400)
        candidates = await manager.plan(task, n=body.get("n", 4))
        return JSONResponse({"candidates": candidates})

    async def interrupt(request: Request):
        run_id = request.path_params["run_id"]
        return JSONResponse({"interrupted": manager.interrupt(run_id)})

    async def permission(request: Request):
        body = await request.json()
        request_id = body.get("request_id", "")
        resolved = manager.resolve_permission(request_id, bool(body.get("approved")))
        return JSONResponse({"resolved": resolved})

    async def events(request: Request):
        run_id = request.path_params["run_id"]
        replay = request.query_params.get("replay", "1") != "0"
        # Stop the stream after a terminal event only if the client asked for a
        # one-shot follow; default keeps it open so follow-up turns keep streaming.
        stop_on = TERMINAL if request.query_params.get("once") == "1" else None

        async def gen():
            async for ev in manager.stream(run_id, replay=replay, stop_on=stop_on):
                yield _sse(ev.type, {"seq": ev.seq, "payload": ev.payload,
                                     "created_at": ev.created_at})

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def sessions(request: Request):
        return JSONResponse(manager.sessions())

    async def delete_session(request: Request):
        manager.delete(request.path_params["run_id"])
        return JSONResponse({"deleted": True})

    async def health_route(request: Request):
        payload = health() if callable(health) else (health or {"status": "ok"})
        return JSONResponse(payload)

    async def index(request: Request):
        return HTMLResponse(index_html())

    @asynccontextmanager
    async def lifespan(app):
        if on_startup is not None:
            await on_startup()
        yield

    return Starlette(
        routes=[
            Route("/", index, methods=["GET"]),
            Route("/session", start_session, methods=["POST"]),
            Route("/session/plan", plan_route, methods=["POST"]),
            Route("/session/{run_id}/message", send_message, methods=["POST"]),
            Route("/session/{run_id}/interrupt", interrupt, methods=["POST"]),
            Route("/session/{run_id}/permission", permission, methods=["POST"]),
            Route("/session/{run_id}/events", events, methods=["GET"]),
            Route("/session/{run_id}", delete_session, methods=["DELETE"]),
            Route("/sessions", sessions, methods=["GET"]),
            Route("/health", health_route, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
