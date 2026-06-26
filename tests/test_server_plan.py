"""Server-side plan search: fan-out planning over HTTP.

The TUI in --server mode has no model client of its own, so plan-fork runs on the
server (which owns the live client+caps) via POST /session/plan. This locks that
endpoint: a task in, a best-first list of {text, score} candidates out — exercised
over ASGI against a scripted mock, on the sequential best-of-n path (no parallel_n).
"""

import httpx

from local_harness.server.app import create_server_app

from mocks import chat_response
from test_server import make_manager

# plan_search seeds the n forks at base_seed=100, 101, ... on the sequential
# (non-parallel_n) path; MeanLogprobVerifier scores from each fork's logprobs.
PLAN_SCRIPT = {
    100: chat_response(content="Plan A: do the safe thing."),
    101: chat_response(content="Plan B: do the bold thing."),
    102: chat_response(content="Plan C: do nothing."),
    103: chat_response(content="Plan D: ask first."),
}


async def test_http_plan_returns_ranked_candidates(tmp_path):
    mgr, _ = make_manager(tmp_path, script=PLAN_SCRIPT)
    app = create_server_app(mgr)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://srv") as c:
        r = await c.post("/session/plan", json={"task": "ship the feature", "n": 4})
        assert r.status_code == 200
        candidates = r.json()["candidates"]
        assert len(candidates) == 4
        assert all("text" in cand and "score" in cand for cand in candidates)
        assert all(cand["text"] for cand in candidates)
        # best-first: scores are non-increasing
        scores = [cand["score"] for cand in candidates]
        assert scores == sorted(scores, reverse=True)

        # missing task -> 400
        assert (await c.post("/session/plan", json={})).status_code == 400
