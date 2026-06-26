"""The event bus: persistent (via log listener) vs ephemeral channels, catch-up,
many subscribers.

Locks the OpenCode-parity contract — many clients observe one session, a late
joiner sees the whole stream, and ephemeral token deltas never hit the log. The
agent loop writes only to the EventLog; the bus turns those writes into a live
broadcast with no agent-side changes.
"""

import asyncio

import pytest

from local_harness.events.bus import EventBus, TOKEN_DELTA, TERMINAL
from local_harness.events.log import EventLog, MODEL_CALL, RUN_COMPLETED, TOOL_CALL


@pytest.fixture
def bus(tmp_path):
    return EventBus(EventLog(tmp_path / "events.db"))


async def _drain(bus, run_id, stop_on, replay=True):
    return [ev async for ev in bus.subscribe(run_id, replay=replay, stop_on=stop_on)]


async def test_ephemeral_deltas_never_persist(bus):
    run_id = bus.create_run("t")
    bus.log.append(run_id, MODEL_CALL, {"call_index": 0})        # persisted
    bus.publish_delta(run_id, TOKEN_DELTA, {"text": "hel"})      # ephemeral
    bus.publish_delta(run_id, TOKEN_DELTA, {"text": "lo"})       # ephemeral
    bus.log.append(run_id, RUN_COMPLETED, {"answer": "hello"})   # persisted

    persisted = bus.log.events(run_id)
    types = [e.type for e in persisted]
    assert TOKEN_DELTA not in types
    assert types == ["run_started", MODEL_CALL, RUN_COMPLETED]
    assert all(e.seq >= 0 for e in persisted)


async def test_live_subscriber_gets_persistent_and_ephemeral(bus):
    run_id = bus.create_run("t")

    async def producer():
        await asyncio.sleep(0.01)
        bus.publish_delta(run_id, TOKEN_DELTA, {"text": "hi"})
        bus.log.append(run_id, MODEL_CALL, {"call_index": 0})
        bus.log.append(run_id, RUN_COMPLETED, {"answer": "hi"})

    asyncio.create_task(producer())
    got = await _drain(bus, run_id, stop_on=TERMINAL)
    types = [e.type for e in got]
    assert "run_started" in types          # catch-up
    assert TOKEN_DELTA in types            # live ephemeral
    assert MODEL_CALL in types
    assert types[-1] == RUN_COMPLETED      # stopped on terminal


async def test_late_joiner_gets_full_catchup(bus):
    run_id = bus.create_run("t")
    bus.log.append(run_id, MODEL_CALL, {"call_index": 0})
    bus.log.append(run_id, TOOL_CALL, {"name": "web_search"})
    bus.log.append(run_id, RUN_COMPLETED, {"answer": "done"})
    got = await _drain(bus, run_id, stop_on=TERMINAL)
    assert [e.type for e in got] == ["run_started", MODEL_CALL, TOOL_CALL, RUN_COMPLETED]


async def test_two_subscribers_one_session(bus):
    run_id = bus.create_run("t")

    async def producer():
        await asyncio.sleep(0.01)
        bus.log.append(run_id, MODEL_CALL, {"call_index": 0})
        bus.log.append(run_id, RUN_COMPLETED, {"answer": "x"})

    asyncio.create_task(producer())
    a, b = await asyncio.gather(
        _drain(bus, run_id, stop_on=TERMINAL),
        _drain(bus, run_id, stop_on=TERMINAL),
    )
    assert [e.type for e in a] == [e.type for e in b]
    assert a[-1].type == RUN_COMPLETED
    assert bus.subscriber_count(run_id) == 0   # both cleaned up on exit


async def test_no_catchup_duplication_when_event_lands_mid_subscribe(bus):
    """Events published between catch-up read and live tailing must not double."""
    run_id = bus.create_run("t")
    bus.log.append(run_id, MODEL_CALL, {"call_index": 0})
    seen_seqs = []

    async def consume():
        async for ev in bus.subscribe(run_id, stop_on=TERMINAL):
            if ev.seq >= 0:
                seen_seqs.append(ev.seq)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    bus.log.append(run_id, TOOL_CALL, {"name": "f"})
    bus.log.append(run_id, RUN_COMPLETED, {"answer": "x"})
    await task
    assert seen_seqs == sorted(set(seen_seqs))   # strictly increasing, no dupes


async def test_agent_writing_to_shared_log_broadcasts(bus):
    """The integration contract: an agent that holds the SAME EventLog and only
    ever calls log methods still drives live subscribers — no bus awareness."""
    run_id = bus.create_run("t")
    received = []

    async def consume():
        async for ev in bus.subscribe(run_id, replay=False, stop_on=TERMINAL):
            received.append(ev.type)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    # simulate the agent loop writing through its log handle (same instance)
    bus.log.append(run_id, MODEL_CALL, {"call_index": 0})
    bus.log.append(run_id, TOOL_CALL, {"name": "calc"})
    bus.log.append(run_id, RUN_COMPLETED, {"answer": "42"})
    await task
    assert received == [MODEL_CALL, TOOL_CALL, RUN_COMPLETED]
