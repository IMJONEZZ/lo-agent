from local_harness.events.log import MODEL_CALL, RUN_COMPLETED, RUN_STARTED, EventLog


def test_append_and_ordering(tmp_path):
    log = EventLog(tmp_path / "e.db")
    run_id = log.create_run("do a thing")

    log.append(run_id, MODEL_CALL, {"call_index": 0})
    log.append(run_id, MODEL_CALL, {"call_index": 1})
    log.append(run_id, RUN_COMPLETED, {"answer": "done"})

    events = log.events(run_id)
    assert [e.type for e in events] == [RUN_STARTED, MODEL_CALL, MODEL_CALL, RUN_COMPLETED]
    assert [e.seq for e in events] == [0, 1, 2, 3]

    only_calls = log.events(run_id, type=MODEL_CALL)
    assert [e.payload["call_index"] for e in only_calls] == [0, 1]


def test_delete_run_removes_it_and_its_events(tmp_path):
    log = EventLog(tmp_path / "e.db")
    keep = log.create_run("keep")
    drop = log.create_run("drop")
    log.append(drop, MODEL_CALL, {"call_index": 0})

    log.delete_run(drop)

    assert [r.run_id for r in log.runs()] == [keep]  # only the survivor remains
    assert log.run(drop) is None
    assert log.events(drop) == []
    assert log.events(keep)  # untouched


def test_run_status_transitions(tmp_path):
    log = EventLog(tmp_path / "e.db")
    run_id = log.create_run("task")
    assert log.run(run_id).status == "running"
    log.append(run_id, RUN_COMPLETED, {"answer": "x"})
    assert log.run(run_id).status == "completed"


def test_runs_are_isolated(tmp_path):
    log = EventLog(tmp_path / "e.db")
    a = log.create_run("a")
    b = log.create_run("b")
    log.append(a, MODEL_CALL, {"call_index": 0})
    assert len(log.events(a)) == 2
    assert len(log.events(b)) == 1
    assert {r.run_id for r in log.runs()} == {a, b}


def test_persistence_across_connections(tmp_path):
    path = tmp_path / "e.db"
    log = EventLog(path)
    run_id = log.create_run("persist me")
    log.append(run_id, MODEL_CALL, {"call_index": 0})
    log.close()

    reopened = EventLog(path)
    assert reopened.run(run_id).task == "persist me"
    assert len(reopened.events(run_id)) == 2


def test_rewind_points_lists_answer_and_followups(tmp_path):
    from local_harness.events.log import USER_MESSAGE
    log = EventLog(tmp_path / "e.db")
    rid = log.create_run("first task")
    log.append(rid, MODEL_CALL, {"call_index": 0})
    log.append(rid, RUN_COMPLETED, {"answer": "a1"})
    log.append(rid, USER_MESSAGE, {"content": "follow up please"})
    log.append(rid, MODEL_CALL, {"call_index": 1})
    log.append(rid, RUN_COMPLETED, {"answer": "a2"})

    pts = log.rewind_points(rid)
    kinds = [k for _, k, _ in pts]
    assert kinds == ["answer", "follow-up"]
    # the answer point is the first MODEL_CALL; the follow-up point is the USER_MESSAGE
    assert pts[0][0] == 1 and pts[1][0] == 3
    assert "follow up" in pts[1][2]


def test_rewind_archives_tail_and_truncates(tmp_path):
    from local_harness.events.log import USER_MESSAGE
    log = EventLog(tmp_path / "e.db")
    rid = log.create_run("task")
    log.append(rid, MODEL_CALL, {"call_index": 0})
    log.append(rid, RUN_COMPLETED, {"answer": "a1"})
    log.append(rid, USER_MESSAGE, {"content": "again"})
    log.append(rid, MODEL_CALL, {"call_index": 1})
    log.append(rid, RUN_COMPLETED, {"answer": "a2"})
    before = len(log.events(rid))

    # rewind to before the follow-up (seq 3) — removes 3 events (USER_MESSAGE + call + completed)
    archive_id = log.rewind(rid, 3)
    assert archive_id is not None
    remaining = log.events(rid)
    assert len(remaining) == before - 3
    assert remaining[-1].type == RUN_COMPLETED and remaining[-1].payload["answer"] == "a1"

    # the removed tail is preserved losslessly in the archive run
    archived = log.events(archive_id)
    assert any(e.type == USER_MESSAGE for e in archived)
    assert any(e.payload.get("answer") == "a2" for e in archived if e.type == RUN_COMPLETED)

    # rewinding past the end is a no-op
    assert log.rewind(rid, 999) is None
