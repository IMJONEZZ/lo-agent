"""Sidecar teardown: run-state file, process stopping, and `lo lens down`.

Regression cover for the leak where `lo lens up` left an orphaned jlens-server
holding the model's RAM/VRAM and its port with no command to reap it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import pytest

from local_harness.jlens import cli, manager


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Never touch a real ~/.lo/jlens or a real lens on the default ports."""
    monkeypatch.setattr(manager, "LO_JLENS_HOME", tmp_path)
    monkeypatch.setattr(manager, "STATE_PATH", tmp_path / "state.json")


def _spawn(*, ignore_term: bool = False) -> subprocess.Popen:
    """A cheap child to stop. With ignore_term it survives SIGTERM."""
    code = ("import signal,time\n"
            + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
            + "time.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, "-c", code])
    time.sleep(0.5)  # let the SIGTERM handler install before we signal it
    return proc


def _args(**kw):
    return argparse.Namespace(**{"service_port": None, "sidecar_port": None,
                                 "timeout": 5.0, "force": False, **kw})


# ------------------------------------------------------------- state file --


def test_state_roundtrip_and_clear():
    manager.write_state(service_pid=1, sidecar_pid=2, sidecar_port=8091)
    assert manager.read_state()["sidecar_pid"] == 2
    manager.clear_state()
    assert manager.read_state() == {}


def test_read_state_tolerates_missing_and_corrupt(tmp_path):
    assert manager.read_state() == {}          # missing
    manager.STATE_PATH.write_text("{not json")
    assert manager.read_state() == {}          # corrupt -> not a crash


# -------------------------------------------------------- process control --


def test_pid_alive():
    proc = _spawn()
    assert manager.pid_alive(proc.pid)
    proc.kill()
    proc.wait()
    assert not manager.pid_alive(proc.pid)
    assert not manager.pid_alive(0)


def test_stop_pid_graceful():
    proc = _spawn()
    assert manager.stop_pid(proc.pid, timeout=5)
    assert not manager.pid_alive(proc.pid)
    proc.wait()


def test_stop_pid_escalates_to_sigkill():
    """The old atexit hook fired SIGTERM and exited; a child ignoring it lived on."""
    proc = _spawn(ignore_term=True)
    started = time.time()
    assert manager.stop_pid(proc.pid, timeout=1)
    assert not manager.pid_alive(proc.pid)
    assert time.time() - started >= 1  # waited out the grace period first
    proc.wait()


def test_stop_pid_on_dead_process_is_a_noop():
    proc = _spawn()
    proc.kill()
    proc.wait()
    assert manager.stop_pid(proc.pid, timeout=1)


def test_stop_sidecar_waits_for_the_child():
    proc = _spawn()
    assert manager.stop_sidecar(proc, timeout=5)
    assert proc.poll() is not None  # reaped, not just signalled


def test_stop_sidecar_kills_a_stubborn_child():
    proc = _spawn(ignore_term=True)
    assert manager.stop_sidecar(proc, timeout=1)
    assert proc.poll() is not None


# --------------------------------------------------------- `lo lens down` --


def test_down_stops_service_and_sidecar(capsys):
    svc, side = _spawn(), _spawn()
    manager.write_state(service_pid=svc.pid, service_port=19092,
                        sidecar_pid=side.pid, sidecar_port=19091, sidecar_owned=True)

    cli.cmd_down(_args())

    assert not manager.pid_alive(svc.pid)
    assert not manager.pid_alive(side.pid)
    assert manager.read_state() == {}  # state cleared so a later down is a no-op
    assert "lens is down" in capsys.readouterr().out
    svc.wait()
    side.wait()


def test_down_with_nothing_running(capsys, monkeypatch):
    monkeypatch.setattr(manager, "pid_on_port", lambda port: None)
    cli.cmd_down(_args())
    assert "nothing to stop" in capsys.readouterr().out


def test_down_finds_an_orphan_by_port_with_no_state(capsys, monkeypatch):
    """A run killed with SIGKILL leaves no state file — the port still finds it."""
    orphan = _spawn()
    monkeypatch.setattr(manager, "pid_on_port",
                        lambda port: orphan.pid if port == 8091 else None)

    cli.cmd_down(_args())

    assert not manager.pid_alive(orphan.pid)
    assert "sidecar" in capsys.readouterr().out
    orphan.wait()


def test_down_leaves_a_reused_sidecar_alone(capsys):
    """`up` reuses a sidecar it didn't start; `down` must not reap it."""
    side = _spawn()
    manager.write_state(service_pid=None, service_port=19092,
                        sidecar_pid=side.pid, sidecar_port=19091, sidecar_owned=False)

    cli.cmd_down(_args())

    assert manager.pid_alive(side.pid)
    out = capsys.readouterr().out
    assert "not started by" in out
    assert "nothing to stop" not in out  # don't contradict the line above
    side.kill()
    side.wait()


def test_down_force_stops_a_reused_sidecar():
    side = _spawn()
    manager.write_state(service_pid=None, service_port=19092,
                        sidecar_pid=side.pid, sidecar_port=19091, sidecar_owned=False)

    cli.cmd_down(_args(force=True))

    assert not manager.pid_alive(side.pid)
    side.wait()


def test_down_port_overrides_beat_recorded_state(capsys, monkeypatch):
    """--sidecar-port targets a box whose state file is stale or absent."""
    orphan = _spawn()
    manager.write_state(service_pid=None, service_port=19092,
                        sidecar_pid=None, sidecar_port=19091, sidecar_owned=True)
    monkeypatch.setattr(manager, "pid_on_port",
                        lambda port: orphan.pid if port == 12345 else None)

    args = _args()
    args.sidecar_port = 12345
    cli.cmd_down(args)

    assert not manager.pid_alive(orphan.pid)
    orphan.wait()
