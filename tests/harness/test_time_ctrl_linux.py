"""Real-process tests for time_ctrl: a frozen tree must not age its clocks or timers."""

from __future__ import annotations

import os
import re
import sys
import time

import pytest

from agency.configs.agconfig import agconfig, ptraceconfig
from agency.harness.ptrace.supervisor import agProxyPtrace, ptrace_available

pytestmark = [
    pytest.mark.skipif(not ptrace_available(), reason="ptrace not usable on this host"),
    pytest.mark.timeout(120),
]

_PRELUDE = """
import ctypes, os, select, signal, socket, threading, time
libc = ctypes.CDLL(None)
class Timespec(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_long), ("nanoseconds", ctypes.c_long)]
class Itimerspec(ctypes.Structure):
    _fields_ = [("interval", Timespec), ("value", Timespec)]
def timespec(seconds):
    return Timespec(int(seconds), int((seconds % 1) * 1e9))
def new_timerfd(value, interval=0.0, absolute=False):
    fd = libc.timerfd_create(1, 0)
    libc.timerfd_settime(fd, int(absolute), ctypes.byref(Itimerspec(timespec(interval), timespec(value))), None)
    return fd
print("ready", flush=True)
start = time.monotonic()
"""
_EPILOGUE = "\nprint('perceived=%.2f' % (time.monotonic() - start), flush=True)\n"
_THREADS_PROGRAM = """
errors = []
def keep_waiting(kind):
    for _ in range(6):
        before = time.monotonic()
        {
            "sleep": lambda: time.sleep(0.4),
            "event": lambda: threading.Event().wait(0.4),
            "select": lambda: select.select([], [], [], 0.4),
            "timerfd": lambda: os.read(new_timerfd(0.4), 8),
        }[kind]()
        if abs(time.monotonic() - before - 0.4) > 0.3:
            errors.append(kind)
threads = [threading.Thread(target=keep_waiting, args=(kind,)) for kind in ("sleep", "event", "select", "timerfd")]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print("errors=%d" % len(errors), errors)
"""

# name -> (program body, perceived seconds the program should report)
_PROGRAMS = {
    "sleep": ("time.sleep(3)", 3.0),
    "event_wait": ("threading.Event().wait(3)", 3.0),
    "select": ("select.select([], [], [], 3)", 3.0),
    "socket_timeout": (
        """
server = socket.socket(); server.bind(("127.0.0.1", 0)); server.listen(1)
client = socket.create_connection(server.getsockname()); client.settimeout(3)
try:
    client.recv(1)
except socket.timeout:
    pass
""",
        3.0,
    ),
    "interval_timer": (
        """
fired = []
signal.signal(signal.SIGALRM, lambda *_: fired.append(1))
signal.setitimer(signal.ITIMER_REAL, 3.0)
while not fired:
    signal.pause()
""",
        3.0,
    ),
    "timerfd_relative": ("os.read(new_timerfd(3.0), 8)", 3.0),
    "timerfd_absolute": ("os.read(new_timerfd(time.monotonic() + 3.0, absolute=True), 8)", 3.0),
    "timerfd_periodic": (
        "fd = new_timerfd(1.0, interval=1.0)\nfor _ in range(5):\n    os.read(fd, 8)",
        5.0,
    ),
    "timerfd_expired_before_freeze": (
        "fd = new_timerfd(0.9)\ntime.sleep(1.6)\nos.read(fd, 8)",
        1.6,
    ),
    "timerfd_under_signal_storm": (
        """
signal.signal(signal.SIGALRM, lambda *_: None)
signal.setitimer(signal.ITIMER_REAL, 0.003, 0.003)
result = []
reader = threading.Thread(target=lambda: result.append(os.read(new_timerfd(3.0), 8)))
reader.start(); reader.join()
signal.setitimer(signal.ITIMER_REAL, 0)
""",
        3.0,
    ),
}


def _run_with_freeze(
    program: str, *, time_control: bool = True, time_realtime: bool = False
) -> "tuple[str, float]":
    """Run *program*, freeze its tree 1.5s in for 4s; return (its stdout, wall seconds)."""
    handle = agProxyPtrace(
        agconfig(ptraceconfig(time_control=time_control, time_realtime=time_realtime)),
        allow_initial_exec=True,
    ).launch(
        [sys.executable, "-u", "-c", _PRELUDE + program + _EPILOGUE],
        dict(os.environ),
        policy=type("AllowAll", (), {"check": lambda self, _agent, _event: True})(),
    )
    try:
        deadline = time.monotonic() + 20
        while "ready" not in handle.terminal_output():
            assert time.monotonic() < deadline, "program never started"
            time.sleep(0.05)
        started = time.monotonic()
        time.sleep(1.5)
        handle.pause()
        assert handle.wait_paused(15)
        time.sleep(4)
        handle.resume()
        stdout, stderr, returncode = handle.wait(timeout=60)
        wall_seconds = time.monotonic() - started
    finally:
        handle.close()
    assert returncode == 0, stderr
    return stdout, wall_seconds


@pytest.mark.parametrize("name", list(_PROGRAMS))
def test_frozen_time_is_not_counted(name):
    program, expected_seconds = _PROGRAMS[name]
    stdout, wall_seconds = _run_with_freeze(program + _EPILOGUE)
    perceived_seconds = float(re.search(r"perceived=([\d.]+)", stdout)[1])
    assert abs(perceived_seconds - expected_seconds) < 0.6
    assert wall_seconds > expected_seconds + 3  # the freeze really happened


def test_without_time_control_a_freeze_shortens_the_wait():
    stdout, _ = _run_with_freeze(_PROGRAMS["sleep"][0] + _EPILOGUE, time_control=False)
    assert float(re.search(r"perceived=([\d.]+)", stdout)[1]) > 4.5


@pytest.mark.parametrize("time_realtime, frozen_wall_clock", [(True, True), (False, False)])
def test_wall_clock_follows_the_freeze_only_when_enabled(time_realtime, frozen_wall_clock):
    program = "wall_start = time.time()\ntime.sleep(3)\nprint('wall_clock=%.2f' % (time.time() - wall_start))"
    stdout, _ = _run_with_freeze(program, time_realtime=time_realtime)
    wall_clock_seconds = float(re.search(r"wall_clock=([\d.]+)", stdout)[1])
    assert (wall_clock_seconds < 3.6) == frozen_wall_clock


def test_threads_with_mixed_timed_waits_all_keep_virtual_time():
    stdout, _ = _run_with_freeze(_THREADS_PROGRAM)
    assert "errors=0" in stdout, stdout
