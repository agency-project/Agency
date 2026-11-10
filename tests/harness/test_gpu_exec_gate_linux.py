"""Real-process tests for the exec gate: an exec that needs GPUs waits with the tree frozen."""

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

_PARENT_PROGRAM = """
import subprocess, sys, time
start = time.monotonic()
try:
    child = subprocess.run(
        [sys.executable, "-c", "import os; print('gpu-child', os.environ.get('CUDA_VISIBLE_DEVICES'))"],
        capture_output=True, text=True,
    )
    print(child.stdout.strip())
except PermissionError:
    print("denied")
print("perceived=%.2f" % (time.monotonic() - start))
"""


class _GpuPolicy:
    """Allows everything; the exec whose script mentions gpu-child gets *gpu_state*."""

    def __init__(self, gpu_state, wait_result=None, wait_seconds=0.0):
        self.gpu_state = gpu_state
        self.wait_result = wait_result
        self.wait_seconds = wait_seconds
        self.waited_pids = []
        self.exited_holder_pids = []

    def check(self, _agent, event):
        if event.syscall == "execve" and any("gpu-child" in part for part in event.argv or []):
            env_overrides = {"CUDA_VISIBLE_DEVICES": "0"} if self.gpu_state == "granted" else None
            return True, None, None, env_overrides, self.gpu_state
        return True

    def wait_for_gpus(self, pid):
        self.waited_pids.append(pid)
        time.sleep(self.wait_seconds)
        return self.wait_result

    def gpu_holder_exited(self, pid):
        self.exited_holder_pids.append(pid)


def _run(policy, *, time_control: bool = True) -> "tuple[str, float]":
    handle = agProxyPtrace(
        agconfig(ptraceconfig(time_control=time_control)), allow_initial_exec=True
    ).launch([sys.executable, "-u", "-c", _PARENT_PROGRAM], dict(os.environ), policy=policy)
    started = time.monotonic()
    try:
        stdout, stderr, returncode = handle.wait(timeout=60)
    finally:
        handle.close()
    assert returncode == 0, stderr
    return stdout, time.monotonic() - started


def test_exec_waiting_for_gpus_freezes_the_tree_and_gets_the_environment_on_grant():
    policy = _GpuPolicy("pending", {"CUDA_VISIBLE_DEVICES": "1"}, wait_seconds=3.0)
    stdout, wall_seconds = _run(policy)
    assert "gpu-child 1" in stdout
    assert float(re.search(r"perceived=([\d.]+)", stdout)[1]) < 2.5  # the 3s wait is not counted
    assert wall_seconds > 3
    assert len(policy.waited_pids) == 1
    assert policy.exited_holder_pids == policy.waited_pids


def test_exec_with_a_free_gpu_proceeds_without_waiting_and_is_reported_when_it_exits():
    policy = _GpuPolicy("granted")
    stdout, _ = _run(policy)
    assert "gpu-child 0" in stdout
    assert policy.waited_pids == []
    assert len(policy.exited_holder_pids) == 1


def test_exec_is_denied_when_the_wait_for_gpus_is_cancelled():
    policy = _GpuPolicy("pending", None)
    stdout, _ = _run(policy)
    assert "denied" in stdout
    assert policy.exited_holder_pids == []


def test_exec_gate_without_time_control_still_waits_and_gets_the_environment():
    policy = _GpuPolicy("pending", {"CUDA_VISIBLE_DEVICES": "1"}, wait_seconds=3.0)
    stdout, _ = _run(policy, time_control=False)
    assert "gpu-child 1" in stdout
    assert float(re.search(r"perceived=([\d.]+)", stdout)[1]) > 2.9  # nothing hides the wait
