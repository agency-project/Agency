"""The real-GPU lease bookkeeping every backend inherits from
``agsandbox_backend`` (agency/sandbox/base.py): holders, release when idle,
release by holder, and the facade's ``acquire_real_gpus_for``.

Nothing here is specific to a backend, so each test runs for chroot, docker and
podman (``backend_kind``, never skipped: no daemon is touched). The per-backend
*gating* of the GPU release on stop()/rm_container()/destroy() differs and is
tested in the family files (test_container_backends.py, test_chroot_backend.py).
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from agency.sandbox import base as sandbox_base
from agency.sandbox.agsandbox import agSandbox
from tests.sandbox.conftest import make_backend


@pytest.fixture(autouse=True)
def _kind(backend_kind):
    """Expose the parametrized backend kind to the module-level ``_backend``."""
    global _KIND
    _KIND = backend_kind


_KIND = "docker"


def _backend():
    return make_backend(_KIND, agname="agent", name="real-gpu-lease-test")


def _reserved(backend, gpu_ids=None):
    released = []
    backend._virtual_gpu_count_requested = 1
    backend._real_gpu_acquire_fn = lambda count: [0]
    backend._real_gpu_release_fn = lambda ids: released.append(list(ids))
    backend._real_gpu_ids = list(gpu_ids or [])
    return released


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TestHoldRealGpus:
    def test_returns_none_and_acquires_nothing_without_a_reservation(self):
        backend = _backend()
        assert backend._hold_real_gpus() is None
        assert backend._real_gpu_ids == []
        assert backend._real_gpu_holders == set()

    def test_acquires_and_registers_a_holder_when_reserved(self):
        backend = _backend()
        _reserved(backend)
        holder = backend._hold_real_gpus()
        assert holder is not None
        assert backend._real_gpu_ids == [0]
        assert backend._real_gpu_holders == {holder}

    def test_second_holder_reuses_the_held_gpus(self):
        backend = _backend()
        calls = []
        _reserved(backend)
        backend._real_gpu_acquire_fn = lambda count: calls.append(count) or [0]
        first = backend._hold_real_gpus()
        second = backend._hold_real_gpus()
        assert calls == [1]
        assert backend._real_gpu_holders == {first, second}


class TestReleaseRealGpuWhenIdle:
    def test_releases_when_the_last_holder_ends_and_nothing_runs_in_background(self):
        backend = _backend()
        released = _reserved(backend)
        holder = backend._hold_real_gpus()
        with patch.object(backend, "get_live_pids", return_value=set()):
            backend.release_real_gpu_when_idle(holder)
        assert released == [[0]]
        assert backend._real_gpu_ids == []
        assert backend._virtual_gpu_count_requested == 1

    def test_keeps_the_gpus_while_another_holder_is_in_flight(self):
        backend = _backend()
        released = _reserved(backend)
        first = backend._hold_real_gpus()
        second = backend._hold_real_gpus()
        with patch.object(backend, "get_live_pids", return_value=set()):
            backend.release_real_gpu_when_idle(first)
            assert released == []
            assert backend._real_gpu_ids == [0]
            backend.release_real_gpu_when_idle(second)
        assert released == [[0]]
        assert backend._real_gpu_ids == []

    def test_unknown_holder_does_not_release_gpus_held_by_others(self):
        backend = _backend()
        released = _reserved(backend)
        backend._hold_real_gpus()
        with patch.object(backend, "get_live_pids", return_value=set()):
            backend.release_real_gpu_when_idle("someone-else")
        assert released == []
        assert backend._real_gpu_ids == [0]

    def test_is_a_noop_when_nothing_is_held(self):
        backend = _backend()
        released = _reserved(backend)
        backend.release_real_gpu_when_idle("x")
        assert released == []

    def test_background_process_keeps_the_gpus_until_it_exits(self):
        backend = _backend()
        released = _reserved(backend)
        live = {1234}
        holder = backend._hold_real_gpus()
        with patch.object(sandbox_base, "_REAL_GPU_IDLE_POLL_INTERVAL_S", 0.01):
            with patch.object(backend, "get_live_pids", side_effect=lambda: set(live)):
                backend.release_real_gpu_when_idle(holder)
                assert released == []
                assert backend._real_gpu_ids == [0]
                time.sleep(0.1)
                assert released == []
                live.clear()
                assert _wait_for(lambda: released == [[0]])
        assert backend._real_gpu_ids == []
        assert _wait_for(lambda: backend._real_gpu_idle_thread is None)

    def test_new_holder_stops_the_background_release(self):
        backend = _backend()
        released = _reserved(backend)
        live = {1234}
        first = backend._hold_real_gpus()
        with patch.object(sandbox_base, "_REAL_GPU_IDLE_POLL_INTERVAL_S", 0.01):
            with patch.object(backend, "get_live_pids", side_effect=lambda: set(live)):
                backend.release_real_gpu_when_idle(first)
                second = backend._hold_real_gpus()
                live.clear()
                assert _wait_for(lambda: backend._real_gpu_idle_thread is None)
                assert released == []
                assert backend._real_gpu_ids == [0]
                backend.release_real_gpu_when_idle(second)
        assert released == [[0]]

    def test_release_virtual_and_real_resources_forgets_in_flight_holders(self):
        backend = _backend()
        _reserved(backend)
        backend._hold_real_gpus()
        backend.release_virtual_and_real_resources(None)
        assert backend._real_gpu_holders == set()
        assert backend._virtual_gpu_count_requested == 0


class TestReleaseRealGpuHolder:
    def test_releases_at_once_when_the_last_holder_is_dropped(self):
        backend = _backend()
        released = _reserved(backend)
        first, second = backend._hold_real_gpus(), backend._hold_real_gpus()
        backend.release_real_gpu_holder(first)
        assert released == []
        backend.release_real_gpu_holder(second)
        assert released == [[0]]
        assert backend._real_gpu_ids == []
        assert backend._virtual_gpu_count_requested == 1

    def test_does_not_wait_for_background_processes(self):
        backend = _backend()
        released = _reserved(backend)
        holder = backend._hold_real_gpus()
        with patch.object(backend, "get_live_pids", return_value={4242}):
            backend.release_real_gpu_holder(holder)
        assert released == [[0]]


class TestAcquireRealGpusFor:
    @staticmethod
    def _sandbox(backend):
        sandbox = agSandbox.__new__(agSandbox)
        sandbox._agname = "lease-test"
        sandbox._destroyed = True  # nothing to tear down in __del__
        sandbox._backend = backend
        return sandbox

    def test_takes_the_lease_once_and_registers_every_holder(self):
        backend = _backend()
        calls = []
        _reserved(backend)
        backend._real_gpu_acquire_fn = lambda count, timeout=None, is_cancelled=None: (
            calls.append((count, timeout)) or [3]
        )
        sandbox = self._sandbox(backend)
        assert sandbox.acquire_real_gpus_for("10", block=True) == [3]
        assert sandbox.acquire_real_gpus_for("11", block=False) == [3]
        assert calls == [(1, None)]
        assert backend._real_gpu_holders == {"10", "11"}

    def test_non_blocking_attempt_returns_none_when_no_gpu_is_free(self):
        backend = _backend()
        _reserved(backend)
        timeouts = []

        def none_free(count, timeout=None, is_cancelled=None):
            timeouts.append(timeout)
            raise TimeoutError

        backend._real_gpu_acquire_fn = none_free
        assert self._sandbox(backend).acquire_real_gpus_for("10", block=False) is None
        assert timeouts == [0]
        assert backend._real_gpu_holders == set()

    def test_blocking_wait_passes_the_cancel_callback_through(self):
        backend = _backend()
        _reserved(backend)
        seen = []
        backend._real_gpu_acquire_fn = lambda count, timeout=None, is_cancelled=None: (
            seen.append(is_cancelled) or [0]
        )
        cancel_check = lambda: False  # noqa: E731
        self._sandbox(backend).acquire_real_gpus_for("10", block=True, is_cancelled=cancel_check)
        assert seen == [cancel_check]
