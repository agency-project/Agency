# Resource cleanup when a run dies at each layer: harness, tracer, daemon, host engine.
# The sandbox is a real container backend holding a real pool lease; only the runtime calls are faked.

from __future__ import annotations

import functools
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agency.agcontext import agcontext
from agency.agdata import agdata, agerror
from agency.agpolicy import agpolicy
from agency.configs.agconfig import agconfig as agconfig_cls
from agency.engine import engine as mod
from agency.engine.engine import AgentEngine
from agency.harness.protocol import HarnessAttemptResult
from agency.orchestrator.agresources import agResourcePool
from agency.sandbox.docker import _DockerBackend

from .test_engine import _FakeAgent, _install_fake_host_server_manager


class _LeasedSandbox:
    def __init__(self, pool):
        self._name = "leased-sandbox"
        self._lock = threading.RLock()
        self.running = True
        self.slots_released = 0
        backend = _DockerBackend(
            "agent",
            name=self._name,
            checkpoint_image=None,
            base_image="img",
            mounts={},
            agconfig=agconfig_cls(),
        )
        backend._container_running = lambda: self.running
        backend._container_status = lambda: "running" if self.running else ""
        backend._rm_container = lambda _name: setattr(self, "running", False)
        backend._run = lambda *_args, **_kwargs: setattr(self, "running", False)
        backend._release_runtime_slot = lambda: setattr(
            self, "slots_released", self.slots_released + 1
        )
        backend._virtual_gpu_count_requested = 1
        backend._real_gpu_release_fn = functools.partial(pool.release_real_gpus, self)
        self._backend = backend

    @property
    def _real_gpu_ids(self):
        return self._backend._real_gpu_ids

    @_real_gpu_ids.setter
    def _real_gpu_ids(self, value):
        self._backend._real_gpu_ids = value

    def commit(self):
        pass

    def _has_pending_background_work(self):
        return False

    def stop(self):
        self._backend._stop_image()

    def rm_container(self):
        self._backend.rm_container()


@pytest.fixture
def leased():
    pool = agResourcePool(gpus=[0], mark_gpus=False)
    sandbox = _LeasedSandbox(pool)
    pool.acquire_real_gpus(sandbox, 1)
    assert pool._free_gpus == set()
    return pool, sandbox


def _skill():
    return SimpleNamespace(
        sandbox_mcp_tools=[], policy=agpolicy(), output_schema=None, max_output_schema_retries=3
    )


def _engine(monkeypatch):
    engine = AgentEngine(_FakeAgent())
    monkeypatch.setattr(engine, "_build_prompt_payload", lambda *_args, **_kwargs: "prompt")
    return engine


def _execute(engine, sandbox):
    return engine.execute(agcontext(), _skill(), SimpleNamespace(), SimpleNamespace(), sandbox)


def _assert_everything_released(pool, sandbox, holder):
    assert pool._free_gpus == {0}
    assert pool._gpus_acquired == 0
    assert sandbox._real_gpu_ids == []
    assert sandbox.running is False
    assert sandbox.slots_released == 1
    assert holder["manager"].stopped is True
    assert sandbox._lock.acquire(blocking=False)


def test_successful_run_gives_back_the_lease_when_the_container_hibernates(monkeypatch, leased):
    pool, sandbox = leased
    holder = _install_fake_host_server_manager(
        monkeypatch, results=[HarnessAttemptResult(ok=True, final_text="done")]
    )
    engine = _engine(monkeypatch)
    monkeypatch.setattr(engine, "_build_execution_result", lambda *_args: agdata(done=True))

    _execute(engine, sandbox)

    _assert_everything_released(pool, sandbox, holder)


@pytest.mark.parametrize(
    "message",
    [
        "native_harness exited with code 137: Killed",
        "native_harness exited with code -1: ",
    ],
    ids=["harness_killed", "tracer_died"],
)
def test_failed_attempt_gives_back_the_lease(monkeypatch, leased, message):
    pool, sandbox = leased
    holder = _install_fake_host_server_manager(
        monkeypatch, results=[HarnessAttemptResult(ok=False, error_message=message)]
    )
    engine = _engine(monkeypatch)

    result = _execute(engine, sandbox)

    assert isinstance(result, agerror)
    assert message in result.error
    assert len(holder["requests"]) == 1
    _assert_everything_released(pool, sandbox, holder)


def test_daemon_that_stops_answering_gives_back_the_lease(monkeypatch, leased):
    pool, sandbox = leased
    holder = _install_fake_host_server_manager(monkeypatch, results=[])
    never_returns = threading.Event()
    client = holder["client"]
    client.run_harness_attempt = lambda _request: never_returns.wait(10)
    client.is_ready = MagicMock(side_effect=ConnectionError("daemon unreachable"))
    monkeypatch.setattr(mod, "_HEALTH_CHECK_INTERVAL_S", 0.01)
    monkeypatch.setattr(mod, "_HEALTH_CHECK_FAILURE_LIMIT", 3)
    engine = _engine(monkeypatch)

    try:
        result = _execute(engine, sandbox)
    finally:
        never_returns.set()

    assert isinstance(result, agerror)
    assert "health check" in result.error
    _assert_everything_released(pool, sandbox, holder)


@pytest.mark.parametrize("raised_by", ["daemon_launch", "attempt_rpc"])
def test_host_engine_exception_gives_back_the_lease(monkeypatch, leased, raised_by):
    pool, sandbox = leased
    holder = _install_fake_host_server_manager(monkeypatch, results=[])
    if raised_by == "daemon_launch":
        monkeypatch.setattr(
            mod,
            "ensure_harness_daemon",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("launch failed")),
        )
    else:
        holder["client"].run_harness_attempt = MagicMock(side_effect=RuntimeError("rpc failed"))
    engine = _engine(monkeypatch)

    with pytest.raises(RuntimeError, match="failed"):
        _execute(engine, sandbox)

    _assert_everything_released(pool, sandbox, holder)
