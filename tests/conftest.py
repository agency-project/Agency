"""Reset mutable agent class-level and module-level state between tests."""

import os

import pytest
import agency.utils.agutil as _agutil_module
from agency.harness import pins
from agency.agent import agent
from agency.agname import agname as _agname


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    """Set stream-batch delay to zero so tests don't sleep 100 ms per LLM call."""
    monkeypatch.setattr(_agutil_module, "_BATCH_INTERVAL_S", 0.0)


@pytest.fixture(scope="session")
def _fake_harness_home(tmp_path_factory):
    """Stand-ins for the pinned harness installs: they exist and report the
    pinned version, which is all the tests that don't run a real CLI need."""
    home = tmp_path_factory.mktemp("harness-home")
    for harness, version in pins.harness_pins().items():
        binary = pins.install_dir(harness, home) / pins.BINARY_RELPATH[harness]
        binary.parent.mkdir(parents=True)
        binary.write_text(f"#!/bin/sh\necho {version}\n")
        binary.chmod(0o755)
    return home


@pytest.fixture(autouse=True)
def _harness_installs(monkeypatch, _fake_harness_home):
    """Keep tests independent of what `agency install-harnesses` put on this
    machine, unless the run is meant to use the real installs: CI, an opt-in
    real-harness run, or an explicit AGENCY_HARNESS_HOME."""
    if (
        os.environ.get("CI")
        or os.environ.get("AGENCY_TEST_EXTERNAL_HARNESSES") == "1"
        or pins.HOME_ENV in os.environ
    ):
        return
    monkeypatch.setenv(pins.HOME_ENV, str(_fake_harness_home))


@pytest.fixture(autouse=True)
def reset_agent_state():
    saved = {
        "log_dir": agent.log_dir,
        "output_dir": agent.output_dir,
    }
    yield
    from agency.orchestrator.orchestrator import _reset_orchestrator_for_tests

    _reset_orchestrator_for_tests()
    agent.log_dir = saved["log_dir"]
    agent.output_dir = saved["output_dir"]
    # Reset module-level name registry so tests don't bleed agnames into each other
    _agname._allocated.clear()
    _agname._noun_counters.clear()
    # WeakSet clears itself as objects die; force a GC pass to help along
    import gc

    gc.collect()
