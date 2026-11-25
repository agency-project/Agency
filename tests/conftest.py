"""Reset mutable agent class-level and module-level state between tests."""

import os
import sys

import pytest
import agency.utils.agutil as _agutil_module
from agency.agent import agent
from agency.agname import agname as _agname

from tests.harness_installs import container_image_problem, skip_or_fail

CONTAINER_BACKEND = os.environ.get("AGENCY_TEST_CONTAINER_BACKEND", "docker")


@pytest.fixture(scope="module")
def golden_image():
    image = os.environ.get("AGENCY_TEST_HARNESS_IMAGE", "docker.io/library/python:3.12-slim")
    problem = None
    if not sys.platform.startswith("linux"):
        problem = "golden profiling and real external harnesses require Linux"
    else:
        problem = container_image_problem(CONTAINER_BACKEND, image)
    if problem:
        skip_or_fail(problem)
    return image


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    """Set stream-batch delay to zero so tests don't sleep 100 ms per LLM call."""
    monkeypatch.setattr(_agutil_module, "_BATCH_INTERVAL_S", 0.0)


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
