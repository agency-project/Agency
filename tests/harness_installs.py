"""Gate for tests that run the real external harness CLIs.

The CLIs are the pinned installs from `agency install-harnesses`. A missing
prerequisite skips the test on a development machine and fails it in CI, so CI
always exercises every harness.
"""

import os

import pytest

from agency.harness.pins import managed_binary


def skip_or_fail(problem: str) -> None:
    if os.environ.get("CI"):
        pytest.fail(problem)
    pytest.skip(problem)


def require_harness(harness: str):
    """The managed install of `harness`; skips (fails in CI) if it is missing."""
    binary = managed_binary(harness)
    if binary is None:
        skip_or_fail(f"{harness} is not installed; run `agency install-harnesses {harness}`")
    return binary
