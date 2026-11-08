"""Gate for tests that run the real external harness CLIs.

The CLIs are the pinned installs from `agency install-harnesses`. A missing
prerequisite skips the test on a development machine and fails it in CI, so CI
always exercises every harness.
"""

import os
import subprocess

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


def container_image_problem(backend: str, image: str) -> "str | None":
    """Why `image` can't be used with `backend`, or None. An image that isn't
    local yet is pulled, so a clean machine (CI) needs no earlier test to have
    fetched it; a local-only image that is missing can't be pulled and is reported."""
    try:
        subprocess.run([backend, "info"], capture_output=True, check=True, timeout=15)
        if subprocess.run([backend, "image", "inspect", image], capture_output=True).returncode:
            subprocess.run([backend, "pull", image], capture_output=True, check=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"requires {backend} and the image {image}: {exc}"
    return None
