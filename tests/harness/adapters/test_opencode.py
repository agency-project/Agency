"""Availability is separate from interactive execution (test_external_pty.py)."""

from agency.harness.adapters.opencode import opencode_available
from agency.harness.pins import managed_binary


def test_opencode_available_reflects_the_managed_install():
    assert opencode_available() == (managed_binary("opencode") is not None)
