"""Availability is separate from interactive execution (test_external_pty.py)."""

from agency.harness.adapters.codex import codex_available
from agency.harness.pins import managed_binary


def test_codex_available_reflects_the_managed_install():
    assert codex_available() == (managed_binary("codex") is not None)
