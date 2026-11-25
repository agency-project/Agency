"""Availability is separate from interactive execution (test_external_pty.py)."""

from unittest.mock import MagicMock

from agency.configs.agconfig import agconfig, harnessadapterconfig
from agency.harness.adapters.base import AdapterRuntime
from agency.harness.adapters.grok import GrokAdapter, GrokDriver, grok_available
from agency.harness.pins import managed_binary

STAND_IN = "/stand-in/harness-cli"  # adapters only build argv; nothing runs it


def test_grok_available_reflects_the_managed_install():
    assert grok_available() == (managed_binary("grok") is not None)


def _grok_argv(tmp_path, config):
    runtime = AdapterRuntime(
        config, "test-model", "agent", "http://daemon", "attempt-key", MagicMock()
    )
    return GrokDriver(GrokAdapter(config), runtime, tmp_path, None, None, None).argv


def _runtime(has_sandbox_mcp_tools):
    config = agconfig(harnessadapterconfig(binary_path=STAND_IN))
    return AdapterRuntime(
        config,
        "test-model",
        "agent",
        "http://daemon",
        "attempt-key",
        MagicMock(),
        has_sandbox_mcp_tools=has_sandbox_mcp_tools,
    )


def test_grok_disallows_subagents_by_default(tmp_path):
    assert "--no-subagents" in _grok_argv(
        tmp_path, agconfig(harnessadapterconfig(binary_path=STAND_IN))
    )


def test_grok_omits_no_subagents_when_allowed(tmp_path):
    argv = _grok_argv(
        tmp_path, agconfig(harnessadapterconfig(allow_subagents=True, binary_path=STAND_IN))
    )
    assert "--no-subagents" not in argv


def test_grok_config_wires_the_agency_mcp_servers(tmp_path):
    config = agconfig(harnessadapterconfig(binary_path=STAND_IN))
    GrokDriver(GrokAdapter(config), _runtime(True), tmp_path, None, None, None)
    toml_text = (tmp_path / "config.toml").read_text()
    assert '[mcp_servers.agency]\nurl = "http://daemon/mcp"' in toml_text
    assert 'headers = { Authorization = "Bearer attempt-key" }' in toml_text
    assert "[mcp_servers.agency-sandbox]" in toml_text
    assert 'url = "http://daemon/sandbox/mcp"' in toml_text


def test_grok_config_omits_the_sandbox_server_without_sandbox_tools(tmp_path):
    config = agconfig(harnessadapterconfig(binary_path=STAND_IN))
    GrokDriver(GrokAdapter(config), _runtime(False), tmp_path, None, None, None)
    toml_text = (tmp_path / "config.toml").read_text()
    assert "[mcp_servers.agency]" in toml_text
    assert "agency-sandbox" not in toml_text
