"""Availability is separate from interactive execution (test_external_pty.py)."""

import json
from unittest.mock import MagicMock

from agency.configs.agconfig import agconfig, harnessadapterconfig
from agency.harness.adapters.base import AdapterRuntime
from agency.harness.adapters.opencode import OpenCodeAdapter, opencode_available
from agency.harness.adapters.pty.driver import driver_for
from agency.harness.pins import managed_binary

STAND_IN = "/stand-in/harness-cli"


def test_opencode_available_reflects_the_managed_install():
    assert opencode_available() == (managed_binary("opencode") is not None)


def _driver(tmp_path, has_sandbox_mcp_tools):
    runtime = AdapterRuntime(
        agconfig(harnessadapterconfig(binary_path=STAND_IN)),
        "test-model",
        "agent",
        "http://daemon",
        "attempt-key",
        MagicMock(),
        has_sandbox_mcp_tools=has_sandbox_mcp_tools,
    )
    return driver_for(OpenCodeAdapter(runtime.agconfig), runtime, tmp_path, None, None, None)


def test_opencode_config_wires_the_agency_mcp_servers(tmp_path):
    _driver(tmp_path, has_sandbox_mcp_tools=True)
    config = json.loads((tmp_path / "opencode.json").read_text())
    assert config["mcp"]["agency"] == {
        "type": "remote",
        "url": "http://daemon/mcp",
        "headers": {"Authorization": "Bearer attempt-key"},
        "oauth": False,
    }
    assert config["mcp"]["agency-sandbox"]["url"] == "http://daemon/sandbox/mcp"


def test_opencode_config_omits_the_sandbox_server_without_sandbox_tools(tmp_path):
    _driver(tmp_path, has_sandbox_mcp_tools=False)
    config = json.loads((tmp_path / "opencode.json").read_text())
    assert set(config["mcp"]) == {"agency"}
