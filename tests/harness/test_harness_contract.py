"""Harness contracts every external CLI must satisfy, tested uniformly.

Each test builds the harness object through the same adapter/driver seam and
asserts the common behavior Agency relies on: the Agency MCP servers are
installed into the launch and the host-supplied context limit reaches the
harness. Compaction arming follows from that limit (90% of it, plus each
CLI's own reserve) -- the firing behavior itself is verified per harness
against the real CLIs in test_context_window_live.py, and per-harness
dialect details live in each harness's own test file. This file must stay
free of harness-specific code.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from agency.configs.agconfig import agconfig, harnessadapterconfig
from agency.harness.adapters.base import AdapterRuntime, HarnessAdapter
from agency.harness.adapters.pty.driver import driver_for
from agency.utils.agutil import agency_config_homes_dir

STAND_IN = "/stand-in/harness-cli"

# Every harness Agency supports, except native: it is not a PTY dialect
# (its "CLI" is Agency's own react loop in a subprocess, covered by the
# native-specific files and test_daemon_seam.py).
PTY_HARNESSES = ("claude_code", "codex", "grok", "kimi", "opencode")

# Every harness Agency supports, including native.
ALL_HARNESSES = ("native", *PTY_HARNESSES)


def _runtime(**overrides) -> AdapterRuntime:
    defaults = dict(
        agconfig=agconfig(harnessadapterconfig(binary_path=STAND_IN)),
        model="test-model",
        engine_name="test-agent",
        harness_base_url="http://127.0.0.1:8766",
        token="attempt-token",
        syscall_policy=MagicMock(),
    )
    return AdapterRuntime(**{**defaults, **overrides})


def _patch_context_limit(monkeypatch, value):
    """Point every adapter's context-limit lookup at one deterministic value."""
    for module in ("base", "claude_code", "codex", "grok", "kimi", "opencode"):
        monkeypatch.setattr(
            f"agency.harness.adapters.{module}.fetch_context_limit",
            lambda *args, **kwargs: value,
        )


def _driver(harness: str, runtime: AdapterRuntime, tmp_path):
    return driver_for(
        HarnessAdapter.for_config(harness, runtime.agconfig), runtime, tmp_path, None, None, 4
    )


@pytest.mark.parametrize("harness", PTY_HARNESSES)
def test_harness_installs_the_agency_mcp_servers(harness, tmp_path):
    runtime = _runtime()
    driver = _driver(harness, runtime, tmp_path)
    assert driver.installed_mcp_servers() == HarnessAdapter.for_config(
        harness, runtime.agconfig
    ).launch_mcp_servers(runtime)


@pytest.mark.parametrize("harness", PTY_HARNESSES)
@pytest.mark.parametrize("has_sandbox_tools", [False, True])
def test_harness_installs_the_sandbox_mcp_server_only_when_present(
    harness, tmp_path, has_sandbox_tools
):
    runtime = _runtime(has_sandbox_mcp_tools=has_sandbox_tools)
    driver = _driver(harness, runtime, tmp_path)
    servers = driver.installed_mcp_servers()
    expected = {"agency", "agency-sandbox"} if has_sandbox_tools else {"agency"}
    assert set(servers) == expected
    assert servers["agency"]["url"] == f"{runtime.harness_base_url}/mcp"
    if has_sandbox_tools:
        assert servers["agency-sandbox"]["url"] == f"{runtime.harness_base_url}/sandbox/mcp"


@pytest.mark.parametrize("harness", PTY_HARNESSES)
def test_harness_receives_the_context_limit(harness, tmp_path, monkeypatch):
    _patch_context_limit(monkeypatch, 42_000)
    runtime = _runtime()
    driver = _driver(harness, runtime, tmp_path)
    assert driver.configured_context_limit() == 42_000


@pytest.mark.parametrize("harness", PTY_HARNESSES)
def test_harness_reports_no_context_limit_when_the_host_has_none(harness, tmp_path, monkeypatch):
    _patch_context_limit(monkeypatch, None)
    runtime = _runtime()
    driver = _driver(harness, runtime, tmp_path)
    assert driver.configured_context_limit() is None


@pytest.mark.parametrize("harness", ALL_HARNESSES)
@pytest.mark.parametrize("has_sandbox_tools", [False, True])
def test_every_launch_carries_the_agency_mcp_servers(monkeypatch, harness, has_sandbox_tools):
    """The launch itself carries the MCP servers: native and claude_code as
    an --mcp-config argument, every other dialect inside its config files.
    Asserted uniformly by scanning the launch the adapter actually built
    for each server's URL and bearer token -- no dialect knowledge here."""
    import json

    captured = {}

    def launch(_self, argv, envp, **kwargs):
        captured["argv"] = argv
        captured["envp"] = envp
        # Snapshot the config home's files now: the attempt's cleanup
        # deletes them before the test can scan for the MCP servers.
        for path in agency_config_homes_dir().rglob("*"):
            if path.is_file() and path.stat().st_size < 1_000_000:
                captured.setdefault("files", []).append(path.read_text(errors="replace"))
        handle = MagicMock()
        handle.wait.return_value = ('{"result": "done"}', "", 0)
        handle.terminal_screen.return_value = ([""] * 40, 0, 0, 1)
        handle.is_paused.return_value = False
        # Native's react loop reaps once the process has exited; the PTY
        # dialects never look at returncode before their readiness timeout.
        handle.returncode = 0
        return handle

    monkeypatch.setattr("agency.harness.ptrace.supervisor.agProxyPtrace.launch", launch)
    monkeypatch.setattr(
        "agency.utils.agutil.ensure_python_packages_locally", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "agency.harness.agharness.materialize_config_home_in_container",
        lambda *args, **kwargs: "/tmp/agharness-contract",
    )
    monkeypatch.setattr(
        "agency.harness.agharness.cleanup_config_home_in_container", lambda *args: None
    )
    # The blank mocked screen never satisfies any dialect's readiness poll,
    # so the PTY attempt ends quickly in a startup timeout -- after the
    # launch (whose argv/env we are here to inspect) has already happened.
    monkeypatch.setattr("agency.harness.adapters.pty.driver.PtyDriver.START_TIMEOUT", 0.1)
    _patch_context_limit(monkeypatch, None)
    runtime = replace(
        _runtime(sandbox=_ContractSandbox() if harness == "native" else None),
        has_sandbox_mcp_tools=has_sandbox_tools,
    )
    adapter = HarnessAdapter.for_config(harness, runtime.agconfig)
    # The same startup timeout surfaces here as a RuntimeError (the daemon
    # normally converts it into a failed-attempt result); this test only
    # needs the captured launch, so the exception is incidental.
    try:
        adapter.run_daemon_attempt(
            runtime, prompt="work", resume_session_id=None, prior_session_blob=None, max_steps=2
        )
    except RuntimeError:
        pass
    assert "argv" in captured, f"{harness} never launched"

    servers = adapter.launch_mcp_servers(runtime)
    expected = {"agency", "agency-sandbox"} if has_sandbox_tools else {"agency"}
    assert set(servers) == expected

    # Whatever dialect the launch used, the built artifacts must connect to
    # every server URL with the attempt's bearer token: native and
    # claude_code carry an --mcp-config argument, the other dialects write
    # the servers into their isolated config home. Scan both, uniformly.
    rendered = (
        json.dumps(captured["argv"])
        + json.dumps(captured["envp"])
        + "".join(captured.get("files", []))
    )
    for name in expected:
        assert servers[name]["url"] in rendered, f"{harness} launch never mentions {name}"
        assert runtime.token in rendered
        assert servers[name]["headers"]["Authorization"] == f"Bearer {runtime.token}"


class _ContractSandbox:
    """The minimal sandbox surface the native adapter touches."""

    class _Backend:
        IMAGE_KIND = "container"

        def ingest_ptrace_pids(self, **_changes):
            return None

    def __init__(self):
        self._backend = self._Backend()

    def exec(self, cmd, workdir="/workspace", timeout=600):
        return "", 0

    def write_file_bytes(self, path, data):
        return None
