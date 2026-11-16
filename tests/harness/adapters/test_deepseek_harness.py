"""DeepSeek Harness adapter: configuration, output parsing, session files and
installer helpers (no CLI is run here)."""

from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace

from agency.configs.agconfig import agconfig
from agency.harness import install
from agency.harness.adapters.deepseek_harness import DeepSeekHarnessAdapter, parse_json_events
from agency.harness.adapters.pty.execution import (
    restore_session,
    session_file_allowed,
    snapshot_session,
)


def test_deepseek_harness_parses_final_and_usage():
    stdout = "\n".join(
        json.dumps(event)
        for event in [
            {"type": "session", "sessionId": "session-1"},
            {
                "type": "status",
                "phase": "step_end",
                "usage": {"inputTokens": 10, "outputTokens": 2},
            },
            {"type": "status", "phase": "step_end", "usage": {"inputTokens": 5, "outputTokens": 1}},
            {
                "type": "final",
                "sessionId": "session-1",
                "reason": {"kind": "completed"},
                "text": "ok",
            },
        ]
    )
    final, input_tokens, output_tokens = parse_json_events(stdout)
    assert final["text"] == "ok" and final["reason"]["kind"] == "completed"
    assert (input_tokens, output_tokens) == (15, 3)


def test_deepseek_harness_patch(tmp_path):
    adapter = DeepSeekHarnessAdapter(agconfig())
    path = adapter._write_patch(
        tmp_path,
        "http://127.0.0.1:9",
        "secret-token-1",
        "DeepSeek-V4.1-Flash",
        has_sandbox_mcp_tools=False,
        context_limit=100_000,
    )
    patch = {row.get("id", "insert"): row for row in json.loads(path.read_text())}
    provider = patch["llm-pi-ai"]["config"]["providers"]["agency"]
    assert provider["api"] == "openai-completions"
    assert provider["baseURL"] == "http://127.0.0.1:9/v1"
    assert provider["apiKeyEnv"] == "AGENCY_TOKEN" and "secret-token-1" not in json.dumps(provider)
    assert provider["models"][0]["id"] == "DeepSeek-V4.1-Flash"
    assert provider["models"][0]["contextWindow"] == 100_000
    assert patch["agent-default-model"]["config"] == {
        "provider": "agency",
        "model": "DeepSeek-V4.1-Flash",
    }
    assert patch["session-log-deepseek"]["disabled"] is True
    assert patch["session-title-llm"]["disabled"] is True
    inserts = {row["id"]: row for row in patch["insert"]["insert"]}
    assert inserts["mcp-agency"]["config"]["url"] == "http://127.0.0.1:9/mcp"
    assert inserts["mcp-agency"]["config"]["transport"] == "streamable-http"
    assert "mcp-agency-sandbox" not in inserts
    plugin = Path(inserts["agency-agpolicy"]["name"])
    assert plugin.parent == tmp_path and plugin.read_text().startswith("// Agency policy bridge")


def test_dsh_session_files():
    allowed = lambda path: session_file_allowed("dsh", PurePosixPath(path))  # noqa: E731
    assert allowed("sessions/--workspace--/session-1/session.v4.jsonl.zstd")
    assert not allowed("sessions/--workspace--/session-1/session.lock")
    assert not allowed("sessions/--workspace--/session-1")  # a directory
    assert not allowed("profiles/headless/cordis.yml")


def test_dsh_session_bundle_round_trips(tmp_path):
    source, target = tmp_path / "a", tmp_path / "b"
    session = source / "sessions/--workspace--/session-1"
    session.mkdir(parents=True)
    (session / "session.v4.jsonl.zstd").write_bytes(b"zstd")
    (session / "session.lock").write_text("")
    blob = snapshot_session(source, "dsh", "session-1")
    target.mkdir()
    restore_session(target, "dsh", "session-1", blob)
    assert (
        target / "sessions/--workspace--/session-1/session.v4.jsonl.zstd"
    ).read_bytes() == b"zstd"
    assert not (target / "sessions/--workspace--/session-1/session.lock").exists()


def test_pypi_version_spelling():
    assert install._pep440("0.2.2-rc.2") == "0.2.2rc2"
    assert install._pep440("0.2.2-alpha.3") == "0.2.2a3"
    assert install._pep440("1.0.0") == "1.0.0"


def _final(kind="completed", **extra):
    return [
        {"type": "status", "phase": "step_end", "usage": {"inputTokens": 7, "outputTokens": 2}},
        {
            "type": "final",
            "sessionId": "session-1",
            "reason": {"kind": kind},
            "text": "ok",
            **extra,
        },
    ]


def _attempt(tmp_path, monkeypatch, events, *, rc=0, plugin_loaded=True, session=True, **kwargs):
    """Run one attempt against a stand-in for the traced dsh process, which
    leaves behind what the real one would: its events, session and marker."""
    from agency.harness import agharness
    from agency.harness.adapters import deepseek_harness

    launched = {}

    class Handle:
        returncode = rc

        def kill(self):
            pass

        def wait(self, timeout=None):
            return "\n".join(json.dumps(event) for event in events), "stderr text", rc

        def close(self):
            pass

    class Tracer:
        def __init__(self, *args, **kwargs):
            pass

        def launch(self, argv, envp, **kwargs):
            launched.update(argv=argv, envp=envp)
            if plugin_loaded:
                Path(envp["AGPOLICY_LOADED_FILE"]).write_text("")
            if session:
                directory = tmp_path / "sessions/--workspace--/session-1"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "session.v4.jsonl.zstd").write_bytes(b"zstd")
            return Handle()

    supervisor = ModuleType("agency.harness.ptrace.supervisor")
    supervisor.agProxyPtrace = Tracer
    monkeypatch.setitem(sys.modules, "agency.harness.ptrace.supervisor", supervisor)
    monkeypatch.setattr(agharness, "materialize_config_home", lambda name: tmp_path)
    monkeypatch.setattr(agharness, "cleanup_config_home", lambda root: None)
    monkeypatch.setattr(deepseek_harness, "fetch_context_limit", lambda *args: None)
    monkeypatch.setattr(DeepSeekHarnessAdapter, "binary", lambda self: "/harness/dsh")
    runtime = SimpleNamespace(
        engine_name="worker",
        harness_base_url="http://127.0.0.1:9",
        token="tok",
        model="DeepSeek-V4.1-Flash",
        has_sandbox_mcp_tools=False,
        agconfig=agconfig(),
        syscall_policy=None,
        register_control_handle=lambda handle: None,
    )
    options = dict(prompt="do it", resume_session_id=None, prior_session_blob=None, max_steps=None)
    options.update(kwargs)
    result = DeepSeekHarnessAdapter(agconfig()).run_daemon_attempt(runtime, **options)
    return result, launched


def test_attempt_returns_the_answer_usage_and_session(tmp_path, monkeypatch):
    result, launched = _attempt(tmp_path, monkeypatch, _final())
    assert result.ok and result.final_text == "ok"
    assert (result.input_tokens, result.output_tokens) == (7, 2)
    assert result.session_id == "session-1" and result.session_blob
    assert launched["argv"][-2:] == ["--", "do it"] and "--session-id" not in launched["argv"]


def test_attempt_resumes_the_given_session(tmp_path, monkeypatch):
    first, _ = _attempt(tmp_path, monkeypatch, _final())
    _, launched = _attempt(
        tmp_path,
        monkeypatch,
        _final(),
        resume_session_id="session-1",
        prior_session_blob=first.session_blob,
    )
    argv = launched["argv"]
    assert argv[argv.index("--session-id") + 1] == "session-1"


def test_a_turn_cut_off_at_the_token_limit_is_resumable(tmp_path, monkeypatch):
    # Headless mode exits 1 after a max-tokens end; the session is still carried.
    result, _ = _attempt(tmp_path, monkeypatch, _final("max-tokens"), rc=1)
    assert result.ok and result.session_id == "session-1"


def test_any_other_end_fails_the_attempt_and_keeps_its_usage(tmp_path, monkeypatch):
    result, _ = _attempt(tmp_path, monkeypatch, _final("error"), rc=1)
    assert not result.ok and '"kind": "error"' in result.error_message
    assert (result.input_tokens, result.output_tokens) == (7, 2)
    # An end reason that is not an object is reported, not a crash.
    result, _ = _attempt(tmp_path, monkeypatch, _final() + [_final()[1] | {"reason": "aborted"}])
    assert not result.ok and "aborted" in result.error_message


def test_a_run_without_a_final_event_fails_with_the_cli_output(tmp_path, monkeypatch):
    result, _ = _attempt(tmp_path, monkeypatch, _final()[:1], rc=2)
    assert not result.ok and "code 2" in result.error_message
    assert "stderr text" in result.error_message


def test_a_run_whose_policy_plugin_never_loaded_fails(tmp_path, monkeypatch):
    result, launched = _attempt(tmp_path, monkeypatch, _final(), plugin_loaded=False)
    assert not result.ok and "policy plugin" in result.error_message
    assert launched["envp"]["AGPOLICY_LOADED_FILE"].startswith(str(tmp_path))


def test_a_session_that_cannot_be_saved_fails_the_attempt(tmp_path, monkeypatch):
    # No session file on disk: success without a session would send the next
    # prompt to a new, empty dsh session.
    result, _ = _attempt(tmp_path, monkeypatch, _final(), session=False)
    assert not result.ok and "session could not be saved" in result.error_message
