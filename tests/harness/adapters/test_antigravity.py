"""Antigravity CLI adapter: configuration, output parsing and session files (no
CLI is run here)."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from types import SimpleNamespace

from agency.configs.agconfig import agconfig
from agency.harness.adapters.antigravity import (
    AntigravityAdapter,
    cli_model_for,
    is_monitor_request,
    parse_stream_json,
)
from agency.harness.adapters.pty.execution import (
    restore_session,
    session_file_allowed,
    snapshot_session,
)


def _runtime(**overrides):
    values = dict(
        harness_base_url="http://127.0.0.1:9",
        token="tok",
        model="google/gemini-4-carbon",
        has_sandbox_mcp_tools=True,
        agconfig=agconfig(),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_antigravity_model_names():
    assert cli_model_for("google/gemini-4-carbon") == "gemini-4-carbon"
    assert cli_model_for("gemini-3.5-hydrogen") == "gemini-3.5-hydrogen"
    assert cli_model_for("DeepSeek-V4.1-Flash") == "gemini-4-carbon"
    assert cli_model_for(None) == "gemini-4-carbon"


def test_antigravity_parses_the_result_event():
    stdout = "\n".join(
        [
            json.dumps({"event": "init", "conversation_id": "conv-1"}),
            "not json",
            json.dumps(
                {
                    "event": "result",
                    "result": {
                        "conversation_id": "conv-1",
                        "status": "SUCCESS",
                        "response": "done\n",
                        "usage": {"input_tokens": 3, "output_tokens": 4},
                    },
                }
            ),
        ]
    )
    result, conversation = parse_stream_json(stdout)
    assert conversation == "conv-1"
    assert result["status"] == "SUCCESS" and result["response"] == "done\n"
    assert parse_stream_json("") == (None, None)


def test_antigravity_config_routes_models_tools_and_policy(tmp_path):
    AntigravityAdapter(agconfig())._write_config(tmp_path, _runtime())
    settings = json.loads((tmp_path / ".gemini/antigravity-cli/settings.json").read_text())
    assert settings == {"modelProvider": "gemini"}
    mcp = json.loads((tmp_path / ".gemini/config/mcp_config.json").read_text())["mcpServers"]
    assert mcp["agency"] == {
        "serverUrl": "http://127.0.0.1:9/mcp",
        "headers": {"Authorization": "Bearer tok"},
    }
    assert mcp["agency-sandbox"]["serverUrl"] == "http://127.0.0.1:9/sandbox/mcp"
    hooks = json.loads((tmp_path / ".gemini/config/hooks.json").read_text())["agency"]
    for event in ("PreToolUse", "PostToolUse"):
        command = hooks[event][0]["hooks"][0]["command"]
        assert command.startswith("python3 ") and command.endswith("agpolicy_hook.py")
    assert (tmp_path / "agpolicy_hook.py").is_file()


def test_antigravity_session_files():
    allowed = lambda path: session_file_allowed("antigravity", PurePosixPath(path))  # noqa: E731
    assert allowed(".gemini/antigravity-cli/conversations/c1.db")
    # Turns SQLite has not yet checkpointed into the .db travel with it.
    assert allowed(".gemini/antigravity-cli/conversations/c1.db-wal")
    assert not allowed(".gemini/antigravity-cli/conversations/c1.db-shm")
    assert allowed(".gemini/antigravity-cli/brain/c1/.system_generated/logs/t.jsonl")
    assert not allowed(".gemini/antigravity-cli/brain/c1/.system_generated")
    assert not allowed(".gemini/antigravity-cli/settings.json")
    assert not allowed(".gemini/config/hooks.json")


def test_antigravity_session_bundle_round_trips(tmp_path):
    source, target = tmp_path / "a", tmp_path / "b"
    conversation = source / ".gemini/antigravity-cli/conversations/c1.db"
    conversation.parent.mkdir(parents=True)
    conversation.write_bytes(b"sqlite")
    log = source / ".gemini/antigravity-cli/brain/c1/.system_generated/logs/transcript.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text("{}")
    (source / ".gemini/antigravity-cli/settings.json").write_text("{}")
    blob = snapshot_session(source, "antigravity", "c1")
    target.mkdir()
    restore_session(target, "antigravity", "c1", blob)
    assert (target / ".gemini/antigravity-cli/conversations/c1.db").read_bytes() == b"sqlite"
    assert not (target / ".gemini/antigravity-cli/settings.json").exists()


def test_antigravity_answers_its_safety_monitor_locally():
    review = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "rate this\n<REVIEW_RUBRIC>\n...\n</REVIEW_RUBRIC>"}],
            }
        ]
    }
    turn = {"contents": [{"role": "user", "parts": [{"text": "list the files"}]}]}
    monitor, agent = "gemini-3.1-flash-lite-preview", "DeepSeek-V4.1-Flash"
    adapter = AntigravityAdapter(agconfig())
    assert is_monitor_request(review, monitor, agent)
    assert not is_monitor_request(turn, monitor, agent)
    verdict = json.loads(adapter._local_response(review, monitor, agent)[0]["text"])
    assert set(verdict) == {"reason", "severity", "confidence"}
    assert adapter._local_response(turn, monitor, agent) is None


def test_a_conversation_that_mentions_the_rubric_still_reaches_the_model():
    rubric = {"role": "user", "parts": [{"text": "explain <REVIEW_RUBRIC> to me"}]}
    monitor = "gemini-3.1-flash-lite-preview"
    # Addressed to the model the agent itself runs under: the agent's own turn.
    assert not is_monitor_request({"contents": [rubric]}, "gemini-4-carbon", "DeepSeek-V4.1-Flash")
    assert not is_monitor_request(
        {"contents": [rubric]}, "models/gemini-3.5-hydrogen", "google/gemini-3.5-hydrogen"
    )
    # Mentioned earlier in the history, not what this request asks about.
    later = {"role": "user", "parts": [{"text": "now list the files"}]}
    assert not is_monitor_request({"contents": [rubric, later]}, monitor, "DeepSeek-V4.1-Flash")
    assert not is_monitor_request({}, monitor, None)


def _attempt(tmp_path, monkeypatch, events, *, rc=0, conversation=True):
    """Run one attempt against a stand-in for the traced agy process."""
    import sys
    from types import ModuleType

    from agency.harness import agharness

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
            if conversation:
                db = tmp_path / ".gemini/antigravity-cli/conversations/conv-1.db"
                db.parent.mkdir(parents=True, exist_ok=True)
                db.write_bytes(b"sqlite")
            return Handle()

    supervisor = ModuleType("agency.harness.ptrace.supervisor")
    supervisor.agProxyPtrace = Tracer
    monkeypatch.setitem(sys.modules, "agency.harness.ptrace.supervisor", supervisor)
    monkeypatch.setattr(agharness, "materialize_config_home", lambda name: tmp_path)
    monkeypatch.setattr(agharness, "cleanup_config_home", lambda root: None)
    monkeypatch.setattr(AntigravityAdapter, "binary", lambda self: "/harness/antigravity")
    runtime = _runtime(
        engine_name="worker", syscall_policy=None, register_control_handle=lambda handle: None
    )
    result = AntigravityAdapter(agconfig()).run_daemon_attempt(
        runtime, prompt="do it", resume_session_id=None, prior_session_blob=None, max_steps=None
    )
    return result, launched


def _result(**fields):
    return [{"event": "result", "result": {"conversation_id": "conv-1", **fields}}]


def test_attempt_returns_the_answer_usage_and_conversation(tmp_path, monkeypatch):
    usage = {"input_tokens": 3, "output_tokens": 4}
    result, launched = _attempt(
        tmp_path, monkeypatch, _result(status="SUCCESS", response="done", usage=usage)
    )
    assert result.ok and result.final_text == "done"
    assert (result.input_tokens, result.output_tokens) == (3, 4)
    assert result.session_id == "conv-1" and result.session_blob
    assert launched["argv"][1:3] == ["-p", "do it"]
    assert launched["envp"]["AGPOLICY_HOOK_DIALECT"] == "antigravity"


def test_a_structured_error_is_reported_not_a_crash(tmp_path, monkeypatch):
    error = {"code": 429, "message": "quota"}
    result, _ = _attempt(tmp_path, monkeypatch, _result(status="ERROR", error=error), rc=1)
    assert not result.ok and "quota" in result.error_message and "code 1" in result.error_message
    result, _ = _attempt(tmp_path, monkeypatch, [{"event": "result", "result": "oops"}], rc=1)
    assert not result.ok and "stderr text" in result.error_message


def test_a_conversation_that_cannot_be_saved_fails_the_attempt(tmp_path, monkeypatch):
    result, _ = _attempt(
        tmp_path, monkeypatch, _result(status="SUCCESS", response="done"), conversation=False
    )
    assert not result.ok and "conversation could not be saved" in result.error_message
