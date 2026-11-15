"""DeepSeek Harness and Antigravity CLI adapters: configuration, output parsing,
session files and installer helpers (no CLI is run here)."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from agency.configs.agconfig import agconfig
from agency.harness import install
from agency.harness.adapters.antigravity import (
    AntigravityAdapter,
    cli_model_for,
    is_monitor_request,
    parse_stream_json,
)
from agency.harness.adapters.deepseek_harness import DeepSeekHarnessAdapter, parse_json_events
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


def test_session_files_each_harness_carries():
    allowed = lambda harness, path: session_file_allowed(harness, PurePosixPath(path))  # noqa: E731
    assert allowed("dsh", "sessions/--workspace--/session-1/session.v4.jsonl.zstd")
    assert not allowed("dsh", "sessions/--workspace--/session-1/session.lock")
    assert not allowed("dsh", "sessions/--workspace--/session-1")  # a directory
    assert not allowed("dsh", "profiles/headless/cordis.yml")
    assert allowed("antigravity", ".gemini/antigravity-cli/conversations/c1.db")
    assert allowed("antigravity", ".gemini/antigravity-cli/brain/c1/.system_generated/logs/t.jsonl")
    assert not allowed("antigravity", ".gemini/antigravity-cli/brain/c1/.system_generated")
    assert not allowed("antigravity", ".gemini/antigravity-cli/settings.json")
    assert not allowed("antigravity", ".gemini/config/hooks.json")


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


def test_pypi_version_spelling():
    assert install._pep440("0.2.2-rc.2") == "0.2.2rc2"
    assert install._pep440("0.2.2-alpha.3") == "0.2.2a3"
    assert install._pep440("1.0.0") == "1.0.0"


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
    adapter = AntigravityAdapter(agconfig())
    assert is_monitor_request(review) and not is_monitor_request(turn)
    verdict = json.loads(adapter._local_response(review)[0]["text"])
    assert set(verdict) == {"reason", "severity", "confidence"}
    assert adapter._local_response(turn) is None
