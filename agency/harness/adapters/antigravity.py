"""Antigravity CLI (``agy``) adapter, driven in print mode.

Verified against Antigravity CLI 1.5.9. Each attempt runs one
``agy -p <prompt> --output-format stream-json`` process under the same ptrace
supervision as every other adapter and reads its final ``result`` event. There
is no TUI dialect to drive: the process exits when the turn is done, and a
later attempt resumes the conversation with ``--conversation <id>`` from the
restored session files.

Model traffic: agy's Gemini API client honours ``GOOGLE_GEMINI_BASE_URL`` when
``"modelProvider": "gemini"`` is set, so it talks ``generateContent`` to the
gateway (`GeminiApiProtocol`) with the attempt token as its API key; no Google
account or Google key is involved, and Agency routes the requests to whatever
backend the agent is configured with.

Safety monitors: before running a tool, agy has small models
(``gemini-3.1-flash-lite-preview``, then ``gemini-3.8-flash``) rate the pending
call against its ``REVIEW_RUBRIC``; no setting turns this off. Through the
gateway each review would be a full call to the agent's own model, which made
single tool steps take minutes with a large reasoning model. Agency answers
these reviews itself with an untriggered verdict; tool admission is Agency's
policy (below) and the sandbox, as for every other harness. A request counts as
a review only when its last turn carries the rubric and it is addressed to a
model other than the one the agent itself runs under, so a conversation that
merely mentions the rubric still reaches the agent's model.

Tools and policy: Agency's MCP servers go in ``~/.gemini/config/mcp_config.json``;
``PreToolUse``/``PostToolUse`` command hooks in ``~/.gemini/config/hooks.json``
run the shared `_harness_permission_hook.py`, whose decisions apply in every
permission mode (agy runs with ``--dangerously-skip-permissions``).
"""

from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path

from .base import AdapterRuntime, AttemptResult, HarnessAdapter
from .gemini_api import GeminiApiProtocol
from ..pins import managed_binary
from .pty.execution import restore_session, snapshot_session


def antigravity_available() -> bool:
    return managed_binary("antigravity") is not None


# agy validates --model against its own catalog and tunes its prompts to it.
# The real model is whatever Agency routes to, so a non-Gemini model runs
# under agy's default Gemini profile.
_DEFAULT_CLI_MODEL = "gemini-4-carbon"
_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_CLI_HOME = ".gemini/antigravity-cli"
_POLL_INTERVAL_S = 0.2
_MONITOR_MARKER = "<REVIEW_RUBRIC>"
_MONITOR_VERDICT = json.dumps(
    {
        "reason": "Tool admission is decided by Agency's policy, not this monitor.",
        "severity": 1,
        "confidence": 9,
    }
)
_REAP_TIMEOUT_S = 30.0


def cli_model_for(model: "str | None") -> str:
    """The agy ``--model`` id for an Agency model name (``google/gemini-4-carbon``
    -> ``gemini-4-carbon``)."""
    base = (model or "").rsplit("/", 1)[-1]
    return base if base.startswith("gemini-") else _DEFAULT_CLI_MODEL


def parse_stream_json(stdout: str) -> "tuple[dict | None, str | None]":
    """(the final ``result`` payload, the conversation id) from agy's NDJSON."""
    result, conversation = None, None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        conversation = event.get("conversation_id") or conversation
        if event.get("event") == "result":
            result = event.get("result")
            if not isinstance(result, dict):
                result = {}
            conversation = result.get("conversation_id") or conversation
    return result, conversation


def is_monitor_request(raw_request: dict, target_model: str, agent_model: "str | None") -> bool:
    """Whether a request is one of agy's tool-safety monitor reviews: its last
    turn carries the rubric, and it is sent to one of agy's small monitor models
    rather than to *agent_model*, the Agency model the agent's own turns use."""
    if target_model.rsplit("/", 1)[-1] == cli_model_for(agent_model):
        return False
    contents = raw_request.get("contents") or []
    last = contents[-1] if contents else {}
    parts = last.get("parts") if isinstance(last, dict) else None
    return any(
        isinstance(part, dict) and _MONITOR_MARKER in (part.get("text") or "")
        for part in parts or []
    )


class AntigravityAdapter(GeminiApiProtocol, HarnessAdapter):
    _DEFAULT_BINARY = "antigravity"
    _HARNESS = "antigravity"
    _SESSION_NAME = "antigravity"

    def _local_response(
        self, raw_request: dict, target_model: str, agent_model: "str | None"
    ) -> "list[dict] | None":
        if is_monitor_request(raw_request, target_model, agent_model):
            return [{"text": _MONITOR_VERDICT}]
        return None

    def run_daemon_attempt(
        self,
        runtime: AdapterRuntime,
        *,
        prompt: str,
        resume_session_id: "str | None",
        prior_session_blob: "bytes | None",
        max_steps: "int | None",
    ) -> AttemptResult:
        from .. import agharness
        from ..executable import HARNESS_PATH
        from ..ptrace.supervisor import agProxyPtrace

        root = agharness.materialize_config_home(runtime.engine_name)
        handle = None
        try:
            restore_session(root, self._SESSION_NAME, resume_session_id, prior_session_blob)
            self._write_config(root, runtime)
            argv = [
                self.binary(),
                "-p",
                prompt,
                "--output-format",
                "stream-json",
                "--model",
                cli_model_for(runtime.model),
                "--dangerously-skip-permissions",
                "--disable-slash-commands",
            ]
            effort = runtime.agconfig.llm.reasoning_effort
            if effort in _EFFORTS:
                argv += ["--effort", effort]
            if resume_session_id:
                argv += ["--conversation", resume_session_id]
            envp = {
                "PATH": HARNESS_PATH,
                "HOME": str(root),
                "TERM": "dumb",
                "AGY_CLI_DISABLE_AUTO_UPDATE": "true",
                "GEMINI_API_KEY": runtime.token,
                "GOOGLE_GEMINI_BASE_URL": runtime.harness_base_url,
                "AGPOLICY_BASE_URL": runtime.harness_base_url,
                "AGPOLICY_TOKEN": runtime.token,
                "AGPOLICY_STATE_DIR": str(root),
                # The hook answers in agy's form even for input it cannot read.
                "AGPOLICY_HOOK_DIALECT": "antigravity",
            }
            handle = agProxyPtrace(runtime.agconfig, allow_initial_exec=True).launch(
                argv, envp, cwd="/workspace", policy=runtime.syscall_policy, ag=None
            )
            runtime.register_control_handle(handle)
            while handle.returncode is None:
                time.sleep(_POLL_INTERVAL_S)
            handle.kill()
            stdout, stderr, rc = handle.wait(timeout=_REAP_TIMEOUT_S)

            result, conversation = parse_stream_json(stdout)
            if result is None or result.get("status") != "SUCCESS":
                detail = (result or {}).get("error") or stderr.strip() or stdout.strip()
                if not isinstance(detail, str):
                    detail = json.dumps(detail)
                return AttemptResult(
                    ok=False,
                    error_message=f"antigravity exited with code {rc}: {detail[-4000:]}",
                )
            session_blob = None
            if conversation:
                # Like the terminal adapters, a conversation that cannot be
                # carried forward fails the attempt: returning success without
                # it would send the engine's next prompt to a new, empty one.
                try:
                    session_blob = snapshot_session(root, self._SESSION_NAME, conversation)
                except (RuntimeError, ValueError) as exc:
                    return AttemptResult(
                        ok=False,
                        error_message=(
                            f"antigravity finished but its conversation could not be saved: {exc}"
                        ),
                    )
            usage = result.get("usage") or {}
            return AttemptResult(
                ok=True,
                final_text=result.get("response") or "",
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                session_id=conversation if session_blob is not None else None,
                session_blob=session_blob,
            )
        finally:
            try:
                if handle is not None:
                    handle.close()
            finally:
                agharness.cleanup_config_home(root)

    def _write_config(self, root: Path, runtime: AdapterRuntime) -> None:
        from .. import agharness

        cli_home = root / _CLI_HOME
        cli_home.mkdir(parents=True, exist_ok=True)
        # The Gemini API provider is what honours GOOGLE_GEMINI_BASE_URL.
        (cli_home / "settings.json").write_text(json.dumps({"modelProvider": "gemini"}))

        config_dir = root / ".gemini" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        servers = agharness.mcp_config_for(
            runtime.harness_base_url,
            runtime.token,
            has_sandbox_mcp_tools=runtime.has_sandbox_mcp_tools,
        )["mcpServers"]
        (config_dir / "mcp_config.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        name: {"serverUrl": server["url"], "headers": server["headers"]}
                        for name, server in servers.items()
                    }
                }
            )
        )

        hook = root / "agpolicy_hook.py"
        hook.write_bytes(
            (Path(__file__).parent.parent / "_harness_permission_hook.py").read_bytes()
        )
        command = "python3 " + shlex.quote(str(hook))
        handler = {
            "matcher": ".*",
            "hooks": [{"type": "command", "command": command, "timeout": 30}],
        }
        (config_dir / "hooks.json").write_text(
            json.dumps({"agency": {"PreToolUse": [handler], "PostToolUse": [handler]}})
        )
        os.chmod(hook, 0o644)


__all__ = [
    "AntigravityAdapter",
    "antigravity_available",
    "cli_model_for",
    "is_monitor_request",
    "parse_stream_json",
]
