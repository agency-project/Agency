"""DeepSeek Harness (``dsh``) adapter, driven through its headless profile.

Verified against DeepSeek Harness 0.2.2-rc.2 (the standalone runtime from the
``deepseek-harness-runtime-bin`` wheel). Each attempt runs one
``dsh --profile headless --json <prompt>`` process under the same ptrace
supervision as every other adapter and reads its final ``final`` event; a later
attempt resumes with ``--session-id`` from the restored session files.

Everything Agency changes is one Cordis patch layer passed with ``--patch``:

* model traffic goes to the gateway as a custom ``openai-completions`` provider
  (served by the shared `ChatCompletionsProtocol`), keyed by the attempt token;
* Agency's MCP servers are ``@deepseek-ai/dsh-mcp-client`` rows;
* ``_dsh_agpolicy_plugin.mjs`` puts Agency's tool policy on dsh's
  ``tools/pre-execute``/``tools/post-execute`` seams (the runtime bundles no
  command-hook bridge);
* DeepSeek's session-log upload and the title model call are switched off, and
  telemetry is disabled through ``DSH_TELEMETRY_MODE``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .base import AdapterRuntime, AttemptResult, HarnessAdapter, fetch_context_limit
from .openai_chat_completions import ChatCompletionsProtocol
from ..pins import managed_binary
from .pty.execution import restore_session, snapshot_session


def deepseek_harness_available() -> bool:
    return managed_binary("dsh") is not None


_PROVIDER = "agency"
_POLL_INTERVAL_S = 0.2
_REAP_TIMEOUT_S = 30.0
_PLUGIN = "_dsh_agpolicy_plugin.mjs"


def parse_json_events(stdout: str) -> "tuple[dict | None, int, int]":
    """(the ``final`` event, summed input tokens, summed output tokens) from
    ``dsh --json``'s newline-delimited run events."""
    final, input_tokens, output_tokens = None, 0, 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "final":
            final = event
        elif event.get("type") == "status" and event.get("phase") == "step_end":
            usage = event.get("usage") or {}
            input_tokens += usage.get("inputTokens", 0) or 0
            output_tokens += usage.get("outputTokens", 0) or 0
    return final, input_tokens, output_tokens


class DeepSeekHarnessAdapter(ChatCompletionsProtocol, HarnessAdapter):
    _DEFAULT_BINARY = "dsh"
    _HARNESS = "dsh"
    _SESSION_NAME = "dsh"

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
            patch = self._write_patch(
                root,
                runtime.harness_base_url,
                runtime.token,
                runtime.model or "default",
                has_sandbox_mcp_tools=runtime.has_sandbox_mcp_tools,
                context_limit=fetch_context_limit(runtime.harness_base_url, runtime.token),
            )
            argv = [self.binary(), "--profile", "headless", "--patch", str(patch), "--json"]
            if resume_session_id:
                argv += ["--session-id", resume_session_id]
            argv += ["--", prompt]
            envp = {
                "PATH": HARNESS_PATH,
                "HOME": str(root),
                "DSH_HOME": str(root),
                "TERM": "dumb",
                "AGENCY_TOKEN": runtime.token,
                # The sandbox is the isolation boundary; dsh's own approval
                # prompts would have no one to answer them.
                "DSH_PERMISSION_MODE": "danger-full-access",
                "DSH_TELEMETRY_MODE": "DISABLED",
                "AGPOLICY_BASE_URL": runtime.harness_base_url,
                "AGPOLICY_TOKEN": runtime.token,
            }
            handle = agProxyPtrace(runtime.agconfig, allow_initial_exec=True).launch(
                argv, envp, cwd="/workspace", policy=runtime.syscall_policy, ag=None
            )
            runtime.register_control_handle(handle)
            while handle.returncode is None:
                time.sleep(_POLL_INTERVAL_S)
            handle.kill()
            stdout, stderr, rc = handle.wait(timeout=_REAP_TIMEOUT_S)

            final, input_tokens, output_tokens = parse_json_events(stdout)
            reason = ((final or {}).get("reason") or {}).get("kind")
            if final is None or reason != "completed":
                detail = json.dumps(final) if final else (stderr.strip() or stdout.strip())
                return AttemptResult(
                    ok=False,
                    error_message=f"dsh exited with code {rc}: {detail[-4000:]}",
                )
            session_id = final.get("sessionId")
            session_blob = None
            if session_id:
                try:
                    session_blob = snapshot_session(root, self._SESSION_NAME, session_id)
                except (RuntimeError, ValueError):
                    session_blob = None
            return AttemptResult(
                ok=True,
                final_text=final.get("text") or "",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                session_id=session_id if session_blob is not None else None,
                session_blob=session_blob,
            )
        finally:
            try:
                if handle is not None:
                    handle.close()
            finally:
                agharness.cleanup_config_home(root)

    def _write_patch(
        self,
        root: Path,
        base_url: str,
        token: str,
        model: str,
        *,
        has_sandbox_mcp_tools: bool = False,
        context_limit: "int | None" = None,
    ) -> Path:
        """Write the per-attempt Cordis patch layer (JSON is valid YAML)."""
        from .. import agharness

        model_entry: dict = {"id": model}
        if context_limit is not None:
            model_entry["contextWindow"] = int(context_limit)
            model_entry["maxTokens"] = max(
                self.agconfig.llm.default_max_tokens, int(0.1 * context_limit)
            )
        plugin = root / _PLUGIN
        plugin.write_bytes((Path(__file__).parent / _PLUGIN).read_bytes())
        servers = agharness.mcp_config_for(
            base_url, token, has_sandbox_mcp_tools=has_sandbox_mcp_tools
        )["mcpServers"]
        inserts = [
            {
                "id": f"mcp-{name}",
                "name": "@deepseek-ai/dsh-mcp-client",
                "config": {
                    "serverName": name,
                    "transport": "streamable-http",
                    "url": server["url"],
                    "headers": server["headers"],
                    # Agency's tools must be listed before the first model call.
                    "failOnStartupError": True,
                    "startupTimeoutMs": 60_000,
                },
            }
            for name, server in servers.items()
        ]
        inserts.append({"id": "agency-agpolicy", "name": str(plugin)})
        patch = [
            {
                "id": "llm-pi-ai",
                "config": {
                    "providers": {
                        _PROVIDER: {
                            "apiKeyEnv": "AGENCY_TOKEN",
                            "api": "openai-completions",
                            "baseURL": f"{base_url}/v1",
                            # The gateway's Chat Completions route is not
                            # OpenAI itself: plain system role, max_tokens.
                            "compat": {
                                "supportsDeveloperRole": False,
                                "maxTokensField": "max_tokens",
                            },
                            "models": [model_entry],
                        }
                    }
                },
            },
            {"id": "agent-default-model", "config": {"provider": _PROVIDER, "model": model}},
            # DeepSeek's own session-log upload, and a second model call per
            # session for its title: neither belongs in an Agency run.
            {"id": "session-log-deepseek", "disabled": True},
            {"id": "session-title-llm", "disabled": True},
            {"insert": inserts},
        ]
        path = root / "agency.patch.yml"
        path.write_text(json.dumps(patch, indent=1))
        return path


__all__ = ["DeepSeekHarnessAdapter", "deepseek_harness_available", "parse_json_events"]
