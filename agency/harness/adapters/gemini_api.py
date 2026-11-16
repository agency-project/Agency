"""Gemini API (``generateContent``) seam for harnesses that speak Google's protocol.

Like `openai_chat_completions`, the translation is protocol-level: a CLI whose
Gemini client is pointed at Agency's gateway (``GOOGLE_GEMINI_BASE_URL``) sends
``POST /v1beta/models/<model>:streamGenerateContent`` (or ``:generateContent``)
with its API key in ``x-goog-api-key``; that key is the attempt's bearer token.

Agency blocks map onto Gemini parts as follows:

* ``text`` <-> ``{"text"}``; ``thinking`` <-> ``{"text", "thought": true}``.
* ``tool_use`` <-> ``{"functionCall": {"id", "name", "args"}}``.
* ``tool_result`` <-> ``{"functionResponse": {"id", "name", "response"}}``, as
  one ``tool`` message per response.
* A block's opaque ``signature`` travels as the part's ``thoughtSignature``,
  base64 of its UTF-8 text (the field is bytes on the wire), so whatever the
  backend needs replayed comes back exactly as it left.
* Any other part kind is kept as a ``gemini_<kind>`` block and replayed verbatim.
"""

from __future__ import annotations

import anyio
import base64
import binascii
import json
import uuid

from fastapi import Request

from ...llm.message_store import with_tool_schema
from ..common import extract_bearer_token
from .streaming import start_streaming_response, stream_response, upstream_error

# Google's Go SDK rejects SSE comment lines ("invalid stream chunk: : keepalive"),
# so an idle stream is kept open with an empty response chunk instead.
GEMINI_KEEPALIVE_FRAME = 'data: {"candidates": []}\r\n\r\n'
_NATIVE_PREFIX = "gemini_"
_KNOWN_PART_KEYS = {"text", "thought", "thoughtSignature", "functionCall", "functionResponse"}

_FINISH_REASON = {"max_tokens": "MAX_TOKENS", "refusal": "SAFETY"}

_FUNCTION_CALLING_MODE = {"AUTO": "auto", "ANY": "required", "NONE": "none"}


def gemini_error_frame(exc: BaseException) -> str:
    message, status, _transient = upstream_error(exc)
    error = {"code": status, "message": message, "status": "UNAVAILABLE"}
    return f"data: {json.dumps({'error': error})}\r\n\r\n"


# What a Gemini client sends back for a call that came with no signature.
_PLACEHOLDER_SIGNATURE = "skip_thought_signature_validator"


def _decode_signature(value) -> "str | None":
    if not isinstance(value, str) or not value or value == _PLACEHOLDER_SIGNATURE:
        return None
    try:
        return base64.b64decode(value, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        # Not one of ours (e.g. a real Gemini signature): keep it as sent.
        return value


def _encode_signature(value) -> "str | None":
    if not isinstance(value, str) or not value:
        return None
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _lower_schema_types(schema):
    """Gemini's OpenAPI-style ``parameters`` spell types in upper case
    (``"STRING"``); JSON Schema, which every backend expects, does not."""
    if isinstance(schema, dict):
        out = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, str):
                out[key] = value.lower()
            else:
                out[key] = _lower_schema_types(value)
        return out
    if isinstance(schema, list):
        return [_lower_schema_types(item) for item in schema]
    return schema


def _gemini_tools_to_agency(tools) -> "list[dict] | None":
    converted = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        for decl in tool.get("functionDeclarations") or []:
            if "parametersJsonSchema" in decl:
                parameters = decl["parametersJsonSchema"]
            else:
                parameters = _lower_schema_types(decl.get("parameters"))
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": decl.get("name", ""),
                        "description": decl.get("description", ""),
                        "parameters": parameters or {"type": "object", "properties": {}},
                    },
                }
            )
    return converted or None


def _gemini_tool_choice_to_agency(tool_config):
    config = (tool_config or {}).get("functionCallingConfig") or {}
    mode = _FUNCTION_CALLING_MODE.get(str(config.get("mode", "")).upper())
    names = config.get("allowedFunctionNames") or []
    if mode == "required" and len(names) == 1:
        return {"type": "function", "function": {"name": names[0]}}
    return mode


def _response_text(response) -> str:
    """A function's result as the text a tool result carries."""
    if isinstance(response, dict):
        for key in ("output", "result", "content"):
            if set(response) == {key} and isinstance(response[key], str):
                return response[key]
    if isinstance(response, str):
        return response
    return json.dumps(response)


def _usage_to_gemini(usage: "dict | None") -> dict:
    usage = usage or {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    output = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    meta = {
        "promptTokenCount": prompt,
        "candidatesTokenCount": output,
        "totalTokenCount": usage.get("total_tokens") or prompt + output,
    }
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning:
        meta["thoughtsTokenCount"] = reasoning
    cached = usage.get("cache_read_tokens") or (usage.get("prompt_tokens_details") or {}).get(
        "cached_tokens"
    )
    if cached:
        meta["cachedContentTokenCount"] = cached
    return meta


class GeminiApiProtocol:
    """Mixin supplying the Gemini ``generateContent`` routes and their block mapping."""

    def register(self, app, router) -> None:
        from fastapi.responses import JSONResponse

        async def generate(request: Request, target: str):
            token = (
                request.headers.get("x-goog-api-key")
                or request.query_params.get("key")
                or extract_bearer_token(request)
            )
            if not token or not router.validate_token(token):
                return JSONResponse(
                    {"error": {"code": 401, "message": "unknown or missing API key"}},
                    status_code=401,
                )
            target_model, _, method = target.rpartition(":")
            body = await request.json()
            if method == "countTokens":
                # No backend counts Gemini tokens for us; the CLI only budgets
                # with this, so a size estimate is enough.
                return JSONResponse({"totalTokens": max(1, len(json.dumps(body)) // 4)})
            if method not in ("generateContent", "streamGenerateContent"):
                return JSONResponse(
                    {"error": {"code": 404, "message": f"unsupported method {method!r}"}},
                    status_code=404,
                )
            model = router.resolve_model(token)
            local = self._local_response(body, target_model, model)
            if local is not None:
                return _local_reply(local, model, stream=method == "streamGenerateContent")
            agency_context = self._format_context_harness_to_agency(body)
            if method == "streamGenerateContent":
                frames = stream_response(
                    router,
                    token,
                    agency_context,
                    model,
                    self._format_agency_stream_to_harness,
                    keepalive_frame=GEMINI_KEEPALIVE_FRAME,
                    keepalive_s=self.agconfig.harness_adapter.stream_keepalive_s,
                    error_frame=gemini_error_frame,
                )
                return await start_streaming_response(request, frames)
            agency_response = await anyio.to_thread.run_sync(router.dispatch, token, agency_context)
            return JSONResponse(self._format_context_agency_to_harness(agency_response, model))

        for version in ("v1beta", "v1"):
            app.add_api_route(f"/{version}/models/{{target:path}}", generate, methods=["POST"])

    def _local_response(
        self, raw_request: dict, target_model: str, agent_model: "str | None"
    ) -> "list[dict] | None":
        """Parts to answer a CLI's auxiliary request with, without a model call;
        None (the default) sends the request to the agent's backend.
        *target_model* is the model the CLI addressed, *agent_model* the Agency
        model the agent runs on."""
        return None

    def _format_context_harness_to_agency(self, raw_request: dict) -> dict:
        messages: "list[dict]" = []
        system = raw_request.get("systemInstruction") or raw_request.get("system_instruction")
        if isinstance(system, dict):
            texts = [p.get("text", "") for p in system.get("parts") or [] if "text" in p]
            if texts:
                messages.append(
                    {
                        "role": "system",
                        "blocks": [{"type": "text", "index": 0, "text": "\n".join(texts)}],
                    }
                )

        # Calls without an id are answered by name, in order.
        unanswered: "dict[str, list[str]]" = {}
        for turn, content in enumerate(raw_request.get("contents") or []):
            role = "assistant" if content.get("role") == "model" else "user"
            blocks: "list[dict]" = []
            for position, part in enumerate(content.get("parts") or []):
                if not isinstance(part, dict):
                    continue
                signature = _decode_signature(part.get("thoughtSignature"))
                if "functionCall" in part:
                    call = part["functionCall"] or {}
                    call_id = call.get("id") or f"call_{turn}_{position}"
                    unanswered.setdefault(call.get("name", ""), []).append(call_id)
                    block = {
                        "type": "tool_use",
                        "index": len(blocks),
                        "id": call_id,
                        "name": call.get("name", ""),
                        "arguments": json.dumps(call.get("args") or {}),
                    }
                elif "functionResponse" in part:
                    response = part["functionResponse"] or {}
                    name = response.get("name", "")
                    call_id = response.get("id")
                    pending = unanswered.get(name) or []
                    if call_id in pending:
                        pending.remove(call_id)
                    elif not call_id:
                        call_id = pending.pop(0) if pending else f"call_{turn}_{position}"
                    messages.append(
                        {
                            "role": "tool",
                            "blocks": [
                                {
                                    "type": "tool_result",
                                    "index": 0,
                                    "tool_call_id": call_id,
                                    "text": _response_text(response.get("response")),
                                }
                            ],
                        }
                    )
                    continue
                elif "text" in part:
                    block = {
                        "type": "thinking" if part.get("thought") else "text",
                        "index": len(blocks),
                        "text": part.get("text") or "",
                    }
                else:
                    kind = next((k for k in part if k not in _KNOWN_PART_KEYS), "part")
                    block = {
                        "type": f"{_NATIVE_PREFIX}{kind}",
                        "index": len(blocks),
                        "data": part,
                    }
                    signature = None
                if signature:
                    block["signature"] = signature
                blocks.append(block)
            if blocks:
                messages.append({"role": role, "blocks": blocks})

        return {
            "messages": with_tool_schema(
                messages,
                _gemini_tools_to_agency(raw_request.get("tools")),
                _gemini_tool_choice_to_agency(raw_request.get("toolConfig")),
            )
        }

    def _parts_from_agency(self, message: dict) -> "list[dict]":
        parts = []
        for b in message.get("blocks", []):
            kind = b.get("type")
            if kind == "text":
                part = {"text": b.get("text", "")}
            elif kind == "thinking":
                part = {"text": b.get("text", ""), "thought": True}
            elif kind == "tool_use":
                try:
                    args = json.loads(b.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {"value": args}
                part = {
                    "functionCall": {"id": b.get("id", ""), "name": b.get("name", ""), "args": args}
                }
            elif isinstance(kind, str) and kind.startswith(_NATIVE_PREFIX):
                data = b.get("data")
                if isinstance(data, dict):
                    parts.append(data)
                continue
            else:
                continue
            signature = _encode_signature(b.get("signature"))
            if signature:
                part["thoughtSignature"] = signature
            parts.append(part)
        return parts

    def _format_context_agency_to_harness(self, agency_response: dict, model: str) -> dict:
        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": self._parts_from_agency(agency_response["message"]),
                    },
                    "finishReason": _FINISH_REASON.get(agency_response.get("stop_reason"), "STOP"),
                    "index": 0,
                }
            ],
            "usageMetadata": _usage_to_gemini(agency_response.get("usage")),
            "modelVersion": model,
            "responseId": uuid.uuid4().hex,
        }

    def _format_agency_stream_to_harness(self, agency_stream, model: str):
        for item in agency_stream:
            if item["type"] == "delta":
                # As with Chat Completions: a redirect can replace the draft,
                # so only the authoritative message is published.
                continue
            payload = self._format_context_agency_to_harness(item, model)
            yield f"data: {json.dumps(payload)}\r\n\r\n"
            return


def _local_reply(parts: "list[dict]", model: str, *, stream: bool):
    """A response the gateway makes up without a model call."""
    from fastapi.responses import JSONResponse, Response

    payload = {
        "candidates": [
            {"content": {"role": "model", "parts": parts}, "finishReason": "STOP", "index": 0}
        ],
        "usageMetadata": _usage_to_gemini(None),
        "modelVersion": model,
    }
    if stream:
        return Response(f"data: {json.dumps(payload)}\r\n\r\n", media_type="text/event-stream")
    return JSONResponse(payload)


__all__ = ["GeminiApiProtocol", "gemini_error_frame"]
