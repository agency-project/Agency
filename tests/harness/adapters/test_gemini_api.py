"""Gemini API (generateContent) seam: request and response translation."""

from __future__ import annotations

import base64
import json

from agency.configs.agconfig import agconfig
from agency.harness.adapters.antigravity import AntigravityAdapter
from agency.llm.message_store import split_tool_schema


def _adapter():
    return AntigravityAdapter(agconfig())


def _b64(text):
    return base64.b64encode(text.encode()).decode()


def test_request_maps_system_turns_tools_and_signatures():
    request = {
        "systemInstruction": {"parts": [{"text": "be brief"}]},
        "contents": [
            {"role": "user", "parts": [{"text": "list files"}]},
            {
                "role": "model",
                "parts": [
                    {"text": "planning", "thought": True, "thoughtSignature": _b64("sig-think")},
                    {
                        "functionCall": {"id": "c1", "name": "run_command", "args": {"cmd": "ls"}},
                        "thoughtSignature": _b64("openai_chat_tool_extra:{}"),
                    },
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "id": "c1",
                            "name": "run_command",
                            "response": {"output": "a.txt"},
                        }
                    }
                ],
            },
        ],
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "run_command",
                        "description": "run",
                        "parametersJsonSchema": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                        },
                    },
                    {
                        "name": "legacy",
                        "parameters": {"type": "OBJECT", "properties": {"x": {"type": "STRING"}}},
                    },
                ]
            }
        ],
        "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
    }
    messages, tools, tool_choice, _extra = split_tool_schema(
        _adapter()._format_context_harness_to_agency(request)["messages"]
    )
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[0]["blocks"][0]["text"] == "be brief"
    thinking, call = messages[2]["blocks"]
    assert thinking == {
        "type": "thinking",
        "index": 0,
        "text": "planning",
        "signature": "sig-think",
    }
    assert call["type"] == "tool_use" and call["id"] == "c1" and call["name"] == "run_command"
    assert json.loads(call["arguments"]) == {"cmd": "ls"}
    assert call["signature"] == "openai_chat_tool_extra:{}"
    result = messages[3]["blocks"][0]
    assert result == {"type": "tool_result", "index": 0, "tool_call_id": "c1", "text": "a.txt"}
    assert [t["function"]["name"] for t in tools] == ["run_command", "legacy"]
    assert tools[1]["function"]["parameters"] == {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }
    assert tool_choice == "required"


def test_calls_without_ids_are_answered_by_name_in_order():
    request = {
        "contents": [
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "view_file", "args": {"Path": "a"}}},
                    {"functionCall": {"name": "view_file", "args": {"Path": "b"}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "view_file", "response": {"x": 1}}},
                    {"functionResponse": {"name": "view_file", "response": {"x": 2}}},
                ],
            },
        ]
    }
    messages, *_ = split_tool_schema(
        _adapter()._format_context_harness_to_agency(request)["messages"]
    )
    ids = [b["id"] for b in messages[0]["blocks"]]
    assert len(set(ids)) == 2
    assert [m["blocks"][0]["tool_call_id"] for m in messages[1:]] == ids
    assert json.loads(messages[1]["blocks"][0]["text"]) == {"x": 1}


def test_unknown_parts_round_trip_verbatim():
    part = {"inlineData": {"mimeType": "image/png", "data": "AAAA"}}
    messages, *_ = split_tool_schema(
        _adapter()._format_context_harness_to_agency(
            {"contents": [{"role": "user", "parts": [part]}]}
        )["messages"]
    )
    block = messages[0]["blocks"][0]
    assert block["type"] == "gemini_inlineData"
    assert _adapter()._parts_from_agency({"blocks": [block]}) == [part]


def test_response_maps_blocks_usage_and_signature_round_trip():
    adapter = _adapter()
    response = {
        "message": {
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "text": "hmm"},
                {"type": "text", "text": "running it"},
                {
                    "type": "tool_use",
                    "id": "call_9",
                    "name": "run_command",
                    "arguments": '{"cmd": "ls"}',
                    "signature": 'openai_chat_tool_extra:{"google": 1}',
                },
            ],
        },
        "stop_reason": "tool_use",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }
    out = adapter._format_context_agency_to_harness(response, "m")
    candidate = out["candidates"][0]
    assert candidate["finishReason"] == "STOP"
    thought, text, call = candidate["content"]["parts"]
    assert thought == {"text": "hmm", "thought": True}
    assert text == {"text": "running it"}
    assert call["functionCall"] == {"id": "call_9", "name": "run_command", "args": {"cmd": "ls"}}
    assert out["usageMetadata"] == {
        "promptTokenCount": 100,
        "candidatesTokenCount": 20,
        "totalTokenCount": 120,
    }
    # The signature comes back to Agency exactly as it left.
    echoed = adapter._format_context_harness_to_agency(
        {"contents": [{"role": "model", "parts": [call]}]}
    )
    block = split_tool_schema(echoed["messages"])[0][0]["blocks"][0]
    assert block["signature"] == 'openai_chat_tool_extra:{"google": 1}'


def test_max_tokens_and_stream_frame():
    adapter = _adapter()
    item = {
        "type": "done",
        "message": {"role": "assistant", "blocks": [{"type": "text", "text": "cut"}]},
        "stop_reason": "max_tokens",
        "usage": {},
    }
    frames = list(adapter._format_agency_stream_to_harness(iter([{"type": "delta"}, item]), "m"))
    assert len(frames) == 1 and frames[0].startswith("data: ") and frames[0].endswith("\r\n\r\n")
    payload = json.loads(frames[0][len("data: ") :])
    assert payload["candidates"][0]["finishReason"] == "MAX_TOKENS"
    assert payload["candidates"][0]["content"]["parts"] == [{"text": "cut"}]


def test_placeholder_signature_is_not_forwarded():
    request = {
        "contents": [
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {"id": "c1", "name": "f", "args": {}},
                        "thoughtSignature": "skip_thought_signature_validator",
                    }
                ],
            }
        ]
    }
    messages, *_ = split_tool_schema(
        _adapter()._format_context_harness_to_agency(request)["messages"]
    )
    assert "signature" not in messages[0]["blocks"][0]


def test_keepalive_is_an_empty_response_chunk_not_an_sse_comment():
    from agency.harness.adapters.gemini_api import GEMINI_KEEPALIVE_FRAME

    assert GEMINI_KEEPALIVE_FRAME.startswith("data: ")
    assert json.loads(GEMINI_KEEPALIVE_FRAME[len("data: ") :]) == {"candidates": []}


class _Router:
    def __init__(self):
        self.dispatched = []

    def validate_token(self, token):
        return token == "tok"

    def resolve_model(self, token):
        return "backend-model"

    def dispatch(self, token, context):
        self.dispatched.append(context)
        return {
            "message": {"role": "assistant", "blocks": [{"type": "text", "text": "hi"}]},
            "stop_reason": "end_turn",
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }


def _client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app, router = FastAPI(), _Router()
    _adapter().register(app, router)
    return TestClient(app), router


def test_route_authenticates_with_the_api_key_header_and_dispatches():
    client, router = _client()
    body = {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}
    path = "/v1beta/models/gemini-4-carbon:generateContent"
    assert client.post(path, json=body, headers={"x-goog-api-key": "nope"}).status_code == 401
    reply = client.post(path, json=body, headers={"x-goog-api-key": "tok"}).json()
    assert reply["candidates"][0]["content"]["parts"] == [{"text": "hi"}]
    assert reply["usageMetadata"]["promptTokenCount"] == 3
    assert len(router.dispatched) == 1
    count = client.post(
        "/v1beta/models/gemini-4-carbon:countTokens", json=body, headers={"x-goog-api-key": "tok"}
    )
    assert count.json()["totalTokens"] >= 1


def test_safety_monitor_reviews_are_answered_without_a_model_call():
    client, router = _client()
    review = {
        "contents": [{"role": "user", "parts": [{"text": "<REVIEW_RUBRIC>rate</REVIEW_RUBRIC>"}]}]
    }
    response = client.post(
        "/v1beta/models/gemini-3.1-flash-lite-preview:streamGenerateContent?alt=sse",
        json=review,
        headers={"x-goog-api-key": "tok"},
    )
    assert response.status_code == 200 and response.text.startswith("data: ")
    payload = json.loads(response.text[len("data: ") :])
    verdict = json.loads(payload["candidates"][0]["content"]["parts"][0]["text"])
    assert verdict["severity"] == 1
    assert router.dispatched == []
