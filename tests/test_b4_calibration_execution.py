from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.b4_calibration_execution import AnthropicJudgeAdapter, OpenAIJudgeAdapter


class _Responses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _adapter(response, max_output_tokens=436):
    responses = _Responses(response=response)
    client = SimpleNamespace(responses=responses)
    return OpenAIJudgeAdapter("gpt-5.6-sol", client=client, max_output_tokens=max_output_tokens), responses


def test_openai_no_text_preserves_diagnostics():
    response = {
        "id": "resp_1",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "usage": {"input_tokens": 100, "output_tokens": 436, "total_tokens": 536},
        "output": [{"type": "reasoning"}],
    }
    adapter, responses = _adapter(response)
    with pytest.raises(ValueError) as raised:
        adapter.call("BLIND", "memora")
    assert str(raised.value) == "provider response contained no text"
    diagnostics = raised.value.diagnostics
    assert diagnostics["status"] == "incomplete"
    assert diagnostics["incomplete_reason"] == "max_output_tokens"
    assert diagnostics["incomplete_details"] == {"reason": "max_output_tokens"}
    assert diagnostics["response_id"] == "resp_1"
    assert (diagnostics["input_tokens"], diagnostics["output_tokens"], diagnostics["total_tokens"]) == (100, 436, 536)
    assert diagnostics["configured_max_output_tokens"] == 436
    assert diagnostics["output_item_types"] == ["reasoning"]
    assert diagnostics["content_block_types"] == [] and diagnostics["text_block_count"] == 0
    assert diagnostics["has_raw_text"] is False and diagnostics["raw_text_char_count"] == 0
    assert diagnostics["json_parse_ok"] is False
    assert adapter.last_diagnostics == diagnostics
    assert responses.calls[0]["max_output_tokens"] == 436
    assert responses.calls[0]["model"] == "gpt-5.6-sol"
    assert responses.calls[0]["input"] == [{"role": "user", "content": "BLIND"}]


def test_openai_partial_visible_text_shape_is_captured():
    raw = '{"decision": "pass", "confidence": "high", "short_reason": "trunc'
    response = {
        "id": "resp_2",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "usage": {"input_tokens": 10, "output_tokens": 436},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": raw}]}],
    }
    adapter, _ = _adapter(response)
    assert adapter.call("BLIND", "memora") == raw
    diagnostics = adapter.last_diagnostics
    assert diagnostics["json_parse_ok"] is False
    assert diagnostics["has_raw_text"] is True and diagnostics["raw_text_char_count"] == len(raw)
    assert diagnostics["raw_text_preview"] == raw
    assert diagnostics["text_block_count"] == 1
    assert diagnostics["output_item_types"] == ["message"] and diagnostics["content_block_types"] == ["output_text"]


def test_openai_diagnostics_never_persist_hidden_reasoning():
    secret = "SECRET-CHAIN-OF-THOUGHT-TOKEN"
    response = {
        "id": "resp_3",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "usage": {"input_tokens": 1, "output_tokens": 436},
        "output": [{
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": secret}],
            "content": [{"type": "reasoning_text", "thinking": secret}],
        }],
    }
    adapter, _ = _adapter(response)
    with pytest.raises(ValueError):
        adapter.call("BLIND", "memora")
    assert secret not in json.dumps(adapter.last_diagnostics)
    assert adapter.last_diagnostics["output_item_types"] == ["reasoning"]
    assert adapter.last_diagnostics["text_block_count"] == 0
    assert adapter.last_diagnostics["has_raw_text"] is False


def test_openai_valid_response_diagnostics_and_request_shape():
    body = json.dumps({"decision": "pass", "confidence": "high", "short_reason": "met"})
    response = {
        "id": "resp_4",
        "status": "completed",
        "usage": {"input_tokens": 5, "output_tokens": 6},
        "output_text": body,
        "output": [{"type": "message", "content": [{"type": "output_text", "text": body}]}],
    }
    adapter, responses = _adapter(response, max_output_tokens=1024)
    assert adapter.call("BLIND", "memora") == body
    diagnostics = adapter.last_diagnostics
    assert diagnostics["status"] == "completed"
    assert diagnostics["json_parse_ok"] is True
    assert diagnostics["has_raw_text"] is True and diagnostics["raw_text_char_count"] == len(body)
    assert diagnostics["configured_max_output_tokens"] == 1024
    assert diagnostics["total_tokens"] == 11
    request = responses.calls[0]
    assert request["max_output_tokens"] == 1024
    assert request["text"]["format"]["type"] == "json_schema"
    assert "reasoning" not in json.dumps(diagnostics).lower()


def test_openai_diagnostics_do_not_change_error_class_or_message():
    response = {"output": [{"type": "reasoning"}], "usage": {"output_tokens": 436}}
    adapter, _ = _adapter(response)
    with pytest.raises(ValueError) as raised:
        adapter.call("BLIND", "memora")
    assert type(raised.value) is ValueError
    assert str(raised.value) == "provider response contained no text"


class _AnthropicMessages:
    def __init__(self, response=None):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_anthropic_no_text_thinking_only_max_tokens_preserves_diagnostics():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="thinking")],
        stop_reason="max_tokens",
        usage=SimpleNamespace(input_tokens=6279, output_tokens=2236, total_tokens=8515),
        request_id="req-escalation",
    )
    messages = _AnthropicMessages(response=response)
    adapter = AnthropicJudgeAdapter("claude-sonnet-5", client=SimpleNamespace(messages=messages))
    with pytest.raises(ValueError) as raised:
        adapter.call("BLIND", "memora")
    assert str(raised.value) == "provider response contained no text"
    diagnostics = raised.value.diagnostics
    assert diagnostics["content_block_types"] == ["thinking"]
    assert diagnostics["text_block_count"] == 0
    assert diagnostics["has_raw_text"] is False and diagnostics["raw_text_char_count"] == 0
    assert diagnostics["stop_reason"] == "max_tokens"
    assert diagnostics["output_tokens"] == 2236
    assert diagnostics["input_tokens"] == 6279 and diagnostics["total_tokens"] == 8515
    assert diagnostics["json_parse_ok"] is False
    assert adapter.last_diagnostics == diagnostics
    assert messages.calls[0]["model"] == "claude-sonnet-5"
