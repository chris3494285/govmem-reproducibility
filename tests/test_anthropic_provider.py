from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from eval.openai_reconstruction import ReconstructionError, _ResponseSchema
from eval.reconstruction import GovernanceReconstructionInput
from eval.reconstruction_provider import ProviderTransportError
from eval.reconstruction_engine import GovernanceStateReconstructionEngine
from eval.interfaces import Query, RawInteraction
from eval.providers.anthropic_provider import AnthropicMessagesProvider, anthropic_transport_schema
from eval.openai_reconstruction import _cache_payload
from scripts.run_rq1_pilot import parse_args as parse_pilot_args


def _input() -> GovernanceReconstructionInput:
    return GovernanceReconstructionInput(
        raw_interactions=(RawInteraction("i0", "u", "memory", datetime(2025, 1, 1)),),
        query=Query("q", "u", "What state applies?", datetime(2025, 2, 1)),
        runtime_context={"user_id": "u", "allowed_scopes": ()},
    )


class _Messages:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _client(payload):
    response = SimpleNamespace(
        content=[SimpleNamespace(text=json.dumps(payload))],
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        request_id="anthropic-request",
        stop_reason="end_turn",
    )
    messages = _Messages(response=response)
    return SimpleNamespace(messages=messages), messages


def test_native_anthropic_transport_preserves_model_and_structured_payload():
    client, messages = _client({"states": [{"interaction_id": "i0", "is_deleted": None, "missing_fields": ["is_deleted"]}]})
    provider = AnthropicMessagesProvider(client=client, model="claude-opus-5")
    result = provider.request_structured_reconstruction(prompt="same prompt", schema=SimpleNamespace(model_json_schema=lambda: {"type": "object"}), request_metadata={})
    assert provider.provider_name == "anthropic"
    assert provider.model == "claude-opus-5"
    assert result.parsed["states"][0]["interaction_id"] == "i0"
    assert result.telemetry.input_tokens == 100
    assert result.telemetry.output_tokens == 20
    assert result.telemetry.total_tokens == 120
    assert result.telemetry.stop_reason == "end_turn"
    assert messages.calls[0]["model"] == "claude-opus-5"
    assert messages.calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_downstream_uses_exact_answer_schema_temperature_and_telemetry():
    client, messages = _client({"answer": "structured answer"})
    provider = AnthropicMessagesProvider(client=client, model="claude-opus-5", temperature=0)
    result = provider.request_downstream("same prompt")

    request = messages.calls[0]
    sent_schema = request["output_config"]["format"]["schema"]
    assert result.parsed == {"answer": "structured answer"}
    assert request["model"] == "claude-opus-5"
    assert request["temperature"] == 0
    assert set(sent_schema["properties"]) == {"answer"}
    assert sent_schema["properties"]["answer"] == {"title": "Answer", "type": "string"}
    assert sent_schema["required"] == ["answer"]
    assert result.telemetry.input_tokens == 100
    assert result.telemetry.output_tokens == 20
    assert result.telemetry.total_tokens == 120
    assert result.telemetry.request_id == "anthropic-request"
    assert result.telemetry.stop_reason == "end_turn"


def test_claude_sonnet_5_downstream_omits_deprecated_temperature():
    client, messages = _client({"answer": "structured answer"})
    provider = AnthropicMessagesProvider(client=client, model="claude-sonnet-5", temperature=0)
    provider.request_downstream("same prompt")
    assert "temperature" not in messages.calls[0]
    assert provider.requested_temperature == 0
    assert provider.effective_temperature is None


def test_anthropic_reconstruction_does_not_add_temperature_unless_configured():
    client, messages = _client({"states": []})
    AnthropicMessagesProvider(client=client, model="claude-opus-5").request_structured_reconstruction(
        prompt="prompt", schema=SimpleNamespace(model_json_schema=lambda: {"type": "object"}), request_metadata={}
    )
    assert "temperature" not in messages.calls[-1]

    AnthropicMessagesProvider(client=client, model="claude-opus-5", temperature=0.25).request_structured_reconstruction(
        prompt="prompt", schema=SimpleNamespace(model_json_schema=lambda: {"type": "object"}), request_metadata={}
    )
    assert messages.calls[-1]["temperature"] == 0.25

    sonnet = AnthropicMessagesProvider(client=client, model="claude-sonnet-5", temperature=0)
    sonnet.request_structured_reconstruction(
        prompt="prompt", schema=SimpleNamespace(model_json_schema=lambda: {"type": "object"}), request_metadata={}
    )
    assert "temperature" not in messages.calls[-1]


def test_anthropic_downstream_malformed_json_is_normalized():
    response = SimpleNamespace(
        content=[SimpleNamespace(text="not-json")],
        usage=None,
        request_id="anthropic-request",
        stop_reason="end_turn",
    )
    provider = AnthropicMessagesProvider(
        client=SimpleNamespace(messages=_Messages(response=response)), model="claude-sonnet-5", temperature=0
    )
    with pytest.raises(ProviderTransportError) as raised:
        provider.request_downstream("prompt")
    assert raised.value.details["error_category"] == "structural_invalid"


def test_anthropic_downstream_explicit_token_limit_is_nonretryable_and_retains_raw_usage():
    response = SimpleNamespace(
        content=[SimpleNamespace(text='{"answer":"partial')],
        usage=SimpleNamespace(input_tokens=12, output_tokens=7),
        request_id="anthropic-request",
        stop_reason="max_tokens",
        status="incomplete",
    )
    messages = _Messages(response=response)
    provider = AnthropicMessagesProvider(client=SimpleNamespace(messages=messages), model="claude-sonnet-5", temperature=0)
    with pytest.raises(ProviderTransportError) as raised:
        provider.request_downstream("prompt", max_tokens=7)
    assert raised.value.details["error_category"] == "token_limit_truncation"
    assert raised.value.details["token_limit_truncation"] is True
    assert raised.value.details["raw_output_text"] == '{"answer":"partial'
    assert raised.value.details["input_tokens"] == 12
    assert raised.value.details["output_tokens"] == 7
    assert messages.calls[0]["max_tokens"] == 7


def test_anthropic_downstream_transport_error_is_normalized():
    provider = AnthropicMessagesProvider(
        client=SimpleNamespace(messages=_Messages(error=RuntimeError("provider unavailable"))),
        model="claude-sonnet-5",
        temperature=0,
    )
    with pytest.raises(ProviderTransportError) as raised:
        provider.request_downstream("prompt")
    assert raised.value.provider == "anthropic"
    assert raised.value.details["provider_exception"] == "RuntimeError"


def test_anthropic_schema_sanitizer_removes_unsupported_constraints_recursively_without_mutation():
    source = {
        "type": "object",
        "required": ["states"],
        "properties": {
            "states": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "exclusiveMinimum": False,
                            "exclusiveMaximum": False,
                            "minLength": 1,
                            "maxLength": 10,
                            "enum": [0, 1],
                        },
                        "nested": {
                            "type": "array",
                            "items": {"type": "number", "minimum": -1, "maximum": 1},
                        },
                    },
                },
            }
        },
        "$defs": {"leaf": {"type": "string", "minLength": 2, "const": "x"}},
    }
    original = json.loads(json.dumps(source))
    transformed = anthropic_transport_schema(source)
    assert source == original
    serialized = json.dumps(transformed)
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength"):
        assert key not in serialized
    assert transformed["type"] == "object"
    assert transformed["required"] == ["states"]
    assert transformed["properties"]["states"]["items"]["additionalProperties"] is False
    assert transformed["properties"]["states"]["items"]["properties"]["confidence"]["enum"] == [0, 1]
    assert transformed["$defs"]["leaf"]["const"] == "x"


def test_anthropic_schema_sanitizer_adds_strict_object_flag_recursively():
    simple = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    assert anthropic_transport_schema(simple) == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    source = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"value": {"type": "string"}}},
                    }
                },
                "required": ["items"],
            },
        },
        "required": ["answer", "nested"],
    }
    transformed = anthropic_transport_schema(source)
    assert transformed["additionalProperties"] is False
    assert transformed["properties"]["nested"]["additionalProperties"] is False
    assert transformed["properties"]["nested"]["properties"]["items"]["items"]["additionalProperties"] is False
    assert transformed["properties"]["answer"] == {"type": "string"}
    assert transformed["required"] == ["answer", "nested"]


def test_anthropic_schema_sanitizer_preserves_stricter_additional_properties():
    schema = {
        "type": "object",
        "additionalProperties": {"type": "string"},
        "properties": {},
    }
    transformed = anthropic_transport_schema(schema)
    assert transformed["additionalProperties"] == {"type": "string"}


def test_anthropic_downstream_sends_strict_top_level_answer_schema():
    client, messages = _client({"answer": "ok"})
    AnthropicMessagesProvider(client=client, model="claude-sonnet-5", temperature=0).request_downstream("prompt")
    schema = messages.calls[-1]["output_config"]["format"]["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False


def test_canonical_schema_constraints_and_validation_remain_unchanged():
    canonical = _ResponseSchema.model_json_schema()
    transport = anthropic_transport_schema(canonical)
    confidence = canonical["$defs"]["_StateOutput"]["properties"]["confidence"]["anyOf"][0]
    assert confidence["maximum"] == 1.0
    assert "maximum" not in json.dumps(transport)
    with pytest.raises(Exception):
        _ResponseSchema.model_validate({"states": [{"interaction_id": "i0", "confidence": 2.0}]})


def test_anthropic_request_uses_only_transformed_schema():
    client, messages = _client({"states": []})
    provider = AnthropicMessagesProvider(client=client, model="claude-opus-5")
    provider.request_structured_reconstruction(prompt="prompt", schema=_ResponseSchema, request_metadata={})
    sent_schema = messages.calls[0]["output_config"]["format"]["schema"]
    assert "maximum" not in json.dumps(sent_schema)
    assert sent_schema["properties"]["states"]["type"] == "array"


def test_anthropic_output_uses_shared_validator_without_alias_repair(tmp_path):
    client, _ = _client({"states": [{"interaction_id": "i0", "validated_until": "bad"}]})
    provider = AnthropicMessagesProvider(client=client, model="claude-opus-5")
    engine = GovernanceStateReconstructionEngine(provider=provider, cache_dir=tmp_path, max_input_tokens=100000)
    with pytest.raises(ReconstructionError) as raised:
        engine.reconstruct(_input())
    assert raised.value.status == "structural_invalid"


def test_anthropic_transport_failure_is_common_transport_error_without_key():
    client = SimpleNamespace(messages=_Messages(error=RuntimeError("request failed")))
    provider = AnthropicMessagesProvider(client=client, model="claude-opus-5")
    with pytest.raises(ProviderTransportError) as raised:
        provider.request_structured_reconstruction(prompt="prompt", schema=SimpleNamespace(model_json_schema=lambda: {}), request_metadata={})
    assert raised.value.provider == "anthropic"
    assert raised.value.details["provider_exception"] == "RuntimeError"
    assert "ANTHROPIC_API_KEY" not in str(raised.value.details)


def test_anthropic_cache_identity_isolated_from_openai():
    input_data = _input()
    openai = _cache_payload("claude-opus-5", "same", input_data, "openai")
    anthropic = _cache_payload("claude-opus-5", "same", input_data, "anthropic")
    assert openai != anthropic
    assert anthropic["provider"] == "anthropic"


def test_pilot_accepts_anthropic_provider_without_constructing_a_client():
    args = parse_pilot_args([
        "--manifest", "manifest.json", "--results-dir", "results",
        "--condition", "reconstruction", "--provider", "anthropic",
        "--model", "claude-opus-5", "--dry-run",
    ])
    assert args.provider == "anthropic"
    assert args.model == "claude-opus-5"


def _diagnostic_provider(response):
    return AnthropicMessagesProvider(
        client=SimpleNamespace(messages=_Messages(response=response)),
        model="claude-sonnet-5",
        temperature=0,
    )


def test_anthropic_no_text_with_max_tokens_is_token_limit_truncation_with_metadata():
    response = SimpleNamespace(
        content=[],
        usage=SimpleNamespace(input_tokens=30, output_tokens=0),
        request_id="req-limit",
        stop_reason="max_tokens",
        status="incomplete",
    )
    with pytest.raises(ProviderTransportError) as raised:
        _diagnostic_provider(response).request_downstream("prompt", max_tokens=8)
    details = raised.value.details
    assert details["error_category"] == "token_limit_truncation"
    assert details["token_limit_truncation"] is True
    assert details["stop_reason"] == "max_tokens"
    assert (details["input_tokens"], details["output_tokens"], details["total_tokens"]) == (30, 0, 30)
    assert details["request_id"] == "req-limit"
    assert details["content_block_types"] == [] and details["content_block_count"] == 0
    assert details["text_block_count"] == 0
    assert details["has_raw_text"] is False and details["raw_text_char_count"] == 0
    assert details["json_parse_ok"] is False
    assert details["raw_output_text"] is None


def test_anthropic_no_text_with_end_turn_remains_failure_with_metadata():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="tool_use")],
        usage=SimpleNamespace(input_tokens=11, output_tokens=2, total_tokens=13),
        request_id="req-notxt",
        stop_reason="end_turn",
    )
    with pytest.raises(ProviderTransportError) as raised:
        _diagnostic_provider(response).request_downstream("prompt")
    details = raised.value.details
    assert details["error_category"] == "structural_invalid"
    assert details["stop_reason"] == "end_turn"
    assert (details["input_tokens"], details["output_tokens"], details["total_tokens"]) == (11, 2, 13)
    assert details["request_id"] == "req-notxt"
    assert details["content_block_types"] == ["thinking", "tool_use"]
    assert details["content_block_count"] == 2 and details["text_block_count"] == 0
    assert details["has_non_text_content"] is True and details["non_text_block_types"] == ["thinking", "tool_use"]
    assert details["has_raw_text"] is False and details["json_parse_ok"] is False


def test_anthropic_malformed_json_preserves_bounded_truncation_diagnostic():
    raw = '{"decision": "correct", "confidence": "high", "short_reason": "unterminated'
    response = SimpleNamespace(
        content=[SimpleNamespace(text=raw)],
        usage=SimpleNamespace(input_tokens=4, output_tokens=9),
        request_id="req-malformed",
        stop_reason="end_turn",
    )
    with pytest.raises(ProviderTransportError) as raised:
        _diagnostic_provider(response).request_downstream("prompt")
    details = raised.value.details
    assert details["error_category"] == "structural_invalid"
    assert details["json_parse_ok"] is False
    assert details["has_raw_text"] is True and details["raw_text_char_count"] == len(raw)
    assert details["text_block_count"] == 1 and details["stop_reason"] == "end_turn"
    assert details["raw_text_preview"] == raw and details["raw_text_tail"] == raw
    assert details["raw_text_open_braces"] == 1 and details["raw_text_close_braces"] == 0
    assert details["raw_text_preview_truncated"] is False
    assert details["raw_text_last_char"] == raw[-1]


def test_anthropic_malformed_json_diagnostic_redacts_secrets_and_bounds_size():
    raw = '{"answer": "sk-ant-SECRETVALUE0123"'
    response = SimpleNamespace(
        content=[SimpleNamespace(text=raw)], usage=None, request_id=None, stop_reason="end_turn",
    )
    with pytest.raises(ProviderTransportError) as raised:
        _diagnostic_provider(response).request_downstream("prompt")
    preview = raised.value.details["raw_text_preview"]
    assert "sk-ant-" not in preview and "[REDACTED]" in preview
    oversized = "{" + "x" * 4000
    response = SimpleNamespace(
        content=[SimpleNamespace(text=oversized)], usage=None, request_id=None, stop_reason="end_turn",
    )
    with pytest.raises(ProviderTransportError) as raised:
        _diagnostic_provider(response).request_downstream("prompt")
    details = raised.value.details
    assert details["raw_text_char_count"] == len(oversized)
    assert len(details["raw_text_preview"]) == 256 and len(details["raw_text_tail"]) == 256
    assert details["raw_text_preview_truncated"] is True


def test_anthropic_valid_structured_json_behavior_is_unchanged():
    client, messages = _client({"answer": "structured answer"})
    provider = AnthropicMessagesProvider(client=client, model="claude-opus-5", temperature=0)
    result = provider.request_downstream("same prompt")
    assert result.parsed == {"answer": "structured answer"}
    request = messages.calls[0]
    assert request["model"] == "claude-opus-5"
    assert request["max_tokens"] == 4096
    assert request["messages"] == [{"role": "user", "content": "same prompt"}]
    assert set(request["output_config"]["format"]["schema"]["properties"]) == {"answer"}


def test_anthropic_valid_json_wrong_shape_is_still_rejected():
    client, _ = _client({"states": []})
    provider = AnthropicMessagesProvider(client=client, model="claude-sonnet-5", temperature=0)
    with pytest.raises(ProviderTransportError) as raised:
        provider.request_downstream("prompt")
    assert raised.value.details["error_category"] == "structural_invalid"
