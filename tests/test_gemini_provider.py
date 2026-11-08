from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eval.providers.gemini_provider import DEFAULT_GEMINI_MODEL, GeminiGenerativeProvider, gemini_answer_schema, gemini_request_config
from eval.reconstruction_provider import ProviderTransportError
from scripts.run_final_main_smoke_deepseek import _retryable_provider_error


class _Models:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def response(text='{"answer":"ok"}', finish_reason="STOP"):
    return SimpleNamespace(text=text, usage_metadata=SimpleNamespace(prompt_token_count=11, candidates_token_count=4, total_token_count=15, thoughts_token_count=2), candidates=[SimpleNamespace(finish_reason=finish_reason)], response_id="r1")


def test_gemini_default_target_and_sdk_compatible_config_without_tools():
    assert DEFAULT_GEMINI_MODEL == "gemini-3-flash-preview"
    config = gemini_request_config(max_output_tokens=128)
    from google.genai import types
    types.GenerateContentConfig(**config)
    assert "tools" not in config
    assert config["thinking_config"] == {"thinking_level": "LOW"}


def test_gemini_request_uses_exact_logical_schema_and_no_tools():
    config = gemini_request_config(max_output_tokens=128)
    assert config["response_mime_type"] == "application/json"
    assert config["response_schema"] == gemini_answer_schema()
    assert "tools" not in config
    assert config["max_output_tokens"] == 128


def test_gemini_mocked_response_parses_and_extracts_usage():
    models = _Models(response=response())
    provider = GeminiGenerativeProvider(client=SimpleNamespace(models=models), model="gemini-3-flash-preview")
    result = provider.request_downstream("prompt", max_tokens=128)
    assert result.parsed == {"answer": "ok"}
    assert result.telemetry.input_tokens == 11
    assert result.telemetry.output_tokens == 4
    assert result.telemetry.total_tokens == 15
    assert result.telemetry.thoughts_tokens == 2
    assert models.calls[0]["model"] == "gemini-3-flash-preview"


def test_gemini_malformed_and_max_token_outputs_are_explicit_failures():
    malformed = GeminiGenerativeProvider(client=SimpleNamespace(models=_Models(response=response("not json"))), model="gemini-3-flash-preview")
    with pytest.raises(ProviderTransportError) as error:
        malformed.request_downstream("prompt")
    assert error.value.details["error_category"] == "structural_invalid"
    truncated = GeminiGenerativeProvider(client=SimpleNamespace(models=_Models(response=response('{"answer":"partial', "MAX_TOKENS"))), model="gemini-3-flash-preview")
    with pytest.raises(ProviderTransportError) as error:
        truncated.request_downstream("prompt")
    assert error.value.details["error_category"] == "token_limit_truncation"
    assert error.value.details["retryable"] is False


def test_gemini_http_retry_classification_is_conservative():
    def failure(status):
        return ProviderTransportError("provider failure", provider="gemini", model=DEFAULT_GEMINI_MODEL, details={"http_status": status})

    for status in (400, 401, 403, 404):
        assert _retryable_provider_error(failure(status)) is False
    for status in (429, 500, 503):
        assert _retryable_provider_error(failure(status)) is True


def test_gemini_configuration_metadata_does_not_claim_thinking_disabled():
    # The transport uses explicit LOW rather than claiming that thinking is
    # disabled.
    assert gemini_request_config(max_output_tokens=128)["thinking_config"]["thinking_level"] == "LOW"
