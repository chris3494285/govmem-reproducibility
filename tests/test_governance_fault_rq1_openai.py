from __future__ import annotations

from types import SimpleNamespace

import pytest

from eval.providers.openai_provider import DEFAULT_OPENAI_MODEL, DownstreamAnswer, OpenAIDownstreamProvider
from eval.reconstruction_provider import ProviderResponse, ProviderTelemetry, ProviderTransportError
from scripts.run_governance_fault_rq1 import construct_context, decide_admission, load_manifest, run_manifest


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _response(answer="ok"):
    return SimpleNamespace(
        output_parsed=DownstreamAnswer(answer=answer),
        usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
        _request_id="openai-request",
    )


def test_openai_structured_request_and_telemetry():
    responses = FakeResponses(_response("structured"))
    provider = OpenAIDownstreamProvider(
        client=SimpleNamespace(responses=responses), model="arbitrary-model", reasoning_effort="none", temperature=0
    )

    result = provider.request_downstream("same prompt")

    assert result.parsed == {"answer": "structured"}
    assert responses.calls[0]["model"] == "arbitrary-model"
    assert responses.calls[0]["reasoning"] == {"effort": "none"}
    assert responses.calls[0]["temperature"] == 0
    assert responses.calls[0]["text_format"] is DownstreamAnswer
    assert result.telemetry.input_tokens == 11
    assert result.telemetry.output_tokens == 7
    assert result.telemetry.total_tokens == 18
    assert result.telemetry.request_id == "openai-request"


def test_openai_reasoning_omits_temperature_and_preserves_requested_setting():
    responses = FakeResponses(_response())
    provider = OpenAIDownstreamProvider(
        client=SimpleNamespace(responses=responses), model="gpt-5.6-sol", reasoning_effort="high", temperature=0
    )
    provider.request_downstream("prompt")
    request = responses.calls[0]
    assert request["reasoning"] == {"effort": "high"}
    assert "temperature" not in request
    assert provider.requested_temperature == 0
    assert provider.effective_temperature is None
    assert provider.reasoning_effort == "high"


def test_openai_provider_error_is_normalized():
    responses = FakeResponses(error=RuntimeError("no service"))
    provider = OpenAIDownstreamProvider(client=SimpleNamespace(responses=responses), model="arbitrary-model")
    with pytest.raises(ProviderTransportError) as raised:
        provider.request_downstream("prompt")
    assert raised.value.provider == "openai"
    assert raised.value.details["provider_exception"] == "RuntimeError"


def test_openai_preflight_does_not_construct_or_call_provider(tmp_path):
    calls = []
    output = tmp_path / "preflight.jsonl"
    results = run_manifest(
        manifest="data/governance_fault_rq1_restrictive_frozen.jsonl",
        output=output,
        provider="openai",
        model="gpt-5.6-sol",
        reasoning_effort="none",
        temperature=0,
        limit=2,
        dry_run=True,
        overwrite=True,
        provider_factory=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert len(results) == 4
    assert len({row["example_id"] for row in results}) == 2
    assert calls == []
    assert all(row["status"] == "dry_run" for row in results)
    assert all(row["requested_reasoning_effort"] == "none" for row in results)
    assert all(row["requested_temperature"] == 0 and row["effective_temperature"] == 0 for row in results)


def test_openai_downstream_context_is_identical_to_deepseek_context(tmp_path):
    openai_fake = _RecordingDownstream("openai", "arbitrary-model")
    deepseek_fake = _RecordingDownstream("deepseek", "deepseek-v4-pro")
    run_manifest(
        manifest="data/governance_fault_rq1_restrictive_frozen.jsonl", output=tmp_path / "openai.jsonl",
        provider="openai", model="arbitrary-model", limit=1,
        provider_factory=lambda model, **kwargs: openai_fake, overwrite=True,
    )
    run_manifest(
        manifest="data/governance_fault_rq1_restrictive_frozen.jsonl", output=tmp_path / "deepseek.jsonl",
        provider="deepseek", model="deepseek-v4-pro", limit=1,
        provider_factory=lambda model, **kwargs: deepseek_fake, overwrite=True,
    )
    assert openai_fake.prompts == deepseek_fake.prompts


class _RecordingDownstream:
    def __init__(self, provider, model):
        self.provider = provider
        self.model = model
        self.prompts = []

    def request_downstream(self, prompt):
        self.prompts.append(prompt)
        return ProviderResponse(
            {"answer": "ok"}, ProviderTelemetry(self.provider, self.model, 1, 1, 2, 0, "request")
        )


def test_openai_reuse_is_within_base_and_reasoning_is_cache_identity(tmp_path):
    fake = _RecordingDownstream("openai", "arbitrary-model")
    results = run_manifest(
        manifest="data/governance_fault_rq1_pilot_frozen.jsonl",
        output=tmp_path / "reuse.jsonl",
        provider="openai",
        model="arbitrary-model",
        reasoning_effort="none",
        limit=1,
        provider_factory=lambda model, **kwargs: fake,
        overwrite=True,
    )
    assert len(fake.prompts) == 2
    assert sum(row["response_reused"] for row in results) == 1

    second = _RecordingDownstream("openai", "arbitrary-model")
    run_manifest(
        manifest="data/governance_fault_rq1_pilot_frozen.jsonl",
        output=tmp_path / "reasoning.jsonl",
        provider="openai",
        model="arbitrary-model",
        reasoning_effort="high",
        limit=1,
        provider_factory=lambda model, **kwargs: second,
        overwrite=True,
    )
    assert len(second.prompts) == 2


def test_openai_default_model_is_explicit():
    assert DEFAULT_OPENAI_MODEL == "gpt-5.6-sol"
