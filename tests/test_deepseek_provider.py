from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace

import pytest

from eval.interfaces import Query, RawInteraction
from eval.openai_reconstruction import ReconstructionError, _ResponseSchema
from eval.reconstruction import GovernanceReconstructionInput
from eval.reconstruction_engine import GovernanceStateReconstructionEngine
from eval.reconstruction_provider import ProviderTransportError
from eval.providers.deepseek_provider import (
    DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DeepSeekChatCompletionsProvider,
    DeepSeekDownstreamProvider,
    deepseek_transport_prompt,
)
from scripts.run_rq1_pilot import parse_args as parse_pilot_args


class _Completions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def _response(payload):
    return SimpleNamespace(
        id="deepseek-request",
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(payload)),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
    )


def _input() -> GovernanceReconstructionInput:
    return GovernanceReconstructionInput(
        raw_interactions=(RawInteraction("i0", "u", "memory", datetime(2025, 1, 1)),),
        query=Query("q", "u", "What governance state applies?", datetime(2025, 2, 1)),
        runtime_context={"user_id": "u", "allowed_scopes": ()},
    )


def test_deepseek_request_shape_and_telemetry():
    completions = _Completions(_response({"states": [{"interaction_id": "i0", "is_deleted": False}]}))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = DeepSeekChatCompletionsProvider(client=client)
    result = provider.request_structured_reconstruction(
        prompt="Return JSON.", schema=_ResponseSchema, request_metadata={}
    )
    request = completions.calls[0]
    assert provider.provider_name == "deepseek"
    assert provider.model == DEFAULT_DEEPSEEK_MODEL
    assert request["model"] == "deepseek-v4-pro"
    assert request["response_format"] == {"type": "json_object"}
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert result.parsed["states"][0]["interaction_id"] == "i0"
    assert result.telemetry.provider == "deepseek"
    assert result.telemetry.input_tokens == 100
    assert result.telemetry.output_tokens == 20
    assert result.telemetry.total_tokens == 120
    assert result.telemetry.request_id == "deepseek-request"
    assert result.telemetry.stop_reason == "stop"
    assert isinstance(result.telemetry.latency_ms, (int, float))


def test_deepseek_downstream_temperature_defaults_to_zero_and_allows_override():
    completions = _Completions(_response({"answer": "ok"}))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = DeepSeekDownstreamProvider(client=client)
    provider.request_downstream("answer")
    assert completions.calls[-1]["temperature"] == 0
    override = DeepSeekDownstreamProvider(client=client, temperature=0.25)
    override.request_downstream("answer")
    assert completions.calls[-1]["temperature"] == 0.25


def test_deepseek_uses_only_deepseek_key_and_base_url(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    provider = DeepSeekChatCompletionsProvider()
    provider._client()
    assert captured == {"api_key": "deepseek-secret", "base_url": DEEPSEEK_BASE_URL}


def test_downstream_provider_reuses_thread_local_clients_and_separates_threads(monkeypatch):
    created = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

    import openai
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    provider = DeepSeekDownstreamProvider(model="deepseek-v4-flash")
    same_thread = provider._client()
    assert provider._client() is same_thread
    barrier = threading.Barrier(2)

    def from_worker(_):
        client = provider._client()
        barrier.wait()
        return client

    with ThreadPoolExecutor(max_workers=2) as pool:
        other_threads = list(pool.map(from_worker, range(2)))
    assert all(client is not same_thread for client in other_threads)
    assert len({id(client) for client in [same_thread, *other_threads]}) == 3
    provider.close()


def test_deepseek_transport_prompt_does_not_mutate_canonical_schema():
    schema = {"type": "object", "properties": {"x": {"type": "number", "minimum": 0}}}
    original = json.loads(json.dumps(schema))
    prompt = deepseek_transport_prompt("shared prompt", schema)
    assert schema == original
    assert "Return the reconstruction as valid JSON" in prompt
    assert '"minimum":0' in prompt


def test_deepseek_semantic_validation_is_shared_and_not_retried(tmp_path):
    completions = _Completions(_response({"states": [{"interaction_id": "i0", "validated_until": "bad"}]}))
    provider = DeepSeekChatCompletionsProvider(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)), model="deepseek-v4-pro"
    )
    engine = GovernanceStateReconstructionEngine(provider=provider, cache_dir=tmp_path, max_input_tokens=100000)
    with pytest.raises(ReconstructionError) as raised:
        engine.reconstruct(_input())
    assert raised.value.status == "structural_invalid"
    assert len(completions.calls) == 1


def test_deepseek_malformed_json_is_structural_failure():
    response = SimpleNamespace(
        id="request",
        choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"), finish_reason="stop")],
        usage=None,
    )
    completions = _Completions(response)
    provider = DeepSeekChatCompletionsProvider(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )
    with pytest.raises(ProviderTransportError) as raised:
        provider.request_structured_reconstruction(prompt="json", schema=_ResponseSchema, request_metadata={})
    assert raised.value.details["error_category"] == "structural_invalid"


def test_deepseek_transport_error_remains_transport_error():
    completions = _Completions(error=RuntimeError("provider unavailable"))
    provider = DeepSeekChatCompletionsProvider(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )
    with pytest.raises(ProviderTransportError) as raised:
        provider.request_structured_reconstruction(prompt="json", schema=_ResponseSchema, request_metadata={})
    assert raised.value.provider == "deepseek"
    assert raised.value.details["provider_exception"] == "RuntimeError"
    assert "DEEPSEEK_API_KEY" not in str(raised.value.details)


def test_rq1_cli_accepts_deepseek_without_constructing_a_provider():
    args = parse_pilot_args([
        "--manifest", "manifest.json", "--results-dir", "results",
        "--condition", "reconstruction", "--provider", "deepseek",
        "--model", "deepseek-v4-pro", "--dry-run",
    ])
    assert args.provider == "deepseek"
    assert args.model == "deepseek-v4-pro"
