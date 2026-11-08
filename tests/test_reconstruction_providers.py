from __future__ import annotations

from datetime import datetime

import pytest

from eval.openai_reconstruction import ReconstructionError, _ResponseSchema
from eval.openai_reconstruction import OpenAIGovernanceStateReconstructor
from eval.reconstruction import GovernanceReconstructionInput
from eval.reconstruction_engine import GovernanceStateReconstructionEngine
from eval.reconstruction_provider import ProviderResponse, ProviderTelemetry, ProviderTransportError, RecordingFakeProvider
from eval.reconstruction_provider import OpenAIResponsesProvider
from eval.interfaces import Query, RawInteraction
from eval.openai_reconstruction import _cache_payload


def _input(count: int = 8) -> GovernanceReconstructionInput:
    return GovernanceReconstructionInput(
        raw_interactions=tuple(
            RawInteraction(f"i{index}", "u", f"fact-{index}-" + ("x" * 250), datetime(2025, 1, index + 1))
            for index in range(count)
        ),
        query=Query("q", "u", "What governance state applies?", datetime(2025, 2, 1)),
        runtime_context={"user_id": "u", "current_time": datetime(2025, 2, 1), "allowed_scopes": ()},
    )


def _provider(name: str, count: int = 8):
    responses = [{"states": [{"interaction_id": f"i{count - 1}", "is_deleted": False}]} for _ in range(8)]
    return RecordingFakeProvider(name, "fake-model", responses)


def _engine(tmp_path, provider):
    engine = GovernanceStateReconstructionEngine(
        provider=provider, cache_dir=tmp_path, max_input_tokens=45,
        streaming_chunk_tokens=35, output_reserve_tokens=5,
    )
    engine._count_tokens = lambda text: list(range(max(1, len(text) // 100)))
    engine.token_counting = "test_deterministic_counter"
    return engine


def test_engine_is_composed_not_openai_subclass():
    assert not issubclass(GovernanceStateReconstructionEngine, OpenAIGovernanceStateReconstructor)


def test_fake_providers_receive_identical_chunks_and_carry_forward(tmp_path):
    first = _provider("fake_a")
    second = _provider("fake_b")
    _engine(tmp_path / "a", first).reconstruct(_input())
    _engine(tmp_path / "b", second).reconstruct(_input())

    assert len(first.requests) == len(second.requests) > 1
    for left, right in zip(first.requests, second.requests):
        assert left["prompt"] == right["prompt"]
        assert left["schema"] is right["schema"]
        assert left["prompt"].count("previous_model_generated_state") == right["prompt"].count("previous_model_generated_state")

    def ids(request):
        payload = request["prompt"].split("Model-visible reconstruction input (JSON):\n", 1)[1]
        payload = payload.split("\n\nIncremental reconstruction instructions", 1)[0]
        import json
        return [item["interaction_id"] for item in json.loads(payload)["raw_interactions"]]

    left_ids = [ids(request) for request in first.requests]
    right_ids = [ids(request) for request in second.requests]
    assert left_ids == right_ids
    assert [item for chunk in left_ids for item in chunk] == [f"i{i}" for i in range(8)]


def test_fake_provider_normalizes_to_canonical_state_and_provenance(tmp_path):
    provider = RecordingFakeProvider("fake_a", "fake-v1", [{
        "states": [{"interaction_id": "i0", "is_deleted": False, "missing_fields": ["is_deleted"]}],
    }])
    engine = GovernanceStateReconstructionEngine(provider=provider, cache_dir=tmp_path, max_input_tokens=100000)
    states = engine.reconstruct(_input(1))
    assert states[0].interaction_id == "i0"
    assert states[0].is_deleted is False
    assert "is_deleted" not in states[0].missing_fields
    assert '"provider":"fake_a"' in states[0].provenance


def test_semantic_invalid_outputs_are_shared_and_not_transport_retried(tmp_path):
    provider = RecordingFakeProvider("fake_a", "fake-v1", [{
        "states": [{"interaction_id": "i0", "validated_until": "bad"}],
    }])
    engine = GovernanceStateReconstructionEngine(provider=provider, cache_dir=tmp_path, max_input_tokens=100000)
    with pytest.raises(ReconstructionError) as raised:
        engine.reconstruct(_input(1))
    assert raised.value.status == "structural_invalid"
    assert len(provider.requests) == 1


def test_invalid_interval_is_semantic_invalid_for_fake_providers(tmp_path):
    for name in ("fake_a", "fake_b"):
        provider = RecordingFakeProvider(name, "fake-v1", [{
            "states": [{"interaction_id": "i0", "valid_from": "2025-02-02", "valid_until": "2025-02-01"}],
        }])
        engine = GovernanceStateReconstructionEngine(provider=provider, cache_dir=tmp_path / name, max_input_tokens=100000)
        with pytest.raises(ReconstructionError) as raised:
            engine.reconstruct(_input(1))
        assert raised.value.status == "semantic_invalid"


def test_provider_namespaces_cache_identity(tmp_path):
    for name in ("fake_a", "fake_b"):
        provider = RecordingFakeProvider(name, "same-model", [{"states": []}])
        engine = GovernanceStateReconstructionEngine(provider=provider, cache_dir=tmp_path, max_input_tokens=100000)
        engine.reconstruct(_input(1))
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_exact_model_and_provider_are_preserved_and_cache_identity_differs():
    input_data = _input(1)
    prompt = "same prompt"
    openai_key = _cache_payload("gpt-4o-mini", prompt, input_data, "openai")
    sol_key = _cache_payload("gpt-5.6-sol", prompt, input_data, "openai")
    assert sol_key["model"] == "gpt-5.6-sol"
    assert openai_key != sol_key


def test_openai_transport_accepts_exact_gpt56_model_without_branching():
    transport = OpenAIResponsesProvider(client=object(), model="gpt-5.6-sol")
    assert transport.provider_name == "openai"
    assert transport.model == "gpt-5.6-sol"


def test_provider_telemetry_is_normalized_at_engine_boundary(tmp_path):
    class Metered(RecordingFakeProvider):
        def request_structured_reconstruction(self, *, prompt, schema, request_metadata):
            self.requests.append({"prompt": prompt, "schema": schema, "request_metadata": dict(request_metadata)})
            return ProviderResponse(
                parsed={"states": []},
                telemetry=ProviderTelemetry(self.provider_name, self.model, 11, 7, 18, 1.5, "fake-request"),
            )

    provider = Metered("fake_a", "fake-v1", [])
    engine = GovernanceStateReconstructionEngine(provider=provider, cache_dir=tmp_path, max_input_tokens=100000)
    engine.reconstruct(_input(1))
    assert engine.last_telemetry["input_tokens"] == 11
    assert engine.last_telemetry["output_tokens"] == 7
    assert engine.last_telemetry["total_tokens"] == 18


def test_provider_transport_error_uses_common_taxonomy(tmp_path):
    class Failing(RecordingFakeProvider):
        def request_structured_reconstruction(self, **kwargs):
            raise ProviderTransportError("network", provider=self.provider_name, model=self.model)

    engine = GovernanceStateReconstructionEngine(provider=Failing("fake_a", "fake-v1", []), cache_dir=tmp_path, max_input_tokens=100000)
    with pytest.raises(ReconstructionError) as raised:
        engine.reconstruct(_input(1))
    assert raised.value.status == "transport_error"
