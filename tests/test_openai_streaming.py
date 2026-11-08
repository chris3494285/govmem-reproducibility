import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from eval.interfaces import Query, RawInteraction
from eval.openai_reconstruction import (
    OpenAIGovernanceStateReconstructor,
    ReconstructionError,
    STREAMING_POLICY,
    SINGLE_CALL_POLICY,
)
from eval.reconstruction import GovernanceReconstructionInput


def _input(count=8):
    return GovernanceReconstructionInput(
        raw_interactions=tuple(
            RawInteraction(f"i{index}", "u", f"fact-{index}-" + ("x" * 250), datetime(2025, 1, index + 1))
            for index in range(count)
        ),
        query=Query("q", "u", "What governance state applies?", datetime(2025, 2, 1)),
        runtime_context={"user_id": "u", "current_time": datetime(2025, 2, 1), "allowed_scopes": ()},
    )


def _payload(prompt):
    payload = prompt.split("Model-visible reconstruction input (JSON):\n", 1)[1]
    payload = payload.split("\n\nIncremental reconstruction instructions", 1)[0]
    return json.loads(payload)


class StreamingResponses:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def parse(self, **kwargs):
        prompt = kwargs["input"][0]["content"]
        self.calls.append(prompt)
        call_index = len(self.calls) - 1
        if self.fail_at == call_index:
            raise RuntimeError("chunk request failed")
        ids = [item["interaction_id"] for item in _payload(prompt)["raw_interactions"]]
        previous = ["i0"] if call_index else []
        state_ids = list(dict.fromkeys(previous + ids[-1:]))
        return SimpleNamespace(
            output_parsed={
                "states": [{"interaction_id": interaction_id, "is_deleted": False} for interaction_id in state_ids]
            },
            usage=SimpleNamespace(input_tokens=10, output_tokens=2, total_tokens=12),
        )


class SemanticFailureResponses(StreamingResponses):
    def parse(self, **kwargs):
        prompt = kwargs["input"][0]["content"]
        if len(self.calls) == self.fail_at:
            self.calls.append(prompt)
            ids = [item["interaction_id"] for item in _payload(prompt)["raw_interactions"]]
            return SimpleNamespace(
                output_parsed={
                    "states": [{
                        "interaction_id": ids[-1],
                        "valid_from": "2025-02-02T00:00:00",
                        "valid_until": "2025-02-01T00:00:00",
                    }]
                },
                usage=SimpleNamespace(input_tokens=10, output_tokens=2, total_tokens=12),
            )
        return super().parse(**kwargs)


def _provider(tmp_path, responses, **kwargs):
    provider = OpenAIGovernanceStateReconstructor(
        client=SimpleNamespace(responses=responses), cache_dir=tmp_path, **kwargs
    )
    # The installed tiktoken package cannot load model encoding data offline in
    # this environment. Use a deterministic test counter, not a production
    # heuristic, to force predictable chunk boundaries.
    provider._count_tokens = lambda text: list(range(max(1, len(text) // 100)))
    provider.token_counting = "test_deterministic_counter"
    return provider


def test_small_history_preserves_one_call_path(tmp_path):
    responses = StreamingResponses()
    provider = _provider(tmp_path, responses, max_input_tokens=100000, streaming_chunk_tokens=50000)
    states = provider.reconstruct(_input(1))
    assert len(responses.calls) == 1
    assert provider.last_telemetry["reconstruction_policy"] == SINGLE_CALL_POLICY
    assert provider.last_telemetry["streaming_diagnostics"]["chunk_count"] == 1
    assert states


def test_streaming_is_chronological_exactly_once_and_carries_state(tmp_path):
    responses = StreamingResponses()
    provider = _provider(tmp_path, responses, max_input_tokens=45, streaming_chunk_tokens=35, output_reserve_tokens=5)
    states = provider.reconstruct(_input())
    assert len(responses.calls) > 1
    chunks = [_payload(prompt)["raw_interactions"] for prompt in responses.calls]
    flattened = [item["interaction_id"] for chunk in chunks for item in chunk]
    assert flattened == [f"i{index}" for index in range(8)]
    assert len(flattened) == len(set(flattened)) == 8
    assert all(
        set(item["interaction_id"] for item in chunks[index])
        .isdisjoint(set(item["interaction_id"] for later in chunks[index + 1:] for item in later))
        for index in range(len(chunks))
    )
    assert any("previous_model_generated_state" in prompt for prompt in responses.calls[1:])
    assert "gold_answer" not in "\n".join(responses.calls)
    assert "expected_action" not in "\n".join(responses.calls)
    assert provider.last_telemetry["reconstruction_policy"] == STREAMING_POLICY
    diagnostics = provider.last_telemetry["streaming_diagnostics"]
    assert diagnostics["all_interactions_processed"] is True
    assert diagnostics["chunk_count"] == len(responses.calls)
    assert diagnostics["total_input_tokens"] == 10 * len(responses.calls)
    assert diagnostics["total_output_tokens"] == 2 * len(responses.calls)
    assert states


def test_streaming_failure_identifies_chunk_and_keeps_transport_status(tmp_path):
    responses = StreamingResponses(fail_at=1)
    provider = _provider(tmp_path, responses, max_input_tokens=45, streaming_chunk_tokens=35, output_reserve_tokens=5)
    with pytest.raises(ReconstructionError) as raised:
        provider.reconstruct(_input())
    assert raised.value.status == "transport_error"
    assert raised.value.details["chunk_index"] == 1
    assert provider.last_telemetry["streaming_diagnostics"]["all_interactions_processed"] is False


def test_semantic_streaming_failure_is_provider_neutral_and_preserves_partial_telemetry(tmp_path):
    responses = SemanticFailureResponses(fail_at=1)
    provider = _provider(tmp_path, responses, max_input_tokens=45, streaming_chunk_tokens=35, output_reserve_tokens=5)
    with pytest.raises(ReconstructionError) as raised:
        provider.reconstruct(_input())
    assert raised.value.status == "semantic_invalid"
    assert "OpenAI" not in str(raised.value)
    assert "Structured reconstruction returned malformed streaming output" in str(raised.value)
    assert len(responses.calls) == 2
    diagnostics = provider.last_telemetry["streaming_diagnostics"]
    assert diagnostics["completed_chunk_count"] == 1
    assert diagnostics["failed_chunk_index"] == 1
    assert diagnostics["chunk_count"] == 2
    assert diagnostics["total_input_tokens"] == 20
    assert diagnostics["total_output_tokens"] == 4
    assert diagnostics["total_tokens"] == 24
    assert isinstance(provider.last_telemetry["latency_ms"], (int, float))
    assert provider.last_telemetry["latency_ms"] == diagnostics["total_api_latency_ms"]
    assert len(diagnostics["per_chunk"]) == 2
    assert diagnostics["per_chunk"][1]["error"] == "semantic_invalid"
    assert raised.value.details["completed_chunk_count"] == 1
    assert raised.value.details["failed_chunk_index"] == 1


def test_streaming_chunk_cache_replay_avoids_api_calls(tmp_path):
    first_responses = StreamingResponses()
    first = _provider(tmp_path, first_responses, max_input_tokens=45, streaming_chunk_tokens=35, output_reserve_tokens=5)
    first.reconstruct(_input())
    first_call_count = len(first_responses.calls)
    second_responses = StreamingResponses(fail_at=0)
    second = _provider(tmp_path, second_responses, max_input_tokens=45, streaming_chunk_tokens=35, output_reserve_tokens=5)
    second.reconstruct(_input())
    assert len(second_responses.calls) == 0
    assert second.last_telemetry["cache_hit"] is True
    assert second.last_telemetry["streaming_diagnostics"]["chunk_count"] == first_call_count
