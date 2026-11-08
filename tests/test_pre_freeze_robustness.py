"""Offline pre-freeze invariants for adapters, streaming, and RQ1 plumbing."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.memconflict_adapter import MemConflictDatasetAdapter
from eval.memora_adapter import MemoraDatasetAdapter
from eval.openai_reconstruction import (
    OpenAIGovernanceStateReconstructor,
    ReconstructionError,
    SINGLE_CALL_POLICY,
    STREAMING_POLICY,
    build_reconstruction_prompt,
)
from eval.interfaces import Query, RawInteraction
from eval.reconstruction import GovernanceReconstructionInput
from eval.rq1_metrics import evaluate_records
from scripts.run_reconstruction_batch import run_batch


def _memora_scope(root: Path, *, count_each: int = 1) -> Path:
    persona = root / "weekly" / "software_engineer"
    conversations = persona / "conversations"
    conversations.mkdir(parents=True)
    for session_id in (2, 1):
        (conversations / f"session_{session_id:04d}.json").write_text(json.dumps({
            "session_id": session_id,
            "date": f"2025-06-0{session_id}",
            "conversation": [{"turn": 1, "speaker": "user", "message": f"memory-{session_id}"}],
        }), encoding="utf-8")
    questions = {}
    for task_type in ("reasoning", "remembering", "recommending"):
        questions[task_type] = [
            {"question_id": f"{task_type}-{index}", "question": f"Question {task_type} {index}",
             "question_date": "2025-06-02", "evaluation": {"gold": "SECRET_GOLD_ANSWER_DO_NOT_EXPOSE"}}
            for index in range(count_each)
        ]
    (persona / "evaluation_questions_software_engineer.json").write_text(json.dumps({
        "persona": "software_engineer", "questions": questions,
    }), encoding="utf-8")
    return persona


def _official_conflict_fixture(path: Path) -> Path:
    records = []
    for record_id in ("b", "a"):
        records.append({
            "ID": record_id,
            "Fixed_Profile": {}, "Dynamic_Profile": {}, "Preference_Profile": {},
            "Personality": {}, "Life_Goal": {}, "Others_Profile": {},
            "metadata": {"sentinel": "SECRET_ORACLE_STATE_DO_NOT_EXPOSE"},
            "Full_Session_Chain": [{
                "Session_ID": 1, "Date": "2025-01-01", "Session_Type": "initial",
                "Session_Questions": [
                    {"question_id": "dynamic", "question": "Dynamic?", "answer": "gold-d", "conflict_type": "dynamic_conflict"},
                    {"question_id": "conditional", "question": "Conditional?", "answer": "gold-c", "conflict_type": "conditional_conflict"},
                    {"question_id": "static", "question": "Static?", "answer": "gold-s", "conflict_type": "static_conflict"},
                ],
                "Session_Dialogue": {"dialogue_turn_2": [{"role": "user", "content": f"later-{record_id}"}],
                                     "dialogue_turn_1": [{"role": "user", "content": f"earlier-{record_id}"}]},
            }],
        })
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _adapter_signature(examples):
    return [
        (
            example.query.query_id,
            example.query.metadata.get("task_type"),
            tuple(item.interaction_id for item in example.interactions),
            json.dumps(dict(example.evaluator_metadata), sort_keys=True, default=str),
        )
        for example in examples
    ]


def test_official_adapter_loading_is_deterministic(tmp_path):
    memora = _memora_scope(tmp_path / "memora")
    first = MemoraDatasetAdapter().load(memora)
    second = MemoraDatasetAdapter().load(memora)
    assert _adapter_signature(first) == _adapter_signature(second)
    assert [item.query.query_id for item in first] == [
        "reasoning-0", "remembering-0", "recommending-0"
    ]
    assert [item.interaction_id for item in first[0].interactions] == [
        "software_engineer::1::1", "software_engineer::2::1"
    ]

    conflict_path = _official_conflict_fixture(tmp_path / "Step4_4.jsonl")
    first = MemConflictDatasetAdapter().load(conflict_path)
    second = MemConflictDatasetAdapter().load(conflict_path)
    assert _adapter_signature(first) == _adapter_signature(second)
    assert [item.query.query_id for item in first] == [
        "b::s1::dynamic", "b::s1::conditional", "b::s1::static",
        "a::s1::dynamic", "a::s1::conditional", "a::s1::static",
    ]
    assert [item.interaction_id for item in first[0].interactions] == [
        "b::s1::dialogue_turn_1::0", "b::s1::dialogue_turn_2::0"
    ]


class _BatchReconstructor:
    model = "mock"

    def __init__(self):
        self.calls = []

    def reconstruct(self, input_data):
        self.calls.append(input_data.query.query_id)
        return ()


def _batch_records(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_batch_filters_before_limit_for_all_experiment_slices(tmp_path):
    gate = Path("tests/fixtures/gatemem-mini")
    out = tmp_path / "gate.jsonl"
    run_batch(dataset="gatemem", input_path=gate, output_path=out, query_type="privacy", limit=1,
              reconstructor=_BatchReconstructor())
    assert [row["task_type"] for row in _batch_records(out)] == ["privacy"]
    out = tmp_path / "safety.jsonl"
    run_batch(dataset="gatemem", input_path=gate, output_path=out, query_type="safety", limit=1,
              reconstructor=_BatchReconstructor())
    assert [row["task_type"] for row in _batch_records(out)] == ["safety"]

    conflict = Path("tests/fixtures/memconflict-mini/memconflict.jsonl")
    for task_type in ("dynamic", "conditional", "static"):
        out = tmp_path / f"{task_type}.jsonl"
        run_batch(dataset="memconflict", input_path=conflict, output_path=out, query_type=task_type, limit=1,
                  reconstructor=_BatchReconstructor())
        assert [row["task_type"] for row in _batch_records(out)] == [task_type]

    memora = _memora_scope(tmp_path / "memora", count_each=2)
    out = tmp_path / "memora.jsonl"
    run_batch(dataset="memora", input_path=memora, output_path=out, query_type="remembering", limit=1,
              reconstructor=_BatchReconstructor())
    assert [row["task_type"] for row in _batch_records(out)] == ["remembering"]


def _stream_input(count=6, *, gold=False):
    return GovernanceReconstructionInput(
        raw_interactions=tuple(
            RawInteraction(f"i{index}", "u", f"fact-{index}-" + "x" * 220, datetime(2025, 1, index + 1))
            for index in range(count)
        ),
        query=Query(
            "q", "u", "What governance state applies?", datetime(2025, 2, 1),
            gold_answer="SECRET_GOLD_ANSWER_DO_NOT_EXPOSE" if gold else None,
            metadata={
                "expected_action": "SECRET_EXPECTED_ACTION_DO_NOT_EXPOSE",
                "oracle": "SECRET_ORACLE_STATE_DO_NOT_EXPOSE",
            },
        ),
        runtime_context={
            "user_id": "u", "allowed_scopes": (),
            "gold_answer": "SECRET_GOLD_ANSWER_DO_NOT_EXPOSE",
        },
    )


def _prompt_payload(prompt):
    payload = prompt.split("Model-visible reconstruction input (JSON):\n", 1)[1]
    payload = payload.split("\n\nIncremental reconstruction instructions", 1)[0]
    return json.loads(payload)


class _StreamingProvider:
    def __init__(self, *, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def parse(self, **kwargs):
        prompt = kwargs["input"][0]["content"]
        index = len(self.calls)
        self.calls.append(prompt)
        if self.fail_at == index:
            raise RuntimeError(f"failure-{index}")
        payload = _prompt_payload(prompt)
        ids = [item["interaction_id"] for item in payload["raw_interactions"]]
        carry_text = prompt.split("Incremental reconstruction instructions (model-generated state only):\n", 1)
        carry = json.loads(carry_text[1].split("\nUpdate, revise", 1)[0]) if len(carry_text) == 2 else {}
        previous_ids = [item["interaction_id"] for item in carry.get("previous_model_generated_state", [])]
        state_id = ids[-1] if ids else (previous_ids[-1] if previous_ids else "i0")
        return SimpleNamespace(
            output_parsed={"states": [{"interaction_id": state_id, "is_deleted": False}]},
            usage=SimpleNamespace(input_tokens=100 + index, output_tokens=20 + index, total_tokens=120 + 2 * index),
        )


def _stream_provider(tmp_path, responses, **kwargs):
    provider = OpenAIGovernanceStateReconstructor(
        client=SimpleNamespace(responses=responses), cache_dir=tmp_path, **kwargs
    )
    provider._count_tokens = lambda text: list(range(max(1, len(text) // 100)))
    provider.token_counting = "test_exact_counter"
    return provider


def test_streaming_boundary_policies_and_oversized_interaction(tmp_path):
    input_data = _stream_input(1)
    base = build_reconstruction_prompt(input_data)
    provider = _stream_provider(tmp_path / "fit", _StreamingProvider(), max_input_tokens=len(base) + 5,
                                streaming_chunk_tokens=100000, output_reserve_tokens=5)
    provider.reconstruct(input_data)
    assert provider.last_telemetry["reconstruction_policy"] == SINGLE_CALL_POLICY
    assert provider.last_telemetry["streaming_diagnostics"]["chunk_count"] == 1

    long_input = _stream_input(6)
    provider = _stream_provider(tmp_path / "stream", _StreamingProvider(), max_input_tokens=100000,
                                streaming_chunk_tokens=100000, output_reserve_tokens=5)
    one_chunk = provider._stream_prompt(long_input, (long_input.raw_interactions[0],), (), 0)
    # The full six-interaction prompt is larger than the budget, while the
    # carry-aware second chunk still fits.  This exercises the boundary without
    # confusing stream-instruction overhead with an oversized interaction.
    provider.max_input_tokens = 38
    provider.streaming_chunk_tokens = 33
    provider.reconstruct(long_input)
    assert provider.last_telemetry["reconstruction_policy"] == STREAMING_POLICY
    assert provider.last_telemetry["streaming_diagnostics"]["chunk_count"] > 1

    oversized = _stream_input(1)
    oversized = GovernanceReconstructionInput(
        raw_interactions=(RawInteraction("huge", "u", "x" * 10000, datetime(2025, 1, 1)),),
        query=oversized.query, runtime_context=oversized.runtime_context,
    )
    provider = _stream_provider(tmp_path / "oversized", _StreamingProvider(), max_input_tokens=100,
                                streaming_chunk_tokens=50, output_reserve_tokens=5)
    with pytest.raises(ReconstructionError) as error:
        provider.reconstruct(oversized)
    assert error.value.status == "configuration_error"
    assert provider.last_telemetry["streaming_diagnostics"]["all_interactions_processed"] is False


def test_streaming_budget_includes_carry_state_and_has_no_leakage(tmp_path):
    responses = _StreamingProvider()
    provider = _stream_provider(tmp_path, responses, max_input_tokens=45, streaming_chunk_tokens=35,
                                output_reserve_tokens=5)
    provider.reconstruct(_stream_input(6, gold=True))
    prompts = "\n".join(responses.calls)
    for secret in ("SECRET_GOLD_ANSWER_DO_NOT_EXPOSE", "SECRET_EXPECTED_ACTION_DO_NOT_EXPOSE",
                   "SECRET_ORACLE_STATE_DO_NOT_EXPOSE"):
        assert secret not in prompts
    diagnostics = provider.last_telemetry["streaming_diagnostics"]
    for chunk in diagnostics["per_chunk"]:
        assert chunk["request_size_diagnostics"]["estimated_input_tokens"] <= diagnostics["chunk_budget"]
    chunks = [_prompt_payload(prompt)["raw_interactions"] for prompt in responses.calls]
    flattened = [item["interaction_id"] for chunk in chunks for item in chunk]
    assert flattened == [f"i{index}" for index in range(6)]
    assert len(flattened) == len(set(flattened))
    assert all(
        not any(item["interaction_id"] in {later["interaction_id"] for later in chunks[index + 1]}
                for item in chunks[index])
        for index in range(len(chunks) - 1)
    )
    first_carry = json.loads(
        responses.calls[0].split("Incremental reconstruction instructions (model-generated state only):\n", 1)[1]
        .split("\nUpdate, revise", 1)[0]
    )
    assert first_carry["previous_model_generated_state"] == []
    assert all("previous_model_generated_state" in prompt for prompt in responses.calls[1:])


def test_streaming_cache_equivalence_partial_replay_and_latency_accounting(tmp_path):
    first_responses = _StreamingProvider()
    first = _stream_provider(tmp_path, first_responses, max_input_tokens=45, streaming_chunk_tokens=35,
                             output_reserve_tokens=5)
    expected = first.reconstruct(_stream_input(6))
    cache_files = sorted((tmp_path / "streaming").glob("*.json"))
    assert len(cache_files) > 2
    cache_files[-1].unlink()

    replay_responses = _StreamingProvider()
    replay = _stream_provider(tmp_path, replay_responses, max_input_tokens=45, streaming_chunk_tokens=35,
                              output_reserve_tokens=5)
    actual = replay.reconstruct(_stream_input(6))
    assert [
        (state.interaction_id, state.is_deleted, state.entity_key, state.missing_fields)
        for state in actual
    ] == [
        (state.interaction_id, state.is_deleted, state.entity_key, state.missing_fields)
        for state in expected
    ]
    chunks = replay.last_telemetry["streaming_diagnostics"]["per_chunk"]
    assert any(chunk["cache_hit"] for chunk in chunks)
    assert any(not chunk["cache_hit"] for chunk in chunks)
    fresh_latency = round(sum(chunk["api_latency_ms"] or 0 for chunk in chunks if chunk["api_request_executed"]), 3)
    assert replay.last_telemetry["streaming_diagnostics"]["total_api_latency_ms"] == fresh_latency
    assert replay.last_telemetry["streaming_diagnostics"]["total_original_api_latency_ms"] >= fresh_latency
    assert replay.last_telemetry["streaming_diagnostics"]["all_interactions_processed"] is True
    assert replay.last_telemetry["streaming_diagnostics"]["interaction_count"] == 6
    assert replay.last_telemetry["streaming_diagnostics"]["chunk_count"] == len(chunks)


def test_midstream_failure_is_not_a_successful_empty_reconstruction(tmp_path):
    responses = _StreamingProvider(fail_at=1)
    provider = _stream_provider(tmp_path, responses, max_input_tokens=45, streaming_chunk_tokens=35,
                                output_reserve_tokens=5)
    with pytest.raises(ReconstructionError) as error:
        provider.reconstruct(_stream_input(6))
    assert error.value.status == "transport_error"
    assert error.value.details["chunk_index"] == 1
    assert provider.last_result["status"] == "transport_error"
    assert provider.last_result["states"] == ()
    assert provider.last_telemetry["streaming_diagnostics"]["all_interactions_processed"] is False

    summary = evaluate_records([{
        "query_id": "failed", "dataset": "memora", "task_type": "remembering",
        "applicable_governance_dimensions": ["lifecycle_deletion"], "skipped": False,
        "reconstruction_status": "error", "validation_status": "transport_error",
        "reconstructed_states": [], "error": {"type": "ReconstructionError"},
        "evaluator_only": {"model_visible_interaction_ids": ["i0"]},
    }])
    metrics = summary["structural"]
    assert metrics["provider_execution_failure_count"] == 1
    assert metrics["empty_reconstruction_rate"] == 0
    assert summary["per_example"][0]["reconstruction_error_labels"] == ["invalid_prediction", "provider_error"]


def test_mixed_rq1_denominators_exclude_skips_and_failures():
    def row(query_id, status="success", validation="valid", skipped=False, states=None):
        return {
            "query_id": query_id, "dataset": "gatemem", "task_type": "privacy",
            "applicable_governance_dimensions": [] if skipped else ["authorization_scope"],
            "skipped": skipped, "reconstruction_status": "skipped" if skipped else status,
            "validation_status": "not_applicable" if skipped else validation,
            "reconstructed_states": states or [],
            "evaluator_only": {"model_visible_interaction_ids": ["i1"]},
        }
    summary = evaluate_records([
        row("valid", states=[{"interaction_id": "i1", "authorized_scopes": ["x"], "missing_fields": []}]),
        row("empty"),
        row("provider", status="error", validation="transport_error"),
        row("structural", status="error", validation="structural_invalid"),
        row("skipped", skipped=True),
    ])
    metrics = summary["structural"]
    assert metrics["n_examples"] == 4
    assert metrics["valid_prediction_rate"] == 0.5
    assert metrics["provider_execution_failure_count"] == 1
    assert metrics["model_reconstruction_failure_count"] == 1
    assert metrics["empty_reconstruction_rate"] == 0.25
    assert metrics["abstention_rate"] == 0.25
    assert metrics["relevant_field_denominator"] == 1
    assert metrics["grounding_denominator"] == 1
