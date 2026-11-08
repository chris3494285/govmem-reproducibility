from __future__ import annotations

import json
import httpx
from pathlib import Path
from types import SimpleNamespace
from openai import BadRequestError

from eval.interfaces import ReconstructedGovernanceState
from eval.openai_reconstruction import OpenAIGovernanceStateReconstructor, ReconstructionError
from eval.reconstruction import GovernanceReconstructionInput
from scripts.run_reconstruction_batch import run_batch


GATEMEM_FIXTURE = "tests/fixtures/gatemem-mini"
MEMCONFLICT_FIXTURE = "tests/fixtures/memconflict-mini"


def _state(interaction_id: str = "mini_office_001::t001") -> ReconstructedGovernanceState:
    return ReconstructedGovernanceState(
        interaction_id=interaction_id,
        authorized_scopes=("default",),
        is_deleted=False,
        valid_from=None,
        valid_until=None,
        provenance='{"provider":"openai","model":"mock","prompt_schema_version":"test","cache_hit":false}',
        metadata={"validation_status": "valid"},
    )


class RecordingReconstructor:
    model = "mock-model"

    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def reconstruct(self, input_data):
        assert input_data.query.gold_answer is None
        assert not hasattr(input_data, "oracle_governance")
        assert "gold" not in input_data.runtime_context
        self.calls.append(input_data.query.query_id)
        if self.fail:
            raise ReconstructionError("mock provider failure", status="transport_error")
        return (_state(input_data.raw_interactions[0].interaction_id),)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_batch_writes_one_record_per_example_and_skips_non_applicable(tmp_path):
    output = tmp_path / "results.jsonl"
    reconstructor = RecordingReconstructor()
    summary = run_batch(
        dataset="gatemem",
        input_path=GATEMEM_FIXTURE,
        output_path=output,
        reconstructor=reconstructor,
    )
    records = _records(output)
    assert len(records) == 3
    assert len(reconstructor.calls) == 2
    utility = next(record for record in records if record["query_id"] == "ckpt_authorized")
    assert utility["skipped"] is True
    assert utility["skipped_reason"] == "no_applicable_governance_dimensions"
    assert utility["reconstructed_states"] == []
    assert utility["evaluator_only"]["evaluator_metadata"]["gold"]["expected_action"] == "answer"
    assert utility["evaluator_only"]["oracle_available"] is False
    assert utility["evaluator_only"]["oracle_fields_supported"] == []
    assert utility["evaluator_only"]["model_visible_interaction_ids"]
    privacy = next(record for record in records if record["query_id"] == "ckpt_unauthorized")
    assert privacy["reconstruction_status"] == "success"
    assert summary["skipped"] == 1
    assert summary["valid"] == 2


def test_batch_resume_skips_completed_query_ids(tmp_path):
    output = tmp_path / "results.jsonl"
    first = RecordingReconstructor()
    run_batch(dataset="gatemem", input_path=GATEMEM_FIXTURE, output_path=output, limit=3, reconstructor=first)
    second = RecordingReconstructor(fail=True)
    summary = run_batch(dataset="gatemem", input_path=GATEMEM_FIXTURE, output_path=output, limit=3, reconstructor=second)
    assert second.calls == []
    assert summary["selected"] == 3
    assert summary["applicable"] == 2
    assert summary["non_applicable"] == 1
    assert summary["already_completed"] == 3
    assert summary["executed"] == 0
    assert len(_records(output)) == 3


def test_provider_failure_is_recorded_and_batch_continues(tmp_path):
    output = tmp_path / "results.jsonl"

    class FailsFirst:
        model = "mock-model"

        def __init__(self):
            self.calls = 0

        def reconstruct(self, input_data):
            self.calls += 1
            if self.calls == 1:
                raise ReconstructionError("mock failure", status="transport_error")
            return (_state(input_data.raw_interactions[0].interaction_id),)

    reconstructor = FailsFirst()
    summary = run_batch(
        dataset="memconflict",
        input_path=MEMCONFLICT_FIXTURE,
        output_path=output,
        query_type="dynamic",
        reconstructor=reconstructor,
    )
    record = _records(output)[0]
    assert reconstructor.calls == 1
    assert summary["invalid_or_error"] == 1
    assert record["reconstruction_status"] == "error"
    assert record["validation_status"] == "transport_error"
    assert record["error"]["message"] == "mock failure"


def test_batch_serializes_detailed_bad_request_and_request_diagnostics(tmp_path):
    response = httpx.Response(
        400,
        headers={"x-request-id": "req_batch_456", "authorization": "Bearer secret"},
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )
    provider_error = BadRequestError(
        "Request too large",
        response=response,
        body={"error": {
            "message": "Request too large",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        }},
    )
    provider = OpenAIGovernanceStateReconstructor(
        client=SimpleNamespace(responses=SimpleNamespace(parse=lambda **kwargs: (_ for _ in ()).throw(provider_error))),
        model="mock-openai",
        cache_dir=tmp_path / "cache",
    )
    output = tmp_path / "bad-request.jsonl"
    run_batch(
        dataset="gatemem",
        input_path=GATEMEM_FIXTURE,
        output_path=output,
        query_type="privacy",
        reconstructor=provider,
    )
    record = _records(output)[0]
    assert record["validation_status"] == "transport_error"
    assert record["error"]["provider_exception"] == "BadRequestError"
    assert record["error"]["http_status"] == 400
    assert record["error"]["provider_code"] == "context_length_exceeded"
    assert record["error"]["provider_type"] == "invalid_request_error"
    assert record["error"]["provider_message"] == "Request too large"
    assert record["error"]["request_id"] == "req_batch_456"
    diagnostics = record["request_size_diagnostics"]
    assert diagnostics["interaction_count"] > 0
    assert diagnostics["serialized_prompt_input_byte_count"] > 0
    serialized = json.dumps(record)
    assert "Bearer secret" not in serialized
    assert "x-request-id" not in serialized


def test_batch_task_type_filtering_gate_mem_and_memconflict(tmp_path):
    gate_output = tmp_path / "gate.jsonl"
    gate_reconstructor = RecordingReconstructor()
    run_batch(
        dataset="gatemem",
        input_path=GATEMEM_FIXTURE,
        output_path=gate_output,
        query_type="privacy",
        reconstructor=gate_reconstructor,
    )
    assert [record["task_type"] for record in _records(gate_output)] == ["privacy"]

    conflict_output = tmp_path / "conflict.jsonl"
    conflict_reconstructor = RecordingReconstructor()
    run_batch(
        dataset="memconflict",
        input_path=MEMCONFLICT_FIXTURE,
        output_path=conflict_output,
        query_type="conditional",
        reconstructor=conflict_reconstructor,
    )
    assert [record["task_type"] for record in _records(conflict_output)] == ["conditional"]


def test_batch_propagates_provider_telemetry_to_jsonl(tmp_path):
    output = tmp_path / "telemetry.jsonl"

    class TelemetryReconstructor(RecordingReconstructor):
        model = "telemetry-model"
        last_telemetry = {
            "provider": "openai",
            "cache_hit": False,
            "latency_ms": 12.5,
            "api_latency_ms": 12.5,
            "usage_available": True,
            "usage": {"input_tokens": 101, "output_tokens": 23, "total_tokens": 124},
            "input_tokens": 101,
            "output_tokens": 23,
            "total_tokens": 124,
        }

    reconstructor = TelemetryReconstructor()
    run_batch(
        dataset="gatemem",
        input_path=GATEMEM_FIXTURE,
        output_path=output,
        query_type="privacy",
        reconstructor=reconstructor,
    )
    telemetry = _records(output)[0]["telemetry"]
    assert telemetry == {
        "provider": "openai",
        "cache_hit": False,
        "latency_ms": 12.5,
        "api_latency_ms": 12.5,
        "api_request_executed": None,
        "original_api_latency_ms": None,
        "usage_available": True,
        "usage": {"input_tokens": 101, "output_tokens": 23, "total_tokens": 124},
        "input_tokens": 101,
        "output_tokens": 23,
        "total_tokens": 124,
        "request_size_diagnostics": None,
        "reconstruction_policy": None,
        "streaming_diagnostics": None,
        "chunk_count": None,
        "interaction_count": None,
        "all_interactions_processed": None,
        "per_chunk": None,
        "total_input_tokens": None,
        "total_output_tokens": None,
        "total_api_latency_ms": None,
    }


def test_batch_marks_cache_hit_without_api_latency(tmp_path):
    output = tmp_path / "cache-hit.jsonl"

    class CachedReconstructor(RecordingReconstructor):
        model = "cached-model"
        last_telemetry = {
            "provider": "openai",
            "cache_hit": True,
            "latency_ms": None,
            "api_latency_ms": None,
            "usage_available": False,
            "usage": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    reconstructor = CachedReconstructor()
    run_batch(
        dataset="gatemem",
        input_path=GATEMEM_FIXTURE,
        output_path=output,
        query_type="privacy",
        reconstructor=reconstructor,
    )
    record = _records(output)[0]
    assert record["cache_hit"] is True
    assert record["telemetry"]["latency_ms"] is None
    assert record["telemetry"]["usage"] is None


def test_real_provider_path_propagates_telemetry_to_batch_jsonl(tmp_path):
    output = tmp_path / "provider-telemetry.jsonl"

    class Responses:
        def parse(self, **kwargs):
            return SimpleNamespace(
                output_parsed={"states": [{"interaction_id": "mini_office_001::t001", "is_deleted": False}]},
                usage=SimpleNamespace(input_tokens=17, output_tokens=5, total_tokens=22),
            )

    provider = OpenAIGovernanceStateReconstructor(
        client=SimpleNamespace(responses=Responses()),
        model="mock-openai",
        cache_dir=tmp_path / "new-cache",
    )
    run_batch(
        dataset="gatemem",
        input_path=GATEMEM_FIXTURE,
        output_path=output,
        query_type="privacy",
        reconstructor=provider,
    )
    record = _records(output)[0]
    assert record["cache_hit"] is False
    assert record["usage"] == {
        "input_tokens": 17,
        "output_tokens": 5,
        "total_tokens": 22,
    }
    assert isinstance(record["latency_ms"], (int, float))
    assert record["request_size_diagnostics"]["interaction_count"] > 0
    assert record["telemetry"]["usage"] == {
        "input_tokens": 17,
        "output_tokens": 5,
        "total_tokens": 22,
    }
    assert record["telemetry"]["latency_ms"] >= 0


def test_batch_cache_replay_preserves_original_telemetry(tmp_path):
    output = tmp_path / "cache-replay.jsonl"
    cache_dir = tmp_path / "cache"

    class Responses:
        def __init__(self, should_fail=False):
            self.should_fail = should_fail

        def parse(self, **kwargs):
            if self.should_fail:
                raise AssertionError("cache replay unexpectedly called the API")
            return SimpleNamespace(
                output_parsed={"states": []},
                usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120),
            )

    first = OpenAIGovernanceStateReconstructor(
        client=SimpleNamespace(responses=Responses()),
        model="cache-replay-model",
        cache_dir=cache_dir,
    )
    run_batch(
        dataset="gatemem",
        input_path=GATEMEM_FIXTURE,
        output_path=output,
        query_type="privacy",
        reconstructor=first,
    )
    second = OpenAIGovernanceStateReconstructor(
        client=SimpleNamespace(responses=Responses(should_fail=True)),
        model="cache-replay-model",
        cache_dir=cache_dir,
    )
    run_batch(
        dataset="gatemem",
        input_path=GATEMEM_FIXTURE,
        output_path=output,
        query_type="privacy",
        overwrite=True,
        reconstructor=second,
    )
    record = _records(output)[0]
    assert record["cache_hit"] is True
    assert record["usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }
    assert record["latency_ms"] is None
    assert isinstance(record["api_latency_ms"], (int, float))


def test_legacy_cache_without_telemetry_still_loads(tmp_path):
    from eval.openai_reconstruction import _cache_payload, build_reconstruction_prompt
    import hashlib

    from eval.gatemem_adapter import GateMemDatasetAdapter

    example = next(
        item for item in GateMemDatasetAdapter().load(GATEMEM_FIXTURE)
        if item.query.metadata.get("query_type") == "privacy"
    )
    provider = OpenAIGovernanceStateReconstructor(
        client=SimpleNamespace(responses=SimpleNamespace(parse=lambda **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")))),
        model="legacy-cache-model",
        cache_dir=tmp_path,
    )
    reconstruction_input = GovernanceReconstructionInput.from_example(example)
    prompt = build_reconstruction_prompt(reconstruction_input)
    cache_key = hashlib.sha256(
        json.dumps(_cache_payload(provider.model, prompt, reconstruction_input), sort_keys=True, default=str).encode()
    ).hexdigest()
    (tmp_path / f"{cache_key}.json").write_text(
        json.dumps({"states": [{"interaction_id": example.interactions[0].interaction_id, "is_deleted": False}]}),
        encoding="utf-8",
    )
    states = provider.reconstruct(reconstruction_input)
    assert states
    assert provider.last_telemetry["cache_hit"] is True
    assert provider.last_telemetry["latency_ms"] is None
    assert provider.last_telemetry["usage"] is None


def test_batch_preserves_incomplete_trailing_line_on_resume(tmp_path):
    output = tmp_path / "results.jsonl"
    output.write_text('{"query_id":"complete"}\n{"query_id":"partial"', encoding="utf-8")
    reconstructor = RecordingReconstructor()
    run_batch(dataset="gatemem", input_path=GATEMEM_FIXTURE, output_path=output, limit=1, reconstructor=reconstructor)
    raw_lines = output.read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line) for line in raw_lines)
