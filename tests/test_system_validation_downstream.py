import json
from pathlib import Path

import pytest

from eval.reconstruction_provider import ProviderResponse, ProviderTelemetry, ProviderTransportError
from scripts.run_system_validation_downstream import (
    FORMAL_STATUS,
    MANIFEST_SHA256,
    build_downstream_prompt,
    prompt_sha256,
    run_downstream,
    _condition_metadata,
    validate_formal_system_result,
)


def _formal_result(path: Path, *, status=FORMAL_STATUS, manifest=MANIFEST_SHA256):
    records = []
    for base in range(60):
        for condition in ("clean", "fault_a", "fault_b"):
            records.append({
                "base_example_id": f"base-{base:02d}",
                "source_dataset": "synthetic",
                "governance_dimension": "authorization",
                "condition": condition,
                "fault_type": "clean" if condition == "clean" else "fault",
                "memory_system": "mem0",
                "manifest_sha256": manifest,
                "requested_retrieval_k": 1,
                "experiment_status": status,
                "cohort_version": "v1",
                "query": "What is the answer?",
                "final_model_visible_context": "A fixed context.",
            })
    payload = {
        "summary": {
            "status": status,
            "cohort_version": "v1",
            "manifest_sha256": manifest,
            "selected_base_examples": 60,
            "condition_rows": 180,
            "native_retrieval_error_n": 0,
            "native_retrieval_empty_n": 0,
            "invariant_failure_n": 0,
            "requested_retrieval_k": 1,
            "memory_system": "mem0",
        },
        "records": records,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


class FakeProvider:
    provider_name = "fake"
    model = "fake-model"

    def __init__(self, fail=False):
        self.prompts = []
        self.fail = fail

    def request_downstream(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise ProviderTransportError("fake failure", provider="fake", model=self.model)
        return ProviderResponse(
            parsed={"answer": "determined"},
            telemetry=ProviderTelemetry("fake", self.model, 3, 2, 5, 0.1, "req-1"),
        )


def test_valid_formal_input_and_prompt_has_only_query_context(tmp_path):
    system_result = tmp_path / "system.json"
    _formal_result(system_result)
    provider = FakeProvider()
    result = run_downstream(
        system_result=system_result,
        provider="openai",
        model="fake-model",
        results_dir=tmp_path / "out",
        provider_factory=lambda _provider, _model: provider,
    )
    assert result["summary"]["inference_success_n"] == 180
    assert len(provider.prompts) == 180
    prompt = provider.prompts[0]
    assert "What is the answer?" in prompt
    assert "A fixed context." in prompt
    assert "mem0" not in prompt
    assert "fault_a" not in prompt
    assert result["records"][0]["prompt_sha256"] == prompt_sha256(prompt)
    assert result["records"][0]["downstream_response"] == "determined"


def test_invalid_formal_input_is_rejected_before_provider_creation(tmp_path):
    system_result = tmp_path / "system.json"
    _formal_result(system_result, status="NON-FROZEN / SMOKE")
    called = False

    def factory(_provider, _model):
        nonlocal called
        called = True
        return FakeProvider()

    with pytest.raises(ValueError, match="formal system result failed validation"):
        run_downstream(
            system_result=system_result,
            provider="openai",
            model="fake-model",
            results_dir=tmp_path / "out",
            provider_factory=factory,
        )
    assert not called


@pytest.mark.parametrize("field", ["manifest_sha256", "native_retrieval_error_n", "native_retrieval_empty_n", "invariant_failure_n"])
def test_formal_gate_fields_are_enforced(tmp_path, field):
    system_result = tmp_path / "system.json"
    payload = _formal_result(system_result)
    payload["summary"][field] = "wrong" if "sha" in field else 1
    system_result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        run_downstream(
            system_result=system_result,
            provider="openai",
            model="fake-model",
            results_dir=tmp_path / "out",
            provider_factory=lambda *_: pytest.fail("provider must not be constructed"),
        )


def test_resume_avoids_duplicate_calls_and_malformed_cache_rejected(tmp_path):
    system_result = tmp_path / "system.json"
    _formal_result(system_result)
    out = tmp_path / "out"
    first = FakeProvider()
    run_downstream(system_result=system_result, provider="openai", model="fake-model", results_dir=out, provider_factory=lambda *_: first)
    assert len(first.prompts) == 180
    second = FakeProvider()
    resumed = run_downstream(system_result=system_result, provider="openai", model="fake-model", results_dir=out, provider_factory=lambda *_: second)
    assert len(second.prompts) == 0
    assert resumed["summary"]["response_reused_n"] == 180

    (out / "system_validation_downstream.jsonl").write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed downstream cache"):
        run_downstream(system_result=system_result, provider="openai", model="fake-model", results_dir=out, provider_factory=lambda *_: FakeProvider())


def test_provider_errors_are_recorded_without_scoring(tmp_path):
    system_result = tmp_path / "system.json"
    _formal_result(system_result)
    result = run_downstream(
        system_result=system_result,
        provider="openai",
        model="fake-model",
        results_dir=tmp_path / "out",
        provider_factory=lambda *_: FakeProvider(fail=True),
    )
    assert result["summary"]["inference_error_n"] == 180
    assert result["summary"]["scoring_performed"] is False
    assert result["records"][0]["downstream_response"] is None
    assert result["records"][0]["downstream_error"]["type"] == "ProviderTransportError"


def test_dry_run_writes_separate_auditable_prompts_without_provider_calls(tmp_path):
    system_result = tmp_path / "system.json"
    _formal_result(system_result)

    def never_construct_provider(*_args):
        pytest.fail("dry-run must not construct a provider")

    result = run_downstream(
        system_result=system_result,
        provider="openai",
        model="fake-model",
        results_dir=tmp_path / "out",
        provider_factory=never_construct_provider,
        dry_run=True,
    )
    summary = result["summary"]
    assert summary["status"] == "FROZEN / FORMAL / DRY-RUN"
    assert summary["condition_rows"] == 180
    assert summary["unique_base_examples"] == 60
    assert summary["api_calls_made"] == 0
    assert summary["validation_errors"] == 0
    assert len(result["records"]) == 180
    assert all("downstream_response" not in row for row in result["records"])
    assert (tmp_path / "out" / "system_validation_downstream_dry_run.json").exists()
    assert (tmp_path / "out" / "system_validation_downstream_dry_run_summary.json").exists()
    assert not (tmp_path / "out" / "system_validation_downstream.jsonl").exists()
    assert "fault_a" not in result["records"][0]["prompt_text"]
    assert "mem0" not in result["records"][0]["prompt_text"]


def test_dry_run_allows_metadata_words_that_are_part_of_inputs(tmp_path):
    system_result = tmp_path / "system.json"
    payload = _formal_result(system_result)
    payload["records"][0]["query"] = "Explain the condition in this memory."
    payload["records"][0]["final_model_visible_context"] = "The governance gold is not part of the memory."
    system_result.write_text(json.dumps(payload), encoding="utf-8")
    result = run_downstream(
        system_result=system_result,
        provider="openai",
        model="fake-model",
        results_dir=tmp_path / "out",
        provider_factory=lambda *_: pytest.fail("dry-run must not construct a provider"),
        dry_run=True,
    )
    assert "condition" in result["records"][0]["prompt_text"]
    assert "governance gold" in result["records"][0]["prompt_text"]


def test_prompt_template_is_provider_and_condition_invariant():
    first = build_downstream_prompt("same query", "same context")
    second = build_downstream_prompt("same query", "same context")
    assert first == second
    assert "system" not in first.lower()
    assert "condition" not in first.lower()
    assert prompt_sha256(first) == prompt_sha256(second)


@pytest.mark.parametrize(
    ("dimension", "labels"),
    [
        ("authorization", ("clean", "conflicting", "flipped")),
        ("deletion", ("clean", "flipped", "stale")),
        ("temporal", ("clean", "missing", "stale")),
    ],
)
def test_original_experimental_condition_labels_are_preserved(dimension, labels):
    del dimension  # The normalization is dimension-independent.
    assert _condition_metadata({"condition": "clean", "fault_type": "clean"}) == ("clean", None)
    for label in labels[1:]:
        assert _condition_metadata({"condition": "fault", "fault_type": label}) == (label, label)


def test_explicit_condition_is_preserved_and_not_prompt_metadata():
    assert _condition_metadata({"condition": "missing", "fault_type": "missing"}) == ("missing", "missing")
    prompt = build_downstream_prompt("query", "context")
    assert "fault_type" not in prompt
