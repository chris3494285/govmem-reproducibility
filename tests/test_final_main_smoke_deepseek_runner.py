import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.reconstruction_provider import ProviderResponse, ProviderTelemetry
from eval.reconstruction_provider import ProviderTransportError
from scripts.run_final_main_smoke_deepseek import (
    EXPECTED_MANIFEST_SHA,
    render_prompt,
    run,
)


MANIFEST = Path("data/final_main_governance_smoke_condition_manifest_v1.jsonl")


class FakeProvider:
    def __init__(self, model, **kwargs):
        self.model = model
        self.calls = []

    def request_downstream(self, prompt):
        self.calls.append(prompt)
        return ProviderResponse(
            {"answer": "ok"},
            ProviderTelemetry("deepseek", self.model, 10, 4, 14, 1.0, "req-1"),
            raw=SimpleNamespace(fake=True),
        )


def test_dry_run_validates_frozen_manifest_without_provider(tmp_path):
    called = []
    result = run(manifest=MANIFEST, results_dir=tmp_path, dry_run=True, provider_factory=lambda *a, **k: called.append(1))
    assert result["summary"]["api_calls_made"] == 0
    assert len(result["rows"]) == 160
    assert result["summary"]["condition_counts"] == {"clean": 24, "corrupted": 68, "enforced": 68}
    assert called == []
    assert result["summary"]["manifest_sha256"] == EXPECTED_MANIFEST_SHA


def test_prompt_is_deterministic_and_condition_specific():
    rows = [json.loads(x) for x in MANIFEST.read_text().splitlines()]
    clean = next(r for r in rows if r["condition_type"] == "clean")
    corrupted = next(r for r in rows if r["condition_type"] == "corrupted")
    assert render_prompt(clean) == render_prompt(clean)
    assert render_prompt(clean) != render_prompt(corrupted)
    assert hashlib.sha256(render_prompt(clean).encode()).hexdigest() == hashlib.sha256(render_prompt(clean).encode()).hexdigest()


def test_oaks_prompt_requires_neutral_label_without_answer_example():
    rows = [json.loads(x) for x in Path("data/final_main_governance_smoke_condition_manifest_v2.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r["source_dataset"] == "oaks_novel" and r["condition_type"] == "clean")
    prompt = render_prompt(row)
    instruction, options = prompt.split("\n\nOptions:\n", 1)
    expected_labels = ", ".join(chr(65 + index) for index in range(len(row["task_options"])))
    assert f"option labels {expected_labels}" in instruction
    assert 'JSON with exactly one string field named "answer"' in instruction
    for label in "ABCDE":
        assert '{"answer":"' + label + '"}' not in prompt
    assert all(f"{label}. " in options for label in "ABCDE")
    # The oracle option is available only through the benchmark-native options
    # list, never as an instruction/example outside that list.
    oracle = row["task_oracle_answer"]
    assert oracle in options
    assert oracle not in instruction


@pytest.mark.parametrize("count", [5, 6, 7])
def test_oaks_prompt_uses_dynamic_option_labels(count):
    row = {"task_answer_type": "multiple_choice", "task_question": "Q", "task_options": [f"option {i}" for i in range(count)], "model_visible_context_text": "evidence", "condition_type": "clean"}
    prompt = render_prompt(row)
    options = prompt.split("\n\nOptions:\n", 1)[1]
    labels = [chr(65 + i) for i in range(count)]
    assert all(f"{label}. option" in options for label in labels)
    assert f"option labels {', '.join(labels)}" in prompt


def test_mocked_success_records_parse_and_telemetry(tmp_path):
    instances = []

    def factory(model, **kwargs):
        value = FakeProvider(model, **kwargs)
        instances.append(value)
        return value

    result = run(manifest=MANIFEST, results_dir=tmp_path, limit=2, force=True, provider_factory=factory)
    assert len(result["rows"]) == 2
    assert all(row["parse_success"] for row in result["rows"])
    assert result["rows"][0]["input_tokens"] == 10
    assert result["rows"][0]["request_id"] == "req-1"
    assert len(instances[0].calls) == 2


def test_resume_reuses_successes_and_rejects_identity_mismatch(tmp_path):
    first = run(manifest=MANIFEST, results_dir=tmp_path, limit=2, force=True, provider_factory=FakeProvider)
    assert first["summary"]["successful_rows"] == 2
    second_calls = []

    def factory(model, **kwargs):
        provider = FakeProvider(model, **kwargs)
        provider.calls = second_calls
        return provider

    second = run(manifest=MANIFEST, results_dir=tmp_path, limit=2, resume=True, provider_factory=factory)
    assert second_calls == []
    cached = json.loads((tmp_path / "smoke_results.jsonl").read_text().splitlines()[0])
    cached["prompt_sha256"] = "wrong"
    (tmp_path / "smoke_results.jsonl").write_text(json.dumps(cached) + "\n")
    with pytest.raises(ValueError, match="cached prompt identity"):
        run(manifest=MANIFEST, results_dir=tmp_path, limit=2, resume=True, provider_factory=factory)


def test_no_api_key_is_written_to_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-test-value")
    result = run(manifest=MANIFEST, results_dir=tmp_path, dry_run=True)
    payload = json.dumps(result)
    assert "sk-secret-test-value" not in payload


class SequenceProvider:
    def __init__(self, sequence, **kwargs):
        self.sequence = list(sequence)
        self.calls = []

    def request_downstream(self, prompt, *, max_tokens=None):
        self.calls.append((prompt, max_tokens))
        value = self.sequence.pop(0)
        if isinstance(value, Exception):
            raise value
        return ProviderResponse({"answer": value}, ProviderTelemetry("deepseek", "test", 1, 1, 2, 1.0, "r"))


def test_malformed_json_then_valid_response_recovers(tmp_path):
    provider = SequenceProvider([json.JSONDecodeError("bad", "", 0), "ok"])
    result = run(manifest=MANIFEST, results_dir=tmp_path, limit=1, force=True, provider_factory=lambda *a, **k: provider)
    row = result["rows"][0]
    assert len(provider.calls) == 2
    assert row["attempt_count"] == 2
    assert row["retry_count"] == 1
    assert row["recovered_by_retry"] is True
    assert result["summary"]["rows_recovered_by_retry"] == 1
    assert provider.calls[0] == provider.calls[1]


def test_provider_transport_error_then_valid_response_recovers(tmp_path):
    error = ProviderTransportError("temporary", provider="deepseek", model="test", details={"provider_exception": "APIConnectionError"})
    provider = SequenceProvider([error, "ok"])
    result = run(manifest=MANIFEST, results_dir=tmp_path, limit=1, force=True, provider_factory=lambda *a, **k: provider)
    assert result["rows"][0]["recovered_by_retry"] is True
    assert result["summary"]["total_provider_attempts"] == 2


def test_three_malformed_responses_persist_final_failure(tmp_path):
    provider = SequenceProvider([json.JSONDecodeError("bad", "", 0)] * 3)
    result = run(manifest=MANIFEST, results_dir=tmp_path, limit=1, force=True, provider_factory=lambda *a, **k: provider)
    row = result["rows"][0]
    assert row["status"] == "error"
    assert row["attempt_count"] == 3
    assert row["retry_count"] == 2
    assert len(row["attempt_errors"]) == 3
    assert row["resolved_max_output_tokens"] == 256
    assert result["summary"]["rows_failed_after_retries"] == 1


def test_retry_budget_and_prompt_are_invariant(tmp_path):
    provider = SequenceProvider([json.JSONDecodeError("bad", "", 0), "ok"])
    result = run(manifest=MANIFEST, results_dir=tmp_path, limit=1, force=True, provider_factory=lambda *a, **k: provider)
    assert provider.calls[0][0].encode() == provider.calls[1][0].encode()
    assert provider.calls[0][1] == provider.calls[1][1] == result["rows"][0]["resolved_max_output_tokens"]
