from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval.reconstruction_provider import ProviderResponse, ProviderTelemetry, ProviderTransportError
from eval.providers.openai_provider import DownstreamAnswer, OpenAIDownstreamProvider
from eval.system_validation_scoring import max_output_tokens_for_row
from scripts.run_final_main_smoke_deepseek import render_prompt as deepseek_prompt
from scripts.run_final_main_openai import EXPECTED_MANIFEST_SHA, main, run, select_long_output_validation_ids, select_stratified_validation_ids
from scripts.run_final_main_smoke_deepseek import _execute_condition, _explicit_token_limit_truncation
from scripts.run_final_main_smoke_deepseek import render_prompt as shared_prompt


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"


def test_openai_full_dry_run_is_frozen_and_makes_no_provider_calls(tmp_path):
    calls = []
    result = run(
        manifest=FULL,
        results_dir=tmp_path,
        dry_run=True,
        workers=8,
        provider_factory=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    summary = result["summary"]
    assert summary["provider"] == "openai"
    assert summary["model"] == "gpt-5.6-sol"
    assert summary["manifest_sha256"] == EXPECTED_MANIFEST_SHA
    assert summary["total_rows"] == 4891
    assert summary["scoring_preflight"]["total_supported"] == 4891
    assert summary["api_calls_made"] == 0
    assert calls == []


def test_openai_uses_same_prompt_and_task_budgets_as_deepseek():
    rows = [json.loads(line) for line in FULL.read_text(encoding="utf-8").splitlines()]
    for row in rows[::613]:
        assert shared_prompt(row) == deepseek_prompt(row)
        assert hashlib.sha256(shared_prompt(row).encode()).hexdigest() == hashlib.sha256(deepseek_prompt(row).encode()).hexdigest()
        assert max_output_tokens_for_row(row) in {128, 256, 512, 768}


def test_openai_runner_rejects_invalid_workers(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        run(manifest=FULL, results_dir=tmp_path, dry_run=True, workers=0)


class _FakeOpenAIProvider:
    provider_name = "openai"

    def __init__(self):
        self.calls = []
        self.closed = 0

    def request_downstream(self, prompt, *, max_tokens=None):
        self.calls.append((prompt, max_tokens))
        return ProviderResponse(
            {"answer": "cannot be determined"},
            ProviderTelemetry("openai", "gpt-5.6-sol", 1, 1, 2, 0, "fake"),
        )

    def close(self):
        self.closed += 1


def test_openai_runner_uses_provider_label_and_closes_provider(tmp_path):
    fake = _FakeOpenAIProvider()
    result = run(
        manifest=FULL,
        results_dir=tmp_path,
        limit=2,
        provider_factory=lambda *args, **kwargs: fake,
        workers=1,
    )
    assert len(result["rows"]) == 2
    assert all(row["provider"] == "openai" for row in result["rows"])
    assert [budget for _prompt, budget in fake.calls] == [256, 256]
    assert fake.closed == 1


def test_stratified_validation_helper_is_deterministic_and_spans_strata():
    rows = [json.loads(line) for line in FULL.read_text(encoding="utf-8").splitlines()]
    ids_a = select_stratified_validation_ids(rows, limit=60)
    ids_b = select_stratified_validation_ids(rows, limit=60)
    assert ids_a == ids_b
    selected = {row["condition_id"] for row in rows if row["condition_id"] in set(ids_a)}
    assert len(selected) == 60
    assert {row["source_dataset"] for row in rows if row["condition_id"] in selected} == {"longmemeval", "memora", "oaks_novel"}


def test_long_output_validation_is_exactly_30_project_proposal_bases_and_90_rows():
    rows = [json.loads(line) for line in FULL.read_text(encoding="utf-8").splitlines()]
    ids_a = select_long_output_validation_ids(rows)
    ids_b = select_long_output_validation_ids(rows)
    assert ids_a == ids_b
    selected = [row for row in rows if row["condition_id"] in set(ids_a)]
    assert len(selected) == 90
    assert len({row["base_id"] for row in selected}) == 30
    assert {row["condition_type"] for row in selected} == {"clean", "corrupted", "enforced"}
    assert all(row["source_dataset"] == "memora" for row in selected)
    from eval.final_main_token_budget import memora_task_subtype
    assert all(memora_task_subtype(row) == "content_project_proposal" for row in selected)
    bases = {row["base_id"] for row in selected}
    assert all(sum(row["condition_type"] == kind for row in selected if row["base_id"] == base) == 1 for base in bases for kind in ("clean", "corrupted", "enforced"))


def test_transport_truncation_detection_is_explicit_and_provider_neutral():
    assert not _explicit_token_limit_truncation({"error_type": "ValidationError", "error_message": "EOF while parsing JSON"})
    assert _explicit_token_limit_truncation({"attempt_errors": [{"incomplete_details": {"reason": "max_output_tokens"}}]})


def test_openai_raw_response_is_retained_before_schema_validation():
    class Responses:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return type("Response", (), {
                "status": "completed",
                "output_text": '{"answer":"complete"}',
                "usage": type("Usage", (), {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5})(),
                "id": "response-1",
            })()

    provider = OpenAIDownstreamProvider(client=type("Client", (), {"responses": Responses()})(), max_tokens=768)
    result = provider.request_downstream("prompt", max_tokens=768)
    assert result.parsed == {"answer": "complete"}
    assert provider.client.responses.kwargs["max_output_tokens"] == 768
    assert provider.client.responses.kwargs["text"]["format"]["type"] == "json_schema"
    schema = provider.client.responses.kwargs["text"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert list(schema["properties"]) == ["answer"]


def test_openai_explicit_token_limit_is_reported_without_retry_signal():
    class Responses:
        def create(self, **kwargs):
            return type("Response", (), {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output_text": '{"answer":"partial',
                "usage": {"input_tokens": 2, "output_tokens": 768, "total_tokens": 770},
            })()

    provider = OpenAIDownstreamProvider(client=type("Client", (), {"responses": Responses()})())
    with pytest.raises(ProviderTransportError) as raised:
        provider.request_downstream("prompt", max_tokens=768)
    assert raised.value.details["error_category"] == "token_limit_truncation"
    assert raised.value.details["raw_output_text"] == '{"answer":"partial'


def test_final_token_limit_telemetry_is_propagated_to_result_row():
    class Provider:
        def request_downstream(self, prompt, *, max_tokens=None):
            raise ProviderTransportError(
                "token limit",
                provider="openai",
                model="model",
                details={
                    "error_category": "token_limit_truncation",
                    "token_limit_truncation": True,
                    "input_tokens": 420,
                    "output_tokens": 768,
                    "total_tokens": 1188,
                    "raw_output_text": '{"answer":"partial',
                    "response_status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            )

    row = {
        "condition_id": "telemetry",
        "base_id": "base",
        "condition_type": "clean",
        "source_dataset": "memora",
        "analysis_event_type": "update",
        "task_question": "q",
        "task_oracle_answer": {"content_data": {"project_title": "x"}},
        "task_scoring_spec": {"subtype": "content_project_proposal"},
        "oracle_state": {"current_status": "valid"},
    }
    result = _execute_condition(row, "same prompt", Provider(), model="model", manifest_sha="sha")
    assert result["status"] == "error"
    assert result["attempt_count"] == 1
    assert result["input_tokens"] == 420
    assert result["output_tokens"] == 768
    assert result["total_tokens"] == 1188
    assert result["raw_output_text"] == '{"answer":"partial'
    assert result["response_status"] == "incomplete"
    assert result["incomplete_details"] == {"reason": "max_output_tokens"}
    assert result["truncated_output"] is True
    assert result["truncation_reason"] == "max_output_tokens"


def test_common_runner_does_not_retry_explicit_token_limit_for_either_provider():
    class Provider:
        provider_name = "provider"

        def __init__(self):
            self.calls = 0

        def request_downstream(self, prompt, *, max_tokens=None):
            self.calls += 1
            raise ProviderTransportError(
                "limited", provider="provider", model="model",
                details={"error_category": "token_limit_truncation", "token_limit_truncation": True},
            )

    row = {
        "condition_id": "x",
        "base_id": "b",
        "condition_type": "clean",
        "source_dataset": "longmemeval",
        "analysis_event_type": "temporal",
        "task_question": "q",
        "task_oracle_answer": "a",
        "oracle_state": {"current_status": "valid"},
    }
    for provider_name in ("openai", "deepseek"):
        provider = Provider()
        result = _execute_condition(row, "same prompt", provider, model="model", manifest_sha="sha")
        assert provider.calls == 1
        assert result["attempt_count"] == 1
        assert result["truncated_output"] is True


def test_common_runner_does_not_retry_deterministic_openai_400_but_retries_429_and_5xx():
    row = {
        "condition_id": "x", "base_id": "b", "condition_type": "clean",
        "source_dataset": "longmemeval", "analysis_event_type": "temporal",
        "task_question": "q", "task_oracle_answer": "a",
        "oracle_state": {"current_status": "valid"},
    }

    class SequenceProvider:
        def __init__(self, status):
            self.status = status
            self.calls = 0

        def request_downstream(self, prompt, *, max_tokens=None):
            self.calls += 1
            if self.calls < 3:
                raise ProviderTransportError("provider error", provider="openai", model="model", details={"http_status": self.status})
            return ProviderResponse({"answer": "a"}, ProviderTelemetry("openai", "model", 1, 1, 2, 0, "id"))

    bad = SequenceProvider(400)
    bad_result = _execute_condition(row, "prompt", bad, model="model", manifest_sha="sha", sleep_fn=lambda _: None)
    assert bad.calls == 1
    assert bad_result["attempt_count"] == 1

    for status in (429, 500):
        transient = SequenceProvider(status)
        recovered = _execute_condition(row, "prompt", transient, model="model", manifest_sha="sha", sleep_fn=lambda _: None, random_fn=lambda: 0)
        assert transient.calls == 3
        assert recovered["status"] == "success"


def test_openai_help_exposes_targeted_condition_options(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "--condition-id" in help_text
    assert "--condition-ids-file" in help_text


def test_exact_condition_id_selects_one_manifest_row_without_provider_calls(tmp_path):
    first = json.loads(FULL.read_text(encoding="utf-8").splitlines()[0])
    calls = []
    result = run(manifest=FULL, results_dir=tmp_path, dry_run=True, selected_condition_ids={first["condition_id"]}, provider_factory=lambda *args, **kwargs: calls.append(1))
    assert len(result["rows"]) == 1
    assert result["rows"][0]["condition_id"] == first["condition_id"]
    assert calls == []


def test_condition_ids_file_preserves_frozen_manifest_order(tmp_path):
    rows = [json.loads(line) for line in FULL.read_text(encoding="utf-8").splitlines()]
    requested = [rows[4]["condition_id"], rows[0]["condition_id"]]
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("\n".join(requested) + "\n", encoding="utf-8")
    result = run(manifest=FULL, results_dir=tmp_path / "out", dry_run=True, selected_condition_ids=set(requested))
    assert [row["condition_id"] for row in result["rows"]] == [requested[1], requested[0]]


def test_targeted_cli_rejects_missing_duplicate_and_conflicting_selection(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        run(manifest=FULL, results_dir=tmp_path / "missing", dry_run=True, selected_condition_ids={"does-not-exist"})
    ids_file = tmp_path / "duplicate.txt"
    ids_file.write_text("x\nx\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--condition-ids-file", str(ids_file), "--dry-run"])
    with pytest.raises(SystemExit):
        main(["--condition-id", "x", "--limit", "1", "--dry-run"])
    with pytest.raises(SystemExit):
        main(["--condition-id", "x", "--stratified-validation", "--dry-run"])
