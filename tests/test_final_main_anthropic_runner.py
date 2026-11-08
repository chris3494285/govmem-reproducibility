from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.providers.anthropic_provider import AnthropicMessagesProvider
from eval.reconstruction_provider import ProviderTransportError
from scripts.run_final_main_anthropic import run


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
BUDGETS = ROOT / "configs/final_main_token_budget_v2.json"
REPAIR_BUDGETS = ROOT / "configs/final_main_token_budget_anthropic_repair_v2.json"
TRUNCATED_IDS = ROOT / "results/final_main_anthropic_sonnet5/truncated_condition_ids.txt"
NONTRUNCATED_IDS = ROOT / "results/final_main_anthropic_sonnet5/nontrunc_failure_condition_ids.txt"


def test_anthropic_dry_run_covers_frozen_cohort_without_provider_calls(tmp_path: Path):
    calls = []
    result = run(manifest=MANIFEST, results_dir=tmp_path, model="claude-sonnet-5", workers=8, dry_run=True, token_budget_config=BUDGETS, provider_factory=lambda *args, **kwargs: calls.append(1))
    summary = result["summary"]
    assert summary["total_rows"] == 4891
    assert summary["manifest_sha256"] == "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"
    assert summary["scoring_preflight"]["total_supported"] == 4891
    assert summary["scoring_preflight"]["unresolved"] == []
    assert summary["api_calls_made"] == 0
    assert summary["workers"] == 8
    assert summary["resolved_token_budget_class_counts"]["memora_project_proposal"] == 581
    assert summary["resolved_token_budget_class_counts"]["memora_meeting_notes"] == 245
    assert summary["resolved_max_output_tokens"]["memora_project_proposal"] == 1536
    assert summary["resolved_max_output_tokens"]["memora_meeting_notes"] == 256
    assert calls == []


def test_anthropic_cost_estimate_uses_standard_local_prices(tmp_path: Path):
    result = run(manifest=MANIFEST, results_dir=tmp_path, model="claude-sonnet-5", dry_run=True, token_budget_config=BUDGETS)
    cost = result["summary"]["cost_estimate"]
    assert cost["input_price_per_million"] == 2.0
    assert cost["output_price_per_million"] == 10.0
    assert cost["estimated_synchronous_upper_bound_cost"] == pytest.approx(
        cost["estimated_input_tokens"] / 1_000_000 * 2.0 + cost["estimated_max_output_tokens"] / 1_000_000 * 10.0
    )


def test_anthropic_repair_selects_exact_truncated_ids_and_uses_repair_budgets(tmp_path: Path):
    truncated = {line.strip() for line in TRUNCATED_IDS.read_text().splitlines() if line.strip()}
    nontruncated = {line.strip() for line in NONTRUNCATED_IDS.read_text().splitlines() if line.strip()}
    assert len(truncated) == 386
    assert not truncated & nontruncated
    result = run(
        manifest=MANIFEST, results_dir=tmp_path, model="claude-sonnet-5", workers=4,
        dry_run=True, token_budget_config=REPAIR_BUDGETS, selected_condition_ids=truncated,
    )
    summary = result["summary"]
    assert summary["requested_rows"] == 386
    assert summary["api_calls_made"] == 0
    assert summary["manifest_sha256"] == "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"
    assert summary["scoring_preflight"]["total_supported"] == 386
    assert set(row["condition_id"] for row in result["rows"]) == truncated
    assert "memora_project_proposal" not in summary["resolved_token_budget_class_counts"]
    assert summary["resolved_max_output_tokens"]["longmemeval"] == 512
    assert summary["resolved_max_output_tokens"]["memora_email_writeup"] == 1024
    assert summary["resolved_max_output_tokens"]["memora_meeting_notes"] == 1024
    assert summary["resolved_max_output_tokens"]["memora_preference"] == 768
    assert summary["resolved_max_output_tokens"]["memora_todos"] == 512
    assert summary["resolved_max_output_tokens"]["oaks_novel"] == 256


def test_anthropic_condition_id_selection_rejects_missing_ids(tmp_path: Path):
    with pytest.raises(ValueError, match="not found in frozen manifest"):
        run(
            manifest=MANIFEST, results_dir=tmp_path, dry_run=True,
            token_budget_config=REPAIR_BUDGETS, selected_condition_ids={"does-not-exist"},
        )


def test_repair_budget_is_rejected_for_normal_full_run(tmp_path: Path):
    with pytest.raises(ValueError, match="require --condition-ids-file"):
        run(manifest=MANIFEST, results_dir=tmp_path, dry_run=True, token_budget_config=REPAIR_BUDGETS)


class _Messages:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _response(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=4),
        request_id="req-1",
        stop_reason=stop_reason,
    )


def test_anthropic_provider_parses_mocked_response_and_preserves_request_contract():
    messages = _Messages(response=_response(json.dumps({"answer": "ok"})))
    provider = AnthropicMessagesProvider(client=SimpleNamespace(messages=messages), model="claude-sonnet-5", temperature=0, max_tokens=256)
    result = provider.request_downstream("same prompt", max_tokens=128)
    assert result.parsed == {"answer": "ok"}
    assert messages.calls[0]["model"] == "claude-sonnet-5"
    assert messages.calls[0]["max_tokens"] == 128
    assert "temperature" not in messages.calls[0]
    assert messages.calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_malformed_and_token_limit_responses_are_transport_failures():
    malformed = AnthropicMessagesProvider(client=SimpleNamespace(messages=_Messages(response=_response("not-json"))), model="claude-sonnet-5")
    with pytest.raises(ProviderTransportError) as malformed_error:
        malformed.request_downstream("prompt")
    assert malformed_error.value.details["error_category"] == "structural_invalid"

    truncated_response = _response('{"answer":"partial', stop_reason="max_tokens")
    truncated = AnthropicMessagesProvider(client=SimpleNamespace(messages=_Messages(response=truncated_response)), model="claude-sonnet-5")
    with pytest.raises(ProviderTransportError) as truncated_error:
        truncated.request_downstream("prompt")
    assert truncated_error.value.details["error_category"] == "token_limit_truncation"
