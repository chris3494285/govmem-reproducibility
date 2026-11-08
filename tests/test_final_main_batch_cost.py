from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.final_main_batch import batch_cost_estimate, batch_custom_id, build_batch_requests, validate_batch_output
from scripts.run_final_main_openai import run


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"


def _rows():
    return [json.loads(line) for line in FULL.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_batch_custom_ids_are_deterministic_and_unique():
    ids = [row["condition_id"] for row in _rows()[:20]]
    custom = [batch_custom_id(cid) for cid in ids]
    assert custom == [batch_custom_id(cid) for cid in ids]
    assert len(custom) == len(set(custom))


def test_batch_request_body_matches_sync_provider_request():
    from eval.providers.openai_provider import OpenAIDownstreamProvider

    rows = _rows()[:2]
    prompts = {row["condition_id"]: "same scientific prompt" for row in rows}
    budgets = {row["condition_id"]: 256 for row in rows}
    provider = OpenAIDownstreamProvider(client=object(), model="gpt-5.6-sol", reasoning_effort="none")
    requests, reverse = build_batch_requests(rows, provider, prompts, budgets)
    assert reverse[requests[0]["custom_id"]] == rows[0]["condition_id"]
    assert requests[0]["body"] == provider.request_payload(prompts[rows[0]["condition_id"]], max_tokens=256)
    assert requests[0]["body"]["model"] == "gpt-5.6-sol"
    assert requests[0]["body"]["max_output_tokens"] == 256


def test_batch_output_order_is_irrelevant_and_duplicates_fail():
    reverse = {"x": "condition-x", "y": "condition-y"}
    lines = [{"custom_id": "y", "response": {}}, {"custom_id": "x", "response": {}}]
    assert set(validate_batch_output(lines, reverse)) == {"x", "y"}
    with pytest.raises(ValueError, match="Duplicate"):
        validate_batch_output(lines + [{"custom_id": "x"}], reverse)


def test_batch_cost_estimate_labels_upper_bound_and_discount():
    estimate = batch_cost_estimate(input_tokens=1_000_000, max_output_tokens=2_000_000, input_price_per_million=1.0, cached_input_price_per_million=0.1, output_price_per_million=2.0, batch_discount=0.5)
    assert estimate["estimated_synchronous_upper_bound_cost"] == 5.0
    assert estimate["estimated_batch_upper_bound_cost"] == 2.5
    assert "upper bound" in estimate["pricing_basis"]


def test_openai_batch_dry_run_has_zero_api_calls_and_preserves_frozen_rows(tmp_path):
    calls = []
    result = run(manifest=FULL, results_dir=tmp_path, execution_mode="batch", dry_run=True, limit=2, provider_factory=lambda *args, **kwargs: calls.append(1))
    assert result["summary"]["execution_mode"] == "batch"
    assert result["summary"]["api_calls_made"] == 0
    assert result["summary"]["total_rows"] == 2
    assert calls == [1]  # construction only; no provider request/API call
