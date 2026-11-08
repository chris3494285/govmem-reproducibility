from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.final_main_token_budget import GROK_V1_BUDGETS, load_grok_v1_budget_config
from eval.providers.grok_provider import DEFAULT_GROK_BASE_URL, DEFAULT_GROK_MODEL, GrokDownstreamProvider
from scripts.run_final_main_grok import run

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
CONFIG = ROOT / "configs/final_main_token_budget_grok_v1.json"
SHA = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"


def test_grok_full_dry_run_is_offline_and_complete(tmp_path):
    result = run(manifest=MANIFEST, results_dir=tmp_path, dry_run=True, workers=4)
    summary = result["summary"]
    assert summary["requested_rows"] == 4891
    assert summary["api_calls_made"] == 0
    assert summary["manifest_sha256"] == SHA
    assert summary["provider"] == "xai"
    assert summary["model"] == DEFAULT_GROK_MODEL
    assert summary["reasoning_effort"] == "none"
    assert summary["scoring_preflight"]["total_supported"] == 4891
    assert summary["resolved_token_budget_class_counts"] == {"longmemeval": 1111, "memora_email_writeup": 238, "memora_meeting_notes": 245, "memora_preference": 1400, "memora_project_proposal": 581, "memora_todos": 196, "oaks_novel": 1120}


def test_grok_budget_loader_is_strict(tmp_path):
    assert load_grok_v1_budget_config(CONFIG)["budgets"] == GROK_V1_BUDGETS
    altered = json.loads(CONFIG.read_text())
    altered["budgets"]["oaks_novel"] = 2048
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="approved measurement ceilings"):
        load_grok_v1_budget_config(bad)


def test_grok_request_has_none_reasoning_and_no_optional_tools():
    provider = GrokDownstreamProvider(client=object())
    payload = provider.request_payload("question", max_tokens=1024)
    assert payload["model"] == "grok-4.3"
    assert payload["reasoning_effort"] == "none"
    assert payload["max_tokens"] == 1024
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "tools" not in payload
    assert "functions" not in payload


def test_grok_request_parameters_uniform_across_conditions(tmp_path):
    result = run(manifest=MANIFEST, results_dir=tmp_path, dry_run=True)
    common = {(row["model_parameters"]["reasoning_effort"], row["model_parameters"]["temperature"], row["model_parameters"]["top_p"], row["model_parameters"]["tools"], row["model_parameters"]["base_url"]) for row in result["rows"]}
    assert common == {("none", None, None, None, DEFAULT_GROK_BASE_URL)}
    assert {row["model_parameters"]["max_output_tokens"] for row in result["rows"]} == set(GROK_V1_BUDGETS.values())


def test_grok_mocked_success_extracts_usage():
    response = SimpleNamespace(id="grok-1", choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"ok"}'), finish_reason="stop")], usage=SimpleNamespace(prompt_tokens=17, completion_tokens=4, total_tokens=21))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    result = GrokDownstreamProvider(client=client).request_downstream("p", max_tokens=1024)
    assert result.parsed == {"answer": "ok"}
    assert result.telemetry.input_tokens == 17
    assert result.telemetry.output_tokens == 4
    assert result.telemetry.total_tokens == 21


def _frozen_condition_id(*, source_dataset, condition_type):
    """Select a condition id from the local frozen manifest instead of embedding one.

    The manifest is a licensed local input; concrete condition ids are
    source-derived benchmark identifiers, so the public test source never
    hard-codes them. This test is skipped by the public release scope when the
    manifest is absent.
    """
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("source_dataset") == source_dataset and row.get("condition_type") == condition_type:
            return row["condition_id"]
    raise AssertionError(f"no {source_dataset}/{condition_type} condition in the frozen manifest")


def test_grok_condition_selection_and_base_url(tmp_path):
    cid = _frozen_condition_id(source_dataset="longmemeval", condition_type="clean")
    result = run(manifest=MANIFEST, results_dir=tmp_path, condition_ids={cid}, base_url="https://example.test/v1", dry_run=True)
    assert result["summary"]["requested_rows"] == 1
    assert result["summary"]["base_url"] == "https://example.test/v1"
    assert result["summary"]["api_calls_made"] == 0
