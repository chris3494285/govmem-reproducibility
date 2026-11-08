from __future__ import annotations

from pathlib import Path
from scripts.run_final_main_gemini import run
from scripts.select_final_main_gemini_canary import select
from scripts.run_final_main_smoke_deepseek import _execute_condition
from scripts.run_final_main_smoke_deepseek import _load_cache, _model_parameters
from eval.reconstruction_provider import ProviderResponse, ProviderTelemetry
import json
import pytest
import hashlib


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
BUDGETS = ROOT / "configs/final_main_token_budget_v2.json"


def rows():
    return [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]


def test_gemini_full_dry_run_is_offline_and_authoritative(tmp_path):
    calls = []
    result = run(manifest=MANIFEST, results_dir=tmp_path, model="gemini-3-flash-preview", workers=4, dry_run=True, token_budget_config=BUDGETS, provider_factory=lambda *a, **k: calls.append(1))
    summary = result["summary"]
    assert summary["requested_rows"] == 4891
    assert summary["api_calls_made"] == 0
    assert summary["manifest_sha256"] == "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"
    assert summary["scoring_preflight"]["total_supported"] == 4891
    assert summary["resolved_token_budget_class_counts"] == {"longmemeval": 1111, "memora_email_writeup": 238, "memora_meeting_notes": 245, "memora_preference": 1400, "memora_project_proposal": 581, "memora_todos": 196, "oaks_novel": 1120}
    assert summary["model_parameters"]["thinking"] == "low"
    assert summary["model_parameters"]["thinking_level"] == "LOW"
    assert summary["model_parameters"]["thinking"] != "disabled"
    assert calls == []


def test_gemini_mocked_live_result_records_low_thinking_metadata():
    row = next(item for item in rows() if item["source_dataset"] == "longmemeval")

    class Provider:
        def request_downstream(self, prompt, *, max_tokens=None):
            return ProviderResponse({"answer": str(row["task_oracle_answer"])}, ProviderTelemetry("gemini", "gemini-3-flash-preview", 1, 1, 2, 0.0, "r"))

    result = _execute_condition(
        row, "prompt", Provider(), model="gemini-3-flash-preview", manifest_sha="sha",
        model_parameters={"thinking": "low", "thinking_level": "LOW", "tools": None},
    )
    assert result["model_parameters"]["thinking"] == "low"
    assert result["model_parameters"]["thinking_level"] == "LOW"
    assert result["model_parameters"]["thinking"] != "disabled"


def test_gemini_canary_covers_all_classes_and_risk_targets():
    chosen = select(rows())
    assert 7 <= len(chosen) <= 10
    assert {item["row"].get("source_dataset") for item in chosen} >= {"longmemeval", "memora", "oaks_novel"}
    assert {__import__("eval.final_main_token_budget", fromlist=["budget_class"]).budget_class(item["row"]) for item in chosen} == {"longmemeval", "memora_email_writeup", "memora_meeting_notes", "memora_preference", "memora_project_proposal", "memora_todos", "oaks_novel"}
    assert sum("longest-risk" in item["reason"] for item in chosen) >= 1
    assert len({item["row"]["condition_id"] for item in chosen}) == len(chosen)


def test_gemini_flash_cache_identity_uses_caller_parameters_and_row_budget(tmp_path):
    cid = "longmemeval::cache-test::clean"
    prompt_hash = hashlib.sha256(b"prompt").hexdigest()
    cached = {
        "condition_id": cid,
        "manifest_sha256": "manifest",
        "model": "gemini-3-flash-preview",
        "prompt_sha256": prompt_hash,
        "model_parameters": _model_parameters(1536, {"thinking": "low", "thinking_level": "LOW", "tools": None}),
        "status": "success",
    }
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(cached) + "\n")

    loaded = _load_cache(
        path,
        model="gemini-3-flash-preview",
        manifest_sha="manifest",
        prompt_hashes={cid: prompt_hash},
        token_budgets={cid: 1536},
        model_parameters={"thinking": "low", "thinking_level": "LOW", "tools": None},
    )
    assert loaded[cid]["model_parameters"]["max_output_tokens"] == 1536


@pytest.mark.parametrize("override", [
    {"thinking": "disabled", "thinking_level": "LOW", "tools": None},
    {"thinking": "low", "thinking_level": "MEDIUM", "tools": None},
    {"thinking": "low", "thinking_level": "LOW", "tools": None, "max_output_tokens": 768},
])
def test_gemini_cache_rejects_mismatched_model_settings(tmp_path, override):
    cid = "longmemeval::cache-test::clean"
    prompt_hash = hashlib.sha256(b"prompt").hexdigest()
    expected = {"thinking": "low", "thinking_level": "LOW", "tools": None}
    cached = {
        "condition_id": cid,
        "manifest_sha256": "manifest",
        "model": "gemini-3-flash-preview",
        "prompt_sha256": prompt_hash,
        "model_parameters": _model_parameters(1536, override),
    }
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(cached) + "\n")
    with pytest.raises(ValueError, match="model-settings identity"):
        _load_cache(path, model="gemini-3-flash-preview", manifest_sha="manifest", prompt_hashes={cid: prompt_hash}, token_budgets={cid: 1536}, model_parameters=expected)
