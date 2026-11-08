import hashlib
import json
from pathlib import Path

import pytest

from eval.system_validation_scoring import max_output_tokens_for_row
from scripts.run_final_main_deepseek import EXPECTED_COUNTS, EXPECTED_MANIFEST_SHA, run
from scripts.run_final_main_smoke_deepseek import render_prompt, validate_manifest


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
SMOKE = ROOT / "data/final_main_governance_smoke_condition_manifest_v2.jsonl"


def test_full_manifest_is_frozen_and_complete():
    assert hashlib.sha256(FULL.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA
    rows = [json.loads(line) for line in FULL.read_text().splitlines()]
    assert len(rows) == 4891
    assert {kind: sum(row["condition_type"] == kind for row in rows) for kind in EXPECTED_COUNTS} == EXPECTED_COUNTS


def test_full_runner_rejects_wrong_and_smoke_manifests(tmp_path):
    wrong = tmp_path / "wrong.jsonl"
    wrong.write_text("{}\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run(manifest=wrong, results_dir=tmp_path / "wrong-results", dry_run=True)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run(manifest=SMOKE, results_dir=tmp_path / "smoke-results", dry_run=True)


def test_smoke_validator_rejects_full_manifest():
    rows = [json.loads(line) for line in FULL.read_text().splitlines()]
    digest = hashlib.sha256(FULL.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_manifest(rows, digest)


def test_full_dry_run_covers_all_rows_without_provider(tmp_path):
    result = run(manifest=FULL, results_dir=tmp_path, dry_run=True, provider_factory=lambda: (_ for _ in ()).throw(AssertionError("provider called")))
    assert result["summary"]["total_rows"] == 4891
    assert result["summary"]["api_calls_made"] == 0
    assert result["summary"]["scoring_preflight"]["total_supported"] == 4891
    assert result["summary"]["scoring_preflight"]["unresolved"] == []


def test_full_runner_uses_shared_prompt_and_budget_resolution():
    row = {"task_question": "Q", "model_visible_context_text": "C", "task_answer_type": "text", "source_dataset": "longmemeval"}
    assert render_prompt(row) == render_prompt(dict(row))
    assert max_output_tokens_for_row({"source_dataset": "longmemeval"}) == 256
    assert max_output_tokens_for_row({"source_dataset": "oaks_novel"}) == 128
