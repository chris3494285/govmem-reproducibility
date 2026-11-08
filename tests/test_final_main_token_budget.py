from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.final_main_token_budget import FROZEN_BUDGETS, GEMINI_BUDGETS, GEMINI_FLASH_BUDGETS, GEMINI_FLASH_REPAIR_BUDGETS, GEMINI_FLASH_REPAIR_V2_BUDGETS, budget_class, load_budget_config, load_gemini_budget_config, load_gemini_flash_budget_config, load_gemini_flash_repair_budget_config, load_gemini_flash_repair_v2_budget_config, load_repair_budget_config, memora_task_subtype, project_proposal_condition_ids, resolve_max_output_tokens
from scripts.run_final_main_deepseek import run as run_deepseek
from scripts.run_final_main_openai import run as run_openai


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
CONFIG = ROOT / "configs/final_main_token_budget_v2.json"


def _rows():
    return [json.loads(line) for line in FULL.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_proposal_slice_is_derived_from_authoritative_source_ids():
    rows = _rows()
    ids = project_proposal_condition_ids(rows)
    selected = [row for row in rows if row["condition_id"] in ids]
    assert len(selected) == 581
    assert {budget_class(row) for row in selected} == {"memora_project_proposal"}
    assert {row["condition_type"] for row in selected} == {"clean", "corrupted", "enforced"}


def test_authoritative_memora_subtype_resolver_distinguishes_source_question_families():
    rows = _rows()
    observed = {memora_task_subtype(row) for row in rows if row.get("source_dataset") == "memora"}
    assert observed == {
        "content_project_proposal",
        "content_meeting_notes",
        "content_email_writeup",
        "preference_recommendation",
        "activity_todos",
    }
    assert len(project_proposal_condition_ids(rows)) == 581
    config = load_budget_config(CONFIG)
    meeting = next(row for row in rows if memora_task_subtype(row) == "content_meeting_notes")
    proposal = next(row for row in rows if memora_task_subtype(row) == "content_project_proposal")
    assert resolve_max_output_tokens(meeting, config) == 256
    assert resolve_max_output_tokens(proposal, config) == 1536


def test_versioned_config_changes_only_proposals():
    config = load_budget_config(CONFIG)
    for key, value in FROZEN_BUDGETS.items():
        expected = 1536 if key == "memora_project_proposal" else value
        assert config["budgets"][key] == expected


def test_frozen_loader_rejects_anthropic_repair_config():
    with pytest.raises(ValueError, match="Unsupported final-main token-budget configuration"):
        load_budget_config(ROOT / "configs/final_main_token_budget_anthropic_repair_v2.json")


def test_gemini_measurement_loader_accepts_only_explicit_gemini_ceilings():
    config = load_gemini_budget_config(ROOT / "configs/final_main_token_budget_gemini_v1.json")
    assert config["provider"] == "gemini"
    assert config["budgets"] == GEMINI_BUDGETS
    with pytest.raises(ValueError):
        load_budget_config(ROOT / "configs/final_main_token_budget_gemini_v1.json")


def test_gemini_flash_measurement_loader_is_model_specific():
    config = load_gemini_flash_budget_config(ROOT / "configs/final_main_token_budget_gemini_flash_v1.json")
    assert config["model"] == "gemini-3-flash-preview"
    assert config["budgets"] == GEMINI_FLASH_BUDGETS
    assert config["budgets"]["longmemeval"] == 1536
    assert config["budgets"]["memora_email_writeup"] == 3072
    assert config["budgets"]["memora_preference"] == 1536
    assert config["budgets"]["oaks_novel"] == 1024
    assert config["budgets"]["memora_todos"] == 3072
    with pytest.raises(ValueError):
        load_budget_config(ROOT / "configs/final_main_token_budget_gemini_flash_v1.json")


def test_gemini_flash_loader_rejects_unauthorized_ceiling_change(tmp_path):
    source = json.loads((ROOT / "configs/final_main_token_budget_gemini_flash_v1.json").read_text())
    source["budgets"]["oaks_novel"] = 2048
    path = tmp_path / "unauthorized_flash.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="approved measurement ceilings"):
        load_gemini_flash_budget_config(path)


def test_gemini_flash_repair_loader_accepts_only_repair_ceilings():
    config = load_gemini_flash_repair_budget_config(ROOT / "configs/final_main_token_budget_gemini_flash_repair_v1.json")
    assert config["budgets"] == GEMINI_FLASH_REPAIR_BUDGETS
    assert config["budgets"]["memora_todos"] == 6144
    with pytest.raises(ValueError):
        load_gemini_flash_budget_config(ROOT / "configs/final_main_token_budget_gemini_flash_repair_v1.json")


def test_gemini_flash_repair_loader_rejects_unauthorized_ceiling_change(tmp_path):
    config = json.loads((ROOT / "configs/final_main_token_budget_gemini_flash_repair_v1.json").read_text())
    config["budgets"]["oaks_novel"] = 4096
    path = tmp_path / "bad_repair.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="approved repair ceilings"):
        load_gemini_flash_repair_budget_config(path)


def test_gemini_flash_repair_v2_loader_accepts_only_v2_ceilings(tmp_path):
    config = load_gemini_flash_repair_v2_budget_config(ROOT / "configs/final_main_token_budget_gemini_flash_repair_v2.json")
    assert config["budgets"] == GEMINI_FLASH_REPAIR_V2_BUDGETS
    assert config["budgets"]["memora_meeting_notes"] == 6144
    with pytest.raises(ValueError):
        load_gemini_flash_repair_budget_config(ROOT / "configs/final_main_token_budget_gemini_flash_repair_v2.json")
    altered = json.loads((ROOT / "configs/final_main_token_budget_gemini_flash_repair_v2.json").read_text())
    altered["budgets"]["memora_meeting_notes"] = 3072
    path = tmp_path / "bad_repair_v2.json"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="approved repair-v2 ceilings"):
        load_gemini_flash_repair_v2_budget_config(path)


def test_global_frozen_budget_does_not_change_with_flash_ceiling():
    assert FROZEN_BUDGETS["oaks_novel"] == 128
    assert FROZEN_BUDGETS["longmemeval"] == 256


def test_anthropic_repair_loader_accepts_only_the_explicit_repair_contract():
    config = load_repair_budget_config(ROOT / "configs/final_main_token_budget_anthropic_repair_v2.json")
    assert config["budgets"]["memora_meeting_notes"] == 1024
    assert config["budgets"]["memora_preference"] == 768


def test_deepseek_and_openai_resolve_same_versioned_proposal_budget():
    row = next(row for row in _rows() if budget_class(row) == "memora_project_proposal")
    config = load_budget_config(CONFIG)
    assert resolve_max_output_tokens(row, config) == 1536


def test_both_full_wrappers_proposal_dry_run_make_no_provider_calls(tmp_path):
    rows = _rows()
    for runner, name in ((run_deepseek, "deepseek"), (run_openai, "openai")):
        calls = []
        kwargs = dict(
            manifest=FULL,
            results_dir=tmp_path / name,
            dry_run=True,
            proposal_only=True,
            budget_config_path=CONFIG,
            workers=2,
            provider_factory=lambda *args, **kw: calls.append(1),
        )
        result = runner(**kwargs)
        assert result["summary"]["total_rows"] == 581
        assert result["summary"]["estimated_total_max_output_tokens"] == 581 * 1536
        assert calls == []
