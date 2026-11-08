import hashlib
import json
from pathlib import Path

from scripts.run_final_main_smoke_deepseek import render_prompt


ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "a2601a7ecb46034faf5dcc7ad00b5a851e19ae01561876fa681fd18af168ca79"
CONDITION_SHA = "3aea17d92a2116400116a5e9082ddb2f2efa585cff550c40925eab9e5162af70"


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def test_v2_has_complete_task_payload_and_frozen_counts():
    rows = _rows(ROOT / "data/final_main_governance_condition_manifest_v2.jsonl")
    assert len(rows) == 4891
    assert sum(r["condition_type"] == "clean" for r in rows) == 731
    assert sum(r["condition_type"] == "corrupted" for r in rows) == 2080
    assert sum(r["condition_type"] == "enforced" for r in rows) == 2080
    assert all(isinstance(r.get("task_question"), str) and r["task_question"].strip() for r in rows)
    assert all(r.get("task_oracle_answer") not in (None, "", []) for r in rows)


def test_task_payload_is_invariant_across_siblings():
    rows = _rows(ROOT / "data/final_main_governance_condition_manifest_v2.jsonl")
    groups = {}
    for row in rows:
        groups.setdefault(row["base_id"], []).append(row)
    assert len(groups) == 731
    for siblings in groups.values():
        assert len({r["task_question"] for r in siblings}) == 1
        assert {json.dumps(r["task_oracle_answer"], sort_keys=True) for r in siblings}.__len__() == 1


def test_source_payloads_are_present_for_all_three_sources():
    rows = _rows(ROOT / "data/final_main_governance_condition_manifest_v2.jsonl")
    sources = {r["source_dataset"] for r in rows}
    assert sources == {"longmemeval", "memora", "oaks_novel"}
    for row in rows:
        assert row["task_source_metadata"]["source_dataset"] == row["source_dataset"]
        assert isinstance(row["task_evidence"], dict) and row["task_evidence"]
    assert any(r["task_answer_type"] == "structured_memory_evidence" for r in rows)
    assert any(r["task_answer_type"] == "multiple_choice" for r in rows)


def test_prompt_uses_question_without_experiment_labels():
    rows = _rows(ROOT / "data/final_main_governance_condition_manifest_v2.jsonl")
    sample = next(r for r in rows if r["condition_type"] == "clean")
    prompt = render_prompt(sample)
    assert sample["task_question"] in prompt
    assert "corrupted" not in prompt.lower()
    assert "fault state" not in prompt.lower()
    assert "enforcement state" not in prompt.lower()
    assert sample["source_dataset"] not in prompt


def test_parent_artifacts_and_hashes_are_unchanged():
    assert hashlib.sha256((ROOT / "data/final_main_governance_base_manifest_v2.jsonl").read_bytes()).hexdigest() == BASE_SHA
    assert hashlib.sha256((ROOT / "data/final_main_governance_condition_manifest_v1.jsonl").read_bytes()).hexdigest() == CONDITION_SHA


def test_smoke_v2_is_same_base_subset_and_complete():
    base_ids = {r["base_id"] for r in _rows(ROOT / "data/final_main_governance_smoke_base_manifest_v1.jsonl")}
    rows = _rows(ROOT / "data/final_main_governance_smoke_condition_manifest_v2.jsonl")
    assert len(rows) == 160
    assert {r["base_id"] for r in rows} == base_ids
    assert all(r.get("task_question") and r.get("task_oracle_answer") not in (None, "", []) for r in rows)


def test_memora_numeric_content_projection_is_preserved_and_fault_projected():
    rows = _rows(ROOT / "data/final_main_governance_condition_manifest_v2.jsonl")
    base = next(r["base_id"] for r in rows if r["source_dataset"] == "memora" and "budget: " in r["model_visible_context_text"])
    group = [r for r in rows if r["base_id"] == base]
    clean = next(r for r in group if r["condition_type"] == "clean")
    missing = next(r for r in group if r["condition_type"] == "corrupted" and r.get("fault_operator") == "missing")
    enforced = next(r for r in group if r["condition_type"] == "enforced" and r.get("fault_operator") == "missing")
    budget = next(line for line in clean["model_visible_context_text"].splitlines() if line.startswith("budget: "))
    assert budget in clean["model_visible_context_text"]
    assert budget not in missing["model_visible_context_text"]
    assert budget in enforced["model_visible_context_text"]
