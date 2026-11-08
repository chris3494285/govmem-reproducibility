import json
from pathlib import Path

import pytest

from eval.final_main_result_join import EXPECTED_MANIFEST_SHA256, join_results_to_manifest, load_condition_manifest
from eval.system_validation_scoring import score_task_answer


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"


def test_full_manifest_has_unique_condition_ids():
    rows, digest = load_condition_manifest(MANIFEST)
    assert len(rows) == 4891
    assert digest == EXPECTED_MANIFEST_SHA256
    assert len({row["condition_id"] for row in rows}) == 4891


def test_valid_result_joins_authoritative_manifest_metadata_without_mutation():
    manifest, _ = load_condition_manifest(MANIFEST)
    target = next(row for row in manifest if row["source_dataset"] == "memora")
    result = {"condition_id": target["condition_id"], "status": "error", "parsed_answer": None}
    original = json.loads(json.dumps(result))
    joined = join_results_to_manifest([result], manifest)[0]
    assert result == original
    assert joined["source_dataset"] == target["source_dataset"]
    assert joined["analysis_event_type"] == target["analysis_event_type"]
    assert joined["task_scoring_spec"].get("subtype")


def test_join_rejects_missing_and_duplicate_ids():
    manifest = [{"condition_id": "c1", "source_dataset": "longmemeval"}]
    with pytest.raises(ValueError, match="no frozen manifest match"):
        join_results_to_manifest([{"condition_id": "missing"}], manifest)
    with pytest.raises(ValueError, match="duplicate condition_id"):
        join_results_to_manifest([{"condition_id": "c1"}], manifest + manifest)


def test_oaks_document_id_and_failure_result_are_preserved():
    manifest, _ = load_condition_manifest(MANIFEST)
    target = next(row for row in manifest if row["source_dataset"] == "oaks_novel")
    result = {"condition_id": target["condition_id"], "status": "error", "parse_success": False}
    joined = join_results_to_manifest([result], manifest)[0]
    assert joined["task_source_metadata"]["document_id"]
    assert joined["result"]["parse_success"] is False


def test_successful_scoring_semantics_remain_unchanged_after_join():
    manifest, _ = load_condition_manifest(MANIFEST)
    target = next(row for row in manifest if row["source_dataset"] == "longmemeval")
    joined = join_results_to_manifest([{"condition_id": target["condition_id"]}], manifest)[0]
    assert score_task_answer(joined["manifest"], target["task_oracle_answer"])[0] is True
