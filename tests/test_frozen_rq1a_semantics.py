from __future__ import annotations

from datetime import datetime

from eval.interfaces import AdaptedExample, Query
from scripts.audit_frozen_rq1a_semantics import audit_manifest_rows


def _pair(clean: dict) -> list[dict]:
    fault = dict(clean)
    if clean["dimension"] == "temporal":
        fault_operator = "stale"
        fault_direction = "unsafe"
        fault_observed = {"temporal": {"status": "expired"}}
    else:
        fault_operator = "flipped"
        fault_direction = "restrictive"
        fault_observed = {"deletion": {"status": "deleted"}}
    fault.update({
        "run_id": clean["example_id"] + "::fault",
        "condition_type": "fault",
        "fault_spec": {"dimension": clean["dimension"], "operator": fault_operator, "direction": fault_direction},
        "observed_governance_state": fault_observed,
    })
    stale = dict(clean)
    stale.update({
        "run_id": clean["example_id"] + "::fault2",
        "condition_type": "fault",
        "fault_spec": {"dimension": clean["dimension"], "operator": "missing", "direction": "ambiguous"},
        "observed_governance_state": {"deletion": None} if clean["dimension"] == "deletion" else {"temporal": None},
    })
    return [clean, fault, stale]


def _memora_source(query_id: str, *, delete_date: str | None = "2024-01-02") -> AdaptedExample:
    operations = [
        {"session_id": 1, "operation": "add", "operation_details": {"item": "item-a"}, "date": "2024-01-01"},
    ]
    if delete_date:
        operations.append({"session_id": 2, "operation": "delete", "operation_details": {"item": "item-a"}, "date": delete_date})
    return AdaptedExample(
        query=Query(query_id, "u", "What is item-a?", datetime.fromisoformat("2024-01-03"), metadata={"dataset": "memora", "task_type": "remembering"}),
        interactions=(), oracle_governance=(), metadata={"dataset": "memora"},
        evaluator_metadata={"session_operations": operations, "gold": {}},
    )


def _memora_row(query_id: str, status: str = "deleted") -> dict:
    return {
        "run_id": f"memora::{query_id}::clean", "example_id": f"memora::{query_id}::deletion::item0",
        "source_dataset": "memora", "dimension": "deletion", "condition_type": "clean", "fault_spec": None,
        "task": "What is item-a?", "memory": [{"item": "item-a", "lifecycle": []}],
        "gold_governance_state": {"deletion": {"status": status, "epoch": 2}},
        "observed_governance_state": {"deletion": {"status": status, "epoch": 2}},
        "gold_admission_decision": "deny" if status == "deleted" else "allow",
        "utility_target": None, "source_metadata": {"query_id": query_id, "item": "item-a"},
    }


def test_deleted_memora_row_remains_valid_despite_candidate_pool_drift():
    row = _memora_row("q-deleted")
    report = audit_manifest_rows(_pair(row), {("memora", "q-deleted"): _memora_source("q-deleted")})
    assert report["invalid_base_examples"] == 0


def test_memora_delete_after_evaluation_boundary_is_invalid():
    row = _memora_row("q-late-delete")
    source = _memora_source("q-late-delete", delete_date="2024-01-04")
    report = audit_manifest_rows(_pair(row), {("memora", "q-late-delete"): source})
    assert report["invalid_base_examples"] == 1
    assert "no_unique_visible_matching_native_delete" in report["invalid_by_reason"]


def _conflict_source(query_id: str) -> AdaptedExample:
    update = {"session_id": 2, "date": "2024-02-01", "attributes": [{"Attribute": "Residence", "Before": "Darwin, Australia", "After": "Melbourne, Australia"}]}
    return AdaptedExample(
        query=Query(query_id, "u", "Did the user's residence change recently?", datetime.fromisoformat("2024-02-01"), metadata={"dataset": "memconflict", "conflict_type": "dynamic"}),
        interactions=(), oracle_governance=(), metadata={"dataset": "memconflict"},
        evaluator_metadata={"gold": {"updated_attributes": [update], "session_history": [{"session_id": 1, "date": "2024-01-01"}, update]}},
    )


def _conflict_row(query_id: str, attribute: str = "Residence") -> dict:
    metadata = {"query_id": query_id, "attribute": attribute, "current_value": "Melbourne, Australia", "previous_value": "Darwin, Australia", "current_session": {"session_id": 2}, "historical_temporal_states": [{"status": "expired"}]}
    return {
        "run_id": f"memconflict::{query_id}::clean", "example_id": f"memconflict::{query_id}::temporal", "source_dataset": "memconflict", "dimension": "temporal", "condition_type": "clean", "fault_spec": None,
        "task": "Did the user's residence change recently?", "memory": [{"attribute": attribute, "current_value": "Melbourne, Australia", "previous_value": "Darwin, Australia"}],
        "gold_governance_state": {"temporal": {"status": "valid", "version": 2}}, "observed_governance_state": {"temporal": {"status": "valid", "version": 2}}, "gold_admission_decision": "allow", "utility_target": None, "source_metadata": metadata,
    }


def test_valid_single_attribute_memconflict_row_passes():
    row = _conflict_row("q-residence")
    report = audit_manifest_rows(_pair(row), {("memconflict", "q-residence"): _conflict_source("q-residence")})
    assert report["invalid_base_examples"] == 0


def test_old_multi_attribute_misbinding_is_invalid():
    row = _conflict_row("q-old-bug", attribute="Career_Status")
    report = audit_manifest_rows(_pair(row), {("memconflict", "q-old-bug"): _conflict_source("q-old-bug")})
    assert report["invalid_base_examples"] == 1
    assert "stored_attribute_mismatches_relevant_updated_attribute" in report["invalid_by_reason"]
