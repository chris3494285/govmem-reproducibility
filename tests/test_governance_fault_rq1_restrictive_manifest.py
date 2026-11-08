from __future__ import annotations

import copy

import pytest

from govmem.governance.enforcement import decide_admission
from scripts.build_governance_fault_rq1_restrictive_manifest import (
    MANIFEST_VERSION,
    build_from_candidates,
    validate_restrictive_manifest,
)


def _candidate(dataset: str, index: int) -> dict:
    if dataset == "memora":
        return {
            "candidate_id": f"memora::{index}",
            "candidate_cluster_id": f"query::{index}",
            "source_dataset": "memora",
            "dimension": "deletion",
            "task": f"What is item {index}?",
            "memory": [{"item": f"active item {index}", "lifecycle": [{"operation": "add", "session_id": 1}]}],
            "gold_governance_state": {"deletion": {"status": "active", "epoch": index}},
            "gold_admission_decision": "allow",
            "utility_target": None,
            "source_metadata": {"item": f"active item {index}", "lifecycle": [{"operation": "add", "session_id": 1}]},
        }
    return {
        "candidate_id": f"conflict::{index}",
        "transition_id": f"transition::{index}",
        "source_dataset": "memconflict",
        "dimension": "temporal",
        "attribute": "Residence" if index % 2 else "Social_Status",
        "current_value": "Melbourne, Australia",
        "previous_value": "Darwin, Australia",
        "task": f"What changed in case {index}?",
        "memory": [{"attribute": "Residence", "current_value": "Melbourne, Australia", "previous_value": "Darwin, Australia"}],
        "gold_governance_state": {"temporal": {"status": "valid", "version": index}},
        "gold_admission_decision": "allow",
        "utility_target": None,
        "source_metadata": {"attribute": "Residence", "current_value": "Melbourne, Australia", "previous_value": "Darwin, Australia", "current_session": {"session_id": index}},
    }


def test_restrictive_manifest_has_30_bases_and_60_rows():
    rows = build_from_candidates([_candidate("memora", i) for i in range(15)] + [_candidate("memconflict", i) for i in range(15)])
    assert len(rows) == 60
    assert len({row["example_id"] for row in rows}) == 30
    assert {row["source_dataset"] for row in rows} == {"memora", "memconflict"}
    assert sum(row["source_dataset"] == "memora" and row["condition_type"] == "clean" for row in rows) == 15
    assert sum(row["source_dataset"] == "memconflict" and row["condition_type"] == "clean" for row in rows) == 15
    assert all(row["manifest_version"] == MANIFEST_VERSION for row in rows)
    assert all(row["gold_admission_decision"] == "allow" for row in rows)


def test_pairs_only_change_observed_state_and_fault_spec():
    rows = build_from_candidates([_candidate("memora", i) for i in range(15)] + [_candidate("memconflict", i) for i in range(15)])
    for example_id in {row["example_id"] for row in rows}:
        pair = [row for row in rows if row["example_id"] == example_id]
        clean, fault = sorted(pair, key=lambda row: row["condition_type"] == "fault")
        for key in ("task", "memory", "gold_governance_state", "gold_admission_decision", "source_metadata"):
            assert clean[key] == fault[key]
        assert clean["fault_spec"] is None
        assert fault["fault_spec"]["direction"] == "restrictive"


def test_restrictive_observed_states_deny():
    rows = build_from_candidates([_candidate("memora", i) for i in range(15)] + [_candidate("memconflict", i) for i in range(15)])
    for row in rows:
        if row["condition_type"] == "fault":
            assert decide_admission(row).actual_admission == "deny"
        else:
            assert decide_admission(row).actual_admission == "allow"


def test_unique_clusters_and_invalid_duplicate_pair_rejected():
    rows = build_from_candidates([_candidate("memora", i) for i in range(15)] + [_candidate("memconflict", i) for i in range(15)])
    assert len({row["candidate_cluster_id"] for row in rows if row["source_dataset"] == "memora"}) == 15
    assert len({row["transition_id"] for row in rows if row["source_dataset"] == "memconflict"}) == 15
    with pytest.raises(ValueError):
        validate_restrictive_manifest(rows + [copy.deepcopy(rows[0])])
