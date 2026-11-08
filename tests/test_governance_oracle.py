from eval.gatemem_adapter import GateMemDatasetAdapter
from eval.memconflict_adapter import MemConflictDatasetAdapter
from eval.memora_adapter import MemoraDatasetAdapter
from govmem.governance.oracle import derive_base_example
from govmem.governance.faults import inject_fault


def test_gatemem_authorization_uses_explicit_scope_membership_only():
    example = GateMemDatasetAdapter().load("tests/fixtures/gatemem-mini")[1]
    result = derive_base_example(example)
    assert result.supported is True
    assert result.example.gold_governance_state.authorization.decision == "deny"
    assert result.example.gold_governance_state.authorization.scope_id is None
    assert result.example.source_metadata["leak_targets"] == ("Friday",)


def test_memora_remembering_has_deterministic_deletion_evidence():
    adapter = MemoraDatasetAdapter()
    example = adapter.adapt_question(
        {"question_id": "q", "question": "What happened to the note?", "question_date": "2025-06-03", "task_type": "remembering", "memory_evidence": {}, "forgetting_evidence": {}},
        {"persona": "p"},
        [
            {"session_id": 1, "date": "2025-06-01", "session_type": "activity", "operation": "add", "operation_details": {"item": "note-a"}, "conversation": []},
            {"session_id": 2, "date": "2025-06-02", "session_type": "activity", "operation": "delete", "operation_details": {"item": "note-a"}, "conversation": []},
        ],
    )
    # The native evidence identifies the concrete lifecycle item; operations
    # provide the deterministic add/delete event pair.
    example = example.__class__(
        query=example.query,
        interactions=example.interactions,
        oracle_governance=example.oracle_governance,
        metadata=example.metadata,
        evaluator_metadata={**example.evaluator_metadata, "gold": {"memory_evidence": {}, "forgetting_evidence": {"forgotten_items": [{"value": "note-a"}]}}},
    )
    result = derive_base_example(example)
    assert result.supported is True
    assert result.example.dimension == "deletion"
    assert result.example.gold_governance_state.deletion.status == "deleted"
    assert result.example.source_metadata["historical_deletion_states"][0].status == "active"


def test_memconflict_static_and_conditional_are_not_forced_into_temporal_gold():
    examples = MemConflictDatasetAdapter().load("tests/fixtures/memconflict-mini/memconflict.jsonl")
    assert all(not derive_base_example(example).supported for example in examples if example.query.metadata["conflict_type"] != "dynamic")
    assert derive_base_example(examples[0]).supported is False


def test_memconflict_updated_attributes_maps_exact_transition(tmp_path):
    import json
    path = tmp_path / "official.jsonl"
    row = {
        "ID": "p1", "Full_Session_Chain": [
            {"Session_ID": 1, "Date": "2024-01-01", "Session_Dialogue": {"d1": [{"role": "user", "content": "I live in A."}]}},
            {"Session_ID": 2, "Date": "2024-06-01", "Updated_Attributes": [{"Attribute": "Residence", "Before": "A", "After": "B"}], "Session_Questions": [{"question_id": "Q", "question": "Did my residence change?", "answer": "Yes", "conflict_type": "dynamic_conflict"}], "Session_Dialogue": {"d1": [{"role": "user", "content": "I moved to B."}]}}
        ]
    }
    path.write_text(json.dumps(row) + "\n")
    result = derive_base_example(MemConflictDatasetAdapter().load(path)[0])
    assert result.supported is True
    state = result.example.gold_governance_state.temporal
    assert result.example.source_metadata["attribute"] == "Residence"
    assert result.example.source_metadata["previous_value"] == "A"
    assert state.status == "valid"
    assert result.example.source_metadata["historical_temporal_states"][0].status == "expired"
    stale = inject_fault(result.example.gold_governance_state, "temporal", "stale", result.example.source_metadata["historical_temporal_states"])
    assert stale.temporal == result.example.source_metadata["historical_temporal_states"][0]
    conflict = inject_fault(result.example.gold_governance_state, "temporal", "conflicting", result.example.source_metadata["historical_temporal_states"])
    assert {item.status for item in conflict.temporal_assertions} == {"valid", "expired"}


def test_memconflict_ambiguous_attribute_linkage_is_unsupported(tmp_path):
    import json
    path = tmp_path / "ambiguous.jsonl"
    row = {"ID": "p1", "Full_Session_Chain": [{"Session_ID": 1, "Date": "2024-01-01", "Session_Dialogue": {}, "Updated_Attributes": [{"Attribute": "Residence", "Before": "A", "After": "B"}, {"Attribute": "Residence", "Before": "B", "After": "C"}], "Session_Questions": [{"question_id": "Q", "question": "Did my residence change?", "conflict_type": "dynamic_conflict"}]}]}
    path.write_text(json.dumps(row) + "\n")
    assert derive_base_example(MemConflictDatasetAdapter().load(path)[0]).supported is False


def test_memora_active_item_is_compact_and_traceable():
    adapter = MemoraDatasetAdapter()
    example = adapter.adapt_question(
        {"question_id": "q", "question": "What is active?", "question_date": "2025-06-03", "task_type": "remembering", "memory_evidence": {"active_memory": {"value": "note-a", "session_id": 1}}, "forgetting_evidence": {}},
        {"persona": "p"},
        [{"session_id": 1, "date": "2025-06-01", "operation": "add", "operation_details": {"item": "note-a"}, "conversation": []}],
    )
    result = derive_base_example(example)
    assert result.supported is True
    assert result.example.gold_governance_state.deletion.status == "active"
    assert len(result.example.memory) == 1
    assert "lifecycle" in result.example.memory[0]
    assert len(str(result.example.memory)) < 1000
