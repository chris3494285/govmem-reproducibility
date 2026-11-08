from __future__ import annotations

import json

from eval.gatemem_adapter import GateMemDatasetAdapter
from eval.memconflict_adapter import MemConflictDatasetAdapter
from eval.memora_adapter import MemoraDatasetAdapter
from scripts.audit_rq1_restrictive_candidates import audit_examples


def _memora_example(*, deleted: bool = False):
    operations = [
        {
            "session_id": 1,
            "operation": "add",
            "operation_details": {"item": "Prepare lecture materials"},
            "date": "2024-01-01T00:00:00",
        }
    ]
    forgetting = []
    if deleted:
        operations.append(
            {
                "session_id": 2,
                "operation": "delete",
                "operation_details": {"item": "Prepare lecture materials"},
                "date": "2024-01-02T00:00:00",
            }
        )
        forgetting = [{"value": "Prepare lecture materials"}]
    question = {
        "question_id": "q1",
        "question": "What should I do?",
        "question_date": "2024-01-03T00:00:00",
        "task_type": "remembering",
        "memory_evidence": {"active_memory": {"value": "Prepare lecture materials"}} if not deleted else {},
        "forgetting_evidence": {"forgotten_items": forgetting},
    }
    question_data = {"persona": "p", "questions": {"remembering": [question]}}
    sessions = [
        {
            "session_id": item["session_id"],
            "date": item["date"],
            "operation": item["operation"],
            "operation_details": item["operation_details"],
            "conversation": [],
        }
        for item in operations
    ]
    return MemoraDatasetAdapter().adapt_question(question, question_data, sessions)


def _memconflict_example():
    row = {
        "ID": "case-1",
        "Full_Session_Chain": [
            {"Session_ID": "s1", "Date": "2024-01-01", "Session_Dialogue": {"turn_1": [{"role": "user", "content": "I work in Darwin."}]}},
            {
                "Session_ID": "s2",
                "Date": "2024-02-01",
                "Updated_Attributes": [{"Attribute": "Work status", "Before": "On_Business_Trip", "After": "Free"}],
                "Session_Dialogue": {"turn_1": [{"role": "user", "content": "My work status is Free."}]},
                "Session_Questions": [{"question_id": "Q_001", "question": "What is the user's work status?", "conflict_type": "dynamic_conflict"}],
            },
        ],
    }
    return list(MemConflictDatasetAdapter()._official_questions(row))


def _multi_attribute_memconflict_examples():
    row = {
        "ID": "case-multi",
        "Full_Session_Chain": [
            {"Session_ID": "s1", "Date": "2024-01-01", "Session_Dialogue": {"turn_1": [{"role": "user", "content": "Initial state."}]}},
            {
                "Session_ID": "s2",
                "Date": "2024-02-01",
                "Updated_Attributes": [
                    {"Attribute": "Residence", "Before": "Darwin, Australia", "After": "Melbourne, Australia"},
                    {"Attribute": "Social_Status", "Before": {"Current_State": "Working_Alone"}, "After": {"Current_State": "Joining_Social_Activities"}},
                ],
                "Session_Dialogue": {"turn_1": [{"role": "user", "content": "Several attributes changed."}]},
                "Session_Questions": [
                    {"question_id": "Q_RES", "question": "Did the user's residence change recently?", "conflict_type": "dynamic_conflict"},
                    {"question_id": "Q_RES_2", "question": "Did the user's residence change recently?", "conflict_type": "dynamic_conflict"},
                    {"question_id": "Q_SOC", "question": "Did the user's social status change recently?", "conflict_type": "dynamic_conflict"},
                    {"question_id": "Q_NONE", "question": "Did the user's favorite color change recently?", "conflict_type": "dynamic_conflict"},
                ],
            },
        ],
    }
    return list(MemConflictDatasetAdapter()._official_questions(row))


def test_active_memora_item_is_restrictive_candidate_and_fault_is_observed_only():
    report = audit_examples({"memora": [_memora_example()], "gatemem": [], "memconflict": []})
    assert report["restrictive_candidates_total"] == 1
    candidate = report["candidates"][0]
    assert candidate["gold_deletion_status"] == "active"
    assert candidate["gold_admission_decision"] == "allow"
    assert candidate["observed_governance_state"]["deletion"]["status"] == "deleted"
    assert candidate["memory"][0]["item"] == "Prepare lecture materials"
    assert len(candidate["memory"]) == 1


def test_deleted_memora_item_is_not_restrictive_candidate():
    report = audit_examples({"memora": [_memora_example(deleted=True)], "gatemem": [], "memconflict": []})
    assert report["restrictive_candidates_total"] == 0
    assert report["unsupported_by_reason"]["memora_item_not_active_at_evaluation"] == 1


def test_memconflict_valid_current_assertion_can_be_marked_expired_without_changing_memory():
    examples = _memconflict_example()
    assert len(examples) == 1
    report = audit_examples({"memora": [], "gatemem": [], "memconflict": examples})
    assert report["restrictive_candidates_total"] == 1
    candidate = report["candidates"][0]
    assert candidate["gold_temporal_status"] == "valid"
    assert candidate["observed_governance_state"]["temporal"]["status"] == "expired"
    assert candidate["current_value"] == "Free"
    assert candidate["memory"][0]["current_value"] == "Free"
    assert candidate["memory"][0]["previous_value"] == "On_Business_Trip"


def test_memconflict_multi_attribute_linkage_keeps_attribute_and_values_aligned():
    examples = _multi_attribute_memconflict_examples()
    report = audit_examples({"memora": [], "gatemem": [], "memconflict": examples})
    assert report["restrictive_candidates_raw"] == 3
    # The transition identity includes the target attribute, so one
    # multi-attribute session yields one identity per independently targeted
    # transition.
    assert report["memconflict_unique_transition_clusters"] == 2
    candidates = {item["task_query_id"]: item for item in report["candidates"]}

    assert candidates["case-multi::ss2::Q_RES"]["transition_id"] == candidates["case-multi::ss2::Q_RES_2"]["transition_id"]

    residence = candidates["case-multi::ss2::Q_RES"]
    assert residence["attribute"] == "Residence"
    assert residence["current_value"] == "Melbourne, Australia"
    assert residence["previous_value"] == "Darwin, Australia"
    assert residence["memory"][0]["attribute"] == residence["attribute"]
    assert residence["memory"][0]["current_value"] == residence["current_value"]
    assert residence["memory"][0]["previous_value"] == residence["previous_value"]

    social = candidates["case-multi::ss2::Q_SOC"]
    assert social["attribute"] == "Social_Status"
    assert social["current_value"] == {"Current_State": "Joining_Social_Activities"}
    assert social["memory"][0]["attribute"] == "Social_Status"
    assert social["memory"][0]["current_value"] == social["current_value"]
    assert "Q_NONE" not in candidates


def test_memconflict_unlinked_question_is_excluded_not_guessed():
    examples = _multi_attribute_memconflict_examples()
    report = audit_examples({"memora": [], "gatemem": [], "memconflict": [examples[-1]]})
    assert report["restrictive_candidates_total"] == 0
    assert report["unsupported_by_reason"]["no_supported_dynamic_updated_attributes_transition"] == 1


def test_ambiguous_gatemem_action_is_rejected(tmp_path):
    root = tmp_path / "office"
    root.mkdir()
    episode = {
        "episode_id": "e1",
        "domain": "office",
        "turns": [{"turn_id": "t1", "speaker": {"principal_id": "u", "role": "user"}, "text": "secret"}],
    }
    checkpoint = {
        "checkpoint_id": "c1",
        "episode_id": "e1",
        "as_of_turn_id": "t1",
        "asker": {"principal_id": "u", "role": "employee"},
        "query_type": "privacy",
        "query_text": "Tell me the secret",
        "gold": {"expected_action": "answer_or_refuse", "leak_targets": ["secret"]},
    }
    (root / "episodes.jsonl").write_text(json.dumps(episode) + "\n")
    (root / "checkpoints.jsonl").write_text(json.dumps(checkpoint) + "\n")
    example = GateMemDatasetAdapter().load(root)[0]
    report = audit_examples({"gatemem": [example], "memora": [], "memconflict": []})
    assert report["restrictive_candidates_total"] == 0
    assert any("no_unambiguous" in reason for reason in report["unsupported_by_reason"])


def test_gate_mem_fixture_has_no_relationship_inferred_allow():
    examples = GateMemDatasetAdapter().load("tests/fixtures/gatemem-mini")
    report = audit_examples({"gatemem": examples, "memora": [], "memconflict": []})
    assert report["restrictive_candidates_total"] == 0
