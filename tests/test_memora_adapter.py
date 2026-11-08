import json
from pathlib import Path

from eval.memora_adapter import MemoraDatasetAdapter


FIXTURE = Path(__file__).parent / "fixtures" / "memora-mini"


def test_memora_maps_active_memory_and_session_fields():
    example = MemoraDatasetAdapter().load(FIXTURE)[0]

    assert example.query.query_id == "deadline_active"
    assert example.query.user_id == "software_engineer"
    assert example.query.current_time.isoformat() == "2025-06-03T00:00:00"
    assert example.interactions[0].session_id == "1"
    assert example.interactions[0].metadata["date"] == "2025-06-01"
    assert example.evaluator_metadata["governance_source"] == "dataset_native_gold"
    assert example.evaluator_metadata["gold"]["memory_evidence"]["active_memory"]["value"] == "Monday"


def test_memora_preserves_forgetting_gold_and_lifecycle_operations():
    example = MemoraDatasetAdapter().load(FIXTURE)[0]
    operations = example.evaluator_metadata["session_operations"]

    assert [operation["operation"] for operation in operations] == ["add", "update", "delete"]
    assert operations[1]["operation_details"]["new"] == "The weekly report is due Monday."
    assert example.evaluator_metadata["session_turn_annotations"][0]["share_memory"] is True
    assert example.evaluator_metadata["gold"]["forgetting_evidence"]["forgotten_items"][0]["value"] == "Friday"
    evaluation = example.evaluator_metadata["gold"]["evaluation"]["evaluation_questions"]
    assert [item["expected_answer"] for item in evaluation] == ["yes", "no"]


def test_memora_evaluator_gold_does_not_leak_into_model_visible_text():
    example = MemoraDatasetAdapter().load(FIXTURE)[0]
    visible_text = "\n".join([example.query.text, *(interaction.content for interaction in example.interactions)])

    assert "memory_evidence" not in visible_text
    assert "forgetting_evidence" not in visible_text
    assert "expected_answer" not in visible_text
    assert "session_operations" not in visible_text


def test_memora_fixture_is_json_and_uses_public_file_layout():
    for path in [FIXTURE / "evaluation_questions_software_engineer.json", *sorted((FIXTURE / "conversations").glob("*.json"))]:
        with path.open(encoding="utf-8") as handle:
            json.load(handle)


def test_memora_official_persona_scope_preserves_order_and_task_applicability(tmp_path):
    persona = tmp_path / "weekly" / "software_engineer"
    conversations = persona / "conversations"
    conversations.mkdir(parents=True)
    (conversations / "session_0001.json").write_text(json.dumps({
        "session_id": 1, "session_type": "memory_introduction", "operation": "add",
        "operation_details": {}, "date": "2025-06-01", "persona": "software_engineer",
        "conversation": [
            {"turn": 1, "speaker": "user", "message": "Remember A.", "share_memory": True},
            {"turn": 2, "speaker": "ai_agent", "message": "Okay."},
        ],
    }), encoding="utf-8")
    (conversations / "session_0002.json").write_text(json.dumps({
        "session_id": 2, "session_type": "memory_update", "operation": "delete",
        "operation_details": {"value": "A"}, "date": "2025-06-02", "persona": "software_engineer",
        "conversation": [{"turn": 1, "speaker": "user", "message": "Forget A.", "share_memory": False}],
    }), encoding="utf-8")
    (persona / "evaluation_questions_software_engineer.json").write_text(json.dumps({
        "persona": "software_engineer", "date_range": {"start_date": "2025-06-01", "end_date": "2025-06-02"},
        "questions": {
            "remembering": [{"question_id": "remember", "question": "What is A?", "question_date": "2025-06-02", "memory_evidence": {"x": 1}, "forgetting_evidence": {"forgotten_items": [{"value": "A"}]}, "evaluation": {}}],
            "reasoning": [{"question_id": "reason", "question": "Why?", "question_date": "2025-06-02", "memory_evidence": {}, "evaluation": {}}],
            "recommending": [],
        },
    }), encoding="utf-8")
    examples = MemoraDatasetAdapter().load(persona)
    assert [example.query.metadata["task_type"] for example in examples] == ["remembering", "reasoning"]
    assert examples[0].query.metadata["lifecycle_applicable"] is True
    assert examples[1].query.metadata["lifecycle_applicable"] is False
    assert [item.interaction_id for item in examples[0].interactions] == [
        "software_engineer::1::1", "software_engineer::1::2", "software_engineer::2::1"
    ]
    assert examples[0].evaluator_metadata["gold"]["forgetting_evidence"]["forgotten_items"][0]["value"] == "A"
    visible = " ".join(item.content for item in examples[0].interactions)
    assert "forgetting_evidence" not in visible


def test_memora_explicit_period_parent_scope_prefixes_persona_ids(tmp_path):
    for persona_name in ("a", "b"):
        persona = tmp_path / "weekly" / persona_name
        (persona / "conversations").mkdir(parents=True)
        (persona / "conversations" / "session_0001.json").write_text(json.dumps({
            "session_id": 1, "date": "2025-01-01", "conversation": [],
        }), encoding="utf-8")
        (persona / f"evaluation_questions_{persona_name}.json").write_text(json.dumps({
            "persona": persona_name, "questions": {"remembering": [{"question_id": "q", "question": "Q", "question_date": "2025-01-01"}]}
        }), encoding="utf-8")
    examples = MemoraDatasetAdapter().load(tmp_path / "weekly")
    assert [example.query.query_id for example in examples] == ["a::q", "b::q"]
