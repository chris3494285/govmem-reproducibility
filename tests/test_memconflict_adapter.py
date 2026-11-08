import json
from pathlib import Path

from eval.memconflict_adapter import MemConflictDatasetAdapter


FIXTURE = Path(__file__).parent / "fixtures" / "memconflict-mini" / "memconflict.jsonl"


def test_memconflict_maps_native_categories_to_validity_dimensions():
    examples = MemConflictDatasetAdapter().load(FIXTURE)
    by_type = {example.metadata["conflict_type"]: example for example in examples}

    assert by_type["dynamic"].query.gold_answer == "Seattle"
    assert by_type["dynamic"].query.metadata["validity_dimension"] == "temporal_validity"
    assert by_type["static"].query.metadata["validity_dimension"] == "factual_validity"
    assert by_type["conditional"].query.metadata["validity_dimension"] == "contextual_applicability"


def test_memconflict_preserves_supporting_memory_and_dynamic_context():
    examples = {example.query.query_id: example for example in MemConflictDatasetAdapter().load(FIXTURE)}

    dynamic_gold = examples["dynamic_001"].evaluator_metadata["gold"]
    conditional_gold = examples["conditional_001"].evaluator_metadata["gold"]
    assert dynamic_gold["supporting_memory"]["session_id"] == "s2"
    assert dynamic_gold["dynamic_state"]["current"] == "Seattle"
    assert conditional_gold["conditional_context"]["condition"] == "weekdays"
    assert examples["dynamic_001"].interactions[1].session_id == "s2"
    assert examples["dynamic_001"].interactions[1].timestamp.isoformat() == "2024-06-01T00:00:00"


def test_memconflict_evaluator_gold_does_not_leak_into_model_visible_text():
    for example in MemConflictDatasetAdapter().load(FIXTURE):
        visible_text = "\n".join([example.query.text, *(interaction.content for interaction in example.interactions)])
        assert "gold_answer" not in visible_text
        assert "supporting_memory" not in visible_text
        assert "dynamic_state" not in visible_text
        assert "conditional_context" not in visible_text


def test_memconflict_fixture_is_jsonl():
    with FIXTURE.open(encoding="utf-8") as handle:
        for line in handle:
            json.loads(line)


def test_memconflict_official_question_schema_uses_native_question_type(tmp_path):
    official = {
        "ID": "official-1",
        "metadata": {"persona_seed": "seed"},
        "Fixed_Profile": {}, "Dynamic_Profile": {}, "Preference_Profile": {},
        "Personality": {}, "Life_Goal": {}, "Others_Profile": {},
        "Full_Session_Chain": [
            {
                "Session_ID": 0, "Date": "2022-01-01", "Session_Type": "initial_reveal",
                "Session_Questions": [{
                    "question_id": "Q_001", "question": "Did the residence change?",
                    "answer": "Yes", "conflict_type": "dynamic_conflict",
                    "ability_target": "track_state_over_time", "difficulty": "easy",
                }],
                "Session_Dialogue": {
                    "dialogue_turn_1": [{"role": "user", "content": "I moved."}, {"role": "assistant", "content": "Noted."}]
                },
            },
            {
                "Session_ID": 1, "Date": "2022-01-02", "Session_Type": "update",
                "Session_Questions": [
                    {"question_id": "Q_001", "question": "Which school?", "answer": "A", "conflict_type": "static_conflict", "ability_target": "recover_truth", "difficulty": "hard"},
                    {"question_id": "Q_002", "question": "When?", "answer": "Later", "conflict_type": "conditional_conflict", "ability_target": "bind_condition", "difficulty": "hard"},
                ],
                "Session_Dialogue": {
                    "dialogue_turn_1": [{"role": "user", "content": "More context."}]
                },
            },
        ],
    }
    path = tmp_path / "Step4_4.jsonl"
    path.write_text(json.dumps(official) + "\n", encoding="utf-8")
    examples = MemConflictDatasetAdapter().load(path)
    assert [example.query.metadata["task_type"] for example in examples] == ["dynamic", "static", "conditional"]
    assert examples[0].query.query_id == "official-1::s0::Q_001"
    assert [item.content for item in examples[1].interactions] == ["I moved.", "Noted.", "More context."]
    assert examples[1].evaluator_metadata["gold"]["answer"] == "A"
    assert "A" not in " ".join(item.content for item in examples[1].interactions)
