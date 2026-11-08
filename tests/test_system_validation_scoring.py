import pytest

from eval.system_validation_scoring import max_output_tokens_for_row, memora_subtype, scoring_spec, score_task_answer, score_task_answer_detailed


def test_longmemeval_uses_native_exact_answer():
    row = {"source_dataset": "longmemeval", "task_oracle_answer": "25"}
    spec = scoring_spec(row)
    assert spec["deterministic"] is True
    assert spec["exact_match_appropriate"] is True
    assert score_task_answer(row, "25")[0] is True
    assert score_task_answer(row, "24")[0] is False


def test_oaks_uses_native_option_state():
    row = {
        "source_dataset": "oaks_novel",
        "task_oracle_answer": "Southwards, returning to England",
        "task_options": ["Geneva", "Southwards, returning to England"],
    }
    spec = scoring_spec(row)
    assert spec["method"] == "normalized_native_option_match"
    assert score_task_answer(row, "Southwards, returning to England")[0] is True
    assert score_task_answer(row, "Geneva")[0] is False


def test_memora_structured_native_facts_are_scored_without_exact_match():
    row = {
        "source_dataset": "memora",
        "task_oracle_answer": {
            "remaining_tasks": [
                {"value": "Organize home office"},
                {"value": "Plan quiet research time"},
            ]
        },
    }
    spec = scoring_spec(row)
    assert spec["method"] == "memora_activity_todos_v1"
    assert spec["exact_match_appropriate"] is False
    assert score_task_answer_detailed(row, "Organize home office and plan quiet research time")["correctness"] is True


def test_memora_numeric_formatting_and_list_order_are_deterministic():
    row = {
        "source_dataset": "memora",
        "task_oracle_answer": {"content_data": {"budget": 125000, "deliverables": ["Report", "Prototype"]}},
    }
    scored = score_task_answer_detailed(row, "The budget is $125k. Deliverables: Prototype and Report.")
    assert scored["correctness"] is True
    assert scored["scoring_details"]["required_item_recall"] == 1.0


def test_oaks_ambiguous_containment_is_unscored():
    row = {
        "source_dataset": "oaks_novel",
        "task_oracle_answer": "black cat",
        "task_options": ["cat", "black cat"],
    }
    assert score_task_answer_detailed(row, "The answer is a black cat.")["correctness"] is None


@pytest.mark.parametrize("count", [5, 6, 7])
def test_oaks_resolves_only_labels_valid_for_actual_option_count(count):
    from eval.system_validation_scoring import resolve_oaks_option
    options = [f"Option {i}" for i in range(count)]
    row = {"source_dataset": "oaks_novel", "task_options": options}
    assert resolve_oaks_option(row, chr(65 + count - 1)) == options[-1]
    assert resolve_oaks_option(row, "Z") is None


def test_memora_preference_constraints_reject_disliked_recommendation():
    row = {"source_dataset": "memora", "task_oracle_answer": {
        "specific_items": [{"item": "Modest Mouse"}, {"item": "Pavement"}],
        "subcategory_data": {"likes": [{"item": "Modest Mouse"}, {"item": "Pavement"}], "dislikes": [{"item": "Talking Heads"}]},
    }}
    assert memora_subtype(row["task_oracle_answer"]) == "preference_recommendation"
    assert score_task_answer_detailed(row, "Modest Mouse and Pavement")["correctness"] is True
    bad = score_task_answer_detailed(row, "Modest Mouse, Pavement, and Talking Heads")
    assert bad["correctness"] is False
    assert bad["scoring_details"]["forbidden_item_violation_count"] == 1


def test_memora_travel_nested_constraints_include_dislikes_and_visited():
    row = {"source_dataset": "memora", "task_oracle_answer": {"memory_items": {
        "climates": {"likes": [{"item": "tropical"}], "dislikes": []},
        "regions": {"likes": [], "dislikes": [{"item": "Santorini"}]},
        "already_visited_list": {"likes": [{"item": "Canaves Oia Hotel, Santorini"}], "dislikes": []},
    }}}
    scored = score_task_answer_detailed(row, "Santorini")
    assert scored["correctness"] is False
    assert scored["scoring_details"]["forbidden_item_violation_count"] == 1
    assert scored["scoring_details"]["already_visited_items"] == ["Canaves Oia Hotel, Santorini"]


def test_memora_todos_and_source_budgets():
    row = {"source_dataset": "memora", "task_oracle_answer": {"remaining_tasks": [{"value": "Buy groceries"}]}}
    assert score_task_answer_detailed(row, "Buy groceries")["correctness"] is True
    assert max_output_tokens_for_row(row) == 256
    assert max_output_tokens_for_row({"source_dataset": "oaks_novel"}) == 128


def test_memora_open_ended_without_native_criterion_is_explicitly_unscored():
    row = {
        "source_dataset": "memora",
        "task_oracle_answer": {"session_history": [123], "task_count": 1},
    }
    spec = scoring_spec(row)
    assert spec["method"] == "unscored_no_native_criterion"
    assert score_task_answer(row, "A plausible proposal")[0] is None
    assert score_task_answer(row, "A plausible proposal")[1] == "unscored_no_native_criterion"
