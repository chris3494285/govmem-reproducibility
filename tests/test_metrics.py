from pathlib import Path

from eval.dataset import load_dataset, to_query_context
from eval.metrics import answer_matches, evidence_hit, exact_match, normalize_text, token_f1, policy_violation_count, stale_memory_error_count, underspecified_answer


DATASET = Path("data/sample_longmemeval.jsonl")


def test_exact_match_and_token_f1():
    assert exact_match("Project Iris", "project iris")
    assert normalize_text("  Seattle\n") == "seattle"
    assert normalize_text(4) == "4"
    assert normalize_text(None) == ""
    assert normalize_text("  White   Adidas Sneakers ") == "white adidas sneakers"
    assert exact_match(4, "4")
    assert answer_matches(4, "4")
    assert answer_matches("The answer is Seattle.", "Seattle")
    assert answer_matches("Seattle", "Seattle")
    assert not answer_matches("Portland", "Seattle")
    score = token_f1("green tea", "tea")
    assert 0 < score < 1
    assert token_f1("", "") == 1.0
    assert token_f1("a b", "x y") == 0.0


def test_evidence_hit_detects_gold_answer_in_text():
    assert evidence_hit("Seattle", ["I moved to Seattle last month."])
    assert not evidence_hit("Seattle", ["I moved to Portland last month."])
    assert evidence_hit(4, ["I participated in 4 charity events."])
    assert not evidence_hit(4, ["I participated in four charity events."])
    assert evidence_hit(4, [4])
    assert not evidence_hit(4, [None])


def test_normalize_text_handles_common_container_types():
    assert normalize_text([" White ", 4, None, True]) == "white 4 true"
    assert normalize_text({"b": "Two", "a": 1}) == "1 two"


def test_underspecified_answer_detection():
    assert underspecified_answer("Adidas sneakers", "white Adidas sneakers")
    assert not underspecified_answer("white Adidas sneakers", "white Adidas sneakers")
    assert not underspecified_answer("unknown", "white Adidas sneakers")


def test_violation_metrics_detect_bad_memories():
    examples = load_dataset(DATASET)
    example = next(item for item in examples if item.id == "ex5_deleted_memory")
    context = to_query_context(example)
    deleted_record = next(record for record in example.memories if record.memory_id == "m5_deleted")
    valid_record = next(record for record in example.memories if record.memory_id == "m5_valid")

    assert policy_violation_count([deleted_record, valid_record], context) == 1
    assert stale_memory_error_count([deleted_record, valid_record], context) == 1
