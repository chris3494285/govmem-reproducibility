import json
from pathlib import Path

import pytest

from scripts.audit_oaks_novel_for_main_dataset import audit, audit_question


def _doc(question):
    return {"meta": {"bid": "d1", "title": "Title", "author": "Author", "num_chunks": 3, "num_qas": 1}, "data": {"chunks": {"0": "one", "1": "two", "2": "three"}, "qas": {question["question"]: question}}}


def _question(qid, mapping, **extra):
    sources = {}
    for chunk, answer in mapping.items():
        chunk = {"c1": "0", "c2": "1", "c3": "2"}.get(chunk, chunk)
        sources.setdefault(answer, {})[chunk] = [f"evidence for {answer} at {chunk}"]
    return {"question_id": qid, "options": list(set(mapping.values())), "option_sources": sources, "question": "What is the state?", **extra}


def test_stable_answer_is_not_update_and_one_transition_is_update(tmp_path):
    path = tmp_path / "oaks.json"
    path.write_text(json.dumps([_doc(_question("stable", {"c1": "A", "c2": "A"})), _doc(_question("changed", {"c1": "A", "c2": "B"}))]))
    # Duplicate document IDs are intentionally changed for a valid composite cohort.
    value = json.loads(path.read_text())
    value[1]["meta"]["bid"] = "d2"
    path.write_text(json.dumps(value))
    summary = audit(path, tmp_path / "out")
    assert summary["raw_unique_question_count"] == 2
    assert summary["eligible_unique_question_count"] == 1
    candidate = json.loads((tmp_path / "out/oaks_novel_candidates.jsonl").read_text().splitlines()[0])
    assert candidate["primary_event_type"] == "update"


def test_multiple_transitions_are_preserved_not_inflated(tmp_path):
    path = tmp_path / "oaks.json"
    doc = _doc(_question("multi", {"c1": "A", "c2": "B", "c3": "A"}))
    path.write_text(json.dumps([doc]))
    summary = audit(path, tmp_path / "out")
    row = json.loads((tmp_path / "out/oaks_novel_candidates.jsonl").read_text().splitlines()[0])
    assert summary["raw_unique_question_count"] == 1
    assert summary["total_answer_transition_count"] == 2
    assert row["transition_count"] == 2
    assert summary["eligible_unique_question_count"] == 1
    assert summary["projected_govmem_event_counts"]["update"] == 79


def test_explicit_temporal_dependency_and_ambiguous_exclusion(tmp_path):
    path = tmp_path / "oaks.json"
    temporal = _question("temporal", {"c1": "A", "c2": "B"}, requires_temporal_reasoning=True)
    ambiguous = _question("ambiguous", {"c1": "A", "c2": "B"})
    ambiguous.pop("option_sources")
    second = _doc(ambiguous)
    second["meta"]["bid"] = "d2"
    path.write_text(json.dumps([_doc(temporal), second]))
    summary = audit(path, tmp_path / "out")
    assert summary["counts_by_primary_event_type"] == {"mixed": 1}
    assert summary["unclear_count"] == 1


def test_update_recovery_requires_evidence_and_ordered_chunks(tmp_path):
    path = tmp_path / "oaks.json"
    no_evidence = _question("noe", {"c1": "A", "c2": "B"})
    no_evidence.pop("option_sources")
    path.write_text(json.dumps([_doc(no_evidence)]))
    summary = audit(path, tmp_path / "out")
    assert summary["eligible_unique_question_count"] == 0
    bad = _question("bad", {"unknown": "A", "c2": "B"})
    with pytest.raises(ValueError, match="unordered chunk"):
        audit_question(_doc(bad), bad, "d", {"c1": 0, "c2": 1, "c3": 2}, "hash")


def test_duplicate_ids_fail_loudly(tmp_path):
    path = tmp_path / "oaks.json"
    doc = _doc(_question("same", {"c1": "A", "c2": "B"}))
    path.write_text(json.dumps([doc, doc]))
    with pytest.raises(ValueError, match="Duplicate OAKS"):
        audit(path, tmp_path / "out")


def test_native_nested_meta_bid_and_document_metadata():
    from scripts.audit_oaks_novel_for_main_dataset import _doc_id
    doc = _doc(_question("q", {"c1": "A", "c2": "B"}))
    assert _doc_id(doc, 0) == "d1"
    question = next(iter(doc["data"]["qas"].values()))
    row = audit_question(doc, question, "d1", {"c1": 0, "c2": 1, "c3": 2}, "hash", doc["meta"])
    assert row["document_title"] == "Title"
    assert row["document_author"] == "Author"
    assert row["num_chunks"] == 3
    assert row["num_qas"] == 1


def test_missing_native_meta_bid_fails():
    from scripts.audit_oaks_novel_for_main_dataset import _doc_id
    doc = _doc(_question("q", {"c1": "A"}))
    doc["meta"].pop("bid")
    with pytest.raises(ValueError, match="meta.bid"):
        _doc_id(doc, 0)


def test_previous_answer_still_holds_is_persistence_not_transition(tmp_path):
    question = {"question_id": "hold", "options": ["A", "B"], "option_sources": {"A": {"0": ["a evidence"]}, "Previous answer still holds": {"1": [], "2": []}}}
    question["question"] = "What is the state?"
    doc = {"meta": {"bid": "hold-doc"}, "data": {"chunks": {"0": "x", "1": "y", "2": "z"}, "qas": {question["question"]: question}}}
    path = tmp_path / "hold.json"
    path.write_text(json.dumps([doc]))
    summary = audit(path, tmp_path / "out")
    assert summary["eligible_unique_question_count"] == 0
    excluded = json.loads((tmp_path / "out/oaks_novel_excluded.jsonl").read_text().splitlines()[0])
    assert excluded["transition_count"] == 0


def test_repeat_is_byte_identical(tmp_path):
    path = tmp_path / "oaks.json"
    path.write_text(json.dumps([_doc(_question("q", {"c1": "A", "c2": "B"}))]))
    audit(path, tmp_path / "a")
    audit(path, tmp_path / "b")
    for name in ("oaks_novel_candidates.jsonl", "oaks_novel_excluded.jsonl", "summary.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
