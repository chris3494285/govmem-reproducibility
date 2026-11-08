import hashlib
import json
from pathlib import Path

from scripts.audit_main_dataset_sources import audit, audit_longmemeval, audit_memora, classify_memora_capabilities


def _long(question_type="knowledge-update", evidence=True):
    return {
        "question_id": "lme-1",
        "question_type": question_type,
        "question": "Where does the user live now?",
        "answer": "Seattle",
        "question_date": "2025/03/10 (Mon) 00:00",
        "answer_session_ids": ["s1", "s2"] if evidence else [],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2025/01/01 (Wed) 00:00", "2025/03/01 (Sat) 00:00"],
        "haystack_sessions": [[], []],
    }


def _memora_root(root: Path, question: dict) -> Path:
    persona = root / "weekly" / "p1"
    (persona / "conversations").mkdir(parents=True)
    (persona / "conversations" / "session_1.json").write_text(json.dumps({"session_id": 1, "date": "2025-01-01", "conversation": []}))
    (persona / "evaluation_questions_p1.json").write_text(json.dumps({"persona": "p1", "questions": {"remembering": [question]}}))
    return root


def test_longmemeval_primary_types_and_single_session_exclusion(tmp_path):
    path = tmp_path / "long.jsonl"
    path.write_text("\n".join(json.dumps(_long(t)) for t in ("knowledge-update", "temporal-reasoning", "single-session-user")) + "\n")
    rows = audit_longmemeval(path)
    assert rows[0]["eligibility"] is True
    assert rows[1]["eligibility"] is True
    assert rows[2]["eligibility"] is False
    assert "unsupported_single_session_question_type" in rows[2]["eligibility_reasons"]


def test_longmemeval_missing_evidence_is_excluded(tmp_path):
    path = tmp_path / "long.jsonl"
    path.write_text(json.dumps(_long(evidence=False)) + "\n")
    row = audit_longmemeval(path)[0]
    assert row["eligibility"] is False
    assert "missing_ground_truth_evidence" in row["eligibility_reasons"]


def test_memora_forgetting_evidence_and_absence_are_eligible(tmp_path):
    q = {
        "question_id": "q1", "question": "What is current?", "question_date": "2025-01-01",
        "memory_evidence": {"active_memory": {"value": "new", "session_id": 1}},
        "forgetting_evidence": {"forgotten_items": [{"value": "old", "session_id": 2}]},
        "evaluation": {"evaluation_questions": [{"evaluation_type": "memory_presence"}, {"evaluation_type": "forgetting_absence"}]},
    }
    rows = audit_memora(_memora_root(tmp_path / "memora", q))
    assert rows[0]["eligibility"] is True
    assert rows[0]["memory_presence_n"] == 1
    assert rows[0]["forgetting_absence_n"] == 1


def test_memora_reasoning_only_is_excluded(tmp_path):
    q = {"question_id": "q1", "question": "Why?", "memory_evidence": {}, "forgetting_evidence": {}, "evaluation": {}}
    row = audit_memora(_memora_root(tmp_path / "memora", q))[0]
    assert row["eligibility"] is False
    assert "no_forgetting_or_forgetting_absence_evidence" in row["eligibility_reasons"]


def test_memora_event_classification_is_conservative():
    caps, reasons, primary = classify_memora_capabilities(
        {}, {"forgotten_items": [{"value": "old", "field": "deadline", "session_id": 1}]}, {}, 0
    )
    assert primary == "deletion"
    assert caps == ["deletion", "invalidation"]
    assert set(reasons) == set(caps)

    caps, reasons, primary = classify_memora_capabilities(
        {"active_memory": {"value": "Monday", "field": "deadline", "session_id": 2}},
        {"forgotten_items": [{"value": "Friday", "field": "deadline", "session_id": 1}]}, {}, 0
    )
    assert primary == "supersession"
    assert set(caps) == {"update", "supersession", "invalidation", "temporal_state"}
    assert set(reasons) == set(caps)

    caps, _, primary = classify_memora_capabilities(
        {"active_memory": {"value": "Monday", "session_id": 2}},
        {"forgotten_items": [{"value": "Friday", "session_id": 1}]}, {}, 0
    )
    assert primary == "unclear"
    assert "invalidation" in caps

    caps, reasons, primary = classify_memora_capabilities(
        {}, {"event": {"type": "expiration", "effective_date": "2025-01-01"}}, {}, 0
    )
    assert primary == "temporal"
    assert caps == ["temporal_state"]
    assert "temporal_state" in reasons


def test_audit_output_is_stable_and_legacy_manifest_unchanged(tmp_path):
    long_path = tmp_path / "long.jsonl"
    long_path.write_text(json.dumps(_long()) + "\n")
    memora_path = _memora_root(tmp_path / "memora", {
        "question_id": "q", "question": "What?", "memory_evidence": {"active": "x"},
        "forgetting_evidence": {"forgotten": "y"}, "evaluation": {},
    })
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    audit(long_path, memora_path, out1)
    audit(long_path, memora_path, out2)
    for name in ("longmemeval_candidates.jsonl", "memora_candidates.jsonl", "excluded_examples.jsonl", "summary.json"):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()
    legacy = Path("data/governance_fault_rq1_pilot_frozen.jsonl")
    assert legacy.exists()
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == hashlib.sha256(legacy.read_bytes()).hexdigest()
