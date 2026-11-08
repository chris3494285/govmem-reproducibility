"""Audit LongMemEval and Memora source records for governance eligibility.

This is an audit only: it never imports a provider, samples records, or calls
an external service.  Native source fields are retained in the audit rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_LONGMEMEVAL = ROOT / "data/official_longmemeval_oracle_200.jsonl"
DEFAULT_MEMORA = ROOT / "data/memora"
OUT_DIR = ROOT / "results/main_dataset_audit_v3"
PRIMARY_LONG_TYPES = {"knowledge-update", "temporal-reasoning"}
UNSUPPORTED_LONG_TYPES = {
    "single-session-user", "single-session-assistant", "single-session-preference"
}
MEMORA_PERIODS = {"weekly", "monthly", "quarterly"}


def _explicit_identity(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    for key in ("semantic_key", "memory_key", "entity", "entity_id", "field", "attribute", "topic", "item_id"):
        if value.get(key) is not None:
            return str(value[key]).lower()
    return None


def _evidence_values(value: Any, role: str) -> list[dict[str, Any]]:
    """Collect native old/current value records without interpreting prose."""
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if key_lower in {"forgotten_items", "deleted_items", "obsolete_items", "previous", "old_value", "before", "prior_value"}:
                if isinstance(item, list):
                    for child in item:
                        if isinstance(child, dict):
                            found.append({"role": "old", "value": child.get("value", child.get("content", child.get("item"))), "identity": _explicit_identity(child), "session_id": child.get("session_id")})
                elif isinstance(item, dict):
                    found.append({"role": "old", "value": item.get("value", item.get("content", item.get("item"))), "identity": _explicit_identity(item), "session_id": item.get("session_id")})
            elif key_lower in {"active_memory", "current", "current_value", "new_value", "replacement", "after"}:
                if isinstance(item, dict):
                    found.append({"role": "new", "value": item.get("value", item.get("content", item.get("item"))), "identity": _explicit_identity(item), "session_id": item.get("session_id")})
            found.extend(_evidence_values(item, role))
    elif isinstance(value, list):
        for item in value:
            found.extend(_evidence_values(item, role))
    return [item for item in found if item.get("value") is not None]


def _contains_explicit_transition(value: Any) -> bool:
    if isinstance(value, dict):
        if any(key in value for key in ("old_value", "previous_value", "new_value", "replacement", "supersedes")):
            return True
        return any(_contains_explicit_transition(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_explicit_transition(item) for item in value)
    return False


def _contains_temporal_marker(value: Any) -> bool:
    if isinstance(value, dict):
        if any(str(key).lower() in {"event_date", "effective_date", "valid_from", "valid_until", "timestamp", "occurred_at", "expires_at"} for key in value):
            return True
        return any(_contains_temporal_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_temporal_marker(item) for item in value)
    return False


def classify_memora_capabilities(memory_evidence: Any, forgetting_evidence: Any, question: dict[str, Any], absence_n: int) -> tuple[list[str], dict[str, str], str]:
    """Classify only event types established by Memora-native evidence.

    In particular, a forgotten item plus an unrelated active item is not
    treated as an update.  A matching semantic identity or an explicit native
    replacement marker is required for old-to-new classification.
    """
    old_values = _evidence_values(forgetting_evidence, "old")
    new_values = _evidence_values(memory_evidence, "new")
    reasons: dict[str, str] = {}
    capabilities: list[str] = []
    matched = False
    if old_values and new_values:
        for old in old_values:
            for new in new_values:
                if old.get("identity") and old.get("identity") == new.get("identity"):
                    matched = True
                elif old.get("identity") is not None or new.get("identity") is not None:
                    continue
                # A native explicit old/new pair is sufficient; generic
                # active/forgotten values without identity are intentionally
                # ambiguous.
    explicit_transition = _contains_explicit_transition(forgetting_evidence) or _contains_explicit_transition(memory_evidence)
    if old_values and new_values and (matched or explicit_transition):
        capabilities.extend(["update", "supersession", "invalidation"])
        reasons["update"] = "Memora evidence identifies an old value and a replacement for the same semantic entity."
        reasons["supersession"] = "The newer native value supersedes the earlier value for the matched semantic entity."
        reasons["invalidation"] = "The native forgetting evidence marks the earlier value as no longer current."
        primary = "supersession" if matched else "update"
    elif old_values and not new_values:
        capabilities.extend(["deletion", "invalidation"])
        reasons["deletion"] = "Native forgetting evidence identifies an obsolete item and no current replacement value."
        reasons["invalidation"] = "Native forgetting evidence identifies the earlier item as no longer valid/current."
        primary = "deletion"
    elif old_values and new_values:
        capabilities.append("invalidation")
        reasons["invalidation"] = "Native forgetting evidence establishes obsolescence, but current and obsolete values lack a deterministic shared identity."
        primary = "unclear"
    elif absence_n:
        primary = "unclear"
        reasons["invalidation"] = "A forgetting-absence evaluation exists, but released evidence does not expose an identifiable state transition."
    else:
        primary = "unclear"
    session_ids = [item.get("session_id") for item in old_values + new_values if item.get("session_id") is not None]
    if len(session_ids) >= 2 and len(set(str(x) for x in session_ids)) >= 2:
        capabilities.append("temporal_state")
        reasons["temporal_state"] = "Native evidence places obsolete and current values in distinct sessions, so chronology affects validity."
    elif _contains_temporal_marker(forgetting_evidence) or _contains_temporal_marker(memory_evidence):
        capabilities.append("temporal_state")
        reasons["temporal_state"] = "Native evidence contains an explicit lifecycle/effective-time field relevant to validity."
    if "temporal_state" in capabilities and primary == "unclear" and not old_values:
        primary = "temporal"
    return sorted(set(capabilities)), reasons, primary


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset source is missing: {path}")
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON array in {path}")
        records = value
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if any(not isinstance(item, dict) for item in records):
        raise ValueError(f"Every record in {path} must be an object")
    return records


def _nonempty(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _ids(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower().endswith("session_id") or key.lower().endswith("session_ids"):
                if isinstance(item, list):
                    found.extend(str(x) for x in item)
                elif item is not None:
                    found.append(str(item))
            found.extend(_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_ids(item))
    return sorted(set(found))


def _dataset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    elif path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        for item in files:
            digest.update(str(item.relative_to(path)).encode())
            digest.update(b"\0")
            digest.update(item.read_bytes())
    else:
        raise FileNotFoundError(f"Dataset source is missing: {path}")
    return digest.hexdigest()


def _long_row(raw: dict[str, Any], source_path: Path) -> dict[str, Any]:
    required = ("question_id", "question_type", "question", "question_date", "answer")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"LongMemEval {raw.get('question_id', '<unknown>')} missing {missing}")
    question_id = str(raw["question_id"])
    qtype = raw["question_type"]
    if not isinstance(qtype, str) or not qtype:
        raise ValueError(f"LongMemEval {question_id} has invalid question_type")
    sessions = raw.get("haystack_sessions")
    session_ids = raw.get("haystack_session_ids")
    dates = raw.get("haystack_dates")
    reasons: list[str] = []
    evidence = raw.get("answer_session_ids")
    if not isinstance(sessions, list) or not isinstance(session_ids, list) or not isinstance(dates, list):
        reasons.append("missing_session_timestamp_metadata")
        sessions, session_ids, dates = sessions or [], session_ids or [], dates or []
    evidence_ids = [str(x) for x in evidence] if isinstance(evidence, list) else []
    if not evidence_ids:
        reasons.append("missing_ground_truth_evidence")
    known_ids = {str(x) for x in session_ids}
    if any(item not in known_ids for item in evidence_ids):
        reasons.append("evidence_session_not_in_haystack")
    if len(session_ids) != len(dates) or any(not item for item in dates):
        reasons.append("insufficient_session_timestamps")
    if qtype in UNSUPPORTED_LONG_TYPES:
        reasons.append("unsupported_single_session_question_type")
    if qtype == "multi-session":
        reasons.append("multi_session_audit_only")
    if qtype not in PRIMARY_LONG_TYPES and qtype != "multi-session" and qtype not in UNSUPPORTED_LONG_TYPES:
        reasons.append("question_type_not_in_primary_scope")
    if qtype in PRIMARY_LONG_TYPES and len(set(evidence_ids)) < 1:
        reasons.append("no_deterministic_evidence_state")
    # A transition requires native evidence in at least two dated sessions;
    # no transition is synthesized from message wording.
    evidence_dates = {str(dates[list(session_ids).index(item)]) for item in evidence_ids if item in known_ids}
    if qtype in PRIMARY_LONG_TYPES and len(evidence_dates) < 2:
        reasons.append("no_deterministic_state_transition")
    eligible = qtype in PRIMARY_LONG_TYPES and not reasons
    capabilities = ["temporal_state"]
    if qtype == "knowledge-update":
        capabilities.append("update")
    if qtype == "temporal-reasoning":
        capabilities.append("invalidation")
    return {
        "source_dataset": "longmemeval",
        "source_example_id": question_id,
        "question_id": question_id,
        "question_type": qtype,
        "question": raw["question"],
        "expected_answer": raw["answer"],
        "question_date": raw["question_date"],
        "answer_session_ids": evidence_ids,
        "evidence_session_ids": evidence_ids,
        "abstention": bool(raw.get("is_abstention", False) or qtype == "abstention"),
        "eligibility": eligible,
        "eligibility_reasons": reasons or ["eligible_primary_question_type_with_native_transition_evidence"],
        "governance_capabilities": sorted(set(capabilities)) if eligible else [],
        "primary_event_type": "update" if qtype == "knowledge-update" else "temporal" if qtype == "temporal-reasoning" else "unclear",
        "source_path": str(source_path),
    }


def _long_converted_row(raw: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Audit a repository-converted record without mistaking it for native data.

    Conversion deliberately drops question type and answer-session evidence.
    Such records remain useful for inventory, but cannot become governance
    candidates without reconstructing labels, which this audit never does.
    """
    required = ("id", "question", "answer", "memories")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"LongMemEval converted record missing {missing}: {source_path}")
    memories = raw["memories"]
    if not isinstance(memories, list):
        raise ValueError(f"LongMemEval converted memories must be a list: {source_path}")
    session_ids = sorted({str(item.get("official_session_id")) for item in memories if isinstance(item, dict) and item.get("official_session_id") is not None})
    return {
        "source_dataset": "longmemeval",
        "source_example_id": str(raw["id"]),
        "question_id": str(raw["id"]),
        "question_type": None,
        "question": raw["question"],
        "expected_answer": raw["answer"],
        "question_date": raw.get("current_time"),
        "answer_session_ids": [],
        "evidence_session_ids": session_ids,
        "abstention": False,
        "eligibility": False,
        "eligibility_reasons": ["converted_source_lacks_native_question_type_and_ground_truth_evidence"],
        "governance_capabilities": [],
        "primary_event_type": "unclear",
        "source_path": str(source_path),
    }


def audit_longmemeval(path: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in _load_records(path):
        if "question_id" in raw or "question_type" in raw:
            rows.append(_long_row(raw, path))
        elif {"id", "question", "answer", "memories"}.issubset(raw):
            rows.append(_long_converted_row(raw, path))
        else:
            raise ValueError(f"Unsupported LongMemEval record schema in {path}: {sorted(raw)}")
    return rows


def _memora_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Memora source directory is missing: {root}")
    files = sorted(root.rglob("evaluation_questions_*.json"))
    if not files:
        raise ValueError(f"No Memora evaluation_questions_*.json files found under {root}")
    return files


def _memora_row(question: dict[str, Any], data: dict[str, Any], source_path: Path, period: str) -> dict[str, Any]:
    required = ("question_id", "question")
    missing = [key for key in required if key not in question]
    if missing:
        raise ValueError(f"Memora question in {source_path} missing {missing}")
    task_type = question.get("task_type")
    memory_evidence = question.get("memory_evidence", {})
    forgetting_evidence = question.get("forgetting_evidence", {})
    evaluation = question.get("evaluation", {}) or {}
    eval_items = evaluation.get("evaluation_questions", []) if isinstance(evaluation, dict) else []
    presence_n = sum(item.get("evaluation_type") == "memory_presence" for item in eval_items if isinstance(item, dict))
    absence_n = sum(item.get("evaluation_type") == "forgetting_absence" for item in eval_items if isinstance(item, dict))
    reasons: list[str] = []
    if not _nonempty(forgetting_evidence) and absence_n == 0:
        reasons.append("no_forgetting_or_forgetting_absence_evidence")
    if not _nonempty(memory_evidence):
        reasons.append("missing_current_memory_evidence")
    if not isinstance(forgetting_evidence, (dict, list)):
        reasons.append("invalid_forgetting_evidence_shape")
    if not _nonempty(forgetting_evidence) and absence_n == 0:
        reasons.append("no_native_invalidation_event")
    eligible = not reasons
    capabilities, capability_reasons, primary_event_type = classify_memora_capabilities(
        memory_evidence, forgetting_evidence, question, absence_n
    )
    conversation_root = source_path.parent / "conversations"
    session_ids = []
    if conversation_root.is_dir():
        for session_file in sorted(conversation_root.glob("session_*.json")):
            session = json.loads(session_file.read_text(encoding="utf-8"))
            if "session_id" not in session:
                raise ValueError(f"Memora session missing session_id: {session_file}")
            session_ids.append(str(session["session_id"]))
    persona = data.get("persona") or source_path.parent.name
    return {
        "source_dataset": "memora",
        "source_example_id": f"{period}::{persona}::{question['question_id']}",
        "question_id": question["question_id"],
        "persona": persona,
        "period": period,
        "task_type": task_type,
        "question": question["question"],
        "question_date": question.get("question_date"),
        "memory_evidence": memory_evidence,
        "forgetting_evidence": forgetting_evidence,
        "memory_presence_n": presence_n,
        "forgetting_absence_n": absence_n,
        "relevant_source_session_ids": session_ids,
        "eligibility": eligible,
        "eligibility_reasons": reasons or ["native_current_and_invalidation_evidence"],
        "governance_capabilities": capabilities if eligible else [],
        "governance_capability_reasons": capability_reasons if eligible else {},
        "primary_event_type": primary_event_type,
        "source_path": str(source_path),
    }


def audit_memora(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in _memora_files(root):
        data = json.loads(path.read_text(encoding="utf-8"))
        period = next((part for part in path.parts if part in MEMORA_PERIODS), "unknown")
        if period == "unknown":
            raise ValueError(f"Memora question is not under weekly/monthly/quarterly: {path}")
        questions = data.get("questions")
        if not isinstance(questions, dict):
            raise ValueError(f"Memora questions must be an object: {path}")
        for task_type, entries in questions.items():
            if not isinstance(entries, list):
                raise ValueError(f"Memora task {task_type} is not a list: {path}")
            for question in entries:
                if not isinstance(question, dict):
                    raise ValueError(f"Memora question is not an object: {path}")
                output.append(_memora_row({**question, "task_type": task_type}, data, path, period))
    return output


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def audit(longmemeval: Path, memora: Path, output_dir: Path, allow_missing_source: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_rows: list[dict[str, Any]] = []
    memora_rows: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    source_status: dict[str, str] = {}
    if longmemeval.exists():
        long_rows = audit_longmemeval(longmemeval)
        fingerprints["longmemeval"] = _dataset_fingerprint(longmemeval)
        source_status["longmemeval"] = "audited"
    elif not allow_missing_source:
        raise FileNotFoundError(f"LongMemEval source is missing: {longmemeval}")
    else:
        source_status["longmemeval"] = "missing"
    if memora.exists():
        memora_rows = audit_memora(memora)
        fingerprints["memora"] = _dataset_fingerprint(memora)
        source_status["memora"] = "audited"
    elif not allow_missing_source:
        raise FileNotFoundError(f"Memora source is missing: {memora}")
    else:
        source_status["memora"] = "missing"
    all_rows = long_rows + memora_rows
    seen: set[tuple[str, str]] = set()
    for row in all_rows:
        key = (row["source_dataset"], row["source_example_id"])
        if key in seen:
            raise ValueError(f"Duplicate source ID: {key}")
        seen.add(key)
    candidates = [row for row in all_rows if row["eligibility"]]
    excluded = []
    for row in all_rows:
        if not row["eligibility"]:
            excluded_row = row.copy()
            excluded_row["exclusion_reasons"] = list(row["eligibility_reasons"])
            excluded.append(excluded_row)
    _write_jsonl(output_dir / "longmemeval_candidates.jsonl", [row for row in long_rows if row["eligibility"]])
    _write_jsonl(output_dir / "memora_candidates.jsonl", [row for row in memora_rows if row["eligibility"]])
    _write_jsonl(output_dir / "excluded_examples.jsonl", excluded)
    capability_counts = Counter(cap for row in candidates for cap in row["governance_capabilities"])
    summary = {
        "source_status": source_status,
        "raw_count_per_source": {"longmemeval": len(long_rows), "memora": len(memora_rows)},
        "counts_by_original_task_type": dict(Counter(row.get("question_type", row.get("task_type")) or "unknown" for row in all_rows)),
        "eligible_count_per_source": {"longmemeval": sum(row["eligibility"] for row in long_rows), "memora": sum(row["eligibility"] for row in memora_rows)},
        "excluded_count_per_source": {"longmemeval": sum(not row["eligibility"] for row in long_rows), "memora": sum(not row["eligibility"] for row in memora_rows)},
        "counts_by_governance_capability": dict(capability_counts),
        "counts_by_primary_event_type": dict(Counter(row.get("primary_event_type") for row in candidates)),
        "unclear_example_count": sum(row.get("primary_event_type") == "unclear" for row in candidates),
        "longmemeval_abstention_count": sum(row.get("abstention", False) for row in long_rows),
        "memora_counts_by_period": dict(Counter(row.get("period") or "unknown" for row in memora_rows)),
        "memora_counts_by_persona": dict(Counter(row.get("persona") or "unknown" for row in memora_rows)),
        "memora_counts_by_task": dict(Counter(row.get("task_type") or "unknown" for row in memora_rows)),
        "memora_counts_with_memory_presence": sum(row.get("memory_presence_n", 0) > 0 for row in memora_rows),
        "memora_counts_with_forgetting_absence": sum(row.get("forgetting_absence_n", 0) > 0 for row in memora_rows),
        "dataset_fingerprints_sha256": fingerprints,
        "candidate_count_total": len(candidates),
        "excluded_count_total": len(excluded),
        "duplicate_source_id_checks": {"checked": True, "duplicate_n": 0},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Main dataset source audit")
    print(f"LongMemEval: raw={len(long_rows)} eligible={sum(row['eligibility'] for row in long_rows)} excluded={sum(not row['eligibility'] for row in long_rows)} ({source_status['longmemeval']})")
    print(f"Memora:      raw={len(memora_rows)} eligible={sum(row['eligibility'] for row in memora_rows)} excluded={sum(not row['eligibility'] for row in memora_rows)} ({source_status['memora']})")
    print(f"Total candidates: {len(candidates)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit LongMemEval and Memora sources without inference")
    parser.add_argument("--longmemeval", type=Path, default=DEFAULT_LONGMEMEVAL)
    parser.add_argument("--memora", type=Path, default=DEFAULT_MEMORA)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--allow-missing-source", action="store_true", help="Audit available sources while marking missing sources")
    args = parser.parse_args()
    audit(args.longmemeval, args.memora, args.output_dir, args.allow_missing_source)


if __name__ == "__main__":
    main()
