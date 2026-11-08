"""Deterministically audit OAKS-Novel question transitions.

The auditor treats one document/question pair as one candidate base unit and
never calls a model or expands transitions into independent samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/oaks_novel_main_dataset_audit"
SCHEMA_VERSION = "oaks-novel-main-audit-v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_documents(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"OAKS-Novel release is missing: {path}")
    text = path.read_text(encoding="utf-8")
    value = json.loads(text) if text.lstrip().startswith(("[", "{")) else [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        for key in ("documents", "docs", "data", "examples"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            value = [value]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"OAKS-Novel input must contain document objects: {path}")
    return value


def _meta(doc: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = doc.get("meta")
    if not isinstance(metadata, dict):
        raise ValueError(f"OAKS document {index} is missing the native meta object")
    bid = metadata.get("bid")
    if bid is None or not str(bid).strip():
        raise ValueError(f"OAKS document {index} has missing/empty meta.bid")
    return metadata


def _doc_id(doc: dict[str, Any], index: int) -> str:
    return str(_meta(doc, index)["bid"])


def _data(doc: dict[str, Any], index: int) -> Any:
    if "data" not in doc:
        raise ValueError(f"OAKS document {index} is missing native data")
    return doc["data"]


def _chunks(doc: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    data = doc.get("data", {})
    source = data if isinstance(data, dict) else doc
    raw = source.get("context_chunks", source.get("chunks", source.get("context", [])))
    if isinstance(raw, dict):
        try:
            raw = [{"chunk_id": key, "content": raw[key]} for key in sorted(raw, key=lambda key: int(str(key)))]
        except (TypeError, ValueError) as exc:
            raise ValueError("OAKS chunk IDs must be numerically orderable") from exc
    if not isinstance(raw, list):
        raise ValueError("OAKS document context chunks must be a list or object")
    ids: list[str] = []
    for index, chunk in enumerate(raw):
        if isinstance(chunk, dict):
            value = chunk.get("chunk_id", chunk.get("id", chunk.get("index", index)))
        else:
            value = index
        ids.append(str(value))
    if len(ids) != len(set(ids)):
        raise ValueError("OAKS document has duplicate chunk IDs")
    return ids, {value: index for index, value in enumerate(ids)}


def _questions(doc: dict[str, Any]) -> list[dict[str, Any]]:
    data = doc.get("data")
    if isinstance(data, dict):
        value = data.get("questions", data.get("qas", data.get("question_data", [])))
    elif isinstance(data, list):
        value = data
    else:
        value = []
    if isinstance(value, dict):
        output = []
        for key, question in value.items():
            if not isinstance(question, dict):
                raise ValueError("OAKS question map values must be objects")
            if not question.get("question_id"):
                raise ValueError(f"OAKS question {key!r} is missing question_id")
            output.append({**question, "question": key, "question_id": question["question_id"]})
        return output
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("OAKS questions must be a list or object map")
    return value


def _answer_mapping(question: dict[str, Any]) -> Any:
    for key in ("chunk_to_answer", "chunk_to_answers", "answers_by_chunk"):
        if key in question:
            return question[key]
    return None


def _native_timeline(question: dict[str, Any], chunk_order: dict[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Reconstruct active answers from option_sources without semantic guessing."""
    option_sources = question.get("option_sources")
    if not isinstance(option_sources, dict):
        return [], [], False
    by_chunk: dict[int, list[dict[str, Any]]] = {}
    for option, source_map in option_sources.items():
        if not isinstance(source_map, dict):
            raise ValueError(f"Question {question.get('question_id')} has invalid option_sources")
        if option == "Previous answer still holds":
            continue
        for chunk, evidence in source_map.items():
            if not isinstance(evidence, list) or not evidence:
                continue
            chunk_id = str(chunk)
            if chunk_id in chunk_order:
                index = chunk_order[chunk_id]
            else:
                try:
                    index = int(chunk_id)
                except ValueError as exc:
                    raise ValueError(f"Question {question.get('question_id')} has unordered chunk {chunk!r}") from exc
            by_chunk.setdefault(index, []).append({"answer": option, "chunk_id": chunk_id, "evidence": evidence})
    holds: set[int] = set()
    previous = option_sources.get("Previous answer still holds")
    if isinstance(previous, dict):
        for chunk, evidence in previous.items():
            if not isinstance(evidence, list):
                raise ValueError(f"Question {question.get('question_id')} has invalid persistence evidence")
            chunk_id = str(chunk)
            index = chunk_order.get(chunk_id)
            if index is None:
                try:
                    index = int(chunk_id)
                except ValueError as exc:
                    raise ValueError(f"Question {question.get('question_id')} has unordered hold chunk {chunk!r}") from exc
            holds.add(index)
    timeline: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for index in sorted(set(by_chunk) | holds):
        explicit = by_chunk.get(index, [])
        if len({item["answer"] for item in explicit}) > 1:
            return [], [], True
        if explicit:
            current = explicit[0]
        elif index in holds and active is not None:
            current = {"answer": active["answer"], "chunk_id": str(index), "evidence": []}
        else:
            continue
        if active is not None and active["answer"] != current["answer"]:
            transitions.append({
                "from_chunk_index": active["chunk_index"], "from_chunk_id": active["chunk_id"],
                "old_answer": active["answer"], "evidence_before": active["evidence"],
                "to_chunk_index": index, "to_chunk_id": current["chunk_id"],
                "new_answer": current["answer"], "evidence_after": current["evidence"],
            })
        active = {"answer": current["answer"], "chunk_id": current["chunk_id"], "chunk_index": index, "evidence": current["evidence"]}
        timeline.append({"chunk_index": index, "chunk_id": current["chunk_id"], "answer": current["answer"], "evidence": current["evidence"], "explicit": bool(explicit)})
    return timeline, transitions, False


def _state_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ordered_states(question: dict[str, Any], chunk_order: dict[str, int]) -> list[dict[str, Any]]:
    mapping = _answer_mapping(question)
    if not isinstance(mapping, (dict, list)):
        return []
    pairs = list(mapping.items()) if isinstance(mapping, dict) else [
        (item.get("chunk_id", item.get("chunk_index")), item.get("answer", item.get("state")))
        for item in mapping if isinstance(item, dict)
    ]
    if any(chunk is None for chunk, _ in pairs):
        raise ValueError(f"Question {question.get('question_id')} has an answer without a chunk identifier")
    keyed: list[tuple[int, str, Any]] = []
    for chunk, answer in pairs:
        chunk_id = str(chunk)
        if chunk_id in chunk_order:
            index = chunk_order[chunk_id]
        else:
            try:
                index = int(chunk_id)
            except ValueError as exc:
                raise ValueError(f"Question {question.get('question_id')} has unordered chunk {chunk!r}") from exc
        keyed.append((index, chunk_id, answer))
    keyed.sort(key=lambda item: item[0])
    if len({item[0] for item in keyed}) != len(keyed):
        raise ValueError(f"Question {question.get('question_id')} maps multiple answers to one chunk")
    return [{"chunk_index": index, "chunk_id": chunk, "answer": answer} for index, chunk, answer in keyed]


def _explicit_temporal(question: dict[str, Any]) -> bool:
    for key in ("temporal", "requires_temporal_reasoning", "temporal_dependency", "chronological", "currentness_required"):
        if question.get(key) is True:
            return True
    qtype = str(question.get("question_type", question.get("task_type", ""))).lower()
    return qtype in {"temporal", "temporal-reasoning", "temporal_reasoning", "currentness"}


def audit_question(doc: dict[str, Any], question: dict[str, Any], document_id: str, chunk_order: dict[str, int], source_fingerprint: str, document_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    question_id = question.get("question_id", question.get("id"))
    if question_id is None or question.get("question") is None:
        raise ValueError(f"OAKS document {document_id} has a question missing question_id/question")
    states, changes, ambiguous = _native_timeline(question, chunk_order)
    explicit_temporal = _explicit_temporal(question)
    has_evidence = bool(states and any(state["evidence"] for state in states))
    reasons: list[str] = []
    if ambiguous:
        reasons.append("multiple_explicit_answers_at_chunk_without_deterministic_winner")
    if not changes:
        reasons.append("no_native_answer_state_change")
    if not has_evidence:
        reasons.append("missing_native_supporting_evidence")
    update_eligible = bool(changes and len(states) >= 2 and has_evidence)
    temporal_eligible = bool((explicit_temporal or len(changes) >= 2) and has_evidence and changes)
    if update_eligible and temporal_eligible:
        event = "mixed"
    elif update_eligible:
        event = "update"
    elif temporal_eligible:
        event = "temporal"
    else:
        event = "unclear"
    transitions = changes
    reconstructed_chunk_answers = {state["chunk_id"]: state["answer"] for state in states if state.get("explicit")}
    row = {
        "source_dataset": "oaks_novel", "document_id": document_id,
        "document_title": (document_meta or {}).get("title"),
        "document_author": (document_meta or {}).get("author"),
        "num_chunks": (document_meta or {}).get("num_chunks"),
        "num_qas": (document_meta or {}).get("num_qas"),
        "source_example_id": f"{document_id}::{question_id}", "question_id": str(question_id),
        "question": question["question"], "options": question.get("options", question.get("answer_options")),
        "primary_event_type": event, "governance_capabilities": sorted(set((["update"] if update_eligible else []) + (["temporal_state"] if temporal_eligible else []))),
        "chunk_to_answer": _answer_mapping(question) if _answer_mapping(question) is not None else reconstructed_chunk_answers, "ordered_answer_states": states,
        "transition_count": len(changes), "transition_chunks": changes,
        "old_current_state_information": changes,
        "old_answer": changes[0]["old_answer"] if changes else None,
        "new_answer": changes[-1]["new_answer"] if changes else None,
        "transition_chunk": changes[0]["to_chunk_id"] if changes else None,
        "previous_valid_interval": [states[0]["chunk_index"], changes[0]["to_chunk_index"] - 1] if changes else None,
        "new_valid_interval": [changes[0]["to_chunk_index"], states[-1]["chunk_index"]] if changes else None,
        "evidence_before": changes[0]["evidence_before"] if changes else None,
        "evidence_after": changes[-1]["evidence_after"] if changes else None,
        "evidence_fields_used": [key for key in ("option_sources", "supporting_evidence", "evidence") if question.get(key)],
        "option_sources": question.get("option_sources"), "eligibility": event != "unclear",
        "eligibility_reason": "Native answer-state changes with ordered chunks and supporting evidence." if event != "unclear" else "; ".join(reasons),
        "source_fingerprint": source_fingerprint, "schema_version": SCHEMA_VERSION,
    }
    return row


def audit(path: Path, output_dir: Path = OUT_DIR, current_deletion: int = 380, current_temporal: int = 113, current_update: int = 78) -> dict[str, Any]:
    documents = load_documents(path)
    fingerprint = _sha(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_question_ids = 0
    for index, doc in enumerate(documents):
        metadata = _meta(doc, index)
        doc_id = _doc_id(doc, index)
        _data(doc, index)
        _, chunk_order = _chunks(doc)
        for question in _questions(doc):
            row = audit_question(doc, question, doc_id, chunk_order, fingerprint, metadata)
            if row["source_example_id"] in seen:
                duplicate_question_ids += 1
                raise ValueError(f"Duplicate OAKS question ID: {row['source_example_id']}")
            seen.add(row["source_example_id"])
            rows.append(row)
    rows.sort(key=lambda row: row["source_example_id"])
    candidates = [row for row in rows if row["eligibility"]]
    excluded = [row for row in rows if not row["eligibility"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "oaks_novel_candidates.jsonl", candidates)
    _write(output_dir / "oaks_novel_excluded.jsonl", excluded)
    counts = Counter(row["primary_event_type"] for row in candidates)
    transition_distribution = Counter(row["transition_count"] for row in rows)
    projected = {"deletion": current_deletion, "temporal": current_temporal + counts.get("temporal", 0) + counts.get("mixed", 0), "update": current_update + counts.get("update", 0) + counts.get("mixed", 0)}
    summary = {
        "raw_document_count": len(documents), "raw_unique_question_count": len(rows),
        "total_answer_transition_count": sum(row["transition_count"] for row in rows),
        "eligible_unique_question_count": len(candidates), "excluded_unique_question_count": len(excluded),
        "counts_by_primary_event_type": dict(counts), "update_capable_count": counts.get("update", 0) + counts.get("mixed", 0),
        "temporal_capable_count": counts.get("temporal", 0) + counts.get("mixed", 0), "mixed_count": counts.get("mixed", 0),
        "unclear_count": sum(row["primary_event_type"] == "unclear" for row in rows),
        "transition_count_distribution": {str(k): v for k, v in sorted(transition_distribution.items())},
        "duplicate_question_id_count": duplicate_question_ids, "missing_question_id_count": 0,
        "dataset_sha256": fingerprint, "output_sha256": None,
        "projected_govmem_event_counts": projected, "current_govmem_event_counts": {"deletion": current_deletion, "temporal": current_temporal, "update": current_update},
        "schema_version": SCHEMA_VERSION,
    }
    output_bytes = b"".join((output_dir / name).read_bytes() for name in ("oaks_novel_candidates.jsonl", "oaks_novel_excluded.jsonl"))
    summary["output_sha256"] = hashlib.sha256(output_bytes).hexdigest()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OAKS-Novel: documents={len(documents)} questions={len(rows)} eligible={len(candidates)} excluded={len(excluded)} transitions={summary['total_answer_transition_count']}")
    print(f"Primary events: {dict(counts)}; projected GovMem counts: {projected}")
    return summary


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit OAKS-Novel native update/temporal evidence without inference")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    audit(args.input, args.output_dir)


if __name__ == "__main__":
    main()
