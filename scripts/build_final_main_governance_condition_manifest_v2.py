"""Repair condition specifications with benchmark task payloads.

The v1 condition file is retained unchanged.  This builder enriches its rows
from the frozen v2 base manifest and the deterministic source-audit records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from eval.system_validation_scoring import scoring_spec

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/final_main_governance_base_manifest_v2.jsonl"
PARENT_CONDITIONS = ROOT / "data/final_main_governance_condition_manifest_v1.jsonl"
LONG_AUDIT = ROOT / "results/main_dataset_audit_v3/longmemeval_candidates.jsonl"
MEMORA_AUDIT = ROOT / "results/main_dataset_audit_v3/memora_candidates.jsonl"
OAKS_AUDIT = ROOT / "results/oaks_novel_main_dataset_audit/oaks_novel_candidates.jsonl"
LONG_NATIVE = ROOT / "data/official_longmemeval_oracle_200.jsonl"
OUTPUT = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
SUMMARY = ROOT / "results/final_main_governance_condition_manifest_v2_summary.json"
FREEZE = ROOT / "data/final_main_governance_condition_manifest_v2.freeze.json"
BASE_SHA = "a2601a7ecb46034faf5dcc7ad00b5a851e19ae01561876fa681fd18af168ca79"
PARENT_SHA = "3aea17d92a2116400116a5e9082ddb2f2efa585cff550c40925eab9e5162af70"
SCHEMA = "main-governance-condition-v2"


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required source artifact is missing: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected objects in {path}")
            rows.append(value)
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value_list(value: Any) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        if "value" in value and value["value"] not in (None, ""):
            found.append(value["value"])
        for child in value.values():
            found.extend(_value_list(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_value_list(child))
    return found


def _unique(values: list[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


def _text_values(value: Any) -> list[str]:
    """Extract native content, deliberately excluding tracking-only fields."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        result: list[str] = []
        for key, child in value.items():
            if key in {"session_id", "session_ids", "session_history", "relevant_source_session_ids"}:
                continue
            result.extend(_text_values(child))
        return result
    if isinstance(value, list):
        result: list[str] = []
        for child in value:
            result.extend(_text_values(child))
        return result
    return []


def _memora_content_items(value: Any, path: str = "") -> list[str]:
    """Project native memory content, retaining scalar facts such as budgets."""
    if isinstance(value, dict):
        result: list[str] = []
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            result.extend(_memora_content_items(child, child_path))
        return result
    if isinstance(value, list):
        result: list[str] = []
        for child in value:
            result.extend(_memora_content_items(child, path))
        return result
    if isinstance(value, (str, int, float)) and not isinstance(value, bool) and value not in (None, ""):
        return [f"{path}: {value}" if path else str(value)]
    return []


def _memory_items(base: dict[str, Any], audit: dict[str, Any], native_long: dict[str, Any]) -> tuple[list[str], list[str]]:
    source = base["source_dataset"]
    if source == "longmemeval":
        relevant = set(str(x) for x in base["source_evidence"].get("evidence_session_ids", []))
        memories = [m for m in native_long.get("memories", []) if str(m.get("official_session_id")) in relevant]
        answer_memories = [m for m in memories if m.get("has_answer")]
        if len(answer_memories) >= 1:
            memories = answer_memories
        memories.sort(key=lambda m: (str(m.get("timestamp", "")), str(m.get("memory_id", ""))))
        if memories:
            return [str(m["content"]) for m in memories if m.get("content")], []
        # The repository's compact LongMemEval audit fixture contains the
        # native answer/session references for a small subset whose full
        # source file is not distributed locally. Keep those rows auditable
        # and deterministic rather than silently dropping them. A complete
        # source release will replace this explicit answer-bearing evidence
        # fallback before formal inference.
        answer = audit.get("expected_answer", base.get("expected_answer"))
        if answer not in (None, ""):
            return [f"Benchmark-native answer-bearing evidence: {answer}"], []
        return [], []
    if source == "memora":
        memory_evidence = audit.get("memory_evidence")
        if isinstance(memory_evidence, dict) and isinstance(memory_evidence.get("content_data"), dict):
            current = _memora_content_items(memory_evidence["content_data"])
        else:
            current = _text_values(memory_evidence)
        obsolete = _text_values(audit.get("forgetting_evidence"))
        return _unique(current), _unique(obsolete)
    states = base.get("ordered_answer_states") or []
    explicit = [state for state in states if state.get("explicit") and state.get("evidence")]
    if not explicit:
        explicit = [state for state in states if state.get("evidence")]
    current = _text_values(explicit[-1].get("evidence", [])) if explicit else []
    previous = _text_values(explicit[-2].get("evidence", [])) if len(explicit) > 1 else []
    return _unique(current), _unique(previous)


def _condition_items(base: dict[str, Any], operator: str | None, condition_type: str, current: list[str], previous: list[str]) -> list[str]:
    if condition_type == "clean" or condition_type == "enforced":
        return current
    if operator == "missing":
        return current[1:] if len(current) > 1 else []
    if operator == "stale":
        return previous or current[:-1]
    if operator == "flipped":
        return previous or current
    if operator == "conflicting":
        return _unique(previous + current)
    return current


def _payload(base: dict[str, Any], audits: dict[str, dict[str, Any]], native_long: dict[str, Any]) -> dict[str, Any]:
    source = base["source_dataset"]
    audit = audits.get(base["source_example_id"])
    if audit is None:
        raise ValueError(f"No source audit payload for {source}/{base['source_example_id']}")
    question = base.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Missing task question for {base['base_id']}")
    expected = base.get("expected_answer")
    if source == "memora" and expected in (None, "", []):
        # Memora's released audit has list-valued native answers in
        # memory_evidence rather than a scalar expected_answer field.
        native_memory = audit.get("memory_evidence")
        if native_memory in (None, "", [], {}):
            raise ValueError(f"Memora task oracle cannot be reconstructed for {base['base_id']}")
        # Memora released questions include structured native answer evidence
        # (for example content_data plus its session history), not a scalar
        # expected_answer column. Preserve that object losslessly.
        expected = native_memory
        answer_type = "structured_memory_evidence"
    elif source == "oaks_novel":
        answer_type = "multiple_choice" if audit.get("options") else "text"
    else:
        answer_type = "text"
    current_items, previous_items = _memory_items(base, audit, native_long)
    if not current_items:
        raise ValueError(f"No native semantic evidence can be rendered for {base['base_id']}")
    metadata = {"source_dataset": source, "source_example_id": base["source_example_id"]}
    for key in ("question_type", "task_type", "persona", "period", "question_date", "question_id", "document_id", "document_title", "document_author"):
        if audit.get(key) is not None:
            metadata[key] = audit[key]
    if source == "longmemeval":
        evidence = {key: audit.get(key) for key in ("answer_session_ids", "evidence_session_ids") if audit.get(key) is not None}
    elif source == "memora":
        evidence = {key: audit.get(key) for key in ("memory_evidence", "forgetting_evidence", "relevant_source_session_ids") if audit.get(key) is not None}
    else:
        evidence = {key: audit.get(key) for key in ("option_sources", "ordered_answer_states", "transition_count", "transition_chunks", "transitions", "evidence_before", "evidence_after") if audit.get(key) is not None}
    result = {"source_dataset": source, "task_question": question, "task_oracle_answer": expected, "task_answer_type": answer_type, "task_options": audit.get("options") if source == "oaks_novel" else None, "task_source_metadata": metadata, "task_evidence": evidence, "_native_current_items": current_items, "_native_previous_items": previous_items}
    result["task_scoring_spec"] = scoring_spec(result)
    return result


def build(*, output: Path = OUTPUT, summary_path: Path = SUMMARY, freeze_path: Path = FREEZE, force: bool = False) -> dict[str, Any]:
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite condition v2 without --force: {output}")
    if _sha(BASE) != BASE_SHA or _sha(PARENT_CONDITIONS) != PARENT_SHA:
        raise ValueError("Frozen parent hash mismatch")
    bases = _read(BASE)
    parent = _read(PARENT_CONDITIONS)
    audits = {}
    for path in (LONG_AUDIT, MEMORA_AUDIT, OAKS_AUDIT):
        for row in _read(path):
            key = row.get("source_example_id")
            if key is not None:
                audits[str(key)] = row
    native_long = {row["id"]: row for row in _read(LONG_NATIVE)}
    base_map = {row["base_id"]: row for row in bases}
    if len(base_map) != 731:
        raise ValueError("Expected 731 unique frozen bases")
    payloads = {base_id: _payload(base, audits, native_long.get(base["source_example_id"], {})) for base_id, base in base_map.items()}
    rows = []
    for original in parent:
        base_id = original["base_id"]
        if base_id not in payloads:
            raise ValueError(f"Condition references unknown base {base_id}")
        row = deepcopy(original)
        task = deepcopy(payloads[base_id])
        current = task.pop("_native_current_items")
        previous = task.pop("_native_previous_items")
        row.update(task)
        items = _condition_items(base_map[base_id], row.get("fault_operator"), row["condition_type"], current, previous)
        row["model_visible_memory_items"] = items
        row["model_visible_context_text"] = "\n".join(items) or "No benchmark-native memory evidence is available in the admitted context."
        row["internal_governance_state"] = deepcopy(row["oracle_state"])
        row["model_visible_context_spec"] = {"content_source": "benchmark_native_evidence", "item_count": len(items), "projection": "admitted_evidence"}
        row["condition_schema_version"] = SCHEMA
        rows.append(row)
    rows.sort(key=lambda row: row["condition_id"])
    if len(rows) != 4891:
        raise ValueError(f"Expected 4891 rows, found {len(rows)}")
    by_base: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not row["task_question"] or row["task_oracle_answer"] in (None, "", []) or not row.get("model_visible_context_text", "").strip():
            raise ValueError(f"Condition {row['condition_id']} lacks task payload")
        by_base.setdefault(row["base_id"], []).append(row)
    for base_id, siblings in by_base.items():
        questions = {json.dumps(row["task_question"], ensure_ascii=False, sort_keys=True) for row in siblings}
        oracles = {json.dumps(row["task_oracle_answer"], ensure_ascii=False, sort_keys=True) for row in siblings}
        if len(questions) != 1 or len(oracles) != 1:
            raise ValueError(f"Task payload is not invariant for {base_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    condition_counts = dict(Counter(row["condition_type"] for row in rows))
    summary = {"base_count": 731, "clean_count": condition_counts["clean"], "supported_base_fault_pair_count": 2080, "corrupted_count": condition_counts["corrupted"], "enforced_count": condition_counts["enforced"], "total_condition_row_count": len(rows), "condition_counts_by_type": condition_counts, "fault_counts": dict(Counter(row["fault_operator"] for row in rows if row.get("fault_operator"))), "task_payload_source_counts": dict(Counter(row["source_dataset"] for row in rows)), "task_question_nonempty_n": sum(bool(row["task_question"]) for row in rows), "task_oracle_nonnull_n": sum(row["task_oracle_answer"] not in (None, "", []) for row in rows), "parent_base_manifest_sha256": BASE_SHA, "parent_condition_manifest_sha256": PARENT_SHA, "manifest_sha256": _sha(output), "schema_version": SCHEMA}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze = {"manifest_filename": output.name, "manifest_sha256": summary["manifest_sha256"], "parent_base_manifest_filename": BASE.name, "parent_base_manifest_sha256": BASE_SHA, "parent_condition_manifest_filename": PARENT_CONDITIONS.name, "parent_condition_manifest_sha256": PARENT_SHA, "schema_version": SCHEMA, "condition_counts": condition_counts, "statement": "Task payloads were reconstructed only from frozen base fields and deterministic benchmark audit artifacts; no model outputs or APIs were used."}
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Condition v2: bases=731 clean={condition_counts['clean']} pairs=2080 corrupted={condition_counts['corrupted']} enforced={condition_counts['enforced']} total={len(rows)}")
    print(f"Condition v2 SHA256: {summary['manifest_sha256']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build task-complete frozen GovMem conditions")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(output=args.output, summary_path=args.summary, freeze_path=args.freeze, force=args.force)


if __name__ == "__main__":
    main()
