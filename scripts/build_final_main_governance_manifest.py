"""Build the deterministic final GovMem base cohort (571 + 160 OAKS rows)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLEAR_MANIFEST = ROOT / "data/main_governance_base_manifest.jsonl"
OAKS_CANDIDATES = ROOT / "results/oaks_novel_main_dataset_audit/oaks_novel_candidates.jsonl"
LEGACY = ROOT / "data/governance_fault_rq1_pilot_frozen.jsonl"
OUTPUT = ROOT / "data/final_main_governance_base_manifest.jsonl"
SUMMARY = ROOT / "results/final_main_governance_base_manifest_summary.json"
SEED = 42
SCHEMA = "final-main-governance-base-v1"
OPS = ("missing", "stale", "flipped", "conflicting")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required cohort artifact is missing: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON at {path}:{line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_no}")
        rows.append(row)
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_key(*parts: Any) -> str:
    return hashlib.sha256(":\x1f".join([str(SEED), *(str(part) for part in parts)]).encode()).hexdigest()


def _oaks_support(row: dict[str, Any]) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    states = row.get("ordered_answer_states")
    transitions = row.get("transitions", row.get("transition_chunks", []))
    if not isinstance(states, list) or not states or not isinstance(transitions, list) or not transitions:
        raise ValueError(f"OAKS row lacks a deterministic transition: {row.get('source_example_id')}")
    if any("answer" not in state or "chunk_index" not in state for state in states):
        raise ValueError(f"OAKS row has malformed ordered answer state: {row.get('source_example_id')}")
    oracle = {
        "current_status": "valid",
        "ordered_answer_states": states,
        "transitions": transitions,
        "relevant_chunk_ids": [state.get("chunk_id") for state in states],
    }
    supported = ["missing", "stale", "conflicting"]
    reasons = {
        "missing": "Native answer-state timeline permits omission of the active state at a known chunk.",
        "stale": "Native transition evidence permits retaining the prior answer as current after a replacement.",
        "conflicting": "Native old/new answer states permit simultaneous incompatible state exposure.",
        "flipped": "OAKS exposes answer states and evidence, but no native binary governance field for inversion.",
    }
    return supported, reasons, oracle


def _oaks_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    supported, reasons, oracle = _oaks_support(row)
    source_id = str(row["source_example_id"])
    return {
        "base_id": f"oaks_novel::{source_id}",
        "source_dataset": "oaks_novel",
        "source_example_id": source_id,
        "primary_event_type": row["primary_event_type"],
        "question": row["question"],
        "expected_answer": row.get("expected_answer", row.get("new_answer")),
        "supported_fault_operators": supported,
        "unsupported_fault_operators": {"flipped": reasons["flipped"]},
        "fault_support_reasons": {op: reasons[op] for op in supported},
        "oracle_state": oracle,
        "source_evidence": {
            key: row[key] for key in ("document_id", "document_title", "document_author", "num_chunks", "num_qas", "options", "option_sources", "chunk_to_answer", "ordered_answer_states", "transition_count", "transition_chunks", "transitions", "evidence_fields_used") if key in row
        },
        "source_fingerprint": row.get("source_fingerprint"),
        "manifest_schema_version": SCHEMA,
        "document_id": row.get("document_id"),
        "document_title": row.get("document_title"),
        "document_author": row.get("document_author"),
        "transition_count": row.get("transition_count"),
        "ordered_answer_states": row.get("ordered_answer_states"),
        "transition_metadata": row.get("transitions", row.get("transition_chunks")),
    }


def _select_oaks(rows: list[dict[str, Any]], total: int = 160, mixed_n: int = 110, update_n: int = 50) -> list[dict[str, Any]]:
    if len(rows) < total:
        raise ValueError(f"OAKS has only {len(rows)} candidates; need {total}")
    docs = sorted({str(row.get("document_id")) for row in rows})
    if len(docs) != 19:
        raise ValueError(f"Expected 19 OAKS documents, found {len(docs)}")
    doc_order = sorted(docs, key=lambda doc: _stable_key("document", doc))
    quotas = {doc: total // len(docs) for doc in docs}
    for doc in doc_order[: total % len(docs)]:
        quotas[doc] += 1
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        event = row.get("primary_event_type")
        if event not in {"mixed", "update"}:
            raise ValueError(f"Unsupported OAKS primary event: {event!r}")
        grouped[str(row["document_id"])][event].append(row)
    update_quota = {doc: min(int(quotas[doc] * update_n / total), len(grouped[doc]["update"])) for doc in docs}
    while sum(update_quota.values()) < update_n:
        eligible = [doc for doc in docs if update_quota[doc] < len(grouped[doc]["update"])]
        if not eligible:
            raise ValueError("Cannot satisfy deterministic OAKS update quota")
        doc = max(eligible, key=lambda item: (len(grouped[item]["update"]) - update_quota[item], _stable_key("update", item)))
        update_quota[doc] += 1
    def take(items: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            count = int(item.get("transition_count", 0))
            buckets["1" if count == 1 else "2" if count == 2 else "3" if count == 3 else "4+"] .append(item)
        for bucket in buckets.values():
            bucket.sort(key=lambda item: _stable_key("candidate", item["source_example_id"]))
        chosen: list[dict[str, Any]] = []
        bucket_names = sorted(buckets)
        while len(chosen) < n:
            progressed = False
            for bucket in bucket_names:
                if buckets[bucket]:
                    chosen.append(buckets[bucket].pop(0))
                    progressed = True
                    if len(chosen) == n:
                        break
            if not progressed:
                raise ValueError("OAKS document/event stratum cannot satisfy quota")
        return chosen
    selected: list[dict[str, Any]] = []
    for doc in docs:
        selected.extend(take(grouped[doc]["update"], update_quota[doc]))
        selected.extend(take(grouped[doc]["mixed"], quotas[doc] - update_quota[doc]))
    selected.sort(key=lambda item: item["source_example_id"])
    if len(selected) != total or Counter(item["primary_event_type"] for item in selected) != Counter({"mixed": mixed_n, "update": update_n}):
        raise ValueError("OAKS selection failed exact size/event quotas")
    if {item["document_id"] for item in selected} != set(docs):
        raise ValueError("OAKS selection omitted a document")
    return selected


def build(output: Path = OUTPUT, summary_path: Path = SUMMARY, force: bool = False) -> dict[str, Any]:
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite final manifest without --force: {output}")
    clear = _read_jsonl(CLEAR_MANIFEST)
    oaks = _read_jsonl(OAKS_CANDIDATES)
    if len(clear) != 571:
        raise ValueError(f"Expected 571 existing clear bases, found {len(clear)}")
    selected_oaks = _select_oaks(oaks)
    legacy_ids = {(row.get("source_dataset"), row.get("source_example_id")) for row in _read_jsonl(LEGACY)}
    all_rows = clear + [_oaks_manifest_row(row) for row in selected_oaks]
    base_ids = [row.get("base_id") for row in all_rows]
    source_ids = [(row.get("source_dataset"), row.get("source_example_id")) for row in all_rows]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("Duplicate final base IDs")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate final source IDs")
    if any(pair in legacy_ids for pair in source_ids):
        raise ValueError("Legacy 100-example source reused")
    all_rows.sort(key=lambda row: row["base_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in all_rows), encoding="utf-8")
    event_counts = Counter(row["primary_event_type"] for row in all_rows)
    oaks_events = Counter(row["primary_event_type"] for row in selected_oaks)
    operator_counts = Counter(op for row in all_rows for op in row["supported_fault_operators"])
    distribution = Counter(len(row["supported_fault_operators"]) for row in all_rows)
    conservative_events = {"deletion": event_counts.get("deletion", 0), "temporal": event_counts.get("temporal", 0), "update": event_counts.get("update", 0)}
    capability_events = {"deletion": event_counts.get("deletion", 0), "temporal": event_counts.get("temporal", 0) + event_counts.get("mixed", 0), "update": event_counts.get("update", 0) + event_counts.get("mixed", 0)}
    summary = {
        "total_final_base_count": len(all_rows), "base_count_by_source": dict(Counter(row["source_dataset"] for row in all_rows)),
        "base_count_by_primary_event_type": dict(event_counts), "oaks_count_by_document": dict(Counter(row["document_id"] for row in selected_oaks)),
        "oaks_count_by_primary_event_type": dict(oaks_events), "oaks_transition_count_distribution": dict(Counter(str(row.get("transition_count")) for row in selected_oaks)),
        "supported_count_by_fault_operator": dict(operator_counts), "supported_operator_count_distribution": {str(k): v for k, v in sorted(distribution.items())},
        "number_supporting_exactly_1_operator": distribution.get(1, 0), "number_supporting_exactly_2_operators": distribution.get(2, 0), "number_supporting_exactly_3_operators": distribution.get(3, 0), "number_supporting_exactly_4_operators": distribution.get(4, 0),
        "duplicate_base_id_count": 0, "duplicate_source_id_count": 0, "manifest_sha256": _sha(output),
        "source_dataset_fingerprints": {"existing_clear_manifest": _sha(CLEAR_MANIFEST), "oaks_candidates": _sha(OAKS_CANDIDATES)},
        "schema_version": SCHEMA, "seed": SEED, "oaks_selection_rule": "160 questions, per-document 8/9 quota, deterministic hash tie-breaks, event quota 110 mixed/50 update, round-robin transition buckets 1/2/3/4+",
        "thresholds_by_event_type": {event: {str(n): event_counts.get(event, 0) >= n for n in (100, 150, 200)} for event in ("deletion", "temporal", "update")},
        "conservative_primary_event_counts": conservative_events,
        "capability_projected_event_counts": capability_events,
        "thresholds_by_capability_projection": {event: {str(n): capability_events[event] >= n for n in (100, 150, 200)} for event in capability_events},
        "thresholds_by_fault_operator": {op: {str(n): operator_counts.get(op, 0) >= n for n in (100, 150, 200)} for op in OPS},
        "oaks_selection_reproducible": True,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Final main cohort: {len(all_rows)} bases; sources={summary['base_count_by_source']}; events={dict(event_counts)}")
    print(f"Fault support: {dict(operator_counts)}; operator-count distribution={dict(sorted(distribution.items()))}")
    print(f"Manifest SHA256: {summary['manifest_sha256']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final deterministic GovMem main base cohort")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.output, args.summary, args.force)


if __name__ == "__main__":
    main()
