"""Assign mutually exclusive analysis event types to the frozen v1 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "data/final_main_governance_base_manifest.jsonl"
OUTPUT = ROOT / "data/final_main_governance_base_manifest_v2.jsonl"
SUMMARY = ROOT / "results/final_main_governance_base_manifest_v2_summary.json"
FREEZE = ROOT / "data/final_main_governance_base_manifest_v2.freeze.json"
SCHEMA = "final-main-governance-base-v2"
SEED = 42


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest is missing: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object in {path}")
            rows.append(value)
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(*parts: Any) -> str:
    return hashlib.sha256("\x1f".join([str(SEED), *(str(part) for part in parts)]).encode()).hexdigest()


def _assign_mixed(rows: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    mixed = [row for row in rows if row.get("primary_event_type") == "mixed"]
    if len(mixed) != 110:
        raise ValueError(f"Expected 110 mixed OAKS rows, found {len(mixed)}")
    assignments: dict[str, tuple[str, str]] = {}
    ambiguous_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mixed:
        count = int(row.get("transition_count", 0))
        if count >= 4:
            assignments[row["base_id"]] = ("temporal", "evidence_preferred_temporal")
        elif count == 2:
            assignments[row["base_id"]] = ("update", "evidence_preferred_update")
        else:
            ambiguous_by_doc[str(row.get("document_id"))].append(row)
    # Transition-count==3 is the explicitly predeclared ambiguity stratum.
    # Allocate 23 of its 37 rows temporally, proportionally by document, then
    # use stable hashes for remainder/ties.  This keeps the 8/9-per-document
    # OAKS cohort document-balanced while targeting 56/54 mixed assignments.
    need_temporal = 23
    quotas = {doc: (len(items) * need_temporal) // sum(len(v) for v in ambiguous_by_doc.values()) for doc, items in ambiguous_by_doc.items()}
    while sum(quotas.values()) < need_temporal:
        eligible = [doc for doc in ambiguous_by_doc if quotas[doc] < len(ambiguous_by_doc[doc])]
        doc = max(eligible, key=lambda item: (len(ambiguous_by_doc[item]) * need_temporal / 37 - quotas[item], _key("doc", item)))
        quotas[doc] += 1
    for doc, items in ambiguous_by_doc.items():
        for row in sorted(items, key=lambda item: _key("mixed-transition-3", item["base_id"]))[: quotas[doc]]:
            assignments[row["base_id"]] = ("temporal", "deterministic_balance_tiebreak")
        for row in items:
            assignments.setdefault(row["base_id"], ("update", "deterministic_balance_tiebreak"))
    if Counter(value[0] for value in assignments.values()) != Counter({"temporal": 56, "update": 54}):
        raise ValueError("Mixed analysis assignment did not meet deterministic target")
    return assignments


def build(parent: Path = PARENT, output: Path = OUTPUT, summary_path: Path = SUMMARY, freeze_path: Path = FREEZE, force: bool = False) -> dict[str, Any]:
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite v2 manifest without --force: {output}")
    rows = _read(parent)
    if len(rows) != 731:
        raise ValueError(f"Expected 731 parent rows, found {len(rows)}")
    base_ids = [row.get("base_id") for row in rows]
    source_ids = [(row.get("source_dataset"), row.get("source_example_id")) for row in rows]
    if len(base_ids) != len(set(base_ids)) or len(source_ids) != len(set(source_ids)):
        raise ValueError("Parent manifest has duplicate base/source IDs")
    assignments = _assign_mixed(rows)
    output_rows = []
    for original in rows:
        row = deepcopy(original)
        primary = row.get("primary_event_type")
        if primary in {"deletion", "temporal", "update"}:
            analysis, reason = primary, "direct_inheritance_from_primary_event_type"
        elif primary == "mixed":
            analysis, reason = assignments[row["base_id"]]
        else:
            raise ValueError(f"Unsupported primary event type: {primary!r}")
        row["analysis_event_type"] = analysis
        row["analysis_event_assignment_reason"] = reason
        row["analysis_event_assignment_rule_version"] = SCHEMA
        output_rows.append(row)
    output_rows.sort(key=lambda row: row["base_id"])
    if len(output_rows) != 731 or any(row["analysis_event_type"] not in {"deletion", "temporal", "update"} for row in output_rows):
        raise ValueError("Invalid v2 analysis-event assignment")
    counts = Counter(row["analysis_event_type"] for row in output_rows)
    if min(counts.values()) < 150 or sum(counts.values()) != 731:
        raise ValueError(f"Analysis buckets do not satisfy minimum size: {counts}")
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    mixed = [row for row in output_rows if row["primary_event_type"] == "mixed"]
    operator_counts = Counter(op for row in output_rows for op in row.get("supported_fault_operators", []))
    summary = {
        "total_base_count": len(output_rows), "source_counts": dict(Counter(row["source_dataset"] for row in output_rows)),
        "primary_event_type_counts": dict(Counter(row["primary_event_type"] for row in output_rows)), "analysis_event_type_counts": dict(counts),
        "mixed_assignment_counts": dict(Counter(row["analysis_event_type"] for row in mixed)), "mixed_assignment_reason_counts": dict(Counter(row["analysis_event_assignment_reason"] for row in mixed)),
        "oaks_mixed_assignment_by_document": {doc: dict(Counter(row["analysis_event_type"] for row in mixed if row.get("document_id") == doc)) for doc in sorted({row.get("document_id") for row in mixed})},
        "oaks_mixed_assignment_by_transition_count_bucket": {bucket: dict(Counter(row["analysis_event_type"] for row in mixed if ("1" if row.get("transition_count") == 1 else "2" if row.get("transition_count") == 2 else "3" if row.get("transition_count") == 3 else "4+") == bucket)) for bucket in ("1", "2", "3", "4+")},
        "fault_support_counts": dict(operator_counts), "duplicate_base_id_count": 0, "duplicate_source_id_count": 0,
        "manifest_sha256": _sha(output), "parent_manifest_sha256": _sha(parent), "schema_version": SCHEMA,
        "thresholds": {event: {str(n): counts[event] >= n for n in (100, 150, 200)} for event in ("deletion", "temporal", "update")},
        "assignment_rule": "non-mixed direct inheritance; mixed transition_count>=4 temporal, ==2 update, ==3 proportional per-document deterministic hash balancing (23 temporal/14 update in that stratum)",
        "seed": SEED,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze = {
        "manifest_filename": output.name, "manifest_sha256": summary["manifest_sha256"], "parent_manifest_sha256": summary["parent_manifest_sha256"],
        "schema_version": SCHEMA, "source_dataset_fingerprints": {"parent_manifest": _sha(parent)}, "source_counts": summary["source_counts"],
        "analysis_event_counts": summary["analysis_event_type_counts"], "fault_operator_counts": summary["fault_support_counts"],
        "assignment_rule": summary["assignment_rule"], "seed": SEED, "statement": "No model outputs or performance results were used in dataset selection or analysis-event assignment.",
    }
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Final v2 analysis events: {dict(counts)}")
    print(f"Mixed temporal/update: {summary['mixed_assignment_counts']}")
    print(f"Fault support: {dict(operator_counts)}")
    print(f"Manifest SHA256: {summary['manifest_sha256']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze mutually exclusive GovMem analysis event types")
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.parent, args.output, args.summary, args.freeze, args.force)


if __name__ == "__main__":
    main()
