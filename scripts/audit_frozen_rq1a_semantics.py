#!/usr/bin/env python3
"""Read-only semantic audit of the frozen governance-fault RQ1-A artifact."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.gatemem_adapter import GateMemDatasetAdapter
from eval.memconflict_adapter import MemConflictDatasetAdapter
from eval.memora_adapter import MemoraDatasetAdapter
from govmem.governance.oracle import derive_gatemem_authorization, derive_memconflict_temporal


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _same_item(left: Any, right: Any) -> bool:
    if left == right:
        return True
    if isinstance(left, dict) and not isinstance(right, dict):
        return str(right) in {str(v) for v in left.values()}
    if isinstance(right, dict) and not isinstance(left, dict):
        return str(left) in {str(v) for v in right.values()}
    return False


def _operation_item(operation: dict[str, Any]) -> Any:
    return (operation.get("operation_details") or {}).get("item")


def _visible(operation: dict[str, Any], question_time: datetime | None) -> bool:
    if question_time is None or not operation.get("date"):
        return True
    try:
        timestamp = datetime.fromisoformat(str(operation["date"]).replace("Z", "+00:00"))
    except ValueError:
        return False
    if timestamp.tzinfo is None and question_time.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=question_time.tzinfo)
    if question_time.tzinfo is None and timestamp.tzinfo is not None:
        question_time = question_time.replace(tzinfo=timestamp.tzinfo)
    return timestamp <= question_time


def _query_id(row: dict[str, Any]) -> str | None:
    source = row.get("source_metadata") or {}
    return source.get("query_id")


def _reason(record: dict[str, Any], reason: str) -> None:
    record.setdefault("invalid_reasons", []).append(reason)


def _audit_gatemem(row: dict[str, Any], source: Any, record: dict[str, Any]) -> None:
    gold = source.evaluator_metadata.get("gold", {}) or {}
    if source.query.metadata.get("query_type") != "privacy":
        _reason(record, "not_privacy_checkpoint")
    if str(gold.get("expected_action", "")).strip().lower() != "refuse":
        _reason(record, "native_expected_action_is_not_refuse")
    stored_state = (row.get("gold_governance_state") or {}).get("authorization") or {}
    if stored_state.get("decision") != "deny" or row.get("gold_admission_decision") != "deny":
        _reason(record, "stored_authorization_gold_is_not_deny")
    stored_targets = (row.get("source_metadata") or {}).get("leak_targets", [])
    native_targets = gold.get("leak_targets") or []
    if stored_targets != native_targets:
        _reason(record, "stored_leak_targets_mismatch_native_source")
    derivation = derive_gatemem_authorization(source)
    if not derivation.supported or derivation.example is None:
        _reason(record, "corrected_oracle_does_not_support_source_example")
    elif _plain(row.get("memory")) != _plain(derivation.example.memory):
        _reason(record, "stored_memory_mismatches_native_privacy_evidence")


def _audit_memora(row: dict[str, Any], source: Any, record: dict[str, Any]) -> None:
    if source.query.metadata.get("task_type") != "remembering":
        _reason(record, "not_remembering_lifecycle_task")
    stored_source = row.get("source_metadata") or {}
    item = stored_source.get("item")
    operations = list(source.evaluator_metadata.get("session_operations", ()))
    adds = [op for op in operations if op.get("operation") == "add" and _same_item(_operation_item(op), item)]
    deletes = [
        op for op in operations
        if op.get("operation") in {"delete", "remove", "forget"}
        and _same_item(_operation_item(op), item)
        and _visible(op, source.query.current_time)
    ]
    state = (row.get("gold_governance_state") or {}).get("deletion") or {}
    status = state.get("status")
    if len(adds) != 1:
        _reason(record, "no_unique_matching_native_add")
    stored_lifecycle = stored_source.get("lifecycle") or []
    for lifecycle_entry in stored_lifecycle:
        matching = [
            operation for operation in operations
            if operation.get("operation") == lifecycle_entry.get("operation")
            and str(operation.get("session_id")) == str(lifecycle_entry.get("session_id"))
            and _same_item(_operation_item(operation), lifecycle_entry.get("item"))
        ]
        if len(matching) != 1:
            _reason(record, "stored_lifecycle_mismatches_native_operations")
            break
    if status == "deleted":
        if len(deletes) != 1:
            _reason(record, "no_unique_visible_matching_native_delete")
        if row.get("gold_admission_decision") != "deny":
            _reason(record, "stored_deleted_item_admission_is_not_deny")
        if deletes and state.get("epoch") is not None and str(state["epoch"]) != str(deletes[0].get("session_id")):
            _reason(record, "stored_delete_epoch_mismatches_native_delete_session")
    elif status == "active":
        if deletes:
            _reason(record, "visible_native_delete_exists_for_stored_active_item")
        if row.get("gold_admission_decision") != "allow":
            _reason(record, "stored_active_item_admission_is_not_allow")
    else:
        _reason(record, "stored_deletion_state_is_invalid")
    if adds and not any(_same_item(_operation_item(op), item) for op in adds):
        _reason(record, "stored_item_mismatches_native_add")


def _attribute_matches_question(attribute: str, question: str) -> bool:
    import re
    phrase = re.sub(r"[_-]+", " ", attribute).lower()
    return phrase in question.lower()


def _audit_memconflict(row: dict[str, Any], source: Any, record: dict[str, Any]) -> None:
    if source.query.metadata.get("conflict_type") != "dynamic":
        _reason(record, "not_dynamic_conflict")
    updates = list((source.evaluator_metadata.get("gold", {}) or {}).get("updated_attributes", ()))
    matches = []
    for update in updates:
        for attribute in update.get("attributes", ()):
            name = str(attribute.get("Attribute", ""))
            if name and _attribute_matches_question(name, source.query.text):
                matches.append((update, attribute))
    if len(matches) != 1:
        _reason(record, "question_does_not_link_to_exactly_one_updated_attribute")
        return
    update, attribute = matches[0]
    metadata = row.get("source_metadata") or {}
    if metadata.get("attribute") != attribute.get("Attribute"):
        _reason(record, "stored_attribute_mismatches_relevant_updated_attribute")
    if metadata.get("current_value") != attribute.get("After"):
        _reason(record, "stored_current_value_mismatches_relevant_updated_attribute")
    if metadata.get("previous_value") != attribute.get("Before"):
        _reason(record, "stored_previous_value_mismatches_relevant_updated_attribute")
    memory = (row.get("memory") or [{}])[0]
    if memory.get("attribute") != attribute.get("Attribute"):
        _reason(record, "stored_memory_attribute_mismatch")
    if memory.get("current_value") != attribute.get("After") or memory.get("previous_value") != attribute.get("Before"):
        _reason(record, "stored_memory_values_mismatch")
    state = (row.get("gold_governance_state") or {}).get("temporal") or {}
    if state.get("status") != "valid" or row.get("gold_admission_decision") != "allow":
        _reason(record, "stored_temporal_gold_is_not_valid_allow")
    if str(metadata.get("current_session", {}).get("session_id")) != str(update.get("session_id")):
        _reason(record, "stored_current_session_mismatch")
    history = (row.get("source_metadata") or {}).get("historical_temporal_states") or []
    if not history:
        _reason(record, "stored_temporal_predecessor_is_missing")


def _audit_clean_row(row: dict[str, Any], source: Any | None) -> dict[str, Any]:
    record = {
        "example_id": row.get("example_id"),
        "source_dataset": row.get("source_dataset"),
        "dimension": row.get("dimension"),
        "query_id": _query_id(row),
        "status": "valid",
        "invalid_reasons": [],
    }
    if source is None:
        _reason(record, "source_example_not_found")
        return record
    if row.get("task") != source.query.text:
        _reason(record, "stored_task_mismatches_source_query")
    if row.get("source_dataset") == "gatemem":
        _audit_gatemem(row, source, record)
    elif row.get("source_dataset") == "memora":
        _audit_memora(row, source, record)
    elif row.get("source_dataset") == "memconflict":
        _audit_memconflict(row, source, record)
    else:
        _reason(record, "unsupported_dataset")
    if record["invalid_reasons"]:
        record["status"] = "invalid"
    return record


def _pair_audit(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("example_id"))].append(row)
    invalid: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for example_id, group in groups.items():
        clean = next((row for row in group if row.get("condition_type") == "clean"), None)
        if clean is None or len(group) != 3:
            record = {"example_id": example_id, "source_dataset": clean.get("source_dataset") if clean else None, "status": "invalid", "invalid_reasons": ["fault_pair_must_have_one_clean_and_two_fault_rows"]}
            invalid.append(record)
            reasons.update(record["invalid_reasons"])
            continue
        for row in group:
            if row is clean:
                continue
            for field in ("task", "memory", "gold_governance_state", "gold_admission_decision", "source_metadata"):
                if row.get(field) != clean.get(field):
                    invalid.append({"example_id": example_id, "source_dataset": clean.get("source_dataset"), "status": "invalid", "invalid_reasons": [f"fault_pair_{field}_mismatch"]})
                    reasons[f"fault_pair_{field}_mismatch"] += 1
                    break
            spec = row.get("fault_spec") or {}
            allowed = {
                "authorization": {"missing", "flipped", "conflicting"},
                "deletion": {"missing", "stale", "flipped"},
                "temporal": {"missing", "stale", "conflicting"},
            }
            if spec.get("operator") not in allowed.get(row.get("dimension"), set()):
                invalid.append({"example_id": example_id, "source_dataset": clean.get("source_dataset"), "status": "invalid", "invalid_reasons": ["unsupported_fault_operator"]})
                reasons["unsupported_fault_operator"] += 1
            if spec.get("operator") == "stale":
                key = "historical_deletion_states" if row.get("dimension") == "deletion" else "historical_temporal_states"
                if not (row.get("source_metadata") or {}).get(key):
                    invalid.append({"example_id": example_id, "source_dataset": clean.get("source_dataset"), "status": "invalid", "invalid_reasons": ["stale_fault_lacks_predecessor"]})
                    reasons["stale_fault_lacks_predecessor"] += 1
            if row.get("observed_governance_state") == clean.get("observed_governance_state"):
                invalid.append({"example_id": example_id, "source_dataset": clean.get("source_dataset"), "status": "invalid", "invalid_reasons": ["fault_does_not_change_observed_state"]})
                reasons["fault_does_not_change_observed_state"] += 1
            dimension = row.get("dimension")
            observed = row.get("observed_governance_state") or {}
            if dimension == "authorization":
                auth = observed.get("authorization")
                assertions = observed.get("authorization_assertions") or []
                if spec.get("operator") == "missing" and auth is not None:
                    invalid.append({"example_id": example_id, "source_dataset": clean.get("source_dataset"), "status": "invalid", "invalid_reasons": ["authorization_missing_fault_not_missing"]})
                    reasons["authorization_missing_fault_not_missing"] += 1
                if spec.get("operator") == "conflicting" and len(assertions) < 2:
                    invalid.append({"example_id": example_id, "source_dataset": clean.get("source_dataset"), "status": "invalid", "invalid_reasons": ["authorization_conflict_lacks_assertions"]})
                    reasons["authorization_conflict_lacks_assertions"] += 1
                if spec.get("operator") == "flipped" and auth == (clean.get("gold_governance_state") or {}).get("authorization"):
                    invalid.append({"example_id": example_id, "source_dataset": clean.get("source_dataset"), "status": "invalid", "invalid_reasons": ["authorization_flip_did_not_flip"]})
                    reasons["authorization_flip_did_not_flip"] += 1
            elif dimension == "deletion":
                deletion = observed.get("deletion")
                if spec.get("operator") == "missing" and deletion is not None:
                    invalid.append({"example_id": example_id, "source_dataset": clean.get("source_dataset"), "status": "invalid", "invalid_reasons": ["deletion_missing_fault_not_missing"]})
                    reasons["deletion_missing_fault_not_missing"] += 1
                if spec.get("operator") == "flipped" and deletion == (clean.get("gold_governance_state") or {}).get("deletion"):
                    invalid.append({"example_id": example_id, "source_dataset": clean.get("source_dataset"), "status": "invalid", "invalid_reasons": ["deletion_flip_did_not_flip"]})
                    reasons["deletion_flip_did_not_flip"] += 1
            elif dimension == "temporal":
                temporal = observed.get("temporal")
                assertions = observed.get("temporal_assertions") or []
                if spec.get("operator") == "missing" and temporal is not None:
                    invalid.append({"example_id": example_id, "source_dataset": clean.get("source_dataset"), "status": "invalid", "invalid_reasons": ["temporal_missing_fault_not_missing"]})
                    reasons["temporal_missing_fault_not_missing"] += 1
                if spec.get("operator") == "conflicting" and len(assertions) < 2:
                    invalid.append({"example_id": example_id, "source_dataset": clean.get("source_dataset"), "status": "invalid", "invalid_reasons": ["temporal_conflict_lacks_assertions"]})
                    reasons["temporal_conflict_lacks_assertions"] += 1
    return invalid, reasons


def audit_manifest_rows(rows: list[dict[str, Any]], sources: dict[tuple[str, str], Any]) -> dict[str, Any]:
    clean_rows = [row for row in rows if row.get("condition_type") == "clean"]
    records = [_audit_clean_row(row, sources.get((str(row.get("source_dataset")), str(_query_id(row))))) for row in clean_rows]
    pair_invalid, pair_reasons = _pair_audit(rows)
    invalid = [record for record in records if record["status"] == "invalid"]
    invalid.extend(pair_invalid)
    reason_counts = Counter(reason for record in invalid for reason in record.get("invalid_reasons", ()))
    reason_counts.update(pair_reasons)
    by_dataset = Counter(record["source_dataset"] for record in records)
    invalid_by_dataset = Counter(record["source_dataset"] for record in invalid if record.get("source_dataset"))
    return {
        "audit_version": "rq1a-semantic-audit-v1",
        "base_examples_total": len(clean_rows),
        "valid_base_examples": len(clean_rows) - len({record["example_id"] for record in invalid if record.get("example_id")}),
        "invalid_base_examples": len({record["example_id"] for record in invalid if record.get("example_id")}),
        "valid_by_dataset": {dataset: by_dataset[dataset] - invalid_by_dataset[dataset] for dataset in sorted(by_dataset)},
        "invalid_by_dataset": dict(sorted(invalid_by_dataset.items())),
        "invalid_by_reason": dict(sorted(reason_counts.items())),
        "fault_pair_invalid_examples": sorted({record["example_id"] for record in pair_invalid}),
        "invalid_records": invalid,
        "records": records,
    }


def _load_sources(paths: dict[str, str | None], wanted: dict[str, set[str]]) -> dict[tuple[str, str], Any]:
    adapters = {"gatemem": GateMemDatasetAdapter, "memora": MemoraDatasetAdapter, "memconflict": MemConflictDatasetAdapter}
    result: dict[tuple[str, str], Any] = {}
    for dataset, path in paths.items():
        if not path:
            continue
        examples = adapters[dataset]().load(path)
        for example in examples:
            if example.query.query_id in wanted.get(dataset, set()):
                result[(dataset, example.query.query_id)] = example
    return result


def audit_file(manifest: str | Path, gatemem_input: str | None, memora_input: str | None, memconflict_input: str | None) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    paths = {"gatemem": gatemem_input, "memora": memora_input, "memconflict": memconflict_input}
    reports = []
    # Isolate official-source parsing so large histories from separate
    # benchmarks do not accumulate in one process.
    for dataset, path in paths.items():
        if not path:
            continue
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            child_output = Path(handle.name)
        try:
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--manifest", str(manifest),
                 "--output", str(child_output), "--_dataset-only", dataset, "--_dataset-input", str(path)],
                check=True, stdout=subprocess.DEVNULL,
            )
            reports.append(json.loads(child_output.read_text(encoding="utf-8")))
        finally:
            child_output.unlink(missing_ok=True)
    if reports:
        records = [record for report in reports for record in report["records"]]
        invalid = [record for report in reports for record in report["invalid_records"]]
        reasons = Counter(reason for record in invalid for reason in record.get("invalid_reasons", ()))
        clean_total = sum(report["base_examples_total"] for report in reports)
        invalid_ids = {record["example_id"] for record in invalid}
        clean_by_dataset = Counter(record["source_dataset"] for record in records)
        invalid_by_dataset = Counter(record.get("source_dataset") for record in invalid if record.get("source_dataset"))
        fault_pairs = sorted({example_id for report in reports for example_id in report.get("fault_pair_invalid_examples", [])})
        source_found = {
            dataset: next((report.get("source_examples_found", {}).get(dataset, 0) for report in reports if report.get("dataset_only") == dataset), 0)
            for dataset in paths
        }
        return {
            "audit_version": "rq1a-semantic-audit-v1",
            "manifest": str(manifest),
            "base_examples_total": clean_total,
            "valid_base_examples": clean_total - len(invalid_ids),
            "invalid_base_examples": len(invalid_ids),
            "valid_by_dataset": {dataset: clean_by_dataset[dataset] - invalid_by_dataset[dataset] for dataset in sorted(clean_by_dataset)},
            "invalid_by_dataset": dict(sorted(invalid_by_dataset.items())),
            "invalid_by_reason": dict(sorted(reasons.items())),
            "fault_pair_invalid_examples": fault_pairs,
            "invalid_records": invalid,
            "records": records,
            "source_examples_found": source_found,
        }
    clean = [row for row in rows if row.get("condition_type") == "clean"]
    wanted = defaultdict(set)
    for row in clean:
        if _query_id(row) is not None:
            wanted[str(row.get("source_dataset"))].add(str(_query_id(row)))
    sources = _load_sources({"gatemem": gatemem_input, "memora": memora_input, "memconflict": memconflict_input}, wanted)
    report = audit_manifest_rows(rows, sources)
    report["manifest"] = str(manifest)
    report["source_examples_found"] = {dataset: sum(1 for key in sources if key[0] == dataset) for dataset in ("gatemem", "memora", "memconflict")}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gatemem-input")
    parser.add_argument("--memora-input")
    parser.add_argument("--memconflict-input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--_dataset-only", choices=("gatemem", "memora", "memconflict"), help=argparse.SUPPRESS)
    parser.add_argument("--_dataset-input", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._dataset_only:
        rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        selected = [row for row in rows if row.get("source_dataset") == args._dataset_only]
        clean = [row for row in selected if row.get("condition_type") == "clean"]
        wanted = {str(_query_id(row)) for row in clean if _query_id(row) is not None}
        sources = _load_sources({args._dataset_only: args._dataset_input}, {args._dataset_only: wanted})
        report = audit_manifest_rows(selected, sources)
        report["dataset_only"] = args._dataset_only
        report["source_examples_found"] = {args._dataset_only: sum(1 for key in sources if key[0] == args._dataset_only)}
    else:
        report = audit_file(args.manifest, args.gatemem_input, args.memora_input, args.memconflict_input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("base_examples_total", "valid_base_examples", "invalid_base_examples", "valid_by_dataset", "invalid_by_dataset", "invalid_by_reason", "fault_pair_invalid_examples")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
