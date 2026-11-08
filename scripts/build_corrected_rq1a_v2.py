#!/usr/bin/env python3
"""Create the minimally corrected v2 copy of the historical RQ1-A manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from eval.memconflict_adapter import MemConflictDatasetAdapter
from govmem.governance.oracle import derive_memconflict_temporal


V2_VERSION = "governance-fault-rq1-v2-corrected"
CORRECTION_REASON = "multi_attribute_binding_bug"


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_id(row: dict[str, Any]) -> str:
    source = row.get("source_metadata") or {}
    query_id = source.get("query_id")
    if not query_id:
        raise ValueError(f"Frozen row lacks source query_id: {row.get('example_id')}")
    return str(query_id)


def _corrected_transition(row: dict[str, Any], source: Any) -> tuple[str, Any, Any, str]:
    derivation = derive_memconflict_temporal(source)
    if not derivation.supported or derivation.example is None:
        raise ValueError(f"Corrected oracle cannot support {row['example_id']}")
    metadata = derivation.example.source_metadata
    attribute = metadata.get("attribute")
    current = metadata.get("current_value")
    previous = metadata.get("previous_value")
    stored = row.get("source_metadata") or {}
    if stored.get("current_value") != current or stored.get("previous_value") != previous:
        raise ValueError(f"Native Before/After values changed for {row['example_id']}; refusing repair")
    session = (metadata.get("current_session") or {}).get("session_id")
    source_record = metadata.get("source_record_id") or source.query.metadata.get("source_record_id")
    transition_id = f"{source_record}::s{session}::{attribute}"
    return str(attribute), current, previous, transition_id


def _transition_source_session(source: Any, attribute: str, current: Any, previous: Any) -> dict[str, Any] | None:
    for update in (source.evaluator_metadata.get("gold", {}) or {}).get("updated_attributes", ()):
        for native in update.get("attributes", ()):
            if native.get("Attribute") == attribute and native.get("After") == current and native.get("Before") == previous:
                return {"session_id": update.get("session_id"), "date": update.get("date")}
    return None


def _set_attribute(row: dict[str, Any], attribute: str) -> None:
    source = row.setdefault("source_metadata", {})
    source["attribute"] = attribute
    memories = row.get("memory") or []
    if not memories or any("attribute" not in memory for memory in memories):
        raise ValueError(f"Frozen memory lacks an attribute label: {row['example_id']}")
    for memory in memories:
        memory["attribute"] = attribute


def _without_allowed_attribute_changes(row: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(row)
    value.pop("manifest_version", None)
    value.get("source_metadata", {}).pop("attribute", None)
    for memory in value.get("memory", []):
        memory.pop("attribute", None)
    return value


def validate_strict_diff(v1: list[dict[str, Any]], v2: list[dict[str, Any]], affected: set[str]) -> None:
    if len(v1) != len(v2):
        raise ValueError("v1/v2 row count changed")
    for old, new in zip(v1, v2):
        if old.get("run_id") != new.get("run_id") or old.get("example_id") != new.get("example_id"):
            raise ValueError("Example or run identity changed")
        if old.get("example_id") not in affected and old != {**new, "manifest_version": new.get("manifest_version")}:
            # The explicit form above allows only the version addition for
            # historical v1 rows, which had no manifest_version.
            old_without_version = dict(old)
            new_without_version = dict(new)
            new_without_version.pop("manifest_version", None)
            if old_without_version != new_without_version:
                raise ValueError(f"Unaffected row changed: {old.get('example_id')}")
        elif old.get("example_id") in affected:
            if _without_allowed_attribute_changes(old) != _without_allowed_attribute_changes(new):
                raise ValueError(f"Disallowed v1/v2 change in affected row: {old.get('example_id')}")


def build_v2(manifest_path: str | Path, memconflict_input: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    v1 = _read(manifest_path)
    if len(v1) != 300:
        raise ValueError(f"Historical RQ1-A must contain 300 rows, got {len(v1)}")
    clean = [row for row in v1 if row.get("condition_type") == "clean"]
    if len(clean) != 100:
        raise ValueError("Historical RQ1-A must contain 100 clean rows")
    source_map = {example.query.query_id: example for example in MemConflictDatasetAdapter().load(memconflict_input)}
    affected: dict[str, dict[str, Any]] = {}
    for row in clean:
        if row.get("source_dataset") != "memconflict":
            continue
        source = source_map.get(_source_id(row))
        if source is None:
            raise ValueError(f"Frozen MemConflict source is missing: {row['example_id']}")
        corrected, current, previous, transition_id = _corrected_transition(row, source)
        old_attribute = (row.get("source_metadata") or {}).get("attribute")
        memory_attributes = {memory.get("attribute") for memory in (row.get("memory") or [])}
        if old_attribute != corrected or memory_attributes != {corrected}:
            affected[row["example_id"]] = {
                "example_id": row["example_id"],
                "query_id": _source_id(row),
                "old_attribute": old_attribute,
                "corrected_attribute": corrected,
                "previous_value": previous,
                "current_value": current,
                "source_session": _transition_source_session(source, corrected, current, previous),
                "transition_id": transition_id,
                "reason": CORRECTION_REASON,
            }
    if len(affected) != 13:
        raise ValueError(f"Expected exactly 13 historical semantic corrections, found {len(affected)}")

    v2 = copy.deepcopy(v1)
    for row in v2:
        row["manifest_version"] = V2_VERSION
        if row.get("example_id") in affected:
            source = source_map[_source_id(row)]
            corrected, _, _, _ = _corrected_transition(row, source)
            _set_attribute(row, corrected)
    validate_strict_diff(v1, v2, set(affected))
    return v2, [affected[key] for key in sorted(affected)]


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    Path(path).write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/governance_fault_rq1_pilot_frozen.jsonl")
    parser.add_argument("--memconflict-input", required=True)
    parser.add_argument("--output", default="data/governance_fault_rq1_pilot_frozen_v2.jsonl")
    parser.add_argument("--corrections-output", default="data/governance_fault_rq1_pilot_v2_corrections.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if (Path(args.output).exists() or Path(args.corrections_output).exists()) and not args.overwrite:
        raise SystemExit("Refusing to overwrite v2 artifacts; use --overwrite explicitly")
    rows, corrections = build_v2(args.manifest, args.memconflict_input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    digest = write_jsonl(rows, args.output)
    Path(args.corrections_output).write_text(json.dumps({"manifest_version": V2_VERSION, "correction_count": len(corrections), "corrections": corrections}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output, "corrections_output": args.corrections_output, "correction_count": len(corrections), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
