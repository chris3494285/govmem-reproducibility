#!/usr/bin/env python3
"""Preflight validator for the frozen system-validation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED = {
    "gatemem": ("authorization", {"clean", "conflicting", "flipped"}),
    "memora": ("deletion", {"clean", "flipped", "stale"}),
    "memconflict": ("temporal", {"clean", "missing", "stale"}),
}


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Frozen system-validation manifest must contain JSON objects")
    return rows


def validate_manifest(manifest: str | Path, metadata_path: str | Path) -> dict[str, Any]:
    manifest = Path(manifest)
    metadata_path = Path(metadata_path)
    rows = read_rows(manifest)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "FROZEN":
        raise ValueError("Metadata status must be FROZEN")
    if metadata.get("version") != "v1" or metadata.get("seed") != 42:
        raise ValueError("Frozen system-validation metadata must specify version v1 and seed 42")
    if len(rows) != 180 or len({row.get("base_example_id") for row in rows}) != 60:
        raise ValueError("Frozen system-validation manifest must contain 180 rows and 60 unique bases")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_pairs: set[tuple[Any, Any]] = set()
    for row in rows:
        base = row.get("base_example_id")
        if not base:
            raise ValueError("Every row requires base_example_id")
        pair = (base, row.get("condition"))
        if pair in seen_pairs:
            raise ValueError(f"Duplicate base/condition: {base}/{row.get('condition')}")
        seen_pairs.add(pair)
        groups[str(base)].append(row)
        dataset = row.get("source_dataset")
        if dataset not in EXPECTED:
            raise ValueError(f"Unsupported source dataset: {dataset}")
        dimension, _ = EXPECTED[dataset]
        if row.get("dimension") != dimension or row.get("governance_dimension") != dimension:
            raise ValueError(f"Dimension mismatch for {base}")
        if "NON-FROZEN" in json.dumps(row, sort_keys=True) or "SMOKE" in json.dumps(row, sort_keys=True):
            raise ValueError(f"Frozen manifest contains smoke marker: {base}")
        if any(token in row for token in ("llm_judge", "embedding_score", "generated_gold", "judge_label")):
            raise ValueError(f"LLM-generated gold field introduced in {base}")
    per_dimension_bases: dict[str, set[str]] = defaultdict(set)
    per_dimension_rows: dict[str, int] = defaultdict(int)
    for base, group in groups.items():
        dataset = group[0]["source_dataset"]
        dimension, expected_conditions = EXPECTED[dataset]
        conditions = [row.get("condition") for row in group]
        if len(group) != 3 or set(conditions) != expected_conditions or len(set(conditions)) != 3:
            raise ValueError(f"Invalid condition set for {base}: {conditions}")
        clean = next(row for row in group if row["condition"] == "clean")
        for row in group:
            for field in ("query", "underlying_memory_content", "underlying_memory_sha256"):
                if row.get(field) != clean.get(field):
                    raise ValueError(f"Sibling invariant failed for {base}: {field}")
            if row["condition"] == "clean":
                if row.get("fault_spec") is not None or row.get("fault_type") != "clean":
                    raise ValueError(f"Clean condition has fault metadata: {base}")
            else:
                if (row.get("fault_spec") or {}).get("operator") != row["condition"]:
                    raise ValueError(f"Fault operator mismatch for {base}")
        if dataset == "gatemem":
            if clean.get("gold_admission_decision") != "deny" or (clean.get("gold_governance_state") or {}).get("authorization", {}).get("decision") != "deny":
                raise ValueError(f"GateMem clean is not deny: {base}")
        elif dataset == "memora":
            source = clean.get("source_metadata") or {}
            if not source.get("historical_deletion_states"):
                raise ValueError(f"Memora lacks native deletion lifecycle history: {base}")
            stale = next(row for row in group if row["condition"] == "stale")
            if not (stale.get("observed_governance_state") or {}).get("deletion"):
                raise ValueError(f"Memora stale state missing: {base}")
        else:
            source = clean.get("source_metadata") or {}
            if not source.get("current_value") or not source.get("previous_value") or not source.get("historical_temporal_states"):
                raise ValueError(f"MemConflict lacks native current/predecessor semantics: {base}")
            stale = next(row for row in group if row["condition"] == "stale")
            if not (stale.get("observed_governance_state") or {}).get("temporal"):
                raise ValueError(f"MemConflict stale state missing: {base}")
        per_dimension_bases[dimension].add(base)
        per_dimension_rows[dimension] += len(group)
    expected_bases = {"authorization": 20, "deletion": 20, "temporal": 20}
    expected_rows = {dimension: 60 for dimension in expected_bases}
    if {key: len(value) for key, value in per_dimension_bases.items()} != expected_bases:
        raise ValueError(f"Per-dimension base counts mismatch: {per_dimension_bases}")
    if dict(per_dimension_rows) != expected_rows:
        raise ValueError(f"Per-dimension row counts mismatch: {per_dimension_rows}")
    actual_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if metadata.get("manifest_sha256") != actual_hash:
        raise ValueError("Manifest SHA256 does not match metadata")
    if metadata.get("base_example_count") != 60 or metadata.get("condition_row_count") != 180:
        raise ValueError("Metadata count fields are incorrect")
    if metadata.get("per_dimension_base_counts") != {"authorization": 20, "deletion": 20, "temporal": 20}:
        raise ValueError("Metadata per-dimension base counts are incorrect")
    if metadata.get("per_dimension_condition_counts") != {"authorization": 60, "deletion": 60, "temporal": 60}:
        raise ValueError("Metadata per-dimension condition counts are incorrect")
    return {"status": "FROZEN", "base_example_count": 60, "condition_row_count": 180, "manifest_sha256": actual_hash}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/system_validation_frozen_v1.jsonl")
    parser.add_argument("--metadata", default="data/system_validation_frozen_v1_metadata.json")
    args = parser.parse_args(argv)
    result = validate_manifest(args.manifest, args.metadata)
    print("SYSTEM VALIDATION FROZEN PREFLIGHT PASS")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
