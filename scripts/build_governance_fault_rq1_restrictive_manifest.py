#!/usr/bin/env python3
"""Freeze the deterministic restrictive-fault supplement for governance RQ1-B."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.audit_rq1_restrictive_candidates import audit_inputs
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from audit_rq1_restrictive_candidates import audit_inputs


MANIFEST_VERSION = "governance-fault-rq1-restrictive-v1"
MEMORA_TARGET = 15
MEMCONFLICT_TARGET = 15


def _candidate_sort_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_id", ""))


def _select_memora(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        clusters[str(candidate["candidate_cluster_id"])].append(candidate)
    if len(clusters) < MEMORA_TARGET:
        raise ValueError(f"Memora has only {len(clusters)} unique query clusters; need {MEMORA_TARGET}")
    selected = []
    for cluster_id in sorted(clusters)[:MEMORA_TARGET]:
        selected.append(sorted(clusters[cluster_id], key=_candidate_sort_key)[0])
    return selected


def _select_memconflict(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        transitions[str(candidate["transition_id"])].append(candidate)
    if len(transitions) < MEMCONFLICT_TARGET:
        raise ValueError(
            f"MemConflict has only {len(transitions)} unique transitions; need {MEMCONFLICT_TARGET}"
        )

    representatives = {
        transition_id: sorted(rows, key=_candidate_sort_key)[0]
        for transition_id, rows in transitions.items()
    }
    by_attribute: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in representatives.values():
        by_attribute[str(candidate.get("attribute", ""))].append(candidate)
    for rows in by_attribute.values():
        rows.sort(key=_candidate_sort_key)

    selected: list[dict[str, Any]] = []
    positions = {attribute: 0 for attribute in sorted(by_attribute)}
    while len(selected) < MEMCONFLICT_TARGET:
        progressed = False
        for attribute in sorted(by_attribute):
            position = positions[attribute]
            rows = by_attribute[attribute]
            if position >= len(rows):
                continue
            selected.append(rows[position])
            positions[attribute] += 1
            progressed = True
            if len(selected) == MEMCONFLICT_TARGET:
                break
        if not progressed:
            raise ValueError("Unable to select the requested MemConflict transitions")
    return selected


def _fault_observed(candidate: dict[str, Any], dataset: str) -> dict[str, Any]:
    observed = copy.deepcopy(candidate["gold_governance_state"])
    if dataset == "memora":
        observed["deletion"]["status"] = "deleted"
    elif dataset == "memconflict":
        observed["temporal"]["status"] = "expired"
    else:
        raise ValueError(f"Unsupported restrictive dataset: {dataset}")
    return observed


def _row(candidate: dict[str, Any], condition: str, observed: dict[str, Any]) -> dict[str, Any]:
    dataset = str(candidate["source_dataset"])
    fault_spec = None
    if condition == "fault":
        operator = "flipped" if dataset == "memora" else "expired"
        fault_spec = {"dimension": candidate["dimension"], "operator": operator, "direction": "restrictive"}
    row = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": f"{candidate['candidate_id']}::{condition}",
        "example_id": candidate["candidate_id"],
        "source_dataset": dataset,
        "dimension": candidate["dimension"],
        "condition_type": condition,
        "fault_spec": fault_spec,
        "gold_governance_state": copy.deepcopy(candidate["gold_governance_state"]),
        "observed_governance_state": observed,
        "gold_admission_decision": candidate["gold_admission_decision"],
        "task": candidate["task"],
        "memory": copy.deepcopy(candidate["memory"]),
        "utility_target": copy.deepcopy(candidate.get("utility_target")),
        "source_metadata": copy.deepcopy(candidate["source_metadata"]),
    }
    for key in ("candidate_cluster_id", "transition_id", "attribute", "item", "current_value", "previous_value"):
        if key in candidate:
            row[key] = copy.deepcopy(candidate[key])
    return row


def validate_restrictive_manifest(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 60:
        raise ValueError(f"Restrictive manifest must contain 60 rows, got {len(rows)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_ids: set[str] = set()
    for row in rows:
        if row["run_id"] in run_ids:
            raise ValueError(f"Duplicate run_id: {row['run_id']}")
        run_ids.add(row["run_id"])
        groups[str(row["example_id"])].append(row)
    if len(groups) != 30:
        raise ValueError(f"Restrictive manifest must contain 30 bases, got {len(groups)}")

    memora_clusters: set[str] = set()
    conflict_transitions: set[str] = set()
    for example_id, pair in groups.items():
        clean = [row for row in pair if row["condition_type"] == "clean"]
        fault = [row for row in pair if row["condition_type"] == "fault"]
        if len(clean) != 1 or len(fault) != 1:
            raise ValueError(f"{example_id} must have exactly one clean and one fault row")
        clean, fault = clean[0], fault[0]
        if clean["gold_admission_decision"] != "allow" or fault["gold_admission_decision"] != "allow":
            raise ValueError(f"{example_id} does not have allow gold admission")
        for key in ("task", "memory", "utility_target", "gold_governance_state", "source_metadata", "source_dataset", "dimension"):
            if clean[key] != fault[key]:
                raise ValueError(f"{example_id} changed invariant field {key}")
        if clean["fault_spec"] is not None or clean["observed_governance_state"] != clean["gold_governance_state"]:
            raise ValueError(f"{example_id} clean row is not a gold-state condition")
        spec = fault["fault_spec"] or {}
        if spec.get("direction") != "restrictive":
            raise ValueError(f"{example_id} fault is not restrictive")
        if fault["source_dataset"] == "memora":
            if spec.get("dimension") != "deletion" or spec.get("operator") != "flipped":
                raise ValueError(f"{example_id} has invalid Memora restrictive fault")
            memora_clusters.add(str(fault["candidate_cluster_id"]))
            if fault["observed_governance_state"]["deletion"]["status"] != "deleted":
                raise ValueError(f"{example_id} does not observe deletion")
        elif fault["source_dataset"] == "memconflict":
            if spec.get("dimension") != "temporal" or spec.get("operator") != "expired":
                raise ValueError(f"{example_id} has invalid MemConflict restrictive fault")
            conflict_transitions.add(str(fault["transition_id"]))
            if fault["observed_governance_state"]["temporal"]["status"] != "expired":
                raise ValueError(f"{example_id} does not observe expiration")
        else:
            raise ValueError(f"Unsupported restrictive dataset: {fault['source_dataset']}")
    if len(memora_clusters) != MEMORA_TARGET or len(conflict_transitions) != MEMCONFLICT_TARGET:
        raise ValueError("Restrictive cohort cluster counts are incorrect")


def build_from_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    memora = _select_memora([row for row in candidates if row["source_dataset"] == "memora"])
    conflict = _select_memconflict([row for row in candidates if row["source_dataset"] == "memconflict"])
    rows: list[dict[str, Any]] = []
    for candidate in [*memora, *conflict]:
        rows.append(_row(candidate, "clean", copy.deepcopy(candidate["gold_governance_state"])))
        rows.append(_row(candidate, "fault", _fault_observed(candidate, candidate["source_dataset"])))
    validate_restrictive_manifest(rows)
    return rows


def write_manifest(rows: list[dict[str, Any]], output: str | Path, overwrite: bool = False) -> str:
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {output}; use --overwrite explicitly")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    output.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gatemem-input")
    parser.add_argument("--memora-input", required=True)
    parser.add_argument("--memconflict-input", required=True)
    parser.add_argument("--output", default="data/governance_fault_rq1_restrictive_frozen.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = audit_inputs(args.gatemem_input, args.memora_input, args.memconflict_input)
    rows = build_from_candidates(report["candidates"])
    digest = write_manifest(rows, args.output, args.overwrite)
    bases = {row["example_id"] for row in rows}
    by_dataset = Counter(row["source_dataset"] for row in rows)
    print(json.dumps({
        "output": args.output,
        "manifest_version": MANIFEST_VERSION,
        "sha256": digest,
        "base_examples": len(bases),
        "condition_rows": len(rows),
        "base_examples_by_dataset": {dataset: len({row["example_id"] for row in rows if row["source_dataset"] == dataset}) for dataset in sorted(by_dataset)},
        "condition_rows_by_dataset": dict(by_dataset),
        "memora_unique_query_clusters": len({row["candidate_cluster_id"] for row in rows if row["source_dataset"] == "memora"}),
        "memconflict_unique_transition_clusters": len({row["transition_id"] for row in rows if row["source_dataset"] == "memconflict"}),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
