#!/usr/bin/env python3
"""Build the deterministic, FROZEN GovMem system-validation cohort."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from govmem.governance.enforcement import decide_admission  # noqa: E402
from govmem.memory_systems.harness import _content  # noqa: E402
from scripts.run_governance_fault_rq1 import load_manifest  # noqa: E402


SEED = 42
DEFAULT_SOURCE = "data/governance_fault_rq1_pilot_frozen_v2.jsonl"
DEFAULT_OUTPUT = "data/system_validation_frozen_v1.jsonl"
DEFAULT_METADATA = "data/system_validation_frozen_v1_metadata.json"
TARGETS = {
    ("gatemem", "authorization"): (20, ("clean", "conflicting", "flipped")),
    ("memora", "deletion"): (20, ("clean", "flipped", "stale")),
    ("memconflict", "temporal"): (20, ("clean", "missing", "stale")),
}


def _operator(row: dict[str, Any]) -> str:
    return "clean" if row.get("condition_type") == "clean" else str((row.get("fault_spec") or {}).get("operator"))


def _cluster_id(row: dict[str, Any]) -> str:
    source = row.get("source_metadata") or {}
    return str(source.get("episode_id") or source.get("query_id") or row.get("example_id"))


def _select_ids(groups: dict[str, list[dict[str, Any]]], count: int, seed: int) -> list[str]:
    if len(groups) < count:
        raise ValueError(f"Only {len(groups)} eligible base examples available; need {count}")
    clusters: dict[str, list[str]] = defaultdict(list)
    for example_id, rows in groups.items():
        clusters[_cluster_id(rows[0])].append(example_id)
    rng = random.Random(seed)
    for ids in clusters.values():
        ids.sort()
        rng.shuffle(ids)
    selected: list[str] = []
    for cluster in sorted(clusters):
        # Keep the seeded within-cluster order; sorting here would silently
        # turn the formal selection into an unseeded lexical prefix.
        pass
    while len(selected) < count:
        progressed = False
        for cluster in sorted(clusters):
            if clusters[cluster] and len(selected) < count:
                selected.append(clusters[cluster].pop(0))
                progressed = True
        if not progressed:
            break
    return sorted(selected)


def _fault_state(clean: dict[str, Any], operator: str) -> dict[str, Any]:
    state = copy.deepcopy(clean["gold_governance_state"])
    dataset = clean["source_dataset"]
    if dataset == "gatemem":
        auth = state.get("authorization")
        if not isinstance(auth, dict):
            raise ValueError(f"Authorization state missing for {clean['example_id']}")
        if operator == "conflicting":
            opposite = {**auth, "decision": "allow" if auth["decision"] == "deny" else "deny"}
            state["authorization"] = None
            state["authorization_assertions"] = [auth, opposite]
        elif operator == "flipped":
            state["authorization"] = {**auth, "decision": "allow" if auth["decision"] == "deny" else "deny"}
            state["authorization_assertions"] = []
        else:
            raise ValueError(f"Unsupported GateMem formal fault: {operator}")
    elif dataset == "memora":
        current = state.get("deletion")
        if operator == "flipped":
            state["deletion"] = {**current, "status": "active" if current["status"] == "deleted" else "deleted"}
        elif operator == "stale":
            history = (clean.get("source_metadata") or {}).get("historical_deletion_states") or []
            previous = next((item for item in reversed(history) if item != current), None)
            if previous is None:
                raise ValueError(f"Memora row has no native deletion predecessor: {clean['example_id']}")
            state["deletion"] = copy.deepcopy(previous)
        else:
            raise ValueError(f"Unsupported Memora formal synthesis fault: {operator}")
    elif dataset == "memconflict":
        if operator == "missing":
            state["temporal"] = None
            state["temporal_assertions"] = []
        elif operator == "stale":
            current = state.get("temporal")
            history = (clean.get("source_metadata") or {}).get("historical_temporal_states") or []
            previous = next((item for item in reversed(history) if item != current), None)
            if previous is None:
                raise ValueError(f"MemConflict row has no native temporal predecessor: {clean['example_id']}")
            state["temporal"] = copy.deepcopy(previous)
            state["temporal_assertions"] = []
        else:
            raise ValueError(f"Unsupported MemConflict formal synthesis fault: {operator}")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return state


def _fault_spec(dimension: str, operator: str) -> dict[str, str]:
    direction = "unsafe" if operator in {"flipped", "stale"} and dimension != "authorization" else "ambiguous"
    if operator == "flipped" and dimension == "authorization":
        direction = "unsafe"
    return {"dimension": dimension, "operator": operator, "direction": direction}


def _row_for_condition(clean: dict[str, Any], existing: dict[str, Any] | None, operator: str) -> dict[str, Any]:
    row = copy.deepcopy(existing if existing is not None else clean)
    row["base_example_id"] = str(clean["example_id"])
    row["governance_dimension"] = row["dimension"]
    row["condition"] = operator
    row["condition_type"] = "clean" if operator == "clean" else "fault"
    row["fault_type"] = operator
    row["fault_direction"] = "clean" if operator == "clean" else _fault_spec(clean["dimension"], operator)["direction"]
    if operator == "clean":
        row["fault_spec"] = None
        row["observed_governance_state"] = copy.deepcopy(clean["gold_governance_state"])
    else:
        row["fault_spec"] = _fault_spec(clean["dimension"], operator)
        if existing is None:
            row["observed_governance_state"] = _fault_state(clean, operator)
    row["query"] = row["task"]
    contents = [_content(item) for item in row.get("memory") or []]
    row["underlying_memory_content"] = contents
    row["underlying_memory_sha256"] = hashlib.sha256(json.dumps(contents, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    row["governance_state"] = copy.deepcopy(row["observed_governance_state"])
    admission = decide_admission(row)
    row["expected_governance_action"] = "governance_admit" if admission.actual_admission == "allow" else "governance_deny"
    return row


def build_manifest(rows: list[dict[str, Any]], *, seed: int = SEED, source_manifest: str = DEFAULT_SOURCE) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[(str(row.get("source_dataset")), str(row.get("dimension")))][str(row["example_id"])].append(row)
    result: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    for (dataset, dimension), (count, conditions) in TARGETS.items():
        candidates: dict[str, list[dict[str, Any]]] = {}
        for example_id, group in groups[(dataset, dimension)].items():
            clean = next((row for row in group if row.get("condition_type") == "clean"), None)
            if clean is None:
                continue
            if dataset == "memora" and not ((clean.get("source_metadata") or {}).get("historical_deletion_states")):
                continue
            if dataset == "memconflict":
                metadata = clean.get("source_metadata") or {}
                if not metadata.get("current_value") or not metadata.get("previous_value") or not metadata.get("historical_temporal_states"):
                    continue
            candidates[example_id] = group
        selected_ids = _select_ids(candidates, count, seed + len(selection))
        selection[f"{dataset}:{dimension}"] = {"selected_base_example_ids": selected_ids, "cluster_ids": sorted({_cluster_id(candidates[item][0]) for item in selected_ids}), "conditions": list(conditions)}
        for example_id in selected_ids:
            group = candidates[example_id]
            clean = next(row for row in group if row.get("condition_type") == "clean")
            by_operator = {_operator(row): row for row in group}
            for operator in conditions:
                result.append(_row_for_condition(clean, by_operator.get(operator), operator))
    result.sort(key=lambda row: (row["source_dataset"], row["governance_dimension"] if "governance_dimension" in row else row["dimension"], row["base_example_id"], conditions_index(row["condition"])))
    return result, {"source_manifest": source_manifest, "selection": selection}


def conditions_index(condition: str) -> int:
    return {"clean": 0, "conflicting": 1, "flipped": 1, "missing": 1, "stale": 2}[condition]


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def write_frozen(source: str | Path, output: str | Path, metadata_path: str | Path, *, seed: int = SEED, force: bool = False) -> dict[str, Any]:
    output = Path(output)
    metadata_path = Path(metadata_path)
    if (output.exists() or metadata_path.exists()) and not force:
        raise FileExistsError(f"Frozen output exists; pass --force to overwrite: {output} or {metadata_path}")
    rows, info = build_manifest(load_manifest(source, require_full=True), seed=seed, source_manifest=str(source))
    payload = _jsonl_bytes(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    script_path = Path(__file__).resolve()
    counts = {dimension: sum(1 for row in rows if row["dimension"] == dimension) for dimension in ("authorization", "deletion", "temporal")}
    bases = {dimension: len({row["base_example_id"] for row in rows if row["dimension"] == dimension}) for dimension in counts}
    metadata = {
        "status": "FROZEN", "version": "v1", "seed": seed,
        "base_example_count": len({row["base_example_id"] for row in rows}),
        "condition_row_count": len(rows),
        "per_dimension_base_counts": bases,
        "per_dimension_condition_counts": counts,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "builder_script": str(script_path.relative_to(ROOT)),
        "builder_script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "source_manifest": str(source),
        "selection_method": "seeded_cluster_balanced_sampling_with_native_fault_completion",
        **info,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.seed != SEED:
        parser.error("Formal system-validation v1 requires --seed 42")
    metadata = write_frozen(args.source_manifest, args.output, args.metadata, seed=args.seed, force=args.force)
    print(json.dumps({"status": metadata["status"], "base_example_count": metadata["base_example_count"], "condition_row_count": metadata["condition_row_count"], "manifest_sha256": metadata["manifest_sha256"], "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
