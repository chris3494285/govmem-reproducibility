"""Deterministic, not-yet-frozen system-validation cohort construction."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


REQUIRED_CONDITIONS = {
    "gatemem": ("clean", "flipped", "conflicting"),
    "memora": ("clean", "flipped", "stale"),
    "memconflict": ("clean", "stale", "missing"),
}
DIMENSIONS = {"gatemem": "authorization", "memora": "deletion", "memconflict": "temporal"}


def _fault_type(row: dict[str, Any]) -> str:
    return "clean" if row.get("condition_type") == "clean" else str((row.get("fault_spec") or {}).get("operator"))


def _cluster_id(row: dict[str, Any]) -> str:
    metadata = row.get("source_metadata") or {}
    return str(
        metadata.get("episode_id")
        or metadata.get("query_id")
        or row.get("candidate_cluster_id")
        or row.get("example_id")
    )


def _balanced_ids(groups: dict[str, list[dict[str, Any]]], count: int) -> list[str]:
    clusters: dict[str, list[str]] = defaultdict(list)
    for example_id, rows in groups.items():
        clusters[_cluster_id(rows[0])].append(example_id)
    for ids in clusters.values():
        ids.sort()
    ordered_clusters = sorted(clusters)
    selected: list[str] = []
    while len(selected) < count:
        progressed = False
        for cluster in ordered_clusters:
            if len(selected) >= count:
                break
            if clusters[cluster]:
                selected.append(clusters[cluster].pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"Unable to select {count} complete base examples; found {len(selected)}")
    return selected


def build_system_validation_manifest(
    rows: Iterable[dict[str, Any]],
    *,
    per_dimension: int = 20,
    source_manifest: str | Path = "data/governance_fault_rq1_pilot_frozen_v2.jsonl",
    status: str = "NON-FROZEN / DRAFT",
) -> dict[str, Any]:
    if per_dimension < 1:
        raise ValueError("per_dimension must be positive")
    all_rows = [deepcopy(row) for row in rows]
    groups_by_dimension: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        key = (str(row.get("source_dataset")), str(row.get("dimension")))
        groups_by_dimension[key][str(row.get("example_id"))].append(row)

    selected: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    for dataset in ("gatemem", "memora", "memconflict"):
        dimension = DIMENSIONS[dataset]
        groups = groups_by_dimension.get((dataset, dimension), {})
        candidates: dict[str, list[dict[str, Any]]] = {}
        for example_id, group in groups.items():
            by_type = {_fault_type(row): row for row in group}
            if all(condition in by_type for condition in REQUIRED_CONDITIONS[dataset]):
                candidates[example_id] = group
        selected_ids = _balanced_ids(candidates, per_dimension)
        selection[f"{dataset}:{dimension}"] = {
            "selected_base_example_ids": selected_ids,
            "cluster_ids": sorted({_cluster_id(candidates[example_id][0]) for example_id in selected_ids}),
            "conditions": list(REQUIRED_CONDITIONS[dataset]),
        }
        for example_id in selected_ids:
            cluster = _cluster_id(candidates[example_id][0])
            for row in sorted(candidates[example_id], key=lambda item: (item.get("condition_type") != "clean", item.get("run_id", ""))):
                row["base_example_id"] = example_id
                row["system_validation_cluster_id"] = cluster
                selected.append(row)

    selected.sort(key=lambda row: (row["source_dataset"], row["dimension"], row["example_id"], row["condition_type"], row.get("run_id", "")))
    return {
        "manifest_version": "system-validation-cohort-v1",
        "status": status,
        "frozen_source_manifest": str(source_manifest),
        "selection_method": "deterministic_cluster_balanced_complete_pairs",
        "per_dimension": per_dimension,
        "selection": selection,
        "records": selected,
        "selected_base_examples": per_dimension * 3,
        "selected_condition_rows": len(selected),
    }


def build_system_validation_smoke(rows: Iterable[dict[str, Any]], *, source_manifest: str | Path) -> dict[str, Any]:
    manifest = build_system_validation_manifest(rows, per_dimension=3, source_manifest=source_manifest, status="NON-FROZEN / SMOKE")
    manifest["manifest_version"] = "system-validation-smoke-v1"
    return manifest
