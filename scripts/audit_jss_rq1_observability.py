#!/usr/bin/env python3
"""Audit frozen inputs for the JSS RQ1 testing-observability analysis.

The audit is offline and read-only.  It verifies the frozen condition and base
manifests, reconstructs the clean/corrupted/enforced triplets, checks all six
frozen aggregate score files against manifest identifiers and context hashes,
and verifies the stable join used by the Formal-300 native validation panel.

No benchmark text, model response text, or external service is used.  Output
contains counts, hashes, paths, condition identifiers, and schema diagnostics
only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONDITION_MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v3.jsonl"
DEFAULT_BASE_MANIFEST = ROOT / "data/final_main_governance_base_manifest_v2.jsonl"
DEFAULT_CONDITION_FREEZE = ROOT / "data/final_main_governance_condition_manifest_v3.freeze.json"
DEFAULT_BASE_FREEZE = ROOT / "data/final_main_governance_base_manifest_v2.freeze.json"
DEFAULT_RESULTS_DIR = ROOT / "results/final_main_sanitized"
DEFAULT_FORMAL_OBSERVATIONS = ROOT / "results/formal_300_validation_v1/analysis/resolved_observations.jsonl"
DEFAULT_FORMAL_BLOCKS = ROOT / "results/formal_300_validation_v1/analysis/resolved_blocks.jsonl"
DEFAULT_FORMAL_SUMMARY = ROOT / "results/formal_300_validation_v1/analysis/formal_validation_summary.json"
DEFAULT_FINAL_FORMAL_BLOCKS = (
    ROOT
    / "results/b4_scoring_validation/private/formal_300_validation_v1/final_analysis/"
    "formal_300_block_level_results.csv"
)
DEFAULT_FINAL_FORMAL_METRICS = (
    ROOT
    / "results/b4_scoring_validation/private/formal_300_validation_v1/final_analysis/"
    "formal_300_final_metrics.json"
)
DEFAULT_HUMAN_ADJUDICATION = (
    ROOT
    / "results/b4_scoring_validation/private/formal_300_validation_v1/"
    "human_adjudication/human_adjudication_decisions.jsonl"
)
DEFAULT_SEMANTIC_PAYLOADS = (
    ROOT
    / "results/b4_scoring_validation/private/formal_300_validation_v1/"
    "semantic_payloads.jsonl"
)
DEFAULT_EVIDENCE_BASE_LEVEL = (
    ROOT
    / "results/answer_leakage_repair/final_v3_analysis/"
    "governance_evidence_audit/governance_evidence_base_level.csv"
)

CONDITIONS = ("clean", "corrupted", "enforced")
PROVIDER_FILES = ("anthropic", "deepseek", "gemini", "grok", "kimi", "openai")
EXPECTED_FROZEN_SCHEMAS = {
    "condition": "main-governance-condition-v3",
    "base": "final-main-governance-base-v2",
}
COUNTABLE_CURRENT_STATUS = {"valid", "invalid", "expired", "deleted", "active"}
NATIVE_PANEL_MODES = ("paper_final_298", "public_297")


class AuditError(RuntimeError):
    """Raised when provenance or a required stable join cannot be verified."""


@dataclass
class AuditBundle:
    report: dict[str, Any]
    triplets: list[dict[str, Any]]
    provider_results: dict[str, dict[str, dict[str, Any]]]
    formal_observations: list[dict[str, Any]]
    formal_blocks: list[dict[str, Any]]
    evidence_support: dict[str, bool]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AuditError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def path_locator(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"Expected a JSON object in {path}")
    return value


def _verify_freeze(manifest: Path, freeze_path: Path, *, kind: str) -> dict[str, Any]:
    if not manifest.is_file():
        raise AuditError(f"Required {kind} manifest is absent: {manifest}")
    if not freeze_path.is_file():
        raise AuditError(f"Required {kind} freeze record is absent: {freeze_path}")
    freeze = _load_json(freeze_path)
    manifest_sha256 = sha256_file(manifest)
    declared_sha256 = freeze.get("manifest_sha256")
    if manifest_sha256 != declared_sha256:
        raise AuditError(
            f"{kind} manifest SHA256 mismatch: observed {manifest_sha256}, "
            f"freeze declares {declared_sha256}"
        )
    declared_name = freeze.get("manifest_filename")
    if declared_name and manifest.name != declared_name:
        raise AuditError(
            f"{kind} manifest filename mismatch: observed {manifest.name}, "
            f"freeze declares {declared_name}"
        )
    declared_schema = freeze.get("schema_version")
    if declared_schema != EXPECTED_FROZEN_SCHEMAS[kind]:
        raise AuditError(
            f"Unexpected frozen {kind} schema: {declared_schema!r}; "
            f"expected {EXPECTED_FROZEN_SCHEMAS[kind]!r}"
        )
    return {
        "path": path_locator(freeze_path),
        "sha256": sha256_file(freeze_path),
        "schema_version": declared_schema,
        "manifest_sha256": manifest_sha256,
        "superseded_manifest_sha256": freeze.get("superseded_manifest_sha256"),
    }


def _duplicate_values(values: Iterable[Any]) -> list[str]:
    counts = Counter(str(value) for value in values)
    return sorted(value for value, count in counts.items() if count > 1)


def _pair_id(base_id: str, fault_operator: str) -> str:
    return f"{base_id}::{fault_operator}"


def build_triplets(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct unique clean/corrupted/enforced triplets from a manifest."""
    condition_ids = [row.get("condition_id") for row in rows]
    missing_condition_ids = sum(not value for value in condition_ids)
    duplicate_condition_ids = _duplicate_values(value for value in condition_ids if value)

    clean_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unpaired_members: list[str] = []
    for row in rows:
        condition_id = row.get("condition_id")
        condition_type = row.get("condition_type")
        if not condition_id:
            raise AuditError("Condition row lacks condition_id")
        if condition_type == "clean":
            clean_by_base[str(row.get("base_id"))].append(row)
            continue
        if condition_type not in {"corrupted", "enforced"}:
            raise AuditError(f"Unsupported condition_type in {condition_id}: {condition_type!r}")
        pair_id = row.get("pair_id")
        if not pair_id:
            unpaired_members.append(str(condition_id))
            continue
        pair_members[str(pair_id)].append(row)

    duplicate_clean = sorted(base for base, group in clean_by_base.items() if len(group) != 1)
    if missing_condition_ids or duplicate_condition_ids:
        raise AuditError(
            "Condition manifest identifier failure: "
            f"missing={missing_condition_ids}, duplicates={duplicate_condition_ids[:5]}"
        )
    if duplicate_clean:
        raise AuditError(f"Each base must have exactly one clean row: {duplicate_clean[:5]}")
    if unpaired_members:
        raise AuditError(f"Corrupted/enforced rows lack pair_id: {unpaired_members[:5]}")

    triplets: list[dict[str, Any]] = []
    orphan_pairs: list[str] = []
    duplicate_members: list[str] = []
    for pair_id, members in sorted(pair_members.items()):
        by_condition: dict[str, dict[str, Any]] = {}
        for row in members:
            condition_type = str(row["condition_type"])
            if condition_type in by_condition:
                duplicate_members.append(pair_id)
            by_condition[condition_type] = row
        if set(by_condition) != {"corrupted", "enforced"} or duplicate_members:
            orphan_pairs.append(pair_id)
            continue
        corrupted = by_condition["corrupted"]
        enforced = by_condition["enforced"]
        base_id = str(corrupted.get("base_id"))
        clean = clean_by_base.get(base_id, [])
        if not clean:
            orphan_pairs.append(pair_id)
            continue
        fault_operator = corrupted.get("fault_operator")
        if not fault_operator:
            raise AuditError(f"Corrupted condition lacks fault_operator: {pair_id}")
        expected_pair_id = _pair_id(base_id, str(fault_operator))
        if pair_id != expected_pair_id:
            raise AuditError(f"Unstable pair_id: observed {pair_id}, expected {expected_pair_id}")
        if enforced.get("fault_operator") != fault_operator:
            raise AuditError(f"Fault operator drift inside {pair_id}")
        if (enforced.get("enforcement_specification") or {}).get("corrects_operator") != fault_operator:
            raise AuditError(f"Enforcement does not target the declared fault: {pair_id}")
        triplet_rows = {"clean": clean[0], "corrupted": corrupted, "enforced": enforced}
        for field in ("base_id", "source_dataset", "analysis_event_type"):
            values = {row.get(field) for row in triplet_rows.values()}
            if len(values) != 1:
                raise AuditError(f"{field} drift inside {pair_id}: {sorted(map(str, values))}")
        triplets.append(
            {
                "base_id": base_id,
                "pair_id": pair_id,
                "fault_operator": str(fault_operator),
                "source_dataset": str(corrupted.get("source_dataset")),
                "analysis_event_type": str(corrupted.get("analysis_event_type")),
                "conditions": triplet_rows,
            }
        )
    if orphan_pairs:
        raise AuditError(f"Incomplete or ambiguous pair groups: {orphan_pairs[:5]}")
    if len(pair_members) != len(triplets):
        raise AuditError("Not all pair groups reconstructed as triplets")
    return triplets, {
        "clean_rows": sum(len(group) for group in clean_by_base.values()),
        "pair_groups": len(pair_members),
        "complete_triplets": len(triplets),
        "duplicate_clean_rows": 0,
        "duplicate_condition_ids": 0,
        "duplicate_pair_members": len(duplicate_members),
        "orphan_pair_groups": 0,
    }


def _state_issues(state: Any, source_dataset: str) -> list[str]:
    """Evaluate source-specific structural constraints, independent of fault labels."""
    if not isinstance(state, dict):
        return ["state_not_an_object"]
    issues: list[str] = []
    status = state.get("current_status")
    if status not in COUNTABLE_CURRENT_STATUS:
        issues.append("invalid_current_status")

    ordered = state.get("ordered_evidence")
    if ordered is not None:
        if not isinstance(ordered, list):
            issues.append("ordered_evidence_not_list")
        else:
            ordinals = [item.get("ordinal") for item in ordered if isinstance(item, dict)]
            session_ids = [item.get("session_id") for item in ordered if isinstance(item, dict)]
            if len(ordinals) != len(ordered) or len(session_ids) != len(ordered):
                issues.append("ordered_evidence_item_schema")
            elif ordinals != list(range(len(ordinals))):
                issues.append("ordered_evidence_ordinals")
            if len(session_ids) != len(set(map(str, session_ids))):
                issues.append("ordered_evidence_duplicate_session")
            relevant = state.get("relevant_session_ids")
            if isinstance(relevant, list) and not {str(value) for value in session_ids} <= {
                str(value) for value in relevant
            }:
                issues.append("ordered_evidence_not_in_relevant_sessions")

    relevant = state.get("relevant_session_ids")
    if relevant is not None:
        if not isinstance(relevant, list):
            issues.append("relevant_session_ids_not_list")
        elif len(relevant) != len({str(value) for value in relevant}):
            issues.append("duplicate_relevant_session_id")

    if source_dataset == "memora":
        obsolete = state.get("obsolete_item")
        if obsolete is not None and not isinstance(obsolete, list):
            issues.append("obsolete_item_not_list")
    elif source_dataset == "oaks_novel":
        states = state.get("ordered_answer_states")
        if states is not None:
            if not isinstance(states, list):
                issues.append("ordered_answer_states_not_list")
            else:
                indices = [item.get("chunk_index") for item in states if isinstance(item, dict)]
                if len(indices) != len(states):
                    issues.append("ordered_answer_state_item_schema")
                elif any(not isinstance(value, int) or value < 0 for value in indices):
                    issues.append("ordered_answer_state_chunk_index")
                elif indices != sorted(indices) or len(indices) != len(set(indices)):
                    issues.append("ordered_answer_states_not_monotone_unique")
    elif source_dataset != "longmemeval" and ordered is None:
        issues.append("no_lifecycle_constraint_surface")
    return issues


def _context_items(row: Mapping[str, Any]) -> list[str]:
    items = row.get("model_visible_memory_items")
    if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
        raise AuditError(f"Invalid model_visible_memory_items for {row.get('condition_id')}")
    return items


def _normalize_item(value: str) -> str:
    return " ".join(value.split())


def canonical_items(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(_normalize_item(value) for value in _context_items(row))


def effective_context_text(row: Mapping[str, Any]) -> str:
    """Reproduce the frozen runner's deterministic empty-projection fallback."""
    text = row.get("model_visible_context_text")
    if isinstance(text, str) and text.strip():
        return text
    base = "No benchmark-native memory evidence is available in the admitted context."
    if row.get("condition_type") == "clean":
        return base
    return base + "\nAdditional admitted evidence is unavailable."


def context_membership(clean: Mapping[str, Any], corrupted: Mapping[str, Any]) -> dict[str, int]:
    clean_counts = Counter(canonical_items(clean))
    corrupt_counts = Counter(canonical_items(corrupted))
    return {
        "added_item_instances": sum((corrupt_counts - clean_counts).values()),
        "removed_item_instances": sum((clean_counts - corrupt_counts).values()),
    }


def parse_proxy_outcome(row: Mapping[str, Any] | None) -> tuple[str, bool | None, str | None]:
    """Return (state, binary score, unresolved reason) without inferring text equality."""
    if row is None:
        return "missing", None, "missing_condition_row"
    if row.get("status") != "success" or row.get("scoring_status") != "scored":
        return "unresolved", None, str(row.get("error_type") or row.get("scoring_status") or "not_scored")
    for field in ("correctness", "primary_correctness"):
        value = row.get(field)
        if isinstance(value, bool):
            return "scored", value, None
    primary_score = row.get("primary_score")
    if primary_score in (0, 0.0):
        return "scored", False, None
    if primary_score in (1, 1.0):
        return "scored", True, None
    return "unresolved", None, "no_usable_binary_score"


def _validate_condition_row(row: Mapping[str, Any]) -> None:
    condition_id = str(row.get("condition_id"))
    base_id = row.get("base_id")
    if not base_id:
        raise AuditError(f"{condition_id} lacks base_id")
    if row.get("condition_type") not in CONDITIONS:
        raise AuditError(f"{condition_id} has invalid condition_type")
    if not row.get("source_dataset") or not row.get("analysis_event_type"):
        raise AuditError(f"{condition_id} lacks source/event stratification fields")
    if not isinstance(row.get("internal_governance_state"), dict):
        raise AuditError(f"{condition_id} lacks internal_governance_state")
    if not isinstance(row.get("oracle_state"), dict):
        raise AuditError(f"{condition_id} lacks oracle_state")
    row_items = _context_items(row)
    spec = row.get("model_visible_context_spec")
    if not isinstance(spec, dict):
        raise AuditError(f"{condition_id} lacks model_visible_context_spec")
    if spec.get("item_count") != len(row_items):
        raise AuditError(f"{condition_id} item_count does not match memory items")
    text = row.get("model_visible_context_text")
    if not isinstance(text, str):
        raise AuditError(f"{condition_id} lacks model_visible_context_text")


def audit_condition_rows(
    condition_rows: Sequence[dict[str, Any]],
    base_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_ids = [row.get("base_id") for row in base_rows]
    if any(not value for value in base_ids):
        raise AuditError("Base manifest contains a missing base_id")
    if _duplicate_values(base_ids):
        raise AuditError("Base manifest contains duplicate base_ids")
    for row in condition_rows:
        _validate_condition_row(row)
    triplets, integrity = build_triplets(condition_rows)

    condition_bases = {str(row["base_id"]) for row in condition_rows}
    base_set = {str(base_id) for base_id in base_ids}
    if condition_bases != base_set:
        raise AuditError(
            f"Base/condition coverage mismatch: missing_clean_bases={len(base_set - condition_bases)}, "
            f"unknown_condition_bases={len(condition_bases - base_set)}"
        )

    state_changed = 0
    oracle_comparison_anomalies = 0
    independent_state_anomalies = 0
    context_changed = 0
    admission_changed = 0
    exclusion_changed = 0
    context_restored = 0
    corrupt_specs = 0
    for triplet in triplets:
        clean = triplet["conditions"]["clean"]
        corrupted = triplet["conditions"]["corrupted"]
        enforced = triplet["conditions"]["enforced"]
        source = triplet["source_dataset"]
        state_changed += corrupted["internal_governance_state"] != clean["internal_governance_state"]
        oracle_comparison_anomalies += corrupted["internal_governance_state"] != corrupted["oracle_state"]
        independent_state_anomalies += bool(_state_issues(corrupted["internal_governance_state"], source))
        changed = (
            canonical_items(clean) != canonical_items(corrupted)
            or effective_context_text(clean) != effective_context_text(corrupted)
        )
        context_changed += changed
        counts = context_membership(clean, corrupted)
        admission_changed += counts["added_item_instances"] > 0
        exclusion_changed += counts["removed_item_instances"] > 0
        context_restored += canonical_items(enforced) == canonical_items(clean) and (
            effective_context_text(enforced) == effective_context_text(clean)
        )
        corrupt_specs += bool(corrupted.get("corruption_specification"))

    counts = {
        "base_count": len(base_rows),
        "condition_count": len(condition_rows),
        "condition_counts_by_type": dict(sorted(Counter(row["condition_type"] for row in condition_rows).items())),
        "fault_pair_count": len(triplets),
        "fault_pairs_by_operator": dict(sorted(Counter(t["fault_operator"] for t in triplets).items())),
        "fault_pairs_by_source": dict(sorted(Counter(t["source_dataset"] for t in triplets).items())),
        "fault_pairs_by_event": dict(sorted(Counter(t["analysis_event_type"] for t in triplets).items())),
        "state_internal_changed_vs_clean": state_changed,
        "state_oracle_comparison_anomalies": oracle_comparison_anomalies,
        "state_independent_constraint_anomalies": independent_state_anomalies,
        "context_changed_clean_vs_corrupted": context_changed,
        "context_added_item_pairs": admission_changed,
        "context_removed_item_pairs": exclusion_changed,
        "context_restored_clean_vs_enforced": context_restored,
        "empty_manifest_context_fallback_conditions": sum(
            not str(row.get("model_visible_context_text") or "").strip() for row in condition_rows
        ),
        "corruption_specification_present": corrupt_specs,
    }
    checks = {
        "unique_base_ids": True,
        "unique_condition_ids": True,
        "one_clean_per_base": True,
        "complete_corrupted_enforced_pairs": True,
        "condition_base_coverage_matches_base_manifest": True,
        "all_condition_rows_have_required_fields": True,
        "all_pair_ids_follow_base_and_fault": True,
        "all_clean_and_enforced_contexts_restored": context_restored == len(triplets),
        "corruption_specification_present_on_every_pair": corrupt_specs == len(triplets),
    }
    if not all(checks.values()):
        raise AuditError(f"Condition audit failed: {checks}")
    return triplets, {"integrity": integrity, "counts": counts, "checks": checks}


def audit_provider_results(
    results_dir: Path,
    condition_rows: Sequence[dict[str, Any]],
    condition_sha_by_schema: Mapping[str, str],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    if not results_dir.is_dir():
        raise AuditError(f"Frozen provider results directory is absent: {results_dir}")
    expected_condition_ids = {str(row["condition_id"]) for row in condition_rows}
    condition_by_id = {str(row["condition_id"]): row for row in condition_rows}
    provider_results: dict[str, dict[str, dict[str, Any]]] = {}
    provider_report: dict[str, Any] = {}

    for file_stem in PROVIDER_FILES:
        path = results_dir / f"{file_stem}.jsonl"
        if not path.is_file():
            raise AuditError(f"Required frozen provider result file is absent: {path}")
        rows = read_jsonl(path)
        indexed: dict[str, dict[str, Any]] = {}
        labels: set[str] = set()
        resolved_conditions = 0
        unresolved_conditions = 0
        empty_projection_fallback_rows = 0
        status_counts: Counter[str] = Counter()
        for row in rows:
            condition_id = row.get("condition_id")
            if not condition_id:
                raise AuditError(f"{path} contains a row without condition_id")
            if condition_id in indexed:
                raise AuditError(f"{path} contains duplicate condition_id {condition_id}")
            indexed[str(condition_id)] = row
            labels.add(str(row.get("provider")))
            status, _score, reason = parse_proxy_outcome(row)
            status_counts[status] += 1
            if status == "scored":
                resolved_conditions += 1
            else:
                unresolved_conditions += 1
                status_counts[f"unresolved:{reason}"] += 1
        if set(indexed) != expected_condition_ids:
            raise AuditError(
                f"{path} condition-id coverage mismatch: missing={len(expected_condition_ids - set(indexed))}, "
                f"extra={len(set(indexed) - expected_condition_ids)}"
            )
        if len(labels) != 1 or not next(iter(labels)):
            raise AuditError(f"{path} does not expose one stable provider label: {sorted(labels)}")
        provider_label = next(iter(labels))
        if file_stem == "grok" and provider_label != "xai":
            raise AuditError(f"{path} expected provider label 'xai', found {provider_label!r}")

        metadata_mismatches = 0
        hash_mismatches = 0
        manifest_sha_mismatches = 0
        for condition_id, result in indexed.items():
            manifest_row = condition_by_id[condition_id]
            for field in (
                "base_id",
                "pair_id",
                "condition_type",
                "fault_operator",
                "analysis_event_type",
                "source_dataset",
            ):
                if result.get(field) != manifest_row.get(field):
                    metadata_mismatches += 1
            if result.get("model_visible_context_sha256") != sha256_text(effective_context_text(manifest_row)):
                hash_mismatches += 1
            schema_version = manifest_row.get("condition_schema_version")
            expected_sha = condition_sha_by_schema.get(str(schema_version))
            if expected_sha is None or result.get("manifest_sha256") != expected_sha:
                manifest_sha_mismatches += 1
            if not str(manifest_row.get("model_visible_context_text") or "").strip():
                empty_projection_fallback_rows += 1
        if metadata_mismatches or hash_mismatches or manifest_sha_mismatches:
            raise AuditError(
                f"{path} join verification failed: metadata={metadata_mismatches}, "
                f"context_hash={hash_mismatches}, manifest_hash={manifest_sha_mismatches}"
            )
        provider_results[provider_label] = indexed
        provider_report[provider_label] = {
            "artifact": path_locator(path),
            "artifact_sha256": sha256_file(path),
            "file_stem": file_stem,
            "row_count": len(rows),
            "unique_condition_ids": len(indexed),
            "scored_condition_rows": resolved_conditions,
            "unresolved_condition_rows": unresolved_conditions,
            "empty_projection_fallback_rows": empty_projection_fallback_rows,
            "status_counts": dict(sorted(status_counts.items())),
            "metadata_mismatches": metadata_mismatches,
            "context_hash_mismatches": hash_mismatches,
            "manifest_sha_mismatches": manifest_sha_mismatches,
        }
    return provider_results, {
        "provider_count": len(provider_results),
        "providers": sorted(provider_results),
        "combined_rows": sum(item["row_count"] for item in provider_report.values()),
        "per_provider": provider_report,
        "checks": {
            "six_provider_files_present": len(provider_results) == len(PROVIDER_FILES),
            "exact_condition_id_coverage": True,
            "unique_provider_condition_ids": True,
            "manifest_metadata_matches": True,
            "model_visible_context_hashes_match_effective_projection": True,
            "result_manifest_hash_matches_condition_schema_version": True,
        },
    }


def _fama_from_adjudication(decision_rows: Sequence[Mapping[str, Any]], prompt: str) -> float:
    criteria = json.loads(prompt)["criteria"]
    kinds = {str(item["criterion_id"]): str(item["evaluation_type"]) for item in criteria}
    decisions = {str(item["criterion_id"]): str(item["decision"]) for item in decision_rows}
    if set(decisions) != set(kinds) or any(value not in {"pass", "fail"} for value in decisions.values()):
        raise AuditError("Human adjudication does not cover the complete frozen criterion set")
    presence = [key for key, kind in kinds.items() if kind == "memory_presence"]
    absence = [key for key, kind in kinds.items() if kind == "forgetting_absence"]
    mpa = sum(decisions[key] == "pass" for key in presence) / len(presence) if presence else 1.0
    faa = sum(decisions[key] == "pass" for key in absence) / len(absence) if absence else 1.0
    return max(0.0, mpa - (len(absence) / len(kinds)) * (1.0 - faa))


def _final_block_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AuditError(f"Final Formal-300 block artifact is absent: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    converted = []
    for row in rows:
        converted.append(
            {
                **row,
                "scores": {
                    condition: float(row[f"{condition}_score"])
                    for condition in CONDITIONS
                },
            }
        )
    return converted


def audit_formal_panel(
    *,
    native_panel: str,
    observations_path: Path,
    public_blocks_path: Path,
    public_summary_path: Path,
    final_blocks_path: Path,
    final_metrics_path: Path,
    human_adjudication_path: Path,
    semantic_payloads_path: Path,
    condition_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load and validate either the public 297 or final human-adjudicated 298 panel."""
    if native_panel not in NATIVE_PANEL_MODES:
        raise AuditError(f"Unsupported native panel mode: {native_panel}")
    if not observations_path.is_file():
        raise AuditError(f"Formal observation artifact is absent: {observations_path}")
    observations = [dict(row) for row in read_jsonl(observations_path)]
    condition_by_id = {str(row["condition_id"]): row for row in condition_rows}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        block_id = row.get("pilot_block_id")
        if not block_id:
            raise AuditError("Formal observation lacks pilot_block_id")
        grouped[str(block_id)].append(row)
    bad_groups = {
        block_id: len(rows)
        for block_id, rows in grouped.items()
        if len(rows) != 3 or {row.get("condition") for row in rows} != set(CONDITIONS)
    }
    if bad_groups:
        raise AuditError(f"Formal observation groups are not complete triplets: {bad_groups}")

    human_overlay: dict[str, Any] | None = None
    if native_panel == "public_297":
        if not public_blocks_path.is_file() or not public_summary_path.is_file():
            raise AuditError("Public Formal-300 block or summary artifact is absent")
        resolved_blocks = read_jsonl(public_blocks_path)
        block_summary = _load_json(public_summary_path)
        expected_resolved = block_summary.get("resolved_matched_triplets")
        expected_unresolved = block_summary.get("unresolved_observations")
        expected_unresolved_ids: set[str] | None = None
        source_artifacts = {
            "observations": observations_path,
            "resolved_blocks": public_blocks_path,
            "summary": public_summary_path,
        }
    else:
        for path, label in (
            (final_blocks_path, "final block"),
            (final_metrics_path, "final metrics"),
            (human_adjudication_path, "human adjudication"),
            (semantic_payloads_path, "semantic payload"),
        ):
            if not path.is_file():
                raise AuditError(f"Required final Formal-300 {label} artifact is absent: {path}")
        resolved_blocks = _final_block_rows(final_blocks_path)
        block_summary = _load_json(final_metrics_path)
        expected_resolved = block_summary.get("resolved_matched_triplets")
        expected_unresolved = block_summary.get("unresolved_observations")
        expected_unresolved_ids = {
            str(row.get("observation_id"))
            for row in block_summary.get("unresolved_details") or []
            if row.get("observation_id")
        }
        if len(expected_unresolved_ids) != expected_unresolved:
            raise AuditError(
                "Final metrics unresolved details do not match the declared unresolved count"
            )
        human_rows = [
            row
            for row in read_jsonl(human_adjudication_path)
            if row.get("resolution_method") == "human_adjudication"
        ]
        if len(human_rows) != 1:
            raise AuditError(f"Expected one frozen human adjudication, found {len(human_rows)}")
        human = human_rows[0]
        observation_id = str(human.get("observation_id"))
        payload_by_id = {
            str(row["observation_id"]): row for row in read_jsonl(semantic_payloads_path)
        }
        payload = payload_by_id.get(observation_id)
        if payload is None:
            raise AuditError(f"Human adjudication observation lacks a semantic payload: {observation_id}")
        adjudicated_score = _fama_from_adjudication(human.get("decisions") or [], str(payload.get("prompt")))
        matching = [row for row in observations if str(row.get("observation_id")) == observation_id]
        if len(matching) != 1:
            raise AuditError(f"Human adjudication observation does not uniquely join observations: {observation_id}")
        if matching[0].get("condition") != payload.get("condition"):
            raise AuditError("Human adjudication condition does not match the frozen payload")
        matching[0]["formal_score"] = adjudicated_score
        matching[0]["resolution_method"] = "human_adjudication"
        matching[0]["resolution_source"] = "private_human_adjudication"
        human_overlay = {
            "observation_id": observation_id,
            "pilot_block_id": str(human.get("pilot_block_id")),
            "condition": str(matching[0].get("condition")),
            "score": adjudicated_score,
            "decision_count": len(human.get("decisions") or []),
            "path": path_locator(human_adjudication_path),
            "sha256": sha256_file(human_adjudication_path),
        }
        if block_summary.get("human_adjudicated_observations") != 1:
            raise AuditError("Final metrics do not declare exactly one human adjudication")
        source_artifacts = {
            "observations": observations_path,
            "final_blocks": final_blocks_path,
            "final_metrics": final_metrics_path,
            "human_adjudication": human_adjudication_path,
            "semantic_payloads": semantic_payloads_path,
        }

    resolved_ids = [str(row.get("pilot_block_id")) for row in resolved_blocks]
    if _duplicate_values(resolved_ids):
        raise AuditError("Resolved Formal-300 blocks contain duplicates")
    unknown_resolved = set(resolved_ids) - set(grouped)
    if unknown_resolved:
        raise AuditError(f"Resolved Formal blocks absent from observations: {sorted(unknown_resolved)[:5]}")
    unresolved_block_ids = sorted(set(grouped) - set(resolved_ids))
    unresolved_observations = sum(row.get("formal_score") is None for row in observations)
    actual_unresolved_ids = {
        str(row.get("observation_id"))
        for row in observations
        if row.get("formal_score") is None
    }
    if expected_resolved != len(resolved_blocks):
        raise AuditError(
            f"Formal summary resolved count mismatch: declared={expected_resolved}, blocks={len(resolved_blocks)}"
        )
    if expected_unresolved != unresolved_observations:
        raise AuditError(
            f"Formal summary unresolved count mismatch: declared={expected_unresolved}, "
            f"observations={unresolved_observations}"
        )
    if expected_unresolved_ids is not None and expected_unresolved_ids != actual_unresolved_ids:
        raise AuditError(
            "Final metrics unresolved observation IDs do not match the resolved observations"
        )
    for block_id in unresolved_block_ids:
        if not any(row.get("formal_score") is None for row in grouped[block_id]):
            raise AuditError(f"Unresolved Formal block has no unresolved observation: {block_id}")

    block_by_id = {str(row["pilot_block_id"]): row for row in resolved_blocks}
    enriched_blocks: list[dict[str, Any]] = []
    for block_id in sorted(resolved_ids):
        block = block_by_id[block_id]
        by_condition = {str(row["condition"]): row for row in grouped[block_id]}
        declared_scores = (
            block.get("scores")
            if "scores" in block
            else {condition: float(block[f"{condition}_score"]) for condition in CONDITIONS}
        )
        manifest_rows: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            condition_id = str(by_condition[condition].get("condition_id"))
            manifest_row = condition_by_id.get(condition_id)
            if manifest_row is None:
                raise AuditError(f"Formal condition_id does not join the manifest: {condition_id}")
            manifest_rows[condition] = manifest_row
            observed_score = by_condition[condition].get("formal_score")
            if observed_score is None or float(observed_score) != float(declared_scores[condition]):
                raise AuditError(
                    f"Formal block score mismatch for {block_id}/{condition}: "
                    f"observation={observed_score}, block={declared_scores[condition]}"
                )
        bases = {row.get("base_id") for row in manifest_rows.values()}
        sources = {row.get("source_dataset") for row in manifest_rows.values()}
        events = {row.get("analysis_event_type") for row in manifest_rows.values()}
        if len(bases) != 1 or len(sources) != 1 or len(events) != 1:
            raise AuditError(f"Formal block manifest join drifts within {block_id}")
        source = next(iter(sources))
        fault_operator = str(manifest_rows["corrupted"].get("fault_operator"))
        if str(block.get("source")) != source:
            raise AuditError(f"Formal block source mismatch for {block_id}")
        if str(block.get("fault_type")) != fault_operator:
            raise AuditError(f"Formal block fault mismatch for {block_id}")
        if str(block.get("generating_provider")) != str(by_condition["clean"].get("generating_provider")):
            raise AuditError(f"Formal block provider mismatch for {block_id}")
        values = [float(value) for value in declared_scores.values()]
        if source == "longmemeval" and any(value not in (0.0, 1.0) for value in values):
            raise AuditError(f"LongMemEval Formal scores are not binary in {block_id}")
        if source == "memora" and any(value < 0 or value > 1 for value in values):
            raise AuditError(f"Memora FAMA score is outside [0,1] in {block_id}")
        answer_hashes = {
            condition: by_condition[condition].get("answer_sha256")
            for condition in CONDITIONS
        }
        enriched_blocks.append(
            {
                "pilot_block_id": block_id,
                "source": source,
                "generating_provider": str(block.get("generating_provider")),
                "fault_type": str(block.get("fault_type")),
                "fault_operator": fault_operator,
                "base_id": next(iter(bases)),
                "analysis_event_type": next(iter(events)),
                "condition_ids": {
                    condition: str(manifest_rows[condition]["condition_id"])
                    for condition in CONDITIONS
                },
                "scores": {
                    condition: float(declared_scores[condition])
                    for condition in CONDITIONS
                },
                "answer_sha256": answer_hashes,
            }
        )

    report = {
        "mode": native_panel,
        "observation_rows": len(observations),
        "observed_block_count": len(grouped),
        "resolved_matched_triplets": len(resolved_blocks),
        "unresolved_blocks": len(unresolved_block_ids),
        "unresolved_observations": unresolved_observations,
        "resolved_observations": len(resolved_blocks) * 3,
        "sources": dict(sorted(Counter(str(row["source"]) for row in enriched_blocks).items())),
        "generating_providers": dict(
            sorted(Counter(str(row["generating_provider"]) for row in enriched_blocks).items())
        ),
        "fault_types": dict(sorted(Counter(str(row["fault_type"]) for row in enriched_blocks).items())),
        "score_units": {
            "longmemeval": "binary_correctness",
            "memora": "fama_continuous_0_to_1",
        },
        "unresolved_block_ids": unresolved_block_ids,
        "human_adjudication": human_overlay,
        "source_artifacts": {
            key: {"path": path_locator(path), "sha256": sha256_file(path)}
            for key, path in source_artifacts.items()
        },
        "checks": {
            "three_observations_per_observed_block": True,
            "resolved_blocks_unique": True,
            "resolved_condition_ids_join_manifest": True,
            "block_scores_match_observations": True,
            "longmemeval_scores_binary": True,
            "memora_scores_continuous_0_to_1": True,
            "unresolved_blocks_preserved": True,
            "human_adjudication_validated": (
                human_overlay is not None if native_panel == "paper_final_298" else True
            ),
        },
    }
    return enriched_blocks, observations, report


def load_evidence_support(path: Path, triplets: Sequence[dict[str, Any]]) -> tuple[dict[str, bool], dict[str, Any]]:
    if not path.is_file():
        raise AuditError(f"Governance evidence base-level audit is absent: {path}")
    support: dict[str, bool] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            base_id = row.get("base_id")
            if not base_id:
                raise AuditError(f"{path} contains a row without base_id")
            support[base_id] = str(row.get("fully_supported", "")).strip().lower() == "true"
    required = {triplet["base_id"] for triplet in triplets}
    missing = required - set(support)
    if missing:
        raise AuditError(f"Evidence support missing for {len(missing)} eligible bases")
    unsupported = sorted(base for base in required if not support[base])
    report = {
        "artifact": path_locator(path),
        "artifact_sha256": sha256_file(path),
        "base_rows": len(support),
        "eligible_bases": len(required),
        "fully_supported_eligible_bases": len(required) - len(unsupported),
        "unsupported_bases": unsupported,
        "checks": {
            "every_eligible_base_present": True,
            "every_eligible_base_fully_supported": not unsupported,
        },
    }
    if unsupported:
        raise AuditError(f"{len(unsupported)} eligible bases are not fully evidence-supported")
    return support, report


def audit_repository(
    *,
    condition_manifest: Path,
    base_manifest: Path,
    condition_freeze: Path | None,
    base_freeze: Path | None,
    results_dir: Path,
    native_panel: str = "paper_final_298",
    formal_observations: Path,
    formal_blocks: Path,
    formal_summary: Path,
    final_formal_blocks: Path,
    final_formal_metrics: Path,
    human_adjudication: Path,
    semantic_payloads: Path,
    evidence_base_level: Path,
) -> AuditBundle:
    if not condition_manifest.is_file():
        raise AuditError(
            f"Frozen condition manifest is unavailable at {condition_manifest}. "
            "For the public checkout, pass the read-only experimental repository path with "
            "--condition-manifest."
        )
    if not base_manifest.is_file():
        raise AuditError(f"Frozen base manifest is unavailable at {base_manifest}")
    if condition_freeze is None or base_freeze is None:
        raise AuditError("Both condition and base freeze records are required")

    condition_freeze_report = _verify_freeze(condition_manifest, condition_freeze, kind="condition")
    base_freeze_report = _verify_freeze(base_manifest, base_freeze, kind="base")
    condition_rows = read_jsonl(condition_manifest)
    base_rows = read_jsonl(base_manifest)
    triplets, condition_report = audit_condition_rows(condition_rows, base_rows)
    superseded_sha256 = condition_freeze_report.get("superseded_manifest_sha256")
    if not superseded_sha256:
        raise AuditError("Condition freeze does not record the superseded v2 manifest SHA256")
    condition_sha_by_schema = {
        "main-governance-condition-v2": str(superseded_sha256),
        "main-governance-condition-v3": str(condition_freeze_report["manifest_sha256"]),
    }
    provider_results, provider_report = audit_provider_results(
        results_dir, condition_rows, condition_sha_by_schema
    )
    formal_blocks_enriched, formal_observations, formal_report = audit_formal_panel(
        native_panel=native_panel,
        observations_path=formal_observations,
        public_blocks_path=formal_blocks,
        public_summary_path=formal_summary,
        final_blocks_path=final_formal_blocks,
        final_metrics_path=final_formal_metrics,
        human_adjudication_path=human_adjudication,
        semantic_payloads_path=semantic_payloads,
        condition_rows=condition_rows,
    )
    evidence_support, evidence_report = load_evidence_support(evidence_base_level, triplets)

    report = {
        "status": "verified_offline_inputs",
        "api_calls_made": 0,
        "schema_version": "jss-rq1-observability-audit-v2",
        "inputs": {
            "condition_manifest": {
                "path": path_locator(condition_manifest),
                "sha256": sha256_file(condition_manifest),
                "freeze": condition_freeze_report,
            },
            "base_manifest": {
                "path": path_locator(base_manifest),
                "sha256": sha256_file(base_manifest),
                "freeze": base_freeze_report,
            },
            "native_panel_mode": native_panel,
        },
        "condition_manifest": condition_report,
        "provider_results": provider_report,
        "formal_panel": formal_report,
        "governance_evidence": evidence_report,
        "join_keys": {
            "clean_condition": "base_id -> unique clean condition_id",
            "matched_pair": "pair_id == base_id + '::' + fault_operator",
            "provider_result": "condition_id plus manifest metadata and model_visible_context_sha256",
            "formal_block": "pilot_block_id -> three condition rows; condition_id -> manifest",
        },
        "limitations": [
            "Corrupted internal_governance_state is a copy of oracle_state in the frozen manifest; "
            "injected fault labels cannot be used as independent state detections.",
            "Some repaired conditions have an empty manifest context; the frozen runners used a "
            "deterministic neutral fallback, which is reconstructed for context-hash verification.",
            "Frozen provider files expose binary scoring outcomes and context hashes, not raw response text.",
            "The Formal-300 panel is a fixed selected validation panel, not a random cohort sample.",
            "LongMemEval binary correctness and Memora FAMA are separate score units.",
            "The paper-final 298 panel consumes one existing blinded human adjudication; "
            "the public 297 panel does not and remains a separate sensitivity analysis.",
        ],
    }
    return AuditBundle(
        report=report,
        triplets=triplets,
        provider_results=provider_results,
        formal_observations=formal_observations,
        formal_blocks=formal_blocks_enriched,
        evidence_support=evidence_support,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-manifest", type=Path, default=DEFAULT_CONDITION_MANIFEST)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--condition-freeze", type=Path, default=DEFAULT_CONDITION_FREEZE)
    parser.add_argument("--base-freeze", type=Path, default=DEFAULT_BASE_FREEZE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--native-panel", choices=NATIVE_PANEL_MODES, default="paper_final_298")
    parser.add_argument("--formal-observations", type=Path, default=DEFAULT_FORMAL_OBSERVATIONS)
    parser.add_argument("--formal-blocks", type=Path, default=DEFAULT_FORMAL_BLOCKS)
    parser.add_argument("--formal-summary", type=Path, default=DEFAULT_FORMAL_SUMMARY)
    parser.add_argument("--final-formal-blocks", type=Path, default=DEFAULT_FINAL_FORMAL_BLOCKS)
    parser.add_argument("--final-formal-metrics", type=Path, default=DEFAULT_FINAL_FORMAL_METRICS)
    parser.add_argument("--human-adjudication", type=Path, default=DEFAULT_HUMAN_ADJUDICATION)
    parser.add_argument("--semantic-payloads", type=Path, default=DEFAULT_SEMANTIC_PAYLOADS)
    parser.add_argument("--evidence-base-level", type=Path, default=DEFAULT_EVIDENCE_BASE_LEVEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        bundle = audit_repository(
            condition_manifest=args.condition_manifest,
            base_manifest=args.base_manifest,
            condition_freeze=args.condition_freeze,
            base_freeze=args.base_freeze,
            results_dir=args.results_dir,
            native_panel=args.native_panel,
            formal_observations=args.formal_observations,
            formal_blocks=args.formal_blocks,
            formal_summary=args.formal_summary,
            final_formal_blocks=args.final_formal_blocks,
            final_formal_metrics=args.final_formal_metrics,
            human_adjudication=args.human_adjudication,
            semantic_payloads=args.semantic_payloads,
            evidence_base_level=args.evidence_base_level,
        )
    except (AuditError, FileNotFoundError, FileExistsError, json.JSONDecodeError) as exc:
        print(f"JSS_RQ1_AUDIT_BLOCKED={exc}", file=sys.stderr)
        return 2
    if args.output:
        write_json(args.output, bundle.report)
    counts = bundle.report["condition_manifest"]["counts"]
    formal = bundle.report["formal_panel"]
    print(f"JSS_RQ1_AUDIT_STATUS={bundle.report['status']}")
    print(f"JSS_RQ1_BASES={counts['base_count']}")
    print(f"JSS_RQ1_CONDITIONS={counts['condition_count']}")
    print(f"JSS_RQ1_FAULT_PAIRS={counts['fault_pair_count']}")
    print(f"JSS_RQ1_NATIVE_PANEL={formal['mode']}")
    print(f"JSS_RQ1_PROVIDERS={bundle.report['provider_results']['provider_count']}")
    print(f"JSS_RQ1_PROVIDER_ROWS={bundle.report['provider_results']['combined_rows']}")
    print(f"JSS_RQ1_FORMAL_RESOLVED_TRIPLETS={formal['resolved_matched_triplets']}")
    print(f"JSS_RQ1_FORMAL_UNRESOLVED_BLOCKS={formal['unresolved_blocks']}")
    print("JSS_RQ1_API_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
