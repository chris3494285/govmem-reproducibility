#!/usr/bin/env python3
"""Evaluate deterministic RQ4 control-plane enforcement policies."""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from govmem.governance.enforcement import decide_admission  # noqa: E402
from scripts.run_governance_fault_rq1 import load_manifest  # noqa: E402


POLICIES = ("permissive", "deny_by_default", "version_validation", "conflict_quarantine", "revalidate_on_uncertainty")
RQ1A_MANIFEST = "data/governance_fault_rq1_pilot_frozen_v2.jsonl"
RQ1B_MANIFEST = "data/governance_fault_rq1_restrictive_frozen.jsonl"
ACTION_ORDER = ("admit", "deny", "quarantine", "revalidate", "not_applicable")


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def observed_state_kind(row: dict[str, Any]) -> str:
    """Classify only deterministic state properties needed by the policies."""
    dimension = row.get("dimension")
    fault = row.get("fault_spec") or {}
    # A stale operator is explicitly an invalid version/lifecycle assertion
    # even when its serialized status still says ``active``.  The permissive
    # baseline intentionally retains the existing enforcement behavior; the
    # robust policies use this classification to reject stale state.
    if _value(fault, "operator") == "stale" and dimension in {"temporal", "deletion"}:
        return "invalid"
    state = row.get("observed_governance_state") or {}
    if dimension == "authorization":
        if _value(state, "authorization_assertions", ()):
            return "conflicting"
        authorization = _value(state, "authorization")
        decision = _value(authorization, "decision")
        return "valid" if decision == "allow" else "invalid" if decision == "deny" else "missing"
    if dimension == "deletion":
        deletion = _value(state, "deletion")
        status = _value(deletion, "status")
        return "valid" if status == "active" else "invalid" if status == "deleted" else "missing"
    if dimension == "temporal":
        if _value(state, "temporal_assertions", ()):
            return "conflicting"
        temporal = _value(state, "temporal")
        status = _value(temporal, "status")
        return "valid" if status == "valid" else "invalid" if status == "expired" else "missing"
    return "uncertain"


def _normal_action(row: dict[str, Any]) -> str:
    return "admit" if decide_admission(row).actual_admission == "allow" else "deny"


def _has_deterministic_current_temporal_state(row: dict[str, Any]) -> bool:
    """Return whether provenance exposes a current value independently of gold."""
    metadata = row.get("source_metadata") or {}
    current_value = metadata.get("current_value")
    current_session = metadata.get("current_session") or {}
    return current_value is not None and current_session.get("session_id") is not None


def _version_validation_action(row: dict[str, Any], kind: str) -> tuple[str, str]:
    if row.get("dimension") == "temporal" and _value(row.get("fault_spec") or {}, "operator") == "stale":
        if _has_deterministic_current_temporal_state(row):
            return "admit", "stale_version_filtered_current_preserved"
        return "deny", "stale_version_unresolved"
    if kind == "conflicting":
        return "deny", "conflicting_version_no_deterministic_winner"
    if kind == "missing":
        return "deny", "missing_version_unresolved"
    return ("admit" if kind == "valid" else "deny"), f"version_validation_{kind}"


def policy_action(policy: str, row: dict[str, Any]) -> tuple[str, bool, str]:
    """Return action, applicability, and deterministic policy rationale."""
    kind = observed_state_kind(row)
    if policy == "permissive":
        return _normal_action(row), True, f"permissive_{kind}"
    if policy == "deny_by_default":
        return ("admit" if kind == "valid" else "deny"), True, f"deny_by_default_{kind}"
    if policy == "version_validation":
        if row.get("dimension") not in {"temporal", "deletion"}:
            return "not_applicable", False, "version_validation_not_applicable_to_dimension"
        action, rationale = _version_validation_action(row, kind)
        return action, True, rationale
    if policy == "conflict_quarantine":
        applicable = row.get("dimension") in {"authorization", "temporal"}
        if not applicable:
            return "not_applicable", False, "conflict_quarantine_not_applicable_to_dimension"
        if kind == "conflicting":
            return "quarantine", True, "conflicting_state_quarantined"
        return _normal_action(row), True, f"conflict_quarantine_no_special_effect_{kind}"
    if policy == "revalidate_on_uncertainty":
        if kind in {"missing", "conflicting", "uncertain"}:
            return "revalidate", True, f"revalidate_{kind}"
        return ("admit" if kind == "valid" else "deny"), True, f"revalidate_clear_{kind}"
    raise ValueError(f"Unknown RQ4 policy: {policy}")


def intervention_applied(policy: str, row: dict[str, Any]) -> bool:
    kind = observed_state_kind(row)
    dimension = row.get("dimension")
    if policy == "permissive":
        return False
    if policy == "deny_by_default":
        return kind != "valid"
    if policy == "version_validation":
        return dimension in {"temporal", "deletion"} and (
            kind != "valid" or _value(row.get("fault_spec") or {}, "operator") == "stale"
        )
    if policy == "conflict_quarantine":
        return dimension in {"authorization", "temporal"} and kind == "conflicting"
    if policy == "revalidate_on_uncertainty":
        return kind in {"missing", "conflicting", "uncertain"}
    raise ValueError(f"Unknown RQ4 policy: {policy}")


def _state_status(state: Any, dimension: str) -> Any:
    return _value(_value(state or {}, dimension), "status")


def governance_exposure(row: dict[str, Any], action: str, policy: str | None = None) -> tuple[bool | None, str]:
    """Classify model-visible governance exposure from frozen state metadata.

    ``ambiguous`` is deliberately separate from both safe and unsafe: an
    admit action over incompatible assertions does not establish a
    deterministic operative state.
    """
    dimension = row.get("dimension")
    kind = observed_state_kind(row)
    if kind == "conflicting":
        return (None, "ambiguous") if action == "admit" else (False, "safe")

    if dimension == "temporal":
        operator = _value(row.get("fault_spec") or {}, "operator")
        if operator == "stale":
            if policy == "version_validation" and action == "admit" and _has_deterministic_current_temporal_state(row):
                return False, "safe"
            return (action == "admit", "unsafe" if action == "admit" else "safe")
        # Missing and restrictive expired conditions do not expose a stale
        # predecessor as current.  Their harm, when any, is access rejection.
        return False, "safe"

    if dimension == "deletion":
        gold_state = row.get("gold_governance_state") or {}
        gold_status = _state_status(gold_state, "deletion")
        operator = _value(row.get("fault_spec") or {}, "operator")
        # A deleted current item or a stale deleted predecessor is unsafe if
        # the policy admits it into the governed context.
        unsafe = action == "admit" and (gold_status == "deleted" or operator == "stale")
        return unsafe, "unsafe" if unsafe else "safe"

    if dimension == "authorization":
        gold = row.get("gold_admission_decision")
        if gold not in {"allow", "deny"}:
            return None, "unscorable"
        unsafe = gold == "deny" and action == "admit"
        return unsafe, "unsafe" if unsafe else "safe"

    return None, "unscorable"


def state_selection(row: dict[str, Any], action: str, policy: str) -> tuple[str | None, bool | None]:
    """Describe which temporal state can be model-visible after enforcement."""
    if row.get("dimension") != "temporal":
        return None, None
    operator = _value(row.get("fault_spec") or {}, "operator")
    if operator == "stale":
        repaired = policy == "version_validation" and action == "admit" and _has_deterministic_current_temporal_state(row)
        if repaired:
            return "current", True
        if action == "admit":
            return "stale_predecessor", False
        return "none", True
    if observed_state_kind(row) == "conflicting":
        return ("ambiguous", False) if action == "admit" else ("none", True)
    if observed_state_kind(row) == "valid" and action == "admit":
        return "current", True
    return "none", True


def _instance(row: dict[str, Any], policy: str, clean_metrics: dict[str, bool | None] | None) -> dict[str, Any]:
    action, applicable, rationale = policy_action(policy, row)
    gold = row.get("gold_admission_decision")
    unsafe_exposure, exposure_status = governance_exposure(row, action, policy)
    operative_state, stale_predecessor_filtered = state_selection(row, action, policy)
    false_rejection = gold == "allow" and action in {"deny", "quarantine", "revalidate"} if gold in {"allow", "deny"} else None
    hard_false_rejection = gold == "allow" and action == "deny" if gold in {"allow", "deny"} else None
    deferred_access = gold == "allow" and action in {"quarantine", "revalidate"} if gold in {"allow", "deny"} else None
    instance = {
        "policy": policy,
        "run_id": row.get("run_id"),
        "base_example_id": row.get("example_id"),
        "source_dataset": row.get("source_dataset"),
        "governance_dimension": row.get("dimension"),
        "condition_type": row.get("condition_type"),
        "fault_type": (row.get("fault_spec") or {}).get("operator", "clean"),
        "fault_direction": (row.get("fault_spec") or {}).get("direction", "clean"),
        "observed_state_kind": observed_state_kind(row),
        "gold_admission_decision": gold,
        "action": action,
        "policy_applicable": applicable,
        "intervention_applied": intervention_applied(policy, row),
        "policy_rationale": rationale,
        "policy_mechanism": rationale,
        "operative_state": operative_state,
        "stale_predecessor_filtered": stale_predecessor_filtered,
        "governance_exposure_status": exposure_status,
        "ambiguous_exposure": exposure_status == "ambiguous" if applicable else None,
        "unsafe_exposure": unsafe_exposure,
        "false_rejection": false_rejection,
        "hard_false_rejection": hard_false_rejection,
        "deferred_access": deferred_access,
        "downstream_reusable": False,
        "reusable_condition": None,
        "delta_unsafe_exposure_vs_clean": None,
        "delta_false_rejection_vs_clean": None,
    }
    if row.get("condition_type") == "fault" and clean_metrics is not None:
        if isinstance(unsafe_exposure, bool) and isinstance(clean_metrics["unsafe_exposure"], bool):
            instance["delta_unsafe_exposure_vs_clean"] = int(unsafe_exposure) - int(clean_metrics["unsafe_exposure"])
        if isinstance(false_rejection, bool) and isinstance(clean_metrics["false_rejection"], bool):
            instance["delta_false_rejection_vs_clean"] = int(false_rejection) - int(clean_metrics["false_rejection"])
    return instance


def evaluate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate rows in stable manifest order for every policy."""
    result: list[dict[str, Any]] = []
    for policy in POLICIES:
        clean_by_base: dict[str, dict[str, bool | None]] = {}
        for row in rows:
            if row.get("condition_type") == "clean":
                clean_instance = _instance(row, policy, None)
                clean_by_base[str(row.get("example_id"))] = {
                    key: clean_instance[key]
                    for key in ("unsafe_exposure", "false_rejection")
                }
        for row in rows:
            result.append(_instance(row, policy, clean_by_base.get(str(row.get("example_id")))))
    return result


def _bool_values(rows: list[dict[str, Any]], field: str, *, applicable_only: bool = True) -> list[bool]:
    return [row[field] for row in rows if (not applicable_only or row["policy_applicable"]) and isinstance(row.get(field), bool)]


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate_group(policy: str, key: tuple[str, str, str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [row for row in rows if row["policy_applicable"]]
    metrics = {}
    for field, output in (
        ("unsafe_exposure", "unsafe_exposure"),
        ("false_rejection", "false_rejection"),
        ("hard_false_rejection", "hard_false_rejection"),
        ("deferred_access", "deferred_access"),
    ):
        values = _bool_values(applicable, field, applicable_only=False)
        metrics[f"{output}_n"] = sum(values)
        metrics[f"{output}_scored_n"] = len(values)
        metrics[f"{output}_rate"] = _rate(values)
    deltas = {}
    for field in ("delta_unsafe_exposure_vs_clean", "delta_false_rejection_vs_clean"):
        values = [row[field] for row in rows if isinstance(row.get(field), int) and not isinstance(row.get(field), bool)]
        deltas[f"mean_{field}"] = sum(values) / len(values) if values else None
        deltas[f"{field}_scored_n"] = len(values)
    counts = {f"{action}_n": sum(row["action"] == action for row in rows) for action in ACTION_ORDER}
    interventions = [row for row in applicable if row["intervention_applied"]]
    return {
        "policy": policy,
        "source_dataset": key[0],
        "governance_dimension": key[1],
        "fault_type": key[2],
        "fault_direction": key[3],
        "n": len(rows),
        "applicable_n": len(applicable),
        "intervention_applied_n": len(interventions),
        "intervention_rate": _rate([True] * len(interventions) + [False] * (len(applicable) - len(interventions))),
        **metrics,
        **deltas,
        **counts,
    }


def aggregate_policy_rows(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        groups[(row["policy"], str(row["source_dataset"]), str(row["governance_dimension"]), str(row["fault_type"]), str(row["fault_direction"]))].append(row)
    return [
        _aggregate_group(policy, key, rows)
        for (policy, *key), rows in sorted(groups.items(), key=lambda item: item[0])
    ]


def _scope_summary(policy: str, dataset: str, dimension: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [row for row in rows if row["policy_applicable"]]
    cohort_keys = frozenset(_instance_key(row) for row in applicable)
    intervention_n = sum(row["intervention_applied"] for row in applicable)
    return {
        "policy": policy,
        "source_dataset": dataset,
        "governance_dimension": dimension,
        "unsafe_exposure_rate": _rate(_bool_values(applicable, "unsafe_exposure", applicable_only=False)),
        "hard_false_rejection_rate": _rate(_bool_values(applicable, "hard_false_rejection", applicable_only=False)),
        "deferred_access_rate": _rate(_bool_values(applicable, "deferred_access", applicable_only=False)),
        "applicable_n": len(applicable),
        "intervention_applied_n": intervention_n,
        "intervention_rate": intervention_n / len(applicable) if applicable else None,
        "cohort_n": len(cohort_keys),
        "cohort_identity": hashlib.sha256(",".join(sorted(cohort_keys)).encode("utf-8")).hexdigest() if cohort_keys else None,
        "_cohort_keys": cohort_keys,
    }


def _instance_key(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(field))
        for field in (
            "source_dataset",
            "governance_dimension",
            "base_example_id",
            "condition_type",
            "fault_type",
            "fault_direction",
            "run_id",
        )
    )


def _tradeoffs(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scope: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        by_scope[(str(row["source_dataset"]), str(row["governance_dimension"]))].append(row)
    rows = []
    for (dataset, dimension), scope_rows in sorted(by_scope.items()):
        summaries = [_scope_summary(policy, dataset, dimension, [row for row in scope_rows if row["policy"] == policy]) for policy in POLICIES]
        for candidate in summaries:
            dominated_by = []
            incomparable_with = []
            candidate_values = [candidate[field] for field in ("unsafe_exposure_rate", "hard_false_rejection_rate", "deferred_access_rate")]
            if all(value is not None for value in candidate_values):
                for other in summaries:
                    if other["policy"] == candidate["policy"]:
                        continue
                    if not candidate["_cohort_keys"] or not other["_cohort_keys"]:
                        continue
                    if candidate["_cohort_keys"] != other["_cohort_keys"]:
                        incomparable_with.append(other["policy"])
                        continue
                    other_values = [other[field] for field in ("unsafe_exposure_rate", "hard_false_rejection_rate", "deferred_access_rate")]
                    if all(value is not None for value in other_values) and all(o <= c for o, c in zip(other_values, candidate_values)) and any(o < c for o, c in zip(other_values, candidate_values)):
                        dominated_by.append(other["policy"])
            rows.append({key: value for key, value in {**candidate, "pareto_dominated_by": sorted(dominated_by), "pareto_incomparable_with": sorted(set(incomparable_with))}.items() if key != "_cohort_keys"})
    return rows


def _overall(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for policy in POLICIES:
        values = [row for row in instances if row["policy"] == policy]
        summary = _aggregate_group(policy, ("__overall__", "__overall__", "__all__", "__all__"), values)
        summary["aggregation_note"] = "pooled descriptive count; not a homogeneous cross-dataset comparison"
        rows.append(summary)
    return rows


def evaluate(manifest_paths: tuple[str | Path, str | Path] = (RQ1A_MANIFEST, RQ1B_MANIFEST)) -> dict[str, Any]:
    rq1a = load_manifest(manifest_paths[0], require_full=True)
    rq1b = load_manifest(manifest_paths[1], require_full=True)
    instances = evaluate_rows(rq1a) + evaluate_rows(rq1b)
    return {
        "analysis_version": "rq4-control-plane-v2",
        "policies": list(POLICIES),
        "source_manifests": {"rq1a": str(manifest_paths[0]), "rq1b": str(manifest_paths[1])},
        "instances": instances,
        "policy_groups": aggregate_policy_rows(instances),
        "overall_policy_summaries": _overall(instances),
        "tradeoffs": _tradeoffs(instances),
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path, fields: list[str] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or sorted({field for row in rows for field in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rq1a-manifest", default=RQ1A_MANIFEST)
    parser.add_argument("--rq1b-manifest", default=RQ1B_MANIFEST)
    parser.add_argument("--output-dir", default="results/rq4")
    args = parser.parse_args(argv)
    summary = evaluate((args.rq1a_manifest, args.rq1b_manifest))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "rq4_policy_evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(summary["policy_groups"], output / "rq4_policy_evaluation.csv")
    write_csv(summary["tradeoffs"], output / "rq4_policy_tradeoffs.csv")
    write_csv(summary["instances"], output / "rq4_policy_instances.csv")
    print(json.dumps({"policies": summary["policies"], "instances": len(summary["instances"]), "output_dir": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
