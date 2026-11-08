"""Validation for governance-fault benchmark rows and paired conditions."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable

from .schema import RunnableGovernanceCondition


def validate_state(state) -> None:
    if state.authorization and state.authorization.decision not in {"allow", "deny"}:
        raise ValueError("Invalid authorization decision")
    if state.deletion and state.deletion.status not in {"active", "deleted"}:
        raise ValueError("Invalid deletion status")
    if state.temporal:
        if state.temporal.status not in {"valid", "expired"}:
            raise ValueError("Invalid temporal status")
        if state.temporal.valid_from and state.temporal.valid_until and state.temporal.valid_from > state.temporal.valid_until:
            raise ValueError("Temporal interval is reversed")
    if state.authorization_assertions:
        decisions = {item.decision for item in state.authorization_assertions}
        if decisions != {"allow", "deny"}:
            raise ValueError("Authorization conflict must contain allow and deny")
    if state.temporal_assertions:
        statuses = {item.status for item in state.temporal_assertions}
        if len(statuses) < 2:
            raise ValueError("Temporal conflict must contain different validity assertions")


def validate_condition(row: RunnableGovernanceCondition) -> None:
    allowed = {
        "authorization": {"missing", "flipped", "conflicting"},
        "deletion": {"missing", "stale", "flipped"},
        "temporal": {"missing", "stale", "conflicting", "expired"},
    }
    if row.condition_type == "clean":
        if row.fault_spec is not None or row.observed_governance_state != row.gold_governance_state:
            raise ValueError("Clean condition must equal gold and have no fault spec")
    else:
        if row.fault_spec is None:
            raise ValueError("Fault condition requires a fault spec")
        if row.fault_spec.dimension != row.dimension:
            raise ValueError("Fault dimension does not match row dimension")
        if row.fault_spec.operator not in allowed[row.dimension]:
            raise ValueError(f"Unsupported operator/dimension pair: {row.dimension}/{row.fault_spec.operator}")
    validate_state(row.gold_governance_state)
    validate_state(row.observed_governance_state)
    if row.gold_admission_decision not in {"allow", "deny"}:
        raise ValueError("Invalid gold admission decision")
    if row.condition_type == "fault":
        target = row.dimension
        for attr in ("authorization", "deletion", "temporal"):
            if attr != target and getattr(row.gold_governance_state, attr) != getattr(row.observed_governance_state, attr):
                raise ValueError(f"Non-target governance dimension changed: {attr}")
        if row.fault_spec.operator == "conflicting":
            assertions = (
                row.observed_governance_state.authorization_assertions
                if target == "authorization" else row.observed_governance_state.temporal_assertions
            )
            if len(assertions) < 2:
                raise ValueError("Conflicting fault does not encode multiple assertions")
        if row.fault_spec.operator == "stale":
            history_key = "historical_deletion_states" if target == "deletion" else "historical_temporal_states"
            history = tuple(row.source_metadata.get(history_key, ()))
            predecessor = [
                getattr(item, target, item)
                for item in history
                if getattr(item, target, item) is not None
            ]
            if not predecessor or getattr(row.observed_governance_state, target) not in predecessor:
                raise ValueError("Stale fault lacks a source-history predecessor")
    if not row.example_id or not row.source_dataset or not row.source_metadata:
        raise ValueError("Row lacks traceability metadata")


def validate_manifest(rows: Iterable[RunnableGovernanceCondition]) -> None:
    rows = tuple(rows)
    run_ids: set[str] = set()
    groups: dict[str, list[RunnableGovernanceCondition]] = defaultdict(list)
    for row in rows:
        if row.run_id in run_ids:
            raise ValueError(f"Duplicate run_id: {row.run_id}")
        run_ids.add(row.run_id)
        validate_condition(row)
        groups[row.example_id].append(row)
    for example_id, group in groups.items():
        clean = [row for row in group if row.condition_type == "clean"]
        if len(clean) != 1:
            raise ValueError(f"Example {example_id} must have exactly one clean condition")
        reference = clean[0]
        for row in group:
            if row.task != reference.task or row.memory != reference.memory or row.utility_target != reference.utility_target:
                raise ValueError(f"Content changed within pair {example_id}")
            if row.gold_governance_state != reference.gold_governance_state or row.gold_admission_decision != reference.gold_admission_decision:
                raise ValueError(f"Gold changed within pair {example_id}")


__all__ = ["validate_condition", "validate_manifest", "validate_state"]
