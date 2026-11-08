"""Deterministic permissive baseline enforcement for fault-RQ1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdmissionResult:
    actual_admission: str
    enforcement_reason: str
    fallback_used: bool


@dataclass(frozen=True)
class MemorySelectionResult:
    selected_memory: tuple[dict[str, Any], ...]
    selected_memory_ids: tuple[str, ...]
    rejected_memory_ids: tuple[str, ...]
    selection_reason: str


def _value(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _decision_from_state(state: Any, dimension: str) -> AdmissionResult:
    if dimension == "authorization":
        assertions = _value(state, "authorization_assertions", ())
        authorization = _value(state, "authorization")
        if assertions:
            return AdmissionResult("allow", "authorization_conflict_permissive_fallback", True)
        decision = _value(authorization, "decision")
        if decision == "allow":
            return AdmissionResult("allow", "authorization_allow", False)
        if decision == "deny":
            return AdmissionResult("deny", "authorization_deny", False)
        return AdmissionResult("allow", "authorization_missing_permissive_fallback", True)
    if dimension == "deletion":
        deletion = _value(state, "deletion")
        status = _value(deletion, "status")
        if status == "active":
            return AdmissionResult("allow", "deletion_active", False)
        if status == "deleted":
            return AdmissionResult("deny", "deletion_deleted", False)
        return AdmissionResult("allow", "deletion_missing_permissive_fallback", True)
    if dimension == "temporal":
        assertions = _value(state, "temporal_assertions", ())
        temporal = _value(state, "temporal")
        if assertions:
            return AdmissionResult("allow", "temporal_conflict_permissive_fallback", True)
        status = _value(temporal, "status")
        if status == "valid":
            return AdmissionResult("allow", "temporal_valid", False)
        if status == "expired":
            return AdmissionResult("deny", "temporal_expired", False)
        return AdmissionResult("allow", "temporal_missing_permissive_fallback", True)
    raise ValueError(f"Unsupported governance dimension: {dimension}")


def decide_admission(condition: Any) -> AdmissionResult:
    """Decide solely from the condition's observed governance state."""
    dimension = condition.get("dimension") if isinstance(condition, dict) else condition.dimension
    state = condition.get("observed_governance_state") if isinstance(condition, dict) else condition.observed_governance_state
    if dimension == "temporal":
        fault = condition.get("fault_spec") if isinstance(condition, dict) else condition.fault_spec
        operator = fault.get("operator") if isinstance(fault, dict) else getattr(fault, "operator", None)
        temporal = _value(state, "temporal")
        if operator == "stale" and _value(temporal, "status") == "expired":
            return AdmissionResult("allow", "temporal_stale_predecessor_admitted", False)
    return _decision_from_state(state, dimension)


def memory_ids(memory: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> list[str]:
    ids: list[str] = []
    for item in memory:
        for key in ("interaction_id", "memory_id", "item_id"):
            if item.get(key) is not None:
                ids.append(str(item[key]))
                break
        if "lifecycle" in item:
            ids.append(str(item.get("item")))
        if "attribute" in item:
            ids.append(str(item["attribute"]))
    return list(dict.fromkeys(ids))


def _temporal_value(row: Any, key: str) -> Any:
    metadata = row.get("source_metadata", {}) if isinstance(row, dict) else getattr(row, "source_metadata", {})
    return metadata.get(key)


def _temporal_memory(row: dict[str, Any], admission: AdmissionResult) -> MemorySelectionResult:
    memory = list(row.get("memory") or [])
    metadata = row.get("source_metadata") or {}
    attribute = metadata.get("attribute") or (memory[0].get("attribute") if memory else "")
    current_value = metadata.get("current_value")
    previous_value = metadata.get("previous_value")
    current_session = metadata.get("current_session") or {}
    previous_session = metadata.get("previous_session") or {}
    current_id = f"{attribute}@{current_session.get('session_id', 'current')}"
    previous_id = f"{attribute}@{previous_session.get('session_id', 'previous')}"
    state = row.get("observed_governance_state") or {}
    temporal = state.get("temporal") if isinstance(state, dict) else getattr(state, "temporal", None)
    status = _value(temporal, "status")
    assertions = state.get("temporal_assertions", []) if isinstance(state, dict) else getattr(state, "temporal_assertions", ())
    fault = row.get("fault_spec") or {}
    operator = fault.get("operator") if isinstance(fault, dict) else getattr(fault, "operator", None)
    rq1b_memconflict = (
        row.get("manifest_version") == "governance-fault-rq1-restrictive-v1"
        and row.get("source_dataset") == "memconflict"
    )
    rejected = [current_id, previous_id]
    if assertions:
        selected = tuple({"attribute": attribute, "value": value} for value in (current_value, previous_value) if value is not None)
        return MemorySelectionResult(selected, (current_id, previous_id), (), "temporal_conflict_both_assertions")
    if status == "valid":
        if rq1b_memconflict:
            selected = tuple(
                entry
                for entry in (
                    {"attribute": attribute, "role": "previous", "value": previous_value},
                    {"attribute": attribute, "role": "current", "value": current_value},
                )
                if entry["value"] is not None
            )
            return MemorySelectionResult(
                selected,
                tuple(identifier for identifier, entry in ((previous_id, previous_value), (current_id, current_value)) if entry is not None),
                (),
                "temporal_comparison_previous_and_current",
            )
        selected = ({"attribute": attribute, "value": current_value},) if current_value is not None else ()
        return MemorySelectionResult(selected, (current_id,) if selected else (), (previous_id,), "temporal_current_assertion")
    if status == "expired" and rq1b_memconflict:
        # RQ1-B's restrictive intervention withholds only the current
        # assertion.  The predecessor remains visible as historical evidence,
        # but is explicitly never promoted to the current role.
        selected = (
            {"attribute": attribute, "role": "previous", "value": previous_value},
        ) if previous_value is not None else ()
        return MemorySelectionResult(
            selected,
            (previous_id,) if selected else (),
            (current_id,),
            "temporal_expired_current_rejected_predecessor_retained",
        )
    if status == "expired" and operator == "stale":
        # The permissive fault experiment exposes the stale assertion's value
        # as historical context while rejecting the current assertion.  The
        # current value is never included in this projection.
        selected = ({"attribute": attribute, "value": previous_value},) if previous_value is not None else ()
        return MemorySelectionResult(selected, (previous_id,) if selected else (), (current_id,), "temporal_stale_predecessor")
    if admission.actual_admission == "allow":
        selected = ({"attribute": attribute, "value": current_value},) if current_value is not None else ()
        return MemorySelectionResult(selected, (current_id,) if selected else (), (previous_id,), "temporal_missing_permissive_fallback")
    return MemorySelectionResult((), (), tuple(rejected), "temporal_expired_rejected")


def select_memory(condition: Any, admission: AdmissionResult | None = None) -> MemorySelectionResult:
    """Select provider-visible memory using observed state and fixed evidence."""
    if admission is None:
        admission = decide_admission(condition)
    row = condition if isinstance(condition, dict) else condition.__dict__
    dimension = row.get("dimension")
    memory = tuple(row.get("memory") or ())
    ids = tuple(memory_ids(memory))
    if dimension == "temporal":
        return _temporal_memory(row, admission)
    if admission.actual_admission == "deny":
        return MemorySelectionResult((), (), ids, f"{dimension}_rejected")
    if dimension == "deletion":
        selected = tuple({"item": item.get("item")} for item in memory if "item" in item)
        if not selected:
            selected = tuple({"content": item.get("content")} for item in memory if "content" in item)
        return MemorySelectionResult(selected, ids, (), f"{dimension}_admitted")
    # Authorization and any future simple dimension expose content only; IDs
    # and timestamps remain evaluator-side bookkeeping.
    selected = tuple({"content": item.get("content")} for item in memory if item.get("content") is not None)
    return MemorySelectionResult(selected, ids, (), f"{dimension}_admitted")


__all__ = ["AdmissionResult", "MemorySelectionResult", "decide_admission", "memory_ids", "select_memory"]
