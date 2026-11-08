"""Deterministic, single-dimension governance-state fault operators."""

from __future__ import annotations

from dataclasses import replace

from .schema import (
    AuthorizationState, DeletionState, FaultSpec, GovernanceDimension, GovernanceState,
    TemporalState,
)


class FaultInjectionError(ValueError):
    pass


def _history_value(item, attribute: str):
    return getattr(item, attribute, None) if hasattr(item, attribute) else item


def _direction_for_flip(dimension: GovernanceDimension, state: GovernanceState) -> str:
    if dimension == "authorization":
        assert state.authorization is not None
        return "unsafe" if state.authorization.decision == "deny" else "restrictive"
    if dimension == "deletion":
        assert state.deletion is not None
        return "unsafe" if state.deletion.status == "deleted" else "restrictive"
    raise FaultInjectionError("Temporal flipped faults are not supported")


def inject_fault(
    gold_state: GovernanceState,
    dimension: GovernanceDimension,
    operator: str,
    source_history: tuple[GovernanceState, ...] | list[GovernanceState] | None = None,
) -> GovernanceState:
    """Return a new observed state; never mutate ``gold_state``.

    ``source_history`` contains only deterministic benchmark-native predecessor
    states.  It is required for stale faults and is never synthesized here.
    """
    if dimension not in {"authorization", "deletion", "temporal"}:
        raise FaultInjectionError(f"Unsupported governance dimension: {dimension}")
    if operator not in {"missing", "stale", "flipped", "conflicting"}:
        raise FaultInjectionError(f"Unsupported fault operator: {operator}")
    if dimension == "authorization" and operator == "stale":
        raise FaultInjectionError("Stale authorization faults are not supported in RQ1 v1")
    if dimension == "temporal" and operator == "flipped":
        raise FaultInjectionError("Flipped temporal faults are not supported in RQ1 v1")

    observed = gold_state
    if dimension == "authorization":
        state = gold_state.authorization
        if state is None:
            raise FaultInjectionError("Authorization gold state is unavailable")
        if operator == "missing":
            observed = replace(gold_state, authorization=None, authorization_assertions=())
        elif operator == "flipped":
            flipped = AuthorizationState("deny" if state.decision == "allow" else "allow", state.scope_id, state.version)
            observed = replace(gold_state, authorization=flipped, authorization_assertions=())
        elif operator == "conflicting":
            opposite = AuthorizationState("deny" if state.decision == "allow" else "allow", state.scope_id, state.version)
            observed = replace(gold_state, authorization=None, authorization_assertions=(state, opposite))
    elif dimension == "deletion":
        state = gold_state.deletion
        if state is None:
            raise FaultInjectionError("Deletion gold state is unavailable")
        if operator == "missing":
            observed = replace(gold_state, deletion=None)
        elif operator == "flipped":
            flipped = DeletionState("active" if state.status == "deleted" else "deleted", state.epoch)
            observed = replace(gold_state, deletion=flipped)
        elif operator == "stale":
            predecessors = tuple(source_history or ())
            previous = next((_history_value(item, "deletion") for item in reversed(predecessors) if _history_value(item, "deletion") is not None and _history_value(item, "deletion") != state), None)
            if previous is None:
                raise FaultInjectionError("No valid historical deletion predecessor supports stale fault")
            observed = replace(gold_state, deletion=previous)
    else:
        state = gold_state.temporal
        if state is None:
            raise FaultInjectionError("Temporal gold state is unavailable")
        if operator == "missing":
            observed = replace(gold_state, temporal=None, temporal_assertions=())
        elif operator == "stale":
            predecessors = tuple(source_history or ())
            previous = next((_history_value(item, "temporal") for item in reversed(predecessors) if _history_value(item, "temporal") is not None and _history_value(item, "temporal") != state), None)
            if previous is None:
                raise FaultInjectionError("No valid historical temporal predecessor supports stale fault")
            observed = replace(gold_state, temporal=previous, temporal_assertions=())
        elif operator == "conflicting":
            predecessors = tuple(source_history or ())
            previous = next((_history_value(item, "temporal") for item in reversed(predecessors) if _history_value(item, "temporal") is not None and _history_value(item, "temporal") != state), None)
            if previous is None:
                raise FaultInjectionError("No valid historical temporal assertion supports conflicting fault")
            observed = replace(gold_state, temporal=None, temporal_assertions=(state, previous))
    return observed


def fault_spec_for(gold_state: GovernanceState, dimension: GovernanceDimension, operator: str) -> FaultSpec:
    """Create a validated spec with deterministic direction."""
    if operator == "conflicting" or operator == "missing":
        direction = "ambiguous"
    elif operator == "stale":
        direction = "unsafe" if dimension in {"deletion", "temporal"} else "ambiguous"
    else:
        direction = _direction_for_flip(dimension, gold_state)
    return FaultSpec(dimension, operator, direction)


__all__ = ["FaultInjectionError", "fault_spec_for", "inject_fault"]
