"""Canonical data contracts for the governance-fault RQ1 benchmark.

These contracts are intentionally separate from the frozen reconstruction RQ1
schemas.  They describe deterministic benchmark inputs and interventions, not
model predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

GovernanceDimension = Literal["authorization", "deletion", "temporal"]
ConditionType = Literal["clean", "fault"]
FaultOperator = Literal["missing", "stale", "flipped", "conflicting", "expired"]
FaultDirection = Literal["unsafe", "restrictive", "ambiguous"]
AdmissionDecision = Literal["allow", "deny"]


@dataclass(frozen=True)
class AuthorizationState:
    decision: AdmissionDecision
    scope_id: str | None = None
    version: int | None = None


@dataclass(frozen=True)
class DeletionState:
    status: Literal["active", "deleted"]
    epoch: int | None = None


@dataclass(frozen=True)
class TemporalState:
    status: Literal["valid", "expired"]
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    evaluated_at: datetime | None = None
    version: int | None = None


@dataclass(frozen=True)
class FaultSpec:
    dimension: GovernanceDimension
    operator: FaultOperator
    direction: FaultDirection


@dataclass(frozen=True)
class GovernanceState:
    authorization: AuthorizationState | None = None
    deletion: DeletionState | None = None
    temporal: TemporalState | None = None
    # A conflict is represented explicitly as assertions with no winner.
    authorization_assertions: tuple[AuthorizationState, ...] = ()
    temporal_assertions: tuple[TemporalState, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaseGovernanceExample:
    example_id: str
    source_dataset: str
    source_example_id: str
    dimension: GovernanceDimension
    task: str
    memory: tuple[dict[str, Any], ...]
    gold_governance_state: GovernanceState
    gold_admission_decision: AdmissionDecision
    utility_target: Any = None
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunnableGovernanceCondition:
    run_id: str
    example_id: str
    source_dataset: str
    dimension: GovernanceDimension
    condition_type: ConditionType
    fault_spec: FaultSpec | None
    gold_governance_state: GovernanceState
    observed_governance_state: GovernanceState
    gold_admission_decision: AdmissionDecision
    task: str
    memory: tuple[dict[str, Any], ...]
    utility_target: Any = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
