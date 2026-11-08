"""Generic dataset and evaluation interfaces used by paper-v2 code.

The existing demo uses ``DatasetExample`` and ``MemoryRecord`` directly.  The
types in this module provide a stable, dataset-independent boundary around
those legacy objects without changing the demo execution path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol


RECONSTRUCTABLE_GOVERNANCE_FIELDS = frozenset({
    "authorized_scopes",
    "is_deleted",
    "valid_from",
    "valid_until",
    "entity_key",
    "contextually_valid",
})


@dataclass(frozen=True)
class RawInteraction:
    """A source interaction before it is interpreted as memory."""

    interaction_id: str
    user_id: str
    content: str
    timestamp: datetime | None
    role: str = "user"
    session_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    """A dataset-independent query presented to a memory system."""

    query_id: str
    user_id: str
    text: str
    current_time: datetime | None
    allowed_scopes: tuple[str, ...] = ()
    gold_answer: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    requester_role: str | None = None


@dataclass(frozen=True)
class OracleGovernanceState:
    """Ground-truth governance facts associated with one interaction."""

    interaction_id: str
    authorized_scopes: tuple[str, ...]
    is_deleted: bool
    valid_from: datetime | None
    valid_until: datetime | None
    entity_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconstructedGovernanceState:
    """A system's reconstructed governance facts and their provenance."""

    interaction_id: str
    authorized_scopes: tuple[str, ...]
    is_deleted: bool | None
    valid_from: datetime | None
    valid_until: datetime | None
    entity_key: str | None = None
    confidence: float | None = None
    provenance: str = "unknown"
    missing_fields: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GovernanceDecision:
    """Auditable result of applying governance to one interaction."""

    interaction_id: str
    authorized: bool
    temporally_valid: bool
    contextually_valid: bool
    selected: bool
    revalidated: bool = False
    reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """Dataset-independent per-query evaluation output."""

    query_id: str
    prediction: str
    gold_answer: str | None
    correct: bool | None
    retrieved_interaction_ids: tuple[str, ...] = ()
    governance_decisions: tuple[GovernanceDecision, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdaptedExample:
    """Normalized example emitted by a dataset adapter."""

    query: Query
    interactions: tuple[RawInteraction, ...]
    oracle_governance: tuple[OracleGovernanceState, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    evaluator_metadata: Mapping[str, Any] = field(default_factory=dict)


class DatasetAdapter(Protocol):
    """Interface implemented by dataset-specific ingestion adapters."""

    name: str

    def load(self, path: str | Path) -> Iterable[AdaptedExample]:
        """Load and normalize a dataset without invoking an answer model."""
