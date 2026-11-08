"""Governance-state reconstruction interfaces and offline test doubles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol

from .interfaces import AdaptedExample, Query, RawInteraction, ReconstructedGovernanceState


@dataclass(frozen=True)
class GovernanceReconstructionInput:
    """The only information supplied to a governance-state reconstructor."""

    raw_interactions: tuple[RawInteraction, ...]
    query: Query
    runtime_context: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_example(cls, example: AdaptedExample) -> "GovernanceReconstructionInput":
        # Query.gold_answer is evaluator data in the shared schema.  The
        # reconstruction boundary must never pass it to a reconstructor.
        query = Query(
            query_id=example.query.query_id,
            user_id=example.query.user_id,
            text=example.query.text,
            current_time=example.query.current_time,
            allowed_scopes=example.query.allowed_scopes,
            metadata=dict(example.query.metadata),
            requester_role=example.query.requester_role,
        )
        return cls(
            raw_interactions=example.interactions,
            query=query,
            runtime_context={
                "user_id": query.user_id,
                "requester_role": query.requester_role,
                "current_time": query.current_time,
                "allowed_scopes": query.allowed_scopes,
            },
        )


class GovernanceStateReconstructor(Protocol):
    """Reconstruct governance state from model-visible runtime information."""

    def reconstruct(
        self, input_data: GovernanceReconstructionInput
    ) -> Iterable[ReconstructedGovernanceState]:
        """Return reconstructed state without access to dataset gold labels."""


class MockGovernanceStateReconstructor:
    """Deterministic, offline reconstructor configured directly by a test."""

    def __init__(
        self,
        states: Iterable[ReconstructedGovernanceState] = (),
        states_by_query: Mapping[str, Iterable[ReconstructedGovernanceState]] | None = None,
    ) -> None:
        self._states = tuple(states)
        self._states_by_query = {
            query_id: tuple(query_states) for query_id, query_states in (states_by_query or {}).items()
        }

    def reconstruct(self, input_data: GovernanceReconstructionInput) -> tuple[ReconstructedGovernanceState, ...]:
        if input_data.query.query_id in self._states_by_query:
            return self._states_by_query[input_data.query.query_id]
        return self._states
