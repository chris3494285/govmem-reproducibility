"""RQ1 comparison metrics for reconstructed governance state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .interfaces import AdaptedExample, OracleGovernanceState, ReconstructedGovernanceState
from .reconstruction_applicability import applicable_governance_dimensions
from .reconstruction import GovernanceReconstructionInput, GovernanceStateReconstructor


@dataclass(frozen=True)
class FieldMetric:
    """Field-level score over only oracle states that expose the field."""

    correct: int = 0
    total: int = 0
    missing: int = 0

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None

    @property
    def missing_rate(self) -> float | None:
        return self.missing / self.total if self.total else None


@dataclass(frozen=True)
class RQ1Evaluation:
    examples_evaluated: int
    oracle_states_evaluated: int
    field_metrics: Mapping[str, FieldMetric]
    reconstructions: Mapping[str, tuple[ReconstructedGovernanceState, ...]]
    applicable_examples_evaluated: int = 0


def _available(oracle: OracleGovernanceState, field: str) -> bool:
    declared = oracle.metadata.get("available_fields")
    if declared is not None:
        return field in declared
    if field == "contextual_validity":
        return "contextually_valid" in oracle.metadata or "contextual_validity" in oracle.metadata
    if field == "temporal_validity":
        return oracle.valid_from is not None or oracle.valid_until is not None
    return field in {"authorization_scope", "lifecycle_deletion"}


def _reconstructed_value(state: ReconstructedGovernanceState, field: str) -> object:
    missing = set(state.missing_fields)
    if field == "authorization_scope":
        return None if "authorized_scopes" in missing else state.authorized_scopes
    if field == "lifecycle_deletion":
        return None if "is_deleted" in missing or state.is_deleted is None else state.is_deleted
    if field == "temporal_validity":
        return None if {"valid_from", "valid_until"} & missing else (state.valid_from, state.valid_until)
    if field == "contextual_validity":
        if "contextually_valid" in missing or "contextual_validity" in missing:
            return None
        return state.metadata.get("contextually_valid", state.metadata.get("contextual_validity"))
    raise KeyError(field)


def _oracle_value(oracle: OracleGovernanceState, field: str) -> object:
    if field == "authorization_scope":
        return oracle.authorized_scopes
    if field == "lifecycle_deletion":
        return oracle.is_deleted
    if field == "temporal_validity":
        return oracle.valid_from, oracle.valid_until
    if field == "contextual_validity":
        return oracle.metadata.get("contextually_valid", oracle.metadata.get("contextual_validity"))
    raise KeyError(field)


def evaluate_rq1(
    examples: Iterable[AdaptedExample], reconstructor: GovernanceStateReconstructor
) -> RQ1Evaluation:
    """Run reconstruction first, then compare outputs to evaluator-only gold."""

    field_names = (
        "authorization_scope",
        "lifecycle_deletion",
        "temporal_validity",
        "contextual_validity",
    )
    totals = {field: [0, 0, 0] for field in field_names}
    all_reconstructions: dict[str, tuple[ReconstructedGovernanceState, ...]] = {}
    examples_evaluated = 0
    oracle_states_evaluated = 0
    applicable_examples_evaluated = 0

    for example in examples:
        examples_evaluated += 1
        applicable_dimensions = set(applicable_governance_dimensions(example))
        if not applicable_dimensions:
            # Preserve the checkpoint in the input pipeline, but do not invoke
            # governance reconstruction or count factual utility tasks as RQ1.
            all_reconstructions[example.query.query_id] = ()
            continue
        applicable_examples_evaluated += 1
        input_data = GovernanceReconstructionInput.from_example(example)
        reconstructed = tuple(reconstructor.reconstruct(input_data))
        all_reconstructions[example.query.query_id] = reconstructed
        by_interaction = {state.interaction_id: state for state in reconstructed}
        for oracle in example.oracle_governance:
            oracle_states_evaluated += 1
            candidate = by_interaction.get(oracle.interaction_id)
            for field in field_names:
                if field not in applicable_dimensions or not _available(oracle, field):
                    continue
                totals[field][1] += 1
                value = _reconstructed_value(candidate, field) if candidate is not None else None
                if value is None:
                    totals[field][2] += 1
                elif value == _oracle_value(oracle, field):
                    totals[field][0] += 1

    return RQ1Evaluation(
        examples_evaluated=examples_evaluated,
        oracle_states_evaluated=oracle_states_evaluated,
        field_metrics={field: FieldMetric(*totals[field]) for field in field_names},
        reconstructions=all_reconstructions,
        applicable_examples_evaluated=applicable_examples_evaluated,
    )
