"""Offline RQ1 structural reconstruction diagnostics for Phase 4A JSONL."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .downstream_evaluation import downstream_record


RECONSTRUCTION_FIELDS = (
    "authorized_scopes",
    "is_deleted",
    "valid_from",
    "valid_until",
    "contextually_valid",
    "entity_key",
)
DIMENSION_FIELDS = {
    "authorization_scope": ("authorized_scopes",),
    "lifecycle_deletion": ("is_deleted",),
    # Phase 4B has no deterministic source signal that an end/supersession
    # boundary must exist.  Open-ended valid_until=None is therefore optional
    # for primary completeness and remains schema-level diagnostic metadata.
    "temporal_validity": ("valid_from",),
    "contextual_validity": ("contextually_valid",),
}

# Temporal bounds are both relevant to interval reconstruction, but ``None``
# is a valid unbounded endpoint.  It is missing only when the model explicitly
# places that bound in missing_fields.


def _states(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = record.get("reconstructed_states", [])
    return [state for state in value if isinstance(state, Mapping)] if isinstance(value, list) else []


def _applicable(record: Mapping[str, Any]) -> tuple[str, ...]:
    value = record.get("applicable_governance_dimensions", [])
    return tuple(str(item) for item in value) if isinstance(value, list) else ()


def _valid(record: Mapping[str, Any]) -> bool:
    return record.get("reconstruction_status") == "success" and record.get("validation_status") == "valid"


def _missing(state: Mapping[str, Any], field: str) -> bool:
    markers = state.get("missing_fields", [])
    if field in markers:
        # [] authorized_scopes + missing is intentionally unknown; it must not
        # be counted as confidently empty authorization.
        return True
    # The shared interval representation treats a null endpoint as an
    # unbounded interval, not as an unsupported value.
    if field in {"valid_from", "valid_until"}:
        return False
    return _state_value(state, field) is None


def _state_value(state: Mapping[str, Any], field: str) -> Any:
    if field == "contextually_valid":
        metadata = state.get("metadata", {})
        if isinstance(metadata, Mapping) and "contextually_valid" in metadata:
            return metadata.get("contextually_valid")
    return state.get(field)


def _relevant_missing(state: Mapping[str, Any], field: str) -> bool:
    """Missingness for primary governance completeness, not raw schema shape."""
    if field == "valid_from":
        # Unlike valid_until, a usable start bound is the primary temporal
        # evidence required by the current Phase 4B representation.
        return "valid_from" in state.get("missing_fields", []) or _state_value(state, field) is None
    return _missing(state, field)


def _structural_metrics(records: list[Mapping[str, Any]], dimension: str | None = None) -> dict[str, Any]:
    selected = [
        record for record in records
        if not record.get("skipped") and (dimension is None or dimension in _applicable(record))
    ]
    n = len(selected)
    valid = [record for record in selected if _valid(record)]
    state_rows = [state for record in valid for state in _states(record)]
    empty = sum(not _states(record) for record in valid)
    invalid = sum(not _valid(record) for record in selected)
    provider_failures = sum(
        not _valid(record)
        and record.get("validation_status") in {"transport_error", "error"}
        for record in selected
    )
    model_failures = invalid - provider_failures
    schema_missing_counts = {field: sum(_missing(state, field) for state in state_rows) for field in RECONSTRUCTION_FIELDS}
    relevant_pairs = [
        (state, field)
        for record in valid
        for state in _states(record)
        for relevant_dimension in (_applicable(record) if dimension is None else (dimension,))
        for field in DIMENSION_FIELDS.get(relevant_dimension, ())
    ]
    relevant_missing_counts = {
        field: sum(_relevant_missing(state, field) for state, relevant_field in relevant_pairs if relevant_field == field)
        for field in RECONSTRUCTION_FIELDS
    }
    visible_metadata_available = False
    grounding_eligible = 0
    grounded = 0
    grounding_missing_id = 0
    grounding_outside_set = 0
    for record in valid:
        visible = record.get("evaluator_only", {}).get("model_visible_interaction_ids")
        if isinstance(visible, list):
            visible_metadata_available = True
            visible_set = set(visible)
            for state in _states(record):
                interaction_id = state.get("interaction_id")
                if not interaction_id:
                    grounding_missing_id += 1
                else:
                    grounding_eligible += 1
                    if interaction_id in visible_set:
                        grounded += 1
                    else:
                        grounding_outside_set += 1
    schema_denominator = len(state_rows) * len(RECONSTRUCTION_FIELDS)
    relevant_denominator = len(relevant_pairs)
    return {
        "n_examples": n,
        "valid_prediction_rate": len(valid) / n if n else None,
        "invalid_or_error_rate": invalid / n if n else None,
        "provider_execution_failure_count": provider_failures,
        "model_reconstruction_failure_count": model_failures,
        "provider_execution_failure_rate": provider_failures / n if n else None,
        "model_reconstruction_failure_rate": model_failures / n if n else None,
        "empty_reconstruction_rate": empty / n if n else None,
        "nonempty_reconstruction_rate": (len(valid) - empty) / n if n else None,
        "mean_states_per_example": sum(len(_states(record)) for record in valid) / n if n else None,
        # Primary completeness metrics are applicability-scoped.
        "missing_field_rate": sum(relevant_missing_counts.values()) / relevant_denominator if relevant_denominator else None,
        "missing_field_rates": {
            field: (count / sum(1 for _, relevant_field in relevant_pairs if relevant_field == field)
                    if any(relevant_field == field for _, relevant_field in relevant_pairs) else None)
            for field, count in relevant_missing_counts.items()
        },
        "relevant_field_missing_rate": sum(relevant_missing_counts.values()) / relevant_denominator if relevant_denominator else None,
        "relevant_field_missing_rates": {
            field: (count / sum(1 for _, relevant_field in relevant_pairs if relevant_field == field)
                    if any(relevant_field == field for _, relevant_field in relevant_pairs) else None)
            for field, count in relevant_missing_counts.items()
        },
        "relevant_field_denominator": relevant_denominator,
        # Schema-wide diagnostics remain available but are not the primary RQ1
        # completeness result.
        "schema_missing_field_rate": sum(schema_missing_counts.values()) / schema_denominator if schema_denominator else None,
        "schema_missing_field_rates": {
            field: (count / len(state_rows) if state_rows else None)
            for field, count in schema_missing_counts.items()
        },
        "grounding_metric": "visible_history_interaction_id_validity",
        "grounding_metadata_available": visible_metadata_available,
        "grounding_coverage": grounded / grounding_eligible if grounding_eligible else None,
        "grounding_denominator": grounding_eligible,
        "grounding_missing_interaction_id": grounding_missing_id,
        "grounding_outside_accepted_set": grounding_outside_set,
        "abstention_rate": empty / n if n else None,
    }


def _field_value(state: Mapping[str, Any], field: str) -> Any:
    if _missing(state, field):
        return None
    return _state_value(state, field)


def _oracle_field_value(oracle: Mapping[str, Any], field: str) -> Any:
    if field == "contextually_valid":
        return oracle.get("metadata", {}).get("contextually_valid")
    return oracle.get(field)


def _native_agreement(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {field: {"n": 0, "correct": 0, "missing_prediction": 0} for field in RECONSTRUCTION_FIELDS}
    for record in records:
        evaluator = record.get("evaluator_only", {})
        if not evaluator.get("oracle_available"):
            continue
        supported = set(evaluator.get("oracle_fields_supported", []))
        oracles = evaluator.get("oracle_governance", [])
        states = {state.get("interaction_id"): state for state in _states(record)}
        if not isinstance(oracles, list):
            continue
        for oracle in oracles:
            if not isinstance(oracle, Mapping):
                continue
            state = states.get(oracle.get("interaction_id"))
            for field in supported.intersection(RECONSTRUCTION_FIELDS):
                totals[field]["n"] += 1
                if state is None or _field_value(state, field) is None:
                    totals[field]["missing_prediction"] += 1
                elif _field_value(state, field) == _oracle_field_value(oracle, field):
                    totals[field]["correct"] += 1
    return {
        field: {
            **values,
            "available": values["n"] > 0,
            "accuracy": values["correct"] / values["n"] if values["n"] else None,
        }
        for field, values in totals.items()
    }


def _error_labels(record: Mapping[str, Any]) -> list[str]:
    if record.get("skipped"):
        return []
    if not _valid(record):
        labels = ["invalid_prediction"]
        error = record.get("error", {})
        if record.get("validation_status") in {"transport_error", "error"}:
            labels.append("provider_error")
        return labels
    states = _states(record)
    labels: list[str] = []
    if not states:
        labels.append("empty_reconstruction")
    dimensions = set(_applicable(record))
    relevant_fields = {
        field for dimension in dimensions for field in DIMENSION_FIELDS.get(dimension, ())
    }
    missing_fields = {
        field for state in states for field in relevant_fields
        if _relevant_missing(state, field)
    }
    if missing_fields:
        labels.append("missing_governance_field")
    if "authorization_scope" in dimensions and "authorized_scopes" in missing_fields:
        labels.append("missing_authorization_state")
    if "lifecycle_deletion" in dimensions and "is_deleted" in missing_fields:
        labels.append("missing_deletion_state")
    if "temporal_validity" in dimensions and "valid_from" in missing_fields:
        labels.append("missing_temporal_state")
    if "contextual_validity" in dimensions and "contextually_valid" in missing_fields:
        labels.append("missing_contextual_state")
    visible = record.get("evaluator_only", {}).get("model_visible_interaction_ids")
    if any(not state.get("interaction_id") for state in states):
        labels.append("missing_interaction_id")
    if isinstance(visible, list) and any(
        state.get("interaction_id") and state.get("interaction_id") not in set(visible)
        for state in states
    ):
        labels.append("wrong_or_unsupported_grounding")
    return labels


def evaluate_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [record for record in records if isinstance(record, Mapping)]
    orchestration_errors = [
        {
            "slice": record.get("slice"),
            "error": record.get("orchestration_error"),
        }
        for record in rows
        if record.get("orchestration_error")
    ]
    applicable = [record for record in rows if not record.get("skipped") and _applicable(record)]
    datasets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    task_types: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    dimensions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in applicable:
        dataset = str(record.get("dataset") or "unknown")
        task = str(record.get("task_type") or record.get("query_type") or "unknown")
        datasets[dataset].append(record)
        task_types[f"{dataset}:{task}"].append(record)
        for dimension in _applicable(record):
            dimensions[f"{dataset}:{task}:{dimension}"].append(record)
    analyses = []
    for record in rows:
        labels = _error_labels(record)
        downstream = downstream_record(record, labels)
        analyses.append({
            "query_id": record.get("query_id"),
            "dataset": record.get("dataset"),
            "task_type": record.get("task_type") or record.get("query_type"),
            "applicable_dimension": list(_applicable(record)),
            "reconstruction_error_labels": labels,
            **{key: downstream[key] for key in ("downstream_gold_available", "downstream_prediction_status", "downstream_correct")},
        })
    return {
        "schema_version": "rq1-metrics-v1",
        "orchestration_errors": orchestration_errors,
        "denominator_rules": {
            "structural": "applicable, non-skipped examples only; provider errors are invalid, empty valid predictions are not errors",
            "native_agreement": "optional native-supported structured agreement only; unavailable oracle has n=0 and accuracy=None",
            "grounding": "visible-history interaction-ID validity only: reconstructed IDs in evaluator-only model_visible_interaction_ids; not semantic target-memory or binding correctness",
            "temporal_completeness": "valid_from is the sole primary temporal field; valid_until is optional for open-ended states and diagnostic unless future evaluator evidence requires an end boundary",
        },
        "structural": _structural_metrics(applicable),
        "by_dataset": {key: _structural_metrics(value) for key, value in sorted(datasets.items())},
        "by_task_type": {key: _structural_metrics(value) for key, value in sorted(task_types.items())},
        "by_governance_dimension": {key: _structural_metrics(value, key.rsplit(":", 1)[-1]) for key, value in sorted(dimensions.items())},
        "native_supported_state_agreement": _native_agreement(applicable),
        "downstream": {
            "status": "not_run",
            "reason": "Phase 4B does not execute a downstream governance decision policy",
            "n_examples_with_native_gold": sum(item["downstream_gold_available"] for item in analyses),
        },
        "per_example": analyses,
    }


__all__ = ["DIMENSION_FIELDS", "RECONSTRUCTION_FIELDS", "evaluate_records"]
