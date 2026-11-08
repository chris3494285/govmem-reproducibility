"""Dataset-native applicability rules for governance reconstruction metrics."""

from __future__ import annotations

from .interfaces import AdaptedExample


GOVERNANCE_DIMENSIONS = (
    "authorization_scope",
    "lifecycle_deletion",
    "temporal_validity",
    "contextual_validity",
)

# Task/conflict types are public dataset metadata, not evaluator answers.
GATEMEM_QUERY_TYPE_DIMENSIONS = {
    "utility": (),
    "privacy": ("authorization_scope",),
    "safety": ("lifecycle_deletion",),
}

MEMORA_TASK_TYPE_DIMENSIONS = {
    "remembering": ("lifecycle_deletion",),
    "forgetting": ("lifecycle_deletion",),
}

MEMCONFLICT_CONFLICT_TYPE_DIMENSIONS = {
    "dynamic": ("temporal_validity",),
    "conditional": ("contextual_validity",),
    "static": (),
}


def dataset_native_task_type(example: AdaptedExample) -> str | None:
    """Return the normalized public task type using explicit dataset rules."""
    dataset = example.query.metadata.get("dataset") or example.metadata.get("dataset")
    if dataset == "gatemem":
        return example.query.metadata.get("task_type") or example.query.metadata.get("query_type")
    if dataset == "memconflict":
        return example.query.metadata.get("task_type") or example.query.metadata.get("conflict_type")
    if dataset == "memora":
        return example.query.metadata.get("task_type")
    return None


def applicable_governance_dimensions(example: AdaptedExample) -> tuple[str, ...]:
    """Return dimensions this dataset-native checkpoint can evaluate.

    Unknown datasets and task types are conservatively excluded rather than
    mapped to a dimension based on model output or evaluator gold.
    """
    dataset = example.query.metadata.get("dataset") or example.metadata.get("dataset")
    if dataset != "gatemem":
        if dataset == "memora":
            return MEMORA_TASK_TYPE_DIMENSIONS.get(example.query.metadata.get("task_type"), ())
        if dataset == "memconflict":
            return MEMCONFLICT_CONFLICT_TYPE_DIMENSIONS.get(example.query.metadata.get("conflict_type"), ())
        # LongMemEval and unknown datasets are utility/control inputs until a
        # dataset-native governance mapping is explicitly added.
        return ()
    query_type = example.query.metadata.get("query_type")
    return GATEMEM_QUERY_TYPE_DIMENSIONS.get(query_type, ())


def is_reconstruction_applicable(example: AdaptedExample, dimension: str | None = None) -> bool:
    dimensions = applicable_governance_dimensions(example)
    return bool(dimensions) if dimension is None else dimension in dimensions


__all__ = [
    "GATEMEM_QUERY_TYPE_DIMENSIONS",
    "GOVERNANCE_DIMENSIONS",
    "MEMORA_TASK_TYPE_DIMENSIONS",
    "MEMCONFLICT_CONFLICT_TYPE_DIMENSIONS",
    "dataset_native_task_type",
    "applicable_governance_dimensions",
    "is_reconstruction_applicable",
]
