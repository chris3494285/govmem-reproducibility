"""Phase 4B downstream-evaluation boundary.

This module deliberately does not execute a governance decision policy.  It
only identifies whether native outcome gold is present so Phase 4C can attach a
decision/answer prediction later without confusing it with reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DownstreamEvaluationRecord:
    query_id: str
    gold_available: bool
    prediction_status: str = "not_run"
    correct: bool | None = None


def native_downstream_gold_available(record: Mapping[str, Any]) -> bool:
    dataset = str(record.get("dataset") or "").lower()
    gold = (
        record.get("evaluator_only", {})
        .get("evaluator_metadata", {})
        .get("gold", {})
    )
    if not isinstance(gold, Mapping):
        return False
    if dataset == "gatemem":
        return any(key in gold for key in ("expected_action", "judge_spec", "answer", "refuse"))
    if dataset == "memora":
        return bool(gold.get("evaluation"))
    if dataset == "memconflict":
        return any(key in gold for key in ("gold_answer", "answer", "supporting_memory", "dynamic_state", "conditional_context"))
    return False


def downstream_record(record: Mapping[str, Any], error_labels: list[str]) -> dict[str, Any]:
    """Create an explicit Phase 4C hand-off record; no prediction is inferred."""
    result = DownstreamEvaluationRecord(
        query_id=str(record.get("query_id", "")),
        gold_available=native_downstream_gold_available(record),
    )
    return {
        "query_id": result.query_id,
        "downstream_gold_available": result.gold_available,
        "downstream_prediction_status": result.prediction_status,
        "downstream_correct": result.correct,
        "reconstruction_error_labels": list(error_labels),
    }


__all__ = [
    "DownstreamEvaluationRecord",
    "downstream_record",
    "native_downstream_gold_available",
]
