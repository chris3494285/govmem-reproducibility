"""Explicit evaluator-only support declarations for structured RQ1 oracles.

These declarations do not derive or invent governance truth.  They record what
the native dataset exposes and whether this repository has a deterministic
mapping to ``OracleGovernanceState``.
"""

from __future__ import annotations

from .interfaces import AdaptedExample
from .reconstruction_applicability import dataset_native_task_type


def structured_oracle_support(example: AdaptedExample) -> dict[str, object]:
    dataset = example.query.metadata.get("dataset") or example.metadata.get("dataset")
    task_type = dataset_native_task_type(example)
    matrix: dict[tuple[str, str | None], dict[str, object]] = {
        ("gatemem", "privacy"): {
            "oracle_available": False,
            "oracle_fields_supported": (),
            "native_gold_fields": ("expected_action", "judge_spec", "leak_targets", "principals", "relationships"),
            "reason": "Native privacy labels describe downstream access decisions, not per-memory authorized scopes.",
        },
        ("gatemem", "safety"): {
            "oracle_available": False,
            "oracle_fields_supported": (),
            "native_gold_fields": ("expected_action", "attack_type", "judge_spec", "leak_targets"),
            "reason": "Native safety labels do not provide a structured per-memory deletion oracle.",
        },
        ("memora", "remembering"): {
            "oracle_available": False,
            "oracle_fields_supported": (),
            "native_gold_fields": ("memory_evidence", "forgetting_evidence", "session_operations", "evaluation"),
            "reason": "Native evidence and session operations are not currently mapped to canonical interaction-level lifecycle states.",
        },
        ("memora", "forgetting"): {
            "oracle_available": False,
            "oracle_fields_supported": (),
            "native_gold_fields": ("forgetting_evidence", "session_operations", "evaluation"),
            "reason": "Native forgetting evidence is not currently mapped to canonical interaction-level lifecycle states.",
        },
        ("memconflict", "dynamic"): {
            "oracle_available": False,
            "oracle_fields_supported": (),
            "native_gold_fields": ("dynamic_state", "supporting_memory", "timestamps"),
            "reason": "Native dynamic conflict labels do not define canonical valid_from/valid_until intervals.",
        },
        ("memconflict", "conditional"): {
            "oracle_available": False,
            "oracle_fields_supported": (),
            "native_gold_fields": ("conditional_context", "supporting_memory"),
            "reason": "Native conditional context is not currently a canonical interaction-level contextual-validity oracle.",
        },
    }
    result = matrix.get((dataset, task_type))
    if result is None:
        return {
            "oracle_available": False,
            "oracle_fields_supported": (),
            "native_gold_fields": (),
            "reason": "No structured oracle support declaration exists for this dataset/task type.",
        }
    return dict(result)


__all__ = ["structured_oracle_support"]
