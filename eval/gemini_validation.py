"""Offline Gemini canary-health and selective-repair utilities."""

from __future__ import annotations

from collections import Counter
from typing import Any

from eval.final_main_token_budget import budget_class, resolve_max_output_tokens


def canary_health(rows: list[dict[str, Any]], budget_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize engineering health; correctness is intentionally absent."""
    enriched = []
    for row in rows:
        budget = resolve_max_output_tokens(row, budget_config)
        output = row.get("output_tokens")
        ratio = output / budget if isinstance(output, (int, float)) and budget else None
        finish = row.get("finish_reason", row.get("stop_reason"))
        truncated = bool(row.get("truncated_output")) or str(finish).upper().endswith("MAX_TOKENS")
        enriched.append({"condition_id": row.get("condition_id"), "budget_class": budget_class(row), "resolved_max_output_tokens": budget, "output_tokens": output, "output_budget_ratio": ratio, "finish_reason": finish, "parse_success": row.get("parse_success"), "truncated_output": truncated})
    review = [item["condition_id"] for item in enriched if item["truncated_output"] or (item["output_budget_ratio"] is not None and item["output_budget_ratio"] >= 0.85)]
    return {"rows": enriched, "review_required": review, "review_required_by_budget_class": dict(sorted(Counter(item["budget_class"] for item in enriched if item["condition_id"] in review).items()))}


def validate_repair_rows(rows: list[dict[str, Any]], expected_ids: set[str], manifest_ids: set[str]) -> None:
    ids = [row.get("condition_id") for row in rows]
    if set(ids) != expected_ids:
        raise ValueError("Repair rows do not exactly match the requested condition IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("Repair rows contain duplicate condition IDs")
    if not expected_ids <= manifest_ids:
        raise ValueError("Repair IDs are not a subset of the frozen manifest")
