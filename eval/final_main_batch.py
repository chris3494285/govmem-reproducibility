"""Offline-safe OpenAI Batch request and result helpers for final-main runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def batch_custom_id(condition_id: str) -> str:
    """Return a deterministic, Batch-safe identifier for one condition."""
    return "govmem-" + hashlib.sha256(condition_id.encode("utf-8")).hexdigest()


def build_batch_requests(rows: Iterable[dict[str, Any]], provider: Any, prompts: dict[str, str], budgets: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    requests: list[dict[str, Any]] = []
    reverse: dict[str, str] = {}
    for row in rows:
        condition_id = row["condition_id"]
        custom_id = batch_custom_id(condition_id)
        if custom_id in reverse:
            raise ValueError(f"Duplicate Batch custom_id generated for {condition_id}")
        reverse[custom_id] = condition_id
        requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": provider.request_payload(prompts[condition_id], max_tokens=budgets[condition_id]),
        })
    return requests, reverse


def write_batch_input(requests: list[dict[str, Any]], path: Path) -> None:
    path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in requests), encoding="utf-8")


def read_batch_lines(content: Any) -> list[dict[str, Any]]:
    if hasattr(content, "text"):
        content = content.text
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    if not isinstance(content, str):
        raise ValueError("Batch output content is not text")
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def validate_batch_output(lines: Iterable[dict[str, Any]], reverse: dict[str, str]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    duplicate_custom_ids: list[str] = []
    unknown_custom_ids: list[str] = []
    for line in lines:
        custom_id = line.get("custom_id")
        if custom_id in results:
            duplicate_custom_ids.append(str(custom_id))
            continue
        if custom_id not in reverse:
            unknown_custom_ids.append(str(custom_id))
            continue
        results[custom_id] = line
    if duplicate_custom_ids:
        raise ValueError(f"Duplicate Batch custom_ids: {', '.join(sorted(duplicate_custom_ids))}")
    if unknown_custom_ids:
        raise ValueError(f"Unknown Batch custom_ids: {', '.join(sorted(unknown_custom_ids))}")
    return results


def batch_cost_estimate(*, input_tokens: int, max_output_tokens: int, input_price_per_million: float | None, cached_input_price_per_million: float | None, output_price_per_million: float | None, batch_discount: float = 0.5, cached_input_tokens: int = 0) -> dict[str, float | int | None]:
    """Estimate upper-bound cost; null means pricing was not supplied."""
    uncached = max(0, input_tokens - cached_input_tokens)
    if input_price_per_million is None or output_price_per_million is None:
        sync = batch = None
    else:
        cached_rate = cached_input_price_per_million if cached_input_price_per_million is not None else input_price_per_million
        sync = (uncached / 1_000_000) * input_price_per_million + (cached_input_tokens / 1_000_000) * cached_rate + (max_output_tokens / 1_000_000) * output_price_per_million
        batch = sync * (1.0 - batch_discount)
    return {"estimated_uncached_input_tokens": uncached, "estimated_cached_input_tokens": cached_input_tokens, "estimated_max_output_tokens": max_output_tokens, "estimated_synchronous_upper_bound_cost": sync, "estimated_batch_upper_bound_cost": batch, "batch_discount": batch_discount, "pricing_basis": "token-budget upper bound; provider billing/usage telemetry is authoritative"}
