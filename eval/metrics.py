from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any, Iterable

from govmem.policy import GovernancePolicyEngine
from govmem.schema import MemoryRecord, QueryContext


def _stringify_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return " ".join(part for part in (_stringify_text(item) for item in value) if part)
    if isinstance(value, dict):
        parts = []
        for key in sorted(value.keys(), key=lambda item: str(item)):
            rendered = _stringify_text(value[key])
            if rendered:
                parts.append(rendered)
        return " ".join(parts)
    return str(value)


def normalize_text(text: Any) -> str:
    text = _stringify_text(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match(prediction: Any, gold: Any) -> bool:
    return normalize_text(prediction) == normalize_text(gold)


def answer_matches(prediction: Any, gold: Any) -> bool:
    normalized_prediction = normalize_text(prediction)
    normalized_gold = normalize_text(gold)
    return exact_match(prediction, gold) or normalized_gold in normalized_prediction


def _coerce_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def token_f1(prediction: Any, gold: Any) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def safe_exact_match(prediction: Any, gold: Any, records: Iterable[MemoryRecord], query_context: QueryContext) -> bool:
    return exact_match(prediction, gold) and policy_violation_count(records, query_context) == 0 and stale_memory_error_count(records, query_context) == 0


def policy_violation_count(records: Iterable[MemoryRecord], query_context: QueryContext, policy_engine: GovernancePolicyEngine | None = None) -> int:
    engine = policy_engine or GovernancePolicyEngine()
    count = 0
    current_time = _coerce_datetime(query_context.current_time)
    for record in records:
        if record.is_deleted:
            count += 1
            continue
        record_valid_from = _coerce_datetime(record.valid_from)
        record_valid_until = _coerce_datetime(record.valid_until)
        if current_time is not None and record_valid_from is not None and record_valid_from > current_time:
            count += 1
            continue
        if current_time is not None and record_valid_until is not None and record_valid_until < current_time:
            count += 1
            continue
        if record.scope not in set(query_context.allowed_scopes):
            count += 1
    return count


def stale_memory_error_count(records: Iterable[MemoryRecord], query_context: QueryContext) -> int:
    count = 0
    current_time = _coerce_datetime(query_context.current_time)
    for record in records:
        if record.is_deleted:
            count += 1
            continue
        record_valid_from = _coerce_datetime(record.valid_from)
        record_valid_until = _coerce_datetime(record.valid_until)
        if current_time is not None and record_valid_from is not None and record_valid_from > current_time:
            count += 1
            continue
        if current_time is not None and record_valid_until is not None and record_valid_until < current_time:
            count += 1
    return count


def evidence_recall(retrieved_metadata: Iterable[dict]) -> float:
    for metadata in retrieved_metadata:
        if metadata.get("has_answer") is True:
            return 1.0
    return 0.0


def evidence_precision(retrieved_metadata: Iterable[dict]) -> float:
    retrieved = list(retrieved_metadata)
    if not retrieved:
        return 0.0
    positive = sum(1 for metadata in retrieved if metadata.get("has_answer") is True)
    return positive / len(retrieved)


def evidence_hit(gold_answer: Any, retrieved_texts: Iterable[Any]) -> bool:
    normalized_gold = normalize_text(gold_answer)
    if not normalized_gold:
        return False
    for text in retrieved_texts:
        if normalized_gold in normalize_text(text):
            return True
    return False


def underspecified_answer(prediction: Any, gold: Any) -> bool:
    normalized_prediction = normalize_text(prediction)
    normalized_gold = normalize_text(gold)
    if not normalized_prediction or not normalized_gold:
        return False
    if normalized_prediction == "unknown" or normalized_prediction == normalized_gold:
        return False
    return normalized_prediction in normalized_gold
