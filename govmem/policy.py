from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable, List

from .schema import MemoryRecord, QueryContext


SOURCE_RANK = {
    "system": 4,
    "user_profile": 3,
    "user_message": 2,
    "inferred": 1,
    "unknown": 0,
}


class GovernancePolicyEngine:
    def _coerce_datetime(self, value: datetime | str | None) -> datetime | None:
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

    def _effective_time(self, record: MemoryRecord) -> datetime | None:
        return self._coerce_datetime(record.valid_from) or self._coerce_datetime(record.timestamp)

    def temporal_relation(self, record: MemoryRecord, query_context: QueryContext) -> str:
        current_time = self._coerce_datetime(query_context.current_time)
        effective_time = self._effective_time(record)
        if current_time is None or effective_time is None:
            return "unknown"
        if effective_time > current_time:
            return "after_query"
        if effective_time < current_time:
            return "before_query"
        return "same_time"

    def filter_deleted(self, records: Iterable[MemoryRecord]) -> List[MemoryRecord]:
        return [record for record in records if not record.is_deleted]

    def filter_valid_time(self, records: Iterable[MemoryRecord], query_context: QueryContext) -> List[MemoryRecord]:
        # Temporal validity is only hard-filtered when we can compare both sides
        # confidently. If query time or memory time is missing, keep the record and
        # let downstream retrieval rank it rather than discarding it as stale.
        current_time = self._coerce_datetime(query_context.current_time)
        if current_time is None:
            return list(records)
        kept: list[MemoryRecord] = []
        for record in records:
            valid_from = self._coerce_datetime(record.valid_from)
            valid_until = self._coerce_datetime(record.valid_until)
            if valid_from is not None and valid_from > current_time:
                continue
            if valid_until is not None and valid_until < current_time:
                continue
            kept.append(record)
        return kept

    def filter_authorized_scope(self, records: Iterable[MemoryRecord], query_context: QueryContext) -> List[MemoryRecord]:
        allowed = set(query_context.allowed_scopes)
        return [record for record in records if record.scope in allowed]

    def _record_sort_key(self, record: MemoryRecord, current_time: datetime | None) -> tuple:
        is_valid = 1
        valid_from = self._coerce_datetime(record.valid_from)
        valid_until = self._coerce_datetime(record.valid_until)
        if current_time is not None and valid_from is not None and valid_from > current_time:
            is_valid = 0
        if current_time is not None and valid_until is not None and valid_until < current_time:
            is_valid = 0
        timestamp_value = self._coerce_datetime(record.timestamp)
        timestamp = timestamp_value.timestamp() if timestamp_value is not None else 0.0
        return (
            0 if record.is_deleted else 1,
            is_valid,
            record.confidence,
            timestamp,
            SOURCE_RANK.get(record.source, 0),
            record.memory_id,
        )

    def resolve_conflicts(self, records: Iterable[MemoryRecord], query_context: QueryContext) -> List[MemoryRecord]:
        grouped: dict[str, list[MemoryRecord]] = defaultdict(list)
        for record in records:
            grouped[record.entity_key].append(record)

        winners: list[MemoryRecord] = []
        for entity_key in sorted(grouped):
            candidates = grouped[entity_key]
            winner = max(candidates, key=lambda record: self._record_sort_key(record, query_context.current_time))
            winners.append(winner)
        return winners

    def apply(self, records: Iterable[MemoryRecord], query_context: QueryContext) -> List[MemoryRecord]:
        filtered = self.filter_deleted(records)
        filtered = self.filter_valid_time(filtered, query_context)
        filtered = self.filter_authorized_scope(filtered, query_context)
        return self.resolve_conflicts(filtered, query_context)
