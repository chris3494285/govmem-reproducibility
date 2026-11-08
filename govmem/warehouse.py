from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List

from .schema import MemoryRecord


class WarehouseMemoryStore:
    def __init__(self) -> None:
        self._records_by_user: dict[str, list[MemoryRecord]] = defaultdict(list)

    def load_records(self, records: Iterable[MemoryRecord]) -> None:
        for record in records:
            self.add_record(record)

    def add_record(self, record: MemoryRecord) -> None:
        self._records_by_user[record.user_id].append(record)

    def query_all(self, user_id: str) -> List[MemoryRecord]:
        return list(self._records_by_user.get(user_id, ()))

    def query_by_scope(self, user_id: str, allowed_scopes: Iterable[str]) -> List[MemoryRecord]:
        allowed = set(allowed_scopes)
        return [record for record in self.query_all(user_id) if record.scope in allowed]

