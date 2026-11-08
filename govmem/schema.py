from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    user_id: str
    content: str
    timestamp: datetime
    source: str
    scope: str
    valid_from: Optional[datetime]
    valid_until: Optional[datetime]
    is_deleted: bool
    confidence: float
    memory_type: str
    entity_key: str


@dataclass(frozen=True)
class QueryContext:
    user_id: str
    question: str
    current_time: Optional[datetime]
    allowed_scopes: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalResult:
    question: str
    records: tuple[MemoryRecord, ...]
    answer: str
