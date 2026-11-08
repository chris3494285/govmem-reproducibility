from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List

from govmem.schema import MemoryRecord, QueryContext


@dataclass(frozen=True)
class DatasetExample:
    id: str
    user_id: str
    question: str
    answer: str
    current_time: datetime
    allowed_scopes: tuple[str, ...]
    memories: tuple[MemoryRecord, ...]
    memory_metadata: tuple[dict[str, Any], ...]
    dataset_mode: str | None = None


def _parse_dt(value):
    if value is None:
        return None
    return datetime.fromisoformat(value)


def load_dataset(path: str | Path) -> List[DatasetExample]:
    path = Path(path)
    examples: list[DatasetExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            memories = []
            memory_metadata = []
            for memory in raw["memories"]:
                memories.append(
                    MemoryRecord(
                        memory_id=memory["memory_id"],
                        user_id=memory["user_id"],
                        content=memory["content"],
                        timestamp=_parse_dt(memory["timestamp"]),
                        source=memory["source"],
                        scope=memory["scope"],
                        valid_from=_parse_dt(memory["valid_from"]),
                        valid_until=_parse_dt(memory["valid_until"]),
                        is_deleted=bool(memory["is_deleted"]),
                        confidence=float(memory["confidence"]),
                        memory_type=memory["memory_type"],
                        entity_key=memory["entity_key"],
                    )
                )
                memory_metadata.append(dict(memory))
            examples.append(
                DatasetExample(
                    id=raw["id"],
                    user_id=raw["user_id"],
                    question=raw["question"],
                    answer=raw["answer"],
                    current_time=_parse_dt(raw["current_time"]),
                    allowed_scopes=tuple(raw["allowed_scopes"]),
                    memories=tuple(memories),
                    memory_metadata=tuple(memory_metadata),
                    dataset_mode=raw.get("dataset_mode"),
                )
            )
    return examples


def to_query_context(example: DatasetExample) -> QueryContext:
    return QueryContext(
        user_id=example.user_id,
        question=example.question,
        current_time=example.current_time,
        allowed_scopes=example.allowed_scopes,
    )
