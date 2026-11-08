"""Common normalized contract for external memory-system adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class MemorySystemDependencyError(RuntimeError):
    """Raised when an optional external memory dependency is unavailable."""


@dataclass(frozen=True)
class MemoryRetrieval:
    content: str
    memory_id: str | None
    rank: int
    score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_payload: Any = None


@dataclass(frozen=True)
class MemoryRetrievalResult:
    memory_system: str
    query: str
    memories: tuple[MemoryRetrieval, ...]
    raw_response: Any = None
    error: dict[str, Any] | None = None

    @property
    def retrieved_memory_ids(self) -> tuple[str, ...]:
        return tuple(memory.memory_id for memory in self.memories if memory.memory_id is not None)

    @property
    def normalized_retrieved_contents(self) -> tuple[str, ...]:
        return tuple(memory.content for memory in self.memories)


def _payload_items(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        for key in ("results", "memories", "data", "items", "documents"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return list(value)
        return [payload]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return list(payload)
    return [payload]


def _field(payload: Any, *names: str) -> Any:
    for name in names:
        if isinstance(payload, Mapping) and name in payload:
            return payload[name]
        if hasattr(payload, name):
            return getattr(payload, name)
    return None


def normalize_retrieval(payload: Any, *, memory_system: str, query: str) -> MemoryRetrievalResult:
    """Normalize native results while retaining IDs, metadata, and raw payloads.

    Native result order is preserved.  Ranks are assigned deterministically
    from that order unless the system exposes a valid one-based rank.
    """
    normalized: list[MemoryRetrieval] = []
    for index, item in enumerate(_payload_items(payload), start=1):
        content = _field(item, "memory", "content", "text", "document", "page_content")
        if content is None and isinstance(item, str):
            content = item
        if content is None:
            content = ""
        raw_metadata = _field(item, "metadata", "meta")
        metadata = deepcopy(dict(raw_metadata)) if isinstance(raw_metadata, Mapping) else {}
        score_value = _field(item, "score", "similarity", "relevance")
        try:
            score = float(score_value) if score_value is not None else None
        except (TypeError, ValueError):
            score = None
        rank_value = _field(item, "rank", "position")
        try:
            rank = int(rank_value) if rank_value is not None and int(rank_value) > 0 else index
        except (TypeError, ValueError):
            rank = index
        memory_id = _field(item, "id", "memory_id", "key", "uuid")
        normalized.append(
            MemoryRetrieval(
                content=str(content),
                memory_id=str(memory_id) if memory_id is not None else None,
                rank=rank,
                score=score,
                metadata=metadata,
                raw_payload=deepcopy(item),
            )
        )
    # A native API can expose scores but omit ranks.  Keep native order as the
    # tie-breaker and make the resulting order reproducible.
    normalized = [item for _, item in sorted(enumerate(normalized), key=lambda pair: (pair[1].rank, pair[0]))]
    return MemoryRetrievalResult(memory_system=memory_system, query=query, memories=tuple(normalized), raw_response=deepcopy(payload))


class MemorySystemAdapter(ABC):
    """Minimal lifecycle/retrieval contract shared by all external systems."""

    memory_system: str
    adapter_version: str = "1"

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def add_memory(self, content: str, *, metadata: Mapping[str, Any] | None = None, **kwargs: Any) -> Any: ...

    @abstractmethod
    def update_memory(self, memory_id: str, content: str | None = None, *, metadata: Mapping[str, Any] | None = None, **kwargs: Any) -> Any: ...

    @abstractmethod
    def delete_memory(self, memory_id: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def retrieve(self, query: str, **kwargs: Any) -> MemoryRetrievalResult: ...

    @abstractmethod
    def snapshot(self) -> Any: ...
