from __future__ import annotations

import re
from typing import Iterable, List

from .policy import GovernancePolicyEngine
from .schema import MemoryRecord, QueryContext


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def lexical_score(question: str, content: str) -> int:
    question_tokens = set(_tokenize(question))
    content_tokens = _tokenize(content)
    overlap = len(question_tokens & set(content_tokens))

    score = overlap * 10
    if re.search(r"answer:\s*", content, flags=re.IGNORECASE):
        score += 5

    answer_part = content.split("answer:", 1)
    if len(answer_part) == 2:
        answer_tokens = _tokenize(answer_part[1])
        score += min(len(answer_tokens), 3)

    return score


def retrieve_top_k(records: Iterable[MemoryRecord], question: str, k: int = 3) -> List[MemoryRecord]:
    ranked = sorted(
        records,
        key=lambda record: (
            -lexical_score(question, record.content),
            -record.timestamp.timestamp(),
            -record.confidence,
            -len(_tokenize(record.content)),
            record.memory_id,
        ),
    )
    return list(ranked[:k])


class WarehouseNoGovRetriever:
    def retrieve(self, records: Iterable[MemoryRecord], question: str, query_context: QueryContext, k: int = 3) -> List[MemoryRecord]:
        return retrieve_top_k(records, question, k=k)


class GovMemRetriever:
    def __init__(self, policy_engine: GovernancePolicyEngine | None = None) -> None:
        self.policy_engine = policy_engine or GovernancePolicyEngine()

    def retrieve(self, records: Iterable[MemoryRecord], question: str, query_context: QueryContext, k: int = 3) -> List[MemoryRecord]:
        governed = self.policy_engine.apply(records, query_context)
        return retrieve_top_k(governed, question, k=k)
