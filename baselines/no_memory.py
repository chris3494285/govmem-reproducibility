from __future__ import annotations

from govmem.agent import DeterministicMemoryAgent
from govmem.schema import QueryContext


class NoMemoryBaseline:
    name = "no_memory"

    def __init__(self) -> None:
        self.agent = DeterministicMemoryAgent()

    def retrieve(self, records, question: str, query_context: QueryContext, k: int = 3):
        return []

    def answer(self, question: str, records, query_context: QueryContext, k: int = 3) -> str:
        return self.agent.answer(question, [])

