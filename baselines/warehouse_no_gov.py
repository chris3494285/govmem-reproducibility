from __future__ import annotations

from govmem.agent import DeterministicMemoryAgent
from govmem.retriever import WarehouseNoGovRetriever
from govmem.schema import QueryContext
from govmem.warehouse import WarehouseMemoryStore


class WarehouseMemoryNoGovernanceBaseline:
    name = "warehouse_no_governance"

    def __init__(self) -> None:
        self.retriever = WarehouseNoGovRetriever()
        self.agent = DeterministicMemoryAgent()

    def retrieve(self, records, question: str, query_context: QueryContext, k: int = 3):
        store = WarehouseMemoryStore()
        store.load_records(records)
        candidates = store.query_all(query_context.user_id)
        return self.retriever.retrieve(candidates, question, query_context, k=k)

    def answer(self, question: str, records, query_context: QueryContext, k: int = 3) -> str:
        return self.agent.answer(question, self.retrieve(records, question, query_context, k=k))

