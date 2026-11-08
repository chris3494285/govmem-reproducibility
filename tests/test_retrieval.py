from pathlib import Path

from eval.dataset import load_dataset, to_query_context
from govmem.policy import GovernancePolicyEngine
from govmem.retriever import GovMemRetriever, WarehouseNoGovRetriever, lexical_score, retrieve_top_k


DATASET = Path("data/sample_longmemeval.jsonl")


def test_lexical_score_prefers_relevant_memory():
    score_a = lexical_score("What tea does Maya prefer now?", "Maya preference update: answer: jasmine tea")
    score_b = lexical_score("What tea does Maya prefer now?", "Sam likes cycling and coffee")
    assert score_a > score_b


def test_govmem_retriever_filters_private_memory():
    examples = load_dataset(DATASET)
    example = next(item for item in examples if item.id == "ex6_unauthorized_private")
    context = to_query_context(example)
    govmem = GovMemRetriever(GovernancePolicyEngine())
    naive = WarehouseNoGovRetriever()

    naive_ids = [record.memory_id for record in naive.retrieve(example.memories, example.question, context, k=2)]
    govmem_ids = [record.memory_id for record in govmem.retrieve(example.memories, example.question, context, k=2)]

    assert govmem_ids == ["m6_public"]
    assert "m6_private" in naive_ids or len(naive_ids) == 1
