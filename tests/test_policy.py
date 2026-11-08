from datetime import datetime
from pathlib import Path

from eval.dataset import load_dataset, to_query_context
from govmem.policy import GovernancePolicyEngine
from govmem.schema import MemoryRecord, QueryContext


DATASET = Path("data/sample_longmemeval.jsonl")


def test_deleted_memory_is_filtered():
    examples = load_dataset(DATASET)
    example = next(item for item in examples if item.id == "ex5_deleted_memory")
    engine = GovernancePolicyEngine()
    kept = engine.apply(example.memories, to_query_context(example))
    assert all(not record.is_deleted for record in kept)
    assert {record.memory_id for record in kept} == {"m5_valid"}


def test_expired_memory_is_filtered():
    examples = load_dataset(DATASET)
    example = next(item for item in examples if item.id == "ex4_expired_memory")
    engine = GovernancePolicyEngine()
    kept = engine.apply(example.memories, to_query_context(example))
    assert {record.memory_id for record in kept} == {"m4_noise"}


def test_unauthorized_memory_is_filtered():
    examples = load_dataset(DATASET)
    example = next(item for item in examples if item.id == "ex6_unauthorized_private")
    engine = GovernancePolicyEngine()
    kept = engine.apply(example.memories, to_query_context(example))
    assert {record.memory_id for record in kept} == {"m6_public"}


def test_conflict_resolution_prefers_newer_higher_quality_memory():
    examples = load_dataset(DATASET)
    example = next(item for item in examples if item.id == "ex3_conflicting_old_new")
    engine = GovernancePolicyEngine()
    kept = engine.apply(example.memories, to_query_context(example))
    assert len(kept) == 1
    assert kept[0].memory_id == "m3_new"
    assert kept[0].content.endswith("Harbor Labs")


def test_not_yet_valid_only_filters_when_time_is_confident():
    engine = GovernancePolicyEngine()
    query_time = datetime.fromisoformat("2024-01-10T00:00:00")
    record_before = MemoryRecord(
        memory_id="m_before",
        user_id="u1",
        content="I moved to Seattle last month.",
        timestamp=datetime.fromisoformat("2024-01-01T00:00:00"),
        source="user_message",
        scope="default",
        valid_from=datetime.fromisoformat("2024-01-01T00:00:00"),
        valid_until=None,
        is_deleted=False,
        confidence=1.0,
        memory_type="episodic",
        entity_key="profile:location",
    )
    record_after = MemoryRecord(
        memory_id="m_after",
        user_id="u1",
        content="I moved to Boston tomorrow.",
        timestamp=datetime.fromisoformat("2024-01-11T00:00:00"),
        source="user_message",
        scope="default",
        valid_from=datetime.fromisoformat("2024-01-11T00:00:00"),
        valid_until=None,
        is_deleted=False,
        confidence=1.0,
        memory_type="episodic",
        entity_key="profile:location",
    )
    record_missing_memory_time = MemoryRecord(
        memory_id="m_missing_memory_time",
        user_id="u1",
        content="I moved to Portland sometime around then.",
        timestamp=datetime.fromisoformat("2024-01-01T00:00:00"),
        source="user_message",
        scope="default",
        valid_from=None,
        valid_until=None,
        is_deleted=False,
        confidence=1.0,
        memory_type="episodic",
        entity_key="profile:location",
    )
    context = QueryContext(user_id="u1", question="Where do I live now?", current_time=query_time, allowed_scopes=("default",))

    assert engine.filter_valid_time([record_before], context) == [record_before]
    assert engine.filter_valid_time([record_after], context) == []
    assert engine.filter_valid_time([record_missing_memory_time], context) == [record_missing_memory_time]


def test_not_yet_valid_does_not_hard_filter_without_query_time():
    engine = GovernancePolicyEngine()
    record_after = MemoryRecord(
        memory_id="m_after",
        user_id="u1",
        content="I moved to Boston tomorrow.",
        timestamp=datetime.fromisoformat("2024-01-11T00:00:00"),
        source="user_message",
        scope="default",
        valid_from=datetime.fromisoformat("2024-01-11T00:00:00"),
        valid_until=None,
        is_deleted=False,
        confidence=1.0,
        memory_type="episodic",
        entity_key="profile:location",
    )
    context = QueryContext(user_id="u1", question="Where do I live now?", current_time=None, allowed_scopes=("default",))
    assert engine.filter_valid_time([record_after], context) == [record_after]


def test_answer_evidence_with_unknown_timestamp_is_not_not_yet_valid():
    engine = GovernancePolicyEngine()
    record = MemoryRecord(
        memory_id="m_answer",
        user_id="u1",
        content="I cleaned my white Adidas sneakers last month.",
        timestamp=datetime.fromisoformat("2024-01-05T00:00:00"),
        source="user_message",
        scope="default",
        valid_from=None,
        valid_until=None,
        is_deleted=False,
        confidence=1.0,
        memory_type="episodic",
        entity_key="profile:shoes",
    )
    context = QueryContext(user_id="u1", question="Which shoes did I clean?", current_time=datetime.fromisoformat("2024-01-10T00:00:00"), allowed_scopes=("default",))
    assert engine.filter_valid_time([record], context) == [record]


def test_not_yet_valid_ignores_unparseable_timestamps():
    engine = GovernancePolicyEngine()
    record = MemoryRecord(
        memory_id="m_bad_time",
        user_id="u1",
        content="I cleaned my white Adidas sneakers last month.",
        timestamp="not-a-datetime",
        source="user_message",
        scope="default",
        valid_from="not-a-datetime",
        valid_until=None,
        is_deleted=False,
        confidence=1.0,
        memory_type="episodic",
        entity_key="profile:shoes",
    )
    context = QueryContext(user_id="u1", question="Which shoes did I clean?", current_time="also-not-a-datetime", allowed_scopes=("default",))
    assert engine.filter_valid_time([record], context) == [record]
    assert engine.temporal_relation(record, context) == "unknown"
