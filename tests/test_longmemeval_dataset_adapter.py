from pathlib import Path

from eval.longmemeval_adapter import LongMemEvalDatasetAdapter, convert_longmemeval_example, load_longmemeval_raw


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "longmemeval_sample_raw.jsonl"


def test_longmemeval_adapter_emits_generic_example():
    adapter = LongMemEvalDatasetAdapter()
    raw = load_longmemeval_raw(str(SOURCE))[0]
    example = adapter.adapt_example(raw)

    assert example.query.query_id == raw["id"]
    assert example.query.text == raw["question"]
    assert example.interactions
    assert example.interactions[0].role == "user"
    assert example.oracle_governance[0].interaction_id == example.interactions[0].interaction_id
    assert example.oracle_governance[0].authorized_scopes == ("default",)
    assert example.metadata["governance_source"] == "legacy_demo_annotation"
    assert example.oracle_governance[0].metadata["governance_source"] == "legacy_demo_annotation"


def test_longmemeval_adapter_load_matches_legacy_conversion_count():
    adapter = LongMemEvalDatasetAdapter()
    adapted = adapter.load(SOURCE)
    legacy = [convert_longmemeval_example(raw) for raw in load_longmemeval_raw(str(SOURCE))]

    assert len(adapted) == len(legacy)
    assert [item.query.query_id for item in adapted] == [item["id"] for item in legacy]
    assert [len(item.interactions) for item in adapted] == [len(item["memories"]) for item in legacy]


def test_longmemeval_adapter_accepts_repository_converted_format():
    source = ROOT / "data" / "official_longmemeval_oracle_20.jsonl"
    example = LongMemEvalDatasetAdapter().load(source)[0]
    assert example.query.metadata["dataset"] == "longmemeval"
    assert example.query.current_time is not None
    assert example.interactions


def test_longmemeval_adapter_rejects_unknown_schema_with_actionable_error(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text('{"id":"bad","question":"not enough schema"}\n', encoding="utf-8")
    try:
        LongMemEvalDatasetAdapter().load(source)
    except ValueError as exc:
        assert "raw records with sessions/question_time" in str(exc)
        assert "converted records with memories/current_time" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected an actionable schema error")
