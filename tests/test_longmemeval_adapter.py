import json
from pathlib import Path

from eval.longmemeval_adapter import convert_longmemeval_example, convert_longmemeval_file, load_longmemeval_raw


RAW_PATH = Path("data/longmemeval_sample_raw.jsonl")


def test_raw_file_loads():
    raw_examples = load_longmemeval_raw(str(RAW_PATH))
    assert len(raw_examples) == 4
    assert raw_examples[0]["id"] == "lme_001"


def test_conversion_produces_expected_top_level_fields():
    raw_examples = load_longmemeval_raw(str(RAW_PATH))
    converted = convert_longmemeval_example(raw_examples[0])
    assert set(converted.keys()) == {"id", "user_id", "question", "answer", "current_time", "allowed_scopes", "memories"}
    assert converted["id"] == "lme_001"
    assert converted["allowed_scopes"] == ["default"]


def test_user_messages_become_memories_and_assistant_messages_are_skipped():
    raw_examples = load_longmemeval_raw(str(RAW_PATH))
    converted = convert_longmemeval_example(raw_examples[0])
    assert len(converted["memories"]) == 2
    assert all(memory["source"] == "user_message" for memory in converted["memories"])
    assert all(memory["memory_type"] == "episodic" for memory in converted["memories"])


def test_entity_key_heuristic_works():
    raw_examples = load_longmemeval_raw(str(RAW_PATH))
    loc = convert_longmemeval_example(raw_examples[0])
    pref = convert_longmemeval_example(raw_examples[1])
    work = convert_longmemeval_example(raw_examples[2])
    general = convert_longmemeval_example(raw_examples[3])

    assert {memory["entity_key"] for memory in loc["memories"]} == {"profile:location"}
    assert {memory["entity_key"] for memory in pref["memories"]} == {"preference:general"}
    assert {memory["entity_key"] for memory in work["memories"]} == {"profile:work"}
    assert {memory["entity_key"] for memory in general["memories"]} == {"memory:general"}


def test_convert_longmemeval_file_writes_valid_jsonl(tmp_path):
    output_path = tmp_path / "converted.jsonl"
    convert_longmemeval_file(str(RAW_PATH), str(output_path))
    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["id"] == "lme_001"
    assert parsed[0]["memories"][0]["source"] == "user_message"

