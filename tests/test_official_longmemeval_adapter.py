from __future__ import annotations

import json
from pathlib import Path

from eval.official_longmemeval import convert_official_example, convert_official_file, load_official_longmemeval
from eval.runner import run_demo


def _write_fixture(path: Path) -> None:
    raw = [
        {
            "question_id": "q1",
            "question_type": "temporal-reasoning",
            "question": "What city do I live in now?",
            "answer": "Seattle",
            "question_date": "2023/05/30 (Tue) 23:40",
            "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 02:21"],
            "haystack_session_ids": ["s1", "s2"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I live in Boston."},
                    {"role": "assistant", "content": "Got it.", "has_answer": False},
                ],
                [
                    {"role": "user", "content": "I moved to Seattle.", "has_answer": True},
                    {"role": "assistant", "content": "Nice.", "has_answer": False},
                ],
            ],
            "answer_session_ids": ["s2"],
        }
    ]
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def test_load_official_longmemeval_limit(tmp_path):
    source = tmp_path / "official.json"
    _write_fixture(source)

    loaded = load_official_longmemeval(str(source), limit=1)
    assert len(loaded) == 1
    assert loaded[0]["question_id"] == "q1"


def test_convert_official_example_structure_and_metadata():
    raw = {
        "question_id": "q1",
        "question_type": "temporal-reasoning",
        "question": "What city do I live in now?",
        "answer": "Seattle",
        "question_date": "2023/05/30 (Tue) 23:40",
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 02:21"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I live in Boston."},
                {"role": "assistant", "content": "Got it.", "has_answer": False},
            ],
            [
                {"role": "user", "content": "I moved to Seattle.", "has_answer": True},
                {"role": "assistant", "content": "Nice.", "has_answer": False},
            ],
        ],
        "answer_session_ids": ["s2"],
    }

    converted = convert_official_example(raw)
    assert set(converted.keys()) == {"id", "user_id", "question", "answer", "current_time", "allowed_scopes", "dataset_mode", "memories"}
    assert converted["dataset_mode"] == "official_longmemeval"
    assert converted["id"] == "q1"
    assert converted["current_time"].startswith("2023-05-30T23:40")
    assert converted["allowed_scopes"] == ["default"]
    assert len(converted["memories"]) == 2
    assert all(memory["source"] == "user_message" for memory in converted["memories"])
    assert converted["memories"][0]["timestamp"].startswith("2023-05-20T02:21")
    assert all(memory["valid_from"] is None for memory in converted["memories"])
    assert converted["memories"][1]["has_answer"] is True
    assert converted["memories"][1]["official_session_id"] == "s2"
    assert converted["memories"][1]["official_question_type"] == "temporal-reasoning"
    assert converted["memories"][1]["role"] == "user"
    assert converted["memories"][0]["entity_key"] != converted["memories"][1]["entity_key"]
    assert converted["memories"][1]["entity_key"] == "official:temporal-reasoning:s2:0"


def test_convert_official_example_include_assistant():
    raw = {
        "question_id": "q1",
        "question_type": "temporal-reasoning",
        "question": "What city do I live in now?",
        "answer": "Seattle",
        "question_date": "2023/05/30 (Tue) 23:40",
        "haystack_dates": ["2023/05/20 (Sat) 02:21"],
        "haystack_session_ids": ["s1"],
        "haystack_sessions": [[
            {"role": "user", "content": "I live in Boston."},
            {"role": "assistant", "content": "Got it.", "has_answer": False},
        ]],
        "answer_session_ids": ["s1"],
    }

    converted = convert_official_example(raw, include_assistant=True)
    assert len(converted["memories"]) == 2
    assert {memory["role"] for memory in converted["memories"]} == {"user", "assistant"}
    assert any(memory.get("has_answer") is False for memory in converted["memories"])


def test_convert_official_example_distinct_entity_keys_allow_retrieval():
    raw = {
        "question_id": "q2",
        "question_type": "temporal-reasoning",
        "question": "What city do I live in now?",
        "answer": "Seattle",
        "question_date": "2023/05/30 (Tue) 23:40",
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 02:21"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I live in Boston."},
            ],
            [
                {"role": "user", "content": "I moved to Seattle and now live in Seattle.", "has_answer": True},
            ],
        ],
        "answer_session_ids": ["s2"],
    }

    converted = convert_official_example(raw)
    entity_keys = [memory["entity_key"] for memory in converted["memories"]]
    assert len(entity_keys) == len(set(entity_keys))
    assert converted["memories"][1]["has_answer"] is True


def test_convert_official_file_writes_jsonl(tmp_path):
    source = tmp_path / "official.json"
    output = tmp_path / "converted.jsonl"
    _write_fixture(source)

    convert_official_file(str(source), str(output), limit=1)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["id"] == "q1"
    assert parsed["dataset_mode"] == "official_longmemeval"
    assert parsed["memories"][0]["official_session_id"] == "s1"


def test_convert_official_file_without_limit_converts_all_examples(tmp_path):
    source = tmp_path / "official.json"
    output = tmp_path / "converted.jsonl"
    raw = [
        {
            "question_id": "q1",
            "question_type": "temporal-reasoning",
            "question": "What city do I live in now?",
            "answer": "Seattle",
            "question_date": "2023/05/30 (Tue) 23:40",
            "haystack_dates": ["2023/05/20 (Sat) 02:21"],
            "haystack_session_ids": ["s1"],
            "haystack_sessions": [[{"role": "user", "content": "I moved to Seattle."}]],
            "answer_session_ids": ["s1"],
        },
        {
            "question_id": "q2",
            "question_type": "temporal-reasoning",
            "question": "What city do I live in now?",
            "answer": "Portland",
            "question_date": "2023/05/30 (Tue) 23:40",
            "haystack_dates": ["2023/05/20 (Sat) 02:21"],
            "haystack_session_ids": ["s2"],
            "haystack_sessions": [[{"role": "user", "content": "I moved to Portland."}]],
            "answer_session_ids": ["s2"],
        },
    ]
    source.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    convert_official_file(str(source), str(output))
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [item["id"] for item in parsed] == ["q1", "q2"]


def test_runner_adds_evidence_metrics_when_has_answer_metadata_present(tmp_path):
    source = tmp_path / "official.json"
    output = tmp_path / "converted.jsonl"
    _write_fixture(source)
    convert_official_file(str(source), str(output), limit=1)

    results = run_demo(output, results_dir=tmp_path / "results")
    for method in results["methods"]:
        assert "evidence_recall" in method
        assert "evidence_precision" in method

    naive = next(entry for entry in results["methods"] if entry["method"] == "naive_memory")
    assert naive["evidence_recall"] == 1.0
    assert naive["evidence_precision"] == 0.5

    traces = json.loads((tmp_path / "results" / "demo_traces.json").read_text(encoding="utf-8"))
    naive_trace = next(item for item in traces if item["method"] == "naive_memory")
    assert any(metadata.get("has_answer") is True for metadata in naive_trace["retrieved_memory_metadata"])
    assert all(item.get("dataset_mode") == "official_longmemeval" for item in traces)

    summary = (tmp_path / "results" / "demo_summary.md").read_text(encoding="utf-8")
    assert "Official Retrieval Sanity Check" in summary
    assert "not full QA evaluation" in summary
