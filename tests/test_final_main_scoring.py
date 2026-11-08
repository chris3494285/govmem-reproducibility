import hashlib
import json
from pathlib import Path

import pytest

from scripts.score_final_main_results import score_results


def _manifest(tmp_path):
    rows = [
        {
            "condition_id": "b::clean",
            "source_dataset": "longmemeval",
            "task_answer_type": "text",
            "task_question": "Where?",
            "task_oracle_answer": "Hallway",
        },
        {
            "condition_id": "b::stale::corrupted",
            "source_dataset": "longmemeval",
            "task_answer_type": "text",
            "task_question": "Where?",
            "task_oracle_answer": "Hallway",
        },
    ]
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def test_pipeline_joins_by_condition_and_is_deterministic(tmp_path):
    manifest = _manifest(tmp_path)
    result = tmp_path / "results.jsonl"
    result.write_text("".join(json.dumps(row) + "\n" for row in [
        {"condition_id": "b::stale::corrupted", "parsed_answer": "hallway", "parse_success": True},
        {"condition_id": "b::clean", "parsed_answer": "wrong", "parse_success": True},
    ]))
    out_a = tmp_path / "scored-a.jsonl"
    out_b = tmp_path / "scored-b.jsonl"
    score_results(result, manifest, out_a)
    score_results(result, manifest, out_b)
    assert out_a.read_bytes() == out_b.read_bytes()
    rows = [json.loads(line) for line in out_a.read_text().splitlines()]
    assert [row["condition_id"] for row in rows] == ["b::clean", "b::stale::corrupted"]
    assert [row["correctness"] for row in rows] == [False, True]
    assert all(row["scoring_version"] for row in rows)


def test_failed_provider_row_remains_unscored(tmp_path):
    manifest = _manifest(tmp_path)
    result = tmp_path / "results.jsonl"
    result.write_text(json.dumps({"condition_id": "b::clean", "error_type": "ProviderError", "parse_success": False}) + "\n")
    output = tmp_path / "scored.jsonl"
    rows = score_results(result, manifest, output)
    assert rows[0]["correctness"] is None
    assert rows[0]["scoring_status"] == "unscored"


def test_historical_retry_error_does_not_block_final_success(tmp_path):
    manifest = _manifest(tmp_path)
    result = tmp_path / "results.jsonl"
    result.write_text(json.dumps({
        "condition_id": "b::clean",
        "status": "success",
        "parse_success": True,
        "parsed_answer": "Hallway",
        "error_type": "JSONDecodeError",
        "attempt_errors": [{"attempt": 1, "error_type": "JSONDecodeError"}],
    }) + "\n")
    rows = score_results(result, manifest, tmp_path / "scored.jsonl")
    assert rows[0]["correctness"] is True
    assert rows[0]["scoring_status"] == "scored"
    assert rows[0]["error_type"] == "JSONDecodeError"


def test_unknown_condition_is_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    result = tmp_path / "results.jsonl"
    result.write_text(json.dumps({"condition_id": "unknown", "parsed_answer": "x"}) + "\n")
    with pytest.raises(ValueError, match="Unknown result condition_id"):
        score_results(result, manifest, tmp_path / "out.jsonl")
