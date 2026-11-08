import json

from scripts.run_rq1_sanity import parse_args, run_sanity


def _args(tmp_path, limit=10):
    return parse_args([
        "--gatemem-root", str(tmp_path / "gate"),
        "--memora-input", str(tmp_path / "memora"),
        "--memconflict-input", str(tmp_path / "conflict"),
        "--results-dir", str(tmp_path / "results"),
        "--model", "mock-model",
        "--cache-dir", str(tmp_path / "cache"),
        "--limit-per-slice", str(limit),
    ])


def test_sanity_builds_five_deterministic_slices_and_evaluates(tmp_path):
    calls = []
    evaluator_commands = []

    def batch_runner(**kwargs):
        calls.append(kwargs)
        return {
            "selected": kwargs["limit"], "executed": 1, "cache_hits": 0,
            "valid": 1, "invalid_or_error": 0,
        }

    def evaluator_runner(command):
        evaluator_commands.append(command)

    result, summary_path = run_sanity(
        _args(tmp_path, limit=10),
        batch_runner=batch_runner,
        evaluator_runner=evaluator_runner,
    )
    assert [call["query_type"] for call in calls] == ["privacy", "safety", "remembering", "dynamic", "conditional"]
    assert [call["limit"] for call in calls] == [10] * 5
    assert [path.name for path in [tmp_path / "results" / name for name in (
        "gatemem_privacy.jsonl", "gatemem_safety.jsonl", "memora_lifecycle.jsonl",
        "memconflict_dynamic.jsonl", "memconflict_conditional.jsonl",
    )]] == [
        "gatemem_privacy.jsonl", "gatemem_safety.jsonl", "memora_lifecycle.jsonl",
        "memconflict_dynamic.jsonl", "memconflict_conditional.jsonl",
    ]
    assert evaluator_commands[0][1].endswith("scripts/evaluate_rq1.py")
    assert evaluator_commands[0][-1] == str(summary_path)
    assert len(result["slices"]) == 5


def test_sanity_writes_explicit_failure_record_and_continues(tmp_path):
    calls = []
    evaluator_commands = []

    def batch_runner(**kwargs):
        calls.append(kwargs)
        if kwargs["query_type"] == "safety":
            raise RuntimeError("mock slice failure")
        return {"selected": 1, "executed": 1, "cache_hits": 0, "valid": 1, "invalid_or_error": 0}

    result, _ = run_sanity(
        _args(tmp_path),
        batch_runner=batch_runner,
        evaluator_runner=lambda command: evaluator_commands.append(command),
    )
    failure_path = tmp_path / "results" / "gatemem_safety.jsonl"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["orchestration_error"]["message"] == "mock slice failure"
    safety = next(item for item in result["slices"] if item["slice"] == "gatemem_safety")
    assert safety["error"]["message"] == "mock slice failure"
    assert len(calls) == 5
    assert evaluator_commands
    assert str(failure_path) in evaluator_commands[0]
