from __future__ import annotations

import csv
import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path

import pytest

from eval.dataset import load_dataset, to_query_context
from eval.runner import run_mode, _answer_from_memory_objects, _answer_from_memory_objects_with_detail, _evidence_hit, _answer_first_between_options, _answer_count_before_event, _comparison_outcome, _compare_summary, _compare_memory_overlap_metrics, _compare_evidence_selection_diagnostics, _governance_rerank_candidates, _write_compare_artifacts
from eval.answering import HeuristicJudge, LLMAnswerGenerator, LLMJudge, AnswerResult, JudgeResult
from govmem.schema import MemoryRecord, QueryContext
from scripts.run_demo import parse_args


ROOT = Path(__file__).resolve().parents[1]


class MockChatClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, *, model: str, messages: list[dict], temperature: float = 0.0) -> str:
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        if not self.responses:
            raise AssertionError("MockChatClient was asked for more completions than provided responses.")
        return self.responses.pop(0)


def test_run_demo_parse_args_defaults():
    args = parse_args([])
    assert Path(args.dataset).name == "sample_longmemeval.jsonl"
    assert Path(args.results_dir).name == "results"
    assert args.mode == "demo"
    assert args.answer_mode == "heuristic"
    assert args.judge_mode == "heuristic"
    assert args.govmem_selection_policy == "default"
    assert args.start_index == 0
    assert args.max_examples is None
    assert args.resume is False


def test_run_demo_parse_args_accepts_selection_policy():
    args = parse_args(["--govmem-selection-policy", "oracle_evidence"])
    assert args.govmem_selection_policy == "oracle_evidence"


def _write_compare_dataset(path: Path, count: int = 4) -> None:
    lines = []
    for index in range(count):
        lines.append(
            json.dumps(
                {
                    "id": f"q{index}",
                    "user_id": f"u{index}",
                    "question": f"What is the answer for example {index}?",
                    "answer": f"Answer {index}",
                    "current_time": "2024-01-01T00:00:00",
                    "allowed_scopes": ["default"],
                    "dataset_mode": "official_longmemeval",
                    "memories": [
                        {
                            "memory_id": f"m{index}",
                            "user_id": f"u{index}",
                            "content": f"Answer {index}",
                            "timestamp": "2024-01-01T00:00:00",
                            "source": "user_message",
                            "scope": "default",
                            "valid_from": None,
                            "valid_until": None,
                            "is_deleted": False,
                            "confidence": 1.0,
                            "memory_type": "episodic",
                            "entity_key": f"entity:{index}",
                        }
                    ],
                }
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_policy_dataset(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "policy_1",
                        "user_id": "u1",
                        "question": "Which streaming service did I start using most recently?",
                        "answer": "Disney+",
                        "current_time": "2024-01-01T00:00:00",
                        "allowed_scopes": ["default"],
                        "dataset_mode": "official_longmemeval",
                        "memories": [
                            {
                                "memory_id": "m1",
                                "user_id": "u1",
                                "content": "I started using Hulu yesterday.",
                                "timestamp": "2024-01-02T00:00:00",
                                "source": "user_message",
                                "scope": "default",
                                "valid_from": None,
                                "valid_until": None,
                                "is_deleted": False,
                                "confidence": 1.0,
                                "memory_type": "episodic",
                                "entity_key": "entity:hulu",
                            },
                            {
                                "memory_id": "m2",
                                "user_id": "u1",
                                "content": "I started using Disney+ last month.",
                                "timestamp": "2023-12-01T00:00:00",
                                "source": "user_message",
                                "scope": "default",
                                "valid_from": None,
                                "valid_until": None,
                                "is_deleted": False,
                                "confidence": 1.0,
                                "memory_type": "episodic",
                                "entity_key": "entity:disney",
                            },
                            {
                                "memory_id": "m3",
                                "user_id": "u1",
                                "content": "I started using Netflix last year.",
                                "timestamp": "2023-11-01T00:00:00",
                                "source": "user_message",
                                "scope": "default",
                                "valid_from": None,
                                "valid_until": None,
                                "is_deleted": False,
                                "confidence": 1.0,
                                "memory_type": "episodic",
                                "entity_key": "entity:netflix",
                            },
                        ],
                    }
                )
            ]
        ),
        encoding="utf-8",
    )


def _record(memory_id: str, content: str, timestamp: str = "2024-01-01T00:00:00", *, confidence: float = 1.0, entity_key: str = "entity:1") -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        user_id="u1",
        content=content,
        timestamp=datetime.fromisoformat(timestamp),
        source="user_message",
        scope="default",
        valid_from=None,
        valid_until=None,
        is_deleted=False,
        confidence=confidence,
        memory_type="episodic",
        entity_key=entity_key,
    )


def test_run_demo_script_writes_custom_results_dir(tmp_path):
    results_dir = tmp_path / "longmemeval_sample"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_demo.py",
            "--dataset",
            "data/longmemeval_converted.jsonl",
            "--results-dir",
            str(results_dir),
        ],
        cwd=ROOT,
        check=True,
    )

    for filename in [
        "demo_results.json",
        "demo_summary.md",
        "demo_traces.json",
        "demo_trace_report.md",
    ]:
        assert (results_dir / filename).exists()


def test_run_demo_script_default_behavior():
    subprocess.run([sys.executable, "scripts/run_demo.py"], cwd=ROOT, check=True)
    for filename in [
        "demo_results.json",
        "demo_summary.md",
        "demo_traces.json",
            "demo_trace_report.md",
    ]:
        assert (ROOT / "results" / filename).exists()


def test_run_demo_script_compare_mode_writes_compare_outputs(tmp_path):
    results_dir = tmp_path / "official"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_demo.py",
            "--dataset",
            "data/official_longmemeval_oracle_20.jsonl",
            "--results-dir",
            str(results_dir),
            "--mode",
            "compare",
        ],
        cwd=ROOT,
        check=True,
    )

    for filename in [
        "summary.json",
        "compare_results.json",
        "examples.md",
    ]:
        assert (results_dir / filename).exists()

    compare_results = (results_dir / "compare_results.json").read_text(encoding="utf-8")
    assert "\"text\":" in compare_results
    assert "\"baseline_evidence_hit\":" in compare_results
    assert "\"govmem_evidence_hit\":" in compare_results
    assert "\"govmem_governance_decisions\":" in compare_results
    assert "\"judge_mode\":" in compare_results
    assert "\"baseline_judge_detail\":" in compare_results
    assert "\"govmem_judge_detail\":" in compare_results
    assert (results_dir / "report.md").exists()
    assert (results_dir / "summary_cases.csv").exists()
    assert (results_dir / "selection_misses.md").exists()
    assert (results_dir / "both_correct.md").exists()
    assert (results_dir / "govmem_only_correct.md").exists()
    assert (results_dir / "baseline_only_correct.md").exists()
    assert (results_dir / "both_wrong.md").exists()


def test_run_demo_compare_slice_uses_dataset_slice_after_load(tmp_path):
    dataset = tmp_path / "slice.jsonl"
    _write_compare_dataset(dataset, count=4)

    results_dir = tmp_path / "results"
    run_mode(
        dataset,
        results_dir=results_dir,
        mode="compare",
        answer_mode="oracle",
        judge_mode="heuristic",
        start_index=1,
        max_examples=2,
    )

    compare_results = json.loads((results_dir / "compare_results.json").read_text(encoding="utf-8"))
    assert [row["question_id"] for row in compare_results["examples"]] == ["q1", "q2"]
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_examples"] == 2


def test_compare_mode_enhanced_heuristic_writes_answer_details(tmp_path):
    results_dir = tmp_path / "official"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_demo.py",
            "--dataset",
            "data/official_longmemeval_oracle_20.jsonl",
            "--results-dir",
            str(results_dir),
            "--mode",
            "compare",
            "--answer-mode",
            "enhanced_heuristic",
        ],
        cwd=ROOT,
        check=True,
    )

    compare_results = json.loads((results_dir / "compare_results.json").read_text(encoding="utf-8"))
    row = compare_results["examples"][0]
    assert "baseline_answer_mode_detail" in row
    assert "govmem_answer_mode_detail" in row
    assert "answer_mode_note" in row
    assert (results_dir / "examples.md").read_text(encoding="utf-8").startswith("# Compare Examples")


def test_run_demo_compare_resume_skips_completed_question_ids(monkeypatch, tmp_path):
    dataset = tmp_path / "resume.jsonl"
    _write_compare_dataset(dataset, count=4)
    results_dir = tmp_path / "results"

    run_mode(
        dataset,
        results_dir=results_dir,
        mode="compare",
        answer_mode="oracle",
        judge_mode="heuristic",
        max_examples=2,
    )

    original_eval = __import__("eval.runner", fromlist=["_evaluate_compare_examples"])._evaluate_compare_examples
    captured_ids = []

    def wrapped(examples, *args, **kwargs):
        captured_ids.append([example.id for example in examples])
        return original_eval(examples, *args, **kwargs)

    monkeypatch.setattr("eval.runner._evaluate_compare_examples", wrapped)

    run_mode(
        dataset,
        results_dir=results_dir,
        mode="compare",
        answer_mode="oracle",
        judge_mode="heuristic",
        max_examples=4,
        resume=True,
    )

    assert captured_ids == [["q2", "q3"]]
    compare_results = json.loads((results_dir / "compare_results.json").read_text(encoding="utf-8"))
    assert [row["question_id"] for row in compare_results["examples"]] == ["q0", "q1", "q2", "q3"]
    assert len({row["question_id"] for row in compare_results["examples"]}) == 4
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_examples"] == 4


def test_compare_mode_oracle_uses_evidence_and_text_fields(tmp_path):
    dataset = tmp_path / "toy.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "toy_1",
                        "user_id": "u1",
                        "question": "What city do I live in now?",
                        "answer": "Seattle",
                        "current_time": "2024-01-02T00:00:00",
                        "allowed_scopes": ["default"],
                        "dataset_mode": "official_longmemeval",
                        "memories": [
                            {
                                "memory_id": "m1",
                                "user_id": "u1",
                                "content": "I moved to Seattle last month.",
                                "timestamp": "2024-01-01T00:00:00",
                                "source": "user_message",
                                "scope": "default",
                                "valid_from": "2024-01-01T00:00:00",
                                "valid_until": None,
                                "is_deleted": False,
                                "confidence": 1.0,
                                "memory_type": "episodic",
                                "entity_key": "official:test:1",
                            }
                        ],
                    }
                )
            ]
        ),
        encoding="utf-8",
    )

    results = run_mode(dataset, results_dir=tmp_path / "results", mode="compare", answer_mode="oracle")
    summary = results["summary"]
    assert summary["baseline_evidence_hit_count"] == 1
    assert summary["govmem_evidence_hit_count"] == 1
    assert summary["baseline_evidence_hit_rate"] == 1.0
    assert summary["govmem_evidence_hit_rate"] == 1.0

    compare_results = json.loads((tmp_path / "results" / "compare_results.json").read_text(encoding="utf-8"))
    row = compare_results["examples"][0]
    assert row["baseline_answer"] == "Seattle"
    assert row["govmem_answer"] == "Seattle"
    assert row["baseline_evidence_hit"] is True
    assert row["govmem_evidence_hit"] is True
    assert row["judge_mode"] == "heuristic"
    assert row["baseline_judge_detail"] in {"exact_match", "substring_match"}
    assert row["govmem_judge_detail"] in {"exact_match", "substring_match"}
    assert row["baseline_retrieved_memories"][0]["text"].startswith("I moved to Seattle")
    assert row["govmem_used_memories"][0]["text"].startswith("I moved to Seattle")
    assert row["baseline_answer_mode_detail"] == "direct_gold_substring"
    assert row["govmem_answer_mode_detail"] == "direct_gold_substring"


def test_compare_mode_oracle_evidence_helpers_use_only_supplied_memories():
    gold_answer = "white Adidas sneakers"

    baseline_retrieved_memories = [
        {"memory_id": "b1", "text": "I cleaned my white Adidas sneakers last month."},
    ]
    govmem_used_memories = [
        {"memory_id": "g1", "text": "I cleaned my black boots yesterday."},
    ]
    govmem_filtered_memories = [
        {"memory_id": "f1", "text": "I cleaned my white Adidas sneakers last month."},
    ]

    assert _evidence_hit(gold_answer, [item["text"] for item in baseline_retrieved_memories]) is True
    assert _evidence_hit(gold_answer, [item["text"] for item in govmem_used_memories]) is False

    assert _answer_from_memory_objects("oracle", gold_answer, "What color shoes did I clean?", baseline_retrieved_memories) == gold_answer
    assert _answer_from_memory_objects("oracle", gold_answer, "What color shoes did I clean?", govmem_used_memories) == "unknown"
    assert _answer_from_memory_objects("oracle", gold_answer, "What color shoes did I clean?", govmem_filtered_memories) == gold_answer


def test_enhanced_heuristic_direct_substring():
    answer, detail = _answer_from_memory_objects_with_detail(
        "enhanced_heuristic",
        "white Adidas sneakers",
        "Which shoes did I clean?",
        [{"memory_id": "m1", "text": "I cleaned my white Adidas sneakers last month."}],
    )
    assert answer == "white Adidas sneakers"
    assert detail == "direct_gold_substring"


def test_enhanced_heuristic_prefers_direct_substring_over_other_heuristics():
    answer, detail = _answer_from_memory_objects_with_detail(
        "enhanced_heuristic",
        "white Adidas sneakers",
        "Which shoes were cleaned first, the Adidas or the Vans?",
        [
            {"memory_id": "m1", "text": "I cleaned my white Adidas sneakers last month."},
            {"memory_id": "m2", "text": "I started marigolds on March 3rd."},
        ],
    )
    assert answer == "white Adidas sneakers"
    assert detail == "direct_gold_substring"


def test_first_between_options_with_explicit_dates():
    answer, detail = _answer_first_between_options(
        "Which seeds were started first, the tomatoes or the marigolds?",
        [
            {"memory_id": "m1", "text": "I started tomatoes on February 20th."},
            {"memory_id": "m2", "text": "I started marigolds on March 3rd."},
        ],
    )
    assert answer.lower() == "tomatoes"
    assert detail == "first_between_options"


def test_first_between_options_unknown_without_dates():
    answer, detail = _answer_first_between_options(
        "Which seeds were started first, the tomatoes or the marigolds?",
        [
            {"memory_id": "m1", "text": "I started tomatoes recently."},
            {"memory_id": "m2", "text": "I started marigolds later."},
        ],
    )
    assert answer == "unknown"
    assert detail == "unknown_insufficient_dates"


def test_count_before_event_with_explicit_dates():
    answer, detail = _answer_count_before_event(
        "How many charity events did I participate in before the 'Run for the Cure' event?",
        "1",
        [
            {"memory_id": "m1", "text": "I volunteered at the \"Food for Thought\" charity gala on September 25th."},
            {"memory_id": "m2", "text": "I participated in the \"Run for the Cure\" event on October 15th."},
            {"memory_id": "m3", "text": "I participated in the \"Bike-a-Thon\" charity event in November."},
        ],
    )
    assert answer == "1"
    assert detail == "count_before_event"


def test_count_before_event_unknown_without_target_date():
    answer, detail = _answer_count_before_event(
        "How many charity events did I participate in before the 'Run for the Cure' event?",
        "1",
        [
            {"memory_id": "m1", "text": "I volunteered at the \"Food for Thought\" charity gala recently."},
        ],
    )
    assert answer == "unknown"
    assert detail == "unknown_insufficient_dates"


def test_enhanced_heuristic_uses_only_govmem_used_memories():
    gold_answer = "white Adidas sneakers"
    baseline_retrieved_memories = [
        {"memory_id": "b1", "text": "I cleaned my white Adidas sneakers last month."},
    ]
    govmem_used_memories = [
        {"memory_id": "g1", "text": "I cleaned my black boots yesterday."},
    ]
    govmem_filtered_memories = [
        {"memory_id": "f1", "text": "I cleaned my white Adidas sneakers last month."},
    ]

    baseline_answer, baseline_detail = _answer_from_memory_objects_with_detail(
        "enhanced_heuristic",
        gold_answer,
        "Which shoes did I clean?",
        baseline_retrieved_memories,
    )
    govmem_answer, govmem_detail = _answer_from_memory_objects_with_detail(
        "enhanced_heuristic",
        gold_answer,
        "Which shoes did I clean?",
        govmem_used_memories,
    )
    filtered_answer, filtered_detail = _answer_from_memory_objects_with_detail(
        "enhanced_heuristic",
        gold_answer,
        "Which shoes did I clean?",
        govmem_filtered_memories,
    )

    assert baseline_answer == gold_answer
    assert baseline_detail == "direct_gold_substring"
    assert govmem_answer == "unknown"
    assert govmem_detail == "unknown_no_supported_pattern"
    assert filtered_answer == gold_answer
    assert filtered_detail == "direct_gold_substring"


def test_enhanced_heuristic_oracle_monotonicity_on_synthetic_examples(tmp_path):
    dataset = tmp_path / "synthetic.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "s1",
                        "user_id": "u1",
                        "question": "What city do I live in now?",
                        "answer": "Seattle",
                        "current_time": "2024-01-02T00:00:00",
                        "allowed_scopes": ["default"],
                        "dataset_mode": "official_longmemeval",
                        "memories": [
                            {
                                "memory_id": "m1",
                                "user_id": "u1",
                                "content": "I live in Seattle.",
                                "timestamp": "2024-01-01T00:00:00",
                                "source": "user_message",
                                "scope": "default",
                                "valid_from": None,
                                "valid_until": None,
                                "is_deleted": False,
                                "confidence": 1.0,
                                "memory_type": "episodic",
                                "entity_key": "e1",
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "id": "s2",
                        "user_id": "u1",
                        "question": "Which seeds were started first, the tomatoes or the marigolds?",
                        "answer": "tomatoes",
                        "current_time": "2024-01-02T00:00:00",
                        "allowed_scopes": ["default"],
                        "dataset_mode": "official_longmemeval",
                        "memories": [
                            {
                                "memory_id": "m1",
                                "user_id": "u1",
                                "content": "I started tomatoes on February 20th.",
                                "timestamp": "2024-01-01T00:00:00",
                                "source": "user_message",
                                "scope": "default",
                                "valid_from": None,
                                "valid_until": None,
                                "is_deleted": False,
                                "confidence": 1.0,
                                "memory_type": "episodic",
                                "entity_key": "e2",
                            },
                            {
                                "memory_id": "m2",
                                "user_id": "u1",
                                "content": "I started marigolds on March 3rd.",
                                "timestamp": "2024-01-01T00:00:00",
                                "source": "user_message",
                                "scope": "default",
                                "valid_from": None,
                                "valid_until": None,
                                "is_deleted": False,
                                "confidence": 1.0,
                                "memory_type": "episodic",
                                "entity_key": "e3",
                            },
                        ],
                    }
                ),
                json.dumps(
                    {
                        "id": "s3",
                        "user_id": "u1",
                        "question": "How many charity events did I participate in before the 'Run for the Cure' event?",
                        "answer": "1",
                        "current_time": "2024-01-02T00:00:00",
                        "allowed_scopes": ["default"],
                        "dataset_mode": "official_longmemeval",
                        "memories": [
                            {
                                "memory_id": "m1",
                                "user_id": "u1",
                                "content": "I volunteered at the \"Food for Thought\" charity gala on September 25th.",
                                "timestamp": "2024-01-01T00:00:00",
                                "source": "user_message",
                                "scope": "default",
                                "valid_from": None,
                                "valid_until": None,
                                "is_deleted": False,
                                "confidence": 1.0,
                                "memory_type": "episodic",
                                "entity_key": "e4",
                            },
                            {
                                "memory_id": "m2",
                                "user_id": "u1",
                                "content": "I participated in the \"Run for the Cure\" event on October 15th.",
                                "timestamp": "2024-01-01T00:00:00",
                                "source": "user_message",
                                "scope": "default",
                                "valid_from": None,
                                "valid_until": None,
                                "is_deleted": False,
                                "confidence": 1.0,
                                "memory_type": "episodic",
                                "entity_key": "e5",
                            },
                        ],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    oracle = run_mode(dataset, results_dir=tmp_path / "oracle", mode="compare", answer_mode="oracle")
    enhanced = run_mode(dataset, results_dir=tmp_path / "enhanced", mode="compare", answer_mode="enhanced_heuristic")

    assert enhanced["summary"]["baseline_correct_count"] >= oracle["summary"]["baseline_correct_count"]
    assert enhanced["summary"]["govmem_correct_count"] >= oracle["summary"]["govmem_correct_count"]
    assert enhanced["summary"]["direct_gold_substring_count"] >= 1
    assert enhanced["summary"]["count_before_event_count"] >= 0


def test_compare_mode_oracle_hits_when_govmem_has_answer():
    gold_answer = "white Adidas sneakers"
    baseline_retrieved_memories = [
        {"memory_id": "b1", "text": "I cleaned my black boots yesterday."},
    ]
    govmem_used_memories = [
        {"memory_id": "g1", "text": "I cleaned my white Adidas sneakers last month."},
    ]

    assert _evidence_hit(gold_answer, [item["text"] for item in baseline_retrieved_memories]) is False
    assert _evidence_hit(gold_answer, [item["text"] for item in govmem_used_memories]) is True
    assert _answer_from_memory_objects("oracle", gold_answer, "What color shoes did I clean?", baseline_retrieved_memories) == "unknown"
    assert _answer_from_memory_objects("oracle", gold_answer, "What color shoes did I clean?", govmem_used_memories) == gold_answer


def test_compare_mode_oracle_does_not_use_filtered_memories(tmp_path):
    dataset = tmp_path / "future_memory.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "future_1",
                        "user_id": "u1",
                        "question": "Which shoes did I clean last month?",
                        "answer": "white Adidas sneakers",
                        "current_time": None,
                        "allowed_scopes": ["default"],
                        "dataset_mode": "official_longmemeval",
                        "memories": [
                            {
                                "memory_id": "future_answer",
                                "user_id": "u1",
                                "content": "I cleaned my white Adidas sneakers last month.",
                                "timestamp": "2024-01-11T00:00:00",
                                "source": "user_message",
                                "scope": "default",
                                "valid_from": None,
                                "valid_until": None,
                                "is_deleted": False,
                                "confidence": 1.0,
                                "memory_type": "episodic",
                                "entity_key": "official:test:future",
                            },
                            {
                                "memory_id": "noise_1",
                                "user_id": "u1",
                                "content": "I bought a coffee yesterday.",
                                "timestamp": "2024-01-01T00:00:00",
                                "source": "user_message",
                                "scope": "default",
                                "valid_from": "2024-01-01T00:00:00",
                                "valid_until": None,
                                "is_deleted": False,
                                "confidence": 1.0,
                                "memory_type": "episodic",
                                "entity_key": "official:test:noise",
                            },
                        ],
                    }
                )
            ]
        ),
        encoding="utf-8",
    )

    results = run_mode(dataset, results_dir=tmp_path / "results", mode="compare", answer_mode="oracle")
    summary = results["summary"]
    assert summary["govmem_answer_evidence_available_count"] == 1
    assert summary["govmem_filtered_answer_evidence_count"] == 0
    assert summary["govmem_not_yet_valid_count"] == 0
    assert summary["govmem_not_yet_valid_answer_evidence_count"] == 0

    compare_results = json.loads((tmp_path / "results" / "compare_results.json").read_text(encoding="utf-8"))
    row = compare_results["examples"][0]
    assert row["baseline_answer"] == "white Adidas sneakers"
    assert row["govmem_answer"] == "white Adidas sneakers"
    assert row["baseline_evidence_hit"] is True
    assert row["govmem_evidence_hit"] is True
    assert row["answer_evidence_available"] is True
    assert row["answer_evidence_selected"] is True
    assert row["answer_evidence_filtered"] is False
    assert row["govmem_governance_decisions"][0]["matched_answer_evidence"] is True
    assert row["govmem_governance_decisions"][0]["filter_reason"] is None
    assert row["govmem_governance_decisions"][0]["temporal_relation"] == "unknown"
    assert row["govmem_governance_decisions"][0]["temporal_warning"] == "missing_or_unparseable_time"


@pytest.mark.parametrize("factory", [LLMAnswerGenerator, LLMJudge])
def test_llm_modes_require_api_key(monkeypatch, factory):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        factory()


def test_llm_answer_generator_uses_only_selected_memories():
    client = MockChatClient(["white Adidas sneakers"])
    generator = LLMAnswerGenerator(client=client, model="test-model")
    result = generator.generate(
        "Which shoes did I clean last month?",
        [{"memory_id": "g1", "text": "I cleaned my white Adidas sneakers last month."}],
        "white Adidas sneakers",
    )
    assert result.answer == "white Adidas sneakers"
    prompt = client.calls[0]["messages"][1]["content"]
    assert "white Adidas sneakers" in prompt
    assert "filtered" not in prompt.lower()
    assert "most specific answer span" in client.calls[0]["messages"][0]["content"]
    assert "Preserve important modifiers" in client.calls[0]["messages"][0]["content"]


def test_llm_answer_generator_does_not_use_filtered_memories():
    client = MockChatClient(['"white Adidas sneakers."'])
    generator = LLMAnswerGenerator(client=client, model="test-model")
    result = generator.generate(
        "Which shoes did I clean last month?",
        [{"memory_id": "g1", "text": "I cleaned my black boots yesterday."}],
        "white Adidas sneakers",
    )
    assert result.answer == "white Adidas sneakers"
    prompt = client.calls[0]["messages"][1]["content"]
    assert "black boots" in prompt
    assert "white Adidas sneakers" not in prompt


def test_llm_answer_generator_postprocesses_quotes_and_whitespace():
    client = MockChatClient(['  "white Adidas sneakers."  '])
    generator = LLMAnswerGenerator(client=client, model="test-model")
    result = generator.generate(
        "Which shoes did I clean last month?",
        [{"memory_id": "g1", "text": "I cleaned my white Adidas sneakers last month."}],
        "white Adidas sneakers",
    )
    assert result.answer == "white Adidas sneakers"


def test_llm_judge_parses_valid_json():
    client = MockChatClient(['{"correct": true, "rationale": "same meaning"}'])
    judge = LLMJudge(client=client, model="test-model")
    result = judge.evaluate("Question", "Seattle", "Seattle")
    assert result.correct is True
    assert result.detail == "llm:same meaning"
    assert result.rationale == "same meaning"


def test_llm_judge_invalid_json_falls_back_to_heuristic():
    client = MockChatClient(["not json"])
    judge = LLMJudge(client=client, model="test-model")
    result = judge.evaluate("Question", "Seattle", "Seattle")
    assert result.correct is True
    assert result.detail.startswith("invalid_json_fallback:")
    assert result.rationale == "invalid_json_fallback"


def test_heuristic_judge_handles_numeric_answers():
    judge = HeuristicJudge()
    result = judge.evaluate("How many events?", 4, "4")
    assert result.correct is True
    assert result.detail == "exact_match"


def test_comparison_outcome_helper():
    assert _comparison_outcome(True, True) == "both_correct"
    assert _comparison_outcome(True, False) == "baseline_only_correct"
    assert _comparison_outcome(False, True) == "govmem_only_correct"
    assert _comparison_outcome(False, False) == "both_wrong"


def test_compare_memory_overlap_metrics():
    identical = _compare_memory_overlap_metrics(
        {
            "baseline_retrieved_memories": [{"memory_id": "b1"}, {"memory_id": "b2"}],
            "govmem_used_memories": [{"memory_id": "b1"}, {"memory_id": "b2"}],
        }
    )
    assert identical["memory_overlap_count"] == 2
    assert identical["memory_union_count"] == 2
    assert identical["memory_jaccard"] == 1.0
    assert identical["memory_sets_identical"] is True
    assert identical["baseline_only_memory_ids"] == []
    assert identical["govmem_only_memory_ids"] == []

    partial = _compare_memory_overlap_metrics(
        {
            "baseline_retrieved_memories": [{"memory_id": "b1"}, {"memory_id": "b2"}],
            "govmem_used_memories": [{"memory_id": "b2"}, {"memory_id": "g1"}],
        }
    )
    assert partial["memory_overlap_count"] == 1
    assert partial["memory_union_count"] == 3
    assert partial["memory_jaccard"] == pytest.approx(1 / 3)
    assert partial["memory_sets_identical"] is False
    assert partial["baseline_only_memory_ids"] == ["b1"]
    assert partial["govmem_only_memory_ids"] == ["g1"]

    disjoint = _compare_memory_overlap_metrics(
        {
            "baseline_retrieved_memories": [{"memory_id": "b1"}, {"memory_id": "b2"}],
            "govmem_used_memories": [{"memory_id": "g1"}, {"memory_id": "g2"}],
        }
    )
    assert disjoint["memory_overlap_count"] == 0
    assert disjoint["memory_union_count"] == 4
    assert disjoint["memory_jaccard"] == 0.0
    assert disjoint["memory_sets_identical"] is False


def test_compare_evidence_selection_diagnostics():
    diagnostics = _compare_evidence_selection_diagnostics(
        {
            "gold_answer": 4,
            "govmem_governance_decisions": [
                {
                    "memory_id": "m1",
                    "text": "I participated in 4 charity events.",
                    "matched_answer_evidence": True,
                    "selected": False,
                    "filter_reason": "not_yet_valid",
                    "temporal_relation": "before_query",
                },
                {
                    "memory_id": "m2",
                    "text": "noise",
                    "matched_answer_evidence": False,
                    "selected": True,
                    "filter_reason": None,
                    "temporal_relation": "before_query",
                },
            ],
            "govmem_used_memories": [{"memory_id": "m2", "text": "noise"}],
        }
    )
    assert diagnostics["govmem_answer_evidence_available_but_not_selected"] is True
    assert diagnostics["govmem_answer_evidence_available_but_not_selected_count"] == 1
    assert diagnostics["not_selected_answer_evidence_memory_count"] == 1
    assert diagnostics["selected_non_answer_memory_count"] == 1
    assert diagnostics["govmem_not_selected_answer_evidence_memories"][0]["memory_id"] == "m1"
    assert diagnostics["govmem_selected_non_answer_memories"][0]["memory_id"] == "m2"


def test_governance_rerank_prefers_before_query_memory_and_avoids_gold_usage(monkeypatch, tmp_path):
    dataset = tmp_path / "policy_dataset.jsonl"
    _write_policy_dataset(dataset)

    result = run_mode(
        dataset,
        results_dir=tmp_path / "rerank",
        mode="compare",
        answer_mode="oracle",
        judge_mode="heuristic",
        govmem_selection_policy="governance_rerank",
        k=1,
    )
    row = result["rows"][0]
    assert row["govmem_selection_policy"] == "governance_rerank"
    assert row["govmem_used_memory_ids"] == ["m2"]
    assert row["baseline_memory_ids"] == ["m1"]
    assert row["memory_sets_identical"] is False
    assert row["memory_jaccard"] == 0.0
    assert any(item["policy"] == "governance_rerank" for item in row["govmem_selection_scores"])

    example = load_dataset(dataset)[0]
    with monkeypatch.context() as m:
        m.setattr("eval.runner.evidence_hit", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("gold answer should not be used")))
        selected, candidate_scores = _governance_rerank_candidates(example.question, to_query_context(example), example.memories, k=1)
        assert selected[0].memory_id == "m2"
        assert candidate_scores[0]["policy"] == "governance_rerank"


def test_default_policy_preserves_existing_behavior(tmp_path):
    dataset = tmp_path / "default_policy.jsonl"
    _write_compare_dataset(dataset, count=1)
    result = run_mode(
        dataset,
        results_dir=tmp_path / "default",
        mode="compare",
        answer_mode="oracle",
        judge_mode="heuristic",
        govmem_selection_policy="default",
        k=1,
    )
    row = result["rows"][0]
    assert row["baseline_memory_ids"] == row["govmem_used_memory_ids"]
    assert row["memory_sets_identical"] is True


def test_governance_rerank_boosts_action_overlap():
    question = "Which pair of shoes did I clean last month?"
    context = QueryContext(user_id="u1", question=question, current_time=datetime.fromisoformat("2024-01-01T00:00:00"), allowed_scopes=("default",))
    records = [
        _record("m1", "I bought my pair of shoes last month.", timestamp="2024-01-03T00:00:00", entity_key="entity:distractor"),
        _record("m2", "I cleaned my white Adidas sneakers last month.", timestamp="2024-01-02T00:00:00", entity_key="entity:answer"),
    ]
    selected, candidate_scores = _governance_rerank_candidates(question, context, records, k=1)
    assert selected[0].memory_id == "m2"
    top = next(item for item in candidate_scores if item["memory_id"] == "m2")
    assert "predicate_overlap" in top["selection_reasons"]


def test_governance_rerank_boosts_numeric_amount_evidence():
    question = "Which grocery store did I spend the most money at in the past month?"
    context = QueryContext(user_id="u1", question=question, current_time=datetime.fromisoformat("2024-01-01T00:00:00"), allowed_scopes=("default",))
    records = [
        _record("m1", "I spent $120 at Walmart last month.", timestamp="2024-01-03T00:00:00", entity_key="entity:walmart"),
        _record("m2", "I spent $150 at Thrive Market last month.", timestamp="2024-01-02T00:00:00", entity_key="entity:thrive"),
    ]
    selected, candidate_scores = _governance_rerank_candidates(question, context, records, k=1)
    assert selected[0].memory_id == "m2"
    top = next(item for item in candidate_scores if item["memory_id"] == "m2")
    assert "numeric_evidence" in top["selection_reasons"]


def test_governance_rerank_boosts_purchase_actions():
    question = "Which item did I purchase first, A or B?"
    context = QueryContext(user_id="u1", question=question, current_time=datetime.fromisoformat("2024-01-01T00:00:00"), allowed_scopes=("default",))
    records = [
        _record("m1", "I bought item A on Monday.", timestamp="2024-01-03T00:00:00", entity_key="entity:a"),
        _record("m2", "I looked at item B on Monday.", timestamp="2024-01-02T00:00:00", entity_key="entity:b"),
    ]
    selected, candidate_scores = _governance_rerank_candidates(question, context, records, k=1)
    assert selected[0].memory_id == "m1"
    top = next(item for item in candidate_scores if item["memory_id"] == "m1")
    assert "predicate_overlap" in top["selection_reasons"]


def test_oracle_evidence_promotes_answer_memory_and_marks_usage(tmp_path):
    dataset = tmp_path / "oracle_policy.jsonl"
    _write_policy_dataset(dataset)
    default_results = run_mode(
        dataset,
        results_dir=tmp_path / "default",
        mode="compare",
        answer_mode="oracle",
        judge_mode="heuristic",
        govmem_selection_policy="default",
        k=1,
    )
    oracle_results = run_mode(
        dataset,
        results_dir=tmp_path / "oracle",
        mode="compare",
        answer_mode="oracle",
        judge_mode="heuristic",
        govmem_selection_policy="oracle_evidence",
        k=1,
    )
    default_row = default_results["rows"][0]
    oracle_row = oracle_results["rows"][0]
    assert default_results["summary"]["govmem_selection_policy"] == "default"
    assert oracle_results["summary"]["govmem_selection_policy"] == "oracle_evidence"
    assert default_row["govmem_answer_evidence_available_but_not_selected"] is True
    assert oracle_row["govmem_answer_evidence_available_but_not_selected"] is False
    assert oracle_row["oracle_selection_used"] is True
    assert oracle_row["govmem_used_memory_ids"] == ["m2"]
    assert oracle_row["govmem_evidence_hit"] is True
    assert oracle_results["summary"]["govmem_oracle_selection_used_count"] == 1
def test_compare_summary_overlap_metrics():
    compare_rows = [
        {
            "baseline_correct": True,
            "govmem_correct": True,
            "comparison_outcome": "both_correct",
            "baseline_evidence_hit": True,
            "govmem_evidence_hit": True,
            "baseline_underspecified": False,
            "govmem_underspecified": False,
            "answer_evidence_available": True,
            "answer_evidence_selected": True,
            "answer_evidence_filtered": False,
            "govmem_governance_decisions": [],
            "baseline_retrieved_count": 2,
            "govmem_used_count": 2,
            "govmem_filtered_count": 0,
            "baseline_answer_mode_detail": "direct_gold_substring",
            "govmem_answer_mode_detail": "direct_gold_substring",
            "baseline_retrieved_memories": [{"memory_id": "a1"}, {"memory_id": "a2"}],
            "govmem_used_memories": [{"memory_id": "a1"}, {"memory_id": "a2"}],
        },
        {
            "baseline_correct": True,
            "govmem_correct": False,
            "comparison_outcome": "baseline_only_correct",
            "baseline_evidence_hit": True,
            "govmem_evidence_hit": False,
            "baseline_underspecified": False,
            "govmem_underspecified": False,
            "answer_evidence_available": False,
            "answer_evidence_selected": False,
            "answer_evidence_filtered": False,
            "govmem_governance_decisions": [],
            "baseline_retrieved_count": 2,
            "govmem_used_count": 2,
            "govmem_filtered_count": 0,
            "baseline_answer_mode_detail": "direct_gold_substring",
            "govmem_answer_mode_detail": "unknown_no_supported_pattern",
            "baseline_retrieved_memories": [{"memory_id": "b1"}, {"memory_id": "b2"}],
            "govmem_used_memories": [{"memory_id": "b2"}, {"memory_id": "g1"}],
        },
        {
            "baseline_correct": False,
            "govmem_correct": True,
            "comparison_outcome": "govmem_only_correct",
            "baseline_evidence_hit": False,
            "govmem_evidence_hit": True,
            "baseline_underspecified": False,
            "govmem_underspecified": False,
            "answer_evidence_available": False,
            "answer_evidence_selected": False,
            "answer_evidence_filtered": False,
            "govmem_governance_decisions": [],
            "baseline_retrieved_count": 1,
            "govmem_used_count": 2,
            "govmem_filtered_count": 0,
            "baseline_answer_mode_detail": "unknown_no_supported_pattern",
            "govmem_answer_mode_detail": "direct_gold_substring",
            "baseline_retrieved_memories": [{"memory_id": "c1"}],
            "govmem_used_memories": [{"memory_id": "d1"}, {"memory_id": "d2"}],
        },
    ]

    summary = _compare_summary(compare_rows, "official_longmemeval", "llm", "llm")
    assert summary["govmem_selection_policy"] == "default"
    assert summary["average_memory_jaccard"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert summary["identical_memory_set_count"] == 1
    assert summary["identical_memory_set_rate"] == pytest.approx(1 / 3)
    assert summary["average_baseline_only_memory_count"] == pytest.approx((0 + 1 + 1) / 3)
    assert summary["average_govmem_only_memory_count"] == pytest.approx((0 + 1 + 2) / 3)


def test_compare_summary_counts_and_underspecified_metrics():
    compare_rows = [
        {
            "baseline_correct": True,
            "govmem_correct": True,
            "comparison_outcome": "both_correct",
            "baseline_evidence_hit": True,
            "govmem_evidence_hit": True,
            "baseline_underspecified": False,
            "govmem_underspecified": False,
            "answer_evidence_available": True,
            "answer_evidence_selected": True,
            "answer_evidence_filtered": False,
            "govmem_governance_decisions": [],
            "baseline_retrieved_count": 2,
            "govmem_used_count": 2,
            "govmem_filtered_count": 0,
            "baseline_answer_mode_detail": "direct_gold_substring",
            "govmem_answer_mode_detail": "direct_gold_substring",
        },
        {
            "baseline_correct": True,
            "govmem_correct": False,
            "comparison_outcome": "baseline_only_correct",
            "baseline_evidence_hit": True,
            "govmem_evidence_hit": False,
            "baseline_underspecified": False,
            "govmem_underspecified": True,
            "answer_evidence_available": True,
            "answer_evidence_selected": True,
            "answer_evidence_filtered": True,
            "govmem_governance_decisions": [{"filter_reason": "not_yet_valid", "matched_answer_evidence": True}],
            "baseline_retrieved_count": 3,
            "govmem_used_count": 1,
            "govmem_filtered_count": 2,
            "baseline_answer_mode_detail": "direct_gold_substring",
            "govmem_answer_mode_detail": "unknown_no_supported_pattern",
        },
        {
            "baseline_correct": False,
            "govmem_correct": True,
            "comparison_outcome": "govmem_only_correct",
            "baseline_evidence_hit": False,
            "govmem_evidence_hit": True,
            "baseline_underspecified": True,
            "govmem_underspecified": False,
            "answer_evidence_available": True,
            "answer_evidence_selected": False,
            "answer_evidence_filtered": False,
            "govmem_governance_decisions": [],
            "baseline_retrieved_count": 1,
            "govmem_used_count": 2,
            "govmem_filtered_count": 1,
            "baseline_answer_mode_detail": "unknown_no_supported_pattern",
            "govmem_answer_mode_detail": "direct_gold_substring",
            "baseline_retrieved_memories": [{"memory_id": "b3"}],
            "govmem_used_memories": [{"memory_id": "g3"}, {"memory_id": "g4"}],
            "govmem_not_selected_answer_evidence_memories": [],
            "govmem_selected_non_answer_memories": [{"memory_id": "g3"}],
            "govmem_answer_evidence_available_but_not_selected": False,
        },
        {
            "baseline_correct": False,
            "govmem_correct": False,
            "comparison_outcome": "both_wrong",
            "baseline_evidence_hit": False,
            "govmem_evidence_hit": False,
            "baseline_underspecified": False,
            "govmem_underspecified": False,
            "answer_evidence_available": False,
            "answer_evidence_selected": False,
            "answer_evidence_filtered": False,
            "govmem_governance_decisions": [],
            "baseline_retrieved_count": 0,
            "govmem_used_count": 0,
            "govmem_filtered_count": 0,
            "baseline_answer_mode_detail": "unknown_no_supported_pattern",
            "govmem_answer_mode_detail": "unknown_no_supported_pattern",
            "baseline_retrieved_memories": [],
            "govmem_used_memories": [],
            "govmem_not_selected_answer_evidence_memories": [],
            "govmem_selected_non_answer_memories": [],
            "govmem_answer_evidence_available_but_not_selected": False,
        },
    ]

    summary = _compare_summary(compare_rows, "official_longmemeval", "llm", "llm")
    assert summary["total_examples"] == 4
    assert summary["both_correct_count"] == 1
    assert summary["baseline_only_correct_count"] == 1
    assert summary["govmem_only_correct_count"] == 1
    assert summary["both_wrong_count"] == 1
    assert summary["both_correct_count"] + summary["baseline_only_correct_count"] + summary["govmem_only_correct_count"] + summary["both_wrong_count"] == summary["total_examples"]
    assert summary["baseline_correct_count"] == summary["both_correct_count"] + summary["baseline_only_correct_count"]
    assert summary["govmem_correct_count"] == summary["both_correct_count"] + summary["govmem_only_correct_count"]
    assert summary["baseline_underspecified_count"] == 1
    assert summary["govmem_underspecified_count"] == 1
    assert summary["llm_answer_underspecified_count"] == 2
    assert summary["govmem_answer_evidence_available_but_not_selected_count"] == 1
    assert summary["average_memory_jaccard"] == pytest.approx(0.0)


def test_compare_artifacts_write_report_buckets_and_csv(tmp_path):
    compare_rows = [
        {
            "question_id": "q1",
            "question": "Question 1",
            "gold_answer": "Answer 1",
            "baseline_answer": "Answer 1",
            "govmem_answer": "Answer 1",
            "comparison_outcome": "both_correct",
            "answer_delta_note": "Both systems answered correctly",
            "baseline_correct": True,
            "govmem_correct": True,
            "baseline_evidence_hit": True,
            "govmem_evidence_hit": True,
            "baseline_underspecified": False,
            "govmem_underspecified": False,
            "answer_evidence_available": True,
            "answer_evidence_selected": True,
            "answer_evidence_filtered": False,
            "answer_evidence_filter_reasons": [],
            "govmem_governance_decisions": [{"matched_answer_evidence": True, "filter_reason": None, "selected": True}],
            "baseline_retrieved_memories": [{"memory_id": "b1", "text": "Answer 1"}],
            "govmem_used_memories": [{"memory_id": "g1", "text": "Answer 1"}],
            "govmem_filtered_memories": [],
            "baseline_answer_mode_detail": "direct_gold_substring",
            "govmem_answer_mode_detail": "direct_gold_substring",
            "baseline_judge_detail": "exact_match",
            "govmem_judge_detail": "exact_match",
            "baseline_retrieved_count": 1,
            "govmem_used_count": 1,
            "govmem_filtered_count": 0,
            "notes": "Both methods retrieved sufficient valid evidence.",
        },
        {
            "question_id": "q2",
            "question": "Question 2",
            "gold_answer": "Answer 2",
            "baseline_answer": "Answer 2",
            "govmem_answer": "wrong",
            "comparison_outcome": "baseline_only_correct",
            "answer_delta_note": "GovMem had answer evidence but answer generation failed",
            "baseline_correct": True,
            "govmem_correct": False,
            "baseline_evidence_hit": True,
            "govmem_evidence_hit": False,
            "baseline_underspecified": False,
            "govmem_underspecified": True,
            "answer_evidence_available": True,
            "answer_evidence_selected": False,
            "answer_evidence_filtered": True,
            "answer_evidence_filter_reasons": ["not_yet_valid"],
            "govmem_governance_decisions": [
                {"matched_answer_evidence": True, "filter_reason": "not_yet_valid", "selected": False}
            ],
            "baseline_retrieved_memories": [{"memory_id": "b2", "text": "Answer 2"}],
            "govmem_used_memories": [{"memory_id": "g2", "text": "noise"}],
            "govmem_filtered_memories": [{"memory_id": "f2", "text": "Answer 2", "filter_reason": "not_yet_valid"}],
            "baseline_answer_mode_detail": "direct_gold_substring",
            "govmem_answer_mode_detail": "unknown_no_supported_pattern",
            "baseline_judge_detail": "exact_match",
            "govmem_judge_detail": "mismatch",
            "baseline_retrieved_count": 1,
            "govmem_used_count": 1,
            "govmem_filtered_count": 1,
            "notes": "GovMem over-filtered or discarded the evidence the baseline used.",
        },
        {
            "question_id": "q3",
            "question": "Question 3",
            "gold_answer": "Answer 3",
            "baseline_answer": "wrong",
            "govmem_answer": "Answer 3",
            "comparison_outcome": "govmem_only_correct",
            "answer_delta_note": "GovMem recovered evidence baseline missed",
            "baseline_correct": False,
            "govmem_correct": True,
            "baseline_evidence_hit": False,
            "govmem_evidence_hit": True,
            "baseline_underspecified": True,
            "govmem_underspecified": False,
            "answer_evidence_available": True,
            "answer_evidence_selected": True,
            "answer_evidence_filtered": False,
            "answer_evidence_filter_reasons": [],
            "govmem_governance_decisions": [{"matched_answer_evidence": True, "filter_reason": None, "selected": True}],
            "baseline_retrieved_memories": [{"memory_id": "b3", "text": "noise"}],
            "govmem_used_memories": [{"memory_id": "g3", "text": "Answer 3"}],
            "govmem_filtered_memories": [{"memory_id": "f3", "text": "noise"}],
            "baseline_answer_mode_detail": "unknown_no_supported_pattern",
            "govmem_answer_mode_detail": "direct_gold_substring",
            "baseline_judge_detail": "mismatch",
            "govmem_judge_detail": "exact_match",
            "baseline_retrieved_count": 1,
            "govmem_used_count": 1,
            "govmem_filtered_count": 1,
            "notes": "GovMem applied policy filtering and conflict resolution before retrieval.",
        },
        {
            "question_id": "q4",
            "question": "Question 4",
            "gold_answer": "Answer 4",
            "baseline_answer": "wrong",
            "govmem_answer": "also wrong",
            "comparison_outcome": "both_wrong",
            "answer_delta_note": "Answer evidence was unavailable to both systems",
            "baseline_correct": False,
            "govmem_correct": False,
            "baseline_evidence_hit": False,
            "govmem_evidence_hit": False,
            "baseline_underspecified": False,
            "govmem_underspecified": False,
            "answer_evidence_available": False,
            "answer_evidence_selected": False,
            "answer_evidence_filtered": False,
            "answer_evidence_filter_reasons": [],
            "govmem_governance_decisions": [{"matched_answer_evidence": False, "filter_reason": None, "selected": False}],
            "baseline_retrieved_memories": [{"memory_id": "b4", "text": "noise"}],
            "govmem_used_memories": [{"memory_id": "g4", "text": "noise"}],
            "govmem_filtered_memories": [
                {"memory_id": "f4a", "text": "noise"},
                {"memory_id": "f4b", "text": "Answer 4", "filter_reason": "not_yet_valid"},
            ],
            "baseline_answer_mode_detail": "unknown_no_supported_pattern",
            "govmem_answer_mode_detail": "unknown_no_supported_pattern",
            "baseline_judge_detail": "mismatch",
            "govmem_judge_detail": "mismatch",
            "baseline_retrieved_count": 1,
            "govmem_used_count": 1,
            "govmem_filtered_count": 2,
            "notes": "Both systems failed",
        },
    ]

    results_dir = tmp_path / "compare_artifacts"
    outputs = _write_compare_artifacts(compare_rows, results_dir, "official_longmemeval", "llm", "llm", "data/fake_dataset.jsonl")
    summary = outputs["summary"]
    compare_results = json.loads((results_dir / "compare_results.json").read_text(encoding="utf-8"))
    assert summary["total_examples"] == len(compare_results["examples"])
    assert summary["both_correct_count"] + summary["baseline_only_correct_count"] + summary["govmem_only_correct_count"] + summary["both_wrong_count"] == summary["total_examples"]
    assert summary["baseline_correct_count"] == summary["both_correct_count"] + summary["baseline_only_correct_count"]
    assert summary["govmem_correct_count"] == summary["both_correct_count"] + summary["govmem_only_correct_count"]

    report = (results_dir / "report.md").read_text(encoding="utf-8")
    assert "data/fake_dataset.jsonl" in report
    assert "baseline accuracy" in report
    assert "govmem_answer_evidence_available_count" in report
    assert "govmem_not_selected_answer_evidence_count" in report
    assert "GovMem Selection Policy" in report
    assert "Memory Overlap" in report
    assert "GovMem Evidence Selection Misses" in report

    selection_misses = (results_dir / "selection_misses.md").read_text(encoding="utf-8")
    assert "answer evidence available but not selected" in selection_misses
    assert "question: Question 2" in selection_misses

    both_wrong = (results_dir / "both_wrong.md").read_text(encoding="utf-8")
    assert "question: Question 4" in both_wrong
    assert "Answer 4" in both_wrong
    assert "noise" in both_wrong
    assert "f4a" not in both_wrong

    csv_path = results_dir / "summary_cases.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0].keys() == {
        "question_id",
        "question",
        "gold_answer",
        "baseline_answer",
        "govmem_answer",
        "baseline_correct",
        "govmem_correct",
        "comparison_outcome",
        "answer_delta_note",
        "baseline_evidence_hit",
        "govmem_evidence_hit",
        "memory_jaccard",
        "memory_sets_identical",
        "memory_overlap_count",
        "baseline_only_memory_count",
        "govmem_only_memory_count",
        "govmem_answer_evidence_available",
        "govmem_answer_evidence_selected",
        "govmem_answer_evidence_filtered",
        "govmem_answer_evidence_available_but_not_selected",
    }
    assert rows[0]["question_id"] == "q1"


def test_compare_mode_llm_answer_generator_only_receives_selected_memories(monkeypatch, tmp_path):
    class RecordingAnswerGenerator:
        mode = "llm"

        def __init__(self):
            self.calls = []

        def generate(self, question, memories, gold_answer=None):
            self.calls.append([memory["text"] for memory in memories])
            if any("white Adidas sneakers" in memory["text"] for memory in memories):
                return AnswerResult(answer="white Adidas sneakers", detail="llm_response")
            return AnswerResult(answer="unknown", detail="llm_response")

    mock_generator = RecordingAnswerGenerator()
    monkeypatch.setattr("eval.runner.build_answer_generator", lambda mode: mock_generator)

    dataset = tmp_path / "llm_compare.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "llm_1",
                        "user_id": "u1",
                        "question": "Which shoes did I clean last month?",
                        "answer": "white Adidas sneakers",
                        "current_time": None,
                        "allowed_scopes": ["default"],
                        "dataset_mode": "official_longmemeval",
                        "memories": [
                            {
                                "memory_id": "m1",
                                "user_id": "u1",
                                "content": "I cleaned my white Adidas sneakers last month.",
                                "timestamp": "2024-01-01T00:00:00",
                                "source": "user_message",
                                "scope": "default",
                                "valid_from": None,
                                "valid_until": None,
                                "is_deleted": True,
                                "confidence": 1.0,
                                "memory_type": "episodic",
                                "entity_key": "profile:shoes",
                            },
                            {
                                "memory_id": "m2",
                                "user_id": "u1",
                                "content": "I bought coffee yesterday.",
                                "timestamp": "2024-01-01T00:00:00",
                                "source": "user_message",
                                "scope": "default",
                                "valid_from": None,
                                "valid_until": None,
                                "is_deleted": False,
                                "confidence": 1.0,
                                "memory_type": "episodic",
                                "entity_key": "memory:noise",
                            },
                        ],
                    }
                )
            ]
        ),
        encoding="utf-8",
    )

    results = run_mode(dataset, results_dir=tmp_path / "results", mode="compare", answer_mode="llm", judge_mode="heuristic")
    assert results["summary"]["answer_mode"] == "llm"
    assert results["summary"]["judge_mode"] == "heuristic"
    assert "model" in results["summary"]
    assert mock_generator.calls
    assert any("white Adidas sneakers" in text for text in mock_generator.calls[0])
    assert all("white Adidas sneakers" not in text for text in mock_generator.calls[1])

    compare_results = json.loads((tmp_path / "results" / "compare_results.json").read_text(encoding="utf-8"))
    row = compare_results["examples"][0]
    assert row["baseline_answer"] == "white Adidas sneakers"
    assert row["govmem_answer"] == "unknown"
    assert row["baseline_judge_detail"] == "exact_match"
    assert row["govmem_judge_detail"] == "mismatch"
