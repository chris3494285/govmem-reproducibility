from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.make_experiment_report import build_experiment_report


def _write_result_dir(
    root: Path,
    *,
    policy: str,
    summary: dict,
    compare_rows: list[dict],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    payload["govmem_selection_policy"] = policy
    (root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (root / "compare_results.json").write_text(json.dumps({"examples": compare_rows}, indent=2, sort_keys=True), encoding="utf-8")
    return root


def _base_summary(**overrides) -> dict:
    summary = {
        "dataset_mode": "official_longmemeval",
        "total_examples": 3,
        "answer_mode": "llm",
        "judge_mode": "llm",
        "model": "gpt-4o-mini",
        "baseline_accuracy": 0.33,
        "govmem_accuracy": 0.40,
        "delta_accuracy": 0.07,
        "baseline_correct_count": 1,
        "govmem_correct_count": 2,
        "govmem_only_correct_count": 1,
        "baseline_only_correct_count": 0,
        "both_correct_count": 1,
        "both_wrong_count": 1,
        "baseline_evidence_hit_rate": 0.17,
        "govmem_evidence_hit_rate": 0.20,
        "average_memory_jaccard": 0.75,
        "identical_memory_set_rate": 0.50,
    }
    summary.update(overrides)
    return summary


def _compare_row(question_id: str, outcome: str, baseline_answer: str, govmem_answer: str) -> dict:
    return {
        "question_id": question_id,
        "question": f"Question {question_id}",
        "gold_answer": "Gold",
        "baseline_answer": baseline_answer,
        "govmem_answer": govmem_answer,
        "comparison_outcome": outcome,
        "answer_delta_note": "note",
        "baseline_retrieved_memories": [{"memory_id": f"b-{question_id}", "text": "baseline"}],
        "govmem_used_memories": [{"memory_id": f"g-{question_id}", "text": "govmem"}],
    }


def test_build_experiment_report_writes_markdown_and_csv(tmp_path):
    default_dir = _write_result_dir(
        tmp_path / "default",
        policy="default",
        summary=_base_summary(baseline_accuracy=0.33, govmem_accuracy=0.33, delta_accuracy=0.0, average_memory_jaccard=1.0, identical_memory_set_rate=1.0),
        compare_rows=[_compare_row("d1", "both_correct", "A", "A")],
    )
    rerank_dir = _write_result_dir(
        tmp_path / "rerank",
        policy="governance_rerank",
        summary=_base_summary(baseline_accuracy=0.33, govmem_accuracy=0.40, delta_accuracy=0.07, govmem_only_correct_count=2, baseline_only_correct_count=1, average_memory_jaccard=0.686, identical_memory_set_rate=0.42),
        compare_rows=[
            _compare_row("r1", "govmem_only_correct", "baseline", "govmem"),
            _compare_row("r2", "baseline_only_correct", "baseline", "govmem"),
        ],
    )
    oracle_dir = _write_result_dir(
        tmp_path / "oracle",
        policy="oracle_evidence",
        summary=_base_summary(baseline_accuracy=0.35, govmem_accuracy=0.43, delta_accuracy=0.08, govmem_only_correct_count=3, baseline_only_correct_count=1, average_memory_jaccard=0.6825, identical_memory_set_rate=0.45),
        compare_rows=[_compare_row("o1", "govmem_only_correct", "baseline", "govmem")],
    )

    output_path = tmp_path / "report.md"
    csv_path = tmp_path / "summary.csv"
    markdown, rows = build_experiment_report([default_dir, rerank_dir, oracle_dir], output_path, csv_output=csv_path)

    assert output_path.read_text(encoding="utf-8").startswith("# GovMem Experiment Report")
    assert "## Main Results Table" in markdown
    assert "governance_rerank" in markdown
    assert "oracle_evidence" in markdown
    assert len(rows) == 3

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["policy"] for row in csv_rows] == ["default", "governance_rerank", "oracle_evidence"]
    assert csv_rows[1]["govmem_accuracy"] == "0.400"


def test_build_experiment_report_handles_missing_optional_fields(tmp_path):
    default_dir = _write_result_dir(
        tmp_path / "default",
        policy="default",
        summary={
            "dataset_mode": "official_longmemeval",
            "total_examples": 1,
            "answer_mode": "llm",
            "judge_mode": "llm",
            "baseline_accuracy": 0.0,
            "govmem_accuracy": 0.0,
            "delta_accuracy": 0.0,
            "baseline_correct_count": 0,
            "govmem_correct_count": 0,
            "govmem_only_correct_count": 0,
            "baseline_only_correct_count": 0,
            "both_correct_count": 0,
            "both_wrong_count": 1,
        },
        compare_rows=[],
    )
    output_path = tmp_path / "report.md"
    markdown, _rows = build_experiment_report([default_dir], output_path)
    assert "n/a" in markdown
    assert output_path.exists()
