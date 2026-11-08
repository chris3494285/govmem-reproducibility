import csv
import json
from pathlib import Path

from scripts.build_system_validation_paper_artifacts import build


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_paper_tables_preserve_statistics_and_coverage(tmp_path):
    summary = build(output_dir=tmp_path)
    assert summary["table_rows"] == 7
    effects = _rows(tmp_path / "table_cross_system_effects.csv")
    keys = {(row["governance_dimension"], row["fault_type"], row["metric"]) for row in effects}
    assert len(keys) == 7
    auth = next(row for row in effects if row["governance_dimension"] == "authorization" and row["fault_type"] == "conflicting")
    assert auth["pooled_delta"] == "0.2000"
    deletion = next(row for row in effects if row["governance_dimension"] == "deletion" and row["fault_type"] == "flipped")
    assert deletion["pooled_delta"] == "1.0000"
    missing = next(row for row in effects if row["governance_dimension"] == "temporal" and row["fault_type"] == "missing")
    assert missing["pooled_delta"] == "0.0000"
    stale = next(row for row in effects if row["governance_dimension"] == "temporal" and row["fault_type"] == "stale" and row["metric"] == "governance_violation")
    assert stale["pooled_delta"] == "0.9792"
    assert stale["pooled_ci_low"] == "0.9375"
    assert stale["pooled_ci_high"] == "1.0000"
    utility = next(row for row in effects if row["governance_dimension"] == "temporal" and row["metric"] == "task_utility_success")
    assert utility["pooled_delta"] == "-1.0000"

    coverage = _rows(tmp_path / "table_scoring_coverage.csv")
    assert len(coverage) == 81
    stale_coverage = next(row for row in coverage if row["memory_system"] == "mem0" and row["governance_dimension"] == "temporal" and row["fault_type"] == "stale" and row["metric"] == "task_utility_success")
    assert stale_coverage["total_n"] == "20"
    assert stale_coverage["scored_n"] == "17"
    assert stale_coverage["unscored_n"] == "3"


def test_paper_artifact_files_and_cautious_text(tmp_path):
    build(output_dir=tmp_path)
    for name in (
        "table_cross_system_effects.csv", "table_paired_transitions.csv", "table_scoring_coverage.csv",
        "figure_cross_system_effects.pdf", "figure_cross_system_effects.png",
        "results_external_validation.md", "methods_external_validation.md", "table_cross_system_effects.tex",
    ):
        assert (tmp_path / name).exists()
    results = (tmp_path / "results_external_validation.md").read_text()
    assert "do not establish that every individual system effect is statistically significant" in results
    assert "Temporal coverage was incomplete" in results
    methods = (tmp_path / "methods_external_validation.md").read_text()
    assert "This is not full A-Mem." in methods
    assert "LLM-based write-time processing/evolution disabled" in methods


def test_paper_builder_does_not_need_inference_or_provider(tmp_path):
    build(output_dir=tmp_path)
    assert not (tmp_path / "system_validation_downstream.jsonl").exists()
