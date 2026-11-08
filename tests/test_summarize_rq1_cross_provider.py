from __future__ import annotations

import json

import pytest

from scripts.summarize_rq1_cross_provider import aggregate_summaries, write_csv


def _summary(*, rq1b: bool = False) -> dict:
    clean_key = "memconflict|temporal|clean|clean"
    fault_key = "memconflict|temporal|conflicting|ambiguous" if not rq1b else "memora|deletion|flipped|restrictive"
    clean = {
        "conditions_total": 2,
        "governance_violation_rate": None if rq1b else 0.5,
        "scored_governance_violation_n": 0 if rq1b else 2,
        "task_utility_rate": 1.0,
        "target_utility_rate": 1.0,
        "scored_task_utility_n": 2,
        "scored_target_utility_n": 2,
        "control_plane_false_rejection_rate": 0.0,
        "scored_control_plane_false_rejection_n": 2,
        "downstream_false_denial_rate": None,
        "scored_downstream_false_denial_n": 0,
    }
    fault = {
        "conditions_total": 2,
        "governance_violation_rate": None,
        "scored_governance_violation_n": 0,
        "task_utility_rate": 0.0,
        "target_utility_rate": 0.0,
        "scored_task_utility_n": 2,
        "scored_target_utility_n": 2,
        "control_plane_false_rejection_rate": 1.0,
        "scored_control_plane_false_rejection_n": 2,
        "downstream_false_denial_rate": 0.5,
        "scored_downstream_false_denial_n": 1,
    }
    paired = {
        "paired_n": 2,
        "paired_governance_violation_n": 0 if not rq1b else 2,
        "mean_delta_governance_violation": None if not rq1b else None,
        "paired_task_utility_n": 2,
        "mean_delta_task_utility": -1.0,
    }
    return {
        "groups": {clean_key: clean, fault_key: fault},
        "paired": {fault_key: paired},
    }


def _write_provider_summaries(tmp_path, provider: str, rq1a: dict, rq1b: dict):
    directory = tmp_path / provider
    directory.mkdir()
    a = directory / "rq1a_summary.json"
    b = directory / "rq1b_restrictive_summary.json"
    a.write_text(json.dumps(rq1a), encoding="utf-8")
    b.write_text(json.dumps(rq1b), encoding="utf-8")
    return a, b


def test_cross_provider_summary_preserves_nulls_denominators_and_unscorable_rows(tmp_path):
    paths = {
        provider: _write_provider_summaries(tmp_path, provider, _summary(), _summary(rq1b=True))
        for provider in ("deepseek", "openai", "anthropic")
    }
    summary = aggregate_summaries(paths)
    rq1a_conflict = next(row for row in summary["rows"] if row["track"] == "rq1a" and row["row_type"] == "paired_fault")
    assert rq1a_conflict["paired_n"] == 2
    assert rq1a_conflict["paired_governance_violation_n"] == 0
    assert rq1a_conflict["mean_delta_governance_violation"] is None
    assert rq1a_conflict["scorable"] is True  # task utility remains scored

    rq1b_fault = next(row for row in summary["rows"] if row["track"] == "rq1b" and row["row_type"] == "paired_fault")
    assert rq1b_fault["paired_n"] == 2
    assert rq1b_fault["paired_task_utility_n"] == 2
    assert rq1b_fault["scored_downstream_false_denial_n"] == 1
    assert rq1b_fault["downstream_false_denial_rate"] == 0.5
    assert rq1b_fault["fault_only_control_plane_false_rejection_rate"] == 1.0

    clean = next(row for row in summary["rows"] if row["track"] == "rq1a" and row["row_type"] == "clean_baseline")
    assert clean["clean_governance_violation_rate"] == 0.5
    assert clean["clean_scored_governance_violation_n"] == 2


def test_unscorable_conflicting_condition_is_explicit(tmp_path):
    source = _summary()
    key = "memconflict|temporal|conflicting|ambiguous"
    source["groups"][key]["scored_task_utility_n"] = 0
    source["paired"][key]["paired_task_utility_n"] = 0
    paths = {provider: _write_provider_summaries(tmp_path, provider, source, _summary(rq1b=True)) for provider in ("deepseek", "openai", "anthropic")}
    row = next(row for row in aggregate_summaries(paths)["rows"] if row["track"] == "rq1a" and row["row_type"] == "paired_fault")
    assert row["scorable"] is False
    assert row["unscorable_reason"]


def test_missing_expected_provider_or_file_fails_loudly(tmp_path):
    paths = {provider: _write_provider_summaries(tmp_path, provider, _summary(), _summary(rq1b=True)) for provider in ("deepseek", "openai")}
    with pytest.raises(ValueError, match="Missing expected RQ1 providers"):
        aggregate_summaries(paths)
    paths["anthropic"] = (tmp_path / "missing-a.json", tmp_path / "missing-b.json")
    with pytest.raises(FileNotFoundError):
        aggregate_summaries(paths)


def test_csv_has_paper_columns_and_preserves_null_as_empty_cell(tmp_path):
    paths = {provider: _write_provider_summaries(tmp_path, provider, _summary(), _summary(rq1b=True)) for provider in ("deepseek", "openai", "anthropic")}
    summary = aggregate_summaries(paths)
    output = tmp_path / "cross_provider.csv"
    write_csv(summary["rows"], output)
    text = output.read_text(encoding="utf-8")
    assert "mean_delta_governance_violation" in text.splitlines()[0]
    assert ",," in text
