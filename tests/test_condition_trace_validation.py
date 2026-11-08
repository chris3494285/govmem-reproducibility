"""Offline tests for the generic Clean -> Corrupted -> Enforced trace validator.

Synthetic fixtures mimic the v2/v3 condition-manifest schema without any real
benchmark identifier or text, so the validator logic runs in the public
snapshot.  The end-to-end run against the frozen private manifest is scoped
``requires_private_artifacts``.
"""

from __future__ import annotations

import json

import pytest

from scripts.validate_condition_traces import (
    ALLOWED_TRIPLET_VARIATION,
    build_triplets,
    main,
    validate,
)

BASE = "fixture::base-1"
PAIR = f"{BASE}::missing"


def _row(condition: str, **overrides):
    row = {
        "base_id": BASE,
        "condition_id": f"{BASE}::{condition}",
        "condition_type": condition,
        "pair_id": None if condition == "clean" else PAIR,
        "fault_operator": None if condition == "clean" else "missing",
        "source_dataset": "fixture_source",
        "analysis_event_type": "update",
        "schema_version": "fixture-condition-v1",
        "oracle_state": {"current_status": "valid", "ordered_evidence": [{"ordinal": 0, "session_id": "s0"}]},
        "internal_governance_state": {"current_status": "valid", "ordered_evidence": [{"ordinal": 0, "session_id": "s0"}]},
        "task_question": "How many items?",
        "task_oracle_answer": "4",
        "task_options": None,
        "source_evidence": {"answer_session_ids": ["s0"]},
        "model_visible_context_spec": {
            "content_source": "benchmark_native_evidence",
            "item_count": 2,
            "projection": "admitted_evidence",
        },
        "model_visible_context_text": "clean context",
        "model_visible_memory_items": ["a", "b"],
        "corruption_specification": None,
        "enforcement_specification": None,
    }
    if condition == "corrupted":
        row["corruption_specification"] = {
            "operator": "missing",
            "deterministic": True,
            "operation": "drop_admitted_evidence",
            "source_oracle_fields": ["ordered_evidence"],
        }
        row["model_visible_context_spec"] = {
            "content_source": "benchmark_native_evidence",
            "item_count": 1,
            "projection": "admitted_evidence",
        }
        row["model_visible_context_text"] = "corrupted context"
        row["model_visible_memory_items"] = ["a"]
    if condition == "enforced":
        row["corruption_specification"] = {
            "operator": "missing",
            "deterministic": True,
            "operation": "drop_admitted_evidence",
            "source_oracle_fields": ["ordered_evidence"],
        }
        row["enforcement_specification"] = {
            "corrects_operator": "missing",
            "deterministic": True,
            "operation": "validate_and_apply_corrected_governance_state",
            "oracle_state_reference": {
                "current_status": "valid",
                "ordered_evidence": [{"ordinal": 0, "session_id": "s0"}],
            },
        }
    row.update(overrides)
    return row


def _triplet_rows():
    return [_row("clean"), _row("corrupted"), _row("enforced")]


def test_schema_allows_only_bookkeeping_and_projection_to_vary():
    assert ALLOWED_TRIPLET_VARIATION == {
        "condition_id",
        "condition_type",
        "pair_id",
        "fault_operator",
        "corruption_specification",
        "enforcement_specification",
        "model_visible_context_text",
        "model_visible_memory_items",
        "model_visible_context_spec",
    }


def test_clean_triplet_passes_every_check():
    triplets, integrity = build_triplets(_triplet_rows())
    assert integrity["complete_triplets"] == 1
    result = validate(triplets)
    assert result["triplet_count"] == 1
    assert all(entry["passed"] == 1 for entry in result["triplet_integrity"].values())
    assert all(entry["passed"] == 1 for entry in result["governance_state"].values())
    assert result["surgical_invariance"]["violations_clean_vs_corrupted"] == []
    assert result["surgical_invariance"]["violations_clean_vs_enforced"] == []
    assert result["admission_projection"]["spec_changed"]["rate"] == 1.0
    assert result["admission_projection"]["spec_restored"]["rate"] == 1.0
    assert result["model_visible_context"]["text_changed"]["rate"] == 1.0
    assert result["model_visible_context"]["text_restored"]["rate"] == 1.0


def test_non_projection_field_drift_is_a_surgical_violation():
    rows = _triplet_rows()
    rows[1]["task_question"] = "How many items were added?"
    triplets, _ = build_triplets(rows)
    result = validate(triplets)
    assert result["surgical_invariance"]["violations_clean_vs_corrupted"] == [
        {"field": "task_question", "triplets": 1}
    ]
    assert result["surgical_invariance"]["violating_triplets"] == 1
    assert result["model_visible_context"]["text_restored"]["rate"] == 1.0


def test_broken_oracle_invariance_is_detected():
    rows = _triplet_rows()
    rows[1]["oracle_state"] = {"current_status": "stale", "ordered_evidence": []}
    triplets, _ = build_triplets(rows)
    result = validate(triplets)
    assert result["governance_state"]["oracle_state_invariant"]["passed"] == 0
    assert result["surgical_invariance"]["violations_clean_vs_corrupted"] == [
        {"field": "oracle_state", "triplets": 1}
    ]


def test_enforcement_not_restoring_context_is_detected():
    rows = _triplet_rows()
    rows[2]["model_visible_context_text"] = "partially restored"
    triplets, _ = build_triplets(rows)
    result = validate(triplets)
    assert result["model_visible_context"]["text_restored"] == {"n": 0, "rate": 0.0}
    assert result["model_visible_context"]["text_changed"]["n"] == 1


def test_enforcement_reference_mismatch_is_detected():
    rows = _triplet_rows()
    rows[2]["enforcement_specification"]["oracle_state_reference"] = {"current_status": "invalid"}
    triplets, _ = build_triplets(rows)
    result = validate(triplets)
    assert result["governance_state"]["enforcement_oracle_reference_matches_enforced_state"]["passed"] == 0


def test_orphan_pair_and_missing_clean_are_reported_not_crashed():
    rows = [_row("clean"), _row("corrupted")]
    triplets, integrity = build_triplets(rows)
    assert triplets == []
    assert integrity["orphan_pair_groups"] == 1

    rows = [_row("corrupted"), _row("enforced")]
    triplets, integrity = build_triplets(rows)
    assert triplets == []
    assert integrity["orphan_pair_groups"] == 1


def test_duplicate_clean_row_is_reported():
    rows = _triplet_rows() + [_row("clean", condition_id=f"{BASE}::clean::dup")]
    _triplets, integrity = build_triplets(rows)
    assert integrity["duplicate_clean_rows"] == 1


def test_provider_label_coverage_counts_missing_rows(tmp_path):
    scores = tmp_path / "scores"
    scores.mkdir()
    for provider in ("openai", "anthropic", "deepseek", "gemini", "kimi", "grok"):
        rows = []
        for condition in ("clean", "corrupted", "enforced"):
            row = {
                "condition_id": f"{BASE}::{condition}",
                "provider": provider,
                "status": "success",
                "scoring_status": "scored",
                "primary_score": 1,
            }
            rows.append(row)
        if provider == "kimi":
            rows[1]["primary_score"] = None  # unusable label
        if provider == "grok":
            rows.pop()  # missing row
        (scores / f"{provider}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    triplets, _ = build_triplets(_triplet_rows())
    result = validate(triplets, scores)
    labels = result["provider_frozen_proxy_labels"]["providers"]
    assert labels["openai"]["labelled"] == 3
    assert labels["kimi"]["unlabelled"] == 1
    assert labels["grok"]["missing_rows"] == 1
    assert result["provider_frozen_proxy_labels"]["missing_files"] == []


def test_absent_provider_files_are_listed_not_invented(tmp_path):
    triplets, _ = build_triplets(_triplet_rows())
    result = validate(triplets, tmp_path)
    report = result["provider_frozen_proxy_labels"]
    assert sorted(report["missing_files"]) == sorted(f"{p}.jsonl" for p in (
        "openai", "anthropic", "deepseek", "gemini", "kimi", "grok"
    ))
    assert report["providers"] == {}


def test_script_blocks_when_manifest_is_absent(tmp_path, capsys):
    code = main(["--manifest", str(tmp_path / "absent.jsonl"), "--no-write"])
    assert code == 2
    assert "CONDITION_TRACE_STATUS=blocked_missing_private_input" in capsys.readouterr().out


def test_script_writes_aggregate_only_output(tmp_path):
    manifest = tmp_path / "fixture_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in _triplet_rows()), encoding="utf-8"
    )
    out = tmp_path / "out"
    assert main(["--manifest", str(manifest), "--output", str(out)]) == 0
    text = (out / "condition_trace_validation.json").read_text(encoding="utf-8")
    assert "clean context" not in text
    assert "corrupted context" not in text
    assert "How many items" not in text


@pytest.mark.requires_private_artifacts
def test_frozen_manifest_validation(tmp_path):
    from scripts.validate_condition_traces import DEFAULT_MANIFEST, DEFAULT_SCORES_DIR

    if not DEFAULT_MANIFEST.is_file():
        pytest.skip("frozen v3 condition manifest is not present in this checkout")
    args = ["--manifest", str(DEFAULT_MANIFEST), "--output", str(tmp_path)]
    if (DEFAULT_SCORES_DIR / "openai.jsonl").is_file():
        args += ["--scores-dir", str(DEFAULT_SCORES_DIR)]
    assert main(args) == 0
    result = json.loads((tmp_path / "condition_trace_validation.json").read_text(encoding="utf-8"))
    assert result["triplet_count"] == 2080
    assert result["surgical_invariance"]["violating_triplets"] == 0
    assert result["model_visible_context"]["text_restored"]["n"] == 2080
