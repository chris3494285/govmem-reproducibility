"""Tests for public JSS aggregate sanitization and manuscript-count checks."""

from __future__ import annotations

import copy

import pytest

from scripts.build_jss_public_aggregate import build_public_rq1_summary


def _source():
    identity = {
        "numerator": 46,
        "denominator": 148,
        "estimate": 46 / 148,
        "ci95": [0.2, 0.4],
        "base_clusters": 105,
        "bootstrap_seed": 20261101,
        "bootstrap_iterations": 5000,
        "bootstrap_unit": "base_id cluster",
    }
    score_changed = {
        **identity,
        "numerator": 1,
        "estimate": 1 / 148,
    }
    return {
        "status": "verified",
        "schema_version": "fixture-v1",
        "cohort": {
            "bases": 731,
            "condition_rows": 4891,
            "matched_fault_pairs": 2080,
            "native_resolved_blocks": 298,
            "provider_replications": 12480,
        },
        "state_observability": {},
        "context_projection_observability": {},
        "downstream_output_observability": {
            "native_style": {
                "longmemeval_binary": {
                    "response_identity_unchanged": identity,
                    "response_identity_unchanged_score_changed": score_changed,
                }
            }
        },
        "statistical_design": {},
        "estimable_metrics": [],
        "not_estimable": {},
        "api_calls_made": 0,
        "no_model_generation_performed": True,
        "no_rescoring_performed": True,
        "provenance": {"private_identifier": "must not be copied"},
    }


def test_public_rq1_summary_drops_provenance_and_preserves_counts():
    result = build_public_rq1_summary(_source())

    assert "provenance" not in result
    assert result["cohort"]["bases"] == 731
    assert result["cohort"]["condition_rows"] == 4891
    assert result["cohort"]["matched_fault_pairs"] == 2080
    assert result["public_release_scope"]["identifier_free"] is True


def test_public_rq1_summary_rejects_count_drift():
    source = copy.deepcopy(_source())
    source["cohort"]["condition_rows"] = 4890

    with pytest.raises(ValueError, match="condition_rows"):
        build_public_rq1_summary(source)
