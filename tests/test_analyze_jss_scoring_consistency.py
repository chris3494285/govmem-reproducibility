"""Unit tests for aggregate-only scoring-consistency analysis."""

from __future__ import annotations

from scripts.analyze_jss_scoring_consistency import identity_summary


def _row(*, condition: str, response: str, score: float, source: str = "fixture"):
    return {
        "pilot_block_id": "fixture-block",
        "condition": condition,
        "answer_sha256": response,
        "formal_score": score,
        "source": source,
        "generating_provider": "fixture-provider",
        "fault_type": "fixture-fault",
        "event_macro": "fixture-event",
    }


def test_identity_summary_finds_only_identical_response_score_change():
    triplets = [
        {
            "clean": _row(condition="clean", response="same", score=1.0),
            "corrupted": _row(condition="corrupted", response="same", score=0.0),
            "enforced": _row(condition="enforced", response="same", score=0.0),
        },
        {
            "clean": _row(condition="clean", response="stable", score=1.0),
            "corrupted": _row(condition="corrupted", response="stable", score=1.0),
            "enforced": _row(condition="enforced", response="stable", score=1.0),
        },
    ]

    counts, discrepancies = identity_summary(triplets)

    assert counts["overall"]["resolved_triplets"] == 2
    assert counts["overall"]["response_identity_unchanged"] == 2
    assert counts["overall"]["response_identity_unchanged_score_changed"] == 1
    assert counts["overall"]["response_identity_unchanged_score_unchanged"] == 1
    assert len(discrepancies) == 1
    assert discrepancies[0]["all_three_answer_identities_equal"] is True
    assert discrepancies[0]["frozen_scores"] == {
        "clean": 1.0,
        "corrupted": 0.0,
        "enforced": 0.0,
    }
