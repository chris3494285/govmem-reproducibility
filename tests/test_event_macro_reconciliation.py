"""Offline tests for the B2 event-macro reconciliation.

The weighting logic is exercised on synthetic paired effects, so the public
suite needs neither the private v3 manifest nor the frozen provider scores.
The end-to-end reconciliation against the frozen artifacts is scoped
``requires_private_artifacts`` and skipped unless those inputs are present.
"""

from __future__ import annotations

from statistics import mean

import pytest

from scripts.analyze_final_main_results import _event_macro
from scripts.audit_event_macro_reconciliation import event_values, main


def _effect(base_id, event, fault, corruption_delta, recovery_delta):
    return {
        "base_id": base_id,
        "analysis_event_type": event,
        "fault_operator": fault,
        "corruption_delta": corruption_delta,
        "recovery_delta": recovery_delta,
    }


def _synthetic_effects():
    """One event with mixed fault counts (2 vs 3) plus two single-weight events.

    Base A supports two faults, base B supports three, so pair weighting
    over-weights base B while base weighting treats A and B equally.
    """
    effects = [
        # temporal: base A (2 faults), base B (3 faults)
        _effect("fixture::a", "temporal", "missing", -1.0, 1.0),
        _effect("fixture::a", "temporal", "stale", -1.0, 1.0),
        _effect("fixture::b", "temporal", "missing", 0.0, 0.0),
        _effect("fixture::b", "temporal", "stale", 0.0, 0.0),
        _effect("fixture::b", "temporal", "conflicting", 0.0, 0.0),
        # deletion and update: uniform fault counts
        _effect("fixture::c", "deletion", "missing", -0.5, 0.5),
        _effect("fixture::c", "deletion", "flipped", -0.5, 0.5),
        _effect("fixture::d", "update", "missing", -0.25, 0.25),
        _effect("fixture::d", "update", "conflicting", -0.25, 0.25),
    ]
    return effects


def test_base_weighting_differs_from_pair_weighting_only_when_fault_counts_mix():
    effects = _synthetic_effects()
    base = event_values(effects, "corruption_delta", weighting="base")
    pair = event_values(effects, "corruption_delta", weighting="pair")

    # temporal: base-weighted mean of {-1.0, 0.0} vs pair-weighted mean of 5 pairs
    assert base["temporal"] == pytest.approx(-0.5)
    assert pair["temporal"] == pytest.approx(-0.4)
    # uniform-fault events are identical under both weightings
    for event in ("deletion", "update"):
        assert base[event] == pytest.approx(pair[event])


def test_frozen_event_macro_equals_mean_of_base_weighted_event_values():
    effects = _synthetic_effects()
    macro = _event_macro(effects, "corruption_delta", replicates=50)["mean"]
    base = event_values(effects, "corruption_delta", weighting="base")
    assert macro == pytest.approx(mean(base.values()))
    # ...and therefore differs from the pair-weighted event mean.
    pair = event_values(effects, "corruption_delta", weighting="pair")
    assert macro != pytest.approx(mean(pair.values()))


def test_macro_is_an_unweighted_mean_over_events():
    effects = _synthetic_effects()
    macro = _event_macro(effects, "corruption_delta", replicates=50)["mean"]
    base = event_values(effects, "corruption_delta", weighting="base")
    assert macro == pytest.approx((-0.5 + -0.5 + -0.25) / 3)
    assert macro == pytest.approx(sum(base.values()) / len(base))


def test_recovery_uses_the_same_identity():
    effects = _synthetic_effects()
    macro = _event_macro(effects, "recovery_delta", replicates=50)["mean"]
    base = event_values(effects, "recovery_delta", weighting="base")
    assert macro == pytest.approx(mean(base.values()))


def test_script_blocks_when_private_inputs_are_absent(tmp_path, capsys):
    code = main(
        [
            "--manifest", str(tmp_path / "absent.jsonl"),
            "--scores-dir", str(tmp_path),
            "--no-write",
        ]
    )
    assert code == 2
    assert "EVENT_MACRO_STATUS=blocked_missing_private_input" in capsys.readouterr().out


@pytest.mark.requires_private_artifacts
def test_frozen_six_provider_reconciliation(tmp_path):
    from scripts.audit_event_macro_reconciliation import DEFAULT_MANIFEST, DEFAULT_SCORES_DIR

    if not DEFAULT_MANIFEST.is_file() or not (DEFAULT_SCORES_DIR / "openai.jsonl").is_file():
        pytest.skip("frozen v3 manifest / provider scores are not present in this checkout")

    code = main(["--output", str(tmp_path), "--replicates", "200"])
    assert code == 0

    import json

    result = json.loads((tmp_path / "event_macro_reconciliation.json").read_text(encoding="utf-8"))
    corruption = result["six_provider_mean"]["corruption"]
    recovery = result["six_provider_mean"]["recovery"]
    assert corruption["macro_matches_published"] is True
    assert recovery["macro_matches_published"] is True
    assert corruption["pair_weighted_reproduces_review_value"] is True
    assert recovery["pair_weighted_reproduces_review_value"] is True
    assert result["identity_errors"] == []
    assert result["reconciliation_status"] == "pass"
