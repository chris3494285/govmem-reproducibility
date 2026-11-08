import json
from pathlib import Path

from scripts.analyze_final_main_results import (
    _balanced_base_ids,
    _pair_effects,
    _score,
    bootstrap_base_effects,
)


def _row(base, condition_type, score, fault=None, event="deletion", source="memora", pair=None, **extra):
    return {
        "condition_id": f"{base}::{condition_type}::{fault or 'clean'}",
        "base_id": base,
        "condition_type": condition_type,
        "fault_operator": fault,
        "analysis_event_type": event,
        "source_dataset": source,
        "pair_id": pair,
        "status": "success",
        "parse_success": True,
        "primary_score": score,
        **extra,
    }


def test_final_provider_failure_is_primary_score_zero():
    row = _row("b", "corrupted", None, "missing", status="error", parse_success=False, primary_score=None)
    assert _score(row) == 0
    assert _score(row, complete_case=True) is None


def test_successful_unscored_row_is_not_imputed_from_compatibility_zero():
    row = _row("b", "clean", 0, scoring_status="unscored", correctness=None)
    assert _score(row) is None


def test_pairing_uses_base_and_condition_ids_not_row_position():
    rows = [
        _row("b", "enforced", 1, "missing", pair="b::missing"),
        _row("b", "clean", 1),
        _row("b", "corrupted", 0, "missing", pair="b::missing"),
    ]
    effects = _pair_effects(rows)
    assert len(effects) == 1
    assert effects[0]["corruption_delta"] == -1
    assert effects[0]["recovery_delta"] == 1


def test_bootstrap_is_deterministic_and_base_level():
    effects = []
    for base, delta in (("a", 0), ("b", 1), ("c", 1)):
        effects.append({"base_id": base, "delta": delta})
    assert bootstrap_base_effects(effects, "delta", replicates=100, seed=42) == bootstrap_base_effects(effects, "delta", replicates=100, seed=42)


def test_balanced_selection_is_exactly_169_per_event():
    rows = []
    for event in ("deletion", "temporal", "update"):
        for index in range(200 if event != "temporal" else 169):
            rows.append({"base_id": f"{event}-{index}", "analysis_event_type": event})
    selected = _balanced_base_ids(rows)
    assert len(selected) == 507
    assert all(sum(row["base_id"] in selected for row in rows if row["analysis_event_type"] == event) == 169 for event in ("deletion", "temporal", "update"))


def test_no_provider_modules_are_used_by_analysis():
    source = Path("scripts/analyze_final_main_results.py").read_text()
    assert "from eval.providers" not in source
    assert "import openai" not in source
    assert "import anthropic" not in source
