import copy

from scripts.analyze_system_validation_statistics import (
    analyze,
    bootstrap_mean,
    exact_mcnemar_p_value,
)


def _rows(*, missing=False):
    rows = []
    for system in ("mem0", "langmem", "amem"):
        for base in ("b1", "b2"):
            clean = {
                "memory_system": system, "base_example_id": base,
                "governance_dimension": "temporal", "condition": "clean", "fault_type": None,
                "governance_violation": False, "task_utility_success": True,
            }
            fault = {
                "memory_system": system, "base_example_id": base,
                "governance_dimension": "temporal", "condition": "stale", "fault_type": "stale",
                "governance_violation": None if missing else True,
                "task_utility_success": None if missing else False,
            }
            rows.extend((clean, fault))
    return rows


def test_paired_effects_and_exact_transition_counts():
    result = analyze(_rows(), replicates=100, seed=42)
    row = next(item for item in result["by_system"] if item["memory_system"] == "mem0" and item["metric"] == "governance_violation")
    assert row["total_base_n"] == 2
    assert row["paired_scored_n"] == 2
    assert row["mean_delta"] == 1
    assert row["clean0_fault1_n"] == 2
    assert row["discordant_harm_n"] == 2
    assert row["discordant_improvement_n"] == 0
    assert result["mcnemar"][0]["exact_two_sided_p_value"] == 0.5


def test_bootstrap_is_deterministic_and_base_level():
    values = [0.0, 1.0, 1.0]
    assert bootstrap_mean(values, replicates=200, seed=42) == bootstrap_mean(values, replicates=200, seed=42)
    result = analyze(_rows(), replicates=50, seed=42)
    pooled = next(item for item in result["pooled_clustered"] if item["metric"] == "governance_violation")
    assert pooled["clustered_base_n"] == 2
    assert pooled["systems_contributing_min"] == 3
    assert pooled["systems_contributing_max"] == 3
    assert pooled["mean_systems_per_base"] == 3


def test_missing_temporal_values_are_not_imputed():
    result = analyze(_rows(missing=True), replicates=50)
    row = next(item for item in result["by_system"] if item["metric"] == "task_utility_success")
    assert row["paired_scored_n"] == 0
    assert row["paired_unscored_n"] == 2
    assert row["mean_delta"] is None
    pooled = next(item for item in result["pooled_clustered"] if item["metric"] == "task_utility_success")
    assert pooled["clustered_base_n"] == 0
    assert pooled["mean_clustered_delta"] is None


def test_exact_mcnemar_zero_and_one_sided_cases():
    assert exact_mcnemar_p_value(0, 0) == 1.0
    assert exact_mcnemar_p_value(4, 0) == 0.125
    assert exact_mcnemar_p_value(0, 4) == 0.125


def test_sign_consistency_is_computed_not_assumed():
    rows = _rows()
    for row in rows:
        if row["memory_system"] == "amem" and row["condition"] == "stale":
            row["governance_violation"] = False
    result = analyze(rows, replicates=20)
    sign = next(item for item in result["sign_consistency"] if item["metric"] == "governance_violation")
    assert sign["positive_system_n"] == 2
    assert sign["zero_system_n"] == 1
    assert sign["same_sign_all_systems"] is False

