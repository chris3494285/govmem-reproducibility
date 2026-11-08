from __future__ import annotations

import json

import pytest

from scripts.analyze_rq2_fault_severity import analyze


def _row(base, condition, *, operator=None, direction=None, gvr=None, utility=None, denial=None, control=None):
    return {
        "run_id": f"{base}-{condition}-{operator or 'clean'}",
        "example_id": base,
        "source_dataset": "memconflict",
        "dimension": "temporal",
        "condition_type": condition,
        "fault_spec": None if condition == "clean" else {"operator": operator, "direction": direction},
        "status": "success",
        "governance_violation": gvr,
        "task_utility": utility,
        "downstream_false_denial": denial,
        "control_plane_false_rejection": control,
    }


def _write_case(tmp_path, provider, *, flip=False, tie=False, unscorable=False):
    directory = tmp_path / provider
    directory.mkdir()
    rows = []
    # b1/b2 contain both faults; b3 only A and b4 only B, proving that
    # unmatched group members do not enter the pairwise comparison.
    for base in ("b1", "b2", "b3", "b4"):
        rows.append(_row(base, "clean", gvr=False, utility=True))
    for base in ("b1", "b2", "b3"):
        rows.append(_row(base, "fault", operator="stale", direction="unsafe", gvr=False if tie else flip, utility=False))
    for base in ("b1", "b2", "b4"):
        rows.append(_row(base, "fault", operator="missing", direction="ambiguous", gvr=False if tie else (not flip), utility=True))
    if unscorable:
        for row in rows:
            if (row.get("fault_spec") or {}).get("operator") == "stale":
                row["governance_violation"] = None
    rq1a = directory / "rq1a_scored.jsonl"
    rq1a.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    b_rows = []
    for base in ("t1", "t2"):
        b_rows.append(_row(base, "clean", utility=True, gvr=None, control=False))
        b_rows.append(_row(base, "fault", operator="expired", direction="restrictive", utility=False, denial=True, control=True))
    for base in ("d1", "d2"):
        b_rows.append({**_row(base, "clean", utility=True, gvr=None, control=False), "source_dataset": "memora", "dimension": "deletion"})
        b_rows.append({**_row(base, "fault", operator="flipped", direction="restrictive", utility=False, denial=None, control=True), "source_dataset": "memora", "dimension": "deletion"})
    rq1b = directory / "rq1b_restrictive_scored.jsonl"
    rq1b.write_text("".join(json.dumps(row) + "\n" for row in b_rows), encoding="utf-8")
    return rq1a, rq1b


def test_matched_intersection_nulls_and_win_tie_loss_and_bootstrap(tmp_path):
    paths = {
        "deepseek": _write_case(tmp_path, "deepseek"),
        "openai": _write_case(tmp_path, "openai", tie=True),
        "anthropic": _write_case(tmp_path, "anthropic", unscorable=True),
    }
    first = analyze(paths)
    second = analyze(paths)
    assert first == second

    deepseek = next(row for row in first["rq2a_matched_comparisons"] if row["provider"] == "deepseek")
    assert deepseek["matched_base_n"] == 2
    assert deepseek["scorable_matched_n"] == 2
    assert deepseek["win_a_n"] == 2
    assert deepseek["tie_n"] == 0
    assert deepseek["win_b_n"] == 0
    assert deepseek["matched_difference_ci_low"] is not None

    anthropic = next(row for row in first["rq2a_matched_comparisons"] if row["provider"] == "anthropic")
    assert anthropic["scorable_matched_n"] == 0
    assert anthropic["mean_matched_difference"] is None
    assert anthropic["gvr_unscorable_reason"]

    stability = next(row for row in first["cross_provider_stability"])
    assert stability["providers_a_gt_b_n"] == 1
    assert stability["providers_tied_n"] == 1
    assert stability["providers_b_gt_a_n"] == 0
    assert stability["direction_unanimous"] is False


def test_rq2b_is_descriptive_and_preserves_denominators(tmp_path):
    paths = {provider: _write_case(tmp_path, provider) for provider in ("deepseek", "openai", "anthropic")}
    rows = analyze(paths)["rq2b_restrictive_descriptive"]
    temporal = next(row for row in rows if row["source_dataset"] == "memconflict")
    assert temporal["matched_base_n"] == 2
    assert temporal["scorable_matched_n"] == 2
    assert temporal["mean_delta_utility"] == -1
    assert temporal["downstream_false_denial_rate"] == 1
    assert temporal["downstream_false_denial_denominator_n"] == 2
    assert temporal["fault_only_control_plane_false_rejection_n"] == 2
    assert all(row["source_dataset"] == "memconflict" for row in rows if row["governance_dimension"] == "temporal")
    assert not any(row["source_dataset"] == "memconflict" and row["governance_dimension"] == "deletion" for row in rows)


def test_provider_cohort_misalignment_fails_loudly(tmp_path):
    paths = {provider: _write_case(tmp_path, provider) for provider in ("deepseek", "openai", "anthropic")}
    path = paths["anthropic"][0]
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cohorts do not align"):
        analyze(paths)
