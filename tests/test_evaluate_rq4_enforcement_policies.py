from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from scripts.evaluate_rq4_enforcement_policies import (
    POLICIES,
    RQ1A_MANIFEST,
    RQ1B_MANIFEST,
    _tradeoffs,
    aggregate_policy_rows,
    evaluate,
    evaluate_rows,
    governance_exposure,
    policy_action,
    state_selection,
)


def _row(
    *,
    example_id: str = "base-1",
    dataset: str = "gatemem",
    dimension: str = "authorization",
    condition_type: str = "fault",
    observed: dict | None = None,
    gold: str | None = "allow",
    gold_state: dict | None = None,
    source_metadata: dict | None = None,
    operator: str = "missing",
    direction: str = "ambiguous",
) -> dict:
    if observed is None:
        observed = {
            "authorization": None,
            "authorization_assertions": [],
            "deletion": None,
            "temporal": None,
            "temporal_assertions": [],
        }
    return {
        "run_id": f"{example_id}::{condition_type}",
        "example_id": example_id,
        "source_dataset": dataset,
        "dimension": dimension,
        "condition_type": condition_type,
        "gold_admission_decision": gold,
        "gold_governance_state": gold_state or {},
        "source_metadata": source_metadata or {},
        "observed_governance_state": observed,
        "fault_spec": None if condition_type == "clean" else {"operator": operator, "direction": direction},
    }


def _auth(decision: str | None, assertions: list | None = None) -> dict:
    return {
        "authorization": None if decision is None else {"decision": decision},
        "authorization_assertions": assertions or [],
        "deletion": None,
        "temporal": None,
        "temporal_assertions": [],
    }


def _temporal(status: str | None, assertions: list | None = None) -> dict:
    return {
        "authorization": None,
        "authorization_assertions": [],
        "deletion": None,
        "temporal": None if status is None else {"status": status},
        "temporal_assertions": assertions or [],
    }


def test_policy_actions_cover_required_control_plane_strategies():
    missing = _row(observed=_auth(None))
    conflicting = _row(observed=_auth(None, [{"decision": "allow"}, {"decision": "deny"}]))

    assert policy_action("permissive", missing)[0] == "admit"
    assert policy_action("deny_by_default", missing)[0] == "deny"
    assert policy_action("revalidate_on_uncertainty", missing)[0] == "revalidate"
    assert policy_action("conflict_quarantine", conflicting)[0] == "quarantine"
    action, applicable, _ = policy_action("version_validation", missing)
    assert (action, applicable) == ("not_applicable", False)


def test_version_validation_rejects_stale_or_expired_and_preserves_current():
    expired = _row(
        dimension="temporal",
        observed=_temporal("expired"),
        operator="stale",
        source_metadata={"current_value": {"state": "current"}, "current_session": {"session_id": 7}},
    )
    current = _row(dimension="temporal", observed=_temporal("valid"), operator="missing")
    assert policy_action("version_validation", expired)[0] == "admit"
    assert policy_action("version_validation", current)[0] == "admit"

    stale_deletion = _row(
        dimension="deletion",
        observed={"deletion": {"status": "active"}},
        operator="stale",
    )
    assert policy_action("permissive", stale_deletion)[0] == "admit"
    assert policy_action("deny_by_default", stale_deletion)[0] == "deny"
    assert policy_action("version_validation", stale_deletion)[0] == "deny"


def test_gold_metrics_distinguish_unsafe_exposure_hard_rejection_and_deferred_access():
    unsafe = _row(gold="deny", observed=_auth(None))
    hard = _row(gold="allow", observed=_auth(None))
    instances = evaluate_rows([unsafe, hard])

    permissive_unsafe = next(x for x in instances if x["policy"] == "permissive" and x["base_example_id"] == "base-1")
    # The first row is unsafe under the permissive policy; the second row has
    # the same base id only in this synthetic fixture, so inspect by source
    # order through direct policy evaluation below instead.
    assert policy_action("permissive", unsafe)[0] == "admit"
    assert policy_action("deny_by_default", hard)[0] == "deny"
    assert policy_action("revalidate_on_uncertainty", hard)[0] == "revalidate"
    assert permissive_unsafe["unsafe_exposure"] is True

    deny_instance = next(x for x in instances if x["policy"] == "deny_by_default" and x["gold_admission_decision"] == "allow")
    revalidate_instance = next(x for x in instances if x["policy"] == "revalidate_on_uncertainty" and x["gold_admission_decision"] == "allow")
    assert deny_instance["hard_false_rejection"] is True
    assert deny_instance["deferred_access"] is False
    assert revalidate_instance["hard_false_rejection"] is False
    assert revalidate_instance["deferred_access"] is True


def test_unknown_gold_is_unscorable_and_not_converted_to_zero():
    row = _row(gold=None, observed=_auth(None))
    instance = evaluate_rows([row])[0]
    assert instance["unsafe_exposure"] is None
    assert instance["false_rejection"] is None
    group = aggregate_policy_rows(evaluate_rows([row]))[0]
    assert group["unsafe_exposure_rate"] is None
    assert group["unsafe_exposure_scored_n"] == 0


def test_clean_fault_pair_reports_control_plane_deltas():
    clean = _row(condition_type="clean", observed=_auth("allow"), gold="allow")
    fault = _row(observed=_auth(None), gold="allow")
    groups = aggregate_policy_rows(evaluate_rows([clean, fault]))
    deny = next(x for x in groups if x["policy"] == "deny_by_default" and x["fault_type"] == "missing")
    assert deny["delta_false_rejection_vs_clean_scored_n"] == 1
    assert deny["mean_delta_false_rejection_vs_clean"] == 1


def test_no_special_effect_is_explicitly_not_applicable():
    row = _row(dimension="authorization", observed=_auth("allow"), operator="flipped", direction="unsafe")
    instance = next(x for x in evaluate_rows([row]) if x["policy"] == "conflict_quarantine")
    assert instance["action"] == "admit"
    assert instance["policy_applicable"] is True
    assert instance["intervention_applied"] is False


def test_temporal_state_exposure_is_not_reduced_to_gold_admission():
    stale = _row(dimension="temporal", observed=_temporal("expired"), operator="stale", gold="allow")
    current = _row(dimension="temporal", observed=_temporal("valid"), operator="missing", gold="allow")
    missing = _row(dimension="temporal", observed=_temporal(None), operator="missing", gold="allow")
    restrictive_expired = _row(dimension="temporal", observed=_temporal("expired"), operator="expired", gold="allow")
    assert governance_exposure(stale, "admit") == (True, "unsafe")
    assert governance_exposure(current, "admit") == (False, "safe")
    assert governance_exposure(missing, "admit") == (False, "safe")
    assert governance_exposure(restrictive_expired, "deny") == (False, "safe")
    restrictive_instance = next(x for x in evaluate_rows([restrictive_expired]) if x["policy"] == "deny_by_default")
    assert restrictive_instance["false_rejection"] is True
    assert restrictive_instance["unsafe_exposure"] is False


def test_version_validation_filters_stale_predecessor_and_preserves_current():
    row = _row(
        dimension="temporal",
        observed=_temporal("expired"),
        operator="stale",
        gold="allow",
        source_metadata={
            "current_value": {"Current_State": "current"},
            "current_session": {"session_id": 7},
            "previous_value": {"Current_State": "predecessor"},
            "previous_session": {"session_id": 6},
        },
    )
    instance = next(x for x in evaluate_rows([row]) if x["policy"] == "version_validation")
    assert instance["action"] == "admit"
    assert instance["policy_mechanism"] == "stale_version_filtered_current_preserved"
    assert instance["unsafe_exposure"] is False
    assert instance["hard_false_rejection"] is False
    assert instance["operative_state"] == "current"
    assert instance["stale_predecessor_filtered"] is True
    assert state_selection(row, "admit", "version_validation") == ("current", True)


def test_version_validation_action_does_not_depend_on_gold_labels():
    common = {
        "dimension": "temporal",
        "observed": _temporal("expired"),
        "operator": "stale",
        "source_metadata": {"current_value": {"state": "current"}, "current_session": {"session_id": 7}},
    }
    allow = _row(**common, gold="allow")
    deny = _row(**common, gold="deny")
    assert policy_action("version_validation", allow)[0] == policy_action("version_validation", deny)[0] == "admit"


def test_version_validation_does_not_reconstruct_missing_or_conflicting_state():
    missing = _row(
        dimension="temporal",
        observed=_temporal(None),
        operator="missing",
        gold="allow",
        gold_state={"temporal": {"status": "valid"}},
    )
    conflicting = _row(
        dimension="temporal",
        observed=_temporal("valid", [{"status": "valid"}, {"status": "expired"}]),
        operator="conflicting",
        gold="allow",
        source_metadata={"current_value": {"state": "current"}, "current_session": {"session_id": 7}},
    )
    assert policy_action("version_validation", missing)[0] == "deny"
    assert policy_action("version_validation", conflicting)[0] == "deny"


def test_restrictive_expired_is_not_repaired_without_stale_provenance():
    row = _row(
        dimension="temporal",
        observed=_temporal("expired"),
        operator="expired",
        gold="allow",
        source_metadata={"current_value": {"state": "current"}, "current_session": {"session_id": 7}},
    )
    instance = next(x for x in evaluate_rows([row]) if x["policy"] == "version_validation")
    assert instance["action"] == "deny"
    assert instance["hard_false_rejection"] is True


def test_memora_version_validation_does_not_use_gold_to_repair_flipped_deletion():
    row = _row(
        dimension="deletion",
        observed={"deletion": {"status": "active"}},
        operator="flipped",
        gold="deny",
        gold_state={"deletion": {"status": "deleted"}},
        source_metadata={"lifecycle": [{"operation": "delete", "session_id": 464}]},
    )
    instance = next(x for x in evaluate_rows([row]) if x["policy"] == "version_validation")
    assert instance["action"] == "admit"
    assert instance["unsafe_exposure"] is True


def test_conflicting_temporal_exposure_is_explicitly_ambiguous():
    row = _row(
        dimension="temporal",
        observed=_temporal("valid", [{"status": "valid"}, {"status": "expired"}]),
    )
    permissive = next(x for x in evaluate_rows([row]) if x["policy"] == "permissive")
    quarantined = next(x for x in evaluate_rows([row]) if x["policy"] == "conflict_quarantine")
    assert permissive["action"] == "admit"
    assert permissive["governance_exposure_status"] == "ambiguous"
    assert permissive["unsafe_exposure"] is None
    assert quarantined["action"] == "quarantine"
    assert quarantined["governance_exposure_status"] == "safe"


def test_deletion_state_exposure_is_unsafe_when_deleted_content_is_admitted():
    row = _row(
        dimension="deletion",
        observed={"deletion": {"status": "deleted"}},
        gold="deny",
        gold_state={"deletion": {"status": "deleted"}},
    )
    assert governance_exposure(row, "admit") == (True, "unsafe")


def test_pareto_requires_identical_instance_cohorts():
    rows = []
    for i in range(90):
        base = _row(example_id=f"base-{i}", observed=_auth(None), gold="deny")
        evaluated = evaluate_rows([base])
        rows.append(next(x for x in evaluated if x["policy"] == "permissive"))
        if i < 15:
            rows.append(next(x for x in evaluated if x["policy"] == "conflict_quarantine"))
    tradeoffs = _tradeoffs(rows)
    permissive = next(x for x in tradeoffs if x["policy"] == "permissive")
    quarantine = next(x for x in tradeoffs if x["policy"] == "conflict_quarantine")
    assert permissive["cohort_n"] == 90
    assert quarantine["cohort_n"] == 15
    assert quarantine["pareto_dominated_by"] == []
    assert "permissive" in quarantine["pareto_incomparable_with"]


def test_tradeoffs_never_compare_across_dataset_scopes():
    rows = []
    for dataset in ("gatemem", "memora"):
        row = _row(dataset=dataset, observed=_auth(None), gold="deny")
        rows.extend(evaluate_rows([row]))
    tradeoffs = _tradeoffs(rows)
    assert {(x["source_dataset"], x["governance_dimension"]) for x in tradeoffs} == {
        ("gatemem", "authorization"),
        ("memora", "authorization"),
    }
    assert all("__overall__" not in x["source_dataset"] for x in tradeoffs)


def test_output_order_is_deterministic_and_policy_order_is_fixed():
    row_a = _row(example_id="b", observed=_auth(None))
    row_b = _row(example_id="a", observed=_auth("allow"), condition_type="clean")
    first = evaluate_rows([row_a, row_b])
    second = evaluate_rows([deepcopy(row_a), deepcopy(row_b)])
    assert first == second
    assert [x["policy"] for x in first[: len([row_a, row_b])]] == [POLICIES[0]] * 2


def test_frozen_manifests_are_read_only_inputs():
    paths = [Path(RQ1A_MANIFEST), Path(RQ1B_MANIFEST)]
    before = [path.read_bytes() for path in paths]
    summary = evaluate()
    assert len(summary["instances"]) == 1800
    assert [path.read_bytes() for path in paths] == before
