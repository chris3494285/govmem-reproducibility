from dataclasses import replace

import pytest

from govmem.governance.faults import fault_spec_for, inject_fault
from govmem.governance.schema import AuthorizationState, BaseGovernanceExample, GovernanceState
from govmem.governance.validation import validate_condition, validate_manifest
from scripts.build_governance_fault_rq1_manifest import build_conditions


def _base(identifier: str) -> BaseGovernanceExample:
    state = GovernanceState(authorization=AuthorizationState("allow", "scope"))
    return BaseGovernanceExample(identifier, "gatemem", identifier, "authorization", "query", ({"interaction_id": "i1", "content": "memory"},), state, "allow", source_metadata={"source": identifier})


def test_manifest_build_is_deterministic_and_pairs_are_traceable():
    bases = [_base(f"x{i:02d}") for i in range(5)]
    first, info1 = build_conditions(bases, seed=42)
    second, info2 = build_conditions(bases, seed=42)
    assert [row.run_id for row in first] == [row.run_id for row in second]
    assert info1 == info2
    assert len(first) == 15
    assert all(sum(row.example_id == example_id for row in first) == 3 for example_id in {row.example_id for row in first})
    assert all(row.example_id and row.source_metadata for row in first)
    validate_manifest(first)


def test_manifest_rejects_duplicate_run_ids_and_content_change():
    rows, _ = build_conditions([_base("x")], seed=42)
    with pytest.raises(ValueError, match="Duplicate run_id"):
        validate_manifest(rows + [rows[0]])
    altered = replace(rows[1], task="different")
    with pytest.raises(ValueError, match="Content changed"):
        validate_manifest([rows[0], altered] + rows[2:])


def test_fault_row_cannot_change_non_target_state():
    rows, _ = build_conditions([_base("x")], seed=42)
    bad = replace(rows[1], gold_governance_state=GovernanceState(authorization=AuthorizationState("allow", "other")))
    with pytest.raises(ValueError, match="Gold changed"):
        validate_manifest([rows[0], bad] + rows[2:])


def test_manifest_rejects_unsupported_operator_dimension_pair():
    rows, _ = build_conditions([_base("x")], seed=42)
    bad_spec = fault_spec_for(rows[0].gold_governance_state, "authorization", "flipped")
    bad = replace(rows[1], fault_spec=replace(bad_spec, operator="stale"))
    with pytest.raises(ValueError, match="Unsupported operator/dimension pair"):
        validate_condition(bad)
