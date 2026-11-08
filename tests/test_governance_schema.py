from govmem.governance.schema import (
    AuthorizationState, DeletionState, FaultSpec, GovernanceState, TemporalState,
)


def test_governance_state_dimensions_are_optional_and_typed():
    state = GovernanceState(
        authorization=AuthorizationState("allow", "project-a", 1),
        deletion=DeletionState("active", 2),
        temporal=TemporalState("valid", version=3),
    )
    assert state.authorization.decision == "allow"
    assert state.deletion.status == "active"
    assert state.temporal.status == "valid"
    assert FaultSpec("authorization", "missing", "ambiguous").dimension == "authorization"
