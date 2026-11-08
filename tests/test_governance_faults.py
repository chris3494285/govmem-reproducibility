import pytest

from govmem.governance.faults import FaultInjectionError, fault_spec_for, inject_fault
from govmem.governance.schema import AuthorizationState, DeletionState, GovernanceState, TemporalState


def test_authorization_faults_and_immutability():
    gold = GovernanceState(authorization=AuthorizationState("allow", "scope"), deletion=DeletionState("active"))
    missing = inject_fault(gold, "authorization", "missing")
    flipped = inject_fault(gold, "authorization", "flipped")
    conflict = inject_fault(gold, "authorization", "conflicting")
    assert missing.authorization is None
    assert flipped.authorization.decision == "deny"
    assert {item.decision for item in conflict.authorization_assertions} == {"allow", "deny"}
    assert gold.authorization.decision == "allow"
    assert fault_spec_for(gold, "authorization", "flipped").direction == "restrictive"


def test_authorization_deny_flip_is_unsafe():
    gold = GovernanceState(authorization=AuthorizationState("deny"))
    assert inject_fault(gold, "authorization", "flipped").authorization.decision == "allow"
    assert fault_spec_for(gold, "authorization", "flipped").direction == "unsafe"


def test_deletion_faults_and_real_predecessor():
    predecessor = GovernanceState(deletion=DeletionState("active", 3))
    gold = GovernanceState(deletion=DeletionState("deleted", 4), temporal=TemporalState("valid"))
    assert inject_fault(gold, "deletion", "missing").deletion is None
    assert inject_fault(gold, "deletion", "stale", [predecessor]).deletion == predecessor.deletion
    assert inject_fault(gold, "deletion", "flipped").deletion.status == "active"
    assert inject_fault(predecessor, "deletion", "flipped").deletion.status == "deleted"
    with pytest.raises(FaultInjectionError):
        inject_fault(gold, "deletion", "stale")


def test_temporal_faults_require_predecessor_for_stale_and_conflict():
    previous = GovernanceState(temporal=TemporalState("expired", version=1))
    gold = GovernanceState(temporal=TemporalState("valid", version=2))
    assert inject_fault(gold, "temporal", "missing").temporal is None
    assert inject_fault(gold, "temporal", "stale", [previous]).temporal.status == "expired"
    conflict = inject_fault(gold, "temporal", "conflicting", [previous])
    assert {item.status for item in conflict.temporal_assertions} == {"valid", "expired"}
    with pytest.raises(FaultInjectionError):
        inject_fault(gold, "temporal", "flipped")


def test_invalid_stale_and_unsupported_pairs_are_rejected():
    with pytest.raises(FaultInjectionError):
        inject_fault(GovernanceState(authorization=AuthorizationState("allow")), "authorization", "stale")
