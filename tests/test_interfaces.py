from datetime import datetime

from eval.interfaces import (
    AdaptedExample,
    EvaluationResult,
    GovernanceDecision,
    OracleGovernanceState,
    Query,
    RawInteraction,
    ReconstructedGovernanceState,
)


def test_shared_schemas_are_frozen_and_preserve_typed_fields():
    interaction = RawInteraction("i1", "u1", "hello", datetime(2025, 1, 1))
    query = Query("q1", "u1", "What?", datetime(2025, 1, 2), ("default",), "answer")
    oracle = OracleGovernanceState("i1", ("default",), False, interaction.timestamp, None)
    reconstructed = ReconstructedGovernanceState(
        "i1", ("default",), False, interaction.timestamp, None, confidence=0.8, provenance="test"
    )
    decision = GovernanceDecision("i1", True, True, True, True, reasons=("valid",))
    result = EvaluationResult("q1", "answer", "answer", True, ("i1",), (decision,))
    example = AdaptedExample(query, (interaction,), (oracle,))

    assert example.query == query
    assert result.governance_decisions == (decision,)
    assert reconstructed.provenance == "test"
    try:
        query.text = "changed"
    except AttributeError:
        pass
    else:
        raise AssertionError("shared schemas should be frozen")
