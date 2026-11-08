from datetime import datetime

from eval.interfaces import AdaptedExample, OracleGovernanceState, Query, RawInteraction, ReconstructedGovernanceState
from eval.reconstruction import GovernanceReconstructionInput, MockGovernanceStateReconstructor
from eval.reconstruction_metrics import evaluate_rq1


def _example() -> AdaptedExample:
    interaction = RawInteraction("m1", "u1", "A governed memory", datetime(2025, 1, 1))
    query = Query(
        "q1", "u1", "What is current?", datetime(2025, 1, 2),
        gold_answer="secret answer",
        metadata={"dataset": "memconflict", "conflict_type": "dynamic"},
    )
    oracle = OracleGovernanceState(
        "m1",
        ("project:alpha",),
        False,
        datetime(2025, 1, 1),
        None,
        metadata={
            "available_fields": ("temporal_validity",),
        },
    )
    return AdaptedExample(query, (interaction,), (oracle,), evaluator_metadata={"gold": "must not pass"})


def _state(**kwargs) -> ReconstructedGovernanceState:
    return ReconstructedGovernanceState(
        interaction_id="m1",
        authorized_scopes=kwargs.get("authorized_scopes", ("project:alpha",)),
        is_deleted=kwargs.get("is_deleted", False),
        valid_from=kwargs.get("valid_from", datetime(2025, 1, 1)),
        valid_until=kwargs.get("valid_until"),
        missing_fields=kwargs.get("missing_fields", ()),
        metadata={"contextually_valid": kwargs.get("contextually_valid", True)},
    )


def test_reconstruction_input_excludes_oracle_and_query_gold():
    example = _example()
    input_data = GovernanceReconstructionInput.from_example(example)

    assert input_data.query.gold_answer is None
    assert not hasattr(input_data, "oracle_governance")
    assert "gold" not in input_data.runtime_context
    assert input_data.raw_interactions[0].content == "A governed memory"


def test_evaluator_compares_mock_after_reconstruction_and_perfect_scores():
    example = _example()
    seen = []

    class InspectingMock(MockGovernanceStateReconstructor):
        def reconstruct(self, input_data):
            seen.append(input_data)
            return super().reconstruct(input_data)

    result = evaluate_rq1([example], InspectingMock(states=(_state(),)))

    assert seen[0].query.gold_answer is None
    assert result.oracle_states_evaluated == 1
    assert result.field_metrics["temporal_validity"].accuracy == 1.0
    assert result.field_metrics["authorization_scope"].total == 0


def test_wrong_reconstruction_scores_lower():
    result = evaluate_rq1(
        [_example()],
        MockGovernanceStateReconstructor(
            states=(_state(valid_from=datetime(2026, 1, 1)),)
        ),
    )

    assert result.field_metrics["temporal_validity"].accuracy == 0.0


def test_missing_reconstructed_fields_are_counted():
    result = evaluate_rq1(
        [_example()],
        MockGovernanceStateReconstructor(
            states=(_state(missing_fields=("valid_from", "valid_until")),)
        ),
    )

    assert result.field_metrics["temporal_validity"].missing == 1
    assert result.field_metrics["authorization_scope"].total == 0
