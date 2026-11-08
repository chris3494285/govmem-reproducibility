from dataclasses import replace
from datetime import datetime
import json

from eval.gatemem_adapter import GateMemDatasetAdapter
from eval.interfaces import OracleGovernanceState, ReconstructedGovernanceState
from eval.openai_reconstruction import model_visible_payload
from eval.reconstruction import GovernanceReconstructionInput, MockGovernanceStateReconstructor
from eval.reconstruction_applicability import (
    MEMCONFLICT_CONFLICT_TYPE_DIMENSIONS,
    MEMORA_TASK_TYPE_DIMENSIONS,
    applicable_governance_dimensions,
    dataset_native_task_type,
    is_reconstruction_applicable,
)
from eval.reconstruction_metrics import evaluate_rq1
from eval.oracle_support import structured_oracle_support
from scripts.run_reconstruction_smoke import (
    filter_examples_by_query_type,
    main as smoke_main,
    parse_args,
    reconstruct_smoke_example,
)


FIXTURE = "tests/fixtures/gatemem-mini"


def test_gatemem_applicability_is_task_native_and_dimension_specific():
    examples = {example.query.query_id: example for example in GateMemDatasetAdapter().load(FIXTURE)}
    assert applicable_governance_dimensions(examples["ckpt_authorized"]) == ()
    assert not is_reconstruction_applicable(examples["ckpt_authorized"])
    assert applicable_governance_dimensions(examples["ckpt_unauthorized"]) == ("authorization_scope",)
    assert is_reconstruction_applicable(examples["ckpt_unauthorized"], "authorization_scope")
    assert applicable_governance_dimensions(examples["ckpt_deleted"]) == ("lifecycle_deletion",)
    assert is_reconstruction_applicable(examples["ckpt_deleted"], "lifecycle_deletion")


def test_non_applicable_utility_checkpoint_is_excluded_from_rq1_metrics():
    utility = GateMemDatasetAdapter().load(FIXTURE)[0]

    class MustNotReconstruct:
        def reconstruct(self, input_data):
            raise AssertionError("utility checkpoint should not be reconstructed for RQ1")

    result = evaluate_rq1([utility], MustNotReconstruct())
    assert result.applicable_examples_evaluated == 0
    assert result.oracle_states_evaluated == 0
    assert all(metric.total == 0 for metric in result.field_metrics.values())
    assert result.reconstructions[utility.query.query_id] == ()


def test_applicable_gate_mem_checkpoint_remains_in_rq1():
    examples = {example.query.query_id: example for example in GateMemDatasetAdapter().load(FIXTURE)}
    privacy = examples["ckpt_unauthorized"]
    interaction_id = privacy.interactions[0].interaction_id
    privacy = replace(privacy, oracle_governance=(OracleGovernanceState(
        interaction_id=interaction_id,
        authorized_scopes=("project:apollo",),
        is_deleted=False,
        valid_from=None,
        valid_until=None,
        metadata={"available_fields": ("authorization_scope",)},
    ),))
    reconstructed = ReconstructedGovernanceState(
        interaction_id=interaction_id,
        authorized_scopes=("project:apollo",),
        is_deleted=False,
        valid_from=None,
        valid_until=None,
    )
    result = evaluate_rq1([privacy], MockGovernanceStateReconstructor(states=(reconstructed,)))
    assert result.applicable_examples_evaluated == 1
    assert result.field_metrics["authorization_scope"].total == 1
    assert result.field_metrics["authorization_scope"].accuracy == 1.0
    assert result.field_metrics["lifecycle_deletion"].total == 0


def test_utility_task_metadata_and_gold_stay_out_of_model_input():
    utility = GateMemDatasetAdapter().load(FIXTURE)[0]
    payload = model_visible_payload(GovernanceReconstructionInput.from_example(utility))
    visible = json.dumps(payload)
    assert "query_type" not in visible
    assert "expected_action" not in visible
    assert "judge_spec" not in visible
    assert "leak_targets" not in visible
    assert "Friday" in visible


def test_reconstruction_smoke_query_type_filter_is_available():
    args = parse_args(["--dataset", "gatemem", "--input", FIXTURE, "--query-type", "privacy"])
    assert args.query_type == "privacy"


def test_smoke_filter_matches_gate_mem_privacy():
    examples = GateMemDatasetAdapter().load(FIXTURE)
    assert [e.query.query_id for e in filter_examples_by_query_type(examples, "privacy")] == ["ckpt_unauthorized"]
    assert dataset_native_task_type(examples[1]) == "privacy"


def test_smoke_filter_matches_all_memconflict_native_types():
    from eval.memconflict_adapter import MemConflictDatasetAdapter

    examples = MemConflictDatasetAdapter().load("tests/fixtures/memconflict-mini")
    for task_type in ("dynamic", "static", "conditional"):
        matched = filter_examples_by_query_type(examples, task_type)
        assert len(matched) == 1
        assert dataset_native_task_type(matched[0]) == task_type


def test_smoke_filter_no_match_fails_clearly(capsys):
    result = smoke_main([
        "--dataset", "memconflict",
        "--input", "tests/fixtures/memconflict-mini",
        "--query-type", "missing",
        "--limit", "1",
    ])
    captured = capsys.readouterr()
    assert result != 0
    assert "No memconflict examples matched" in captured.err
    assert "missing" in captured.err


def test_non_applicable_smoke_example_skips_reconstructor():
    utility = GateMemDatasetAdapter().load(FIXTURE)[0]

    class CountingReconstructor:
        calls = 0

        def reconstruct(self, input_data):
            self.calls += 1
            return ()

    reconstructor = CountingReconstructor()
    result = reconstruct_smoke_example(utility, reconstructor)
    assert reconstructor.calls == 0
    assert result["states"] == []
    assert result["skipped"] is True
    assert result["skipped_reason"] == "no_applicable_governance_dimensions"


def test_applicable_smoke_example_calls_reconstructor():
    privacy = {example.query.query_id: example for example in GateMemDatasetAdapter().load(FIXTURE)}["ckpt_unauthorized"]

    class CountingReconstructor:
        calls = 0

        def reconstruct(self, input_data):
            self.calls += 1
            return ()

    reconstructor = CountingReconstructor()
    result = reconstruct_smoke_example(privacy, reconstructor)
    assert reconstructor.calls == 1
    assert result["skipped"] is False
    assert result["states"] == []


def test_other_dataset_applicability_is_conservative():
    from eval.memconflict_adapter import MemConflictDatasetAdapter
    from eval.memora_adapter import MemoraDatasetAdapter

    conflict_examples = {
        example.query.metadata["conflict_type"]: example
        for example in MemConflictDatasetAdapter().load("tests/fixtures/memconflict-mini")
    }
    assert applicable_governance_dimensions(conflict_examples["dynamic"]) == ("temporal_validity",)
    assert applicable_governance_dimensions(conflict_examples["conditional"]) == ("contextual_validity",)
    assert applicable_governance_dimensions(conflict_examples["static"]) == ()
    assert MEMCONFLICT_CONFLICT_TYPE_DIMENSIONS["static"] == ()

    memora = MemoraDatasetAdapter().load("tests/fixtures/memora-mini")[0]
    assert applicable_governance_dimensions(memora) == ("lifecycle_deletion",)
    assert MEMORA_TASK_TYPE_DIMENSIONS["remembering"] == ("lifecycle_deletion",)

    unknown = replace(memora, query=replace(memora.query, metadata={"dataset": "unknown", "task_type": "remembering"}))
    assert applicable_governance_dimensions(unknown) == ()


def test_structured_oracle_support_is_explicit_and_does_not_promote_downstream_gold():
    examples = {example.query.query_id: example for example in GateMemDatasetAdapter().load(FIXTURE)}
    privacy_support = structured_oracle_support(examples["ckpt_unauthorized"])
    safety_support = structured_oracle_support(examples["ckpt_deleted"])
    assert privacy_support["oracle_available"] is False
    assert safety_support["oracle_available"] is False
    assert privacy_support["oracle_fields_supported"] == ()
    assert "expected_action" in privacy_support["native_gold_fields"]
    assert "expected_action" not in privacy_support["oracle_fields_supported"]

    from eval.memconflict_adapter import MemConflictDatasetAdapter
    conflict_examples = MemConflictDatasetAdapter().load("tests/fixtures/memconflict-mini")
    by_type = {example.query.metadata["conflict_type"]: example for example in conflict_examples}
    assert structured_oracle_support(by_type["dynamic"])["oracle_available"] is False
    assert structured_oracle_support(by_type["conditional"])["oracle_available"] is False

    from eval.memora_adapter import MemoraDatasetAdapter
    memora = MemoraDatasetAdapter().load("tests/fixtures/memora-mini")[0]
    memora_support = structured_oracle_support(memora)
    assert memora_support["oracle_available"] is False
    assert memora_support["oracle_fields_supported"] == ()
    assert "forgetting_evidence" in memora_support["native_gold_fields"]


def test_native_answer_and_evidence_gold_does_not_create_rq1_state_agreement():
    from eval.rq1_metrics import evaluate_records

    summary = evaluate_records([{
        "query_id": "memora-remembering",
        "dataset": "memora",
        "task_type": "remembering",
        "applicable_governance_dimensions": ["lifecycle_deletion"],
        "skipped": False,
        "reconstruction_status": "success",
        "validation_status": "valid",
        "reconstructed_states": [{
            "interaction_id": "i1",
            "is_deleted": True,
            "missing_fields": [],
        }],
        "evaluator_only": {
            "oracle_available": False,
            "oracle_fields_supported": [],
            "oracle_governance": [],
            "evaluator_metadata": {
                "gold": {
                    "forgetting_evidence": {"forgotten_items": [{"value": "x", "session_id": 2}]},
                    "evaluation": {"expected_answer": "no"},
                },
            },
        },
    }])
    metric = summary["native_supported_state_agreement"]["is_deleted"]
    assert metric == {"n": 0, "correct": 0, "missing_prediction": 0, "available": False, "accuracy": None}
