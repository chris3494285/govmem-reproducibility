import json
from pathlib import Path

from eval.gatemem_adapter import GateMemDatasetAdapter


FIXTURE = Path(__file__).parent / "fixtures" / "gatemem-mini"


def test_gatemem_maps_authorized_checkpoint_and_native_context():
    example = GateMemDatasetAdapter().load(FIXTURE)[0]

    assert example.query.query_id == "ckpt_authorized"
    assert example.query.user_id == "employee_lee"
    assert example.query.requester_role == "employee"
    assert example.query.metadata["query_type"] == "utility"
    assert example.query.metadata["checkpoint_position"] == "t002"
    assert [item.interaction_id for item in example.interactions] == [
        "mini_office_001::t001",
        "mini_office_001::t002",
    ]
    assert example.metadata["governance_source"] == "dataset_native_gold"
    assert example.metadata["relationships"][0]["type"] == "manager_of"
    assert example.evaluator_metadata["gold"]["expected_action"] == "answer"


def test_gatemem_preserves_refusal_and_deletion_gold_without_leaking_it():
    examples = {example.query.query_id: example for example in GateMemDatasetAdapter().load(FIXTURE)}
    unauthorized = examples["ckpt_unauthorized"]
    deleted = examples["ckpt_deleted"]

    assert unauthorized.evaluator_metadata["gold"]["expected_action"] == "refuse"
    assert deleted.evaluator_metadata["gold"]["expected_action"] == "no_memory"
    assert deleted.evaluator_metadata["gold"]["attack_type"] == "post_delete_recovery"
    assert deleted.evaluator_metadata["gold"]["leak_targets"] == ["old Apollo draft"]

    for example in examples.values():
        visible = [example.query.text, *(interaction.content for interaction in example.interactions)]
        visible_text = "\n".join(visible)
        assert "expected_action" not in visible_text
        assert "leak_targets" not in visible_text
        assert "post_delete_recovery" not in visible_text


def test_gatemem_fixture_uses_public_paired_jsonl_shape():
    for filename in ("episodes.jsonl", "checkpoints.jsonl"):
        with (FIXTURE / filename).open(encoding="utf-8") as handle:
            for line in handle:
                json.loads(line)


def test_gatemem_query_types_are_preserved_for_applicability():
    examples = GateMemDatasetAdapter().load(FIXTURE)
    assert {example.query.metadata["query_type"] for example in examples} == {
        "utility", "privacy", "safety"
    }
