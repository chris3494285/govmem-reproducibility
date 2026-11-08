import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_analysis_spec_is_frozen_and_uses_required_aggregates():
    spec = json.loads((ROOT / "configs/main_analysis_spec_v1.json").read_text())
    assert spec["status"] == "FROZEN"
    assert spec["deterministic_seed"] == 42
    assert spec["frozen_base_manifest_sha256"] == "a2601a7ecb46034faf5dcc7ad00b5a851e19ae01561876fa681fd18af168ca79"
    assert "event-macro" in spec["primary_aggregate_definition"]
    assert "micro" in spec["secondary_aggregate_definition"]
    assert spec["per_event_reporting"] == ["deletion", "temporal", "update"]
    assert spec["balanced_event_robustness"]["counts"] == {"deletion": 169, "temporal": 169, "update": 169}
    assert spec["balanced_event_robustness"]["total"] == 507
    assert spec["oaks_cluster_robustness"]["cluster_field"] == "document_id"
    assert spec["oaks_cluster_robustness"]["sensitivity"] == "leave-one-document-out"
    assert "frozen before model inference" in spec["analysis_policy_statement"]
