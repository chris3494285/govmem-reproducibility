import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_failure_policy_is_frozen_and_pre_scoring():
    spec = json.loads((ROOT / "configs/main_failure_handling_spec_v1.json").read_text())
    assert spec["status"] == "FROZEN_PRE_SCORING"
    assert spec["schema_version"] == "main-failure-handling-spec-v1"
    assert "score=0" in spec["primary_failure_policy"]
    assert "retain all attempted" in spec["primary_denominator_policy"]
    assert "complete-case" in spec["sensitivity_analysis"]


def test_analysis_spec_is_not_modified_by_failure_policy():
    spec = json.loads((ROOT / "configs/main_analysis_spec_v1.json").read_text())
    assert spec["schema_version"] == "main-analysis-spec-v1"
    assert "event-macro" in spec["primary_aggregate_definition"]
