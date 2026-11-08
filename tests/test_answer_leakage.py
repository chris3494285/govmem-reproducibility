import pytest
from pathlib import Path

from scripts.audit_answer_leakage import audit, validate_no_unreported_oracle_substitution

ROOT = Path(__file__).resolve().parents[1]


def test_formal_manifest_leakage_is_machine_readable_and_reported():
    result = audit(ROOT / "data/final_main_governance_condition_manifest_v2.jsonl")
    assert result["status"] == "FAIL"
    assert result["submission_ready"] is False
    assert result["total_condition_rows"] == 4891
    assert result["affected_condition_rows"] == 159
    assert result["affected_base_count"] == 53
    assert result["affected_rows_by_condition_type"] == {"clean": 53, "enforced": 106}
    assert result["affected_rows_by_source"] == {"longmemeval": 159}


def test_corrected_v3_manifest_has_no_leakage_or_unresolved_repair_provenance():
    result = audit(ROOT / "data/final_main_governance_condition_manifest_v3.jsonl")
    assert result["status"] == "PASS"
    assert result["submission_ready"] is False
    assert result["affected_condition_rows"] == 0
    assert result["unresolved_source_provenance_condition_ids"] == []


def test_release_guard_fails_on_oracle_substitution():
    with pytest.raises(RuntimeError, match="direct oracle-answer substitution"):
        validate_no_unreported_oracle_substitution({"affected_condition_rows": 1, "affected_base_count": 1})


def test_release_guard_passes_for_native_evidence_only():
    validate_no_unreported_oracle_substitution({"affected_condition_rows": 0, "affected_base_count": 0})


def test_release_guard_rejects_direct_oracle_provenance():
    result = audit(ROOT / "data/final_main_governance_condition_manifest_v3.jsonl")
    row = {
        "condition_id": "synthetic",
        "source_dataset": "longmemeval",
        "model_visible_context_spec": {
            "content_source": "expected_answer",
            "oracle_answer_substitution": True,
        },
        "source_provenance": {"oracle_answer_substitution": True},
    }
    # The public audit API must fail closed on the same classes of metadata
    # that the v3 builder is forbidden to emit.
    from scripts.audit_answer_leakage import _provenance_problems

    assert _provenance_problems([row], repaired=True) == ["synthetic"]
    result["unresolved_source_provenance_condition_ids"] = ["synthetic"]
    with pytest.raises(RuntimeError):
        validate_no_unreported_oracle_substitution(result)
