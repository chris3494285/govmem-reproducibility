from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.analyze_rq3_error_propagation import _pair_record, analyze


def test_rq3_script_supports_direct_and_module_invocation_from_repo_root():
    root = Path(__file__).resolve().parents[1]
    for command in (
        [sys.executable, "scripts/analyze_rq3_error_propagation.py", "--help"],
        [sys.executable, "-m", "scripts.analyze_rq3_error_propagation", "--help"],
    ):
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        assert "Analyze deterministic RQ1 fault propagation" in completed.stdout


def _manifest_row(base, condition, *, operator=None, direction=None, version=None):
    current = {"Current_State": "CURRENT"}
    previous = {"Current_State": "PREVIOUS"}
    return {
        "run_id": f"{base}::{condition}::{operator or 'clean'}",
        "example_id": base,
        "source_dataset": "memconflict",
        "dimension": "temporal",
        "condition_type": condition,
        "fault_spec": None if condition == "clean" else {"dimension": "temporal", "operator": operator, "direction": direction},
        "manifest_version": version,
        "task": "Did the value change?",
        "memory": [{"interaction_id": base, "content": "memory"}],
        "gold_governance_state": {"temporal": {"status": "valid"}},
        "observed_governance_state": {"temporal": {"status": "valid"}} if condition == "clean" else {"temporal": {"status": "expired"}},
        "gold_admission_decision": "allow",
        "utility_target": None,
        "source_metadata": {"current_value": current, "previous_value": previous, "attribute": "State"},
    }


def _inference(manifest_row, *, admission, context, selected):
    return {
        "run_id": manifest_row["run_id"], "example_id": manifest_row["example_id"],
        "condition_type": manifest_row["condition_type"], "source_dataset": "memconflict", "dimension": "temporal",
        "actual_admission": admission, "constructed_context": context, "selected_memory_ids": selected,
        "rejected_memory_ids": [], "status": "success",
    }


def _scored(inference, *, gvr=None, utility=None, target=None, denial=None):
    return {**inference, "governance_violation": gvr, "task_utility": utility, "target_utility": target,
            "mentions_required_target": target, "downstream_false_denial": denial}


def test_rq1a_downstream_failure_is_causal_gvr_delta():
    manifest = {"source_dataset": "memconflict", "dimension": "temporal", "fault_spec": {"operator": "stale", "direction": "unsafe"}, "source_metadata": {}}
    clean = {"actual_admission": "allow", "selected_memory_ids": [], "constructed_context": "same"}
    fault = {"actual_admission": "allow", "selected_memory_ids": [], "constructed_context": "same"}
    cases = ((False, True, True), (True, True, False), (False, False, False), (None, True, None))
    for clean_gvr, fault_gvr, expected in cases:
        clean_scored = {"governance_violation": clean_gvr}
        fault_scored = {"governance_violation": fault_gvr}
        record = _pair_record("deepseek", "rq1a", manifest, clean, fault, clean_scored, fault_scored)
        assert record["downstream_failure"] is expected
        if record["context_changed"] is False and record["governance_violation_changed"] is False:
            assert record["downstream_failure"] is not True


def _write_case(tmp_path, provider):
    root = tmp_path / provider
    root.mkdir()
    a_manifest = [
        _manifest_row("a1", "clean"),
        _manifest_row("a1", "fault", operator="stale", direction="unsafe"),
        _manifest_row("a1", "fault", operator="missing", direction="ambiguous"),
        _manifest_row("a2", "clean"),
        _manifest_row("a2", "fault", operator="stale", direction="unsafe"),
        _manifest_row("a2", "fault", operator="missing", direction="ambiguous"),
    ]
    b_manifest = [
        _manifest_row("b1", "clean", version="governance-fault-rq1-restrictive-v1"),
        _manifest_row("b1", "fault", operator="expired", direction="restrictive", version="governance-fault-rq1-restrictive-v1"),
        _manifest_row("b2", "clean", version="governance-fault-rq1-restrictive-v1"),
        _manifest_row("b2", "fault", operator="expired", direction="restrictive", version="governance-fault-rq1-restrictive-v1"),
    ]
    def make_rows(manifest, track):
        inference, scored = [], []
        for row in manifest:
            fault = row.get("fault_spec") or {}
            if track == "a" and fault.get("operator") == "stale":
                admission, context, selected = "deny", "PREVIOUS", []
                score = {"gvr": True, "utility": None, "target": None, "denial": None}
            elif track == "a":
                admission, context, selected = "allow", "CURRENT", ["m"]
                score = {"gvr": False, "utility": None, "target": None, "denial": None}
            elif row["condition_type"] == "clean":
                admission, context, selected = "allow", "CURRENT TARGET", ["m"]
                score = {"gvr": None, "utility": True, "target": row["example_id"] == "b1", "denial": None}
            else:
                admission, context, selected = "deny", "NO TARGET", []
                score = {"gvr": None, "utility": False, "target": False, "denial": row["example_id"] == "b1"}
            inf = _inference(row, admission=admission, context=context, selected=selected)
            inference.append(inf)
            scored.append(_scored(inf, **score))
        return inference, scored
    for name, manifest in (("rq1a", a_manifest), ("rq1b_restrictive", b_manifest)):
        inference, scored = make_rows(manifest, "a" if name == "rq1a" else "b")
        (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in inference), encoding="utf-8")
        (root / f"{name}_scored.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scored), encoding="utf-8")
    return root


def test_fixture_setup_and_rq3_analysis(tmp_path):
    # Write manifests separately so the normal frozen-manifest validator is
    # exercised without using any provider or API.
    for provider in ("deepseek", "openai", "anthropic"):
        _write_case(tmp_path, provider)
    a_rows = [_manifest_row("a1", "clean"), _manifest_row("a1", "fault", operator="stale", direction="unsafe"), _manifest_row("a1", "fault", operator="missing", direction="ambiguous"), _manifest_row("a2", "clean"), _manifest_row("a2", "fault", operator="stale", direction="unsafe"), _manifest_row("a2", "fault", operator="missing", direction="ambiguous")]
    b_rows = [_manifest_row("b1", "clean", version="governance-fault-rq1-restrictive-v1"), _manifest_row("b1", "fault", operator="expired", direction="restrictive", version="governance-fault-rq1-restrictive-v1"), _manifest_row("b2", "clean", version="governance-fault-rq1-restrictive-v1"), _manifest_row("b2", "fault", operator="expired", direction="restrictive", version="governance-fault-rq1-restrictive-v1")]
    a_manifest = tmp_path / "rq1a_manifest.jsonl"; b_manifest = tmp_path / "rq1b_manifest.jsonl"
    a_manifest.write_text("".join(json.dumps(row) + "\n" for row in a_rows), encoding="utf-8")
    b_manifest.write_text("".join(json.dumps(row) + "\n" for row in b_rows), encoding="utf-8")
    summary = analyze(input_root=tmp_path, rq1a_manifest=a_manifest, rq1b_manifest=b_manifest)
    assert len(summary["paired_records"]) == 18
    stale = next(row for row in summary["paired_records"] if row["fault_type"] == "stale" and row["provider"] == "deepseek")
    assert stale["admission_changed"] is True
    assert stale["context_changed"] is True
    assert "stale_predecessor_promoted_or_exposed" in stale["context_mechanisms"]
    assert stale["downstream_failure"] is True
    assert stale["full_propagation"] is True
    missing = next(row for row in summary["paired_records"] if row["fault_type"] == "missing" and row["provider"] == "deepseek")
    assert missing["admission_changed"] is False
    assert missing["context_changed"] is False
    assert missing["downstream_failure"] is False
    assert missing["full_propagation"] is False
    missing_group = next(row for row in summary["by_provider_fault"] if row["fault_type"] == "missing" and row["provider"] == "deepseek")
    assert missing_group["admission_change_rate"] == 0
    assert missing_group["context_change_rate"] == 0
    assert missing_group["downstream_failure_rate"] == 0
    assert missing_group["full_propagation_rate"] == 0
    restrictive = [row for row in summary["paired_records"] if row["track"] == "rq1b" and row["provider"] == "deepseek"]
    assert any(row["propagation_stop"] == "clean_failed" for row in restrictive)
    assert any(row["downstream_failure"] is True for row in restrictive)
    assert any(row["downstream_failure"] is None for row in restrictive)
    aggregates = summary["by_provider_fault"]
    assert all(row["provider"] in {"deepseek", "openai", "anthropic"} for row in aggregates)
    assert len(summary["cross_provider_stability"]) == 3
