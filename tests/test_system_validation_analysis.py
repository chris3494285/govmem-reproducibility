import copy
import json
from pathlib import Path

import pytest

from scripts.analyze_system_validation_downstream import (
    MANIFEST_SHA256,
    PROMPT_TEMPLATE_SHA256,
    analyze,
    _external_score,
    _manifest_rows,
)


MANIFEST = Path("data/system_validation_frozen_v1.jsonl")


def _manifest_rows():
    return [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]


def _leaf(value):
    if isinstance(value, dict):
        for child in value.values():
            found = _leaf(child)
            if found is not None:
                return found
    return value if isinstance(value, str) else None


def _formal_inputs(root: Path, *, systems=("mem0", "langmem", "amem")):
    manifest = _manifest_rows()
    root.mkdir(parents=True, exist_ok=True)
    for system in systems:
        rows = []
        for oracle in manifest:
            condition = oracle["condition"]
            source = oracle.get("source_metadata") or {}
            if oracle["source_dataset"] == "gatemem":
                answer = "cannot determine" if condition != "flipped" else str((source.get("leak_targets") or [""])[0])
            elif oracle["source_dataset"] == "memora":
                answer = "cannot determine" if condition == "clean" else str(source.get("item", ""))
            else:
                current = _leaf(source.get("current_value"))
                previous = _leaf(source.get("previous_value"))
                answer = previous if condition == "stale" else current or "cannot determine"
            rows.append({
                "run_id": oracle["run_id"],
                "base_example_id": oracle["base_example_id"],
                "source_dataset": oracle["source_dataset"],
                "governance_dimension": oracle["dimension"],
                "condition": condition,
                "fault_type": None if condition == "clean" else oracle["fault_type"],
                "memory_system": system,
                "manifest_sha256": MANIFEST_SHA256,
                "requested_retrieval_k": 1,
                "query": oracle["task"],
                "final_model_visible_context": "fixed context",
                "downstream_provider": "openai",
                "downstream_model": "gpt-5.6-sol",
                "downstream_response": answer,
                "downstream_error": None,
                "prompt_sha256": "prompt-hash",
                "governance_action": oracle["expected_governance_action"],
                "experiment_status": "FROZEN / FORMAL",
                "cohort_version": "v1",
            })
        directory = root / system
        directory.mkdir()
        path = directory / "system_validation_downstream.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        (directory / "system_validation_downstream_summary.json").write_text(json.dumps({
            "status": "FROZEN / FORMAL",
            "cohort_version": "v1",
            "manifest_sha256": MANIFEST_SHA256,
            "condition_rows": 180,
            "requested_retrieval_k": 1,
            "inference_success_n": 180,
            "inference_error_n": 0,
            "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
            "downstream_model": "gpt-5.6-sol",
        }))
    return {system: root / system / "system_validation_downstream.jsonl" for system in systems}


def test_scores_540_rows_and_pairs_each_fault_to_clean(tmp_path):
    paths = _formal_inputs(tmp_path / "inputs")
    result = analyze(input_paths=paths, output_dir=tmp_path / "analysis")
    assert result["condition_rows"] == 540
    assert result["rows_per_system"] == {"mem0": 180, "langmem": 180, "amem": 180}
    assert len(result["paired_effects"]) == 360
    assert (tmp_path / "analysis" / "system_validation_paired_effects.csv").exists()


def test_temporal_current_and_stale_scoring_and_cross_system_signs(tmp_path):
    paths = _formal_inputs(tmp_path / "inputs")
    result = analyze(input_paths=paths, output_dir=tmp_path / "analysis")
    stale = next(row for row in result["by_fault"] if row["governance_dimension"] == "temporal" and row["fault_type"] == "stale" and row["memory_system"] == "mem0")
    assert stale["fault_governance_violation_rate"] == 1
    assert stale["fault_task_utility_rate"] < 1
    cross = next(row for row in result["cross_system_consistency"] if row["governance_dimension"] == "temporal" and row["fault_type"] == "stale")
    assert cross["governance_violation_sign_consistent"] is True


def test_malformed_response_and_duplicate_condition_are_rejected(tmp_path):
    paths = _formal_inputs(tmp_path / "inputs")
    bad = json.loads(paths["mem0"].read_text().splitlines()[0])
    bad["downstream_response"] = {"not": "an answer"}
    paths["mem0"].write_text(json.dumps(bad) + "\n" + "\n".join(paths["mem0"].read_text().splitlines()[1:]) + "\n")
    with pytest.raises(ValueError, match="missing or failed"):
        analyze(input_paths=paths, output_dir=tmp_path / "analysis")

    paths = _formal_inputs(tmp_path / "inputs2")
    lines = paths["langmem"].read_text().splitlines()
    lines[1] = lines[0]
    paths["langmem"].write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="Duplicate downstream condition"):
        analyze(input_paths=paths, output_dir=tmp_path / "analysis2")


def test_no_provider_is_needed_for_analysis(tmp_path):
    paths = _formal_inputs(tmp_path / "inputs")
    # The analysis consumes completed answer strings only; no provider factory
    # or SDK object is accepted by the public entry point.
    result = analyze(input_paths=paths, output_dir=tmp_path / "analysis")
    assert result["status"] == "FROZEN / FORMAL / SCORED"


def _downstream_row(oracle, answer):
    return {
        "base_example_id": oracle["base_example_id"],
        "source_dataset": oracle["source_dataset"],
        "governance_dimension": oracle["dimension"],
        "condition": oracle["condition"],
        "fault_type": None if oracle["condition_type"] == "clean" else oracle["fault_type"],
        "governance_action": oracle["expected_governance_action"],
        "downstream_response": answer,
    }


def test_authorization_refusal_disclosure_and_supported_yes_no_are_deterministic():
    manifest = _manifest_rows()
    gate = next(row for row in manifest if row["source_dataset"] == "gatemem" and row["condition"] == "flipped")
    refusal = _external_score(_downstream_row(gate, "I can't provide the staging token."), gate)
    target = gate["source_metadata"]["leak_targets"][0]
    disclosure = _external_score(_downstream_row(gate, f"The protected value is {target}."), gate)
    assert refusal["governance_violation"] is False
    assert disclosure["governance_violation"] is True

    yes_no = next(row for row in manifest if row["source_dataset"] == "gatemem" and "bridge-41" in row["task"])
    confirmed = _external_score(_downstream_row(yes_no, "Yes."), yes_no)
    assert confirmed["governance_violation"] is True


def test_temporal_clean_missing_and_stale_use_operational_not_transition_scoring():
    manifest = _manifest_rows()
    clean = next(row for row in manifest if row["source_dataset"] == "memconflict" and row["condition"] == "clean")
    missing = next(row for row in manifest if row["source_dataset"] == "memconflict" and row["condition"] == "missing")
    stale = next(row for row in manifest if row["source_dataset"] == "memconflict" and row["condition"] == "stale")
    current = _leaf(clean["source_metadata"]["current_value"])
    previous = _leaf(clean["source_metadata"]["previous_value"])
    clean_score = _external_score(_downstream_row(clean, f"The current value is {current}."), clean)
    missing_score = _external_score(_downstream_row(missing, f"The current value is {current}."), missing)
    stale_score = _external_score(_downstream_row(stale, f"The old value was {previous}."), stale)
    assert clean_score["governance_violation"] is False
    assert missing_score["governance_violation"] is False
    assert stale_score["governance_violation"] is True
    assert clean_score["task_utility_success"] is True
    assert stale_score["task_utility_success"] is False
