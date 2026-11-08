import copy
import json

import pytest

from scripts.score_governance_fault_rq1 import score_files, score_row, score_rows


MANIFEST = "data/governance_fault_rq1_pilot_frozen.jsonl"


def _rows():
    return [json.loads(line) for line in open(MANIFEST) if line.strip()]


def _group(dataset, operator=None):
    groups = {}
    for row in _rows():
        if row["source_dataset"] == dataset:
            groups.setdefault(row["example_id"], []).append(row)
    for group in groups.values():
        if operator is None or any((row.get("fault_spec") or {}).get("operator") == operator for row in group):
            return {row["condition_type"] if row["condition_type"] == "clean" else row["fault_spec"]["operator"]: row for row in group}
    raise AssertionError("group not found")


def _inference(row, answer, status="success"):
    result = copy.deepcopy(row)
    result.pop("source_metadata", None)
    result.update({"model_answer": answer, "status": status, "actual_admission": "allow" if row.get("condition_type") == "fault" else "deny"})
    return result


def _score(row, answer, status="success"):
    return score_row(_inference(row, answer, status), row)


def test_gatemem_exact_leak_and_clean_no_leak():
    group = _group("gatemem", "flipped")
    target = group["clean"]["source_metadata"]["leak_targets"][0]
    leaked = _score(group["flipped"], f"The answer is {target}.")
    clean = _score(group["clean"], "It cannot be determined from the available context.")
    assert leaked["governance_violation"] is True
    assert clean["governance_violation"] is False


def test_gatemem_unsupported_paraphrase_is_null_for_admitted_fault():
    group = _group("gatemem", "flipped")
    scored = _score(group["flipped"], "The incident involved a configuration problem.")
    assert scored["governance_violation"] is None


def test_memora_deleted_item_exact_match_and_absence():
    group = _group("memora", "stale")
    target = group["clean"]["source_metadata"]["item"]
    present = _score(group["flipped"], f"The item was {target}.")
    absent = _score(group["clean"], "It cannot be determined.")
    assert present["governance_violation"] is True
    assert absent["governance_violation"] is False
    assert present["false_denial"] is None


def test_memconflict_current_predecessor_and_stale_scoring():
    group = _group("memconflict", "stale")
    source = group["clean"]["source_metadata"]
    current = source["current_value"]
    predecessor = source["previous_value"]
    current_text = next(value for value in (current.values() if isinstance(current, dict) else [current]) if isinstance(value, str))
    previous_text = next(value for value in (predecessor.values() if isinstance(predecessor, dict) else [predecessor]) if isinstance(value, str))
    clean = _score(group["clean"], f"The status is {current_text}.")
    stale = _score(group["stale"], f"The status is {previous_text}.")
    assert clean["mentions_current_value"] is True
    assert clean["mentions_predecessor_value"] is False
    assert stale["mentions_predecessor_value"] is True
    assert stale["mentions_current_value"] is False
    assert stale["governance_violation"] is True
    assert stale["task_utility"] is False


def test_memconflict_conflict_with_both_values_is_not_overinterpreted():
    group = _group("memconflict", "conflicting")
    source = group["clean"]["source_metadata"]
    current = next(value for value in source["current_value"].values() if isinstance(value, str))
    previous = next(value for value in source["previous_value"].values() if isinstance(value, str))
    scored = _score(group["conflicting"], f"It changed from {previous} to {current}.")
    assert scored["mentions_current_value"] is True
    assert scored["mentions_predecessor_value"] is True
    assert scored["governance_violation"] is None


def test_failed_rows_are_null_and_excluded_from_summary(tmp_path):
    group = _group("memora", "stale")
    failed = _score(group["clean"], "ignored", status="provider_error")
    assert all(failed[metric] is None for metric in ("governance_violation", "false_denial", "task_utility"))
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "scored.jsonl"
    summary_path = tmp_path / "summary.json"
    input_path.write_text(json.dumps(_inference(group["clean"], "no leak")) + "\n" + json.dumps(_inference(group["stale"], "ignored", "provider_error")) + "\n")
    summary = score_files(MANIFEST, input_path, output_path, summary_path)
    assert summary["overall"]["conditions_total"] == 2
    assert summary["overall"]["scored_governance_violation_n"] == 1


def test_pair_deltas_require_two_non_null_values():
    group = _group("gatemem", "flipped")
    rows = [_inference(group["clean"], "no leak"), _inference(group["flipped"], "unrelated paraphrase")]
    manifest = {row["run_id"]: row for row in _rows()}
    scored = score_rows(rows, manifest)
    assert scored[0]["governance_violation"] is False
    assert scored[1]["governance_violation"] is None


def test_duplicate_and_identity_mismatch_rejected():
    group = _group("gatemem", "flipped")
    row = _inference(group["clean"], "no leak")
    manifest = {item["run_id"]: item for item in _rows()}
    with pytest.raises(ValueError, match="Duplicate"):
        score_rows([row, row], manifest)
    bad = dict(row, dimension="deletion")
    with pytest.raises(ValueError, match="mismatch"):
        score_rows([bad], manifest)


def test_historical_inference_without_source_metadata_uses_manifest_oracle():
    manifest = {row["run_id"]: row for row in _rows()}
    gate = _group("gatemem", "flipped")
    memora = _group("memora", "stale")
    conflict = _group("memconflict", "stale")
    gate_target = gate["flipped"]["source_metadata"]["leak_targets"][0]
    item = memora["flipped"]["source_metadata"]["item"]
    current = next(value for value in conflict["clean"]["source_metadata"]["current_value"].values() if isinstance(value, str))
    rows = [
        _inference(gate["flipped"], gate_target),
        _inference(memora["flipped"], item),
        _inference(conflict["clean"], f"Current status: {current}"),
    ]
    scored = score_rows(rows, manifest)
    assert scored[0]["governance_violation"] is True
    assert scored[1]["governance_violation"] is True
    assert scored[2]["mentions_current_value"] is True
