from __future__ import annotations

import copy
import hashlib
import json

from scripts.build_corrected_rq1a_v2 import V2_VERSION, validate_strict_diff, write_jsonl


V1 = "data/governance_fault_rq1_pilot_frozen.jsonl"
V2 = "data/governance_fault_rq1_pilot_frozen_v2.jsonl"
CORRECTIONS = "data/governance_fault_rq1_pilot_v2_corrections.json"


def _rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def test_v2_preserves_cohort_and_corrects_exactly_thirteen_bases():
    old = _rows(V1)
    new = _rows(V2)
    corrections = json.load(open(CORRECTIONS, encoding="utf-8"))
    assert len(old) == len(new) == 300
    assert len({row["example_id"] for row in old}) == len({row["example_id"] for row in new}) == 100
    assert len(corrections["corrections"]) == 13
    assert all(row["manifest_version"] == V2_VERSION for row in new)
    assert {row["run_id"] for row in old} == {row["run_id"] for row in new}
    assert all((row.get("fault_spec") or {}).get("direction") != "restrictive" for row in new)


def test_v2_only_changes_attribute_binding_and_manifest_version():
    old = _rows(V1)
    new = _rows(V2)
    affected = {item["example_id"] for item in json.load(open(CORRECTIONS, encoding="utf-8"))["corrections"]}
    validate_strict_diff(old, new, affected)
    for before, after in zip(old, new):
        if before["example_id"] not in affected:
            after_without_version = dict(after)
            after_without_version.pop("manifest_version", None)
            assert before == after_without_version
        else:
            assert before["source_metadata"]["current_value"] == after["source_metadata"]["current_value"]
            assert before["source_metadata"]["previous_value"] == after["source_metadata"]["previous_value"]
            assert before["task"] == after["task"]
            assert before["gold_governance_state"] == after["gold_governance_state"]
            assert before["observed_governance_state"] == after["observed_governance_state"]


def test_v2_generation_bytes_are_stable(tmp_path):
    rows = _rows(V2)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    h1 = write_jsonl(copy.deepcopy(rows), first)
    h2 = write_jsonl(copy.deepcopy(rows), second)
    assert h1 == h2
    assert first.read_bytes() == second.read_bytes()
