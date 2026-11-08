from __future__ import annotations

import hashlib
import json

import pytest

from scripts.build_rq1_gpt56_smoke import (
    FROZEN_FINAL_SHA256,
    SMOKE_MANIFEST_VERSION,
    build_smoke_manifest,
    validate_smoke_manifest,
    write_smoke_manifest,
)
from scripts.summarize_rq1_pilot import summarize


def _parent(tmp_path):
    records = []
    for dataset, task in (("memora", "remembering"), ("memconflict", "dynamic"), ("memconflict", "conditional")):
        for index in range(2):
            records.append({"dataset": dataset, "task_type": task, "example_id": f"{dataset}-{task}-{index}", "query_id": f"{dataset}-{task}-{index}"})
    parent = {"manifest_version": "rq1-final-90-v1", "selected_n": 90, "records": records}
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(parent), encoding="utf-8")
    return path


def test_smoke_selection_is_deterministic_and_stratified(tmp_path, monkeypatch):
    parent = _parent(tmp_path)
    monkeypatch.setattr("scripts.build_rq1_gpt56_smoke.FROZEN_FINAL_SHA256", hashlib.sha256(parent.read_bytes()).hexdigest())
    first = build_smoke_manifest(parent)
    second = build_smoke_manifest(parent)
    assert first == second
    assert set(first) == {"manifest_version", "parent_manifest_sha256", "selection_rule", "examples"}
    assert len(first["examples"]) == 3
    assert {(row["dataset"], row["task_type"]) for row in first["examples"]} == {
        ("memora", "remembering"), ("memconflict", "dynamic"), ("memconflict", "conditional")
    }


def test_smoke_manifest_rejects_wrong_parent_and_does_not_overwrite(tmp_path, monkeypatch):
    parent = _parent(tmp_path)
    monkeypatch.setattr("scripts.build_rq1_gpt56_smoke.FROZEN_FINAL_SHA256", hashlib.sha256(parent.read_bytes()).hexdigest())
    manifest = build_smoke_manifest(parent)
    output = tmp_path / "smoke.json"
    write_smoke_manifest(manifest, output)
    with pytest.raises(FileExistsError):
        write_smoke_manifest(manifest, output)
    manifest["parent_manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="parent SHA256"):
        validate_smoke_manifest(manifest)


def _smoke_result(example: dict[str, str]) -> dict[str, object]:
    return {
        "query_id": example["example_id"],
        "dataset": example["dataset"],
        "task_type": example["task_type"],
        "condition": "reconstruction",
        "applicable_governance_dimensions": [],
        "skipped": False,
        "reconstruction_status": "success",
        "validation_status": "valid",
        "reconstructed_states": [],
        "evaluator_only": {"model_visible_interaction_ids": []},
    }


def _write_smoke_inputs(tmp_path, monkeypatch):
    parent = _parent(tmp_path)
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    monkeypatch.setattr("scripts.build_rq1_gpt56_smoke.FROZEN_FINAL_SHA256", parent_sha)
    manifest = build_smoke_manifest(parent)
    manifest_path = tmp_path / "smoke.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result_path = tmp_path / "results.jsonl"
    result_path.write_text(
        "\n".join(json.dumps(_smoke_result(example)) for example in manifest["examples"]) + "\n",
        encoding="utf-8",
    )
    return manifest, manifest_path, result_path


def test_smoke_results_are_summarized_against_manifest_examples(tmp_path, monkeypatch):
    manifest, manifest_path, result_path = _write_smoke_inputs(tmp_path, monkeypatch)
    summary = summarize(manifest_path, [result_path])
    assert summary["manifest_version"] == SMOKE_MANIFEST_VERSION
    assert summary["complete"] is True
    assert summary["conditions"]["reconstruction"]["operational"]["selected_n"] == 3


def test_smoke_summary_rejects_missing_and_unexpected_rows(tmp_path, monkeypatch):
    manifest, manifest_path, result_path = _write_smoke_inputs(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]

    result_path.write_text("\n".join(json.dumps(row) for row in rows[:2]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not exactly match"):
        summarize(manifest_path, [result_path])

    unexpected = dict(rows[2])
    unexpected["query_id"] = "not-in-smoke-manifest"
    result_path.write_text("\n".join(json.dumps(row) for row in [rows[0], rows[1], unexpected]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not exactly match"):
        summarize(manifest_path, [result_path])
