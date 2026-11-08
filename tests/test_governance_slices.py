from __future__ import annotations

import json
from pathlib import Path

from eval.governance_slices import generate_governance_slices, load_governance_source, write_governance_slices
from eval.runner import run_demo


SOURCE = Path("data/longmemeval_converted.jsonl")


def test_governance_slice_generator_writes_jsonl(tmp_path):
    output_path = tmp_path / "governance_slices.jsonl"
    write_governance_slices(SOURCE, output_path)

    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines
    parsed = [json.loads(line) for line in lines]
    assert all("::" in example["id"] for example in parsed)


def test_governance_slice_ids_and_fields(tmp_path):
    raw_examples = load_governance_source(SOURCE)
    slices = generate_governance_slices(raw_examples)
    ids = {example["id"] for example in slices}

    assert "lme_001::conflict" in ids
    assert "lme_001::expired" in ids
    assert "lme_001::deleted" in ids
    assert "lme_001::unauthorized" in ids
    assert "lme_001::abstain" in ids

    unauthorized = next(example for example in slices if example["id"] == "lme_001::unauthorized")
    assert "private" not in unauthorized["allowed_scopes"]
    assert any(memory["scope"] == "private" for memory in unauthorized["memories"])

    deleted = next(example for example in slices if example["id"] == "lme_001::deleted")
    assert any(memory["is_deleted"] for memory in deleted["memories"])

    expired = next(example for example in slices if example["id"] == "lme_001::expired")
    assert any(memory["valid_until"] is not None for memory in expired["memories"])
    assert any(memory["valid_until"] < expired["current_time"] for memory in expired["memories"] if memory["valid_until"] is not None)

    abstain = next(example for example in slices if example["id"] == "lme_001::abstain")
    assert abstain["answer"] == "unknown"
    assert any("answer:" in memory["content"].lower() for memory in abstain["memories"])


def test_governance_slices_are_safe_for_govmem(tmp_path):
    output_path = tmp_path / "governance_slices.jsonl"
    results_dir = tmp_path / "results"
    write_governance_slices(SOURCE, output_path)

    results = run_demo(output_path, results_dir=results_dir)
    naive = next(entry for entry in results["methods"] if entry["method"] == "naive_memory")
    govmem = next(entry for entry in results["methods"] if entry["method"] == "govmem")

    assert govmem["policy_violation_count"] == 0
    assert govmem["stale_memory_error_count"] == 0
    assert govmem["safe_exact_match"] >= naive["safe_exact_match"]

