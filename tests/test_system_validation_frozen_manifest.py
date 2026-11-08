from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_system_validation_frozen_manifest import (
    _fault_state,
    build_manifest,
    write_frozen,
)
from scripts.run_governance_fault_rq1 import load_manifest
from scripts.validate_system_validation_frozen_manifest import validate_manifest


SOURCE = "data/governance_fault_rq1_pilot_frozen_v2.jsonl"


def test_formal_manifest_is_deterministic_and_has_exact_cohort_sizes():
    source = load_manifest(SOURCE, require_full=True)
    first, first_info = build_manifest(source, seed=42, source_manifest=SOURCE)
    second, second_info = build_manifest(source, seed=42, source_manifest=SOURCE)
    assert first == second
    assert first_info == second_info
    assert len(first) == 180
    assert len({row["base_example_id"] for row in first}) == 60
    assert {dimension: len({row["base_example_id"] for row in first if row["dimension"] == dimension}) for dimension in ("authorization", "deletion", "temporal")} == {"authorization": 20, "deletion": 20, "temporal": 20}


def test_formal_condition_sets_and_sibling_invariants():
    rows, _ = build_manifest(load_manifest(SOURCE, require_full=True))
    expected = {
        "gatemem": {"clean", "conflicting", "flipped"},
        "memora": {"clean", "flipped", "stale"},
        "memconflict": {"clean", "missing", "stale"},
    }
    groups = {}
    for row in rows:
        groups.setdefault(row["base_example_id"], []).append(row)
    for group in groups.values():
        assert {row["condition"] for row in group} == expected[group[0]["source_dataset"]]
        assert len({row["query"] for row in group}) == 1
        assert len({row["underlying_memory_sha256"] for row in group}) == 1
        assert len({json.dumps(row["underlying_memory_content"], sort_keys=True) for row in group}) == 1
        assert len({(row["base_example_id"], row["condition"]) for row in group}) == 3


def test_write_frozen_hash_and_overwrite_protection(tmp_path):
    output = tmp_path / "manifest.jsonl"
    metadata = tmp_path / "metadata.json"
    info = write_frozen(SOURCE, output, metadata)
    assert validate_manifest(output, metadata)["manifest_sha256"] == info["manifest_sha256"]
    with pytest.raises(FileExistsError):
        write_frozen(SOURCE, output, metadata)
    write_frozen(SOURCE, output, metadata, force=True)


def test_validator_rejects_manifest_hash_or_metadata_mismatch(tmp_path):
    output = tmp_path / "manifest.jsonl"
    metadata_path = tmp_path / "metadata.json"
    write_frozen(SOURCE, output, metadata_path)
    metadata = json.loads(metadata_path.read_text())
    metadata["manifest_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="SHA256"):
        validate_manifest(output, metadata_path)


def test_memconflict_and_memora_fault_completion_require_native_predecessors():
    rows = load_manifest(SOURCE, require_full=True)
    memconflict = next(row for row in rows if row["source_dataset"] == "memconflict" and row["condition_type"] == "clean")
    broken_conflict = json.loads(json.dumps(memconflict))
    broken_conflict["source_metadata"]["historical_temporal_states"] = []
    with pytest.raises(ValueError, match="predecessor"):
        _fault_state(broken_conflict, "stale")
    memora = next(row for row in rows if row["source_dataset"] == "memora" and row["condition_type"] == "clean")
    broken_memora = json.loads(json.dumps(memora))
    broken_memora["source_metadata"]["historical_deletion_states"] = []
    with pytest.raises(ValueError, match="predecessor"):
        _fault_state(broken_memora, "stale")


def test_gate_mem_clean_rows_are_deny():
    rows, _ = build_manifest(load_manifest(SOURCE, require_full=True))
    clean = [row for row in rows if row["source_dataset"] == "gatemem" and row["condition"] == "clean"]
    assert len(clean) == 20
    assert all(row["gold_admission_decision"] == "deny" for row in clean)
    assert all(row["expected_governance_action"] == "governance_deny" for row in clean)

