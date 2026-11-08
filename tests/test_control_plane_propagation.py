from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_control_plane_propagation import EXPECTED_SHA, MANIFEST, analyze, canonical_admission, canonical_context


def test_frozen_manifest_and_canonical_representations_are_stable():
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == EXPECTED_SHA
    row = json.loads(next(line for line in MANIFEST.read_text().splitlines() if line.strip()))
    assert canonical_admission(row) == canonical_admission(row)
    assert canonical_context(row) == canonical_context(row)


def test_control_plane_analysis_is_complete_and_offline(tmp_path):
    summary = analyze(output_dir=tmp_path)
    assert summary["base_count"] == 731
    assert summary["fault_pair_count"] == 2080
    assert summary["provider_count"] == 6
    assert summary["provider_rows"] == 2080 * 6
    assert summary["overall"]["admission_change"]["denominator"] == 2080 * 6
    assert summary["overall"]["context_change"]["denominator"] == 2080 * 6
    assert all((tmp_path / name).is_file() for name in ("mechanism_instance_level.csv", "mechanism_summary.json", "mechanism_by_provider.csv", "mechanism_by_event.csv", "mechanism_by_fault.csv", "mechanism_by_source.csv"))


def test_analysis_is_deterministic(tmp_path):
    first = analyze(output_dir=tmp_path / "a")
    second = analyze(output_dir=tmp_path / "b")
    assert first == second
    for name in ("mechanism_instance_level.csv", "mechanism_summary.json", "mechanism_by_provider.csv", "mechanism_by_event.csv", "mechanism_by_fault.csv", "mechanism_by_source.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_missing_or_duplicate_result_join_fails(tmp_path):
    paths = dict(__import__("scripts.analyze_control_plane_propagation", fromlist=["PROVIDER_RESULTS"]).PROVIDER_RESULTS)
    altered = tmp_path / "duplicate.jsonl"
    original = paths["deepseek"].read_text()
    altered.write_text(original + original.splitlines()[0] + "\n")
    paths["deepseek"] = altered
    with pytest.raises(ValueError, match="duplicate"):
        analyze(result_paths=paths, output_dir=tmp_path / "out")
