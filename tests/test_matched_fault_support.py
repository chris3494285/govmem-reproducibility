from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_matched_fault_support import analyze


def test_matched_support_analysis_uses_same_bases_and_is_deterministic(tmp_path):
    first = analyze(output_dir=tmp_path / "a")
    second = analyze(output_dir=tmp_path / "b")
    assert first == second
    assert first["base_count"] == 731
    assert first["fault_pair_count"] == 2080
    assert first["support_counts"] == {"missing": 731, "stale": 351, "flipped": 380, "conflicting": 618}
    assert first["bases_by_supported_fault_count"] == {"1": 0, "2": 113, "3": 618, "4": 0}
    assert (tmp_path / "a/matched_fault_instance_level.csv").read_bytes() == (tmp_path / "b/matched_fault_instance_level.csv").read_bytes()


def test_stale_overlap_counts_are_transparent():
    summary = analyze(output_dir=Path("/tmp/govmem-matched-support-test"))
    pairs = {frozenset((row["fault_a"], row["fault_b"])): row["matched_base_count"] for row in summary["overlap_counts"] if "breakdown" not in row}
    assert pairs[frozenset(("stale", "missing"))] == 351
    assert pairs[frozenset(("stale", "flipped"))] == 0
    assert pairs[frozenset(("stale", "conflicting"))] == 238
    assert not any(row["fault_pair"] == "stale_vs_flipped" and row["matched_base_count"] for row in summary["comparisons"])


def test_no_provider_api_path_is_used():
    source = Path("scripts/analyze_matched_fault_support.py").read_text(encoding="utf-8")
    assert "Provider" not in source
    assert "requests" not in source
