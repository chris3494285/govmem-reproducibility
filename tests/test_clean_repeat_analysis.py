"""Offline clean-repeat analysis tests. No provider calls are made."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_clean_repeat_stochasticity import (
    FORMAL_COMPARATOR, _paired_bootstrap_ci, _resolve_clean_b, analyze,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "results/stochasticity_control/clean_repeat_v1"


def test_all_sixty_blocks_resolve_with_frozen_evaluator():
    rows, resolution = _resolve_clean_b(CONTROL)
    assert len(rows) == 60
    assert resolution["n_blocks"] == 60
    assert resolution["n_resolved_clean_b"] == 60
    assert all(r["clean_B_correctness"] is not None for r in rows)
    methods = {r["clean_B_resolution_method"] for r in rows}
    # frozen evaluator routes only; no obsolete unified scorer
    assert methods <= {"dual_primary_binary_agreement", "gemini_binary_majority", "memora_gemini_primary",
                        "native_deterministic", "memora_primary_secondary_agreement", "memora_third_majority",
                        "memora_secondary_disagreement"}
    assert "native_deterministic" in methods


def test_deterministic_block_counted_once():
    rows, _ = _resolve_clean_b(CONTROL)
    det = [r for r in rows if r["repeat_block_id"] == "clean-repeat-v1::xai::longmemeval::43"]
    assert len(det) == 1
    assert det[0]["clean_A_route"] == "deterministic_abstention"
    assert det[0]["clean_B_resolution_method"] == "native_deterministic"
    offline = [json.loads(line) for line in
               (CONTROL / "judging/deterministic_abstention_results.jsonl").read_text().splitlines() if line.strip()]
    assert len(offline) == 1 and offline[0]["paid_call"] is False
    assert det[0]["clean_B_correctness"] == 0.0


def test_analysis_is_deterministic():
    first = analyze(CONTROL)
    second = analyze(CONTROL)
    assert first["overall"] == second["overall"]
    assert first["longmemeval"] == second["longmemeval"]
    assert first["memora"] == second["memora"]
    assert first["blocks"] == second["blocks"]


def test_frozen_denominators_and_comparator():
    metrics = analyze(CONTROL)
    assert metrics["n_blocks"] == 60
    assert metrics["longmemeval"]["n"] == 30
    assert metrics["memora"]["n"] == 30
    assert metrics["overall"]["n"] == 60
    assert metrics["formal_300_comparator"]["corruption_effect_pp"] == FORMAL_COMPARATOR["corruption_effect_pp"]
    # frozen evaluator counts: 29 LME dual-primary judged, 2 escalations, 30 memora
    counts = metrics["resolution"]["evaluator_counts"]
    assert counts["lme_primary_judged"] == 29
    assert counts["lme_escalation_completed"] == 2
    assert counts["memora_observations"] == 30
    assert counts["deterministic"] == 1


def test_paired_bootstrap_uses_frozen_seed_and_is_reproducible():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    first = _paired_bootstrap_ci(values, iterations=200)
    second = _paired_bootstrap_ci(values, iterations=200)
    assert first == second  # fixed seed -> reproducible
    assert first[0] <= first[1]


def test_paper_table_exists_and_matches_metrics():
    metrics = analyze(CONTROL)
    tex = (CONTROL / "analysis/clean_repeat_paper_table.tex").read_text()
    assert "clean-repeat-stochasticity" in tex
    assert f"{metrics['overall']['paired_difference_pp']:+.2f}" in tex
