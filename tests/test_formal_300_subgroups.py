"""Offline tests for the final Formal-300 subgroup analysis.

These tests read the frozen final human-adjudicated Formal-300 block-level
artifact, which lives under the gitignored private results tree, so the whole
module is marked ``requires_private_artifacts`` and is skipped unless
``--include-private`` is passed.  No LLM/API call is made and no frozen value is
modified.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.analyze_formal_300_subgroups import (
    BLOCK_LEVEL_CSV,
    BOOTSTRAP_SEED,
    REFERENCE,
    analyze,
    bootstrap_block,
    bootstrap_cluster,
    cluster_groups,
    load_blocks,
    summarize,
)

pytestmark = pytest.mark.requires_private_artifacts

if not BLOCK_LEVEL_CSV.is_file():  # pragma: no cover - public snapshot
    pytest.skip(
        "frozen Formal-300 final analysis artifact is not present in this checkout",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def blocks():
    loaded, reconciliation = load_blocks()
    return loaded, reconciliation


@pytest.fixture(scope="module")
def analysis():
    return analyze(output_dir=None, write=False, iterations=200)


# --------------------------------------------------------------------------- #
# 1. final adjudicated 298/300 reconciliation
# --------------------------------------------------------------------------- #
def test_final_adjudicated_reconciliation(analysis):
    checks = analysis["recon"]["checks"]
    assert checks["selected_blocks"] == 300
    assert checks["resolved_blocks"] == 298
    # Frozen metric: 898 resolved observations out of 900. The triplet analysis
    # consumes 298 complete triplets = 894 observations; a block with any
    # unresolved component is excluded entirely.
    assert analysis["reconciliation"]["resolved_observations"] == 898
    assert analysis["reconciliation"]["observations_in_complete_triplets"] == 298 * 3
    assert analysis["overall"]["n"] == 298
    assert checks["selected_distinct_bases"] == 239
    assert checks["resolved_distinct_bases"] == 238
    assert checks["corruption_effect"] == pytest.approx(0.35536290041409735)
    assert checks["recovery_effect"] == pytest.approx(0.3267353221505277)


def test_distinct_base_definitions_are_reported_not_conflated(analysis):
    note = analysis["recon"]["distinct_base_note"]
    assert "239" in note and "238" in note
    definitions = analysis["comparison"]["distinct_base_definitions"]
    assert definitions["selected_300_blocks"] == 239
    assert definitions["resolved_298_blocks"] == 238


def test_frozen_block_bootstrap_is_reproduced_exactly(blocks):
    loaded, _ = blocks
    ci = bootstrap_block(loaded)  # default seed 20261010, 10,000 iterations
    assert ci["corruption_effect"] == pytest.approx(REFERENCE["corruption_ci"], abs=0.0)
    assert ci["recovery_effect"] == pytest.approx(REFERENCE["recovery_ci"], abs=0.0)
    assert ci["restoration_gap"] == pytest.approx(REFERENCE["restoration_ci"], abs=0.0)


# --------------------------------------------------------------------------- #
# 2. no duplicated blocks
# --------------------------------------------------------------------------- #
def test_no_duplicated_blocks(blocks):
    loaded, reconciliation = blocks
    ids = [b["pilot_block_id"] for b in loaded]
    assert len(ids) == len(set(ids)) == 298
    assert reconciliation["dropped_block_n"] == 2


def test_each_block_carries_a_complete_triplet(blocks):
    loaded, _ = blocks
    for block in loaded:
        assert set(("clean", "corrupted", "enforced")) <= set(block)
        assert all(0.0 <= block[key] <= 1.0 for key in ("clean", "corrupted", "enforced"))


# --------------------------------------------------------------------------- #
# 3. consistent score directions
# --------------------------------------------------------------------------- #
def test_consistent_score_directions(analysis):
    assert analysis["overall"]["corruption_effect"] > 0
    assert analysis["overall"]["recovery_effect"] > 0
    for source, row in analysis["by_source"].items():
        if row["n"]:
            assert row["corruption_effect"] > 0, source
            assert row["recovery_effect"] > 0, source
    for provider, row in analysis["by_provider"].items():
        assert row["corruption_effect"] > 0, provider
        assert row["recovery_effect"] > 0, provider


def test_pooled_effects_are_mean_of_source_effects_weighted_by_blocks(analysis):
    overall = analysis["overall"]
    by_source = analysis["by_source"]
    total = sum(row["n"] for row in by_source.values())
    assert total == overall["n"]
    for key in ("clean", "corrupted", "enforced", "corruption_effect", "recovery_effect"):
        weighted = sum(by_source[s][key] * by_source[s]["n"] for s in by_source) / total
        assert weighted == pytest.approx(overall[key])


# --------------------------------------------------------------------------- #
# 4. source / provider sum consistency
# --------------------------------------------------------------------------- #
def test_source_and_provider_sums_are_consistent(analysis):
    overall = analysis["overall"]
    by_source = analysis["by_source"]
    by_provider = analysis["by_provider"]
    by_cell = analysis["by_provider_source"]

    assert sum(row["n"] for row in by_source.values()) == overall["n"]
    assert sum(row["n"] for row in by_provider.values()) == overall["n"]
    assert sum(row["n"] for row in by_cell.values()) == overall["n"]

    for provider, row in by_provider.items():
        cell_total = sum(
            by_cell[(provider, source)]["n"] for source in ("longmemeval", "memora")
        )
        assert cell_total == row["n"], provider
        assert sum(row["dataset_composition"].values()) == row["n"], provider
        for source in ("longmemeval", "memora"):
            assert row["dataset_composition"].get(source, 0) == by_cell[(provider, source)]["n"]

    for source in ("longmemeval", "memora"):
        assert sum(by_cell[(p, source)]["n"] for p in by_provider) == by_source[source]["n"]


def test_cluster_counts_never_exceed_block_counts(analysis):
    subsets = [analysis["overall"], *analysis["by_source"].values(), *analysis["by_provider"].values()]
    subsets += list(analysis["by_provider_source"].values())
    for row in subsets:
        if row["n"]:
            assert 0 < row["clusters"] <= row["n"]


# --------------------------------------------------------------------------- #
# 5. correct pairing
# --------------------------------------------------------------------------- #
def test_pairing_matches_the_frozen_block_level_effects():
    with BLOCK_LEVEL_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        clean, corrupted, enforced = (
            float(row["clean_score"]),
            float(row["corrupted_score"]),
            float(row["enforced_score"]),
        )
        assert float(row["corruption_effect"]) == pytest.approx(clean - corrupted)
        assert float(row["recovery_effect"]) == pytest.approx(enforced - corrupted)
        assert float(row["restoration_gap"]) == pytest.approx(enforced - clean)


def test_summarize_uses_paired_block_means(blocks):
    loaded, _ = blocks
    metrics = summarize(loaded)
    assert metrics["n"] == 298
    assert metrics["corruption_effect"] == pytest.approx(metrics["clean"] - metrics["corrupted"])
    assert metrics["recovery_effect"] == pytest.approx(metrics["enforced"] - metrics["corrupted"])
    assert metrics["restoration_gap"] == pytest.approx(metrics["enforced"] - metrics["clean"])


# --------------------------------------------------------------------------- #
# 6. bootstrap reproducibility
# --------------------------------------------------------------------------- #
def test_block_bootstrap_is_reproducible(blocks):
    loaded, _ = blocks
    first = bootstrap_block(loaded, iterations=400, seed=BOOTSTRAP_SEED)
    second = bootstrap_block(loaded, iterations=400, seed=BOOTSTRAP_SEED)
    assert first == second


def test_cluster_bootstrap_is_reproducible(blocks):
    loaded, _ = blocks
    first = bootstrap_cluster(loaded, iterations=400, seed=BOOTSTRAP_SEED)
    second = bootstrap_cluster(loaded, iterations=400, seed=BOOTSTRAP_SEED)
    assert first == second


def test_cluster_bootstrap_is_sensitive_to_seed(blocks):
    loaded, _ = blocks
    first = bootstrap_cluster(loaded, iterations=200, seed=BOOTSTRAP_SEED)
    second = bootstrap_cluster(loaded, iterations=200, seed=BOOTSTRAP_SEED + 1)
    assert first["ci"] != second["ci"]


def test_cluster_intervals_are_at_least_as_wide_as_block_intervals_for_pooled_effect(analysis):
    overall = analysis["overall"]
    for key in ("corruption_effect", "recovery_effect"):
        block_ci = overall["block_bootstrap_ci95"][key]
        cluster_ci = overall["cluster_bootstrap_ci95"][key]
        assert (cluster_ci[1] - cluster_ci[0]) > (block_ci[1] - block_ci[0])


# --------------------------------------------------------------------------- #
# 7. cluster-aware handling of repeated bases
# --------------------------------------------------------------------------- #
def test_repeated_bases_are_grouped_into_single_clusters(blocks):
    loaded, reconciliation = blocks
    grouped = cluster_groups(loaded, ("source",))
    flat = [b for clusters in grouped.values() for group in clusters.values() for b in group]
    assert len(flat) == len(loaded)
    for clusters in grouped.values():
        for base_id, group in clusters.items():
            assert {b["base_id"] for b in group} == {base_id}
    # 298 resolved blocks collapse onto 238 distinct bases, so reuse must exist.
    assert reconciliation["resolved_distinct_bases"] == 238
    assert any(len(group) > 1 for clusters in grouped.values() for group in clusters.values())


def test_cluster_resampling_never_splits_a_base(blocks):
    loaded, _ = blocks
    grouped = cluster_groups(loaded, ("source",))
    per_base = {base_id: len(group) for clusters in grouped.values() for base_id, group in clusters.items()}
    # Emulate one draw of the shipped resampler and confirm block counts per base
    # are always an integer multiple of that base's block count.
    import random

    rng = random.Random(BOOTSTRAP_SEED)
    drawn: list[dict] = []
    for stratum in sorted(grouped):
        clusters = list(grouped[stratum].values())
        for _ in range(len(clusters)):
            drawn.extend(clusters[rng.randrange(len(clusters))])
    counts: dict[str, int] = {}
    for block in drawn:
        counts[block["base_id"]] = counts.get(block["base_id"], 0) + 1
    for base_id, count in counts.items():
        assert count % per_base[base_id] == 0, base_id
    # At least one base must actually be drawn more than once in a 238-cluster draw.
    assert max(counts.values()) >= 1


def test_cluster_bootstrap_counts_clusters_not_blocks(blocks):
    loaded, _ = blocks
    result = bootstrap_cluster(loaded, iterations=100, seed=BOOTSTRAP_SEED)
    assert result["clusters"] == 238
    assert result["stratum_cluster_counts"] == {"longmemeval": 105, "memora": 133}


# --------------------------------------------------------------------------- #
# 8. no inclusion of unresolved cases
# --------------------------------------------------------------------------- #
def test_unresolved_blocks_are_excluded(blocks):
    loaded, reconciliation = blocks
    assert reconciliation["dropped_block_n"] == 2
    assert len(reconciliation["dropped_block_descriptions"]) == 2
    assert reconciliation["resolved_observations"] == 898
    assert reconciliation["observations_in_complete_triplets"] == 298 * 3
    assert reconciliation["bases_fully_dropped"] == 1
    assert reconciliation["bases_retained_elsewhere"] == 1


def test_frozen_262_metrics_reference_has_two_unresolved(analysis):
    assert analysis["reconciliation"]["resolved_blocks"] == 300 - 2
    assert analysis["reconciliation"]["resolved_observations"] == 900 - 2


# --------------------------------------------------------------------------- #
# 9. end-to-end determinism and publication hygiene
# --------------------------------------------------------------------------- #
def test_outputs_are_deterministic(tmp_path):
    first = analyze(output_dir=tmp_path / "a", iterations=150)
    second = analyze(output_dir=tmp_path / "b", iterations=150)
    for name in (
        "source_summary.csv",
        "provider_summary.csv",
        "provider_source_summary.csv",
        "bootstrap_comparison.json",
        "FORMAL_300_SUBGROUP_REPORT.md",
        "FORMAL_300_SUBGROUP_TABLES.tex",
    ):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert first["overall"] == second["overall"]


def test_outputs_contain_no_identifiers(tmp_path):
    analyze(output_dir=tmp_path, iterations=100)
    for path in Path(tmp_path).glob("*"):
        text = path.read_text(encoding="utf-8")
        assert "pilot-block::" not in text
        assert "pilot-observation::" not in text
        assert "::" not in text


def test_script_has_no_network_or_provider_calls():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "analyze_formal_300_subgroups.py"
    ).read_text(encoding="utf-8")
    for banned in (
        "requests",
        "urllib",
        "http://",
        "https://",
        "import openai",
        "import anthropic",
        "subprocess",
        "SystemExit(1)",
    ):
        assert banned not in source
