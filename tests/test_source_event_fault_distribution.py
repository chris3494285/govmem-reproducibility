"""Offline tests for the Source x Event x Fault distribution audit.

The focused tests run on small synthetic fixtures that mimic the frozen
manifest schema, so they need neither licensed benchmark text nor a private
checkout.  One additional test re-verifies the real frozen v3 cohort and is
marked ``requires_licensed_data`` (skipped unless the frozen manifest is
present and ``--include-private`` is passed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_source_event_fault_distribution import (
    DEFAULT_BASE_MANIFEST,
    DEFAULT_CONDITION_MANIFEST,
    DEFAULT_FREEZE,
    FAULTS,
    analyze,
    render_paper_table,
)

ALL_FAULTS = ("missing", "stale", "flipped", "conflicting")


# --------------------------------------------------------------------------- #
# synthetic fixture builders
# --------------------------------------------------------------------------- #
def _base_row(base_id: str, source: str, event: str, faults: tuple[str, ...]) -> dict:
    return {
        "base_id": base_id,
        "source_dataset": source,
        "analysis_event_type": event,
        "primary_event_type": event,
        "supported_fault_operators": list(faults),
        "unsupported_fault_operators": {f: "not applicable" for f in ALL_FAULTS if f not in faults},
    }


def _condition_rows(bases: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for base in bases:
        rows.append(
            {
                "condition_id": f"{base['base_id']}::clean",
                "base_id": base["base_id"],
                "source_dataset": base["source_dataset"],
                "analysis_event_type": base["analysis_event_type"],
                "condition_type": "clean",
                "fault_operator": None,
                "pair_id": None,
            }
        )
        for fault in base["supported_fault_operators"]:
            for kind in ("corrupted", "enforced"):
                rows.append(
                    {
                        "condition_id": f"{base['base_id']}::{fault}::{kind}",
                        "base_id": base["base_id"],
                        "source_dataset": base["source_dataset"],
                        "analysis_event_type": base["analysis_event_type"],
                        "condition_type": kind,
                        "fault_operator": fault,
                        "pair_id": f"{base['base_id']}::{fault}",
                    }
                )
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _make_cohort(tmp_path: Path, spec: dict[str, tuple[str, str, tuple[str, ...]]]):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bases = [_base_row(bid, source, event, faults) for bid, (source, event, faults) in spec.items()]
    base_path = _write_jsonl(tmp_path / "base.jsonl", bases)
    condition_path = _write_jsonl(tmp_path / "conditions.jsonl", _condition_rows(bases))
    return condition_path, base_path


DEFAULT_SPEC = {
    "lm::1": ("longmemeval", "temporal", ("missing", "stale")),
    "lm::2": ("longmemeval", "update", ("missing", "stale", "conflicting")),
    "me::1": ("memora", "deletion", ("missing", "flipped", "conflicting")),
}


def _run(tmp_path: Path, spec=None, **kwargs):
    condition_path, base_path = _make_cohort(tmp_path, spec or DEFAULT_SPEC)
    return analyze(
        condition_manifest=condition_path,
        base_manifest=base_path,
        freeze=tmp_path / "absent.freeze.json",
        output_dir=kwargs.pop("output_dir", None),
        write=kwargs.pop("write", False),
        expected_invariants=kwargs.pop("expected_invariants", None),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# exact global count reconciliation
# --------------------------------------------------------------------------- #
def test_synthetic_global_count_reconciliation(tmp_path):
    result = _run(tmp_path)
    totals = result["global_totals"]
    assert totals == {
        "distinct_bases": 3,
        "matched_fault_pairs": 8,
        "corrupted_conditions": 8,
        "enforced_conditions": 8,
        "clean_conditions": 3,
        "total_conditions": 19,
    }
    assert (
        totals["clean_conditions"] + totals["corrupted_conditions"] + totals["enforced_conditions"]
        == totals["total_conditions"]
    )
    assert 2 * totals["matched_fault_pairs"] == (
        totals["corrupted_conditions"] + totals["enforced_conditions"]
    )


def test_strict_invariant_enforcement_raises_on_mismatch(tmp_path):
    with pytest.raises(ValueError, match="invariant mismatch"):
        _run(tmp_path, expected_invariants={"matched_fault_pairs": 999})


# --------------------------------------------------------------------------- #
# cross-tab sum consistency
# --------------------------------------------------------------------------- #
def test_cross_tab_sums_are_consistent(tmp_path):
    result = _run(tmp_path)
    pairs = result["global_totals"]["matched_fault_pairs"]
    assert sum(cell["fault_pair_count"] for cell in result["cells"]) == pairs
    assert sum(t["fault_pair_count"] for t in result["per_event_totals"].values()) == pairs
    assert sum(t["fault_pair_count"] for t in result["per_fault_totals"].values()) == pairs
    assert sum(t["fault_pair_count"] for t in result["per_source_totals"].values()) == pairs
    for cell in result["cells"]:
        assert cell["fault_pair_count"] == cell["corrupted_condition_count"]
        assert cell["fault_pair_count"] == cell["enforced_condition_count"]
        assert cell["fault_pair_count"] == cell["distinct_base_count"]


# --------------------------------------------------------------------------- #
# unique-base deduplication
# --------------------------------------------------------------------------- #
def test_unique_base_dedup_across_faults(tmp_path):
    result = _run(tmp_path)
    per_fault = result["per_fault_totals"]
    # lm::2 supports three faults but must be counted once per fault cell only.
    assert per_fault["missing"]["distinct_base_count"] == 3
    assert per_fault["stale"]["distinct_base_count"] == 2
    assert per_fault["flipped"]["distinct_base_count"] == 1
    assert per_fault["conflicting"]["distinct_base_count"] == 2
    # distinct-base counts are not additive across faults ...
    assert sum(t["distinct_base_count"] for t in per_fault.values()) > result["global_totals"]["distinct_bases"]
    # ... while pair counts are.
    assert sum(t["fault_pair_count"] for t in per_fault.values()) == result["global_totals"]["matched_fault_pairs"]
    assert result["per_source_totals"]["longmemeval"]["distinct_base_count"] == 2


# --------------------------------------------------------------------------- #
# fault-pair sibling completeness and duplicates
# --------------------------------------------------------------------------- #
def test_incomplete_fault_pair_sibling_is_rejected(tmp_path):
    condition_path, base_path = _make_cohort(tmp_path, DEFAULT_SPEC)
    rows = [json.loads(line) for line in condition_path.read_text().splitlines()]
    rows = [r for r in rows if r["condition_id"] != "lm::1::missing::enforced"]
    _write_jsonl(condition_path, rows)
    with pytest.raises(ValueError, match="incomplete siblings"):
        analyze(condition_manifest=condition_path, base_manifest=base_path,
                freeze=tmp_path / "absent.freeze.json", output_dir=None, write=False,
                expected_invariants=None)


def test_duplicate_fault_pair_sibling_is_rejected(tmp_path):
    condition_path, base_path = _make_cohort(tmp_path, DEFAULT_SPEC)
    rows = [json.loads(line) for line in condition_path.read_text().splitlines()]
    duplicate = dict(next(r for r in rows if r["condition_id"] == "lm::1::missing::enforced"))
    duplicate["condition_id"] = "lm::1::missing::enforced::dup"
    rows.append(duplicate)
    _write_jsonl(condition_path, rows)
    with pytest.raises(ValueError, match="duplicate siblings"):
        analyze(condition_manifest=condition_path, base_manifest=base_path,
                freeze=tmp_path / "absent.freeze.json", output_dir=None, write=False,
                expected_invariants=None)


def test_duplicate_condition_id_is_rejected(tmp_path):
    condition_path, base_path = _make_cohort(tmp_path, DEFAULT_SPEC)
    rows = [json.loads(line) for line in condition_path.read_text().splitlines()]
    rows.append(dict(rows[0]))
    _write_jsonl(condition_path, rows)
    with pytest.raises(ValueError, match="duplicate condition_id"):
        analyze(condition_manifest=condition_path, base_manifest=base_path,
                freeze=tmp_path / "absent.freeze.json", output_dir=None, write=False,
                expected_invariants=None)


def test_fault_applicability_violation_is_rejected(tmp_path):
    condition_path, base_path = _make_cohort(tmp_path, DEFAULT_SPEC)
    rows = [json.loads(line) for line in condition_path.read_text().splitlines()]
    # lm::1 does not support `conflicting`; inject a corrupted record for it.
    rows.append(
        {
            "condition_id": "lm::1::conflicting::corrupted",
            "base_id": "lm::1",
            "source_dataset": "longmemeval",
            "analysis_event_type": "temporal",
            "condition_type": "corrupted",
            "fault_operator": "conflicting",
            "pair_id": "lm::1::conflicting",
        }
    )
    _write_jsonl(condition_path, rows)
    with pytest.raises(ValueError, match="fault applicability"):
        analyze(condition_manifest=condition_path, base_manifest=base_path,
                freeze=tmp_path / "absent.freeze.json", output_dir=None, write=False,
                expected_invariants=None)


# --------------------------------------------------------------------------- #
# missing / unsupported combination handling
# --------------------------------------------------------------------------- #
def test_missing_and_unsupported_combinations_are_distinguished(tmp_path):
    result = _run(tmp_path)
    unused = {
        (c["source"], c["event"])
        for c in result["structurally_absent_cells"]["unused_source_event_combinations"]
    }
    # longmemeval x deletion and memora x temporal/update have no bases.
    assert ("longmemeval", "deletion") in unused
    assert ("memora", "temporal") in unused
    unsupported = {
        (c["source"], c["event"], c["fault"])
        for c in result["structurally_absent_cells"]["unsupported_fault_cells"]
    }
    assert ("longmemeval", "temporal", "flipped") in unsupported
    assert ("memora", "deletion", "stale") in unsupported
    # No cell is supported-in-principle but empty in practice.
    assert result["observed_zero_cells"] == []
    # The paper table renders both structural markers.
    table = render_paper_table(result)
    assert "n/a (no bases)" in table
    assert "--" in table


# --------------------------------------------------------------------------- #
# output determinism and release hygiene
# --------------------------------------------------------------------------- #
def test_outputs_are_written_and_deterministic(tmp_path):
    first = _run(tmp_path / "a", output_dir=tmp_path / "a" / "results", write=True)
    second = _run(tmp_path / "b", output_dir=tmp_path / "b" / "results", write=True)
    for name in (
        "source_event_fault_counts.csv",
        "source_event_counts.csv",
        "source_fault_counts.csv",
        "SOURCE_EVENT_FAULT_PAPER_TABLE.tex",
    ):
        assert (tmp_path / "a" / "results" / name).read_bytes() == (
            tmp_path / "b" / "results" / name
        ).read_bytes()
    assert first["global_totals"] == second["global_totals"]


def test_outputs_contain_no_row_level_identifiers(tmp_path):
    _run(tmp_path / "c", output_dir=tmp_path / "c" / "results", write=True)
    for name in ("source_event_fault_counts.csv", "source_event_counts.csv", "source_fault_counts.csv"):
        text = (tmp_path / "c" / "results" / name).read_text()
        assert "::" not in text  # no base_id / condition_id / pair_id values


def test_script_has_no_network_or_provider_calls():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "analyze_source_event_fault_distribution.py"
    ).read_text(encoding="utf-8")
    for banned in ("requests", "urllib", "http://", "https://", "openai", "anthropic", "subprocess"):
        assert banned not in source


# --------------------------------------------------------------------------- #
# real frozen cohort (private / licensed artifacts; skipped by default)
# --------------------------------------------------------------------------- #
@pytest.mark.requires_licensed_data
@pytest.mark.skipif(
    not DEFAULT_CONDITION_MANIFEST.is_file(),
    reason="frozen v3 condition manifest is not present in this checkout",
)
def test_frozen_cohort_reconciliation():
    result = analyze(output_dir=None, write=False)
    assert result["global_totals"] == {
        "distinct_bases": 731,
        "matched_fault_pairs": 2080,
        "corrupted_conditions": 2080,
        "enforced_conditions": 2080,
        "clean_conditions": 731,
        "total_conditions": 4891,
    }
    assert result["observed_cell_count"] == 14
    assert result["proposed_identity"]["supported"] is True
    assert result["proposed_identity"]["bases_with_three_faults"] == 618
    assert result["proposed_identity"]["bases_with_two_faults"] == 113
    assert result["source_event_confounding"]["confounded"] is True
    assert result["per_fault_totals"]["missing"]["fault_pair_count"] == 731
    assert result["per_fault_totals"]["stale"]["fault_pair_count"] == 351
    assert result["per_fault_totals"]["flipped"]["fault_pair_count"] == 380
    assert result["per_fault_totals"]["conflicting"]["fault_pair_count"] == 618


def test_frozen_cohort_paths_match_repo_convention():
    assert DEFAULT_BASE_MANIFEST.name == "final_main_governance_base_manifest_v2.jsonl"
    assert DEFAULT_CONDITION_MANIFEST.name == "final_main_governance_condition_manifest_v3.jsonl"
    assert DEFAULT_FREEZE.name == "final_main_governance_condition_manifest_v3.freeze.json"
    assert tuple(FAULTS) == ALL_FAULTS
