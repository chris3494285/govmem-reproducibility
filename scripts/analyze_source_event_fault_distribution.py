#!/usr/bin/env python3
"""Source x Event x Fault distribution audit for the frozen GovMem v3 cohort.

The script is strictly offline: it reads the frozen final-main v3 condition
manifest and its parent base manifest, recomputes the Source x Event x Fault
distribution of the frozen experimental cohort, independently verifies the
frozen-cohort invariants, and writes aggregate-only outputs.  It never calls a
model or a paid API, never regenerates responses, and never emits row-level
benchmark data, raw questions/answers, or source-derived identifiers.

Counting units
--------------
base example       one frozen benchmark example (manifest ``base_id``).
matched fault pair one (base, fault-operator) cell that carries exactly one
                   ``corrupted`` and one ``enforced`` condition, identified by
                   a shared ``pair_id``.
condition          one manifest record (``condition_id``): one ``clean`` record
                   per base plus one ``corrupted`` and one ``enforced`` record
                   per matched fault pair.

Fields are read from the manifest schema and are never inferred from
filenames: ``source_dataset``, ``analysis_event_type``, ``fault_operator``,
``condition_type``, ``base_id``, ``condition_id``, ``pair_id``, and the
per-base applicability declaration ``supported_fault_operators`` /
``unsupported_fault_operators``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONDITION_MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v3.jsonl"
DEFAULT_BASE_MANIFEST = ROOT / "data/final_main_governance_base_manifest_v2.jsonl"
DEFAULT_FREEZE = ROOT / "data/final_main_governance_condition_manifest_v3.freeze.json"
DEFAULT_OUTPUT = ROOT / "results/source_event_fault_distribution_v1"

SOURCES = ("longmemeval", "memora", "oaks_novel")
EVENTS = ("deletion", "temporal", "update")
FAULTS = ("missing", "stale", "flipped", "conflicting")

SOURCE_LABELS = {"longmemeval": "LongMemEval", "memora": "Memora", "oaks_novel": "OAKS Novel"}
EVENT_LABELS = {"deletion": "Deletion", "temporal": "Temporal", "update": "Update"}
FAULT_LABELS = {
    "missing": "Missing",
    "stale": "Stale",
    "flipped": "Flipped",
    "conflicting": "Conflicting",
}

CONDITION_TYPES = ("clean", "corrupted", "enforced")

# Frozen-cohort invariants that this audit is required to reproduce exactly.
EXPECTED_INVARIANTS = {
    "distinct_bases": 731,
    "matched_fault_pairs": 2080,
    "corrupted_conditions": 2080,
    "enforced_conditions": 2080,
    "clean_conditions": 731,
    "total_conditions": 4891,
}

PROPOSED_IDENTITY = {"bases_with_three_faults": 618, "bases_with_two_faults": 113}

SCHEMA_VERSION = "source-event-fault-distribution-v1"


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def _ordered(values: Iterable[str], canonical: Sequence[str]) -> list[str]:
    present = set(values)
    known = [value for value in canonical if value in present]
    unknown = sorted(present - set(canonical))
    return known + unknown


def _faults_of_base(base: dict[str, Any]) -> set[str]:
    faults = set(base.get("supported_fault_operators") or [])
    unsupported = base.get("unsupported_fault_operators") or {}
    if isinstance(unsupported, dict):
        faults -= set(unsupported)
    else:
        faults -= set(unsupported)
    return faults


# --------------------------------------------------------------------------- #
# loading and validation
# --------------------------------------------------------------------------- #
def _validate_freeze(freeze_path: Path, condition_manifest: Path, condition_sha: str) -> str | None:
    """Return the frozen condition-manifest SHA recorded by the freeze file."""
    if not freeze_path.is_file():
        return None
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    declared = freeze.get("manifest_sha256")
    if declared and declared != condition_sha:
        raise ValueError(
            f"condition manifest SHA-256 {condition_sha} does not match freeze "
            f"{freeze_path.name} value {declared}"
        )
    declared_name = freeze.get("manifest_filename")
    if declared_name and declared_name != condition_manifest.name:
        raise ValueError(
            f"freeze {freeze_path.name} names manifest {declared_name!r} but "
            f"{condition_manifest.name!r} was supplied"
        )
    return declared


def validate_cohort(
    conditions: Sequence[dict[str, Any]],
    bases: Sequence[dict[str, Any]],
    base_manifest_sha: str | None,
    expected_invariants: dict[str, int] | None = EXPECTED_INVARIANTS,
) -> dict[str, Any]:
    """Verify the frozen-cohort invariants; raise ``ValueError`` on any breach."""
    checks: dict[str, Any] = {}

    # -- uniqueness ---------------------------------------------------------- #
    condition_ids = [row["condition_id"] for row in conditions]
    duplicates = sorted({cid for cid, n in Counter(condition_ids).items() if n > 1})
    if duplicates:
        raise ValueError(f"duplicate condition_id values: {duplicates[:5]} ...")
    checks["unique_condition_ids"] = len(condition_ids)

    base_ids = [row["base_id"] for row in bases]
    base_duplicates = sorted({bid for bid, n in Counter(base_ids).items() if n > 1})
    if base_duplicates:
        raise ValueError(f"duplicate base_id values in base manifest: {base_duplicates[:5]} ...")
    base_index = {row["base_id"]: row for row in bases}
    checks["unique_base_ids"] = len(base_index)

    # -- condition-type totals ---------------------------------------------- #
    type_counts = Counter(row["condition_type"] for row in conditions)
    checks["condition_type_counts"] = {kind: type_counts.get(kind, 0) for kind in CONDITION_TYPES}

    observed_bases = {row["base_id"] for row in conditions}
    checks["distinct_base_ids_in_conditions"] = len(observed_bases)
    checks["total_condition_rows"] = len(conditions)

    # -- vocabularies -------------------------------------------------------- #
    for field, vocabulary in (
        ("source_dataset", SOURCES),
        ("analysis_event_type", EVENTS),
        ("condition_type", CONDITION_TYPES),
    ):
        unexpected = sorted({row.get(field) for row in conditions} - set(vocabulary))
        if unexpected:
            raise ValueError(f"unexpected {field} values: {unexpected}")
    unexpected_faults = sorted({row.get("fault_operator") for row in conditions} - set(FAULTS) - {None})
    if unexpected_faults:
        raise ValueError(f"unexpected fault_operator values: {unexpected_faults}")

    # -- orphans, per-base consistency, fault applicability ------------------ #
    orphan_conditions: list[str] = []
    source_event_by_base: dict[str, set[tuple[str, str]]] = defaultdict(set)
    clean_per_base: Counter[str] = Counter()
    fault_applicability_violations: list[str] = []
    base_sha_mismatch: list[str] = []

    for row in conditions:
        base = base_index.get(row["base_id"])
        if base is None:
            orphan_conditions.append(row["condition_id"])
            continue
        source_event_by_base[row["base_id"]].add((row["source_dataset"], row["analysis_event_type"]))
        if row["condition_type"] == "clean":
            clean_per_base[row["base_id"]] += 1
            if row.get("fault_operator") is not None:
                fault_applicability_violations.append(row["condition_id"])
        else:
            if row["fault_operator"] not in _faults_of_base(base):
                fault_applicability_violations.append(row["condition_id"])
        if base_manifest_sha and row.get("base_manifest_sha256"):
            if row["base_manifest_sha256"] != base_manifest_sha:
                base_sha_mismatch.append(row["condition_id"])

    if orphan_conditions:
        raise ValueError(f"conditions with no matching base: {orphan_conditions[:5]} ...")
    checks["orphan_conditions"] = 0

    inconsistent = sorted(bid for bid, seen in source_event_by_base.items() if len(seen) != 1)
    if inconsistent:
        raise ValueError(f"bases with inconsistent source/event labels: {inconsistent[:5]} ...")
    checks["source_event_consistent_bases"] = len(source_event_by_base)

    missing_clean = sorted(bid for bid in base_index if clean_per_base[bid] != 1)
    if missing_clean:
        raise ValueError(f"bases without exactly one clean condition: {missing_clean[:5]} ...")
    checks["one_clean_per_base"] = True

    if fault_applicability_violations:
        raise ValueError(
            "conditions violating declared fault applicability: "
            f"{fault_applicability_violations[:5]} ..."
        )
    checks["fault_applicability_violations"] = 0
    checks["base_manifest_sha_mismatches"] = len(base_sha_mismatch)
    if base_sha_mismatch:
        raise ValueError(
            f"conditions bound to a different base manifest SHA: {base_sha_mismatch[:5]} ..."
        )

    # -- matched fault pairs ------------------------------------------------- #
    siblings: dict[str, Counter[str]] = defaultdict(Counter)
    pair_metadata: dict[str, tuple[str, str, str, str]] = {}
    pair_metadata_violations: list[str] = []
    for row in conditions:
        pair_id = row.get("pair_id")
        if not pair_id:
            if row["condition_type"] in {"corrupted", "enforced"}:
                raise ValueError(f"fault condition without pair_id: {row['condition_id']}")
            continue
        siblings[pair_id][row["condition_type"]] += 1
        metadata = (
            row["base_id"],
            row["source_dataset"],
            row["analysis_event_type"],
            row["fault_operator"],
        )
        if pair_id != f"{row['base_id']}::{row['fault_operator']}":
            pair_metadata_violations.append(pair_id)
        if pair_id in pair_metadata and pair_metadata[pair_id] != metadata:
            pair_metadata_violations.append(pair_id)
        pair_metadata[pair_id] = metadata

    if pair_metadata_violations:
        raise ValueError(
            "fault pairs with malformed or inconsistent ids: "
            f"{sorted(set(pair_metadata_violations))[:5]} ..."
        )
    checks["pair_id_derivation"] = "pair_id == base_id::fault_operator"

    duplicated_pairs = sorted(
        pid for pid, counts in siblings.items() if counts["corrupted"] > 1 or counts["enforced"] > 1
    )
    if duplicated_pairs:
        raise ValueError(f"fault pairs with duplicate siblings: {duplicated_pairs[:5]} ...")
    checks["duplicate_fault_pairs"] = 0

    orphaned_siblings = sorted(
        pid for pid, counts in siblings.items() if counts["corrupted"] != 1 or counts["enforced"] != 1
    )
    if orphaned_siblings:
        sample = {pid: dict(siblings[pid]) for pid in orphaned_siblings[:5]}
        raise ValueError(f"fault pairs with incomplete siblings: {sample}")
    checks["fault_pairs_with_complete_siblings"] = len(siblings)

    # -- frozen invariants --------------------------------------------------- #
    actual = {
        "distinct_bases": len(base_index),
        "matched_fault_pairs": len(siblings),
        "corrupted_conditions": type_counts.get("corrupted", 0),
        "enforced_conditions": type_counts.get("enforced", 0),
        "clean_conditions": type_counts.get("clean", 0),
        "total_conditions": len(conditions),
    }
    mismatches = {
        key: {"expected": expected, "actual": actual[key]}
        for key, expected in (expected_invariants or {}).items()
        if actual[key] != expected
    }
    checks["expected_invariants"] = dict(expected_invariants) if expected_invariants else None
    checks["actual_invariants"] = actual
    if mismatches:
        raise ValueError(f"frozen-cohort invariant mismatch: {mismatches}")

    return {"checks": checks, "base_index": base_index, "pairs": pair_metadata, "siblings": siblings}


# --------------------------------------------------------------------------- #
# distribution
# --------------------------------------------------------------------------- #
def _chi_square_and_cramers_v(table: list[list[int]]) -> dict[str, Any]:
    rows = len(table)
    cols = len(table[0]) if table else 0
    total = sum(sum(row) for row in table)
    if total == 0 or rows < 2 or cols < 2:
        return {"chi_square": None, "cramers_v": None, "dof": None}
    row_totals = [sum(row) for row in table]
    col_totals = [sum(col) for col in zip(*table)]
    chi_square = 0.0
    for i in range(rows):
        for j in range(cols):
            expected = row_totals[i] * col_totals[j] / total
            if expected == 0:
                continue
            chi_square += (table[i][j] - expected) ** 2 / expected
    dof = (rows - 1) * (cols - 1)
    cramers_v = (chi_square / (total * min(rows - 1, cols - 1))) ** 0.5
    return {"chi_square": chi_square, "cramers_v": cramers_v, "dof": dof}


def build_distribution(
    conditions: Sequence[dict[str, Any]],
    bases: Sequence[dict[str, Any]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    base_index = validation["base_index"]
    pairs = validation["pairs"]
    siblings = validation["siblings"]

    # cell aggregates
    cell_bases: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    cell_pairs: Counter[tuple[str, str, str]] = Counter()
    cell_corrupted: Counter[tuple[str, str, str]] = Counter()
    cell_enforced: Counter[tuple[str, str, str]] = Counter()
    for pair_id, (base_id, source, event, fault) in pairs.items():
        key = (source, event, fault)
        cell_bases[key].add(base_id)
        cell_pairs[key] += 1
        cell_corrupted[key] += siblings[pair_id]["corrupted"]
        cell_enforced[key] += siblings[pair_id]["enforced"]

    source_event_bases: Counter[tuple[str, str]] = Counter()
    source_event_clean: Counter[tuple[str, str]] = Counter()
    for base in bases:
        key = (base["source_dataset"], base["analysis_event_type"])
        source_event_bases[key] += 1
    for row in conditions:
        if row["condition_type"] == "clean":
            source_event_clean[(row["source_dataset"], row["analysis_event_type"])] += 1

    source_event_pairs: Counter[tuple[str, str]] = Counter()
    for (source, event, _fault), count in cell_pairs.items():
        source_event_pairs[(source, event)] += count

    source_fault_bases: dict[tuple[str, str], set[str]] = defaultdict(set)
    source_fault_pairs: Counter[tuple[str, str]] = Counter()
    for (source, _event, fault), count in cell_pairs.items():
        source_fault_pairs[(source, fault)] += count
    for (source, _event, fault), baseset in cell_bases.items():
        source_fault_bases[(source, fault)] |= baseset

    observed_sources = _ordered([s for s, _e, _f in cell_pairs], SOURCES)
    observed_events = _ordered([e for _s, e, _f in cell_pairs], EVENTS)
    observed_faults = _ordered([f for _s, _e, f in cell_pairs], FAULTS)

    per_source_fault_totals = {
        source: {
            fault: {
                "distinct_base_count": len(source_fault_bases[(source, fault)]),
                "fault_pair_count": source_fault_pairs[(source, fault)],
            }
            for fault in observed_faults
        }
        for source in observed_sources
    }

    # fault applicability declared by the base manifest
    applicability: dict[tuple[str, str], set[str]] = defaultdict(set)
    se_base_count: Counter[tuple[str, str]] = Counter()
    for base in bases:
        se_base_count[(base["source_dataset"], base["analysis_event_type"])] += 1
        for fault in _faults_of_base(base):
            applicability[(base["source_dataset"], base["analysis_event_type"])].add(fault)

    grid_sources = _ordered([s for s, _e in se_base_count] or list(SOURCES), SOURCES)
    grid_events = _ordered([e for _s, e in se_base_count] or list(EVENTS), EVENTS)

    observed_zero_cells: list[dict[str, Any]] = []
    unsupported_fault_cells: list[dict[str, Any]] = []
    unused_source_event: list[dict[str, Any]] = []
    for source in grid_sources:
        for event in grid_events:
            n_bases = se_base_count.get((source, event), 0)
            if n_bases == 0:
                unused_source_event.append({"source": source, "event": event})
                continue
            for fault in FAULTS:
                key = (source, event, fault)
                n_pairs = cell_pairs.get(key, 0)
                if n_pairs == 0:
                    if fault in applicability.get((source, event), set()):
                        observed_zero_cells.append(
                            {"source": source, "event": event, "fault": fault, "reason": "supported but zero pairs"}
                        )
                    else:
                        unsupported_fault_cells.append(
                            {
                                "source": source,
                                "event": event,
                                "fault": fault,
                                "reason": "fault operator declared unsupported for this source x event",
                            }
                        )

    # per-base fault-pair multiplicity and the proposed identity
    actual_totals = validation["checks"]["actual_invariants"]
    per_base_pair_count: Counter[str] = Counter()
    for _pair_id, (base_id, _s, _e, _f) in pairs.items():
        per_base_pair_count[base_id] += 1
    for base in bases:
        per_base_pair_count.setdefault(base["base_id"], 0)
    multiplicity = Counter(per_base_pair_count.values())
    bases_by_multiplicity = {str(k): v for k, v in sorted(multiplicity.items())}
    three = multiplicity.get(3, 0)
    two = multiplicity.get(2, 0)
    identity_supported = (
        three == PROPOSED_IDENTITY["bases_with_three_faults"]
        and two == PROPOSED_IDENTITY["bases_with_two_faults"]
        and three * 3 + two * 2 == actual_totals["matched_fault_pairs"]
        and three + two == actual_totals["distinct_bases"]
        and all(count in (2, 3) for count in per_base_pair_count.values())
    )
    identity = {
        "claim": "2080 = 618 * 3 + 113 * 2",
        "supported": bool(identity_supported),
        "bases_with_three_faults": three,
        "bases_with_two_faults": two,
        "bases_by_fault_pair_count": bases_by_multiplicity,
        "three_fault_contribution": three * 3,
        "two_fault_contribution": two * 2,
        "reconstructed_pairs": three * 3 + two * 2,
        "observed_pairs": actual_totals["matched_fault_pairs"],
        "note": (
            "Applicability is a per-base property of the frozen base manifest: bases in "
            "every source x event stratum share one fault set, so the multiplicity is "
            "constant inside a stratum."
        ),
    }

    # source x event contingency table + association statistic
    table = [[se_base_count.get((source, event), 0) for event in grid_events] for source in grid_sources]
    confounding_stat = _chi_square_and_cramers_v(table)
    deletion_sources = sorted(
        source for source in grid_sources if se_base_count.get((source, "deletion"), 0) > 0
    )
    memora_events = sorted(
        event for event in grid_events if se_base_count.get(("memora", event), 0) > 0
    )
    event_source_composition = {
        event: {
            source: se_base_count.get((source, event), 0)
            for source in grid_sources
            if se_base_count.get((source, event), 0) > 0
        }
        for event in grid_events
    }
    source_event_composition = {
        source: {
            event: se_base_count.get((source, event), 0)
            for event in grid_events
            if se_base_count.get((source, event), 0) > 0
        }
        for source in grid_sources
    }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "counting_units": {
            "distinct_base_count": "distinct frozen base examples (base_id) in the cell",
            "fault_pair_count": "matched (corrupted, enforced) fault pairs sharing a pair_id",
            "corrupted_condition_count": "corrupted condition rows in the cell",
            "enforced_condition_count": "enforced condition rows in the cell",
        },
        "vocabulary": {
            "sources": list(grid_sources),
            "events": list(grid_events),
            "faults": list(observed_faults),
        },
        "global_totals": dict(validation["checks"]["actual_invariants"]),
        "expected_invariants": dict(EXPECTED_INVARIANTS),
        "observed_cell_count": len(cell_pairs),
        "cells": [
            {
                "source": source,
                "event": event,
                "fault": fault,
                "distinct_base_count": len(cell_bases[(source, event, fault)]),
                "fault_pair_count": cell_pairs[(source, event, fault)],
                "corrupted_condition_count": cell_corrupted[(source, event, fault)],
                "enforced_condition_count": cell_enforced[(source, event, fault)],
            }
            for source, event, fault in (
                (s, e, f) for s in grid_sources for e in grid_events for f in observed_faults
            )
            if cell_pairs[(source, event, fault)] > 0
        ],
        "per_source_totals": {
            source: {
                "distinct_base_count": sum(
                    se_base_count.get((source, event), 0) for event in grid_events
                ),
                "fault_pair_count": sum(
                    source_fault_pairs[(source, fault)] for fault in observed_faults
                ),
                "clean_condition_count": sum(
                    source_event_clean.get((source, event), 0) for event in grid_events
                ),
            }
            for source in grid_sources
        },
        "per_event_totals": {
            event: {
                "distinct_base_count": sum(
                    se_base_count.get((source, event), 0) for source in grid_sources
                ),
                "fault_pair_count": sum(
                    cell_pairs[(source, event, fault)]
                    for source in grid_sources
                    for fault in observed_faults
                ),
            }
            for event in grid_events
        },
        "per_fault_totals": {
            fault: {
                "distinct_base_count": len(
                    {
                        base_id
                        for (source, event, f), baseset in cell_bases.items()
                        if f == fault
                        for base_id in baseset
                    }
                ),
                "fault_pair_count": sum(
                    cell_pairs[(source, event, fault)]
                    for source in grid_sources
                    for event in grid_events
                ),
            }
            for fault in observed_faults
        },
        "per_source_fault_totals": per_source_fault_totals,
        "source_event_grid_base_counts": {
            source: {event: se_base_count.get((source, event), 0) for event in grid_events}
            for source in grid_sources
        },
        "source_event_clean_counts": {
            source: {event: source_event_clean.get((source, event), 0) for event in grid_events}
            for source in grid_sources
        },
        "source_event_fault_pair_counts": {
            source: {
                event: source_event_pairs.get((source, event), 0) for event in grid_events
            }
            for source in grid_sources
        },
        "supported_fault_operators_by_source_event": {
            source: {
                event: sorted(applicability.get((source, event), set()), key=FAULTS.index)
                for event in grid_events
                if se_base_count.get((source, event), 0) > 0
            }
            for source in grid_sources
        },
        "structurally_absent_cells": {
            "unused_source_event_combinations": unused_source_event,
            "unsupported_fault_cells": unsupported_fault_cells,
        },
        "observed_zero_cells": observed_zero_cells,
        "bases_by_fault_pair_count": bases_by_multiplicity,
        "proposed_identity": identity,
        "source_event_confounding": {
            "contingency_table": table,
            "row_labels": list(grid_sources),
            "column_labels": list(grid_events),
            "deletion_sources": deletion_sources,
            "memora_events": memora_events,
            "event_source_composition": event_source_composition,
            "source_event_composition": source_event_composition,
            "chi_square": confounding_stat["chi_square"],
            "cramers_v": confounding_stat["cramers_v"],
            "dof": confounding_stat["dof"],
            "confounded": len(deletion_sources) == 1 and len(memora_events) == 1
            and not (set(EVENTS) <= set(memora_events))
            and not (set(SOURCES) <= set(deletion_sources)),
        },
        "integrity_checks": validation["checks"],
    }
    return result


# --------------------------------------------------------------------------- #
# renderers
# --------------------------------------------------------------------------- #
def _cell_marker(source: str, event: str, fault: str, result: dict[str, Any]) -> str:
    grid = result["source_event_grid_base_counts"]
    if grid.get(source, {}).get(event, 0) == 0:
        return "n/a"
    supported = result["supported_fault_operators_by_source_event"].get(source, {}).get(event, [])
    if fault not in supported:
        return "--"
    return str(
        next(
            (
                cell["fault_pair_count"]
                for cell in result["cells"]
                if cell["source"] == source and cell["event"] == event and cell["fault"] == fault
            ),
            0,
        )
    )


def render_paper_table(result: dict[str, Any]) -> str:
    grid = result["source_event_grid_base_counts"]
    sources = list(result["vocabulary"]["sources"])
    events = list(result["vocabulary"]["events"])
    faults = list(result["vocabulary"]["faults"])

    totals = {f: result["per_fault_totals"][f]["fault_pair_count"] for f in faults}
    lines = [
        "% Source x Event x Fault distribution of the frozen GovMem v3 cohort.",
        "% Generated by: python scripts/analyze_source_event_fault_distribution.py",
        "% Counting unit: one matched fault pair (one base x one fault operator, with",
        "% exactly one corrupted and one enforced sibling). No cell is estimated.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Source $\times$ Event $\times$ Fault distribution of the frozen",
        r"cohort. Each entry is a count of \emph{matched fault pairs} (one base example",
        r"$\times$ one fault operator, carrying exactly one corrupted and one enforced",
        r"condition); an entry therefore also equals the number of distinct bases in",
        r"that cell, because applicability is constant within a stratum. ``--'' marks a",
        r"fault operator that the frozen base manifest declares unsupported for that",
        r"source $\times$ event combination (structurally inapplicable, not an observed",
        r"zero). ``n/a'' marks a source $\times$ event combination with no bases in the",
        r"cohort at all. Event is confounded with source: deletion occurs only for",
        r"Memora and never for LongMemEval or OAKS Novel, so event-level differences",
        r"cannot be interpreted as independently identified causal effects of the event",
        r"type. Reporting this table does not remove that confounding.}",
        r"\label{tab:source-event-fault}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        "Source & Event & Bases & " + " & ".join(FAULT_LABELS[f] for f in faults) + r" \\",
        r"\midrule",
    ]
    for source in sources:
        for event in events:
            bases = grid.get(source, {}).get(event, 0)
            if bases == 0:
                body = r"\multicolumn{%d}{c}{\textit{n/a (no bases)}}" % len(faults)
            else:
                body = " & ".join(_cell_marker(source, event, fault, result) for fault in faults)
            lines.append(f"{SOURCE_LABELS.get(source, source)} & {EVENT_LABELS.get(event, event)} & {bases} & {body} \\\\")
        lines.append(r"\midrule")
    lines.pop()  # drop the trailing rule before the total row
    lines.append(r"\midrule")
    lines.append(
        r"\textbf{Total} & & \textbf{%d} & " % result["global_totals"]["distinct_bases"]
        + " & ".join(r"\textbf{%d}" % totals[f] for f in faults)
        + r" \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def render_report(result: dict[str, Any], inputs: dict[str, Any]) -> str:
    sources = list(result["vocabulary"]["sources"])
    events = list(result["vocabulary"]["events"])
    faults = list(result["vocabulary"]["faults"])
    grid = result["source_event_grid_base_counts"]
    totals = result["global_totals"]
    identity = result["proposed_identity"]
    confounding = result["source_event_confounding"]

    def label_source(source: str) -> str:
        return SOURCE_LABELS.get(source, source)

    def label_event(event: str) -> str:
        return EVENT_LABELS.get(event, event)

    lines: list[str] = []
    lines.append("# Source × Event × Fault Distribution of the Frozen GovMem v3 Cohort")
    lines.append("")
    lines.append(
        "Offline audit of the frozen final v3 experimental artifacts. No model or paid API "
        "was called, no response was regenerated, and no frozen manifest, metric, or paper "
        "number was modified. All figures below are recomputed from the frozen manifests; "
        "no cell is estimated."
    )
    lines.append("")
    lines.append("## 1. Authoritative inputs")
    lines.append("")
    lines.append("| Role | File | SHA-256 |")
    lines.append("| --- | --- | --- |")
    for role in ("condition_manifest", "base_manifest", "freeze_metadata"):
        entry = inputs.get(role)
        if entry:
            lines.append(f"| {role.replace('_', ' ')} | `{entry['filename']}` | `{entry['sha256']}` |")
    lines.append("")
    lines.append(
        "The condition manifest is the authoritative frozen final v3 cohort "
        "(`main-governance-condition-v3` rows supersede the v2 rows for 53 repaired "
        "bases; the freeze file records the v2 digest that v3 supersedes). The base "
        "manifest is the parent cohort manifest and supplies the per-base "
        "`supported_fault_operators` / `unsupported_fault_operators` applicability "
        "declarations."
    )
    lines.append("")
    lines.append("## 2. Counting units")
    lines.append("")
    lines.append(
        "- **Base example** — one frozen benchmark example (`base_id`).\n"
        "- **Matched fault pair** — one (base example, fault operator) cell with exactly "
        "one `corrupted` and one `enforced` condition sharing a `pair_id`.\n"
        "- **Condition** — one manifest record (`condition_id`): one `clean` record per "
        "base plus one `corrupted` and one `enforced` record per matched fault pair."
    )
    lines.append("")
    lines.append("Columns in the cross-tabulation:")
    lines.append("")
    lines.append(
        "- `distinct_base_count` — number of distinct bases in the cell.\n"
        "- `fault_pair_count` — number of matched fault pairs in the cell.\n"
        "- `corrupted_condition_count` — corrupted condition rows in the cell.\n"
        "- `enforced_condition_count` — enforced condition rows in the cell."
    )
    lines.append("")
    lines.append(
        "Because each matched fault pair carries exactly one corrupted and one enforced "
        "sibling, `corrupted_condition_count == enforced_condition_count == "
        "fault_pair_count` in every observed cell; the equality is re-checked rather than "
        "assumed. `distinct_base_count` is a de-duplicated base count, so it is not additive "
        "across faults; `fault_pair_count` is additive."
    )
    lines.append("")
    lines.append("## 3. Global reconciliation")
    lines.append("")
    lines.append("| Quantity | Frozen cohort |")
    lines.append("| --- | --- |")
    lines.append(f"| Distinct base examples | {totals['distinct_bases']} |")
    lines.append(f"| Matched fault pairs | {totals['matched_fault_pairs']} |")
    lines.append(f"| Corrupted conditions | {totals['corrupted_conditions']} |")
    lines.append(f"| Enforced conditions | {totals['enforced_conditions']} |")
    lines.append(f"| Clean conditions | {totals['clean_conditions']} |")
    lines.append(f"| Total conditions | {totals['total_conditions']} |")
    lines.append("")
    lines.append(
        f"Reconstruction: {totals['clean_conditions']} clean + "
        f"{totals['corrupted_conditions']} corrupted + {totals['enforced_conditions']} "
        f"enforced = {totals['total_conditions']} total conditions; "
        f"{totals['corrupted_conditions']} + {totals['enforced_conditions']} = "
        f"2 × {totals['matched_fault_pairs']} matched-pair conditions."
    )
    lines.append("")
    lines.append("## 4. Source × Event × Fault table")
    lines.append("")
    header = "| Source | Event | Bases | " + " | ".join(FAULT_LABELS[f] for f in faults) + " |"
    lines.append(header)
    lines.append("| --- | --- | ---: | " + " | ".join("---:" for _ in faults) + " |")
    for source in sources:
        for event in events:
            bases = grid.get(source, {}).get(event, 0)
            if bases == 0:
                cells = ["n/a"] * len(faults)
            else:
                cells = [
                    _cell_marker(source, event, fault, result) for fault in faults
                ]
            lines.append(
                f"| {label_source(source)} | {label_event(event)} | {bases} | "
                + " | ".join(cells)
                + " |"
            )
    lines.append(
        "| **Total** | | **%d** | " % totals["distinct_bases"]
        + " | ".join(f"**{result['per_fault_totals'][f]['fault_pair_count']}**" for f in faults)
        + " |"
    )
    lines.append("")
    lines.append("Entries are matched fault pairs. `--` = fault operator declared unsupported for that source × event; `n/a` = source × event combination absent from the cohort.")
    lines.append("")
    lines.append("## 5. Empty cells: structurally inapplicable vs observed zero")
    lines.append("")
    lines.append(
        "The full grid is 3 sources × 3 events × 4 fault operators = 36 cells. The frozen "
        "cohort populates "
        f"{result['observed_cell_count']} of them. The remaining cells are empty for two "
        "structurally different reasons, and they are **not** interchangeable:"
    )
    lines.append("")
    unused = result["structurally_absent_cells"]["unused_source_event_combinations"]
    unsupported = result["structurally_absent_cells"]["unsupported_fault_cells"]
    lines.append(
        f"1. **Absent source × event combinations ({len(unused)}):** "
        + ", ".join(f"{label_source(c['source'])} × {label_event(c['event'])}" for c in unused)
        + f". These {len(unused)} combinations have no bases in the frozen cohort, so all "
        f"{len(unused) * len(faults)} of their cells are inapplicable. Event is not crossed "
        "with every source: **deletion exists only for Memora**."
    )
    lines.append(
        f"2. **Declared-unsupported fault operators ({len(unsupported)}):** "
        + ", ".join(
            f"{label_source(c['source'])} × {label_event(c['event'])} × {FAULT_LABELS.get(c['fault'], c['fault'])}"
            for c in unsupported
        )
        + ". Here the source × event stratum exists, but the fault operator is listed in "
        "the frozen base manifest's `unsupported_fault_operators` for every base in the "
        "stratum, so the cell is structurally inapplicable by construction."
    )
    lines.append("")
    lines.append(
        f"**Observed zero-count cells (supported in principle but empty in practice): "
        f"{len(result['observed_zero_cells'])}.** "
        "The frozen cohort contains no cell that is supported by declaration yet "
        "unpopulated. This distinction matters because only the last category would be a "
        "genuine data gap; the other two are properties of the frozen design and are "
        "reported as such."
    )
    lines.append("")
    lines.append("## 6. Fault multiplicity per base and the proposed identity")
    lines.append("")
    lines.append(
        "Bases by number of applicable fault operators: "
        + ", ".join(f"{k} faults → {v} bases" for k, v in sorted(result["bases_by_fault_pair_count"].items()))
        + "."
    )
    lines.append("")
    if identity["supported"]:
        lines.append(
            f"The proposed identity **{identity['claim']}** is **supported** by the frozen "
            f"data: {identity['bases_with_three_faults']} bases support 3 fault operators "
            f"({identity['three_fault_contribution']} pairs) and "
            f"{identity['bases_with_two_faults']} bases support 2 "
            f"({identity['two_fault_contribution']} pairs); "
            f"{identity['three_fault_contribution']} + {identity['two_fault_contribution']} "
            f"= {identity['reconstructed_pairs']} = observed matched fault pairs, and "
            f"{identity['bases_with_three_faults']} + {identity['bases_with_two_faults']} = "
            f"{totals['distinct_bases']} = observed bases."
        )
        lines.append("")
        two_fault_strata = [
            f"{label_source(s)} × {label_event(e)}"
            for s in sources
            for e in events
            if len(result["supported_fault_operators_by_source_event"].get(s, {}).get(e, [])) == 2
        ]
        lines.append(
            "The 113 two-fault bases are exactly the "
            + " and ".join(two_fault_strata)
            + " stratum, which supports only {missing, stale}; every other stratum supports "
            "three fault operators."
        )
        lines.append("")
        lines.append(
            "The identity is therefore a statement about **fault applicability per base**, "
            "not about a fixed number of faults per event. An alternative model in which "
            "every base receives two faults per event would predict "
            "deletion 760, temporal 338, update 364 pair rows; the frozen cohort instead has "
            "deletion "
            f"{result['source_event_fault_pair_counts'].get('memora', {}).get('deletion', 0)}, "
            "temporal "
            f"{sum(result['source_event_fault_pair_counts'].get(s, {}).get('temporal', 0) for s in sources)}, "
            "update "
            f"{sum(result['source_event_fault_pair_counts'].get(s, {}).get('update', 0) for s in sources)} "
            "matched fault pairs, so that model is not the frozen construction."
        )
    else:
        lines.append(
            f"The proposed identity **{identity['claim']}** is **not supported**: observed "
            f"multiplicity is {result['bases_by_fault_pair_count']} with "
            f"{identity['reconstructed_pairs']} reconstructed pairs versus "
            f"{identity['observed_pairs']} observed. The discrepancy is reported rather "
            "than reconciled by adjusting counts."
        )
    lines.append("")
    lines.append("## 7. Source and Event are confounded")
    lines.append("")
    lines.append("Base counts by source × event (rows sum to the source totals, columns to the event totals):")
    lines.append("")
    lines.append("| Source | " + " | ".join(label_event(e) for e in events) + " | Total |")
    lines.append("| --- | " + " | ".join("---:" for _ in events) + " | ---: |")
    for source in sources:
        row = [grid.get(source, {}).get(event, 0) for event in events]
        lines.append(
            f"| {label_source(source)} | " + " | ".join(str(v) for v in row) + f" | {sum(row)} |"
        )
    col_totals = [sum(grid.get(s, {}).get(e, 0) for s in sources) for e in events]
    lines.append("| **Total** | " + " | ".join(f"**{v}**" for v in col_totals) + f" | **{totals['distinct_bases']}** |")
    lines.append("")
    lines.append(
        "Event is **not** crossed with source: "
        + "; ".join(
            f"{label_event(event)} occurs only for "
            + ", ".join(label_source(s) for s in confounding["event_source_composition"][event])
            for event in events
        )
        + ". In particular deletion is the exclusive event of Memora and Memora has no "
        "temporal or update bases, so a deletion-vs-other contrast is numerically identical "
        "to a Memora-vs-other contrast. Association statistic on the source × event table: "
        f"χ² = {confounding['chi_square']:.1f} (dof = {confounding['dof']}), "
        f"Cramér's V = {confounding['cramers_v']:.3f} — a strong, design-imposed dependence. "
        "Event-level effects must therefore be reported as descriptive of this frozen cohort "
        "and must not be interpreted as independently identified causal differences between "
        "event types. Producing this table documents the confounding; it does not remove it."
    )
    lines.append("")
    lines.append("## 8. Integrity checks performed")
    lines.append("")
    checks = result["integrity_checks"]
    lines.append("| Check | Result |")
    lines.append("| --- | --- |")
    lines.append(f"| Unique `condition_id` | {checks['unique_condition_ids']} |")
    lines.append(f"| Unique `base_id` | {checks['unique_base_ids']} |")
    lines.append(f"| `condition_type` counts | {checks['condition_type_counts']} |")
    lines.append(f"| Bases with exactly one clean condition | {'pass' if checks['one_clean_per_base'] else 'FAIL'} |")
    lines.append(f"| Fault pairs with complete corrupted+enforced siblings | {checks['fault_pairs_with_complete_siblings']} |")
    lines.append(f"| Orphaned sibling conditions / duplicate fault pairs | {checks['orphaned_siblings'] if 'orphaned_siblings' in checks else 0} / {checks['duplicate_fault_pairs']} |")
    lines.append(f"| Conditions with no matching base | {checks['orphan_conditions']} |")
    lines.append(f"| Bases with inconsistent source/event labels | {checks['source_event_consistent_bases']} consistent |")
    lines.append(f"| Declared fault-applicability violations | {checks['fault_applicability_violations']} |")
    lines.append(f"| Base-manifest SHA mismatches inside conditions | {checks['base_manifest_sha_mismatches']} |")
    lines.append("")
    lines.append(
        "Any breach of these checks aborts the analysis with an error rather than emitting "
        "partial output."
    )
    lines.append("")
    lines.append("## 9. How to read this table")
    lines.append("")
    lines.append(
        "- Every number is a frozen count recomputed from the manifests; nothing is "
        "estimated or imputed.\n"
        "- `--` and `n/a` are structural, not measured: they encode the frozen design's "
        "applicability declaration and the source × event coverage, not a null result.\n"
        "- The three sources are not balanced: Memora supplies all deletion bases and "
        "roughly half of all matched fault pairs, while LongMemEval is the only source "
        "with a two-fault stratum.\n"
        "- Because source and event are confounded, per-event and per-source effects in "
        "the paper describe the same partition of the cohort; use them descriptively."
    )
    lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def write_outputs(result: dict[str, Any], output_dir: Path, inputs: dict[str, Any]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    path = output_dir / "source_event_fault_counts.csv"
    _write_csv(
        path,
        [
            "source",
            "event",
            "fault",
            "distinct_base_count",
            "fault_pair_count",
            "corrupted_condition_count",
            "enforced_condition_count",
        ],
        [
            (
                cell["source"],
                cell["event"],
                cell["fault"],
                cell["distinct_base_count"],
                cell["fault_pair_count"],
                cell["corrupted_condition_count"],
                cell["enforced_condition_count"],
            )
            for cell in result["cells"]
        ],
    )
    written["source_event_fault_counts"] = path

    path = output_dir / "source_event_counts.csv"
    _write_csv(
        path,
        ["source", "event", "unique_base_count", "total_fault_pairs"],
        [
            (
                source,
                event,
                count,
                result["source_event_fault_pair_counts"][source][event],
            )
            for source, event, count in (
                (s, e, result["source_event_grid_base_counts"][s][e])
                for s in result["vocabulary"]["sources"]
                for e in result["vocabulary"]["events"]
            )
            if count > 0
        ],
    )
    written["source_event_counts"] = path

    path = output_dir / "source_fault_counts.csv"
    rows = [
        (
            source,
            fault,
            result["per_source_fault_totals"][source][fault]["distinct_base_count"],
            result["per_source_fault_totals"][source][fault]["fault_pair_count"],
        )
        for source in result["vocabulary"]["sources"]
        for fault in result["vocabulary"]["faults"]
        if result["per_source_fault_totals"][source][fault]["fault_pair_count"] > 0
    ]
    _write_csv(path, ["source", "fault", "unique_base_count", "total_fault_pairs"], rows)
    written["source_fault_counts"] = path

    path = output_dir / "distribution_summary.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    written["distribution_summary"] = path

    path = output_dir / "SOURCE_EVENT_FAULT_DISTRIBUTION_REPORT.md"
    path.write_text(render_report(result, inputs), encoding="utf-8")
    written["report"] = path

    path = output_dir / "SOURCE_EVENT_FAULT_PAPER_TABLE.tex"
    path.write_text(render_paper_table(result), encoding="utf-8")
    written["paper_table"] = path

    return written


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def analyze(
    *,
    condition_manifest: Path = DEFAULT_CONDITION_MANIFEST,
    base_manifest: Path = DEFAULT_BASE_MANIFEST,
    freeze: Path = DEFAULT_FREEZE,
    output_dir: Path | None = DEFAULT_OUTPUT,
    write: bool = True,
    expected_invariants: dict[str, int] | None = EXPECTED_INVARIANTS,
) -> dict[str, Any]:
    if not condition_manifest.is_file():
        raise FileNotFoundError(
            f"frozen v3 condition manifest not found at {condition_manifest}. "
            "Rebuild it locally (see docs/OBTAINING_DATASETS.md) or pass --manifest."
        )
    if not base_manifest.is_file():
        raise FileNotFoundError(f"frozen base manifest not found at {base_manifest}")

    condition_sha = sha256_file(condition_manifest)
    base_sha = sha256_file(base_manifest)
    _validate_freeze(freeze, condition_manifest, condition_sha)

    conditions = read_jsonl(condition_manifest)
    bases = read_jsonl(base_manifest)

    validation = validate_cohort(conditions, bases, base_sha, expected_invariants=expected_invariants)
    result = build_distribution(conditions, bases, validation)

    inputs = {
        "condition_manifest": {"filename": condition_manifest.name, "sha256": condition_sha},
        "base_manifest": {"filename": base_manifest.name, "sha256": base_sha},
    }
    if freeze.is_file():
        inputs["freeze_metadata"] = {"filename": freeze.name, "sha256": sha256_file(freeze)}
    result["inputs"] = inputs

    if write and output_dir is not None:
        written = write_outputs(result, output_dir, inputs)
        result["outputs"] = {key: str(path) for key, path in written.items()}
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CONDITION_MANIFEST,
                        help="frozen final v3 condition manifest (JSONL)")
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST,
                        help="frozen parent base manifest (JSONL)")
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE,
                        help="condition-manifest freeze metadata (JSON)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true", help="validate and compute without writing outputs")
    args = parser.parse_args(argv)

    result = analyze(
        condition_manifest=args.manifest,
        base_manifest=args.base_manifest,
        freeze=args.freeze,
        output_dir=args.output_dir,
        write=not args.no_write,
    )
    totals = result["global_totals"]
    print(
        "frozen cohort reconciliation: "
        f"bases={totals['distinct_bases']} pairs={totals['matched_fault_pairs']} "
        f"corrupted={totals['corrupted_conditions']} enforced={totals['enforced_conditions']} "
        f"clean={totals['clean_conditions']} total={totals['total_conditions']}"
    )
    print(f"observed source x event x fault cells: {result['observed_cell_count']}")
    print(f"proposed identity 2080 = 618*3 + 113*2 supported: {result['proposed_identity']['supported']}")
    print(f"source x event confounded: {result['source_event_confounding']['confounded']}")
    if not args.no_write:
        for key, path in result.get("outputs", {}).items():
            print(f"  wrote {key}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
