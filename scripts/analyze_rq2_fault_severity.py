#!/usr/bin/env python3
"""Matched, deterministic RQ2 fault-severity analysis over scored RQ1 rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


EXPECTED_PROVIDERS = ("deepseek", "openai", "anthropic")
RQ1A_FILENAME = "rq1a_scored.jsonl"
RQ1B_FILENAME = "rq1b_restrictive_scored.jsonl"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260824


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Referenced scored RQ1 output is missing: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Scored RQ1 row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def _fault_key(row: dict[str, Any]) -> tuple[str, str] | None:
    fault = row.get("fault_spec") or {}
    if row.get("condition_type") == "clean" or not fault:
        return None
    operator = fault.get("operator")
    direction = fault.get("direction")
    if operator is None or direction is None:
        raise ValueError(f"Fault row lacks operator/direction: {row.get('run_id')}")
    return str(operator), str(direction)


def _scope(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("source_dataset")), str(row.get("dimension"))


def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[tuple[str, str] | None, dict[str, dict[str, Any]]]]:
    indexed: dict[tuple[str, str], dict[tuple[str, str] | None, dict[str, dict[str, Any]]]] = {}
    seen: set[str] = set()
    for row in rows:
        run_id = str(row.get("run_id"))
        if run_id in seen:
            raise ValueError(f"Duplicate scored RQ1 run_id: {run_id}")
        seen.add(run_id)
        base = row.get("example_id")
        if base is None:
            raise ValueError(f"Scored RQ1 row lacks example_id: {run_id}")
        key = _fault_key(row)
        bucket = indexed.setdefault(_scope(row), {}).setdefault(key, {})
        base = str(base)
        if base in bucket:
            raise ValueError(f"Duplicate base/fault condition in scored RQ1 rows: {base}, {key}")
        bucket[base] = row
    return indexed


def _cohort_signature(indexed: dict[tuple[str, str], dict[tuple[str, str] | None, dict[str, dict[str, Any]]]]) -> set[tuple[Any, ...]]:
    signature = set()
    for scope, faults in indexed.items():
        for fault_key, bases in faults.items():
            signature.update((scope[0], scope[1], fault_key, base) for base in bases)
    return signature


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _bootstrap_ci(values: list[float], *, comparison_key: str) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    digest = hashlib.sha256(comparison_key.encode("utf-8")).digest()
    seed = BOOTSTRAP_SEED ^ int.from_bytes(digest[:8], "big")
    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def _direction_counts(values: list[float]) -> tuple[int, int, int]:
    return sum(value > 0 for value in values), sum(value == 0 for value in values), sum(value < 0 for value in values)


def _matched_rows(
    provider: str,
    scope: tuple[str, str],
    fault_a: tuple[str, str],
    fault_b: tuple[str, str],
    faults: dict[tuple[str, str] | None, dict[str, dict[str, Any]]],
    metric: str,
) -> tuple[list[dict[str, Any]], int]:
    clean = faults.get(None, {})
    a_rows = faults.get(fault_a, {})
    b_rows = faults.get(fault_b, {})
    bases = sorted(set(clean) & set(a_rows) & set(b_rows))
    values = []
    for base in bases:
        clean_value = clean[base].get(metric)
        a_value = a_rows[base].get(metric)
        b_value = b_rows[base].get(metric)
        if not all(isinstance(value, bool) for value in (clean_value, a_value, b_value)):
            continue
        if metric == "governance_violation":
            delta_a = int(a_value) - int(clean_value)
            delta_b = int(b_value) - int(clean_value)
            values.append({"base": base, "delta_a": delta_a, "delta_b": delta_b, "difference": delta_a - delta_b})
        else:
            harm_a = -(int(a_value) - int(clean_value))
            harm_b = -(int(b_value) - int(clean_value))
            values.append({"base": base, "harm_a": harm_a, "harm_b": harm_b, "difference": harm_a - harm_b})
    return values, len(bases)


def _pair_row(provider: str, scope: tuple[str, str], a: tuple[str, str], b: tuple[str, str], faults: dict) -> dict[str, Any]:
    gvr_values, matched_n = _matched_rows(provider, scope, a, b, faults, "governance_violation")
    utility_values, utility_matched_n = _matched_rows(provider, scope, a, b, faults, "task_utility")
    # RQ2-A and RQ2-B use separate scorable cohorts; this keeps nulls and
    # denominators explicit rather than silently intersecting metrics.
    gvr_differences = [item["difference"] for item in gvr_values]
    utility_differences = [item["difference"] for item in utility_values]
    ci_low, ci_high = _bootstrap_ci(gvr_differences, comparison_key=f"{provider}|gvr|{scope}|{a}|{b}")
    utility_ci_low, utility_ci_high = _bootstrap_ci(utility_differences, comparison_key=f"{provider}|utility|{scope}|{a}|{b}")
    gvr_wins_a, gvr_ties, gvr_wins_b = _direction_counts(gvr_differences)
    utility_wins_a, utility_ties, utility_wins_b = _direction_counts(utility_differences)
    return {
        "analysis": "rq2a",
        "provider": provider,
        "source_dataset": scope[0],
        "governance_dimension": scope[1],
        "fault_a": a[0],
        "fault_a_direction": a[1],
        "fault_b": b[0],
        "fault_b_direction": b[1],
        "matched_base_n": matched_n,
        "scorable_matched_n": len(gvr_values),
        "mean_delta_gvr_fault_a": _mean([item["delta_a"] for item in gvr_values]),
        "mean_delta_gvr_fault_b": _mean([item["delta_b"] for item in gvr_values]),
        "mean_matched_difference": _mean(gvr_differences),
        "matched_difference_ci_low": ci_low,
        "matched_difference_ci_high": ci_high,
        "win_a_n": gvr_wins_a,
        "tie_n": gvr_ties,
        "win_b_n": gvr_wins_b,
        "utility_scorable_matched_n": len(utility_values),
        "mean_utility_harm_fault_a": _mean([item["harm_a"] for item in utility_values]),
        "mean_utility_harm_fault_b": _mean([item["harm_b"] for item in utility_values]),
        "mean_utility_matched_difference": _mean(utility_differences),
        "utility_difference_ci_low": utility_ci_low,
        "utility_difference_ci_high": utility_ci_high,
        "utility_win_a_n": utility_wins_a,
        "utility_tie_n": utility_ties,
        "utility_win_b_n": utility_wins_b,
        "gvr_scorable": bool(gvr_values),
        "utility_scorable": bool(utility_values),
        "gvr_unscorable_reason": None if gvr_values else "No matched bases had deterministic governance-violation outcomes",
        "utility_unscorable_reason": None if utility_values else "No matched bases had deterministic task-utility outcomes",
    }


def _rq2a(provider: str, indexed: dict) -> list[dict[str, Any]]:
    rows = []
    for scope, faults in sorted(indexed.items()):
        keys = sorted(key for key in faults if key is not None)
        for a, b in itertools.combinations(keys, 2):
            rows.append(_pair_row(provider, scope, a, b, faults))
    return rows


def _rq2b(provider: str, indexed: dict) -> list[dict[str, Any]]:
    requested = {
        ("memconflict", "temporal", "expired", "restrictive"),
        ("memora", "deletion", "flipped", "restrictive"),
    }
    rows = []
    for scope, faults in sorted(indexed.items()):
        for fault_key, fault_rows in sorted((key, value) for key, value in faults.items() if key is not None):
            label = (scope[0], scope[1], fault_key[0], fault_key[1])
            if label not in requested:
                continue
            clean = faults.get(None, {})
            bases = sorted(set(clean) & set(fault_rows))
            utility_deltas = []
            false_denial = []
            control_plane = []
            for base in bases:
                clean_utility = clean[base].get("task_utility")
                fault_utility = fault_rows[base].get("task_utility")
                if isinstance(clean_utility, bool) and isinstance(fault_utility, bool):
                    utility_deltas.append(int(fault_utility) - int(clean_utility))
                value = fault_rows[base].get("downstream_false_denial")
                if isinstance(value, bool):
                    false_denial.append(int(value))
                value = fault_rows[base].get("control_plane_false_rejection")
                if isinstance(value, bool):
                    control_plane.append(int(value))
            ci_low, ci_high = _bootstrap_ci(utility_deltas, comparison_key=f"{provider}|rq2b|{scope}|{fault_key}")
            rows.append({
                "analysis": "rq2b_descriptive",
                "provider": provider,
                "source_dataset": scope[0],
                "governance_dimension": scope[1],
                "fault_type": fault_key[0],
                "fault_direction": fault_key[1],
                "matched_base_n": len(bases),
                "scorable_matched_n": len(utility_deltas),
                "mean_delta_utility": _mean(utility_deltas),
                "mean_utility_harm": _mean([-value for value in utility_deltas]) if utility_deltas else None,
                "utility_difference_ci_low": ci_low,
                "utility_difference_ci_high": ci_high,
                "downstream_false_denial_rate": _mean(false_denial),
                "scored_downstream_false_denial_n": len(false_denial),
                "downstream_false_denial_denominator_n": len(false_denial),
                "fault_only_control_plane_false_rejection_rate": _mean(control_plane),
                "fault_only_control_plane_false_rejection_n": len(control_plane),
                "scorable": bool(utility_deltas or false_denial or control_plane),
                "unscorable_reason": None if (utility_deltas or false_denial or control_plane) else "No deterministic RQ1-B metric values were scored",
            })
    return rows


def _stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in ("source_dataset", "governance_dimension", "fault_a", "fault_a_direction", "fault_b", "fault_b_direction"))
        grouped.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        directions = [row["mean_matched_difference"] for row in values if row.get("gvr_scorable") and row.get("mean_matched_difference") is not None]
        a_n, tie_n, b_n = _direction_counts(directions)
        utility_directions = [
            row["mean_utility_matched_difference"]
            for row in values
            if row.get("utility_scorable") and row.get("mean_utility_matched_difference") is not None
        ]
        utility_a_n, utility_tie_n, utility_b_n = _direction_counts(utility_directions)
        output.append({
            "source_dataset": key[0], "governance_dimension": key[1],
            "fault_a": key[2], "fault_a_direction": key[3], "fault_b": key[4], "fault_b_direction": key[5],
            "providers_scorable_n": len(directions), "providers_a_gt_b_n": a_n,
            "providers_tied_n": tie_n, "providers_b_gt_a_n": b_n,
            "direction_unanimous": len(directions) > 0 and max(a_n, tie_n, b_n) == len(directions),
            "utility_providers_scorable_n": len(utility_directions),
            "utility_providers_a_gt_b_n": utility_a_n,
            "utility_providers_tied_n": utility_tie_n,
            "utility_providers_b_gt_a_n": utility_b_n,
            "utility_direction_unanimous": len(utility_directions) > 0 and max(utility_a_n, utility_tie_n, utility_b_n) == len(utility_directions),
        })
    return output


def analyze(provider_paths: dict[str, tuple[str | Path, str | Path]]) -> dict[str, Any]:
    missing = [provider for provider in EXPECTED_PROVIDERS if provider not in provider_paths]
    if missing:
        raise ValueError(f"Missing expected RQ1 providers: {', '.join(missing)}")
    extra = sorted(set(provider_paths) - set(EXPECTED_PROVIDERS))
    if extra:
        raise ValueError(f"Unexpected providers: {', '.join(extra)}")
    indexed_by_provider = {}
    source_files = {}
    for provider in EXPECTED_PROVIDERS:
        rq1a, rq1b = provider_paths[provider]
        rq1a_path, rq1b_path = Path(rq1a), Path(rq1b)
        a_index = _index_rows(_load_jsonl(rq1a_path))
        b_index = _index_rows(_load_jsonl(rq1b_path))
        indexed_by_provider[provider] = (a_index, b_index)
        source_files[provider] = {"rq1a": str(rq1a_path), "rq1b": str(rq1b_path)}
    for track_index, track_name in ((0, "RQ1-A"), (1, "RQ1-B")):
        signatures = {provider: _cohort_signature(indexes[track_index]) for provider, indexes in indexed_by_provider.items()}
        reference = signatures[EXPECTED_PROVIDERS[0]]
        for provider in EXPECTED_PROVIDERS[1:]:
            if signatures[provider] != reference:
                raise ValueError(f"{track_name} base/fault cohorts do not align: {EXPECTED_PROVIDERS[0]} vs {provider}")
    rq2a = [row for provider in EXPECTED_PROVIDERS for row in _rq2a(provider, indexed_by_provider[provider][0])]
    rq2b = [row for provider in EXPECTED_PROVIDERS for row in _rq2b(provider, indexed_by_provider[provider][1])]
    return {
        "analysis_version": "rq2-fault-severity-v1",
        "providers": list(EXPECTED_PROVIDERS),
        "source_files": source_files,
        "rq2a_matched_comparisons": rq2a,
        "rq2b_restrictive_descriptive": rq2b,
        "cross_provider_stability": _stability(rq2a),
    }


def _csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(summary["rq2a_matched_comparisons"]) + list(summary["rq2b_restrictive_descriptive"])
    stability = {
        tuple(item.get(field) for field in ("source_dataset", "governance_dimension", "fault_a", "fault_a_direction", "fault_b", "fault_b_direction")): item
        for item in summary["cross_provider_stability"]
    }
    for row in rows:
        if row["analysis"] != "rq2a":
            continue
        key = tuple(row.get(field) for field in ("source_dataset", "governance_dimension", "fault_a", "fault_a_direction", "fault_b", "fault_b_direction"))
        row.update({f"stability_{name}": value for name, value in stability.get(key, {}).items() if name not in key})
    return rows


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    keys = sorted({key for row in rows for key in row})
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="results/rq1")
    parser.add_argument("--output-dir", default="results/rq2")
    args = parser.parse_args(argv)
    root, output_dir = Path(args.input_root), Path(args.output_dir)
    paths = {provider: (root / provider / RQ1A_FILENAME, root / provider / RQ1B_FILENAME) for provider in EXPECTED_PROVIDERS}
    summary = analyze(paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rq2_fault_severity.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = _csv_rows(summary)
    write_csv(rows, output_dir / "rq2_fault_severity.csv")
    write_csv(summary["rq2a_matched_comparisons"], output_dir / "rq2_matched_comparisons.csv")
    print(json.dumps({"providers": summary["providers"], "rq2a_comparisons": len(summary["rq2a_matched_comparisons"]), "rq2b_descriptive": len(summary["rq2b_restrictive_descriptive"]), "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
