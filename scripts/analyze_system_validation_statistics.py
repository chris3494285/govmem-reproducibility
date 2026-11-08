#!/usr/bin/env python3
"""Estimate uncertainty for frozen system-validation paired effects."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SYSTEMS = ("mem0", "langmem", "amem")
METRICS = ("governance_violation", "task_utility_success")
MANIFEST_SHA256 = "95abb36509873cbfec811f477291e2887def4b9a6ec2b04e766d27102ee09588"
PROMPT_TEMPLATE_SHA256 = "32506f36a07f6fb72bb643307002d7a086c5f1f3065e8884e6177139a54cd48e"


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_number}: {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Scored row on line {line_number} is not an object")
            rows.append(value)
    return rows


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def bootstrap_mean(values: list[float], *, replicates: int = 10_000, seed: int = 42) -> tuple[float | None, float | None]:
    """Bootstrap a list of base-level effects, never individual row pairs."""
    if not values:
        return None, None
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = random.Random(seed)
    samples = [mean(values[rng.randrange(len(values))] for _ in values) for _ in range(replicates)]
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


def exact_mcnemar_p_value(clean0_fault1: int, clean1_fault0: int) -> float:
    """Exact two-sided binomial test over McNemar discordant pairs."""
    n = clean0_fault1 + clean1_fault0
    if n == 0:
        return 1.0
    lower = sum(_binomial_probability(n, k) for k in range(0, clean0_fault1 + 1))
    upper = sum(_binomial_probability(n, k) for k in range(clean0_fault1, n + 1))
    return min(1.0, 2.0 * min(lower, upper))


def _binomial_probability(n: int, successes: int) -> float:
    import math
    return math.comb(n, successes) / (2 ** n)


def _validate_rows(rows: list[dict[str, Any]], *, strict_total: bool = False) -> None:
    if strict_total and len(rows) != 540:
        raise ValueError(f"Expected 540 scored rows, got {len(rows)}")
    expected_fields = ("memory_system", "base_example_id", "governance_dimension", "condition", "fault_type")
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if any(field not in row for field in expected_fields):
            raise ValueError("Scored row is missing system-validation identity fields")
        if row["condition"] == "clean":
            if row.get("fault_type") not in (None, "clean"):
                raise ValueError("Clean row has a non-clean fault_type")
            continue
        key = (str(row["memory_system"]), str(row["base_example_id"]), str(row["condition"]))
        if key in seen:
            raise ValueError(f"Duplicate scored condition: {key}")
        seen.add(key)
    if strict_total and {row["memory_system"] for row in rows} != set(SYSTEMS):
        raise ValueError("Scored output must contain exactly mem0, langmem, and amem")
    if strict_total:
        counts = {system: sum(row["memory_system"] == system for row in rows) for system in SYSTEMS}
        if counts != {system: 180 for system in SYSTEMS}:
            raise ValueError(f"Scored output must contain 180 rows per system: {counts}")


def _effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["memory_system"]), str(row["base_example_id"]))].append(row)
    effects = []
    for (system, base), group in sorted(groups.items()):
        clean = [row for row in group if row["condition"] == "clean"]
        if len(clean) != 1:
            raise ValueError(f"Expected exactly one clean sibling for {system}/{base}")
        clean = clean[0]
        for fault in sorted((row for row in group if row["condition"] != "clean"), key=lambda row: str(row["condition"])):
            effect = {
                "memory_system": system,
                "base_example_id": base,
                "source_dataset": fault.get("source_dataset"),
                "governance_dimension": fault.get("governance_dimension"),
                "fault_type": fault.get("fault_type") or fault.get("condition"),
            }
            for metric in METRICS:
                clean_value, fault_value = clean.get(metric), fault.get(metric)
                effect[f"clean_{metric}"] = clean_value
                effect[f"fault_{metric}"] = fault_value
                effect[f"delta_{metric}"] = (
                    int(fault_value) - int(clean_value)
                    if isinstance(clean_value, bool) and isinstance(fault_value, bool)
                    else None
                )
            effects.append(effect)
    return effects


def _transition_counts(effects: list[dict[str, Any]], metric: str) -> dict[str, int]:
    counts = {"clean0_fault0_n": 0, "clean0_fault1_n": 0, "clean1_fault0_n": 0, "clean1_fault1_n": 0}
    for row in effects:
        clean, fault = row[f"clean_{metric}"], row[f"fault_{metric}"]
        if isinstance(clean, bool) and isinstance(fault, bool):
            counts[f"clean{int(clean)}_fault{int(fault)}_n"] += 1
    return counts


def _system_statistics(effects: list[dict[str, Any]], *, replicates: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in effects:
        for metric in METRICS:
            groups[(row["memory_system"], row["governance_dimension"], row["fault_type"], metric)].append(row)
    output = []
    for (system, dimension, fault, metric), group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        deltas = [row[f"delta_{metric}"] for row in group if row[f"delta_{metric}"] is not None]
        low, high = bootstrap_mean(deltas, replicates=replicates, seed=seed)
        transitions = _transition_counts(group, metric)
        total = len(group)
        scored = len(deltas)
        row = {
            "memory_system": system, "governance_dimension": dimension, "fault_type": fault, "metric": metric,
            "total_base_n": total, "paired_scored_n": scored, "paired_unscored_n": total - scored,
            "paired_coverage_rate": scored / total if total else None,
            "mean_delta": mean(deltas) if deltas else None,
            "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            **transitions,
        }
        row["discordant_harm_n"] = transitions["clean0_fault1_n"]
        row["discordant_improvement_n"] = transitions["clean1_fault0_n"]
        output.append(row)
    return output


def _pooled_statistics(effects: list[dict[str, Any]], *, replicates: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    all_grouped: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in effects:
        for metric in METRICS:
            key = (row["governance_dimension"], row["fault_type"], metric)
            all_grouped[key].add(row["base_example_id"])
            delta = row[f"delta_{metric}"]
            if delta is not None:
                grouped[(row["governance_dimension"], row["fault_type"], metric)][row["base_example_id"]].append(delta)
    output = []
    for (dimension, fault, metric), base_ids in sorted(all_grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        by_base = grouped.get((dimension, fault, metric), {})
        clustered = [mean(values) for values in by_base.values()]
        low, high = bootstrap_mean(clustered, replicates=replicates, seed=seed)
        system_counts = [len({row["memory_system"] for row in effects if row["base_example_id"] == base and row[f"delta_{metric}"] is not None and row["governance_dimension"] == dimension and row["fault_type"] == fault}) for base in by_base]
        total_base = len(base_ids)
        output.append({
            "governance_dimension": dimension, "fault_type": fault, "metric": metric,
            "total_base_n": total_base, "clustered_base_n": len(clustered),
            "paired_unscored_n": total_base - len(clustered),
            "paired_coverage_rate": len(clustered) / total_base if total_base else None,
            "mean_clustered_delta": mean(clustered) if clustered else None,
            "clustered_bootstrap_ci_low": low, "clustered_bootstrap_ci_high": high,
            "systems_contributing_min": min(system_counts) if system_counts else 0,
            "systems_contributing_max": max(system_counts) if system_counts else 0,
            "mean_systems_per_base": mean(system_counts) if system_counts else None,
        })
    return output


def _mcnemar(effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in effects:
        groups[(row["memory_system"], row["governance_dimension"], row["fault_type"])].append(row)
    for (system, dimension, fault), group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        counts = _transition_counts(group, "governance_violation")
        output.append({"memory_system": system, "governance_dimension": dimension, "fault_type": fault, **counts,
                       "discordant_total_n": counts["clean0_fault1_n"] + counts["clean1_fault0_n"],
                       "exact_two_sided_p_value": exact_mcnemar_p_value(counts["clean0_fault1_n"], counts["clean1_fault0_n"])})
    return output


def _sign_consistency(system_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in system_rows:
        groups[(row["governance_dimension"], row["fault_type"], row["metric"])].append(row)
    output = []
    for (dimension, fault, metric), group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        item = {"governance_dimension": dimension, "fault_type": fault, "metric": metric}
        signs = []
        for system in SYSTEMS:
            found = next((row for row in group if row["memory_system"] == system), None)
            delta = found["mean_delta"] if found else None
            item[f"{system}_mean_delta"] = delta
            if delta is not None:
                signs.append(1 if delta > 0 else -1 if delta < 0 else 0)
        item["positive_system_n"] = sum(sign == 1 for sign in signs)
        item["negative_system_n"] = sum(sign == -1 for sign in signs)
        item["zero_system_n"] = sum(sign == 0 for sign in signs)
        item["scorable_system_n"] = len(signs)
        item["same_sign_all_systems"] = len(signs) == len(SYSTEMS) and len(set(signs)) == 1
        output.append(item)
    return output


def _headlines(system_rows: list[dict[str, Any]], pooled_rows: list[dict[str, Any]], mcnemar_rows: list[dict[str, Any]]) -> dict[str, Any]:
    requested = (
        ("authorization", "conflicting", "governance_violation", "GVR"),
        ("authorization", "flipped", "governance_violation", "GVR"),
        ("deletion", "flipped", "governance_violation", "GVR"),
        ("deletion", "stale", "governance_violation", "GVR"),
        ("temporal", "missing", "governance_violation", "GVR"),
        ("temporal", "stale", "governance_violation", "GVR"),
        ("temporal", "stale", "task_utility_success", "task_utility_success"),
    )
    result: dict[str, Any] = {}
    for dimension, fault, metric, label in requested:
        per_system = [row for row in system_rows if row["governance_dimension"] == dimension and row["fault_type"] == fault and row["metric"] == metric]
        pooled = next((row for row in pooled_rows if row["governance_dimension"] == dimension and row["fault_type"] == fault and row["metric"] == metric), None)
        test = next((row for row in mcnemar_rows if row["governance_dimension"] == dimension and row["fault_type"] == fault), None) if metric == "governance_violation" else None
        result[f"{dimension}/{fault}/{label}"] = {
            "per_system": per_system,
            "pooled_clustered": pooled,
            "exact_paired_test": test,
        }
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(rows: list[dict[str, Any]], *, replicates: int = 10_000, seed: int = 42, strict_total: bool = False) -> dict[str, Any]:
    _validate_rows(rows, strict_total=strict_total)
    effects = _effects(rows)
    by_system = _system_statistics(effects, replicates=replicates, seed=seed)
    pooled = _pooled_statistics(effects, replicates=replicates, seed=seed)
    consistency = _sign_consistency(by_system)
    mcnemar = _mcnemar(effects)
    return {
        "status": "FROZEN / FORMAL / STATISTICAL",
        "manifest_sha256": MANIFEST_SHA256,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "seed": seed,
        "bootstrap_replicates": replicates,
        "row_count": len(rows),
        "paired_effect_count": len(effects),
        "by_system": by_system,
        "pooled_clustered": pooled,
        "mcnemar": mcnemar,
        "sign_consistency": consistency,
        "headlines": _headlines(by_system, pooled, mcnemar),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/system_validation_analysis/system_validation_scored.jsonl")
    parser.add_argument("--output-dir", default="results/system_validation_statistics")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.replicates <= 0:
        parser.error("--replicates must be positive")
    rows = _jsonl(args.input)
    result = analyze(rows, replicates=args.replicates, seed=args.seed, strict_total=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "system_validation_statistics_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output / "system_validation_statistics_by_system.csv", result["by_system"])
    _write_csv(output / "system_validation_statistics_pooled.csv", result["pooled_clustered"])
    _write_csv(output / "system_validation_mcnemar.csv", result["mcnemar"])
    _write_csv(output / "system_validation_sign_consistency.csv", result["sign_consistency"])
    print(json.dumps({"row_count": result["row_count"], "paired_effect_count": result["paired_effect_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
