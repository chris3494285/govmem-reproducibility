#!/usr/bin/env python3
"""Offline same-base comparison of supported GovMem fault operators."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
ACCEPTED_MANIFEST_SHAS = {"58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8", "4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb"}
OUTPUT = ROOT / "results/matched_fault_support"
FAULTS = ("missing", "stale", "flipped", "conflicting")
PROVIDERS = {
    "deepseek": ROOT / "results/final_main_deepseek_v4_pro/scored_results.jsonl",
    "openai": ROOT / "results/final_main_openai_gpt56_sol_final_v2/scored/full_results_scored.jsonl",
    "anthropic": ROOT / "results/final_main_anthropic_sonnet5_final/scored/full_results_scored.jsonl",
    "gemini": ROOT / "results/final_main_gemini_3_flash_final/scored/full_results_scored.jsonl",
    "kimi": ROOT / "results/final_main_kimi_k2_6_nothinking_final/scored/full_results_scored.jsonl",
    "grok": ROOT / "results/final_main_grok_43_final/scored/full_results_scored.jsonl",
}


def _artifact_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _score(row: dict[str, Any]) -> int:
    if row.get("correctness") is True or row.get("primary_correctness") is True:
        return 1
    if row.get("correctness") is False or row.get("primary_score") == 0 or row.get("status") == "error":
        return 0
    raise ValueError(f"Result lacks deterministic score: {row.get('condition_id')}")


def _bootstrap(values: list[float], seed: int = 42, n: int = 10_000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        samples = rng.choice(np.asarray(values), size=(n, len(values)), replace=True).mean(axis=1)
        return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))
    except ImportError:  # pragma: no cover
        rng = random.Random(seed)
        samples = [mean(rng.choices(values, k=len(values))) for _ in range(n)]
        samples.sort()
        return samples[int(.025 * (n - 1))], samples[int(.975 * (n - 1))]


def _comparison(records: list[dict[str, Any]], comparator: str, *, provider: str, first: str = "stale") -> dict[str, Any]:
    values = [record["delta_stale"] - record[f"delta_{comparator}"] for record in records]
    stale_worse = sum(value < 0 for value in values)
    ties = sum(value == 0 for value in values)
    comparator_worse = sum(value > 0 for value in values)
    low, high = _bootstrap(values)
    return {"provider": provider, "fault_pair": f"stale_vs_{comparator}", "matched_base_count": len(values), "mean_difference_pp": mean(values) * 100 if values else None, "median_difference_pp": median(values) * 100 if values else None, "stale_worse_n": stale_worse, "stale_worse_fraction": stale_worse / len(values) if values else None, "tied_n": ties, "tied_fraction": ties / len(values) if values else None, "comparator_worse_n": comparator_worse, "comparator_worse_fraction": comparator_worse / len(values) if values else None, "bootstrap_ci_low_pp": low * 100 if low is not None else None, "bootstrap_ci_high_pp": high * 100 if high is not None else None, "bootstrap_seed": 42, "bootstrap_replicates": 10_000, "interpretation": "negative means stale causes greater degradation on the same base"}


def analyze(*, manifest: Path = MANIFEST, provider_paths: dict[str, Path] = PROVIDERS, output_dir: Path = OUTPUT) -> dict[str, Any]:
    import hashlib
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if manifest_sha not in ACCEPTED_MANIFEST_SHAS:
        raise ValueError(f"Frozen condition manifest SHA256 mismatch: {manifest_sha}")
    conditions = _jsonl(manifest)
    if len(conditions) != 4891 or len({row.get("condition_id") for row in conditions}) != 4891:
        raise ValueError("Frozen condition manifest must contain 4891 unique rows")
    bases: dict[str, dict[str, Any]] = {}
    support: dict[str, set[str]] = defaultdict(set)
    metadata: dict[str, dict[str, Any]] = {}
    for row in conditions:
        base = row["base_id"]
        metadata[base] = row
        if row["condition_type"] in {"corrupted", "enforced"}:
            support[base].add(row["fault_operator"])
        if row["condition_type"] == "clean":
            bases.setdefault(base, {})["clean"] = row
    overlap_rows = []
    for i, first in enumerate(FAULTS):
        for second in FAULTS[i + 1:]:
            matched = [base for base, ops in support.items() if first in ops and second in ops]
            overlap_rows.append({"fault_a": first, "fault_b": second, "matched_base_count": len(matched)})
            for dimension, values in (("event", {event: [b for b in matched if metadata[b]["analysis_event_type"] == event] for event in ("deletion", "temporal", "update")}), ("source", {source: [b for b in matched if metadata[b]["source_dataset"] == source] for source in ("longmemeval", "memora", "oaks_novel")})):
                for name, subset in values.items():
                    overlap_rows.append({"fault_a": first, "fault_b": second, "breakdown": dimension, "breakdown_value": name, "matched_base_count": len(subset)})

    indexes = {}
    expected = {row["condition_id"] for row in conditions}
    for provider, path in provider_paths.items():
        rows = _jsonl(path)
        indexes[provider] = {}
        for row in rows:
            cid = row.get("condition_id")
            if not cid or cid in indexes[provider] or cid not in expected:
                raise ValueError(f"Invalid, duplicate, or unknown {provider} condition_id: {cid}")
            indexes[provider][cid] = row
        if set(indexes[provider]) != expected:
            raise ValueError(f"{provider} results do not exactly match the frozen manifest")

    instances = []
    comparisons = []
    for provider, result_index in indexes.items():
        for base, ops in sorted(support.items()):
            clean = bases[base]["clean"]
            for first, second in (("stale", "missing"), ("stale", "flipped"), ("stale", "conflicting")):
                if first not in ops or second not in ops:
                    continue
                scores = {"clean": _score(result_index[clean["condition_id"]])}
                for fault in (first, second):
                    cid = f"{base}::{fault}::corrupted"
                    scores[fault] = _score(result_index[cid])
                deltas = {f"delta_{fault}": scores[fault] - scores["clean"] for fault in (first, second)}
                instances.append({"provider": provider, "base_id": base, "analysis_event_type": clean["analysis_event_type"], "source_dataset": clean["source_dataset"], "fault_a": first, "fault_b": second, "clean_score": scores["clean"], "score_a": scores[first], "score_b": scores[second], **deltas, "difference_stale_minus_comparator": deltas["delta_stale"] - deltas[f"delta_{second}"]})
        for pair in (("stale", "missing"), ("stale", "flipped"), ("stale", "conflicting")):
            first, second = pair
            records = [record for record in instances if record["provider"] == provider and record["fault_a"] == first and record["fault_b"] == second]
            comparisons.append(_comparison(records, second, provider=provider))

    by_dimension = {}
    for dimension, key in (("event", "analysis_event_type"), ("source", "source_dataset")):
        rows = []
        for provider in indexes:
            for pair in (("stale", "missing"), ("stale", "flipped"), ("stale", "conflicting")):
                for value in sorted({record[key] for record in instances}):
                    subset = [record for record in instances if record["provider"] == provider and record["fault_a"] == pair[0] and record["fault_b"] == pair[1] and record[key] == value]
                    if subset:
                        rows.append({**_comparison(subset, pair[1], provider=provider), "breakdown": dimension, "breakdown_value": value})
        by_dimension[dimension] = rows

    summary = {"status": "OFFLINE MATCHED-SUPPORT FAULT ANALYSIS", "manifest_sha256": manifest_sha, "base_count": 731, "fault_pair_count": 2080, "providers": sorted(indexes), "support_counts": {fault: sum(fault in ops for ops in support.values()) for fault in FAULTS}, "bases_by_supported_fault_count": {str(n): sum(len(ops) == n for ops in support.values()) for n in range(1, 5)}, "overlap_counts": overlap_rows, "matched_instance_count": len(instances), "comparisons": comparisons, "by_event": by_dimension["event"], "by_source": by_dimension["source"], "result_paths": {provider: _artifact_label(path) for provider, path in provider_paths.items()}, "api_calls_made": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_fields = list(dict.fromkeys(field for record in instances for field in record))
    with (output_dir / "matched_fault_instance_level.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=instance_fields); writer.writeheader(); writer.writerows(instances)
    (output_dir / "matched_fault_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, rows in (("provider", comparisons), ("event", by_dimension["event"]), ("source", by_dimension["source"]), ("fault", comparisons)):
        fields = list(rows[0]) if rows else []
        with (output_dir / f"matched_fault_by_{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    overlap_fields = list(dict.fromkeys(field for record in overlap_rows for field in record))
    with (output_dir / "matched_fault_overlap_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=overlap_fields); writer.writeheader(); writer.writerows(overlap_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--provider-result", action="append", nargs=2, metavar=("NAME", "PATH"), help="Complete sanitized result path; repeat for all providers")
    args = parser.parse_args()
    provider_paths = dict(args.provider_result) if args.provider_result else None
    summary = analyze(manifest=args.manifest, provider_paths={key: Path(value) for key, value in provider_paths.items()} if provider_paths else PROVIDERS, output_dir=args.output_dir)
    base_overlaps = {frozenset((row["fault_a"], row["fault_b"])): row["matched_base_count"] for row in summary["overlap_counts"] if "breakdown" not in row}
    print(json.dumps({"stale_vs_missing": base_overlaps[frozenset(("stale", "missing"))], "stale_vs_flipped": base_overlaps[frozenset(("stale", "flipped"))], "stale_vs_conflicting": base_overlaps[frozenset(("stale", "conflicting"))], "providers": summary["providers"], "matched_instances": summary["matched_instance_count"], "api_calls_made": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
