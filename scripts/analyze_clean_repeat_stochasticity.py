#!/usr/bin/env python3
"""Offline analysis of the CLEAN-A / CLEAN-B stochasticity control.

This is the smallest reproducible repair for the missing clean-repeat analysis
pipeline.  It performs **no** provider calls: it re-uses the frozen formal-300
evaluator resolution (`scripts.finalize_formal_300_validation.resolve`) and the
frozen paired-bootstrap method/seed (`BOOTSTRAP_SEED = 20261010`) so no metric is
redefined.  CLEAN-A correctness is read from the frozen manifest; CLEAN-B
correctness is resolved from the durable judge results, the offline deterministic
result, and the frozen semantic payloads.

Writes only into `results/stochasticity_control/clean_repeat_v1/analysis/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.finalize_formal_300_validation import BOOTSTRAP_SEED, resolve
from scripts.prepare_cheap_judge_pilot import read_jsonl
from scripts.run_clean_repeat_stochasticity import (
    PRIVATE, build_judge_plan, build_offline_deterministic_results, load_frozen_lme_source,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stochasticity_control/clean_repeat_v1"
FLOOR = 1e-12
# Frozen formal-300 comparator (results/.../formal_300_validation_v1/final_analysis).
FORMAL_COMPARATOR = {
    "protocol": "formal-300-validation-v1",
    "corruption_effect_pp": 35.536290041409735,
    "corruption_effect_ci95_pp": [29.78457785625372, 41.11590945899807],
    "recovery_effect_pp": 32.67353221505277,
    "recovery_effect_ci95_pp": [26.935454755810795, 38.363827287715324],
    "bootstrap_iterations": 10000,
    "bootstrap_seed": 20261010,
}


def _resolve_clean_b(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve CLEAN-B correctness with the frozen formal-300 evaluator logic."""
    manifest = read_jsonl(out / "manifest.jsonl")
    generation_results = {r["repeat_block_id"]: r for r in read_jsonl(out / "generation_results.jsonl")}
    payloads = {r["observation_id"]: r for r in read_jsonl(PRIVATE / "semantic_payloads.jsonl")}
    plan = build_judge_plan(manifest, generation_results, payloads)
    dynamic = read_jsonl(out / "judging/dynamic_judge_plan.jsonl")
    results_index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(out / "judging/judge_results.jsonl"):
        key = (str(row["clean_A_observation_id"]), str(row["judge_slot"]))
        if key in results_index:
            raise ValueError(f"duplicate durable result: {key}")
        results_index[key] = row
    offline = build_offline_deterministic_results(manifest, generation_results, load_frozen_lme_source())
    deterministic = [{"observation_id": d["clean_A_observation_id"], "binary_correct": d["clean_B_binary_correct"]}
                     for d in offline]
    observations = [{
        "observation_id": b["clean_A_observation_id"], "source": b["source"], "condition": "clean",
        "pilot_block_id": b["formal300_block_id"], "generating_provider": b["generating_provider"],
        "fault_type": b["selected_fault"],
    } for b in manifest]
    resolved, counters = resolve(observations, deterministic, payloads, results_index, plan + dynamic, {})
    by_obs = {r["observation_id"]: r for r in resolved}
    rows: list[dict[str, Any]] = []
    for block in manifest:
        obs = by_obs[block["clean_A_observation_id"]]
        rows.append({
            "repeat_block_id": block["repeat_block_id"], "source": block["source"],
            "generating_provider": block["generating_provider"], "selected_fault": block["selected_fault"],
            "clean_A_route": block["clean_A_route"],
            "clean_A_correctness": float(block["clean_A_correctness"]),
            "clean_B_correctness": None if obs["formal_score"] is None else float(obs["formal_score"]),
            "clean_B_resolution_method": obs["resolution_method"],
        })
    return rows, {"evaluator_counts": dict(counters), "n_blocks": len(rows),
                  "n_resolved_clean_b": sum(1 for r in rows if r["clean_B_correctness"] is not None)}


def _paired_bootstrap_ci(values: list[float], iterations: int = FORMAL_COMPARATOR["bootstrap_iterations"],
                         seed: int = BOOTSTRAP_SEED) -> list[float]:
    """95% paired percentile bootstrap CI (frozen method: resample blocks, fixed seed)."""
    rng = random.Random(seed)
    samples: list[float] = []
    n = len(values)
    for _ in range(iterations):
        drawn = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(statistics.mean(drawn))

    def percentile(ordered: list[float], q: float) -> float:
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    ordered = sorted(samples)
    return [percentile(ordered, 0.025), percentile(ordered, 0.975)]


def _subset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if r["clean_B_correctness"] is not None]
    a = [r["clean_A_correctness"] for r in usable]
    b = [r["clean_B_correctness"] for r in usable]
    deltas = [x - y for x, y in zip(a, b)]
    mean_a = statistics.mean(a) if a else None
    mean_b = statistics.mean(b) if b else None
    disagreement = sum(1 for x, y in zip(a, b) if abs(x - y) > FLOOR)
    transitions = {"correct_correct": 0, "correct_incorrect": 0, "incorrect_correct": 0, "incorrect_incorrect": 0}
    binary = all(abs(v) < FLOOR or abs(v - 1.0) < FLOOR for v in a + b)
    if binary:
        for x, y in zip(a, b):
            key = ("correct" if x > 0.5 else "incorrect") + "_" + ("correct" if y > 0.5 else "incorrect")
            transitions[key] += 1
    return {
        "n": len(usable),
        "excluded_unresolved": len(rows) - len(usable),
        "mean_clean_A": mean_a, "mean_clean_B": mean_b,
        "paired_difference_pp": None if mean_a is None else (mean_a - mean_b) * 100.0,
        "absolute_gap_pp": None if mean_a is None else abs(mean_a - mean_b) * 100.0,
        "agreement_rate": None if not usable else 1.0 - disagreement / len(usable),
        "disagreement_count": disagreement,
        "disagreement_rate": None if not usable else disagreement / len(usable),
        "binary_outcomes": binary,
        "transitions": transitions if binary else None,
        "paired_difference_ci95_pp": None if not deltas else [v * 100.0 for v in _paired_bootstrap_ci(deltas)],
        "max_abs_delta": None if not deltas else max(abs(d) for d in deltas),
        "mean_abs_delta": None if not deltas else statistics.mean(abs(d) for d in deltas),
    }


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row[key])].append(row)
    return {k: _subset_metrics(v) for k, v in sorted(buckets.items())}


def analyze(out: Path) -> dict[str, Any]:
    rows, resolution = _resolve_clean_b(out)
    overall = _subset_metrics(rows)
    by_source = _group(rows, "source")
    by_provider = _group(rows, "generating_provider")
    lme_rows = [r for r in rows if r["source"] == "longmemeval"]
    memora_rows = [r for r in rows if r["source"] == "memora"]
    return {
        "protocol": "clean-repeat-v1",
        "evaluator": "final-native-evaluation-v2",
        "unit": "paired CLEAN-A / CLEAN-B block",
        "n_blocks": len(rows),
        "resolution": resolution,
        "overall": overall,
        "longmemeval": _subset_metrics(lme_rows),
        "memora": _subset_metrics(memora_rows),
        "by_source": by_source,
        "by_generating_provider": by_provider,
        "bootstrap": {"iterations": FORMAL_COMPARATOR["bootstrap_iterations"], "seed": BOOTSTRAP_SEED,
                      "unit": "paired block", "method": "percentile"},
        "formal_300_comparator": FORMAL_COMPARATOR,
        "scoring_protocol": ("LongMemEval dual primaries (OpenAI gpt-5.6-sol + Anthropic claude-sonnet-5) "
                             "with Gemini escalation on disagreement/non-binary; Memora Gemini criterion "
                             "primary with native FAMA; frozen native-deterministic route for the single "
                             "deterministic block. No obsolete unified deterministic scorer."),
        "blocks": rows,
        "api_calls_performed": 0,
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _paper_table(metrics: dict[str, Any]) -> str:
    overall = metrics["overall"]
    lme = metrics["longmemeval"]
    memora = metrics["memora"]
    ci = overall["paired_difference_ci95_pp"]
    cmp_ci = FORMAL_COMPARATOR["corruption_effect_ci95_pp"]

    def ci_cell(values: list[float] | None) -> str:
        return "NA" if not values else f"[{values[0]:+.2f}, {values[1]:+.2f}]"

    lines = [
        "% Auto-generated by scripts/analyze_clean_repeat_stochasticity.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{CLEAN-A vs CLEAN-B repeat-generation stochasticity control "
        "($60$ paired blocks; frozen \\texttt{final-native-evaluation-v2} evaluator). "
        "Correctness is native FAMA for Memora and binary for LongMemEval. "
        "The final row is the frozen formal-300 corruption effect, shown only for scale.}",
        "\\label{tab:clean-repeat-stochasticity}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Subset & $n$ & Clean-A & Clean-B & $\\Delta$ (pp) [95\\% CI] \\\\",
        "\\midrule",
        f"All blocks & {overall['n']} & {_fmt(overall['mean_clean_A'])} & {_fmt(overall['mean_clean_B'])} & "
        f"{overall['paired_difference_pp']:+.2f} {ci_cell(ci)} \\\\",
        f"LongMemEval (binary) & {lme['n']} & {_fmt(lme['mean_clean_A'])} & {_fmt(lme['mean_clean_B'])} & "
        f"{lme['paired_difference_pp']:+.2f} {ci_cell(lme['paired_difference_ci95_pp'])} \\\\",
        f"Memora (FAMA) & {memora['n']} & {_fmt(memora['mean_clean_A'])} & {_fmt(memora['mean_clean_B'])} & "
        f"{memora['paired_difference_pp']:+.2f} {ci_cell(memora['paired_difference_ci95_pp'])} \\\\",
        "\\midrule",
        f"LongMemEval binary flips & {lme['disagreement_count']}/{lme['n']} & \\multicolumn{{3}}{{l}}"
        f"{{agreement {_fmt(lme['agreement_rate'])}}} \\\\",
        f"Memora exact-equal FAMA & {memora['n'] - memora['disagreement_count']}/{memora['n']} & "
        f"\\multicolumn{{3}}{{l}}{{mean $|\\Delta|$ {_fmt(memora['mean_abs_delta'])}}} \\\\",
        "\\midrule",
        f"\\textit{{Formal-300 corruption effect}} & 298 & 0.6903 & 0.3349 & "
        f"{FORMAL_COMPARATOR['corruption_effect_pp']:+.2f} {ci_cell(cmp_ci)} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    metrics = analyze(args.output)
    analysis_dir = args.output / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "clean_repeat_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (analysis_dir / "clean_repeat_block_level.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["repeat_block_id", "source", "generating_provider", "selected_fault",
                         "clean_A_route", "clean_A_correctness", "clean_B_correctness",
                         "clean_B_resolution_method"])
        for row in metrics["blocks"]:
            writer.writerow([row["repeat_block_id"], row["source"], row["generating_provider"],
                             row["selected_fault"], row["clean_A_route"], row["clean_A_correctness"],
                             row["clean_B_correctness"], row["clean_B_resolution_method"]])
    (analysis_dir / "clean_repeat_paper_table.tex").write_text(_paper_table(metrics), encoding="utf-8")

    overall = metrics["overall"]
    lme = metrics["longmemeval"]
    memora = metrics["memora"]
    print(f"CLEAN_REPEAT_BLOCKS={metrics['n_blocks']}")
    print(f"RESOLVED_CLEAN_B={metrics['resolution']['n_resolved_clean_b']}")
    print(f"MEAN_CLEAN_A={_fmt(overall['mean_clean_A'])}")
    print(f"MEAN_CLEAN_B={_fmt(overall['mean_clean_B'])}")
    print(f"PAIRED_DIFFERENCE_PP={overall['paired_difference_pp']:+.4f}")
    print(f"PAIRED_DIFFERENCE_CI95_PP=[{overall['paired_difference_ci95_pp'][0]:+.4f},"
          f"{overall['paired_difference_ci95_pp'][1]:+.4f}]")
    print(f"ABSOLUTE_GAP_PP={overall['absolute_gap_pp']:.4f}")
    print(f"LME_FLIPS={lme['disagreement_count']}/{lme['n']}")
    print(f"LME_AGREEMENT={_fmt(lme['agreement_rate'])}")
    print(f"MEMORA_EXACT_EQUAL={memora['n'] - memora['disagreement_count']}/{memora['n']}")
    print(f"MEMORA_MEAN_ABS_DELTA={_fmt(memora['mean_abs_delta'])}")
    print(f"FORMAL300_CORRUPTION_EFFECT_PP={FORMAL_COMPARATOR['corruption_effect_pp']:+.4f}")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
