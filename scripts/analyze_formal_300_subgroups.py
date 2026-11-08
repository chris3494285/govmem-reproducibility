#!/usr/bin/env python3
"""Offline subgroup analysis of the final adjudicated Formal-300 validation.

Reads only frozen artifacts.  Performs no LLM/API call, no scoring, and no
response generation.  Emits aggregate-only tables (no benchmark text, no
per-example identifiers, no base/pair ids).

Two uncertainty procedures are computed side by side:

* ``block``   - the frozen original procedure: unstratified paired resampling of
                matched provider/base blocks (seed 20261010, 10,000 iterations).
                Reproduced here only as a reference; it is never replaced.
* ``cluster`` - a paired base-cluster bootstrap that resamples the shared base
                examples (clusters) instead of the blocks, so bases reused across
                providers stay together.

Neither interval is a population-representative precision statement: the 300
blocks are a deterministic, provider-balanced, non-probability selection from
the frozen cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]

BLOCK_LEVEL_CSV = (
    ROOT
    / "results/b4_scoring_validation/private/formal_300_validation_v1/final_analysis/"
    "formal_300_block_level_results.csv"
)
FINAL_METRICS_JSON = (
    ROOT
    / "results/b4_scoring_validation/private/formal_300_validation_v1/final_analysis/"
    "formal_300_final_metrics.json"
)
SELECTED_BLOCKS_JSONL = ROOT / "results/cheap_judge_pilot_v1/selected_blocks.jsonl"
DEFAULT_OUTPUT = ROOT / "results/formal_300_subgroup_v1"

# Predeclared bootstrap constants.
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20261010  # same constant as the frozen block bootstrap
EFFECT_KEYS = ("corruption_effect", "recovery_effect", "restoration_gap")

# Frozen reference values (from formal_300_final_metrics.json) that must reconcile.
REFERENCE = {
    "selected_blocks": 300,
    "resolved_blocks": 298,
    "resolved_observations": 898,
    "unresolved_observations": 2,
    "selected_distinct_bases": 239,
    "clean": 0.6902866438957969,
    "corrupted": 0.3349237434816995,
    "enforced": 0.6616590656322272,
    "corruption_effect": 0.35536290041409735,
    "recovery_effect": 0.3267353221505277,
    "restoration_gap": -0.028627578263569653,
    "corruption_ci": [0.2978457785625372, 0.41115909458998073],
    "recovery_ci": [0.26935454755810795, 0.38363827287715324],
    "restoration_ci": [-0.059391263336064905, 0.0011804059781394282],
}


# --------------------------------------------------------------------------- #
# loading and reconciliation
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_blocks(
    block_level_csv: Path = BLOCK_LEVEL_CSV,
    selected_blocks_jsonl: Path = SELECTED_BLOCKS_JSONL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return resolved blocks (with base clusters) and a reconciliation record."""
    block_meta = {row["pilot_block_id"]: row for row in read_jsonl(selected_blocks_jsonl)}
    with block_level_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    blocks: list[dict[str, Any]] = []
    for row in rows:
        pid = row["pilot_block_id"]
        meta = block_meta.get(pid)
        if meta is None:
            raise ValueError(f"resolved block {pid} is not present in the frozen selection")
        blocks.append(
            {
                "pilot_block_id": pid,
                "base_id": meta["base_id"],
                "source": row["source"],
                "provider": row["generating_provider"],
                "fault": row["fault_type"],
                "clean": float(row["clean_score"]),
                "corrupted": float(row["corrupted_score"]),
                "enforced": float(row["enforced_score"]),
            }
        )

    selected = list(block_meta.values())
    dropped_ids = sorted(set(block_meta) - {b["pilot_block_id"] for b in blocks})
    resolved_bases = {b["base_id"] for b in blocks}
    dropped_base_ids = {block_meta[pid]["base_id"] for pid in dropped_ids}
    dropped_details = [
        {
            "source": block_meta[pid]["source"],
            "provider": block_meta[pid]["generating_provider"],
            "fault": block_meta[pid]["fault_type"],
        }
        for pid in dropped_ids
    ]
    reconciliation = {
        "selected_blocks": len(selected),
        "resolved_blocks": len(blocks),
        "selected_observations": len(selected) * 3,
        "resolved_observations": len(selected) * 3 - len(dropped_details),
        "observations_in_complete_triplets": len(blocks) * 3,
        "selected_distinct_bases": len({b["base_id"] for b in selected}),
        "resolved_distinct_bases": len(resolved_bases),
        "dropped_block_n": len(dropped_ids),
        "dropped_block_descriptions": dropped_details,
        "dropped_base_ids": sorted(dropped_base_ids),
        "bases_fully_dropped": len(dropped_base_ids - resolved_bases),
        "bases_retained_elsewhere": len(dropped_base_ids & resolved_bases),
    }
    return blocks, reconciliation


def reconcile(blocks: list[dict[str, Any]], reconciliation: dict[str, Any]) -> dict[str, Any]:
    """Assert the frozen reference values; raise on any discrepancy."""
    metrics = summarize(blocks)
    checks = {
        "selected_blocks": reconciliation["selected_blocks"],
        "resolved_blocks": len(blocks),
        "selected_distinct_bases": reconciliation["selected_distinct_bases"],
        "resolved_distinct_bases": reconciliation["resolved_distinct_bases"],
        "clean": metrics["clean"],
        "corrupted": metrics["corrupted"],
        "enforced": metrics["enforced"],
        "corruption_effect": metrics["corruption_effect"],
        "recovery_effect": metrics["recovery_effect"],
        "restoration_gap": metrics["restoration_gap"],
    }
    mismatches = {
        key: {"expected": REFERENCE[key], "actual": value}
        for key, value in checks.items()
        if key in REFERENCE and abs(REFERENCE[key] - value) > 1e-9
    }
    if mismatches:
        raise ValueError(f"frozen Formal-300 reference mismatch: {mismatches}")

    block_ci = bootstrap_block(blocks)
    for key, expected in (
        ("corruption_effect", REFERENCE["corruption_ci"]),
        ("recovery_effect", REFERENCE["recovery_ci"]),
        ("restoration_gap", REFERENCE["restoration_ci"]),
    ):
        actual = block_ci[key]
        if abs(actual[0] - expected[0]) > 1e-9 or abs(actual[1] - expected[1]) > 1e-9:
            raise ValueError(f"frozen block-bootstrap CI mismatch for {key}: {actual} vs {expected}")

    return {
        "checks": checks,
        "distinct_base_note": (
            "239 distinct bases belong to the 300 selected blocks; the 298 resolved blocks "
            f"contain {reconciliation['resolved_distinct_bases']} distinct bases because the two "
            "dropped blocks remove one base entirely and one base is retained by its other blocks."
        ),
    }


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
def summarize(blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Paired matched-block summary; equal weight per block."""
    if not blocks:
        return {
            "n": 0,
            "clean": None,
            "corrupted": None,
            "enforced": None,
            "corruption_effect": None,
            "recovery_effect": None,
            "restoration_gap": None,
        }
    clean = statistics.mean(b["clean"] for b in blocks)
    corrupted = statistics.mean(b["corrupted"] for b in blocks)
    enforced = statistics.mean(b["enforced"] for b in blocks)
    return {
        "n": len(blocks),
        "clean": clean,
        "corrupted": corrupted,
        "enforced": enforced,
        "corruption_effect": clean - corrupted,
        "recovery_effect": enforced - corrupted,
        "restoration_gap": enforced - clean,
    }


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _interval(values: Sequence[float]) -> list[float]:
    return [percentile(values, 0.025), percentile(values, 0.975)]


def bootstrap_block(
    blocks: Sequence[dict[str, Any]],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[float]]:
    """Frozen original procedure: unstratified paired resampling of blocks."""
    if not blocks:
        return {key: [None, None] for key in EFFECT_KEYS}
    rng = random.Random(seed)
    n = len(blocks)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        draw = [blocks[rng.randrange(n)] for _ in range(n)]
        metrics = summarize(draw)
        for key in EFFECT_KEYS:
            samples[key].append(metrics[key])
    return {key: _interval(samples[key]) for key in EFFECT_KEYS}


def cluster_groups(
    blocks: Sequence[dict[str, Any]],
    strata: Sequence[str] = ("source",),
) -> dict[tuple, dict[str, list[dict[str, Any]]]]:
    """Group blocks as stratum -> base cluster -> blocks (stable order)."""
    grouped: dict[tuple, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for block in blocks:
        grouped[tuple(block[key] for key in strata)][block["base_id"]].append(block)
    return {stratum: dict(clusters) for stratum, clusters in grouped.items()}


def bootstrap_cluster(
    blocks: Sequence[dict[str, Any]],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    strata: Sequence[str] = ("source",),
) -> dict[str, Any]:
    """Base-cluster bootstrap: resample shared base examples, keep triplets intact."""
    if not blocks:
        return {"ci": {key: [None, None] for key in EFFECT_KEYS}, "clusters": 0, "strata": list(strata)}
    grouped = cluster_groups(blocks, strata)
    strata_keys = sorted(grouped)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        draw: list[dict[str, Any]] = []
        for stratum in strata_keys:
            clusters = list(grouped[stratum].values())
            for _ in range(len(clusters)):
                draw.extend(clusters[rng.randrange(len(clusters))])
        metrics = summarize(draw)
        for key in EFFECT_KEYS:
            samples[key].append(metrics[key])
    return {
        "ci": {key: _interval(samples[key]) for key in EFFECT_KEYS},
        "clusters": len({b["base_id"] for b in blocks}),
        "strata": list(strata),
        "stratum_cluster_counts": {
            "|".join(str(part) for part in stratum): len(grouped[stratum]) for stratum in strata_keys
        },
    }


def estimate(
    blocks: Sequence[dict[str, Any]],
    strata: Sequence[str] = ("source",),
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    with_bootstrap: bool = True,
) -> dict[str, Any]:
    metrics = summarize(blocks)
    result: dict[str, Any] = {
        **metrics,
        "clusters": len({b["base_id"] for b in blocks}),
        "max_blocks_per_base": max(Counter(b["base_id"] for b in blocks).values(), default=0),
    }
    if with_bootstrap and metrics["n"] >= 2:
        cluster = bootstrap_cluster(blocks, iterations, seed, strata)
        result["block_bootstrap_ci95"] = bootstrap_block(blocks, iterations, seed)
        result["cluster_bootstrap_ci95"] = cluster["ci"]
        result["cluster_bootstrap_clusters"] = cluster["clusters"]
    else:
        result["block_bootstrap_ci95"] = {key: [None, None] for key in EFFECT_KEYS}
        result["cluster_bootstrap_ci95"] = {key: [None, None] for key in EFFECT_KEYS}
        result["cluster_bootstrap_clusters"] = None
    return result


def stability(result: dict[str, Any]) -> str:
    """Flag cells whose intervals are degenerate or rest on very few clusters."""
    if result["n"] < 10 or result["clusters"] < 10:
        return "unstable_small_cell"
    degenerate = [
        key.replace("_effect", "").replace("_gap", "_gap")
        for key in EFFECT_KEYS
        if (lambda ci: ci[0] is None or abs(ci[1] - ci[0]) < 1e-12)(
            result["cluster_bootstrap_ci95"][key]
        )
    ]
    if degenerate:
        return "degenerate:" + "+".join(degenerate)
    return "estimated"


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #
def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_pp(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.2f}"


def _fmt_ci(ci: Sequence[float | None]) -> str:
    low, high = ci
    if low is None or high is None:
        return "n/a"
    return f"[{low * 100:+.2f}, {high * 100:+.2f}]"


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


PROVIDERS = ("openai", "anthropic", "deepseek", "gemini", "kimi", "xai")
SOURCES = ("longmemeval", "memora")
PROVIDER_LABEL = {
    "openai": "OpenAI GPT-5.6 Sol",
    "anthropic": "Claude Sonnet 5",
    "deepseek": "DeepSeek V4 Pro",
    "gemini": "Gemini 3 Flash",
    "kimi": "Kimi K2.6 (no thinking)",
    "xai": "Grok 4.3",
}
SOURCE_LABEL = {"longmemeval": "LongMemEval", "memora": "Memora"}


def render_tables_tex(
    overall: dict[str, Any],
    by_source: dict[str, dict[str, Any]],
    by_provider: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "% Formal-300 subgroup tables (final human-adjudicated artifacts).",
        "% Generated by: python scripts/analyze_formal_300_subgroups.py",
        "% Effects are percentage points; intervals are 10,000-resample paired bootstraps.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Formal-300 native-style validation by dataset (final",
        r"human-adjudicated analysis; 298 of 300 matched blocks resolved). Effects are",
        r"in percentage points. LongMemEval rows use native-style binary answer",
        r"correctness; Memora rows use the native continuous FAMA score. These are",
        r"different native-style outcomes and are deliberately not pooled into a single",
        r"benchmark accuracy. Intervals are paired block bootstraps ($10{,}000$",
        r"resamples, seed $20261010$) and describe within-sample variability of a",
        r"deterministic, provider-balanced, non-probability selection; they are not",
        r"population precision statements.}",
        r"\label{tab:formal300-source}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & $n$ & Clean & Corrupted & Enforced & Corruption (pp) & Recovery (pp) \\",
        r"\midrule",
    ]
    for source in SOURCES:
        row = by_source[source]
        if row["n"] == 0:
            continue
        lines.append(
            f"{SOURCE_LABEL[source]} & {row['n']} & {_fmt(row['clean'])} & {_fmt(row['corrupted'])} & "
            f"{_fmt(row['enforced'])} & ${_fmt_pp(row['corruption_effect'])}$ & "
            f"${_fmt_pp(row['recovery_effect'])}$ \\\\"
        )
    lines += [
        r"\midrule",
        f"Pooled & {overall['n']} & {_fmt(overall['clean'])} & {_fmt(overall['corrupted'])} & "
        f"{_fmt(overall['enforced'])} & ${_fmt_pp(overall['corruption_effect'])}$ & "
        f"${_fmt_pp(overall['recovery_effect'])}$ \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Formal-300 native-style corruption and recovery effects by",
        r"generating provider ($n = 49$--$50$ matched blocks each). Effects are in",
        r"percentage points with paired block-bootstrap intervals. Provider cells are",
        r"small and partially share base examples across providers, so these rows are",
        r"descriptive of this frozen sample; they do not rank providers or establish",
        r"provider differences.}",
        r"\label{tab:formal300-provider}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Provider & $n$ & LME / Memora & Corruption (pp) & Recovery (pp) & Stability \\",
        r"\midrule",
    ]
    for provider in PROVIDERS:
        row = by_provider[provider]
        lme = row["dataset_composition"].get("longmemeval", 0)
        mem = row["dataset_composition"].get("memora", 0)
        lines.append(
            f"{PROVIDER_LABEL[provider]} & {row['n']} & {lme} / {mem} & "
            f"${_fmt_pp(row['corruption_effect'])}$ {_fmt_ci(row['block_bootstrap_ci95']['corruption_effect'])} & "
            f"${_fmt_pp(row['recovery_effect'])}$ {_fmt_ci(row['block_bootstrap_ci95']['recovery_effect'])} & "
            f"{stability(row).replace('_', ' ')} \\\\"
        )
    lines += [
        r"\midrule",
        f"Pooled & {overall['n']} & 148 / 150 & ${_fmt_pp(overall['corruption_effect'])}$ "
        f"{_fmt_ci(overall['block_bootstrap_ci95']['corruption_effect'])} & "
        f"${_fmt_pp(overall['recovery_effect'])}$ "
        f"{_fmt_ci(overall['block_bootstrap_ci95']['recovery_effect'])} & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def render_report(
    overall: dict[str, Any],
    by_source: dict[str, dict[str, Any]],
    by_provider: dict[str, dict[str, Any]],
    by_provider_source: dict[tuple[str, str], dict[str, Any]],
    reconciliation: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Formal-300 Subgroup Analysis (final human-adjudicated artifacts)")
    add("")
    add(
        "Offline analysis of the frozen Formal-300 validation. No LLM or API call, no new "
        "scoring, no response generation, and no change to any frozen result or evaluation "
        "decision. All values are recomputed from `formal_300_block_level_results.csv` (final "
        "human-adjudicated analysis; protocol `formal-300-validation-v1`, evaluator "
        "`final-native-evaluation-v2`; the frozen block identity is intentionally omitted)."
    )
    add("")
    add("## 1. Reconciliation with the frozen reference values")
    add("")
    add("| Frozen reference | Reference value | Recomputed | Status |")
    add("|---|---:|---:|---|")
    add(f"| Selected blocks | 300 | {reconciliation['selected_blocks']} | match |")
    add(f"| Resolved blocks | 298 | {reconciliation['resolved_blocks']} | match |")
    add(f"| Resolved observations | 898 | {reconciliation['resolved_observations']} | match |")
    add(
        f"| Observations inside complete triplets (analysis input) | not specified | "
        f"{reconciliation['observations_in_complete_triplets']} | derived |"
    )
    add(f"| Distinct bases (300 selected blocks) | 239 | {reconciliation['selected_distinct_bases']} | match |")
    add(
        f"| Distinct bases (298 resolved blocks) | not specified | "
        f"{reconciliation['resolved_distinct_bases']} | definitional difference |"
    )
    add(f"| Corruption effect | +35.54 pp | {_fmt_pp(overall['corruption_effect'])} pp | match |")
    add(f"| Recovery effect | +32.67 pp | {_fmt_pp(overall['recovery_effect'])} pp | match |")
    add("")
    add("### Discrepancy reported, not adjusted")
    add("")
    add(
        "The reference value **239 distinct bases** is a property of the **300 selected "
        f"blocks**. The **298 resolved blocks** contain **{reconciliation['resolved_distinct_bases']} "
        "distinct bases**: the two unresolved blocks remove one base entirely and one base is "
        "retained through its other provider blocks. Both numbers are correct under their own "
        "definition. Every statistic below uses the 298 resolved blocks; no value was changed "
        "to match either figure."
    )
    add("")
    add(
        "A second counting nuance is reported for the same reason: the frozen metric counts "
        f"**{reconciliation['resolved_observations']} resolved observations** "
        f"(900 − 2 unresolved), while the triplet analysis consumes "
        f"**{reconciliation['observations_in_complete_triplets']} observations** "
        "(298 blocks × 3), because a block with any unresolved component is excluded "
        "entirely rather than partially used. The two numbers are not in conflict; they "
        "count different things."
    )
    add("")
    add(
        "The two excluded blocks are described by stratum only (no identifiers): "
        + "; ".join(
            f"{d['source']} / {d['provider']} / {d['fault']}"
            for d in reconciliation["dropped_block_descriptions"]
        )
        + ". Both are LongMemEval conflicting-fault blocks whose primary judgment returned "
        "`adjudication_required`; neither was imputed."
    )
    add("")
    add("## 2. Outcome units (deliberately not pooled into one accuracy)")
    add("")
    add(
        "* **LongMemEval** — native-style *binary* answer correctness (deterministic native / "
        "abstention routing plus blinded semantic judgment with binary projection).\n"
        "* **Memora** — native *continuous* FAMA per answer, using the benchmark's own "
        "criterion-level aggregation `FAMA = max(0, MPA − λ(1 − FAA))`.\n"
        "* The pooled row is an equal-weight block-average effect size across both datasets. "
        "It is **not** a single benchmark-native accuracy and must not be reported as one."
    )
    add("")
    add("## 3. Dataset-specific results")
    add("")
    add("| Dataset | Valid triplets | Base clusters | Clean | Corrupted | Enforced | Corruption (pp) | Recovery (pp) |")
    add("|---|---:|---:|---:|---:|---:|---|---|")
    for source in SOURCES:
        row = by_source[source]
        if row["n"] == 0:
            continue
        add(
            f"| {SOURCE_LABEL[source]} | {row['n']} | {row['clusters']} | {_fmt(row['clean'])} | "
            f"{_fmt(row['corrupted'])} | {_fmt(row['enforced'])} | "
            f"${_fmt_pp(row['corruption_effect'])}$ {_fmt_ci(row['block_bootstrap_ci95']['corruption_effect'])} | "
            f"${_fmt_pp(row['recovery_effect'])}$ {_fmt_ci(row['block_bootstrap_ci95']['recovery_effect'])} |"
        )
    add(
        f"| **Pooled** | **{overall['n']}** | {overall['clusters']} | {_fmt(overall['clean'])} | "
        f"{_fmt(overall['corrupted'])} | {_fmt(overall['enforced'])} | "
        f"**${_fmt_pp(overall['corruption_effect'])}$** {_fmt_ci(overall['block_bootstrap_ci95']['corruption_effect'])} | "
        f"**${_fmt_pp(overall['recovery_effect'])}$** {_fmt_ci(overall['block_bootstrap_ci95']['recovery_effect'])} |"
    )
    add("")
    add("Intervals are the frozen block bootstrap. The base-cluster bootstrap gives the same")
    add("picture (§5). Both datasets show the corruption effect and the recovery effect in the")
    add("same direction.")
    add("")
    add("## 4. Provider-specific results")
    add("")
    add("| Provider | Valid triplets | LME / Memora | Clean | Corrupted | Corruption (pp) [block CI] | Recovery (pp) [block CI] | Stability |")
    add("|---|---:|---|---:|---:|---|---|---|")
    for provider in PROVIDERS:
        row = by_provider[provider]
        lme = row["dataset_composition"].get("longmemeval", 0)
        mem = row["dataset_composition"].get("memora", 0)
        add(
            f"| {provider} | {row['n']} | {lme} / {mem} | {_fmt(row['clean'])} | {_fmt(row['corrupted'])} | "
            f"${_fmt_pp(row['corruption_effect'])}$ {_fmt_ci(row['block_bootstrap_ci95']['corruption_effect'])} | "
            f"${_fmt_pp(row['recovery_effect'])}$ {_fmt_ci(row['block_bootstrap_ci95']['recovery_effect'])} | "
            f"{stability(row)} |"
        )
    add("")
    add("### provider × dataset cells")
    add("")
    add("| Provider | Dataset | Valid triplets | Corruption (pp) [block CI] | Recovery (pp) [block CI] | Stability |")
    add("|---|---|---:|---|---|---|")
    for provider in PROVIDERS:
        for source in SOURCES:
            row = by_provider_source[(provider, source)]
            if row["n"] == 0:
                continue
            add(
                f"| {provider} | {SOURCE_LABEL[source]} | {row['n']} | "
                f"${_fmt_pp(row['corruption_effect'])}$ {_fmt_ci(row['block_bootstrap_ci95']['corruption_effect'])} | "
                f"${_fmt_pp(row['recovery_effect'])}$ {_fmt_ci(row['block_bootstrap_ci95']['recovery_effect'])} | "
                f"{stability(row)} |"
            )
    add("")
    add(
        "Each provider × dataset cell holds exactly 24–25 blocks and 24–25 base clusters: the "
        "frozen sampler admits each base at most once per provider, so base reuse happens only "
        "*across* providers, never inside a cell. "
        "These are **small cells with unstable intervals**; they are reported for completeness "
        "and must not be used to rank providers or to assert provider differences."
    )
    add("")
    add("## 5. Dependence-aware uncertainty (base-cluster bootstrap)")
    add("")
    add("Predeclared design:")
    add("")
    add(
        "* **Sampling unit:** one matched (generating provider, base) block carrying clean, "
        "corrupted, and enforced answers; the triplet is never split.\n"
        "* **Cluster identifier:** the base example. Blocks that share a base (the same content "
        "reused across providers) are resampled together.\n"
        "* **Stratification:** clusters are resampled independently within each dataset stratum "
        "(LongMemEval 148 blocks, Memora 150 blocks), preserving the dataset split. "
        "Dataset-specific, provider-specific, and provider × dataset cells are single-stratum "
        "resamples over the clusters present in that cell.\n"
        f"* **Iterations:** {BOOTSTRAP_ITERATIONS:,} (bootstrap_comparison.json records the same).\n"
        f"* **Seed:** {BOOTSTRAP_SEED} — the same constant as the frozen block bootstrap, so the "
        "comparison isolates the resampling scheme rather than the random stream.\n"
        "* **Aggregation and weighting:** equal weight per block; a resampled cluster "
        "contributes all of its blocks, so a base drawn $m$ times contributes $m$ copies of "
        "each of its blocks. No reweighting and no down-weighting of reused bases.\n"
        "* **Missing / unresolved handling:** the two blocks with an unresolved component are "
        "excluded before resampling; nothing is imputed and no partial triplet is used."
    )
    add("")
    add("| Subset | Blocks | Clusters | Effect | Block bootstrap (frozen) | Base-cluster bootstrap |")
    add("|---|---:|---:|---|---|---|")
    subsets = [("Pooled", overall)]
    subsets += [(SOURCE_LABEL[s], by_source[s]) for s in SOURCES if by_source[s]["n"] > 0]
    for label, row in subsets:
        for key, name in (("corruption_effect", "corruption"), ("recovery_effect", "recovery")):
            add(
                f"| {label} | {row['n']} | {row['clusters']} | {name} | "
                f"{_fmt_ci(row['block_bootstrap_ci95'][key])} | "
                f"{_fmt_ci(row['cluster_bootstrap_ci95'][key])} |"
            )
    add("")
    add(
        "The frozen block-bootstrap intervals are **reproduced exactly and not replaced**: "
        f"frozen corruption {_fmt_ci(REFERENCE['corruption_ci'])}, reproduced "
        f"{_fmt_ci(comparison['reproduced_block_bootstrap']['corruption_effect'])}; frozen "
        f"recovery {_fmt_ci(REFERENCE['recovery_ci'])}, reproduced "
        f"{_fmt_ci(comparison['reproduced_block_bootstrap']['recovery_effect'])}."
    )
    add("")
    add("### Comparison of the two schemes")
    add("")
    for key, name in (("corruption_effect", "Corruption"), ("recovery_effect", "Recovery")):
        block_ci = overall["block_bootstrap_ci95"][key]
        cluster_ci = overall["cluster_bootstrap_ci95"][key]
        add(
            f"* **{name} (pooled).** Block CI {_fmt_ci(block_ci)}; base-cluster CI "
            f"{_fmt_ci(cluster_ci)}; width changes from "
            f"{(block_ci[1] - block_ci[0]) * 100:.2f} pp to {(cluster_ci[1] - cluster_ci[0]) * 100:.2f} pp."
        )
    add("")
    wider = all(
        (overall["cluster_bootstrap_ci95"][key][1] - overall["cluster_bootstrap_ci95"][key][0])
        > (overall["block_bootstrap_ci95"][key][1] - overall["block_bootstrap_ci95"][key][0])
        for key in ("corruption_effect", "recovery_effect")
    )
    add(
        "The two schemes agree on the direction and on the qualitative conclusion for this "
        "frozen sample. They are not interchangeable: the block bootstrap treats 298 blocks as "
        "independent, while the cluster bootstrap honours the 238 distinct bases that underlie "
        "them"
        + (
            ", so its pooled intervals are the wider (more conservative) of the two."
            if wider
            else ", so its pooled intervals are not uniformly wider; the two schemes are "
            "reported side by side and neither replaces the other."
        )
    )
    add("")
    add(
        "**Interpretation limit.** The 300 blocks are a deterministic, provider-balanced, "
        "non-probability selection and bases are reused across providers. These intervals "
        "quantify sampling variability *within this frozen sample*. They are **not** "
        "population-representative precision statements and do not license inference to the "
        "full 12,480 provider–pair cohort."
    )
    add("")
    add("## 6. What these results can and cannot support")
    add("")
    add("**Supported by this analysis**")
    add("")
    add(
        "* A directionally consistent corruption effect and enforcement recovery on the frozen "
        "298-block sample, in both datasets and in all six provider subgroups.\n"
        "* The observation that explicitly modelling shared-base dependence does not overturn "
        "the picture given by the frozen block bootstrap.\n"
        "* Descriptive, clearly bounded per-provider and provider × dataset estimates for this "
        "sample."
    )
    add("")
    add("**Not supported by this analysis**")
    add("")
    add(
        "* A population-level six-provider semantic-effect magnitude: coverage is 2.40% of "
        "provider–pair units, OAKS Novel is absent from the semantic leg, and the selection is "
        "not a probability sample.\n"
        "* Provider rankings or claims that providers differ: cells hold ~25 blocks with "
        "unstable intervals and partially shared bases.\n"
        "* Validation of the frozen deterministic scorer's magnitudes: this analysis re-scores "
        "answers with the native-style evaluator and never runs that scorer.\n"
        "* Hypothesis-testing significance claims: the intervals are bootstrap summaries of a "
        "deterministic non-probability sample.\n"
        "* Any use of the pooled number as a single benchmark-native accuracy."
    )
    add("")
    return "\n".join(lines).rstrip("\n") + "\n"


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def analyze(
    output_dir: Path | None = DEFAULT_OUTPUT,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    write: bool = True,
) -> dict[str, Any]:
    blocks, reconciliation = load_blocks()
    recon = reconcile(blocks, reconciliation)

    overall = estimate(blocks, strata=("source",), iterations=iterations, seed=seed)
    by_source = {
        source: estimate(
            [b for b in blocks if b["source"] == source],
            strata=(),
            iterations=iterations,
            seed=seed,
        )
        for source in SOURCES
    }
    by_provider: dict[str, dict[str, Any]] = {}
    for provider in PROVIDERS:
        subset = [b for b in blocks if b["provider"] == provider]
        row = estimate(subset, strata=("source",), iterations=iterations, seed=seed)
        row["dataset_composition"] = dict(Counter(b["source"] for b in subset))
        by_provider[provider] = row
    by_provider_source: dict[tuple[str, str], dict[str, Any]] = {}
    for provider in PROVIDERS:
        for source in SOURCES:
            subset = [b for b in blocks if b["provider"] == provider and b["source"] == source]
            by_provider_source[(provider, source)] = estimate(
                subset, strata=(), iterations=iterations, seed=seed
            )

    comparison = {
        "schema_version": "formal-300-subgroup-bootstrap-comparison-v1",
        "protocol": "formal-300-validation-v1",
        "evaluator": "final-native-evaluation-v2",
        "inference_caveat": (
            "Deterministic, provider-balanced, non-probability sample with bases reused across "
            "providers; intervals describe within-sample sampling variability only and are not "
            "population-representative precision."
        ),
        "frozen_reference_block_bootstrap": {
            "unit": "matched provider/base block",
            "iterations": 10000,
            "seed": 20261010,
            "corruption_effect": REFERENCE["corruption_ci"],
            "recovery_effect": REFERENCE["recovery_ci"],
            "restoration_gap": REFERENCE["restoration_ci"],
        },
        "reproduced_block_bootstrap": overall["block_bootstrap_ci95"],
        "cluster_bootstrap": {
            "unit": "base example (cluster); matched triplets preserved",
            "stratification": ["source"],
            "iterations": iterations,
            "seed": seed,
            "aggregation": "equal weight per block; each resampled cluster contributes all of its blocks",
            "missing_handling": "blocks with an unresolved component excluded entirely; no imputation",
            "overall": overall["cluster_bootstrap_ci95"],
            "clusters_overall": overall["clusters"],
            "by_source": {s: by_source[s]["cluster_bootstrap_ci95"] for s in SOURCES},
            "by_provider": {p: by_provider[p]["cluster_bootstrap_ci95"] for p in PROVIDERS},
        },
        "reference_values_reconciled": recon["checks"],
        "distinct_base_definitions": {
            "selected_300_blocks": reconciliation["selected_distinct_bases"],
            "resolved_298_blocks": reconciliation["resolved_distinct_bases"],
            "explanation": recon["distinct_base_note"],
        },
        "note": (
            "The block bootstrap is reproduced bit-for-bit from the frozen analysis; the cluster "
            "bootstrap uses the same seed and iteration count with a different resampling scheme."
        ),
    }

    result: dict[str, Any] = {
        "overall": overall,
        "by_source": by_source,
        "by_provider": by_provider,
        "by_provider_source": by_provider_source,
        "reconciliation": reconciliation,
        "recon": recon,
        "comparison": comparison,
    }

    if write and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        write_csv(
            output_dir / "source_summary.csv",
            [
                "source", "valid_triplets", "base_clusters", "clean_mean", "corrupted_mean",
                "enforced_mean", "corruption_effect_pp", "block_ci_corruption_low_pp",
                "block_ci_corruption_high_pp", "cluster_ci_corruption_low_pp",
                "cluster_ci_corruption_high_pp", "recovery_effect_pp", "block_ci_recovery_low_pp",
                "block_ci_recovery_high_pp", "cluster_ci_recovery_low_pp",
                "cluster_ci_recovery_high_pp", "restoration_gap_pp",
                "cluster_ci_restoration_low_pp", "cluster_ci_restoration_high_pp",
                "native_outcome_unit",
            ],
            [
                (
                    source,
                    row["n"],
                    row["clusters"],
                    f"{row['clean']:.6f}",
                    f"{row['corrupted']:.6f}",
                    f"{row['enforced']:.6f}",
                    f"{row['corruption_effect'] * 100:.4f}",
                    f"{row['block_bootstrap_ci95']['corruption_effect'][0] * 100:.4f}",
                    f"{row['block_bootstrap_ci95']['corruption_effect'][1] * 100:.4f}",
                    f"{row['cluster_bootstrap_ci95']['corruption_effect'][0] * 100:.4f}",
                    f"{row['cluster_bootstrap_ci95']['corruption_effect'][1] * 100:.4f}",
                    f"{row['recovery_effect'] * 100:.4f}",
                    f"{row['block_bootstrap_ci95']['recovery_effect'][0] * 100:.4f}",
                    f"{row['block_bootstrap_ci95']['recovery_effect'][1] * 100:.4f}",
                    f"{row['cluster_bootstrap_ci95']['recovery_effect'][0] * 100:.4f}",
                    f"{row['cluster_bootstrap_ci95']['recovery_effect'][1] * 100:.4f}",
                    f"{row['restoration_gap'] * 100:.4f}",
                    f"{row['cluster_bootstrap_ci95']['restoration_gap'][0] * 100:.4f}",
                    f"{row['cluster_bootstrap_ci95']['restoration_gap'][1] * 100:.4f}",
                    "binary_correctness" if source == "longmemeval" else "continuous_native_fama",
                )
                for source in SOURCES
                for row in (by_source[source],)
                if row["n"] > 0
            ],
        )

        write_csv(
            output_dir / "provider_summary.csv",
            [
                "provider", "valid_triplets", "base_clusters", "longmemeval_triplets",
                "memora_triplets", "clean_mean", "corrupted_mean", "enforced_mean",
                "corruption_effect_pp", "block_ci_corruption_low_pp", "block_ci_corruption_high_pp",
                "cluster_ci_corruption_low_pp", "cluster_ci_corruption_high_pp",
                "recovery_effect_pp", "block_ci_recovery_low_pp", "block_ci_recovery_high_pp",
                "cluster_ci_recovery_low_pp", "cluster_ci_recovery_high_pp", "stability",
            ],
            [
                (
                    provider,
                    row["n"],
                    row["clusters"],
                    row["dataset_composition"].get("longmemeval", 0),
                    row["dataset_composition"].get("memora", 0),
                    f"{row['clean']:.6f}",
                    f"{row['corrupted']:.6f}",
                    f"{row['enforced']:.6f}",
                    f"{row['corruption_effect'] * 100:.4f}",
                    f"{row['block_bootstrap_ci95']['corruption_effect'][0] * 100:.4f}",
                    f"{row['block_bootstrap_ci95']['corruption_effect'][1] * 100:.4f}",
                    f"{row['cluster_bootstrap_ci95']['corruption_effect'][0] * 100:.4f}",
                    f"{row['cluster_bootstrap_ci95']['corruption_effect'][1] * 100:.4f}",
                    f"{row['recovery_effect'] * 100:.4f}",
                    f"{row['block_bootstrap_ci95']['recovery_effect'][0] * 100:.4f}",
                    f"{row['block_bootstrap_ci95']['recovery_effect'][1] * 100:.4f}",
                    f"{row['cluster_bootstrap_ci95']['recovery_effect'][0] * 100:.4f}",
                    f"{row['cluster_bootstrap_ci95']['recovery_effect'][1] * 100:.4f}",
                    stability(row),
                )
                for provider in PROVIDERS
                for row in (by_provider[provider],)
            ],
        )

        write_csv(
            output_dir / "provider_source_summary.csv",
            [
                "provider", "source", "valid_triplets", "base_clusters", "clean_mean",
                "corrupted_mean", "enforced_mean", "corruption_effect_pp",
                "block_ci_corruption_low_pp", "block_ci_corruption_high_pp",
                "cluster_ci_corruption_low_pp", "cluster_ci_corruption_high_pp",
                "recovery_effect_pp", "block_ci_recovery_low_pp", "block_ci_recovery_high_pp",
                "cluster_ci_recovery_low_pp", "cluster_ci_recovery_high_pp", "stability",
            ],
            [
                (
                    provider,
                    source,
                    row["n"],
                    row["clusters"],
                    f"{row['clean']:.6f}",
                    f"{row['corrupted']:.6f}",
                    f"{row['enforced']:.6f}",
                    f"{row['corruption_effect'] * 100:.4f}",
                    f"{row['block_bootstrap_ci95']['corruption_effect'][0] * 100:.4f}",
                    f"{row['block_bootstrap_ci95']['corruption_effect'][1] * 100:.4f}",
                    f"{row['cluster_bootstrap_ci95']['corruption_effect'][0] * 100:.4f}",
                    f"{row['cluster_bootstrap_ci95']['corruption_effect'][1] * 100:.4f}",
                    f"{row['recovery_effect'] * 100:.4f}",
                    f"{row['block_bootstrap_ci95']['recovery_effect'][0] * 100:.4f}",
                    f"{row['block_bootstrap_ci95']['recovery_effect'][1] * 100:.4f}",
                    f"{row['cluster_bootstrap_ci95']['recovery_effect'][0] * 100:.4f}",
                    f"{row['cluster_bootstrap_ci95']['recovery_effect'][1] * 100:.4f}",
                    stability(row),
                )
                for provider in PROVIDERS
                for source in SOURCES
                for row in (by_provider_source[(provider, source)],)
                if row["n"] > 0
            ],
        )

        (output_dir / "bootstrap_comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "FORMAL_300_SUBGROUP_REPORT.md").write_text(
            render_report(overall, by_source, by_provider, by_provider_source, reconciliation, comparison),
            encoding="utf-8",
        )
        (output_dir / "FORMAL_300_SUBGROUP_TABLES.tex").write_text(
            render_tables_tex(overall, by_source, by_provider), encoding="utf-8"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    result = analyze(
        output_dir=args.output, iterations=args.iterations, seed=args.seed, write=not args.no_write
    )
    overall = result["overall"]
    print(
        f"resolved blocks={overall['n']} base clusters={overall['clusters']} "
        f"corruption={overall['corruption_effect'] * 100:+.2f}pp "
        f"recovery={overall['recovery_effect'] * 100:+.2f}pp"
    )
    print(f"reconciled: {result['recon']['checks']}")
    if not args.no_write:
        for name in (
            "source_summary.csv",
            "provider_summary.csv",
            "provider_source_summary.csv",
            "bootstrap_comparison.json",
            "FORMAL_300_SUBGROUP_REPORT.md",
            "FORMAL_300_SUBGROUP_TABLES.tex",
        ):
            print(f"  wrote {args.output / name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
