#!/usr/bin/env python3
"""Analyze completed cheap-judge pilot results with paired block resampling."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_cheap_judge_pilot import CONDITIONS, PILOT_SEED, PROTOCOL, read_jsonl, write_json, write_jsonl


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def paired_metrics(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if not blocks:
        return {"n": 0, "clean_accuracy": None, "corrupted_accuracy": None, "enforced_accuracy": None,
                "clean_minus_corrupted": None, "enforced_minus_corrupted": None, "enforced_minus_clean": None}
    clean = statistics.mean(float(b["results"]["clean"]) for b in blocks)
    corrupted = statistics.mean(float(b["results"]["corrupted"]) for b in blocks)
    enforced = statistics.mean(float(b["results"]["enforced"]) for b in blocks)
    return {"n": len(blocks), "clean_accuracy": clean, "corrupted_accuracy": corrupted, "enforced_accuracy": enforced,
            "clean_minus_corrupted": clean - corrupted, "enforced_minus_corrupted": enforced - corrupted,
            "enforced_minus_clean": enforced - clean}


def paired_bootstrap(blocks: list[dict[str, Any]], seed: int = PILOT_SEED, iterations: int = 10_000) -> dict[str, list[float] | None]:
    if not blocks:
        return {k: None for k in ("clean_minus_corrupted", "enforced_minus_corrupted", "enforced_minus_clean")}
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        drawn = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        metric = paired_metrics(drawn)
        for key in ("clean_minus_corrupted", "enforced_minus_corrupted", "enforced_minus_clean"):
            samples[key].append(float(metric[key]))
    return {key: [percentile(values, 0.025), percentile(values, 0.975)] for key, values in samples.items()}


def combine_observation(answer: dict[str, Any], judge_results: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    if answer["evaluator_type"] == "native_deterministic":
        binary = {"correct": True, "abstention_correct": True, "incorrect": False, "abstention_incorrect": False}.get(answer["native_result"])
        return {**answer, "exploratory_binary_result": binary, "resolution": "native_deterministic"}
    values = []
    for provider in ("deepseek", "kimi"):
        row = judge_results.get((answer["observation_id"], provider))
        values.append(None if row is None else row["parsed_result"].get("exploratory_binary"))
    if values[0] is not None and values[0] == values[1]:
        result, resolution = values[0], "cheap_judge_agreement"
    else:
        result, resolution = None, "cheap_judge_disagreement_or_missing"
    return {**answer, "deepseek_binary": values[0], "kimi_binary": values[1],
            "exploratory_binary_result": result, "resolution": resolution}


def load_judge_results(private_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for provider in ("deepseek", "kimi"):
        path = private_root / provider / "judge_results.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            key = (row["observation_id"], provider)
            if key in result:
                raise ValueError(f"duplicate judge result: {key}")
            result[key] = row
    return result


def execution_audit(manifest: list[dict[str, Any]], private_root: Path) -> dict[str, Any]:
    known = {str(r["evaluation_request_id"]): r for r in manifest}
    if len(known) != len(manifest):
        raise ValueError("duplicate request IDs in frozen manifest")
    output = {}
    for provider in ("deepseek", "kimi"):
        planned = {rid for rid, row in known.items() if row["judge_provider"] == provider}
        ledger_path = private_root / provider / "judge_execution_ledger.jsonl"
        result_path = private_root / provider / "judge_results.jsonl"
        events = read_jsonl(ledger_path) if ledger_path.exists() else []
        results = read_jsonl(result_path) if result_path.exists() else []
        if any(str(e.get("evaluation_request_id")) not in planned for e in events):
            raise ValueError(f"{provider} ledger contains non-frozen request")
        result_ids = [str(r.get("evaluation_request_id")) for r in results]
        if any(rid not in planned for rid in result_ids):
            raise ValueError(f"{provider} results contain non-frozen request")
        duplicates = len(result_ids) - len(set(result_ids))
        histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            histories[str(event["evaluation_request_id"])].append(event)
        completed = {rid for rid, history in histories.items() if history[-1].get("state") == "completed"}
        failed = {rid for rid, history in histories.items() if history[-1].get("state") == "failed"}
        started_only = {
            rid for rid, history in histories.items()
            if history[-1].get("state") == "started"
        }
        if completed != set(result_ids):
            raise ValueError(f"{provider} completed ledger/results mismatch")
        usage = Counter()
        usage_reported = 0
        for result in results:
            item = result.get("provider_usage")
            if isinstance(item, dict):
                usage_reported += 1
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if isinstance(item.get(key), int):
                        usage[key] += item[key]
        output[provider] = {
            "planned": len(planned), "completed": len(completed), "failed": len(failed),
            "started_only": len(started_only), "pending": len(planned - set(histories)),
            "duplicates": duplicates, "usage_reported_n": usage_reported,
            "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        }
    return output


def analyze(answers: list[dict[str, Any]], judge_results: dict[tuple[str, str], dict[str, Any]], iterations: int = 10_000,
            execution: dict[str, Any] | None = None) -> dict[str, Any]:
    combined = [combine_observation(a, judge_results) for a in answers]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in combined:
        grouped[row["pilot_block_id"]].append(row)
    blocks = []
    for bid, rows in grouped.items():
        if len(rows) != 3 or {r["condition"] for r in rows} != set(CONDITIONS):
            raise ValueError(f"invalid matched block: {bid}")
        by_condition = {r["condition"]: r for r in rows}
        if all(by_condition[c]["exploratory_binary_result"] is not None for c in CONDITIONS):
            first = rows[0]
            blocks.append({
                "pilot_block_id": bid, "source": first["source"],
                "generating_provider": first["generating_provider"], "fault_type": first["fault_type"],
                "results": {c: bool(by_condition[c]["exploratory_binary_result"]) for c in CONDITIONS},
            })
    semantic = [r for r in combined if r["evaluator_type"] == "cheap_dual_semantic"]
    paired_valid = [r for r in semantic if r.get("deepseek_binary") is not None and r.get("kimi_binary") is not None]
    agreements = sum(r["deepseek_binary"] == r["kimi_binary"] for r in paired_valid)
    disagreements = len(paired_valid) - agreements
    overall = paired_metrics(blocks)
    overall["paired_bootstrap_95_ci"] = paired_bootstrap(blocks, PILOT_SEED, iterations)

    def breakdown(field: str) -> dict[str, Any]:
        values = {}
        for label in sorted({str(b[field]) for b in blocks}):
            subset = [b for b in blocks if str(b[field]) == label]
            values[label] = paired_metrics(subset)
        return values

    by_source = breakdown("source")
    by_provider = breakdown("generating_provider")
    by_fault = breakdown("fault_type")
    complete_rate = len(blocks) / len(grouped)
    agreement_rate = agreements / len(paired_valid) if paired_valid else 0.0
    cis = overall["paired_bootstrap_95_ci"]
    reversals = sum(
        value["n"] and (value["clean_minus_corrupted"] < 0 or value["enforced_minus_corrupted"] < 0)
        for table in (by_source, by_provider) for value in table.values()
    )
    if complete_rate < 0.70 or agreement_rate < 0.75:
        signal = "inconclusive"
    elif cis["clean_minus_corrupted"][0] > 0 and cis["enforced_minus_corrupted"][0] > 0 and reversals == 0:
        signal = "strong_positive"
    elif overall["clean_minus_corrupted"] > 0 and overall["enforced_minus_corrupted"] > 0:
        signal = "mixed" if reversals or cis["clean_minus_corrupted"][0] <= 0 or cis["enforced_minus_corrupted"][0] <= 0 else "strong_positive"
    elif abs(overall["clean_minus_corrupted"]) < 0.02 and abs(overall["enforced_minus_corrupted"]) < 0.02:
        signal = "weak"
    else:
        signal = "mixed"
    summary = {
        "protocol": PROTOCOL, "scientific_status": "exploratory_only",
        "selected_blocks": len(grouped), "complete_matched_triplets": len(blocks),
        "incomplete_matched_triplets": len(grouped) - len(blocks),
        "semantic_observations": len(semantic), "valid_cheap_judge_pairs": len(paired_valid),
        "cheap_judge_agreement_n": agreements, "cheap_judge_disagreement_n": disagreements,
        "cheap_judge_agreement_rate": agreement_rate,
        "overall": overall, "by_source": by_source, "by_generating_provider": by_provider,
        "by_fault_type": by_fault, "pilot_signal": signal,
        "decision_questions": {
            "corrupted_lower_than_clean": overall["clean_minus_corrupted"] > 0 if blocks else None,
            "enforced_higher_than_corrupted": overall["enforced_minus_corrupted"] > 0 if blocks else None,
            "enforced_approximately_returns_toward_clean": abs(overall["enforced_minus_clean"]) <= abs(overall["clean_minus_corrupted"]) if blocks else None,
            "major_source_provider_reversal_n": reversals,
        },
        "bootstrap": {"seed": PILOT_SEED, "iterations": iterations, "unit": "matched provider/base block"},
        "native_deterministic_observations": sum(r["evaluator_type"] == "native_deterministic" for r in combined),
        "unresolved_observations": sum(r["exploratory_binary_result"] is None for r in combined),
        "execution": execution,
        "api_calls_performed": 0,
    }
    return {"summary": summary, "combined": combined, "blocks": blocks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=Path("results/cheap_judge_pilot_v1/selected_answers.jsonl"))
    parser.add_argument("--private-root", type=Path, default=Path("results/b4_scoring_validation/private/cheap_judge_pilot_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("results/cheap_judge_pilot_v1/request_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/cheap_judge_pilot_v1/analysis"))
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args(argv)
    audit = execution_audit(read_jsonl(args.manifest), args.private_root)
    result = analyze(read_jsonl(args.answers), load_judge_results(args.private_root), args.bootstrap_iterations, audit)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "pilot_analysis_summary.json", result["summary"])
    write_jsonl(args.output / "resolved_observations.jsonl", result["combined"])
    write_jsonl(args.output / "complete_matched_blocks.jsonl", result["blocks"])
    summary = result["summary"]
    report = f"""# Cheap-judge exploratory pilot analysis

This is exploratory evidence from DeepSeek and Kimi, both of which failed the formal B4 evaluator-selection gate. It does not replace `final-native-evaluation-v2` and must not be used as final paper correctness evidence.

- Complete matched triplets: {summary['complete_matched_triplets']} / {summary['selected_blocks']}
- Cheap-judge agreement: {summary['cheap_judge_agreement_n']} / {summary['valid_cheap_judge_pairs']} ({summary['cheap_judge_agreement_rate']:.3f})
- Cheap-judge disagreements: {summary['cheap_judge_disagreement_n']}
- Pilot signal: `{summary['pilot_signal']}`

Effects and fixed-seed paired-bootstrap confidence intervals are recorded in `pilot_analysis_summary.json`. API calls performed by this analysis: 0.
"""
    (args.output / "PILOT_ANALYSIS_REPORT.md").write_text(report, encoding="utf-8")
    print(f"PILOT_SIGNAL={summary['pilot_signal']}")
    print(f"TOTAL_BLOCKS={summary['selected_blocks']}")
    print(f"COMPLETE_TRIPLETS={summary['complete_matched_triplets']}")
    print(f"SEMANTIC_OBSERVATIONS={summary['semantic_observations']}")
    print(f"NATIVE_DETERMINISTIC_OBSERVATIONS={summary['native_deterministic_observations']}")
    for provider in ("deepseek", "kimi"):
        x = audit[provider]
        prefix = provider.upper()
        for key in ("planned", "completed", "failed", "duplicates"):
            print(f"{prefix}_{key.upper()}={x[key]}")
        print(f"{prefix}_INPUT_TOKENS={x['input_tokens']}")
        print(f"{prefix}_OUTPUT_TOKENS={x['output_tokens']}")
    print(f"CHEAP_JUDGE_BINARY_AGREEMENT_N={summary['cheap_judge_agreement_n']}")
    print(f"CHEAP_JUDGE_BINARY_AGREEMENT_RATE={summary['cheap_judge_agreement_rate']}")
    print(f"CHEAP_JUDGE_DISAGREEMENT_N={summary['cheap_judge_disagreement_n']}")
    print(f"UNRESOLVED_OBSERVATIONS={summary['unresolved_observations']}")
    print(f"RESOLVED_MATCHED_BLOCKS={summary['complete_matched_triplets']}")
    overall = summary["overall"]
    print(f"CLEAN_ACCURACY={overall['clean_accuracy']}")
    print(f"CORRUPTED_ACCURACY={overall['corrupted_accuracy']}")
    print(f"ENFORCED_ACCURACY={overall['enforced_accuracy']}")
    print(f"CORRUPTION_EFFECT_PP={None if overall['clean_minus_corrupted'] is None else 100 * overall['clean_minus_corrupted']}")
    print(f"CORRUPTION_EFFECT_95CI={json.dumps(None if overall['paired_bootstrap_95_ci']['clean_minus_corrupted'] is None else [100*x for x in overall['paired_bootstrap_95_ci']['clean_minus_corrupted']])}")
    print(f"RECOVERY_EFFECT_PP={None if overall['enforced_minus_corrupted'] is None else 100 * overall['enforced_minus_corrupted']}")
    print(f"RECOVERY_EFFECT_95CI={json.dumps(None if overall['paired_bootstrap_95_ci']['enforced_minus_corrupted'] is None else [100*x for x in overall['paired_bootstrap_95_ci']['enforced_minus_corrupted']])}")
    print(f"RESTORATION_GAP_PP={None if overall['enforced_minus_clean'] is None else 100 * overall['enforced_minus_clean']}")
    print(f"RESTORATION_GAP_95CI={json.dumps(None if overall['paired_bootstrap_95_ci']['enforced_minus_clean'] is None else [100*x for x in overall['paired_bootstrap_95_ci']['enforced_minus_clean']])}")
    print(f"BY_SOURCE={json.dumps(summary['by_source'], sort_keys=True)}")
    print(f"BY_GENERATING_PROVIDER={json.dumps(summary['by_generating_provider'], sort_keys=True)}")
    print(f"BY_FAULT_TYPE={json.dumps(summary['by_fault_type'], sort_keys=True)}")
    print("API_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
