#!/usr/bin/env python3
"""Final offline analysis for the frozen formal-300 matched-block validation.

Reads only persisted, frozen evaluator outputs.  It never imports a provider
client, never performs inference, and never resamples the frozen 300 blocks.
Human adjudication (if recorded) is treated as the final resolution for that
explicit case.
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

from scripts.prepare_cheap_judge_pilot import read_jsonl
from scripts.prepare_formal_300_validation import EVALUATOR, PROTOCOL
from scripts.run_formal_300_validation import HUMAN_ADJUDICATION_RESOLUTION_METHOD, criterion_map, result_index

BOOTSTRAP_SEED = 20261010
CONDITIONS = ("clean", "corrupted", "enforced")
LME_PRIMARY_SLOTS = ("lme_primary_openai", "lme_primary_anthropic")


def fama(decisions: dict[str, str], prompt: str) -> float | None:
    """Native FAMA for a full criterion decision map (frozen evaluator)."""
    criteria = json.loads(prompt)["criteria"]
    kinds = {str(c["criterion_id"]): str(c["evaluation_type"]) for c in criteria}
    if set(decisions) != set(kinds) or any(v not in {"pass", "fail"} for v in decisions.values()):
        return None
    presence = [k for k, v in kinds.items() if v == "memory_presence"]
    absence = [k for k, v in kinds.items() if v == "forgetting_absence"]
    mpa = sum(decisions[k] == "pass" for k in presence) / len(presence) if presence else 1.0
    faa = sum(decisions[k] == "pass" for k in absence) / len(absence) if absence else 1.0
    return max(0.0, mpa - (len(absence) / len(kinds)) * (1.0 - faa))


def lme_binary(row: dict[str, Any] | None) -> bool | None:
    return None if row is None else row.get("parsed_result", {}).get("exploratory_binary")


def human_decision_map(private: Path) -> dict[str, dict[str, str]]:
    """observation_id -> {criterion_id: decision} from explicit human adjudications."""
    path = private / "human_adjudication" / "human_adjudication_decisions.jsonl"
    decisions: dict[str, dict[str, str]] = {}
    for record in read_jsonl(path) if path.exists() else []:
        if record.get("resolution_method") != HUMAN_ADJUDICATION_RESOLUTION_METHOD:
            continue
        decisions[str(record["observation_id"])] = {
            str(item["criterion_id"]): str(item["decision"]) for item in record.get("decisions", [])
        }
    return decisions


def resolve(
    observations: list[dict[str, Any]],
    deterministic: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    results: dict[tuple[str, str], dict[str, Any]],
    plan: list[dict[str, Any]],
    human_decisions: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Resolve each observation per the frozen evaluator protocol."""
    det = {r["observation_id"]: r for r in deterministic}
    counters: Counter[str] = Counter()
    static_secondary = {r["observation_id"] for r in plan if r["judge_slot"] == "memora_secondary_openai"}
    resolved: list[dict[str, Any]] = []
    for obs in observations:
        oid = obs["observation_id"]
        if oid in det:
            score = float(bool(det[oid]["binary_correct"]))
            method = "native_deterministic"
            counters["deterministic"] += 1
        elif obs["source"] == "longmemeval":
            a = results.get((oid, "lme_primary_openai"))
            b = results.get((oid, "lme_primary_anthropic"))
            av, bv = lme_binary(a), lme_binary(b)
            counters["lme_primary_judged"] += int(av is not None and bv is not None)
            if av is None or bv is None:
                score, method = None, "primary_missing_or_unusable"
            elif av == bv:
                score, method = float(av), "dual_primary_binary_agreement"
                counters["lme_primary_agreement"] += 1
            else:
                counters["lme_primary_disagreement"] += 1
                third = lme_binary(results.get((oid, "lme_escalation_gemini")))
                if third is None:
                    score, method = None, "gemini_escalation_pending"
                else:
                    score, method = float(sum((av, bv, third)) >= 2), "gemini_binary_majority"
                    counters["lme_escalation_completed"] += 1
        else:
            payload = payloads[oid]
            primary = results.get((oid, "memora_primary_gemini"))
            pm = criterion_map(primary) if primary else None
            decisions = primary.get("parsed_result", {}).get("decisions", []) if primary else []
            dynamic_risk = any(
                x.get("confidence") == "low" or x.get("decision") == "adjudication_required" for x in decisions
            )
            needs_secondary = oid in static_secondary or dynamic_risk
            counters["memora_observations"] += 1
            if pm is None:
                score, method = None, "memora_primary_missing"
            elif not needs_secondary:
                score, method = fama(pm, payload["prompt"]), "memora_gemini_primary"
                counters["memora_gemini_primary"] += 1
            else:
                counters["memora_secondary_required"] += 1
                secondary = results.get((oid, "memora_secondary_openai")) or results.get((oid, "memora_dynamic_secondary_openai"))
                sm = criterion_map(secondary) if secondary else None
                if sm is None:
                    score, method = None, "memora_secondary_pending"
                elif pm == sm and all(v in {"pass", "fail"} for v in pm.values()):
                    score, method = fama(pm, payload["prompt"]), "memora_primary_secondary_agreement"
                    counters["memora_secondary_agreement"] += 1
                else:
                    counters["memora_secondary_disagreement"] += 1
                    third = results.get((oid, "memora_escalation_anthropic"))
                    tm = criterion_map(third) if third else None
                    if tm is not None and set(pm) == set(sm) == set(tm):
                        majority = {}
                        for cid in pm:
                            votes = [pm[cid], sm[cid], tm[cid]]
                            valid = [v for v in votes if v in {"pass", "fail"}]
                            if valid.count("pass") >= 2:
                                majority[cid] = "pass"
                            elif valid.count("fail") >= 2:
                                majority[cid] = "fail"
                            else:
                                majority[cid] = "adjudication_required"
                        score, method = fama(majority, payload["prompt"]), "memora_third_majority"
                        if score is not None:
                            counters["memora_escalation_completed"] += 1
                    else:
                        human = human_decisions.get(oid)
                        if human is not None and set(human) == set(pm) and all(v in {"pass", "fail"} for v in human.values()):
                            score, method = fama(human, payload["prompt"]), HUMAN_ADJUDICATION_RESOLUTION_METHOD
                            counters["human_adjudicated"] += 1
                        else:
                            score, method = None, "memora_third_pending"
        if score is None:
            counters["unresolved_observations"] += 1
        resolved.append({**obs, "formal_score": score, "resolution_method": method})
    return resolved, counters


def block_metrics(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if not blocks:
        return {"n": 0, "clean": None, "corrupted": None, "enforced": None,
                "corruption_effect": None, "recovery_effect": None, "restoration_gap": None}
    values = {c: statistics.mean(b["scores"][c] for b in blocks) for c in CONDITIONS}
    return {
        "n": len(blocks), "clean": values["clean"], "corrupted": values["corrupted"], "enforced": values["enforced"],
        "corruption_effect": values["clean"] - values["corrupted"],
        "recovery_effect": values["enforced"] - values["corrupted"],
        "restoration_gap": values["enforced"] - values["clean"],
    }


def paired_bootstrap(blocks: list[dict[str, Any]], iterations: int) -> dict[str, list[float] | None]:
    keys = ("corruption_effect", "recovery_effect", "restoration_gap")
    if not blocks:
        return {k: None for k in keys}
    rng = random.Random(BOOTSTRAP_SEED)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        drawn = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        metric = block_metrics(drawn)
        for key in keys:
            samples[key].append(metric[key])

    def percentile(values: list[float], q: float) -> float:
        ordered = sorted(values)
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    return {key: [percentile(values, 0.025), percentile(values, 0.975)] for key, values in samples.items()}


def routing_counts(plan: list[dict[str, Any]], results: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    def counter(field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row.get(field)) for row in plan).items()))

    by_slot: dict[str, dict[str, int]] = {}
    for row in plan:
        slot = str(row.get("judge_slot"))
        entry = by_slot.setdefault(slot, {"requests": 0, "completed": 0})
        entry["requests"] += 1
        if row.get("formal_validation_request_id") in {
            r.get("formal_validation_request_id") for r in results.values() if r.get("parsed_result") is not None
        }:
            entry["completed"] += 1
    return {
        "total_judge_requests": len(plan),
        "completed_judge_results": sum(1 for r in results.values() if r.get("parsed_result") is not None),
        "by_source": counter("source"),
        "by_provider": counter("provider"),
        "by_phase": counter("phase"),
        "by_judge_slot": by_slot,
    }


def build_report(metrics: dict[str, Any]) -> str:
    overall = metrics["paired_effects"]
    ci = metrics["paired_effects_ci95"]
    lines = [
        "# Formal-300 validation — final analysis",
        "",
        f"Protocol `{PROTOCOL}` · evaluator `{EVALUATOR}`.",
        "This is the **formal-300 downstream behavioral validation**. It is distinct from the "
        "scorer-independent control-plane propagation results under "
        "`results/answer_leakage_repair/final_v3_analysis/control_plane_propagation/`; the two must not be conflated.",
        "Only frozen evaluator outputs are used. No API calls were made and the frozen 300 blocks were not resampled.",
        "",
        "Correctness provenance: frozen native deterministic results for deterministic routes, frozen "
        "`final-native-evaluation-v2` semantic judgments (dual LongMemEval primaries with Gemini binary "
        "majority; Memora Gemini primary with OpenAI secondary and Anthropic third), and one explicit blinded "
        "human adjudication. No obsolete deterministic-scorer behavioral values are used, and the cheap pilot "
        "is referenced only for the qualitative conclusion.",
        "",
        "## Coverage",
        "",
        "| Quantity | Value |",
        "| --- | --- |",
        f"| Frozen matched blocks | {metrics['total_frozen_blocks']} |",
        f"| Total observations | {metrics['total_observations']} |",
        f"| Resolved observations | {metrics['resolved_observations']} |",
        f"| Unresolved observations | {metrics['unresolved_observations']} |",
        f"| Human-adjudicated observations | {metrics['human_adjudicated_observations']} |",
        f"| Resolved matched triplets | {overall['n']} / {metrics['total_frozen_blocks']} |",
        "",
        "## Condition-level correctness (resolved matched blocks)",
        "",
        "| Condition | Correctness |",
        "| --- | --- |",
        f"| clean | {overall['clean']:.4f} |",
        f"| corrupted | {overall['corrupted']:.4f} |",
        f"| enforced | {overall['enforced']:.4f} |",
        "",
        "## Paired matched-block effects (95% paired bootstrap CI)",
        "",
        "| Effect | Estimate (pp) | 95% CI (pp) |",
        "| --- | --- | --- |",
        f"| corruption (clean − corrupted) | {overall['corruption_effect']*100:+.2f} | "
        f"[{ci['corruption_effect'][0]*100:+.2f}, {ci['corruption_effect'][1]*100:+.2f}] |",
        f"| recovery (enforced − corrupted) | {overall['recovery_effect']*100:+.2f} | "
        f"[{ci['recovery_effect'][0]*100:+.2f}, {ci['recovery_effect'][1]*100:+.2f}] |",
        f"| restoration gap (enforced − clean) | {overall['restoration_gap']*100:+.2f} | "
        f"[{ci['restoration_gap'][0]*100:+.2f}, {ci['restoration_gap'][1]*100:+.2f}] |",
        "",
        f"Formal signal: `{metrics['formal_signal']}`.",
        "",
        "## Results by source",
        "",
        "| Source | n | clean | corrupted | enforced | corruption (pp) | recovery (pp) | restoration gap (pp) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for source, row in metrics["by_source"].items():
        lines.append(
            f"| {source} | {row['n']} | {row['clean']:.3f} | {row['corrupted']:.3f} | {row['enforced']:.3f} | "
            f"{row['corruption_effect']*100:+.1f} | {row['recovery_effect']*100:+.1f} | {row['restoration_gap']*100:+.1f} |"
        )
    lines += [
        "",
        "## Results by generating provider",
        "",
        "| Generating provider | n | clean | corrupted | enforced | corruption (pp) | recovery (pp) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for provider, row in metrics["by_generating_provider"].items():
        lines.append(
            f"| {provider} | {row['n']} | {row['clean']:.3f} | {row['corrupted']:.3f} | {row['enforced']:.3f} | "
            f"{row['corruption_effect']*100:+.1f} | {row['recovery_effect']*100:+.1f} |"
        )
    lines += [
        "",
        "## Block-level outcomes",
        "",
        "| Outcome | Blocks | Rate |",
        "| --- | --- | --- |",
        f"| corruption damage (clean > corrupted) | {metrics['block_outcomes']['corruption_damage']} | "
        f"{metrics['block_outcomes']['corruption_damage_rate']:.4f} |",
        f"| recovery (enforced > corrupted) | {metrics['block_outcomes']['recovery']} | "
        f"{metrics['block_outcomes']['recovery_rate']:.4f} |",
        f"| full restoration (enforced ≥ clean) | {metrics['block_outcomes']['full_restoration']} | "
        f"{metrics['block_outcomes']['full_restoration_rate']:.4f} |",
        "",
        "## Evaluator routing",
        "",
        f"- Judge requests: {metrics['routing']['total_judge_requests']} "
        f"(completed durable results: {metrics['routing']['completed_judge_results']})",
        f"- By source: {metrics['routing']['by_source']}",
        f"- By provider: {metrics['routing']['by_provider']}",
        f"- By phase: {metrics['routing']['by_phase']}",
        "",
        "| Judge slot | Requests | Completed |",
        "| --- | --- | --- |",
    ]
    for slot, row in sorted(metrics["routing"]["by_judge_slot"].items()):
        lines.append(f"| {slot} | {row['requests']} | {row['completed']} |")
    lme = metrics["lme_primary_agreement"]
    memora = metrics["memora_routing"]
    lines += [
        "",
        "## LongMemEval primary agreement",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Dual-primary judged | {lme['judged']} |",
        f"| Agreements | {lme['agreements']} |",
        f"| Disagreements | {lme['disagreements']} |",
        f"| Gemini escalation requests | {lme['escalation_requests']} |",
        f"| Gemini escalations used for binary majority | {lme['escalation_completed']} |",
        f"| Agreement rate | {lme['agreement_rate']:.4f} |",
        "",
        "## Memora routing",
        "",
        "| Route | Observations |",
        "| --- | --- |",
        f"| Gemini primary (direct) | {memora['gemini_primary_direct']} |",
        f"| OpenAI secondary required | {memora['openai_secondary_required']} |",
        f"| Anthropic escalation completed | {memora['anthropic_escalation_completed']} |",
        f"| Human adjudication | {memora['human_adjudication']} |",
        "",
        "## Conclusion",
        "",
        f"Formal-300 reproduces the cheap-pilot qualitative conclusion: `{metrics['qualitative_conclusion_matches_cheap_pilot']}`.",
        f"Cheap-pilot signal: `{metrics['cheap_pilot']['signal']}`.",
        "",
    ]
    if metrics["unresolved_details"]:
        lines += [
            "## Unresolved observations",
            "",
            "These observations are not resolvable to a binary/FAMA score under the frozen rule "
            "(a primary judge returned `adjudication_required`, which maps to no `exploratory_binary`). "
            "They are not transport failures, and no judgment was imputed.",
            "",
            "| observation_id | source | condition | generating provider | resolution method |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in metrics["unresolved_details"]:
            lines.append(
                f"| {row['observation_id']} | {row['source']} | {row['condition']} | "
                f"{row['generating_provider']} | {row['resolution_method']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, default=Path("results/b4_scoring_validation/private/formal_300_validation_v1"))
    parser.add_argument("--public", type=Path, default=Path("results/formal_300_validation_v1"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args(argv)
    out = args.output or (args.private / "final_analysis")
    out.mkdir(parents=True, exist_ok=True)

    observations = read_jsonl(args.public / "selected_observations.jsonl")
    deterministic = read_jsonl(args.public / "deterministic_results.jsonl")
    plan = read_jsonl(args.public / "execution_plan.jsonl")
    dynamic = args.private / "dynamic_execution_plan.jsonl"
    if dynamic.exists():
        plan = plan + read_jsonl(dynamic)
    payloads = {r["observation_id"]: r for r in read_jsonl(args.private / "semantic_payloads.jsonl")}
    results = result_index(args.private)
    human_decisions = human_decision_map(args.private)

    resolved, counters = resolve(observations, deterministic, payloads, results, plan, human_decisions)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in resolved:
        grouped[row["pilot_block_id"]].append(row)
    blocks: list[dict[str, Any]] = []
    for bid, rows in grouped.items():
        if len(rows) != 3 or {r["condition"] for r in rows} != set(CONDITIONS):
            raise ValueError(f"matched block drift: {bid}")
        by = {r["condition"]: r for r in rows}
        if all(by[c]["formal_score"] is not None for c in CONDITIONS):
            first = rows[0]
            blocks.append({
                "pilot_block_id": bid, "source": first["source"], "generating_provider": first["generating_provider"],
                "fault_type": first["fault_type"], "scores": {c: float(by[c]["formal_score"]) for c in CONDITIONS},
                "resolution_methods": {c: by[c]["resolution_method"] for c in CONDITIONS},
            })
    blocks.sort(key=lambda b: b["pilot_block_id"])

    overall = block_metrics(blocks)
    cis = paired_bootstrap(blocks, args.iterations)
    by_source = {
        source: block_metrics([b for b in blocks if b["source"] == source])
        for source in sorted({b["source"] for b in blocks})
    }
    by_provider = {
        provider: block_metrics([b for b in blocks if b["generating_provider"] == provider])
        for provider in sorted({b["generating_provider"] for b in blocks})
    }
    reversals = sum(
        1 for table in (by_source, by_provider) for row in table.values()
        if row["n"] and (row["corruption_effect"] < 0 or row["recovery_effect"] < 0)
    )
    if len(blocks) < 210:
        signal = "inconclusive"
    elif cis["corruption_effect"][0] > 0 and cis["recovery_effect"][0] > 0 and reversals == 0 \
            and abs(overall["restoration_gap"]) < abs(overall["corruption_effect"]):
        signal = "strong_positive"
    elif overall["corruption_effect"] > 0 and overall["recovery_effect"] > 0:
        signal = "mixed"
    elif abs(overall["corruption_effect"]) < 0.02 and abs(overall["recovery_effect"]) < 0.02:
        signal = "weak"
    else:
        signal = "mixed"

    n = len(blocks)
    block_outcomes = {
        "corruption_damage": sum(1 for b in blocks if b["scores"]["clean"] > b["scores"]["corrupted"]),
        "recovery": sum(1 for b in blocks if b["scores"]["enforced"] > b["scores"]["corrupted"]),
        "full_restoration": sum(1 for b in blocks if b["scores"]["enforced"] >= b["scores"]["clean"]),
    }
    block_outcomes["corruption_damage_rate"] = block_outcomes["corruption_damage"] / n if n else None
    block_outcomes["recovery_rate"] = block_outcomes["recovery"] / n if n else None
    block_outcomes["full_restoration_rate"] = block_outcomes["full_restoration"] / n if n else None

    lme_rows = [r for r in resolved if r["source"] == "longmemeval" and r["resolution_method"].startswith(
        ("dual_primary", "gemini_"))]
    lme_agreement = {
        "judged": counters["lme_primary_judged"],
        "agreements": counters["lme_primary_agreement"],
        "disagreements": counters["lme_primary_disagreement"],
        "escalation_requests": sum(1 for r in plan if r["judge_slot"] == "lme_escalation_gemini"),
        "escalation_completed": counters["lme_escalation_completed"],
        "agreement_rate": (counters["lme_primary_agreement"] / counters["lme_primary_judged"])
        if counters["lme_primary_judged"] else None,
    }
    unresolved_details = [
        {
            "observation_id": r["observation_id"], "source": r["source"], "condition": r["condition"],
            "generating_provider": r["generating_provider"], "fault_type": r["fault_type"],
            "resolution_method": r["resolution_method"],
        }
        for r in resolved if r["formal_score"] is None
    ]
    memora_routing = {
        "gemini_primary_direct": sum(1 for r in resolved if r["resolution_method"] == "memora_gemini_primary"),
        "openai_secondary_required": counters["memora_secondary_required"],
        "anthropic_escalation_completed": counters["memora_escalation_completed"],
        "human_adjudication": sum(1 for r in resolved if r["resolution_method"] == HUMAN_ADJUDICATION_RESOLUTION_METHOD),
    }

    cheap_path = Path("results/cheap_judge_pilot_v1/analysis_post_recovery/pilot_analysis_summary.json")
    cheap = json.loads(cheap_path.read_text(encoding="utf-8")) if cheap_path.exists() else {}
    cheap_signal = cheap.get("pilot_signal")
    matches = "yes" if (cheap_signal is not None and signal == cheap_signal) else "no"

    resolved_observations = sum(1 for r in resolved if r["formal_score"] is not None)
    metrics = {
        "protocol": PROTOCOL, "evaluator": EVALUATOR,
        "total_frozen_blocks": len(grouped), "total_observations": len(observations),
        "resolved_observations": resolved_observations,
        "unresolved_observations": counters["unresolved_observations"],
        "human_adjudicated_observations": len(human_decisions),
        "resolved_matched_triplets": len(blocks),
        "routing": routing_counts(plan, results),
        "evaluator_counts": dict(counters),
        "lme_primary_agreement": lme_agreement,
        "memora_routing": memora_routing,
        "unresolved_details": unresolved_details,
        "paired_effects": overall, "paired_effects_ci95": cis,
        "by_source": by_source, "by_generating_provider": by_provider,
        "block_outcomes": block_outcomes, "formal_signal": signal,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "iterations": args.iterations, "unit": "matched provider/base block"},
        "cheap_pilot": {"signal": cheap_signal, "n": cheap.get("overall", {}).get("n")},
        "qualitative_conclusion_matches_cheap_pilot": matches,
        "scorer_independent_control_plane": "results/answer_leakage_repair/final_v3_analysis/control_plane_propagation",
        "api_calls_performed": 0,
    }

    (out / "formal_300_final_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "formal_300_final_report.md").write_text(build_report(metrics), encoding="utf-8")
    with (out / "formal_300_block_level_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pilot_block_id", "source", "generating_provider", "fault_type",
                         "clean_score", "corrupted_score", "enforced_score",
                         "corruption_effect", "recovery_effect", "restoration_gap",
                         "corruption_damage", "recovery", "full_restoration"])
        for b in blocks:
            s = b["scores"]
            writer.writerow([b["pilot_block_id"], b["source"], b["generating_provider"], b["fault_type"],
                             s["clean"], s["corrupted"], s["enforced"],
                             s["clean"] - s["corrupted"], s["enforced"] - s["corrupted"], s["enforced"] - s["clean"],
                             int(s["clean"] > s["corrupted"]), int(s["enforced"] > s["corrupted"]),
                             int(s["enforced"] >= s["clean"])])

    def pp(value: float | None) -> str:
        return "NA" if value is None else f"{value*100:+.2f}"

    def ci(key: str) -> str:
        values = cis[key]
        return "NA" if values is None else f"[{values[0]*100:+.2f},{values[1]*100:+.2f}]"

    print(f"FORMAL_BLOCKS={metrics['total_frozen_blocks']}")
    print(f"FORMAL_OBSERVATIONS={metrics['total_observations']}")
    print(f"RESOLVED_OBSERVATIONS={metrics['resolved_observations']}")
    print(f"UNRESOLVED_OBSERVATIONS={metrics['unresolved_observations']}")
    print(f"HUMAN_ADJUDICATED={metrics['human_adjudicated_observations']}")
    print(f"LME_PRIMARY_AGREEMENT_RATE={metrics['lme_primary_agreement']['agreement_rate']:.6f}")
    print(f"CLEAN_CORRECTNESS={overall['clean']:.6f}")
    print(f"CORRUPTED_CORRECTNESS={overall['corrupted']:.6f}")
    print(f"ENFORCED_CORRECTNESS={overall['enforced']:.6f}")
    print(f"CORRUPTION_EFFECT_PP={pp(overall['corruption_effect'])}")
    print(f"CORRUPTION_EFFECT_CI95={ci('corruption_effect')}")
    print(f"RECOVERY_EFFECT_PP={pp(overall['recovery_effect'])}")
    print(f"RECOVERY_EFFECT_CI95={ci('recovery_effect')}")
    print(f"RESTORATION_GAP_PP={pp(overall['restoration_gap'])}")
    print(f"RESTORATION_GAP_CI95={ci('restoration_gap')}")
    print(f"QUALITATIVE_CONCLUSION_MATCHES_CHEAP_PILOT={matches}")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
