#!/usr/bin/env python3
"""Prepare the frozen full-corpus native reevaluation without provider calls.

Public artifacts contain identifiers, hashes, routing results, and aggregate
cost estimates only. Licensed benchmark text and existing model answers are
written only to the gitignored private payload area.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.design_final_native_evaluation import (
    DESIGN_VERSION,
    SEED,
    is_valid_answer_row,
    lme_prompt,
    load_answer_paths,
    load_existing_answers,
    load_lme_source,
    load_memora_source,
    memora_key,
    memora_prompt,
    memora_subtype,
    native_lme_answerability,
    public_case_id,
    route_longmemeval,
    sha256_file,
    sha256_value,
    stable_stratified_sample,
)


PROTOCOL_VERSION = "full-native-reevaluation-v1"
B4_EVALUATOR_VERSION = "final-native-evaluation-v2"
LME_OPENAI_MODEL = "gpt-5.6-sol"
LME_ANTHROPIC_MODEL = "claude-sonnet-5"
GEMINI_MODEL = "gemini-3-flash-preview"
MEMORA_SECONDARY_PROVIDER = "openai"
MEMORA_SECONDARY_MODEL = LME_OPENAI_MODEL
MEMORA_THIRD_PROVIDER = "anthropic"
MEMORA_THIRD_MODEL = LME_ANTHROPIC_MODEL
EXPECTED = {
    "lme_total": 6666,
    "lme_deterministic_abstention": 2523,
    "lme_deterministic_native": 853,
    "lme_semantic": 3286,
    "lme_invalid": 4,
    "memora_answers": 15960,
    "memora_semantic_answers": 15955,
    "memora_criteria": 252336,
    "memora_semantic_criteria": 252144,
    "memora_invalid_criteria": 192,
}
INPUT_LIMITS = {
    "openai": 1_050_000,
    "anthropic": 1_000_000,
    "gemini": 1_048_576,
}
INPUT_LIMIT_SOURCES = {
    "openai": "frozen B4 frontier input-limit configuration",
    "anthropic": "frozen B4 frontier input-limit configuration",
    "gemini": "repository-verified B4 Gemini capability configuration",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError(f"frozen reevaluation artifact drift: {path}")
    path.write_text(payload, encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError(f"frozen reevaluation artifact drift: {path}")
    path.write_text(payload, encoding="utf-8")


def token_estimate(prompt: str) -> int:
    """Repository-standard planning estimate, not a provider tokenizer."""
    return math.ceil(len(prompt) / 4)


def output_budget(task: str, criterion_n: int = 0) -> int:
    return 160 if task == "longmemeval" else 40 + 36 * criterion_n


def request_id(case_id: str, judge_slot: str, provider: str, model: str, prompt_hash: str, payload_hash: str) -> str:
    identity = {
        "protocol_version": PROTOCOL_VERSION,
        "case_id": case_id,
        "judge_slot": judge_slot,
        "provider": provider,
        "model": model,
        "prompt_hash": prompt_hash,
        "payload_hash": payload_hash,
    }
    return f"full-native-v1-request::{sha256_value(identity)}"


def summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": 0, "median": 0, "mean": 0, "p90": 0, "max": 0, "total": 0}
    ordered = sorted(values)
    pos = (len(ordered) - 1) * 0.9
    lo, hi = math.floor(pos), math.ceil(pos)
    p90 = ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return {
        "n": len(values), "min": min(values), "median": statistics.median(values),
        "mean": statistics.mean(values), "p90": p90, "max": max(values), "total": sum(values),
    }


def _assert_counts(actual: dict[str, int]) -> None:
    drift = {key: {"expected": EXPECTED[key], "actual": value} for key, value in actual.items() if value != EXPECTED[key]}
    if drift:
        raise ValueError(f"canonical corpus/routing count drift: {json.dumps(drift, sort_keys=True)}")


def build(
    repo_root: Path,
    lme_source_path: Path,
    memora_root: Path,
    output_root: Path,
    private_root: Path,
) -> dict[str, Any]:
    answer_paths = load_answer_paths(repo_root)
    lme_rows, memora_rows, answer_provenance = load_existing_answers(answer_paths)
    lme_source = load_lme_source(lme_source_path)
    memora_source = load_memora_source(memora_root)

    deterministic: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    payload_by_case: dict[str, dict[str, Any]] = {}
    lme_semantic: list[dict[str, Any]] = []
    memora_semantic: list[dict[str, Any]] = []
    lme_routes: Counter[str] = Counter()
    memora_criterion_routes: Counter[str] = Counter()
    memora_criterion_types: Counter[str] = Counter()

    for row in lme_rows:
        qid = str(row["base_id"]).split("::")[-1]
        native = lme_source.get(qid)
        if native is None:
            raise ValueError(f"missing LongMemEval native row: {qid}")
        route = route_longmemeval(row, native)
        lme_routes[route["route"]] += 1
        case_id = public_case_id(row)
        common = {
            "case_id": case_id,
            "source": "longmemeval",
            "condition_id": row["condition_id"],
            "condition": row.get("condition_type"),
            "generating_provider": row.get("provider"),
            "answer_sha256": sha256_value(row.get("parsed_answer")),
            "question_id_sha256": sha256_value(qid),
            "question_type": native.get("question_type"),
            "route": route["route"],
        }
        if route["route"] == "semantic_judge_required":
            prompt = lme_prompt(row, native)
            prompt_hash = sha256_value(prompt)
            payload_hash = sha256_value({"case_id": case_id, "prompt": prompt, "evaluator": B4_EVALUATOR_VERSION})
            private_locator = f"semantic_payloads.jsonl#{case_id}"
            payload = {
                **common,
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "payload_hash": payload_hash,
                "protocol_version": PROTOCOL_VERSION,
                "b4_evaluator_version": B4_EVALUATOR_VERSION,
            }
            payloads.append(payload)
            payload_by_case[case_id] = payload
            lme_semantic.append({**common, "prompt_hash": prompt_hash, "payload_hash": payload_hash, "private_payload_locator": private_locator})
        else:
            raw_decision = route.get("native_decision")
            binary: bool | None
            if route["route"] == "deterministic_native":
                raw_decision, binary = "correct", True
            elif route["route"] == "deterministic_abstention":
                binary = raw_decision == "abstention_correct"
            else:
                raw_decision, binary = "invalid_unscorable", None
            deterministic.append({
                **common,
                "deterministic_raw_decision": raw_decision,
                "deterministic_binary_correct": binary,
                "method": route["reason"],
                "native_answerability": route.get("answerability", native_lme_answerability(native)),
                "routing_design_version": DESIGN_VERSION,
                "b4_evaluator_version": B4_EVALUATOR_VERSION,
            })

    expected_assignments: set[str] = set()
    for row in memora_rows:
        key = memora_key(str(row["base_id"]))
        native = memora_source.get(key)
        if native is None:
            raise ValueError(f"missing Memora native row: {key}")
        criteria = native.get("evaluation", {}).get("evaluation_questions", [])
        if not criteria:
            raise ValueError(f"Memora row has no native criteria: {key}")
        case_id = public_case_id(row)
        route = "semantic_judge_required" if is_valid_answer_row(row) else "invalid_unscorable"
        assignments = []
        for criterion in criteria:
            native_id = str(criterion["evaluation_question_id"])
            assignment_id = f"{case_id}::{native_id}"
            if assignment_id in expected_assignments:
                raise ValueError(f"duplicate Memora criterion assignment: {assignment_id}")
            expected_assignments.add(assignment_id)
            kind = str(criterion["evaluation_type"])
            memora_criterion_routes[route] += 1
            memora_criterion_types[kind] += 1
            assignments.append(assignment_id)
            if route == "invalid_unscorable":
                deterministic.append({
                    "case_id": case_id,
                    "criterion_assignment_id": assignment_id,
                    "source": "memora",
                    "subtype": memora_subtype(key[2]),
                    "condition_id": row["condition_id"],
                    "condition": row.get("condition_type"),
                    "generating_provider": row.get("provider"),
                    "answer_sha256": sha256_value(row.get("parsed_answer")),
                    "route": route,
                    "criterion_type": kind,
                    "deterministic_raw_decision": "invalid_unscorable",
                    "deterministic_binary_correct": None,
                    "method": "missing_or_failed_existing_answer",
                    "routing_design_version": DESIGN_VERSION,
                    "b4_evaluator_version": B4_EVALUATOR_VERSION,
                })
        if route == "semantic_judge_required":
            prompt = memora_prompt(row, native)
            prompt_hash = sha256_value(prompt)
            payload_hash = sha256_value({"case_id": case_id, "prompt": prompt, "evaluator": B4_EVALUATOR_VERSION})
            common = {
                "case_id": case_id,
                "source": "memora",
                "subtype": memora_subtype(key[2]),
                "native_task_group": native["native_task_group"],
                "condition_id": row["condition_id"],
                "condition": row.get("condition_type"),
                "generating_provider": row.get("provider"),
                "answer_sha256": sha256_value(row.get("parsed_answer")),
                "native_question_key_sha256": sha256_value(key),
                "route": route,
                "criterion_assignment_ids": assignments,
                "criterion_n": len(assignments),
                "memory_presence_n": sum(c["evaluation_type"] == "memory_presence" for c in criteria),
                "forgetting_absence_n": sum(c["evaluation_type"] == "forgetting_absence" for c in criteria),
                "prompt_hash": prompt_hash,
                "payload_hash": payload_hash,
                "private_payload_locator": f"semantic_payloads.jsonl#{case_id}",
            }
            payload = {
                **common,
                "prompt": prompt,
                "protocol_version": PROTOCOL_VERSION,
                "b4_evaluator_version": B4_EVALUATOR_VERSION,
            }
            payloads.append(payload)
            payload_by_case[case_id] = payload
            memora_semantic.append(common)

    actual = {
        "lme_total": len(lme_rows),
        "lme_deterministic_abstention": lme_routes["deterministic_abstention"],
        "lme_deterministic_native": lme_routes["deterministic_native"],
        "lme_semantic": lme_routes["semantic_judge_required"],
        "lme_invalid": lme_routes["invalid_unscorable"],
        "memora_answers": len(memora_rows),
        "memora_semantic_answers": len(memora_semantic),
        "memora_criteria": sum(memora_criterion_routes.values()),
        "memora_semantic_criteria": memora_criterion_routes["semantic_judge_required"],
        "memora_invalid_criteria": memora_criterion_routes["invalid_unscorable"],
    }
    _assert_counts(actual)
    if len(payload_by_case) != len(lme_semantic) + len(memora_semantic):
        raise ValueError("duplicate semantic case identity")

    # The frozen v1 risk protocol sampled 200 Memora answers with SEED+2.
    # LME no longer uses that single-primary audit sample because v2 requires
    # two frontier primaries on every routed semantic answer.
    memora_secondary = stable_stratified_sample(
        memora_semantic, min(200, len(memora_semantic)), SEED + 2,
        ("generating_provider", "condition", "native_task_group"),
    )
    secondary_ids = {row["case_id"] for row in memora_secondary}
    if len(secondary_ids) != 200:
        raise ValueError("frozen Memora secondary/risk sample must contain 200 unique answers")

    execution: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []

    def add_slot(case: dict[str, Any], judge_slot: str, provider: str, model: str, role: str) -> None:
        payload = payload_by_case[case["case_id"]]
        prompt = payload["prompt"]
        estimated = token_estimate(prompt)
        if estimated > INPUT_LIMITS[provider]:
            raise ValueError(f"input-limit blocker: {case['case_id']} {provider} {estimated}>{INPUT_LIMITS[provider]}")
        rid = request_id(case["case_id"], judge_slot, provider, model, case["prompt_hash"], case["payload_hash"])
        public = {
            "evaluation_request_id": rid,
            "case_id": case["case_id"],
            "source": case["source"],
            "subtype": case.get("subtype"),
            "condition_id": case["condition_id"],
            "answer_sha256": case["answer_sha256"],
            "judge_slot": judge_slot,
            "judge_role": role,
            "provider": provider,
            "model": model,
            "prompt_hash": case["prompt_hash"],
            "payload_hash": case["payload_hash"],
            "protocol_version": PROTOCOL_VERSION,
            "b4_evaluator_version": B4_EVALUATOR_VERSION,
            "private_payload_locator": case["private_payload_locator"],
            "estimated_input_tokens": estimated,
            "planned_output_tokens": output_budget(case["source"], int(case.get("criterion_n", 0))),
            "configured_max_input_tokens": INPUT_LIMITS[provider],
            "input_limit_source": INPUT_LIMIT_SOURCES[provider],
        }
        manifests.append(public)
        execution.append({
            **public,
            "initial_state": "planned",
            "attempt": 0,
            "max_attempts_without_explicit_recovery": 1,
            "automatic_retries": False,
            "ledger_path": f"<private-root>/ledgers/{provider}.jsonl",
            "result_path": f"<private-root>/results/{provider}.jsonl",
            "resume_policy": "completed_skip; failed_or_started_only_fail_closed; explicit_append_only_recovery_required",
        })

    for case in sorted(lme_semantic, key=lambda item: item["case_id"]):
        add_slot(case, "lme_primary_openai", "openai", LME_OPENAI_MODEL, "dual_primary")
        add_slot(case, "lme_primary_anthropic", "anthropic", LME_ANTHROPIC_MODEL, "dual_primary")
    for case in sorted(memora_semantic, key=lambda item: item["case_id"]):
        add_slot(case, "memora_primary", "gemini", GEMINI_MODEL, "primary")
        if case["case_id"] in secondary_ids:
            add_slot(case, "memora_secondary_risk", MEMORA_SECONDARY_PROVIDER, MEMORA_SECONDARY_MODEL, "frozen_secondary_risk_sample")

    ids = [row["evaluation_request_id"] for row in execution]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evaluation_request_id")

    payloads.sort(key=lambda item: item["case_id"])
    execution.sort(key=lambda item: item["evaluation_request_id"])
    manifests.sort(key=lambda item: item["evaluation_request_id"])
    deterministic.sort(key=lambda item: (item["source"], item["case_id"], item.get("criterion_assignment_id", "")))
    write_jsonl(private_root / "semantic_payloads.jsonl", payloads)
    write_jsonl(output_root / "execution_plan.jsonl", execution)
    write_jsonl(output_root / "request_payload_manifest.jsonl", manifests)
    write_jsonl(output_root / "deterministic_results.jsonl", deterministic)

    groups: dict[str, list[dict[str, Any]]] = {
        "longmemeval_openai": [r for r in manifests if r["judge_slot"] == "lme_primary_openai"],
        "longmemeval_anthropic": [r for r in manifests if r["judge_slot"] == "lme_primary_anthropic"],
        "memora_gemini_primary": [r for r in manifests if r["judge_slot"] == "memora_primary"],
        "memora_secondary_risk": [r for r in manifests if r["judge_slot"] == "memora_secondary_risk"],
    }
    volumes = {}
    for name, rows in groups.items():
        volumes[name] = {
            "request_n": len(rows),
            "estimated_input_tokens": sum(r["estimated_input_tokens"] for r in rows),
            "planned_output_tokens": sum(r["planned_output_tokens"] for r in rows),
            "input_token_distribution": summary([r["estimated_input_tokens"] for r in rows]),
            "max_output_tokens_per_request": max((r["planned_output_tokens"] for r in rows), default=0),
        }
    lme_base = groups["longmemeval_openai"]
    escalation_scenarios = {}
    lme_input_total = sum(r["estimated_input_tokens"] for r in lme_base)
    for rate in (0.01, 0.02, 0.05, 0.10):
        calls = math.ceil(len(lme_base) * rate)
        escalation_scenarios[f"{int(rate * 100)}pct"] = {
            "calls": calls,
            "estimated_input_tokens_proportional": math.ceil(lme_input_total * rate),
            "planned_output_tokens": calls * 160,
            "note": "volume scenario only; exact cases and tokens depend on observed primary disagreement",
        }
    base_calls = len(execution)
    total_input = sum(r["estimated_input_tokens"] for r in manifests)
    total_output = sum(r["planned_output_tokens"] for r in manifests)
    cost = {
        "schema_version": PROTOCOL_VERSION,
        "token_estimation": "sum over actual frozen prompts of ceil(Python character count / 4); planning estimate, not a provider tokenizer",
        "output_budget": "160 tokens per LME judgment; 40 + 36 per Memora native criterion",
        "volumes": volumes,
        "longmemeval_gemini_escalation": {"baseline_calls": 0, "provider": "gemini", "model": GEMINI_MODEL, "scenarios": escalation_scenarios},
        "memora_dynamic_secondary": "all primary low-confidence or adjudication_required outputs not already in the 200-case frozen sample; count is result-dependent and excluded from base calls",
        "memora_third_judge": {"baseline_calls": 0, "provider": MEMORA_THIRD_PROVIDER, "model": MEMORA_THIRD_MODEL, "trigger": "two valid semantic judgments disagree"},
        "base_paid_calls": base_calls,
        "estimated_input_tokens": total_input,
        "estimated_output_token_budget": total_output,
        "usd_estimate": None,
        "usd_estimate_reason": "no repository-local authoritative pricing table covers all exact frozen judge models",
    }
    write_json(output_root / "cost_estimate.json", cost)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "b4_evaluator_version": B4_EVALUATOR_VERSION,
        "routing_design_version": DESIGN_VERSION,
        "answer_generation_calls": 0,
        "paid_calls_performed": 0,
        "population": actual,
        "execution_request_n": len(execution),
        "semantic_payload_n": len(payloads),
        "deterministic_result_n": len(deterministic),
        "memora_secondary_min_calls": len(memora_secondary),
        "answer_artifacts": answer_provenance["answer_artifacts"],
        "source_hashes": {
            "longmemeval_oracle_sha256": sha256_file(lme_source_path),
            "memora_native_tree_fingerprint": sha256_value(sorted((str(key), sha256_value(value.get("evaluation", {}))) for key, value in memora_source.items())),
        },
        "private_payload_sha256": sha256_file(private_root / "semantic_payloads.jsonl"),
        "execution_plan_sha256": sha256_file(output_root / "execution_plan.jsonl"),
        "public_output_policy": "No benchmark text, native criterion wording, or model-answer text is present in public plan artifacts.",
    }
    write_json(output_root / "plan_manifest.json", manifest)

    report = f"""# Full native reevaluation plan

Protocol: `{PROTOCOL_VERSION}`. Frozen evaluator methodology: `{B4_EVALUATOR_VERSION}`. This is an offline plan over existing answers; it performed no answer-generation or semantic-judge calls.

## Verified population and routing

- LongMemEval: {actual['lme_total']:,} rows; {actual['lme_deterministic_native']:,} deterministic-native, {actual['lme_deterministic_abstention']:,} deterministic-abstention, {actual['lme_semantic']:,} semantic, and {actual['lme_invalid']:,} invalid/unscorable.
- Memora: {actual['memora_answers']:,} generated answers and {actual['memora_criteria']:,} native criterion assignments; {actual['memora_semantic_answers']:,} answer batches / {actual['memora_semantic_criteria']:,} criteria require semantic evaluation, while {actual['memora_invalid_criteria']:,} criteria belong to invalid/unscorable answers.
- OAKS is not scheduled here: its frozen native option evaluator remains the final evaluator and requires no semantic calls.

Every LongMemEval semantic case has independent OpenAI `{LME_OPENAI_MODEL}` and Anthropic `{LME_ANTHROPIC_MODEL}` primary slots. Gemini `{GEMINI_MODEL}` is created only after an observed binary disagreement. Every valid Memora answer has one Gemini primary batch containing all of that answer's native criteria. The frozen minimum Memora reliability/risk sample has {len(memora_secondary):,} OpenAI secondary calls; additional low-confidence or `adjudication_required` outputs are dynamic, and Anthropic is invoked only after two valid judgments disagree.

## Base execution and cost scale

- LongMemEval OpenAI: {volumes['longmemeval_openai']['request_n']:,} calls; {volumes['longmemeval_openai']['estimated_input_tokens']:,} estimated input tokens; {volumes['longmemeval_openai']['planned_output_tokens']:,} output-token budget.
- LongMemEval Anthropic: {volumes['longmemeval_anthropic']['request_n']:,} calls; {volumes['longmemeval_anthropic']['estimated_input_tokens']:,} estimated input tokens; {volumes['longmemeval_anthropic']['planned_output_tokens']:,} output-token budget.
- Memora Gemini primary: {volumes['memora_gemini_primary']['request_n']:,} calls; {volumes['memora_gemini_primary']['estimated_input_tokens']:,} estimated input tokens; {volumes['memora_gemini_primary']['planned_output_tokens']:,} output-token budget.
- Memora frozen secondary/risk minimum: {volumes['memora_secondary_risk']['request_n']:,} OpenAI calls; {volumes['memora_secondary_risk']['estimated_input_tokens']:,} estimated input tokens; {volumes['memora_secondary_risk']['planned_output_tokens']:,} output-token budget.
- Base: {base_calls:,} calls, {total_input:,} estimated input tokens, and {total_output:,} planned output tokens.

Token estimates are computed from every actual frozen prompt as `ceil(character_count / 4)` and are planning estimates, not provider-tokenizer measurements. No dollar estimate is asserted because the repository has no authoritative local pricing table covering all exact judge models. LME Gemini escalation scenarios (1%, 2%, 5%, 10%) are recorded in `cost_estimate.json` and begin at zero calls.

## Identity, recovery, and privacy

Every base request binds case ID, source, condition ID, answer hash, judge slot, provider, model, prompt hash, payload hash, `{PROTOCOL_VERSION}`, and `{B4_EVALUATOR_VERSION}`. Execution is max-attempt-1 by default. Completed requests are skipped; failed or started-only requests fail closed and require an explicit append-only recovery authorization. Attempt 3 is not allowed without a later audited recovery. Private prompts are at `{private_root}` and are gitignored by the repository's `results/b4_scoring_validation/private/` rule.

## Concurrency audit

The provider clients use per-thread clients or internal synchronization where implemented, but the shared `DurableLedger` performs append + flush + fsync without an interprocess file lock, and result materialization is not transactionally coordinated with it. The end-to-end writer is therefore **not concurrency-safe**. Recommended operation is serial concurrency 1 per provider ledger. Provider-partitioned execution may run in parallel only with the separate provider ledger/result paths already recorded in the plan, one serial writer per partition, followed by an offline identity-checked merge. Do not point concurrent workers at the same ledger or result file.

Suggested safe operational concurrency: Gemini 1, OpenAI 1, Anthropic 1 per isolated provider partition. Higher provider concurrency requires a separately tested locked writer and is not introduced by this plan.
"""
    report_path = output_root / "FULL_NATIVE_REEVALUATION_PLAN.md"
    if report_path.exists() and report_path.read_text(encoding="utf-8") != report:
        raise ValueError(f"frozen reevaluation artifact drift: {report_path}")
    report_path.write_text(report, encoding="utf-8")
    return {"actual": actual, "cost": cost, "execution_n": len(execution), "secondary_n": len(memora_secondary)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--longmemeval-source", type=Path, default=Path("$LONGMEMEVAL_SOURCE/longmemeval-cleaned/longmemeval_oracle.json"))
    parser.add_argument("--memora-source", type=Path, default=Path("$MEMORA_SOURCE"))
    parser.add_argument("--output-root", type=Path, default=Path("results/full_native_reevaluation_v1"))
    parser.add_argument("--private-root", type=Path, default=Path("results/b4_scoring_validation/private/full_native_reevaluation_v1"))
    args = parser.parse_args(argv)
    result = build(args.repo_root, args.longmemeval_source, args.memora_source, args.output_root, args.private_root)
    actual, cost = result["actual"], result["cost"]
    print(f"PROTOCOL={PROTOCOL_VERSION}")
    print(f"B4_EVALUATOR={B4_EVALUATOR_VERSION}")
    print("ANSWER_GENERATION_CALLS=0")
    print(f"LME_TOTAL_ROWS={actual['lme_total']}")
    print(f"LME_DETERMINISTIC_ROWS={actual['lme_deterministic_native'] + actual['lme_deterministic_abstention']}")
    print(f"LME_SEMANTIC_CASES={actual['lme_semantic']}")
    print(f"LME_OPENAI_PRIMARY_CALLS={cost['volumes']['longmemeval_openai']['request_n']}")
    print(f"LME_ANTHROPIC_PRIMARY_CALLS={cost['volumes']['longmemeval_anthropic']['request_n']}")
    print(f"MEMORA_TOTAL_ANSWERS={actual['memora_answers']}")
    print(f"MEMORA_PRIMARY_CALLS={cost['volumes']['memora_gemini_primary']['request_n']}")
    print(f"MEMORA_SECONDARY_MIN_CALLS={result['secondary_n']}")
    print(f"BASE_PAID_CALLS={cost['base_paid_calls']}")
    print(f"ESTIMATED_INPUT_TOKENS={cost['estimated_input_tokens']}")
    print(f"ESTIMATED_OUTPUT_TOKEN_BUDGET={cost['estimated_output_token_budget']}")
    print("CONCURRENCY_SAFE=no")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
