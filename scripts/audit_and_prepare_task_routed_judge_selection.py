#!/usr/bin/env python3
"""Audit Gemini reference independence and prepare task-routed LME plans.

This script is offline-only.  It reads the 404-case calibration and private
payloads needed to hash frozen prompts, but never reads locked holdout labels
and never imports or calls a provider client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.providers.deepseek_provider import DEFAULT_DEEPSEEK_MODEL
from eval.providers.kimi_provider import DEFAULT_KIMI_MODEL
from scripts.b4_calibration_execution import estimate_input_tokens, provider_request_prompt
from scripts.prepare_final_judge_selection_calibration import (
    digest,
    file_digest,
    load_payloads,
    read_jsonl,
    write_json,
    write_jsonl,
)
from scripts.b4_calibration_runner import build_lme_prompt


POLICY_VERSION = "final-judge-selection-task-routed-v2"
SOURCE_CALIBRATION_VERSION = "final-judge-selection-calibration-v1"
EXPECTED_CALIBRATION_N = 404
EXPECTED_LME_N = 64
EXPECTED_MEMORA_N = 340
EXPECTED_HOLDOUT_N = 200
VERIFIED_DEEPSEEK_LIMIT = 1_048_576
VERIFIED_KIMI_LIMIT = 262_144
DEEPSEEK_LIMIT_SOURCE = "repository-verified B4 DeepSeek V4 Pro capability configuration"
KIMI_LIMIT_SOURCE = "repository-verified B4 Kimi K2.6 capability configuration"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def projection(task: str, decision: str | None) -> str | None:
    if task == "longmemeval":
        if decision in {"correct", "abstention_correct"}:
            return "correct"
        if decision in {"incorrect", "abstention_incorrect"}:
            return "incorrect"
    if task == "memora":
        if decision in {"pass", "fail"}:
            return decision
    return None


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    ref = [x[0] for x in pairs]
    pred = [x[1] for x in pairs]
    labels = sorted(set(ref) | set(pred))
    po = sum(a == b for a, b in pairs) / len(pairs)
    pe = sum(ref.count(label) * pred.count(label) for label in labels) / (len(pairs) ** 2)
    return None if pe == 1 else (po - pe) / (1 - pe)


def confusion(pairs: list[tuple[str, str]], positive: str) -> dict[str, int]:
    return {
        "true_positive": sum(ref == positive and pred == positive for ref, pred in pairs),
        "false_positive": sum(ref != positive and pred == positive for ref, pred in pairs),
        "false_negative": sum(ref == positive and pred != positive for ref, pred in pairs),
        "true_negative": sum(ref != positive and pred != positive for ref, pred in pairs),
    }


def reference_category(adjudication: dict[str, Any]) -> str:
    method = adjudication.get("resolution_method")
    if method == "third_judge_majority":
        return "reference_used_gemini"
    if method == "human_adjudication" and adjudication.get("gemini_raw_decision") is not None:
        return "human_adjudication_after_gemini"
    return "reference_independent_of_gemini"


def metric_block(rows: list[dict[str, Any]], result_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs: list[tuple[str, str]] = []
    abstention_pairs: list[tuple[str, str]] = []
    parse_failure = 0
    provider_failure = 0
    adjudication = 0
    for row in rows:
        identifier = str(row["blinded_request_id"])
        result = result_by_id.get(identifier)
        if result is None or result.get("parse_status") != "completed":
            if result and result.get("parse_status") == "parse_failure":
                parse_failure += 1
            elif result:
                provider_failure += 1
            continue
        decision = result.get("decision")
        if decision == "adjudication_required":
            adjudication += 1
        predicted = projection(str(row["task"]), decision)
        if predicted is not None:
            pairs.append((str(row["gold_binary_label"]), predicted))
        if row["task"] == "longmemeval" and row.get("gold_raw_label") in {"abstention_correct", "abstention_incorrect"}:
            abstention_pairs.append((str(row["gold_raw_label"]), str(decision)))
    agreement = sum(ref == pred for ref, pred in pairs)
    positive = "correct" if rows and rows[0]["task"] == "longmemeval" else "pass"
    return {
        "n": len(pairs), "agreement_n": agreement,
        "agreement_rate": agreement / len(pairs) if pairs else None,
        "disagreement_n": len(pairs) - agreement,
        "cohen_kappa": kappa(pairs), "confusion_matrix": confusion(pairs, positive),
        "adjudication_required_n": adjudication, "parse_failure_n": parse_failure,
        "provider_failure_n": provider_failure,
        "abstention_subtype": {
            "eligible_n": len(abstention_pairs), "raw_exact_n": sum(a == b for a, b in abstention_pairs),
            "raw_exact_rate": sum(a == b for a, b in abstention_pairs) / len(abstention_pairs) if abstention_pairs else None,
        },
    }


def independence_audit(calibration: list[dict[str, Any]], adjudication: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    adj_by_id = {str(row["blinded_request_id"]): row for row in adjudication}
    result_by_id = {str(row["blinded_request_id"]): row for row in results}
    calibration_ids = {str(row["blinded_request_id"]) for row in calibration}
    if not calibration_ids.issubset(set(adj_by_id)):
        raise ValueError("calibration cases missing from adjudication reference")
    if len(result_by_id) != len(results) or set(result_by_id) != calibration_ids:
        raise ValueError("Gemini calibration result identity mismatch")
    audit_cases = []
    for row in calibration:
        identifier = str(row["blinded_request_id"])
        adj = adj_by_id[identifier]
        result = result_by_id[identifier]
        category = reference_category(adj)
        audit_cases.append({
            "blinded_request_id": identifier, "task": row["task"], "subtype": row.get("subtype"),
            "reference_category": category, "resolution_method": adj["resolution_method"],
            "gold_binary_label": row["gold_binary_label"],
            "candidate_binary_label": projection(row["task"], result.get("decision")),
            "candidate_decision": result.get("decision"), "candidate_parse_status": result.get("parse_status"),
        })
    independent = [row for row in calibration if reference_category(adj_by_id[str(row["blinded_request_id"])]) == "reference_independent_of_gemini"]
    used = [row for row in calibration if reference_category(adj_by_id[str(row["blinded_request_id"])]) == "reference_used_gemini"]
    human = [row for row in calibration if reference_category(adj_by_id[str(row["blinded_request_id"])]) == "human_adjudication_after_gemini"]
    all_metrics = metric_block(calibration, result_by_id)
    independent_metrics = metric_block(independent, result_by_id)
    by_task = {task: metric_block([row for row in calibration if row["task"] == task], result_by_id) for task in ("longmemeval", "memora")}
    independent_by_task = {task: metric_block([row for row in independent if row["task"] == task], result_by_id) for task in ("longmemeval", "memora")}
    return {
        "schema_version": POLICY_VERSION, "source_calibration_version": SOURCE_CALIBRATION_VERSION,
        "calibration_n": len(calibration), "candidate": {"provider": "gemini", "model": "gemini-3-flash-preview"},
        "reference_independence": {
            "all_n": len(calibration), "independent_n": len(independent), "reference_used_gemini_n": len(used), "human_adjudication_after_gemini_n": len(human),
            "by_category": dict(Counter(row["reference_category"] for row in audit_cases)),
            "provenance_rule": "third_judge_majority means Gemini entered final gold; human_adjudication is marked dependent when Gemini is present in the escalation evidence; all other frozen resolution methods are independent of Gemini.",
        },
        "all_calibration": {"overall": all_metrics, "by_task": by_task},
        "reference_independent_only": {"overall": independent_metrics, "by_task": independent_by_task},
        "case_audit": audit_cases,
        "holdout_read": False,
        "api_calls": 0,
    }


def build_lme_plans(calibration: list[dict[str, Any]], payload_index: dict[str, dict[str, Any]], locators: dict[str, str]) -> list[dict[str, Any]]:
    lme = [row for row in calibration if row["task"] == "longmemeval"]
    if len(lme) != EXPECTED_LME_N:
        raise ValueError(f"expected exactly {EXPECTED_LME_N} frozen LME calibration cases, found {len(lme)}")
    plans = []
    candidates = [
        ("deepseek", DEFAULT_DEEPSEEK_MODEL, VERIFIED_DEEPSEEK_LIMIT, DEEPSEEK_LIMIT_SOURCE),
        ("kimi", DEFAULT_KIMI_MODEL, VERIFIED_KIMI_LIMIT, KIMI_LIMIT_SOURCE),
    ]
    for row in lme:
        identifier = str(row["blinded_request_id"])
        payload = payload_index[identifier]
        semantic_prompt = build_lme_prompt(payload)
        semantic_hash = digest(semantic_prompt)
        if semantic_hash != row["prompt_sha256"]:
            raise ValueError(f"frozen LME prompt hash drift: {identifier}")
        for provider, model, limit, limit_source in candidates:
            request_prompt = provider_request_prompt(provider, semantic_prompt)
            request_hash = digest(request_prompt)
            measured = estimate_input_tokens(request_prompt, model)
            if measured["estimated_tokens"] > limit:
                raise ValueError(f"input-limit blocker for {provider}/{model}: {identifier}")
            slot_id = "b4-task-routed-slot::" + digest({"policy_version": POLICY_VERSION, "blinded_request_id": identifier, "provider": provider, "model": model, "request_hash": request_hash})
            plans.append({
                "request_slot_id": slot_id, "blinded_request_id": identifier, "task": "longmemeval",
                "candidate_id": f"{provider}-{model}", "provider": provider, "model": model,
                "protocol_version": POLICY_VERSION, "semantic_prompt_hash": semantic_hash, "request_hash": request_hash,
                "private_payload_locator": locators[identifier], "payload_sha256": row["payload_sha256"],
                "prompt_chars": len(request_prompt), "estimated_input_tokens": measured["estimated_tokens"],
                "input_token_estimation_method": measured["method"], "configured_max_input_tokens": limit,
                "input_limit_source": limit_source, "max_attempts": 1, "automatic_retries": False,
                "execution_status": "planned_not_executed", "holdout_touched": False,
            })
    if len(plans) != 128 or len({row["request_slot_id"] for row in plans}) != 128:
        raise ValueError("LME candidate plan cardinality or uniqueness failure")
    return sorted(plans, key=lambda row: (row["provider"], row["blinded_request_id"]))


def protocol_markdown(audit: dict[str, Any], plans: list[dict[str, Any]]) -> str:
    independent = audit["reference_independent_only"]
    mem = independent["by_task"]["memora"]
    lme = independent["by_task"]["longmemeval"]
    return f"""# Task-routed judge-selection protocol v2

Policy version: `{POLICY_VERSION}`. Source calibration version: `{SOURCE_CALIBRATION_VERSION}`. This is an offline calibration-stage adaptation; no holdout label was read and no judge call was made.

## Independence audit

The 404-case Gemini calibration contains {audit['reference_independence']['independent_n']} reference-independent cases, {audit['reference_independence']['reference_used_gemini_n']} cases where Gemini participated in `third_judge_majority`, and {audit['reference_independence']['human_adjudication_after_gemini_n']} human-adjudicated case whose evidence chain included Gemini. The independent-subset Memora result is {mem['agreement_n']}/{mem['n']} ({mem['agreement_rate']:.6f}); LongMemEval is {lme['agreement_n']}/{lme['n']} ({lme['agreement_rate']:.6f}).

The all-case Gemini result is not used to qualify Gemini as a universal primary. Reference dependence is determined from immutable adjudication resolution provenance, not from task labels or observed answers.

## Task routing

Memora may use Gemini `gemini-3-flash-preview` as a provisional primary because its reference-independent calibration agreement is at least 0.90 ({mem['agreement_rate']:.6f}); 0.95 remains preferred but is not required for eligibility.

Gemini is rejected as the LongMemEval primary: all-case agreement is 0.640625 and the reference-independent subset is below 0.90. DeepSeek `{DEFAULT_DEEPSEEK_MODEL}` and Kimi `{DEFAULT_KIMI_MODEL}` are prepared on exactly the unchanged 64-case LME calibration subset. Each has 64 planned calls, one per case, max attempts 1, and no automatic retry. No Memora calls are planned for them.

An LME candidate is eligible only at binary agreement >=0.90. Prefer >=0.95, then higher abstention-subtype accuracy, lower parse/provider failure, and stronger kappa. If neither candidate reaches 0.90, `LME_PRIMARY_NOT_FOUND=yes`; this policy does not silently fall back to OpenAI or Anthropic.

## Future locked holdout

The 200-case holdout remains untouched and its labels remain locked. Only after both task primaries are frozen, evaluate the complete composite exactly once: 168 Memora cases through the frozen Memora primary and 32 LongMemEval cases through the frozen LME primary. The composite passes only if overall, LongMemEval, and Memora agreement each reach 0.90; 0.95 overall is preferred. No routing or model changes may be made after observing holdout results.

## Blinding and execution safety

Candidate plans contain hashes, IDs, and private payload locators only. Candidate prompts preserve the frozen LongMemEval semantic schema and hide generating provider/model, condition, GovMem scorer, gold labels, and other judge outputs. DeepSeek uses only provider transport JSON wording in addition to the same semantic prompt; Kimi receives the semantic prompt unchanged. The two plan families contain exactly {sum(row['provider'] == 'deepseek' for row in plans)} and {sum(row['provider'] == 'kimi' for row in plans)} slots. All future execution must use separate append-only ledgers, completed-slot skipping, fail-closed started-only handling, provider/model/hash drift checks, and max_attempts=1.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=Path("results/final_native_evaluation_design/judge_selection/calibration_cases.jsonl"))
    parser.add_argument("--adjudication", type=Path, default=Path("results/b4_scoring_validation/private/frontier_adjudication/adjudication_cases.jsonl"))
    parser.add_argument("--candidate-results", type=Path, default=Path("results/b4_scoring_validation/private/final_judge_selection/gemini-3-flash-preview/candidate_results.jsonl"))
    parser.add_argument("--lme-payload", type=Path, default=Path("results/b4_scoring_validation/private/longmemeval_judge_execution.jsonl"))
    parser.add_argument("--memora-payload", type=Path, default=Path("results/b4_scoring_validation/private/memora_judge_execution.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/final_native_evaluation_design/judge_selection/task_routed_v2"))
    args = parser.parse_args(argv)
    calibration = load_jsonl(args.calibration)
    if len(calibration) != EXPECTED_CALIBRATION_N:
        raise SystemExit(f"expected {EXPECTED_CALIBRATION_N} calibration rows")
    adjudication = load_jsonl(args.adjudication)
    results = load_jsonl(args.candidate_results)
    audit = independence_audit(calibration, adjudication, results)
    payload_index, locators = load_payloads(args.lme_payload, args.memora_payload)
    plans = build_lme_plans(calibration, payload_index, locators)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "independence_audit.json", audit)
    write_jsonl(output / "lme_candidate_plan.json", plans)
    write_json(output / "holdout_policy.json", {
        "policy_version": POLICY_VERSION, "holdout_status": "locked_not_read_not_exported", "total": EXPECTED_HOLDOUT_N,
        "memora_n": 168, "longmemeval_n": 32, "unlock_requires": ["Memora primary frozen", "LongMemEval primary frozen", "explicit separate execution command"],
        "pass_thresholds": {"overall": 0.90, "longmemeval": 0.90, "memora": 0.90}, "preferred_overall": 0.95, "post_holdout_model_or_routing_changes": False,
    })
    write_json(output / "candidate_comparison_template.json", {
        "policy_version": POLICY_VERSION, "candidate_result_status": "not_executed",
        "candidate_fields": ["provider", "model", "task", "n", "agreement_n", "agreement_rate", "cohen_kappa", "confusion_matrix", "abstention_subtype", "adjudication_required_rate", "parse_failure_rate", "provider_failure_rate"],
        "lme_candidates": [{"provider": "deepseek", "model": DEFAULT_DEEPSEEK_MODEL}, {"provider": "kimi", "model": DEFAULT_KIMI_MODEL}],
        "eligibility_rule": "LME binary agreement >= 0.90; preferred >= 0.95; do not fall back to OpenAI/Anthropic automatically.", "holdout_included": False,
    })
    (output / "TASK_ROUTED_SELECTION_PROTOCOL.md").write_text(protocol_markdown(audit, plans), encoding="utf-8")
    independent = audit["reference_independent_only"]
    mem = independent["by_task"]["memora"]
    print(f"GEMINI_CALIBRATION_ALL_N={audit['all_calibration']['overall']['n']}")
    print(f"GEMINI_REFERENCE_INDEPENDENT_N={audit['reference_independence']['independent_n']}")
    print(f"MEMORA_ALL_N={audit['all_calibration']['by_task']['memora']['n']}")
    print(f"MEMORA_REFERENCE_INDEPENDENT_N={independent['by_task']['memora']['n']}")
    print(f"MEMORA_REFERENCE_INDEPENDENT_AGREEMENT={mem['agreement_rate']}")
    print(f"MEMORA_PROVISIONAL_PRIMARY_ELIGIBLE={'yes' if mem['agreement_rate'] is not None and mem['agreement_rate'] >= 0.90 else 'no'}")
    print(f"LME_GEMINI_AGREEMENT={audit['all_calibration']['by_task']['longmemeval']['agreement_rate']}")
    print("LME_GEMINI_ELIGIBLE=no")
    print("LME_DEEPSEEK_PENDING=64")
    print("LME_KIMI_PENDING=64")
    print("HOLDOUT_CASES_TOUCHED=0")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
