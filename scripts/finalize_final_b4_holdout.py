#!/usr/bin/env python3
"""Offline, recovery-aware finalizer for the executed final-v2 holdout."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_final_judge_selection_calibration import read_jsonl, write_json, write_jsonl


PROTOCOL_VERSION = "final-native-evaluation-v2"
RECOVERY_PROTOCOLS = {
    "final-native-evaluation-v2-execution-recovery-1",
    "final-native-evaluation-v2-execution-recovery-2",
}
DEFAULT_PLAN = Path("results/b4_scoring_validation/private/final_b4_v2/execution_plan.jsonl")
DEFAULT_LEDGER = Path("results/b4_scoring_validation/private/final_b4_v2/judge_execution_ledger.jsonl")
DEFAULT_RESULTS = Path("results/b4_scoring_validation/private/final_b4_v2/holdout_results.jsonl")
DEFAULT_GOLD = Path("results/b4_scoring_validation/private/frontier_adjudication/adjudication_cases.jsonl")
DEFAULT_OUTPUT = Path("results/b4_scoring_validation/private/final_b4_v2/finalized_holdout_results.jsonl")
DEFAULT_PUBLIC = Path("results/final_native_evaluation_design/final_b4_v2")


def identity_keys(row: dict[str, Any]) -> tuple[str, ...]:
    return ("request_slot_id", "blinded_request_id", "provider", "model", "protocol_version", "request_hash")


def validate_lifecycle(plan: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    expected = {str(row["request_slot_id"]): row for row in plan}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        slot_id = str(event.get("request_slot_id"))
        if slot_id not in expected:
            raise ValueError(f"ledger contains unknown slot: {slot_id}")
        row = expected[slot_id]
        for key in identity_keys(row):
            if str(event.get(key)) != str(row.get(key)):
                raise ValueError(f"ledger identity drift for {slot_id}: {key}")
        attempt = event.get("attempt")
        if not isinstance(attempt, int) or attempt not in {0, 1, 2}:
            raise ValueError(f"invalid attempt for {slot_id}")
        if attempt == 0 and event.get("state") != "planned":
            raise ValueError(f"attempt-0 may only be planned: {slot_id}")
        if attempt == 2 and event.get("state") not in {"started", "completed", "failed", "recovery_authorized"}:
            raise ValueError(f"invalid attempt-2 state for {slot_id}")
        grouped[slot_id].append(event)
    for slot_id, history in grouped.items():
        indexed = list(enumerate(history))
        authorizations: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, event in indexed:
            if event.get("state") != "recovery_authorized":
                continue
            recovery_attempt = event.get("recovery_attempt")
            if not isinstance(recovery_attempt, int) or recovery_attempt < 2 or recovery_attempt > 2:
                raise ValueError(f"invalid recovery_attempt for {slot_id}")
            authorizations[recovery_attempt].append((index, event))
        for recovery_attempt, auths in authorizations.items():
            if len(auths) != 1:
                raise ValueError(f"duplicate recovery authorization for attempt {recovery_attempt}: {slot_id}")

        execution_attempts = {
            int(event["attempt"])
            for event in history
            if event.get("state") in {"started", "completed", "failed"}
        }
        if any(attempt >= 3 for attempt in execution_attempts):
            raise ValueError(f"attempt 3+ is invalid: {slot_id}")

        attempt1_completed = any(
            event.get("attempt") == 1 and event.get("state") == "completed"
            for event in history
        )
        if attempt1_completed and (2 in execution_attempts or 2 in authorizations):
            raise ValueError(f"completed attempt-1 cannot be superseded: {slot_id}")

        for attempt in sorted(execution_attempts):
            starts = [(index, event) for index, event in indexed if event.get("attempt") == attempt and event.get("state") == "started"]
            terminals = [(index, event) for index, event in indexed if event.get("attempt") == attempt and event.get("state") in {"completed", "failed"}]
            if len(starts) != 1:
                raise ValueError(f"attempt {attempt} requires exactly one started event: {slot_id}")
            # A previous attempt may be intentionally incomplete when an
            # explicit recovery authorization bridges to the next attempt.
            if attempt == 1 and 2 in execution_attempts and not terminals:
                pass
            elif len(terminals) != 1:
                raise ValueError(f"attempt {attempt} requires exactly one terminal event: {slot_id}")
            if terminals and starts[0][0] >= terminals[0][0]:
                raise ValueError(f"attempt {attempt} terminal precedes start: {slot_id}")
            if attempt > 1:
                auths = authorizations.get(attempt, [])
                if len(auths) != 1:
                    raise ValueError(f"attempt-{attempt} lacks exactly one valid recovery authorization: {slot_id}")
                auth_index, auth = auths[0]
                if auth.get("attempt") != attempt - 1:
                    raise ValueError(f"recovery authorization must refer to attempt {attempt - 1}: {slot_id}")
                if auth.get("recovery_protocol_version") not in RECOVERY_PROTOCOLS:
                    raise ValueError(f"unsupported recovery protocol: {slot_id}")
                if auth_index >= starts[0][0]:
                    raise ValueError(f"recovery authorization occurs after attempt-{attempt} execution: {slot_id}")

        for recovery_attempt in authorizations:
            if recovery_attempt not in execution_attempts:
                raise ValueError(f"recovery authorization has no attempt-{recovery_attempt} lifecycle: {slot_id}")
    return grouped


def materialize_slot_results(plan: list[dict[str, Any]], events: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_lifecycle(plan, events)
    expected = {str(row["request_slot_id"]): row for row in plan}
    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        slot_id = str(row.get("request_slot_id"))
        if slot_id not in expected:
            raise ValueError(f"result contains unknown slot: {slot_id}")
        plan_row = expected[slot_id]
        for key in identity_keys(plan_row):
            if str(row.get(key)) != str(plan_row.get(key)):
                raise ValueError(f"result identity drift for {slot_id}: {key}")
        by_slot[slot_id].append(row)
    materialized = []
    for plan_row in plan:
        slot_id = str(plan_row["request_slot_id"])
        valid = [row for row in by_slot.get(slot_id, []) if row.get("parse_status") == "completed"]
        if len(valid) > 1:
            raise ValueError(f"multiple valid completed semantic results for {slot_id}")
        if valid:
            row = dict(valid[0])
            row["materialization_status"] = "completed_latest_valid_attempt"
            materialized.append(row)
        else:
            materialized.append({**{key: plan_row[key] for key in identity_keys(plan_row)}, "task": plan_row["task"], "subtype": plan_row.get("subtype"), "materialization_status": "unresolved_no_valid_completed_result"})
    if len(materialized) != len(plan):
        raise ValueError("materialized slot count drift")
    return materialized


def binary(task: str, decision: str | None) -> str | None:
    if task == "longmemeval":
        return "correct" if decision in {"correct", "abstention_correct"} else "incorrect" if decision in {"incorrect", "abstention_incorrect"} else None
    return "pass" if decision == "pass" else "fail" if decision == "fail" else None


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted(set(a for a, _ in pairs) | set(b for _, b in pairs))
    po = sum(a == b for a, b in pairs) / len(pairs)
    pe = sum(sum(a == label for a, _ in pairs) * sum(b == label for _, b in pairs) for label in labels) / len(pairs) ** 2
    return None if pe == 1 else (po - pe) / (1 - pe)


def metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    normalized = [("correct" if a == "pass" else "incorrect" if a == "fail" else a, "correct" if b == "pass" else "incorrect" if b == "fail" else b) for a, b in pairs]
    return {"n": len(pairs), "agreement_n": sum(a == b for a, b in normalized), "agreement_rate": sum(a == b for a, b in normalized) / len(normalized) if normalized else None, "cohen_kappa": kappa(normalized), "confusion_matrix": {"tp": sum(a == "correct" and b == "correct" for a, b in normalized), "tn": sum(a == "incorrect" and b == "incorrect" for a, b in normalized), "fp": sum(a == "incorrect" and b == "correct" for a, b in normalized), "fn": sum(a == "correct" and b == "incorrect" for a, b in normalized)}}


def resolve_cases(materialized: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold = {str(row["blinded_request_id"]): row for row in gold_rows}
    case_gold = {str(row["blinded_request_id"]): row for row in gold_rows}
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in materialized:
        by_case[str(row["blinded_request_id"])][str(row["provider"])] = row
    mem_pairs = []
    lme_pairs = []
    lme_primary_pairs = []
    lme_needed = 0
    lme_completed = 0
    unresolved = 0
    provider_failures = sum(row.get("materialization_status") != "completed_latest_valid_attempt" for row in materialized)
    parse_failures = 0
    case_rows = []
    for identifier, reference in case_gold.items():
        entries = by_case.get(identifier, {})
        if reference["task"] == "memora":
            prediction = binary("memora", entries.get("gemini", {}).get("decision"))
            if prediction is not None:
                mem_pairs.append((reference["final_reference_binary_label"], prediction))
        else:
            openai = binary("longmemeval", entries.get("openai", {}).get("decision"))
            anthropic = binary("longmemeval", entries.get("anthropic", {}).get("decision"))
            final = None
            if openai and anthropic:
                lme_primary_pairs.append((openai, anthropic))
                if openai == anthropic:
                    final = openai
                else:
                    lme_needed += 1
                    gemini = binary("longmemeval", entries.get("gemini", {}).get("decision"))
                    if gemini:
                        lme_completed += 1
                        votes = [openai, anthropic, gemini]
                        if votes.count("correct") != votes.count("incorrect"):
                            final = "correct" if votes.count("correct") > votes.count("incorrect") else "incorrect"
            if final is not None:
                lme_pairs.append((reference["final_reference_binary_label"], final))
            else:
                unresolved += 1
        case_rows.append({"blinded_request_id": identifier, "task": reference["task"], "reference_binary_label": reference["final_reference_binary_label"], "resolution_status": "resolved" if ((reference["task"] == "memora" and identifier in by_case and binary("memora", entries.get("gemini", {}).get("decision")) is not None) or (reference["task"] == "longmemeval" and identifier in by_case and ((binary("longmemeval", entries.get("openai", {}).get("decision")) == binary("longmemeval", entries.get("anthropic", {}).get("decision")) and binary("longmemeval", entries.get("openai", {}).get("decision")) is not None) or binary("longmemeval", entries.get("gemini", {}).get("decision")) is not None))) else "escalation_pending_or_unresolved"})
    summary = {"memora": metrics(mem_pairs), "longmemeval": metrics(lme_pairs), "overall_resolved": metrics(mem_pairs + lme_pairs), "lme_primary_dual": metrics(lme_primary_pairs), "lme_escalation_needed_n": lme_needed, "lme_escalation_completed_n": lme_completed, "unresolved_n": unresolved, "provider_failure_n": provider_failures, "parse_failure_n": parse_failures}
    return summary, case_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--public-output", type=Path, default=DEFAULT_PUBLIC)
    args = parser.parse_args(argv)
    plan = read_jsonl(args.plan)
    if len(plan) != 232 or len({row["request_slot_id"] for row in plan}) != 232:
        raise SystemExit("expected exactly 232 frozen planned slots")
    ledger = read_jsonl(args.ledger)
    result_rows = read_jsonl(args.results)
    materialized = materialize_slot_results(plan, ledger, result_rows)
    holdout_ids = {row["blinded_request_id"] for row in plan}
    gold_all = read_jsonl(args.gold)
    gold = [row for row in gold_all if row["blinded_request_id"] in holdout_ids]
    if len(gold) != 200:
        raise SystemExit("frozen holdout gold did not contain exactly 200 identities")
    summary, case_rows = resolve_cases(materialized, gold)
    final_pass = "pending" if summary["unresolved_n"] or summary["lme_escalation_needed_n"] > summary["lme_escalation_completed_n"] else ("yes" if summary["memora"]["agreement_rate"] >= .90 and summary["longmemeval"]["agreement_rate"] >= .90 and summary["overall_resolved"]["agreement_rate"] >= .90 else "no")
    summary.update({"protocol_version": PROTOCOL_VERSION, "finalization_mode": "offline", "holdout_total": 200, "holdout_gold_n": 200, "api_calls_performed": 0, "final_b4_holdout_pass": final_pass, "b4_methodology_closed": final_pass == "yes"})
    args.output.parent.mkdir(parents=True, exist_ok=True); write_jsonl(args.output, materialized)
    args.public_output.mkdir(parents=True, exist_ok=True)
    write_json(args.public_output / "holdout_summary.json", summary)
    (args.public_output / "HOLDOUT_VALIDATION_REPORT.md").write_text("# Final B4 holdout validation\n\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n\nCase resolution rows: " + str(len(case_rows)) + ".\n", encoding="utf-8")
    print("FINALIZATION_MODE=offline"); print("API_CALLS_PERFORMED=0"); print("HOLDOUT_TOTAL=200"); print("MEMORA_HOLDOUT_N=168"); print("LME_HOLDOUT_N=32")
    print(f"MEMORA_HOLDOUT_AGREEMENT={summary['memora']['agreement_rate']}"); print(f"LME_HOLDOUT_AGREEMENT={summary['longmemeval']['agreement_rate']}"); print(f"OVERALL_HOLDOUT_AGREEMENT={summary['overall_resolved']['agreement_rate']}")
    print(f"LME_DUAL_PRIMARY_AGREEMENT={summary['lme_primary_dual']['agreement_rate']}"); print(f"LME_ESCALATION_NEEDED={summary['lme_escalation_needed_n']}"); print(f"LME_ESCALATION_COMPLETED={summary['lme_escalation_completed_n']}"); print(f"UNRESOLVED_N={summary['unresolved_n']}"); print(f"FINAL_B4_HOLDOUT_PASS={final_pass}"); print(f"B4_METHODOLOGY_CLOSED={'yes' if final_pass == 'yes' else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
