#!/usr/bin/env python3
"""Analyze one isolated B4 judge phase from its private plan and ledger."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_b4_private_payloads import read_jsonl


ERROR_STATES = {"provider_error", "retryable_error", "permanently_failed"}


def _task_summary(task: str, plan: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    task_plan = [row for row in plan if row.get("task") == task]
    slot_ids = {str(row["request_slot_id"]) for row in task_plan}
    task_events = [row for row in events if str(row.get("request_slot_id")) in slot_ids]
    latest: dict[str, dict[str, Any]] = {}
    for event in task_events:
        latest[str(event["request_slot_id"])] = event
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_plan:
        by_case[str(row["blinded_request_id"])].append(row)

    exact = disagreement = adjudication = low_confidence = escalation = valid_pairs = pass_fail_exact = 0
    abstentions: Counter[str] = Counter()
    subtype_cases: Counter[str] = Counter()
    escalation_cases: list[dict[str, Any]] = []
    for case_id, slots in by_case.items():
        if task == "memora":
            subtype_cases[str(slots[0].get("subtype", "unknown"))] += 1
        terminal = [latest.get(str(slot["request_slot_id"])) for slot in slots]
        completed = [event for event in terminal if event and event.get("state") == "completed"]
        decisions = [event["parsed_result"]["decision"] for event in completed]
        confidences = [event["parsed_result"]["confidence"] for event in completed]
        pair_valid = len(completed) == 2
        if pair_valid:
            valid_pairs += 1
            if decisions[0] == decisions[1]:
                exact += 1
                if task == "memora" and decisions[0] in {"pass", "fail"}:
                    pass_fail_exact += 1
            else:
                disagreement += 1
        has_adjudication = "adjudication_required" in decisions
        has_low = "low" in confidences
        adjudication += has_adjudication
        low_confidence += has_low
        if task == "longmemeval":
            abstentions.update(decision for decision in decisions if decision.startswith("abstention_"))
        if not pair_valid or (pair_valid and decisions[0] != decisions[1]) or has_adjudication or has_low:
            escalation += 1
            reasons = []
            if not pair_valid:
                reasons.append("incomplete_or_failed_pair")
            if pair_valid and decisions[0] != decisions[1]:
                reasons.append("judge_disagreement")
            if has_adjudication:
                reasons.append("adjudication_required")
            if has_low:
                reasons.append("low_confidence")
            escalation_cases.append({
                "blinded_request_id": case_id,
                "task": task,
                "subtype": slots[0].get("subtype"),
                "reasons": reasons,
            })

    parse_failures = sum(event.get("state") == "parse_failed" for event in latest.values())
    provider_error_slots = sum(event.get("state") in ERROR_STATES for event in latest.values())
    provider_error_events = sum(event.get("state") in ERROR_STATES for event in task_events)
    result: dict[str, Any] = {
        "task": task,
        "total_cases": len(by_case),
        "total_request_slots": len(task_plan),
        "valid_paired_cases": valid_pairs,
        "exact_decision_agreement_n": exact,
        "exact_decision_agreement_rate": exact / valid_pairs if valid_pairs else None,
        "disagreement_n": disagreement,
        "disagreement_rate": disagreement / valid_pairs if valid_pairs else None,
        "adjudication_required_case_n": adjudication,
        "low_confidence_case_n": low_confidence,
        "parse_failure_slot_n": parse_failures,
        "provider_error_slot_n": provider_error_slots,
        "provider_error_event_n": provider_error_events,
        "escalation_case_n": escalation,
        "escalation_cases": escalation_cases,
    }
    if task == "longmemeval":
        result["abstention_decision_counts"] = dict(sorted(abstentions.items()))
    else:
        result["pass_fail_exact_agreement_n"] = pass_fail_exact
        result["subtype_case_counts"] = dict(sorted(subtype_cases.items()))
    return result


def analyze_phase(plan: list[dict[str, Any]], events: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    lme = _task_summary("longmemeval", plan, events)
    memora = _task_summary("memora", plan, events)
    return {
        "schema_version": "b4-calibration-phase-analysis-v1",
        "phase": phase,
        "longmemeval": lme,
        "memora": memora,
        "combined": {
            "total_cases": lme["total_cases"] + memora["total_cases"],
            "total_request_slots": lme["total_request_slots"] + memora["total_request_slots"],
            "valid_paired_cases": lme["valid_paired_cases"] + memora["valid_paired_cases"],
            "exact_decision_agreement_n": lme["exact_decision_agreement_n"] + memora["exact_decision_agreement_n"],
            "disagreement_n": lme["disagreement_n"] + memora["disagreement_n"],
            "parse_failure_slot_n": lme["parse_failure_slot_n"] + memora["parse_failure_slot_n"],
            "provider_error_slot_n": lme["provider_error_slot_n"] + memora["provider_error_slot_n"],
            "escalation_case_n": lme["escalation_case_n"] + memora["escalation_case_n"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--phase", default="cheap_judges")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--escalation-output", type=Path)
    args = parser.parse_args(argv)
    full_plan = args.phase_dir / "full_validation_execution_plan.jsonl"
    plan_path = full_plan if full_plan.exists() else args.phase_dir / "calibration_execution_plan.jsonl"
    plan = read_jsonl(plan_path)
    ledger_path = args.phase_dir / "judge_execution_ledger.jsonl"
    events = read_jsonl(ledger_path) if ledger_path.exists() else []
    summary = analyze_phase(plan, events, args.phase)
    output = args.output or args.phase_dir / "phase_analysis_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    escalation_output = args.escalation_output or args.phase_dir / "escalation_queue.jsonl"
    escalation_rows = summary["longmemeval"]["escalation_cases"] + summary["memora"]["escalation_cases"]
    escalation_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in escalation_rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
