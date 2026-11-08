#!/usr/bin/env python3
"""Preflight or execute the frozen final B4 holdout architecture.

Preflight is label-blind and performs no provider calls. Paid execution is
explicitly gated and uses a private append-only ledger separate from all B4
calibration ledgers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.providers.anthropic_provider import DEFAULT_ANTHROPIC_MODEL
from eval.providers.openai_provider import DEFAULT_OPENAI_MODEL
from eval.providers.gemini_provider import DEFAULT_GEMINI_MODEL
from scripts.b4_calibration_execution import AnthropicJudgeAdapter, DurableLedger, GeminiJudgeAdapter, OpenAIJudgeAdapter
from scripts.b4_calibration_runner import build_lme_prompt, build_memora_prompt, parse_judge_output
from scripts.freeze_final_b4_evaluation_protocol import PROTOCOL_VERSION, build_base_plan
from scripts.prepare_final_judge_selection_calibration import digest, read_jsonl, write_json, write_jsonl


ROOT = Path("results/b4_scoring_validation/private/final_b4_v2")
PLAN = Path("results/final_native_evaluation_design/final_b4_v2/locked_holdout_execution_plan.jsonl")
GOLD = Path("results/b4_scoring_validation/private/frontier_adjudication/adjudication_cases.jsonl")


def safe_error(exc: BaseException) -> str:
    return re.sub(r"(?:AIza|sk-ant-|sk-|key=)[A-Za-z0-9_\-]+", "[REDACTED]", str(exc))[:2000]


def payload_index(plan: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in plan:
        path_text, sep, identifier = str(row["private_payload_locator"]).partition("#")
        if not sep or identifier != row["blinded_request_id"]:
            raise ValueError("payload locator drift")
        for payload in read_jsonl(Path(path_text)):
            result[str(payload["blinded_request_id"])] = payload
    return result


def reconstruct(plan_row: dict[str, Any], payloads: dict[str, dict[str, Any]]) -> str:
    payload = payloads[str(plan_row["blinded_request_id"])]
    prompt = build_lme_prompt(payload) if plan_row["task"] == "longmemeval" else build_memora_prompt(payload)
    if digest(payload) != plan_row["payload_sha256"] or digest(prompt) != plan_row["prompt_sha256"] or digest(prompt) != plan_row["request_hash"]:
        raise ValueError(f"frozen prompt/payload drift: {plan_row['blinded_request_id']}")
    return prompt


def slot_base(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("request_slot_id", "blinded_request_id", "task", "provider", "model", "protocol_version", "request_hash")}


def validate_events(events: list[dict[str, Any]], plan: list[dict[str, Any]]) -> dict[str, str]:
    expected = {row["request_slot_id"]: row for row in plan}
    histories = defaultdict(list)
    for event in events:
        identifier = str(event.get("request_slot_id"))
        if identifier not in expected:
            raise ValueError(f"unknown base ledger slot: {identifier}")
        row = expected[identifier]
        for key in ("blinded_request_id", "provider", "model", "protocol_version", "request_hash"):
            if str(event.get(key)) != str(row[key]):
                raise ValueError(f"ledger drift: {identifier}/{key}")
        histories[identifier].append(event)
    states = {}
    for identifier, history in histories.items():
        if sum(event.get("state") == "started" for event in history) > 1:
            raise ValueError(f"more than one attempt: {identifier}")
        terminal = [event for event in history if event.get("state") in {"completed", "failed"}]
        if len(terminal) > 1:
            raise ValueError(f"multiple terminal attempts: {identifier}")
        if history[-1].get("state") == "started" and not terminal:
            raise ValueError(f"started-only slot requires manual resolution: {identifier}")
        if terminal:
            states[identifier] = str(terminal[-1]["state"])
    return states


def append(ledger: DurableLedger, row: dict[str, Any], state: str, **extra: Any) -> None:
    event = slot_base(row)
    event.update({"state": state, "attempt": 1, "timestamp_ns": time.time_ns(), **extra})
    ledger.append(event)


def binary(task: str, decision: str | None) -> str | None:
    if task == "longmemeval":
        return "correct" if decision in {"correct", "abstention_correct"} else "incorrect" if decision in {"incorrect", "abstention_incorrect"} else None
    return "pass" if decision == "pass" else "fail" if decision == "fail" else None


def adapter(provider: str, model: str) -> Any:
    if provider == "openai": return OpenAIJudgeAdapter(model, max_output_tokens=1024)
    if provider == "anthropic": return AnthropicJudgeAdapter(model, max_output_tokens=1024)
    if provider == "gemini": return GeminiJudgeAdapter(model, max_output_tokens=1024)
    raise ValueError(f"unsupported provider: {provider}")


def prepare_lme_escalations(plan: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create Gemini slots only for valid, binary-disagreeing LME primaries."""
    by_case = defaultdict(dict)
    for result in results:
        if result.get("task") == "longmemeval" and result.get("parse_status") == "completed":
            by_case[str(result["blinded_request_id"])][str(result["provider"])] = result
    base_by_id = {str(row["blinded_request_id"]): row for row in plan if row["task"] == "longmemeval"}
    escalations = []
    for identifier, primary in sorted(by_case.items()):
        if set(primary) != {"openai", "anthropic"}:
            continue
        openai_binary = binary("longmemeval", primary["openai"].get("decision"))
        anthropic_binary = binary("longmemeval", primary["anthropic"].get("decision"))
        if not openai_binary or not anthropic_binary or openai_binary == anthropic_binary:
            continue
        base = base_by_id[identifier]
        slot = dict(base)
        slot["provider"] = "gemini"
        slot["model"] = DEFAULT_GEMINI_MODEL
        slot["request_slot_id"] = "b4-final-v2-lme-escalation-slot::" + digest({"protocol_version": PROTOCOL_VERSION, "blinded_request_id": identifier, "provider": "gemini", "model": DEFAULT_GEMINI_MODEL, "request_hash": base["request_hash"]})
        slot["escalation_reason"] = "lme_binary_primary_disagreement"
        escalations.append(slot)
    return escalations


def validate_escalation_events(events: list[dict[str, Any]], slots: list[dict[str, Any]]) -> dict[str, str]:
    expected = {row["request_slot_id"]: row for row in slots}
    states = {}
    for event in events:
        identifier = str(event.get("request_slot_id"))
        if identifier not in expected:
            raise ValueError(f"unknown escalation slot: {identifier}")
        row = expected[identifier]
        for key in ("blinded_request_id", "provider", "model", "protocol_version", "request_hash"):
            if str(event.get(key)) != str(row[key]):
                raise ValueError(f"escalation ledger drift: {identifier}/{key}")
    grouped = defaultdict(list)
    for event in events:
        grouped[str(event["request_slot_id"])].append(event)
    for identifier, history in grouped.items():
        if sum(event.get("state") == "started" for event in history) > 1 or sum(event.get("state") in {"completed", "failed"} for event in history) > 1:
            raise ValueError(f"escalation slot exceeded one attempt: {identifier}")
        if history[-1].get("state") == "started" and not any(event.get("state") in {"completed", "failed"} for event in history):
            raise ValueError(f"escalation started-only slot requires manual resolution: {identifier}")
        states[identifier] = history[-1].get("state")
    return states


def load_gold_for_analysis(path: Path, ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = {str(row["blinded_request_id"]): row for row in read_jsonl(path)}
    if not ids.issubset(rows):
        raise ValueError("holdout IDs missing from frozen adjudication reference")
    return {identifier: rows[identifier] for identifier in ids}


def analyze(results: list[dict[str, Any]], gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_case = defaultdict(dict)
    for row in results:
        by_case[str(row["blinded_request_id"])][row.get("judge_slot", row.get("provider"))] = row
    resolved = []
    lme_primary_pairs = []
    lme_escalations = 0
    unresolved = 0
    for identifier, reference in gold.items():
        task = reference["task"]
        entries = by_case.get(identifier, {})
        if task == "memora":
            candidate = next(iter(entries.values()), None)
            label = binary(task, candidate.get("decision") if candidate else None)
        else:
            a, b = entries.get("openai"), entries.get("anthropic")
            pa, pb = binary(task, a.get("decision") if a else None), binary(task, b.get("decision") if b else None)
            if pa and pb: lme_primary_pairs.append((pa, pb))
            label = pa if pa == pb and pa else None
            if pa and pb and pa != pb:
                lme_escalations += 1
                third = entries.get("gemini")
                pt = binary(task, third.get("decision") if third else None)
                votes = [x for x in (pa, pb, pt) if x]
                if len(votes) == 3 and votes.count("correct") != votes.count("incorrect"):
                    label = "correct" if votes.count("correct") > votes.count("incorrect") else "incorrect"
            if label is None: unresolved += 1
        if label is not None: resolved.append((reference["final_reference_binary_label"], label, task))
    agreement = sum(reference == prediction for reference, prediction, _ in resolved)
    return {"resolved_n": len(resolved), "agreement_n": agreement, "agreement_rate": agreement / len(resolved) if resolved else None, "unresolved_n": unresolved, "lme_dual_primary_agreement_n": sum(a == b for a, b in lme_primary_pairs), "lme_dual_primary_n": len(lme_primary_pairs), "lme_escalation_n": lme_escalations}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid-judge-calls", action="store_true")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--gold", type=Path, default=GOLD)
    args = parser.parse_args(argv)
    if args.execute != args.confirm_paid_judge_calls:
        raise SystemExit("paid execution requires both --execute and --confirm-paid-judge-calls")
    plan = read_jsonl(args.plan)
    if len(plan) != 232 or sum(row["task"] == "memora" for row in plan) != 168 or sum(row["task"] == "longmemeval" for row in plan) != 64:
        raise SystemExit("frozen base plan must contain 168 Memora and 64 LongMemEval slots")
    payloads = payload_index(plan)
    prompts = {row["request_slot_id"]: reconstruct(row, payloads) for row in plan}
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_copy = args.output_root / "execution_plan.jsonl"
    plan_copy.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in plan), encoding="utf-8")
    ledger = DurableLedger(args.output_root / "judge_execution_ledger.jsonl")
    events = ledger.read()
    states = validate_events(events, plan)
    results_path = args.output_root / "holdout_results.jsonl"
    results = read_jsonl(results_path) if results_path.exists() else []
    if not args.execute:
        pending = [row for row in plan if states.get(row["request_slot_id"]) not in {"completed", "failed"}]
    else:
        required_env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
        missing = [env for env in required_env.values() if not os.environ.get(env)]
        if missing: raise SystemExit("paid execution blocked before ledger start: missing credentials")
        for row in plan:
            if states.get(row["request_slot_id"]) in {"completed", "failed"}: continue
            append(ledger, row, "planned"); append(ledger, row, "started")
            try:
                raw = adapter(row["provider"], row["model"]).call(prompts[row["request_slot_id"]], row["task"])
                parsed = parse_judge_output(raw, row["task"])
                response_hash = hashlib.sha256(raw.encode()).hexdigest()
                result = {**slot_base(row), "provider": row["provider"], "judge_slot": row["provider"], "parse_status": "completed", "response_hash": response_hash, **parsed}
                results.append(result); write_jsonl(results_path, results); append(ledger, row, "completed", response_hash=response_hash, parsed_result=parsed)
            except Exception as exc:
                results.append({**slot_base(row), "provider": row["provider"], "judge_slot": row["provider"], "parse_status": "provider_or_parse_failure", "error_class": type(exc).__name__, "error_message": safe_error(exc)})
                write_jsonl(results_path, results); append(ledger, row, "failed", error_class=type(exc).__name__, error_message=safe_error(exc))
        states = validate_events(ledger.read(), plan)
        pending = [row for row in plan if states.get(row["request_slot_id"]) not in {"completed", "failed"}]
        # Gemini is conditionally invoked only after both valid LME primary
        # judgments disagree in binary projection. Its ledger is independent
        # from the base ledger and is never populated during preflight.
        escalation_slots = prepare_lme_escalations(plan, results)
        escalation_plan_path = args.output_root / "lme_escalation_execution_plan.jsonl"
        escalation_plan_text = "".join(json.dumps({k: v for k, v in row.items() if k != "prompt"}, sort_keys=True) + "\n" for row in escalation_slots)
        if escalation_plan_path.exists() and escalation_plan_path.read_text(encoding="utf-8") != escalation_plan_text:
            raise SystemExit("escalation plan drift")
        escalation_plan_path.write_text(escalation_plan_text, encoding="utf-8")
        escalation_ledger = DurableLedger(args.output_root / "lme_escalation_ledger.jsonl")
        escalation_states = validate_escalation_events(escalation_ledger.read(), escalation_slots)
        if escalation_slots:
            if not os.environ.get("GEMINI_API_KEY"):
                raise SystemExit("Gemini escalation blocked before escalation ledger start: GEMINI_API_KEY is missing")
            gemini = adapter("gemini", DEFAULT_GEMINI_MODEL)
            base_prompt_by_id = {str(row["blinded_request_id"]): prompts[row["request_slot_id"]] for row in plan if row["task"] == "longmemeval"}
            for row in escalation_slots:
                if escalation_states.get(row["request_slot_id"]) in {"completed", "failed"}:
                    continue
                append(escalation_ledger, row, "planned"); append(escalation_ledger, row, "started")
                try:
                    raw = gemini.call(base_prompt_by_id[row["blinded_request_id"]], "longmemeval")
                    parsed = parse_judge_output(raw, "longmemeval")
                    response_hash = hashlib.sha256(raw.encode()).hexdigest()
                    results.append({**slot_base(row), "provider": "gemini", "judge_slot": "gemini", "parse_status": "completed", "response_hash": response_hash, **parsed})
                    write_jsonl(results_path, results)
                    append(escalation_ledger, row, "completed", response_hash=response_hash, parsed_result=parsed)
                except Exception as exc:
                    results.append({**slot_base(row), "provider": "gemini", "judge_slot": "gemini", "parse_status": "provider_or_parse_failure", "error_class": type(exc).__name__, "error_message": safe_error(exc)})
                    write_jsonl(results_path, results)
                    append(escalation_ledger, row, "failed", error_class=type(exc).__name__, error_message=safe_error(exc))
    summary = {"protocol_version": PROTOCOL_VERSION, "execution_mode": "paid_execution" if args.execute else "preflight_only", "holdout_total": 200, "holdout_memora": 168, "holdout_longmemeval": 32, "known_base_paid_calls": 232, "max_additional_escalation_calls": 32, "pending_slots": len(pending), "completed_slots": sum(value == "completed" for value in states.values()), "failed_slots": sum(value == "failed" for value in states.values()), "holdout_cases_touched": 0 if not args.execute else 200, "paid_calls_performed": 0 if not args.execute else sum(value == "completed" for value in states.values())}
    if args.execute:
        gold = load_gold_for_analysis(args.gold, {row["blinded_request_id"] for row in plan})
        summary.update(analyze(results, gold))
        write_json(args.output_root / "holdout_summary.json", summary)
        write_jsonl(args.output_root / "longmemeval_holdout_results.jsonl", [row for row in results if row["task"] == "longmemeval"])
        write_jsonl(args.output_root / "memora_holdout_results.jsonl", [row for row in results if row["task"] == "memora"])
        (args.output_root / "HOLDOUT_VALIDATION_REPORT.md").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_json(args.output_root / "preflight_summary.json", summary)
    print("FINAL_PROTOCOL_VERSION=final-native-evaluation-v2")
    print("HOLDOUT_TOTAL=200"); print("HOLDOUT_MEMORA=168"); print("HOLDOUT_LONGMEMEVAL=32")
    print("KNOWN_BASE_PAID_CALLS=232"); print("MAX_ADDITIONAL_ESCALATION_CALLS=32")
    print("HOLDOUT_CASES_TOUCHED=0"); print(f"PAID_CALLS_PERFORMED={summary['paid_calls_performed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
