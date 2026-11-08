#!/usr/bin/env python3
"""Run one explicitly gated candidate judge on the frozen 404-case calibration.

The default operation is offline preflight.  Holdout artifacts are never read
by this module.  Paid execution requires both ``--execute`` and
``--confirm-paid-judge-calls`` and uses one append-only, one-attempt slot per
calibration case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b4_calibration_execution import (
    DurableLedger,
    GeminiJudgeAdapter,
    estimate_input_tokens,
)
from scripts.b4_calibration_runner import build_lme_prompt, build_memora_prompt, parse_judge_output
from scripts.prepare_b4_frontier_escalation import GEMINI_LIMIT_SOURCE, VERIFIED_GEMINI_MAX_INPUT_TOKENS
from scripts.prepare_final_judge_selection_calibration import (
    EXPECTED_GOLD_N,
    SOURCE_PROTOCOL_VERSION,
    digest,
    file_digest,
    read_jsonl,
    write_json,
    write_jsonl,
)


CANDIDATE_ID = "gemini-3-flash-preview"
CANDIDATE_PROVIDER = "gemini"
CANDIDATE_MODEL = "gemini-3-flash-preview"
PROTOCOL_VERSION = "final-judge-selection-calibration-v1"
CALIBRATION_N = 404
MAX_ATTEMPTS = 1
GEMINI_INPUT_LIMIT_SOURCE = GEMINI_LIMIT_SOURCE
TERMINAL_STATES = {"completed", "failed"}


def candidate_slot_id(blinded_request_id: str, provider: str, model: str, prompt_hash: str) -> str:
    return "b4-candidate-slot::" + digest({
        "blinded_request_id": blinded_request_id,
        "candidate_id": CANDIDATE_ID,
        "provider": provider,
        "model": model,
        "prompt_hash": prompt_hash,
        "protocol_version": PROTOCOL_VERSION,
    })


def safe_error(exc: BaseException) -> str:
    return re.sub(r"(?:AIza|sk-ant-|sk-|key=)[A-Za-z0-9_\-]+", "[REDACTED]", str(exc))[:2000]


def reconstruct_prompt(task: str, payload: dict[str, Any]) -> str:
    return build_lme_prompt(payload) if task == "longmemeval" else build_memora_prompt(payload)


def payload_from_locator(locator: str) -> tuple[Path, str]:
    path_text, separator, identifier = locator.partition("#")
    if not separator or not path_text or not identifier:
        raise ValueError(f"invalid private payload locator: {locator}")
    return Path(path_text), identifier


def load_payload_index(plan: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    paths = sorted({payload_from_locator(str(row["private_payload_locator"]))[0] for row in plan})
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            identifier = str(row["blinded_request_id"])
            if identifier in index:
                raise ValueError(f"duplicate private payload ID: {identifier}")
            index[identifier] = row
    return index


def load_calibration_cases(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != CALIBRATION_N:
        raise ValueError(f"expected {CALIBRATION_N} calibration cases, found {len(rows)}")
    result = {str(row["blinded_request_id"]): row for row in rows}
    if len(result) != CALIBRATION_N:
        raise ValueError("duplicate calibration case IDs")
    return result


def build_execution_plan(
    plan_path: Path,
    calibration_path: Path,
    provider: str,
    model: str,
    max_input_tokens: int,
) -> list[dict[str, Any]]:
    if provider != CANDIDATE_PROVIDER or model != CANDIDATE_MODEL:
        raise ValueError(f"candidate drift: expected {CANDIDATE_PROVIDER}/{CANDIDATE_MODEL}, found {provider}/{model}")
    if max_input_tokens != VERIFIED_GEMINI_MAX_INPUT_TOKENS:
        raise ValueError(f"candidate input limit must be repository-verified {VERIFIED_GEMINI_MAX_INPUT_TOKENS}, found {max_input_tokens}")
    source_plan = read_jsonl(plan_path)
    if len(source_plan) != CALIBRATION_N:
        raise ValueError(f"expected {CALIBRATION_N} frozen candidate plan rows, found {len(source_plan)}")
    calibration = load_calibration_cases(calibration_path)
    if set(row["blinded_request_id"] for row in source_plan) != set(calibration):
        raise ValueError("candidate plan/calibration case identity mismatch")
    payloads = load_payload_index(source_plan)
    slots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in source_plan:
        identifier = str(row["blinded_request_id"])
        if identifier not in payloads:
            raise ValueError(f"missing private payload for {identifier}")
        payload = payloads[identifier]
        if payload.get("blinded_request_id") != identifier:
            raise ValueError(f"payload identity drift for {identifier}")
        payload_path, locator_id = payload_from_locator(str(row["private_payload_locator"]))
        if locator_id != identifier:
            raise ValueError(f"payload locator identity drift for {identifier}")
        if not payload_path.exists():
            raise ValueError(f"private payload path missing: {payload_path}")
        if digest(payload) != row["payload_sha256"]:
            raise ValueError(f"private payload hash drift for {identifier}")
        prompt = reconstruct_prompt(str(row["task"]), payload)
        # The frozen preparation artifact hashes canonical JSON string values
        # through its shared digest function; preserve that exact identity.
        actual_prompt_hash = digest(prompt)
        if actual_prompt_hash != row["semantic_prompt_sha256"]:
            raise ValueError(f"semantic prompt hash drift for {identifier}")
        measured = estimate_input_tokens(prompt, model)
        if measured["estimated_tokens"] > max_input_tokens:
            raise ValueError(f"input-limit blocker for {identifier}: {measured['estimated_tokens']} > {max_input_tokens}")
        slot_id = candidate_slot_id(identifier, provider, model, actual_prompt_hash)
        if slot_id in seen:
            raise ValueError(f"duplicate candidate request slot: {slot_id}")
        seen.add(slot_id)
        slots.append({
            "request_slot_id": slot_id,
            "blinded_request_id": identifier,
            "task": row["task"],
            "subtype": row.get("subtype"),
            "candidate_id": CANDIDATE_ID,
            "judge_provider": provider,
            "judge_model": model,
            "protocol_version": PROTOCOL_VERSION,
            "request_hash": actual_prompt_hash,
            "payload_sha256": row["payload_sha256"],
            "semantic_prompt_hash": actual_prompt_hash,
            "prompt_chars": len(prompt),
            "estimated_input_tokens": measured["estimated_tokens"],
            "input_token_estimation_method": measured["method"],
            "configured_max_input_tokens": max_input_tokens,
            "input_limit_source": GEMINI_INPUT_LIMIT_SOURCE,
            "prompt": prompt,
        })
    return sorted(slots, key=lambda row: row["blinded_request_id"])


def write_frozen_plan(path: Path, slots: list[dict[str, Any]]) -> None:
    public_rows = [{key: value for key, value in row.items() if key != "prompt"} for row in slots]
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in public_rows)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError(f"execution plan drift: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def validate_ledger(events: list[dict[str, Any]], slots: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {str(slot["request_slot_id"]): slot for slot in slots}
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        slot_id = str(event.get("request_slot_id"))
        if slot_id not in expected:
            raise ValueError(f"ledger contains unknown candidate slot: {slot_id}")
        slot = expected[slot_id]
        for key in ("blinded_request_id", "candidate_id", "judge_provider", "judge_model", "protocol_version", "request_hash"):
            if str(event.get(key)) != str(slot[key]):
                raise ValueError(f"ledger identity drift for {slot_id}: {key}")
        histories[slot_id].append(event)
    states: dict[str, str] = {}
    for slot_id, history in histories.items():
        if sum(event.get("state") == "started" for event in history) > MAX_ATTEMPTS:
            raise ValueError(f"candidate slot exceeded one paid attempt: {slot_id}")
        terminal = [event for event in history if event.get("state") in TERMINAL_STATES]
        if len(terminal) > 1:
            raise ValueError(f"candidate slot has multiple terminal attempts: {slot_id}")
        if history[-1].get("state") == "started" and not terminal:
            states[slot_id] = "blocked_started_only"
        elif terminal:
            states[slot_id] = str(terminal[-1]["state"])
        else:
            states[slot_id] = str(history[-1].get("state"))
    completed = sum(state == "completed" for state in states.values())
    failed = sum(state == "failed" for state in states.values())
    blocked = sum(state == "blocked_started_only" for state in states.values())
    return {"histories": histories, "states": states, "completed": completed, "failed": failed, "blocked": blocked}


def append_event(ledger: DurableLedger, slot: dict[str, Any], state: str, **extra: Any) -> None:
    event = {key: slot[key] for key in ("request_slot_id", "blinded_request_id", "candidate_id", "judge_provider", "judge_model", "protocol_version", "request_hash")}
    event.update({"state": state, "attempt": 1, "timestamp_ns": time.time_ns(), **extra})
    ledger.append(event)


def binary_projection(task: str, decision: str | None) -> str | None:
    if task == "longmemeval":
        if decision in {"correct", "abstention_correct"}:
            return "correct"
        if decision in {"incorrect", "abstention_incorrect"}:
            return "incorrect"
    elif task == "memora":
        if decision == "pass":
            return "pass"
        if decision == "fail":
            return "fail"
    return None


def cohen_kappa(reference: list[str], prediction: list[str]) -> float | None:
    if not reference or len(reference) != len(prediction):
        return None
    labels = sorted(set(reference) | set(prediction))
    po = sum(a == b for a, b in zip(reference, prediction)) / len(reference)
    pe = sum(reference.count(label) * prediction.count(label) for label in labels) / (len(reference) ** 2)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def summarize_candidate(calibration: dict[str, dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["blinded_request_id"]): row for row in results if row.get("parse_status") == "completed"}
    if len(by_id) != len(results):
        raise ValueError("duplicate or invalid candidate result IDs")
    metrics: dict[str, Any] = {}
    all_pairs: list[tuple[str, str]] = []
    for task in ("longmemeval", "memora"):
        pairs = []
        unresolved = 0
        parse_failures = sum(row.get("parse_status") == "parse_failure" for row in results if row.get("task") == task)
        provider_failures = sum(row.get("parse_status") == "provider_failure" for row in results if row.get("task") == task)
        for identifier, gold in calibration.items():
            if gold["task"] != task:
                continue
            result = by_id.get(identifier)
            if not result:
                continue
            decision = result.get("decision")
            predicted = binary_projection(task, decision)
            if predicted is None:
                unresolved += 1
            else:
                pairs.append((str(gold["gold_binary_label"]), predicted))
                all_pairs.append((str(gold["gold_binary_label"]), predicted))
        reference = [a for a, _ in pairs]
        prediction = [b for _, b in pairs]
        confusion = {"reference_correct_or_pass_prediction_correct_or_pass": 0, "reference_correct_or_pass_prediction_incorrect_or_fail": 0, "reference_incorrect_or_fail_prediction_correct_or_pass": 0, "reference_incorrect_or_fail_prediction_incorrect_or_fail": 0}
        for ref, pred in pairs:
            if ref in {"correct", "pass"} and pred in {"correct", "pass"}: confusion["reference_correct_or_pass_prediction_correct_or_pass"] += 1
            elif ref in {"correct", "pass"}: confusion["reference_correct_or_pass_prediction_incorrect_or_fail"] += 1
            elif pred in {"correct", "pass"}: confusion["reference_incorrect_or_fail_prediction_correct_or_pass"] += 1
            else: confusion["reference_incorrect_or_fail_prediction_incorrect_or_fail"] += 1
        abstention_pairs = []
        if task == "longmemeval":
            for identifier, gold in calibration.items():
                if gold["task"] == task and gold.get("gold_raw_label") in {"abstention_correct", "abstention_incorrect"} and identifier in by_id:
                    abstention_pairs.append((gold["gold_raw_label"], by_id[identifier].get("decision")))
        metrics[task] = {
            "n": len(pairs), "agreement_n": sum(ref == pred for ref, pred in pairs),
            "disagreement_n": sum(ref != pred for ref, pred in pairs),
            "agreement_rate": sum(ref == pred for ref, pred in pairs) / len(pairs) if pairs else None,
            "cohen_kappa": cohen_kappa(reference, prediction), "confusion": confusion,
            "adjudication_required_n": sum(row.get("decision") == "adjudication_required" for row in results if row.get("task") == task),
            "unresolved_n": unresolved, "parse_failure_n": parse_failures, "provider_failure_n": provider_failures,
            "abstention_subtype": {"eligible_n": len(abstention_pairs), "raw_exact_n": sum(a == b for a, b in abstention_pairs), "raw_exact_rate": sum(a == b for a, b in abstention_pairs) / len(abstention_pairs) if abstention_pairs else None},
        }
    ref_all, pred_all = zip(*all_pairs) if all_pairs else ([], [])
    overall = {
        "valid_judged_n": len(all_pairs), "agreement_n": sum(a == b for a, b in all_pairs),
        "disagreement_n": sum(a != b for a, b in all_pairs),
        "agreement_rate": sum(a == b for a, b in all_pairs) / len(all_pairs) if all_pairs else None,
        "cohen_kappa": cohen_kappa(list(ref_all), list(pred_all)),
        "adjudication_required_n": sum(row.get("decision") == "adjudication_required" for row in results),
        "parse_failure_n": sum(row.get("parse_status") == "parse_failure" for row in results),
        "provider_failure_n": sum(row.get("parse_status") == "provider_failure" for row in results),
    }
    overall["adjudication_required_rate"] = overall["adjudication_required_n"] / CALIBRATION_N
    overall["parse_failure_rate"] = overall["parse_failure_n"] / CALIBRATION_N
    overall["provider_failure_rate"] = overall["provider_failure_n"] / CALIBRATION_N
    eligible = all(metrics[task]["agreement_rate"] is not None and metrics[task]["agreement_rate"] >= 0.90 for task in ("longmemeval", "memora")) and overall["agreement_rate"] is not None and overall["agreement_rate"] >= 0.90
    preferred = eligible and overall["agreement_rate"] >= 0.95
    return {"candidate_id": CANDIDATE_ID, "protocol_version": PROTOCOL_VERSION, "overall": overall, "by_task": metrics, "eligible": eligible, "preferred_threshold_met": preferred, "selection_rule": "all three binary/pass-fail agreement gates >= 0.90; preferred overall >= 0.95"}


def write_report(path: Path, summary: dict[str, Any], preflight: dict[str, Any]) -> None:
    overall = summary["overall"]
    path.write_text(
        "# Gemini candidate calibration report\n\n"
        f"Candidate: `{CANDIDATE_ID}`. Execution mode: `{preflight['execution_mode']}`. Holdout cases touched: 0.\n\n"
        f"Calibration cases: {CALIBRATION_N}. Completed results: {preflight['completed']}. Failed results: {preflight['failed']}. Pending slots: {preflight['pending']}.\n\n"
        f"Valid judged cases: {overall['valid_judged_n']}; agreement: {overall['agreement_n']}; disagreements: {overall['disagreement_n']}; agreement rate: {overall['agreement_rate']}; kappa: {overall['cohen_kappa']}.\n\n"
        f"Adjudication-required: {overall['adjudication_required_n']}; parse failures: {overall['parse_failure_n']}; provider failures: {overall['provider_failure_n']}.\n\n"
        f"Eligibility: `{summary['eligible']}`. Preferred threshold: `{summary['preferred_threshold_met']}`.\n\n"
        "This report contains calibration-only aggregates. The locked holdout was not read or evaluated.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-provider", default=CANDIDATE_PROVIDER)
    parser.add_argument("--candidate-model", default=CANDIDATE_MODEL)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid-judge-calls", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=VERIFIED_GEMINI_MAX_INPUT_TOKENS)
    parser.add_argument("--calibration-cases", type=Path, default=Path("results/final_native_evaluation_design/judge_selection/calibration_cases.jsonl"))
    parser.add_argument("--execution-plan", type=Path, default=Path("results/final_native_evaluation_design/judge_selection/candidate_execution_plan.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/b4_scoring_validation/private/final_judge_selection/gemini-3-flash-preview"))
    args = parser.parse_args(argv)
    if args.execute != args.confirm_paid_judge_calls:
        raise SystemExit("paid execution requires both --execute and --confirm-paid-judge-calls")
    slots = build_execution_plan(args.execution_plan, args.calibration_cases, args.candidate_provider, args.candidate_model, args.max_input_tokens)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_frozen_plan(args.output_dir / "execution_plan.jsonl", slots)
    # Create the append-only/result files during preflight without adding any
    # execution events or semantic results.
    ledger_path = args.output_dir / "judge_execution_ledger.jsonl"
    results_path = args.output_dir / "candidate_results.jsonl"
    if not ledger_path.exists():
        ledger_path.write_text("", encoding="utf-8")
    if not results_path.exists():
        results_path.write_text("", encoding="utf-8")
    ledger = DurableLedger(ledger_path)
    events = ledger.read()
    state = validate_ledger(events, slots)
    if state["blocked"]:
        raise SystemExit(f"fail closed: {state['blocked']} started-only candidate slots require manual resolution")
    pending = [slot for slot in slots if state["states"].get(slot["request_slot_id"]) not in {"completed", "failed", "blocked_started_only"}]
    existing_results = read_jsonl(results_path) if results_path.exists() else []
    result_ids = {str(row["blinded_request_id"]) for row in existing_results}
    if len(result_ids) != len(existing_results):
        raise SystemExit("duplicate candidate result IDs")
    expected_by_id = {str(slot["blinded_request_id"]): slot for slot in slots}
    for result in existing_results:
        identifier = str(result.get("blinded_request_id"))
        if identifier not in expected_by_id:
            raise SystemExit(f"candidate result is outside frozen calibration: {identifier}")
        if state["states"].get(expected_by_id[identifier]["request_slot_id"]) != "completed":
            raise SystemExit(f"candidate result lacks a matching completed ledger event: {identifier}")
    if not args.execute:
        results = existing_results
    else:
        if not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit("paid execution blocked before started events: GEMINI_API_KEY is missing")
        adapter = GeminiJudgeAdapter(CANDIDATE_MODEL, max_output_tokens=1024)
        results = list(existing_results)
        for slot in pending:
            if slot["request_slot_id"] not in state["states"]:
                append_event(ledger, slot, "planned")
            append_event(ledger, slot, "started")
            try:
                raw = adapter.call(slot["prompt"], slot["task"])
            except Exception as exc:
                record = {key: slot[key] for key in ("request_slot_id", "blinded_request_id", "task", "candidate_id", "judge_provider", "judge_model", "protocol_version", "request_hash")}
                record.update({"parse_status": "provider_failure", "error_class": type(exc).__name__, "error_message": safe_error(exc)})
                results.append(record)
                write_jsonl(results_path, results)
                append_event(ledger, slot, "failed", failure_kind="provider_failure", error_class=type(exc).__name__, error_message=safe_error(exc))
                continue
            response_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            try:
                parsed = parse_judge_output(raw, slot["task"])
            except ValueError as exc:
                record = {key: slot[key] for key in ("request_slot_id", "blinded_request_id", "task", "candidate_id", "judge_provider", "judge_model", "protocol_version", "request_hash")}
                record.update({"parse_status": "parse_failure", "error_class": type(exc).__name__, "error_message": safe_error(exc), "response_hash": response_hash})
                results.append(record)
                write_jsonl(results_path, results)
                append_event(ledger, slot, "failed", failure_kind="parse_failure", error_class=type(exc).__name__, error_message=safe_error(exc), response_hash=response_hash)
                continue
            result = {key: slot[key] for key in ("request_slot_id", "blinded_request_id", "task", "candidate_id", "judge_provider", "judge_model", "protocol_version", "request_hash")}
            result.update({"parse_status": "completed", "response_hash": response_hash, "decision": parsed["decision"], "confidence": parsed["confidence"], "short_reason": parsed["short_reason"], **({"answerability_basis": parsed["answerability_basis"]} if slot["task"] == "longmemeval" else {})})
            results.append(result)
            write_jsonl(results_path, results)
            append_event(ledger, slot, "completed", response_hash=response_hash, parsed_result=parsed)
    summary = summarize_candidate(load_calibration_cases(args.calibration_cases), results)
    final_state = validate_ledger(ledger.read(), slots)
    preflight = {"execution_mode": "paid_execution" if args.execute else "preflight_only", "completed": final_state["completed"], "failed": final_state["failed"], "pending": len(slots) - final_state["completed"] - final_state["failed"]}
    summary["preflight"] = preflight
    write_json(args.output_dir / "candidate_summary.json", summary)
    write_report(args.output_dir / "CANDIDATE_CALIBRATION_REPORT.md", summary, preflight)
    print(f"CANDIDATE_ID={CANDIDATE_ID}")
    print(f"CALIBRATION_CASES={len(slots)}")
    print(f"PENDING_SLOTS={preflight['pending']}")
    print(f"COMPLETED_SLOTS={preflight['completed']}")
    print(f"FAILED_SLOTS={preflight['failed']}")
    print(f"ESTIMATED_NEW_PAID_CALLS={preflight['pending'] if not args.execute else 0}")
    print("HOLDOUT_CASES_TOUCHED=0")
    print(f"PAID_CALLS_PERFORMED={0 if not args.execute else preflight['completed']}")
    if args.execute:
        print(f"ELIGIBLE={'yes' if summary['eligible'] else 'no'}")
        print(f"PREFERRED_THRESHOLD_MET={'yes' if summary['preferred_threshold_met'] else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
