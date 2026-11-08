#!/usr/bin/env python3
"""Run or preflight one frozen task-routed LongMemEval candidate.

The default mode is offline-only. Paid execution requires both explicit flags,
uses one append-only ledger per candidate, and never reads the locked holdout.
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

from eval.providers.deepseek_provider import DEFAULT_DEEPSEEK_MODEL
from eval.providers.kimi_provider import DEFAULT_KIMI_MODEL
from scripts.b4_calibration_execution import (
    DeepSeekJudgeAdapter,
    DurableLedger,
    KimiJudgeAdapter,
    estimate_input_tokens,
    provider_request_prompt,
)
from scripts.b4_calibration_runner import build_lme_prompt, parse_judge_output
from scripts.prepare_final_judge_selection_calibration import digest, read_jsonl, write_json, write_jsonl


POLICY_VERSION = "final-judge-selection-task-routed-v2"
EXPECTED_CASES = 64
MAX_ATTEMPTS = 1
LIMITS = {"deepseek": 1_048_576, "kimi": 262_144}
LIMIT_SOURCES = {
    "deepseek": "repository-verified B4 DeepSeek V4 Pro capability configuration",
    "kimi": "repository-verified B4 Kimi K2.6 capability configuration",
}
CANDIDATES = {
    "deepseek": {"model": DEFAULT_DEEPSEEK_MODEL, "directory": "lme-deepseek-v4-pro"},
    "kimi": {"model": DEFAULT_KIMI_MODEL, "directory": "lme-kimi-k2.6"},
}


def safe_error(exc: BaseException) -> str:
    return re.sub(r"(?:AIza|sk-ant-|sk-|key=)[A-Za-z0-9_\-]+", "[REDACTED]", str(exc))[:2000]


def payload_locator(value: str) -> tuple[Path, str]:
    path, sep, identifier = value.partition("#")
    if not sep or not path or not identifier:
        raise ValueError(f"invalid payload locator: {value}")
    return Path(path), identifier


def load_payloads(plan: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    paths = sorted({payload_locator(str(row["private_payload_locator"]))[0] for row in plan})
    payloads: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            identifier = str(row["blinded_request_id"])
            if identifier in payloads:
                raise ValueError(f"duplicate private payload: {identifier}")
            payloads[identifier] = row
    return payloads


def build_slots(plan_path: Path, calibration_path: Path, candidate: str) -> list[dict[str, Any]]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unsupported candidate: {candidate}")
    model = CANDIDATES[candidate]["model"]
    source = read_jsonl(plan_path)
    rows = [row for row in source if row.get("provider") == candidate and row.get("model") == model]
    if len(rows) != EXPECTED_CASES:
        raise ValueError(f"expected exactly {EXPECTED_CASES} frozen {candidate} rows, found {len(rows)}")
    if any(row.get("task") != "longmemeval" for row in rows):
        raise ValueError("candidate plan contains non-LongMemEval rows")
    if any(row.get("protocol_version") != POLICY_VERSION for row in rows):
        raise ValueError("protocol version drift")
    calibration = read_jsonl(calibration_path)
    ids = {str(row["blinded_request_id"]) for row in calibration}
    if len(calibration) != 404 or len(ids) != 404:
        raise ValueError("calibration source must contain exactly 404 unique cases")
    payloads = load_payloads(rows)
    slots = []
    seen: set[str] = set()
    for row in rows:
        identifier = str(row["blinded_request_id"])
        if identifier not in ids or identifier not in payloads:
            raise ValueError(f"candidate case is not in frozen calibration: {identifier}")
        payload = payloads[identifier]
        if digest(payload) != row["payload_sha256"]:
            raise ValueError(f"payload hash drift: {identifier}")
        prompt = build_lme_prompt(payload)
        semantic_hash = digest(prompt)
        if semantic_hash != row["semantic_prompt_hash"]:
            raise ValueError(f"semantic prompt hash drift: {identifier}")
        request_prompt = provider_request_prompt(candidate, prompt)
        request_hash = digest(request_prompt)
        if request_hash != row["request_hash"]:
            raise ValueError(f"request hash drift: {identifier}")
        measured = estimate_input_tokens(request_prompt, model)
        limit = LIMITS[candidate]
        if measured["estimated_tokens"] > limit:
            raise ValueError(f"input-limit blocker: {identifier}")
        slot_id = str(row["request_slot_id"])
        if slot_id in seen:
            raise ValueError(f"duplicate request slot: {slot_id}")
        seen.add(slot_id)
        slots.append({
            "request_slot_id": slot_id,
            "blinded_request_id": identifier,
            "task": "longmemeval",
            "candidate_id": f"{candidate}-{model}",
            "judge_provider": candidate,
            "judge_model": model,
            "protocol_version": POLICY_VERSION,
            "request_hash": request_hash,
            "semantic_prompt_hash": semantic_hash,
            "payload_sha256": row["payload_sha256"],
            "private_payload_locator": row["private_payload_locator"],
            "prompt_chars": len(request_prompt),
            "estimated_input_tokens": measured["estimated_tokens"],
            "input_token_estimation_method": measured["method"],
            "configured_max_input_tokens": limit,
            "input_limit_source": LIMIT_SOURCES[candidate],
            "max_attempts": MAX_ATTEMPTS,
            "automatic_retries": False,
            "prompt": request_prompt,
        })
    if len(slots) != EXPECTED_CASES or len(seen) != EXPECTED_CASES:
        raise ValueError("candidate slot cardinality failure")
    return sorted(slots, key=lambda row: row["blinded_request_id"])


def validate_ledger(events: list[dict[str, Any]], slots: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {row["request_slot_id"]: row for row in slots}
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        slot_id = str(event.get("request_slot_id"))
        if slot_id not in expected:
            raise ValueError(f"ledger contains unknown slot: {slot_id}")
        slot = expected[slot_id]
        for key in ("blinded_request_id", "candidate_id", "judge_provider", "judge_model", "protocol_version", "request_hash"):
            if str(event.get(key)) != str(slot[key]):
                raise ValueError(f"ledger identity drift for {slot_id}: {key}")
        histories[slot_id].append(event)
    states: dict[str, str] = {}
    for slot_id, history in histories.items():
        if sum(event.get("state") == "started" for event in history) > MAX_ATTEMPTS:
            raise ValueError(f"slot exceeded one attempt: {slot_id}")
        terminal = [event for event in history if event.get("state") in {"completed", "failed"}]
        if len(terminal) > 1:
            raise ValueError(f"slot has multiple terminal events: {slot_id}")
        if history[-1].get("state") == "started" and not terminal:
            states[slot_id] = "started_only"
        elif terminal:
            states[slot_id] = str(terminal[-1]["state"])
    blocked = sum(value == "started_only" for value in states.values())
    if blocked:
        raise ValueError(f"fail closed: {blocked} started-only slots")
    return {
        "states": states,
        "completed": sum(value == "completed" for value in states.values()),
        "failed": sum(value == "failed" for value in states.values()),
    }


def append_event(ledger: DurableLedger, slot: dict[str, Any], state: str, **extra: Any) -> None:
    event = {key: slot[key] for key in ("request_slot_id", "blinded_request_id", "candidate_id", "judge_provider", "judge_model", "protocol_version", "request_hash")}
    event.update({"state": state, "attempt": 1, "timestamp_ns": time.time_ns(), **extra})
    ledger.append(event)


def result_record(slot: dict[str, Any], **fields: Any) -> dict[str, Any]:
    base = {key: slot[key] for key in ("request_slot_id", "blinded_request_id", "task", "candidate_id", "judge_provider", "judge_model", "protocol_version", "request_hash")}
    base.update(fields)
    return base


def binary_projection(decision: str | None) -> str | None:
    if decision in {"correct", "abstention_correct"}:
        return "correct"
    if decision in {"incorrect", "abstention_incorrect"}:
        return "incorrect"
    return None


def kappa(reference: list[str], prediction: list[str]) -> float | None:
    if not reference:
        return None
    labels = sorted(set(reference) | set(prediction))
    observed = sum(a == b for a, b in zip(reference, prediction)) / len(reference)
    expected = sum(reference.count(label) * prediction.count(label) for label in labels) / len(reference) ** 2
    return None if expected == 1 else (observed - expected) / (1 - expected)


def summarize(calibration_path: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    all_gold = read_jsonl(calibration_path)
    if len(all_gold) != 404:
        raise ValueError("invalid calibration gold source")
    gold = {str(row["blinded_request_id"]): row for row in all_gold if row.get("task") == "longmemeval"}
    if len(gold) != EXPECTED_CASES:
        raise ValueError("calibration gold does not contain exactly 64 LongMemEval cases")
    if len({str(row["blinded_request_id"]) for row in results}) != len(results):
        raise ValueError("duplicate candidate results")
    valid = []
    abstention = []
    for result in results:
        identifier = str(result["blinded_request_id"])
        if identifier not in gold:
            raise ValueError("result outside calibration")
        if result.get("parse_status") != "completed":
            continue
        predicted = binary_projection(result.get("decision"))
        if predicted is not None:
            valid.append((gold[identifier]["gold_binary_label"], predicted))
        if gold[identifier].get("gold_raw_label") in {"abstention_correct", "abstention_incorrect"}:
            abstention.append((gold[identifier]["gold_raw_label"], result.get("decision")))
    agreement = sum(a == b for a, b in valid)
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for ref, pred in valid:
        if ref == "correct" and pred == "correct": confusion["tp"] += 1
        elif ref == "incorrect" and pred == "incorrect": confusion["tn"] += 1
        elif ref == "incorrect": confusion["fp"] += 1
        else: confusion["fn"] += 1
    return {
        "n": len(gold), "valid_judged_n": len(valid), "agreement_n": agreement,
        "disagreement_n": len(valid) - agreement,
        "agreement_rate": agreement / len(valid) if valid else None,
        "cohen_kappa": kappa([a for a, _ in valid], [b for _, b in valid]),
        "confusion": confusion,
        "eligible_abstention_subtype_n": len(abstention),
        "raw_abstention_subtype_exact_n": sum(a == b for a, b in abstention),
        "raw_abstention_subtype_agreement": (sum(a == b for a, b in abstention) / len(abstention) if abstention else None),
        "adjudication_required_n": sum(row.get("decision") == "adjudication_required" for row in results),
        "low_confidence_n": sum(row.get("confidence") == "low" for row in results),
        "parse_failure_n": sum(row.get("parse_status") == "parse_failure" for row in results),
        "provider_failure_n": sum(row.get("parse_status") == "provider_failure" for row in results),
        "eligible": agreement / len(valid) >= 0.90 if valid else False,
        "preferred_threshold_met": agreement / len(valid) >= 0.95 if valid else False,
    }


def compare(root: Path, calibration_path: Path, output_path: Path) -> None:
    summaries = {}
    for candidate, config in CANDIDATES.items():
        directory = root / config["directory"]
        results = read_jsonl(directory / "candidate_results.jsonl")
        summaries[candidate] = summarize(calibration_path, results)
    selected = None
    eligible = [name for name, summary in summaries.items() if summary["eligible"]]
    if eligible:
        selected = sorted(eligible, key=lambda name: (-summaries[name]["agreement_rate"], -float(summaries[name]["raw_abstention_subtype_agreement"] or -1), -float(summaries[name]["cohen_kappa"] or -1), summaries[name]["adjudication_required_n"] + summaries[name]["low_confidence_n"] + summaries[name]["parse_failure_n"] + summaries[name]["provider_failure_n"]))[0]
    comparison = {"policy_version": POLICY_VERSION, "candidates": summaries, "selected_lme_primary": selected, "lme_primary_not_found": selected is None, "holdout_touched": 0, "paid_calls_performed": 0}
    write_json(output_path / "lme_candidate_comparison.json", comparison)
    lines = ["# LME candidate comparison", "", f"Policy: `{POLICY_VERSION}`. Holdout cases touched: 0.", ""]
    for name in ("deepseek", "kimi"):
        summary = summaries[name]
        label = name.upper()
        lines.extend([f"{label}_AGREEMENT={summary['agreement_rate']}", f"{label}_KAPPA={summary['cohen_kappa']}", f"{label}_ABSTENTION_SUBTYPE_AGREEMENT={summary['raw_abstention_subtype_agreement']}", f"{label}_ELIGIBLE={'yes' if summary['eligible'] else 'no'}", ""])
    lines.extend([f"SELECTED_LME_PRIMARY={selected or 'none'}", f"LME_PRIMARY_NOT_FOUND={'yes' if selected is None else 'no'}", ""])
    (output_path / "LME_CANDIDATE_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=["deepseek", "kimi"], required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid-judge-calls", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--plan", type=Path, default=Path("results/final_native_evaluation_design/judge_selection/task_routed_v2/lme_candidate_plan.json"))
    parser.add_argument("--calibration", type=Path, default=Path("results/final_native_evaluation_design/judge_selection/calibration_cases.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("results/b4_scoring_validation/private/final_judge_selection"))
    parser.add_argument("--comparison-output", type=Path, default=Path("results/final_native_evaluation_design/judge_selection/task_routed_v2"))
    args = parser.parse_args(argv)
    if args.execute != args.confirm_paid_judge_calls:
        raise SystemExit("paid execution requires both --execute and --confirm-paid-judge-calls")
    config = CANDIDATES[args.candidate]
    output = args.root / config["directory"]
    output.mkdir(parents=True, exist_ok=True)
    slots = build_slots(args.plan, args.calibration, args.candidate)
    public_slots = [{key: value for key, value in slot.items() if key != "prompt"} for slot in slots]
    plan_path = output / "execution_plan.jsonl"
    plan_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in public_slots)
    if plan_path.exists() and plan_path.read_text(encoding="utf-8") != plan_text:
        raise SystemExit("execution plan drift")
    plan_path.write_text(plan_text, encoding="utf-8")
    ledger_path = output / "judge_execution_ledger.jsonl"
    results_path = output / "candidate_results.jsonl"
    ledger_path.touch(exist_ok=True)
    results_path.touch(exist_ok=True)
    ledger = DurableLedger(ledger_path)
    state = validate_ledger(ledger.read(), slots)
    results = read_jsonl(results_path)
    result_ids = {str(row["blinded_request_id"]) for row in results}
    if len(result_ids) != len(results):
        raise SystemExit("duplicate candidate result IDs")
    if not args.execute:
        pending = [slot for slot in slots if state["states"].get(slot["request_slot_id"]) not in {"completed", "failed"}]
    else:
        missing = os.environ.get("DEEPSEEK_API_KEY" if args.candidate == "deepseek" else "KIMI_API_KEY")
        if not missing:
            raise SystemExit("paid execution blocked before ledger start: candidate credential missing")
        adapter = DeepSeekJudgeAdapter(config["model"]) if args.candidate == "deepseek" else KimiJudgeAdapter(config["model"])
        for slot in slots:
            if state["states"].get(slot["request_slot_id"]) in {"completed", "failed"}:
                continue
            append_event(ledger, slot, "planned")
            append_event(ledger, slot, "started")
            try:
                raw = adapter.call(slot["prompt"], "longmemeval")
            except Exception as exc:
                record = result_record(slot, parse_status="provider_failure", error_class=type(exc).__name__, error_message=safe_error(exc))
                results.append(record); write_jsonl(results_path, results)
                append_event(ledger, slot, "failed", failure_kind="provider_failure", error_class=type(exc).__name__, error_message=safe_error(exc))
                continue
            response_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            try:
                parsed = parse_judge_output(raw, "longmemeval")
            except ValueError as exc:
                record = result_record(slot, parse_status="parse_failure", response_hash=response_hash, error_class=type(exc).__name__, error_message=safe_error(exc))
                results.append(record); write_jsonl(results_path, results)
                append_event(ledger, slot, "failed", failure_kind="parse_failure", error_class=type(exc).__name__, error_message=safe_error(exc), response_hash=response_hash)
                continue
            record = result_record(slot, parse_status="completed", response_hash=response_hash, **parsed)
            results.append(record); write_jsonl(results_path, results)
            append_event(ledger, slot, "completed", response_hash=response_hash, parsed_result=parsed)
        state = validate_ledger(ledger.read(), slots)
        pending = [slot for slot in slots if state["states"].get(slot["request_slot_id"]) not in {"completed", "failed"}]
    summary = summarize(args.calibration, results)
    summary.update({"candidate_id": config["model"], "provider": args.candidate, "protocol_version": POLICY_VERSION, "execution_mode": "paid_execution" if args.execute else "preflight_only", "completed_slots": state["completed"], "failed_slots": state["failed"], "pending_slots": len(pending), "holdout_cases_touched": 0, "paid_calls_performed": 0 if not args.execute else state["completed"]})
    write_json(output / "candidate_summary.json", summary)
    (output / "CANDIDATE_CALIBRATION_REPORT.md").write_text(f"# {config['model']} LME calibration\n\nExecution mode: `{summary['execution_mode']}`. Cases: 64. Holdout cases touched: 0.\n\nAgreement: {summary['agreement_n']}/{summary['valid_judged_n']} ({summary['agreement_rate']}). Kappa: {summary['cohen_kappa']}. Eligible: `{summary['eligible']}`.\n", encoding="utf-8")
    if args.compare:
        compare(args.root, args.calibration, args.comparison_output)
    print(f"CANDIDATE_ID={config['model']}")
    print("CALIBRATION_CASES=64")
    print(f"PENDING_SLOTS={len(pending)}")
    print(f"COMPLETED_SLOTS={state['completed']}")
    print(f"FAILED_SLOTS={state['failed']}")
    print(f"ESTIMATED_NEW_PAID_CALLS={len(pending) if not args.execute else 0}")
    print("MEMORA_CASES_TOUCHED=0")
    print("HOLDOUT_CASES_TOUCHED=0")
    print(f"PAID_CALLS_PERFORMED={0 if not args.execute else state['completed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
