#!/usr/bin/env python3
"""Complete exactly three frozen final-v2 provider-execution recoveries.

Default mode is offline preflight. Paid execution requires both explicit flags.
No completed or non-target holdout slot is ever scheduled.
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

from scripts.b4_calibration_execution import AnthropicJudgeAdapter, DurableLedger, OpenAIJudgeAdapter
from scripts.b4_calibration_runner import build_lme_prompt, parse_judge_output
from scripts.prepare_final_judge_selection_calibration import digest, read_jsonl


PROTOCOL_VERSION = "final-native-evaluation-v2"
OPENAI_RECOVERY_PROTOCOL = "final-native-evaluation-v2-execution-recovery-1"
ANTHROPIC_RECOVERY_PROTOCOL = "final-native-evaluation-v2-execution-recovery-2"
ROOT = Path("results/b4_scoring_validation/private/final_b4_v2")
PLAN = ROOT / "execution_plan.jsonl"
LEDGER = ROOT / "judge_execution_ledger.jsonl"
RESULTS = ROOT / "holdout_results.jsonl"

OPENAI_TARGETS = {
    "b4-final-v2-slot::00f9b08a5e63c14f61ff9f109f09e968adcfbb31e4bb5a4e193679f66b100ddf",
    "b4-final-v2-slot::04fb1c3675278d9e0b94ef2de3d97ba89b4fea5c8b12c5e94ba6ceae042489d2",
}
ANTHROPIC_TARGET = "b4-final-v2-slot::1fc53c079bc14f82b3bea3fe4cdf1b46fa9b5c51444a0d28e474818b952e0152"
TARGETS = OPENAI_TARGETS | {ANTHROPIC_TARGET}


def safe_error(exc: BaseException) -> str:
    return re.sub(r"(?:AIza|sk-ant-|sk-|key=)[A-Za-z0-9_\-]+", "[REDACTED]", str(exc))[:2000]


def identity(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("request_slot_id", "blinded_request_id", "task", "provider", "model", "protocol_version", "request_hash")}


def group_events(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event.get("request_slot_id"))].append(event)
    return grouped


def classify_targets(plan: list[dict[str, Any]], events: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    plan_by_id = {str(row["request_slot_id"]): row for row in plan}
    if not TARGETS.issubset(plan_by_id) or len(TARGETS) != 3:
        raise ValueError("frozen recovery target set is absent or malformed")
    grouped = group_events(events)
    results_by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        results_by_slot[str(result.get("request_slot_id"))].append(result)
    statuses = {}
    for slot_id in sorted(TARGETS):
        row = plan_by_id[slot_id]
        expected_provider = "openai" if slot_id in OPENAI_TARGETS else "anthropic"
        expected_model = "gpt-5.6-sol" if expected_provider == "openai" else "claude-sonnet-5"
        if row.get("provider") != expected_provider or row.get("model") != expected_model or row.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"target provider/model/protocol drift: {slot_id}")
        history = grouped.get(slot_id, [])
        for event in history:
            for key in ("request_slot_id", "blinded_request_id", "provider", "model", "protocol_version", "request_hash"):
                if str(event.get(key)) != str(row.get(key)):
                    raise ValueError(f"target identity drift: {slot_id}/{key}")
        valid_results = [result for result in results_by_slot.get(slot_id, []) if result.get("parse_status") == "completed"]
        if len(valid_results) > 1:
            raise ValueError(f"duplicate valid target results: {slot_id}")
        attempt2 = [event for event in history if event.get("attempt") == 2 and event.get("state") in {"started", "completed", "failed"}]
        if any(event.get("attempt") not in {0, 1, 2} for event in history):
            raise ValueError(f"attempt 3 is forbidden: {slot_id}")
        if valid_results or any(event.get("state") == "completed" for event in attempt2):
            statuses[slot_id] = "completed"
            continue
        if any(event.get("state") == "failed" for event in attempt2):
            statuses[slot_id] = "attempt2_failed"
            continue
        if attempt2:
            raise ValueError(f"ambiguous attempt-2 started-only state: {slot_id}")
        attempt1_failures = [event for event in history if event.get("attempt") == 1 and event.get("state") == "failed"]
        if len(attempt1_failures) != 1:
            raise ValueError(f"target lacks exactly one attempt-1 failure: {slot_id}")
        authorizations = [event for event in history if event.get("state") == "recovery_authorized"]
        if expected_provider == "openai":
            failure = attempt1_failures[0]
            if failure.get("error_class") != "AuthenticationError" or "expired_secret_key" not in str(failure.get("error_message")):
                raise ValueError(f"OpenAI target is not the frozen expired-key failure: {slot_id}")
            if len(authorizations) != 1 or authorizations[0].get("recovery_attempt") != 2 or authorizations[0].get("recovery_protocol_version") != OPENAI_RECOVERY_PROTOCOL:
                raise ValueError(f"OpenAI attempt-2 authorization drift: {slot_id}")
            statuses[slot_id] = "authorized_attempt2_ready"
        else:
            failure = attempt1_failures[0]
            if failure.get("error_class") != "ValueError" or str(failure.get("error_message")) != "provider response contained no text":
                raise ValueError("Anthropic target is not the frozen no-text failure")
            if not authorizations:
                statuses[slot_id] = "anthropic_authorization_required"
            elif len(authorizations) == 1 and authorizations[0].get("recovery_attempt") == 2 and authorizations[0].get("recovery_protocol_version") == ANTHROPIC_RECOVERY_PROTOCOL:
                statuses[slot_id] = "authorized_attempt2_ready"
            else:
                raise ValueError("Anthropic recovery authorization drift or duplication")
    return {
        "statuses": statuses,
        "openai_already_authorized": sum(statuses[slot] == "authorized_attempt2_ready" for slot in OPENAI_TARGETS),
        "anthropic_newly_recoverable": int(statuses[ANTHROPIC_TARGET] == "anthropic_authorization_required"),
        "completed_targets": sum(status == "completed" for status in statuses.values()),
        "failed_targets": sum(status == "attempt2_failed" for status in statuses.values()),
        "ready_targets": [slot for slot, status in statuses.items() if status in {"authorized_attempt2_ready", "anthropic_authorization_required"}],
    }


def reconstruct_target_prompts(plan: list[dict[str, Any]]) -> dict[str, str]:
    plan_by_id = {str(row["request_slot_id"]): row for row in plan}
    prompts = {}
    payload_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    for slot_id in TARGETS:
        row = plan_by_id[slot_id]
        path_text, sep, blinded_id = str(row["private_payload_locator"]).partition("#")
        if not sep or blinded_id != row["blinded_request_id"]:
            raise ValueError(f"payload locator drift: {slot_id}")
        path = Path(path_text)
        if path not in payload_cache:
            payload_cache[path] = {str(item["blinded_request_id"]): item for item in read_jsonl(path)}
        payload = payload_cache[path][blinded_id]
        prompt = build_lme_prompt(payload)
        if digest(payload) != row["payload_sha256"] or digest(prompt) != row["prompt_sha256"] or digest(prompt) != row["request_hash"]:
            raise ValueError(f"target prompt/payload hash drift: {slot_id}")
        prompts[slot_id] = prompt
    return prompts


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def append_anthropic_authorization(ledger: DurableLedger, row: dict[str, Any]) -> None:
    ledger.append({
        **identity(row), "state": "recovery_authorized", "attempt": 1,
        "recovery_attempt": 2,
        "recovery_protocol_version": ANTHROPIC_RECOVERY_PROTOCOL,
        "recovery_reason": "anthropic_no_text_provider_failure_no_semantic_result",
        "original_error_class": "ValueError", "original_terminal_state": "failed",
        "timestamp_ns": time.time_ns(),
    })


def make_adapter(provider: str, model: str) -> Any:
    if provider == "openai":
        return OpenAIJudgeAdapter(model, max_output_tokens=1024)
    if provider == "anthropic":
        return AnthropicJudgeAdapter(model, max_output_tokens=1024)
    raise ValueError(f"unsupported recovery provider: {provider}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid-judge-calls", action="store_true")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    if args.execute != args.confirm_paid_judge_calls:
        raise SystemExit("paid execution requires both --execute and --confirm-paid-judge-calls")
    plan = read_jsonl(args.plan)
    prompts = reconstruct_target_prompts(plan)
    ledger = DurableLedger(args.ledger)
    events = ledger.read()
    results = read_jsonl(args.results)
    classified = classify_targets(plan, events, results)
    if not args.execute:
        print("TARGET_SLOTS=3")
        print(f"OPENAI_ALREADY_AUTHORIZED_ATTEMPT2={classified['openai_already_authorized']}")
        print(f"ANTHROPIC_NEWLY_RECOVERABLE={classified['anthropic_newly_recoverable']}")
        print(f"COMPLETED_TARGETS={classified['completed_targets']}")
        print("OTHER_SLOTS_TOUCHED=0")
        print(f"ESTIMATED_NEW_PAID_CALLS={len(classified['ready_targets'])}")
        print("PAID_CALLS_PERFORMED=0")
        return 0
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("paid execution blocked before ledger writes: OPENAI_API_KEY and ANTHROPIC_API_KEY are required")
    plan_by_id = {str(row["request_slot_id"]): row for row in plan}
    if classified["statuses"][ANTHROPIC_TARGET] == "anthropic_authorization_required":
        append_anthropic_authorization(ledger, plan_by_id[ANTHROPIC_TARGET])
    executed = {"openai": 0, "anthropic": 0}
    completed = 0
    failed = 0
    for slot_id in sorted(classified["ready_targets"]):
        row = plan_by_id[slot_id]
        ledger.append({**identity(row), "state": "started", "attempt": 2, "timestamp_ns": time.time_ns()})
        executed[row["provider"]] += 1
        try:
            raw = make_adapter(row["provider"], row["model"]).call(prompts[slot_id], "longmemeval")
            response_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            parsed = parse_judge_output(raw, "longmemeval")
        except Exception as exc:
            failure = {**identity(row), "parse_status": "provider_or_parse_failure", "error_class": type(exc).__name__, "error_message": safe_error(exc)}
            append_jsonl(args.results, failure)
            ledger.append({**identity(row), "state": "failed", "attempt": 2, "error_class": type(exc).__name__, "error_message": safe_error(exc), "timestamp_ns": time.time_ns()})
            failed += 1
            continue
        result = {**identity(row), "judge_slot": row["provider"], "parse_status": "completed", "response_hash": response_hash, **parsed}
        append_jsonl(args.results, result)
        ledger.append({**identity(row), "state": "completed", "attempt": 2, "response_hash": response_hash, "parsed_result": parsed, "timestamp_ns": time.time_ns()})
        completed += 1
    print("TARGET_SLOTS=3")
    print(f"OPENAI_ATTEMPT2_EXECUTED={executed['openai']}")
    print(f"ANTHROPIC_ATTEMPT2_EXECUTED={executed['anthropic']}")
    print(f"COMPLETED_TARGETS={completed}")
    print(f"FAILED_TARGETS={failed}")
    print(f"PAID_CALLS_PERFORMED={sum(executed.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
