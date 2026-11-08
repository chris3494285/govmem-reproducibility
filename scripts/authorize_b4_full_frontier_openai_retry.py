#!/usr/bin/env python3
"""Preflight or append-authorize audited OpenAI quota retries for full B4."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_b4_full_frontier_failures import SAFE_DEFINITIVE_RETRY
from scripts.b4_calibration_execution import (
    FULL_FRONTIER_OPENAI_RETRY_VERSION,
    MANUAL_RETRY_AUTHORIZATION_STATE,
    DurableLedger,
    LedgerConflictError,
    authorized_manual_retry_slots,
    build_slots,
    resume_plan,
)
from scripts.prepare_b4_private_payloads import read_jsonl

EXPECTED_RETRY_N = 197
EXPECTED_COMPLETED_N = 1009
EXPECTED_AMBIGUOUS_ANTHROPIC_N = 2
EXPECTED_PROVIDER = "openai"
EXPECTED_MODEL = "gpt-5.6-sol"
EXPECTED_ERROR_FAMILY = "openai_http_429_credit_balance_exhausted"


def _identity(record: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(record.get("request_slot_id", "")), str(record.get("blinded_request_id", "")),
        str(record.get("judge_slot", "")), str(record.get("judge_provider", record.get("provider", ""))),
        str(record.get("judge_model", record.get("model", ""))), str(record.get("request_hash", "")),
    )


def recovery_preflight(
    slots: list[dict[str, Any]],
    events: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    plan_by_id = {str(slot["request_slot_id"]): slot for slot in slots}
    if len(plan_by_id) != len(slots):
        raise LedgerConflictError("frozen execution plan contains duplicate request-slot IDs")
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        histories[str(event.get("request_slot_id"))].append(event)

    audited = audit.get("current_failed_slots", [])
    selected = [
        record for record in audited
        if record.get("retry_classification") == SAFE_DEFINITIVE_RETRY
        and record.get("provider") == EXPECTED_PROVIDER
        and record.get("model") == EXPECTED_MODEL
        and record.get("error_family") == EXPECTED_ERROR_FAMILY
    ]
    ambiguous_anthropic = [
        record for record in audited
        if record.get("provider") == "anthropic"
        and record.get("model") == "claude-sonnet-5"
        and record.get("retry_classification") == "UNSAFE_AMBIGUOUS"
    ]
    if len(selected) != EXPECTED_RETRY_N:
        raise LedgerConflictError(f"expected {EXPECTED_RETRY_N} audited OpenAI quota failures, found {len(selected)}")
    if len({record["request_slot_id"] for record in selected}) != EXPECTED_RETRY_N:
        raise LedgerConflictError("audited retry inventory contains duplicate slots")
    if len(ambiguous_anthropic) != EXPECTED_AMBIGUOUS_ANTHROPIC_N:
        raise LedgerConflictError("Anthropic ambiguous-slot count differs from audited expectation")

    eligible: list[dict[str, Any]] = []
    already_authorized: list[dict[str, Any]] = []
    for record in selected:
        slot_id = str(record["request_slot_id"])
        slot = plan_by_id.get(slot_id)
        if slot is None or _identity(slot) != _identity(record):
            raise LedgerConflictError(f"audit/plan identity or hash drift for {slot_id}")
        history = histories.get(slot_id, [])
        if not history:
            raise LedgerConflictError(f"audited slot lacks ledger history: {slot_id}")
        if any(event.get("state") == "completed" for event in history):
            raise LedgerConflictError(f"completed slot cannot be authorized: {slot_id}")
        if any(_identity(event) != _identity(slot) for event in history):
            raise LedgerConflictError(f"ledger identity/provider/model/hash drift for {slot_id}")
        latest = history[-1]
        if latest.get("state") == MANUAL_RETRY_AUTHORIZATION_STATE:
            if latest.get("repair_version") != FULL_FRONTIER_OPENAI_RETRY_VERSION:
                raise LedgerConflictError(f"slot has an unrelated authorization: {slot_id}")
            already_authorized.append(slot)
        elif (
            latest.get("state") == "permanently_failed"
            and latest.get("error_class") == "RateLimitError"
            and record.get("latest_terminal_slot_state") == "permanently_failed"
        ):
            eligible.append(slot)
        else:
            raise LedgerConflictError(f"audited slot state drift for {slot_id}: {latest.get('state')}")

    latest_by_id = {slot_id: history[-1] for slot_id, history in histories.items() if history}
    completed_ids = {
        slot_id for slot_id, event in latest_by_id.items()
        if event.get("state") == "completed" and slot_id in plan_by_id
    }
    if len(completed_ids) != EXPECTED_COMPLETED_N:
        raise LedgerConflictError(f"expected {EXPECTED_COMPLETED_N} protected completions, found {len(completed_ids)}")
    selected_ids = {record["request_slot_id"] for record in selected}
    ambiguous_ids = {record["request_slot_id"] for record in ambiguous_anthropic}
    if selected_ids & completed_ids or selected_ids & ambiguous_ids:
        raise LedgerConflictError("retry inventory overlaps completed or ambiguous slots")
    for slot_id in ambiguous_ids:
        latest = latest_by_id.get(str(slot_id), {})
        if latest.get("state") == MANUAL_RETRY_AUTHORIZATION_STATE:
            raise LedgerConflictError("an ambiguous Anthropic slot has been authorized")

    return {
        "schema_version": "b4-full-frontier-openai-retry-preflight-v1",
        "mode": "read_only_dry_run",
        "eligible_openai_retry_slots": len(eligible),
        "authorized_openai_retry_slots": len(already_authorized),
        "audited_openai_retry_slots": len(selected),
        "protected_completed_slots": len(completed_ids),
        "blocked_anthropic_ambiguous_slots": len(ambiguous_ids),
        "estimated_new_paid_calls": len(eligible) + len(already_authorized),
        "no_other_slots_eligible": len(completed_ids) + len(selected_ids) + len(ambiguous_ids) == len(slots),
        "eligible_slots": eligible,
        "already_authorized_slots": already_authorized,
    }


def append_authorizations(ledger: DurableLedger, summary: dict[str, Any]) -> int:
    if summary["already_authorized_slots"]:
        raise LedgerConflictError("duplicate authorization blocked: one or more slots are already authorized")
    if len(summary["eligible_slots"]) != EXPECTED_RETRY_N:
        raise LedgerConflictError("authorization requires the complete exact 197-slot audited set")
    events = ledger.read()
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        histories[str(event.get("request_slot_id"))].append(event)
    prepared = []
    for slot in summary["eligible_slots"]:
        history = histories[str(slot["request_slot_id"])]
        terminal = history[-1]
        started_n = sum(event.get("state") == "started" for event in history)
        prepared.append({
            "request_slot_id": slot["request_slot_id"],
            "blinded_request_id": slot["blinded_request_id"],
            "judge_slot": slot["judge_slot"],
            "judge_provider": slot["judge_provider"],
            "judge_model": slot["judge_model"],
            "request_hash": slot["request_hash"],
            "task": slot["task"],
            "state": MANUAL_RETRY_AUTHORIZATION_STATE,
            "previous_terminal_state": terminal["state"],
            "previous_terminal_attempt": terminal.get("attempt"),
            "previous_error_class": terminal.get("error_class"),
            "previous_error_message_hash": hashlib.sha256(str(terminal.get("error_message", "")).encode()).hexdigest(),
            "authorized_attempt": started_n + 1,
            "maximum_new_paid_calls": 1,
            "recovery_reason": "audited_openai_http_429_credit_balance_exhausted_no_completion",
            "repair_version": FULL_FRONTIER_OPENAI_RETRY_VERSION,
            "timestamp_ns": time.time_ns(),
        })
    for event in prepared:
        ledger.append(event)
    return len(prepared)


def _public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key not in {"eligible_slots", "already_authorized_slots"}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, default=Path("results/b4_scoring_validation/private/full_frontier_judges"))
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads((args.phase_dir / "full_validation_manifest.json").read_text(encoding="utf-8"))
    prompts = read_jsonl(args.phase_dir / "full_validation_prompt_plan.jsonl")
    slots = build_slots(prompts, **manifest["judge_architecture"])
    ledger = DurableLedger(args.phase_dir / "judge_execution_ledger.jsonl")
    audit = json.loads((args.phase_dir / "failure_audit.json").read_text(encoding="utf-8"))
    summary = recovery_preflight(slots, ledger.read(), audit)
    if args.authorize:
        appended = append_authorizations(ledger, summary)
        summary = recovery_preflight(slots, ledger.read(), audit)
        summary["mode"] = "append_only_authorize"
        summary["authorization_events_appended"] = appended
    else:
        summary["authorization_events_appended"] = 0
    authorized = authorized_manual_retry_slots(slots, ledger)
    resume = resume_plan(slots, ledger)
    summary["runner_authorized_pending_slots"] = len(authorized)
    summary["runner_completed_slots"] = len(resume["completed"])
    summary["runner_blocked_slots"] = len(resume["blocked"])
    output = args.phase_dir / "openai_retry_preflight.json"
    output.write_text(json.dumps(_public_summary(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_public_summary(summary), sort_keys=True))
    print(f"AUTHORIZED_OPENAI_RETRY_SLOTS={summary['authorized_openai_retry_slots'] or summary['eligible_openai_retry_slots']}")
    print(f"PROTECTED_COMPLETED_SLOTS={summary['protected_completed_slots']}")
    print(f"BLOCKED_ANTHROPIC_AMBIGUOUS_SLOTS={summary['blocked_anthropic_ambiguous_slots']}")
    print(f"ESTIMATED_NEW_PAID_CALLS={summary['estimated_new_paid_calls']}")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
