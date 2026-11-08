#!/usr/bin/env python3
"""Authorize exactly two user-approved Anthropic no-text B4 retries."""

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

from scripts.b4_calibration_execution import (
    MANUAL_RETRY_AUTHORIZATION_STATE,
    TWO_ANTHROPIC_NO_TEXT_RETRY_VERSION,
    DurableLedger,
    LedgerConflictError,
    authorized_manual_retry_slots,
    build_slots,
)
from scripts.prepare_b4_private_payloads import read_jsonl

TARGET_BLINDED_IDS = frozenset({
    "longmemeval::gpt4_a1b77f9c::78f401338953cdae",
    "longmemeval::gpt4_cd90e484::5784a907e475fa49",
})
EXPECTED_PROVIDER = "anthropic"
EXPECTED_MODEL = "claude-sonnet-5"
EXPECTED_TASK = "longmemeval"
EXPECTED_ERROR_CLASS = "ValueError"
EXPECTED_ERROR_MESSAGE = "provider response contained no text"
EXPECTED_ERROR_FAMILY = "provider_response_no_text"
EXPECTED_COMPLETED_N = 1206


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
    target_blinded_ids: frozenset[str] = TARGET_BLINDED_IDS,
) -> dict[str, Any]:
    if len(target_blinded_ids) != 2:
        raise LedgerConflictError("Anthropic no-text recovery requires exactly two explicit blinded IDs")
    targets = [
        slot for slot in slots
        if slot.get("blinded_request_id") in target_blinded_ids
        and slot.get("judge_provider") == EXPECTED_PROVIDER
        and slot.get("judge_model") == EXPECTED_MODEL
        and slot.get("task") == EXPECTED_TASK
    ]
    if len(targets) != 2 or {slot["blinded_request_id"] for slot in targets} != set(target_blinded_ids):
        raise LedgerConflictError("exact two-slot Anthropic target set is absent from the frozen plan")
    if len({slot["request_slot_id"] for slot in targets}) != 2:
        raise LedgerConflictError("target request-slot IDs are not unique")

    audit_by_id = {str(record["request_slot_id"]): record for record in audit.get("current_failed_slots", [])}
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        histories[str(event.get("request_slot_id"))].append(event)
    eligible = []
    authorized = []
    for slot in targets:
        slot_id = str(slot["request_slot_id"])
        record = audit_by_id.get(slot_id)
        if record is None or _identity(record) != _identity(slot):
            raise LedgerConflictError(f"audit/plan identity or request-hash drift for {slot_id}")
        if (
            record.get("retry_classification") != "UNSAFE_AMBIGUOUS"
            or record.get("error_family") != EXPECTED_ERROR_FAMILY
            or record.get("provider") != EXPECTED_PROVIDER
            or record.get("model") != EXPECTED_MODEL
            or record.get("task") != EXPECTED_TASK
        ):
            raise LedgerConflictError(f"target is not the audited Anthropic no-text failure: {slot_id}")
        history = histories.get(slot_id, [])
        if not history:
            raise LedgerConflictError(f"target lacks ledger history: {slot_id}")
        if any(event.get("state") == "completed" for event in history):
            raise LedgerConflictError(f"completed target cannot be retried: {slot_id}")
        if any(_identity(event) != _identity(slot) for event in history):
            raise LedgerConflictError(f"ledger provider/model/hash drift for {slot_id}")
        latest = history[-1]
        if latest.get("state") == MANUAL_RETRY_AUTHORIZATION_STATE:
            if latest.get("repair_version") != TWO_ANTHROPIC_NO_TEXT_RETRY_VERSION:
                raise LedgerConflictError(f"target has an unrelated authorization: {slot_id}")
            authorized.append(slot)
        elif (
            latest.get("state") == "provider_error"
            and latest.get("error_class") == EXPECTED_ERROR_CLASS
            and latest.get("error_message") == EXPECTED_ERROR_MESSAGE
        ):
            eligible.append(slot)
        else:
            raise LedgerConflictError(f"target terminal state or error drift for {slot_id}")

    latest_by_id = {slot_id: history[-1] for slot_id, history in histories.items() if history}
    completed_n = sum(event.get("state") == "completed" for event in latest_by_id.values())
    if completed_n != EXPECTED_COMPLETED_N:
        raise LedgerConflictError(f"expected {EXPECTED_COMPLETED_N} protected completed slots, found {completed_n}")
    target_slot_ids = {slot["request_slot_id"] for slot in targets}
    other_incomplete = [
        slot_id for slot_id, event in latest_by_id.items()
        if slot_id not in target_slot_ids and event.get("state") != "completed"
    ]
    if other_incomplete:
        raise LedgerConflictError(f"unexpected non-target incomplete or authorized slots: {len(other_incomplete)}")
    return {
        "schema_version": "b4-two-anthropic-no-text-retry-preflight-v1",
        "mode": "read_only_dry_run",
        "eligible_slots": eligible,
        "already_authorized_slots": authorized,
        "authorized_anthropic_retry_slots": len(eligible) + len(authorized),
        "protected_completed_slots": completed_n,
        "other_eligible_slots": 0,
        "estimated_new_paid_calls": len(eligible) + len(authorized),
    }


def append_authorizations(ledger: DurableLedger, summary: dict[str, Any]) -> int:
    if summary["already_authorized_slots"]:
        raise LedgerConflictError("duplicate authorization blocked")
    if len(summary["eligible_slots"]) != 2:
        raise LedgerConflictError("authorization requires exactly two eligible target slots")
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in ledger.read():
        histories[str(event.get("request_slot_id"))].append(event)
    prepared = []
    for slot in summary["eligible_slots"]:
        history = histories[str(slot["request_slot_id"])]
        terminal = history[-1]
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
            "authorized_attempt": sum(event.get("state") == "started" for event in history) + 1,
            "maximum_new_paid_calls": 1,
            "recovery_reason": "explicit_user_authorization_for_exact_anthropic_no_text_slot",
            "repair_version": TWO_ANTHROPIC_NO_TEXT_RETRY_VERSION,
            "timestamp_ns": time.time_ns(),
        })
    for event in prepared:
        ledger.append(event)
    return len(prepared)


def _report(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key not in {"eligible_slots", "already_authorized_slots"}}


def print_summary(summary: dict[str, Any]) -> None:
    print(f"AUTHORIZED_ANTHROPIC_RETRY_SLOTS={summary['authorized_anthropic_retry_slots']}")
    print(f"PROTECTED_COMPLETED_SLOTS={summary['protected_completed_slots']}")
    print(f"OTHER_ELIGIBLE_SLOTS={summary['other_eligible_slots']}")
    print(f"ESTIMATED_NEW_PAID_CALLS={summary['estimated_new_paid_calls']}")
    print("PAID_CALLS_PERFORMED=0")


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
        summary["authorization_events_appended"] = append_authorizations(ledger, summary)
        summary = recovery_preflight(slots, ledger.read(), audit) | {
            "mode": "append_only_authorize", "authorization_events_appended": 2,
        }
    else:
        summary["authorization_events_appended"] = 0
    selected = authorized_manual_retry_slots(slots, ledger)
    target_ids = {slot["request_slot_id"] for slot in summary["eligible_slots"] + summary["already_authorized_slots"]}
    selected_target_ids = {slot["request_slot_id"] for slot in selected if slot["request_slot_id"] in target_ids}
    if summary["already_authorized_slots"] and len(selected_target_ids) != 2:
        raise LedgerConflictError("runner does not expose exactly the two authorized targets")
    (args.phase_dir / "anthropic_no_text_retry_preflight.json").write_text(
        json.dumps(_report(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
