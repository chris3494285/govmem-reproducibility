#!/usr/bin/env python3
"""Preflight or append one exact, user-authorized B4 frontier retry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b4_calibration_execution import (
    DurableLedger,
    append_manual_retry_authorization,
    build_slots,
    manual_retry_preflight,
    resume_plan,
)
from scripts.prepare_b4_private_payloads import read_jsonl


def _expected(args: argparse.Namespace) -> dict[str, str]:
    return {
        "request_slot_id": args.request_slot_id,
        "blinded_request_id": args.blinded_request_id,
        "judge_slot": args.judge_slot,
        "judge_provider": args.provider,
        "judge_model": args.model,
        "request_hash": args.request_hash,
        "expected_error_class": args.expected_error_class,
        "expected_error_message": args.expected_error_message,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--request-slot-id", required=True)
    parser.add_argument("--blinded-request-id", required=True)
    parser.add_argument("--judge-slot", choices=("A", "B"), required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request-hash", required=True)
    parser.add_argument("--expected-error-class", required=True)
    parser.add_argument("--expected-error-message", required=True)
    parser.add_argument("--expected-total-slots", type=int, required=True)
    parser.add_argument("--expected-completed-slots", type=int, required=True)
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args(argv)

    manifest = json.loads((args.phase_dir / "calibration_manifest.json").read_text(encoding="utf-8"))
    prompts = read_jsonl(args.phase_dir / "calibration_prompt_plan.jsonl")
    slots = build_slots(prompts, **manifest["judge_architecture"])
    ledger = DurableLedger(args.phase_dir / "judge_execution_ledger.jsonl")
    expected = _expected(args)
    summary = manual_retry_preflight(slots, ledger.read(), expected)
    if summary["total_request_slots"] != args.expected_total_slots:
        raise SystemExit("frontier slot count differs from explicit authorization")
    if summary["completed_slots"] != args.expected_completed_slots:
        raise SystemExit("frontier completed count differs from explicit authorization")

    if args.authorize:
        event = append_manual_retry_authorization(ledger, slots, expected)
        summary = manual_retry_preflight(slots, ledger.read(), expected)
        summary["mode"] = "authorize"
        summary["authorization_event_state"] = event["state"]
    else:
        summary["mode"] = "dry_run"
        summary["would_append_authorization_events"] = summary["authorization_candidate_slots"]

    plan = resume_plan(slots, ledger)
    summary["runner_completed_slots"] = len(plan["completed"])
    summary["runner_pending_slots"] = len(plan["pending"])
    summary["runner_blocked_slots"] = len(plan["blocked"])
    if summary["authorized_recovery_slots"]:
        if len(plan["pending"]) != 1 or plan["pending"][0]["request_slot_id"] != args.request_slot_id:
            raise SystemExit("authorized resume plan is not restricted to the exact target")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
