#!/usr/bin/env python3
"""Dry-run or execute explicitly authorized cheap-judge pilot recoveries."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b4_calibration_execution import DeepSeekJudgeAdapter, DurableLedger, KimiJudgeAdapter
from scripts.prepare_cheap_judge_pilot import JUDGES, PROTOCOL, read_jsonl
from scripts.prepare_cheap_judge_recovery import (
    EXPECTED_RECOVERY,
    IDENTITY_FIELDS,
    RECOVERY_PROTOCOL,
    manifest_hash,
)
from scripts.run_cheap_judge_pilot import append_jsonl, parse_result


def load_authorization(path: Path, recovery_rows: list[dict[str, Any]]) -> dict[str, Any]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "authorized",
        "authorization_timestamp",
        "authorization_version",
        "maximum_authorized_retries_per_request",
        "protocol_version",
        "recovery_protocol_version",
        "recovery_manifest_hash",
        "authorized_request_ids",
    }
    if set(authorization) != required:
        raise RuntimeError("authorization artifact schema mismatch")
    if authorization["authorized"] is not True or not authorization["authorization_timestamp"]:
        raise RuntimeError("recovery is not explicitly authorized")
    if authorization["protocol_version"] != PROTOCOL or authorization["recovery_protocol_version"] != RECOVERY_PROTOCOL:
        raise RuntimeError("authorization protocol drift")
    if authorization["maximum_authorized_retries_per_request"] != 1:
        raise RuntimeError("authorization must allow exactly one retry per request")
    if authorization["recovery_manifest_hash"] != manifest_hash(recovery_rows):
        raise RuntimeError("authorization recovery-manifest hash drift")
    ids = authorization["authorized_request_ids"]
    if not isinstance(ids, list) or len(ids) != len(set(map(str, ids))):
        raise RuntimeError("duplicate or invalid authorized request IDs")
    manifest_ids = {row["request_id"] for row in recovery_rows}
    if set(map(str, ids)) != manifest_ids:
        raise RuntimeError("authorization must bind the exact recovery manifest population")
    return authorization


def validate_recovery_state(
    recovery_row: dict[str, Any],
    plan_row: dict[str, Any],
    events: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> str:
    rid = recovery_row["request_id"]
    if any(str(recovery_row.get(public)) != str(plan_row.get(plan)) for public, plan in (
        ("request_id", "evaluation_request_id"),
        ("provider", "judge_provider"),
        ("model", "judge_model"),
        ("pilot_block_id", "pilot_block_id"),
        ("condition_id", "condition_id"),
        ("source", "source"),
        ("answer_hash", "answer_sha256"),
        ("prompt_hash", "request_hash"),
        ("payload_hash", "payload_hash"),
        ("protocol_version", "protocol_version"),
    )):
        raise RuntimeError(f"recovery/request identity drift: {rid}")
    history = [event for event in events if str(event.get("evaluation_request_id")) == rid]
    for event in history:
        if any(str(event.get(field)) != str(plan_row.get(field)) for field in IDENTITY_FIELDS):
            raise RuntimeError(f"recovery ledger identity drift: {rid}")
    result_rows = [row for row in results if str(row.get("evaluation_request_id")) == rid]
    if len(result_rows) > 1:
        raise RuntimeError(f"duplicate successful result identity: {rid}")
    numbered_attempts = [event.get("attempt") for event in history if isinstance(event.get("attempt"), int)]
    if any(attempt > 2 for attempt in numbered_attempts):
        raise RuntimeError(f"attempt 3+ is forbidden: {rid}")
    attempt_one = [event for event in history if event.get("attempt") == 1]
    attempt_one_states = [event.get("state") for event in attempt_one]
    if recovery_row["prior_status"] == "failed":
        if attempt_one_states.count("started") != 1 or attempt_one_states.count("failed") != 1:
            raise RuntimeError(f"original failed lifecycle drift: {rid}")
    elif recovery_row["prior_status"] == "started_only":
        if attempt_one_states.count("started") != 1 or any(state in {"failed", "completed"} for state in attempt_one_states):
            raise RuntimeError(f"original started-only lifecycle drift: {rid}")
    else:
        raise RuntimeError(f"invalid frozen prior status: {rid}")
    attempt_two = [event for event in history if event.get("attempt") == 2]
    authorizations = [event for event in history if event.get("state") == "recovery_authorized"]
    if len(authorizations) > 1:
        raise RuntimeError(f"duplicate recovery authorization: {rid}")
    if attempt_two:
        states = [event.get("state") for event in attempt_two]
        if "completed" in states:
            if states.count("started") != 1 or states.count("completed") != 1 or len(result_rows) != 1:
                raise RuntimeError(f"invalid recovered completion lifecycle: {rid}")
            return "recovered_completed"
        if "started" in states:
            if "failed" in states:
                return "recovery_failed"
            return "recovery_started_only"
        raise RuntimeError(f"unsupported attempt-2 history: {rid}")
    if result_rows or any(event.get("state") == "completed" for event in history):
        raise RuntimeError(f"completed request cannot be recovered: {rid}")
    if authorizations:
        event = authorizations[0]
        if event.get("recovery_attempt") != 2 or event.get("recovery_protocol_version") != RECOVERY_PROTOCOL:
            raise RuntimeError(f"invalid recovery authorization event: {rid}")
        if history.index(event) < max(i for i, item in enumerate(history) if item.get("attempt") == 1):
            raise RuntimeError(f"recovery authorization precedes original lifecycle: {rid}")
        return "authorized_pending"
    return "eligible"


def _event_base(plan_row: dict[str, Any]) -> dict[str, Any]:
    return {field: plan_row[field] for field in (
        "evaluation_request_id",
        "pilot_block_id",
        "observation_id",
        "source",
        "judge_provider",
        "judge_model",
        "request_hash",
        "payload_hash",
        "protocol_version",
    )}


def execute_authorized(
    provider: str,
    rows: list[dict[str, Any]],
    plan_by_id: dict[str, dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    private_root: Path,
    authorization: dict[str, Any],
    adapter_factory: Callable[[str, int], Any] | None = None,
) -> int:
    env_name = "DEEPSEEK_API_KEY" if provider == "deepseek" else "KIMI_API_KEY"
    if not os.environ.get(env_name):
        raise RuntimeError(f"{env_name} is required before any recovery ledger event is written")
    provider_root = private_root / provider
    ledger = DurableLedger(provider_root / "judge_execution_ledger.jsonl")
    result_path = provider_root / "judge_results.jsonl"
    events = ledger.read()
    results = read_jsonl(result_path)
    states = {
        row["request_id"]: validate_recovery_state(row, plan_by_id[row["request_id"]], events, results)
        for row in rows
    }
    consumed = {rid: state for rid, state in states.items() if state in {"recovery_failed", "recovery_started_only"}}
    if consumed:
        raise RuntimeError(f"attempt-2 already consumed; explicit new recovery would be required: {sorted(consumed)}")
    executable = [row for row in rows if states[row["request_id"]] in {"eligible", "authorized_pending"}]
    if not executable:
        return 0
    lock = provider_root / ".recovery.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"provider recovery partition is already locked: {provider}") from exc
    os.close(fd)
    if adapter_factory is None:
        adapter_factory = lambda name, output: (
            DeepSeekJudgeAdapter(name, max_output_tokens=output)
            if provider == "deepseek"
            else KimiJudgeAdapter(name, max_output_tokens=output)
        )
    max_output = max(int(plan_by_id[row["request_id"]]["planned_output_tokens"]) for row in executable)
    adapter = adapter_factory(JUDGES[provider]["model"], max_output)
    paid = 0
    try:
        for row in executable:
            plan = plan_by_id[row["request_id"]]
            payload = payloads[plan["observation_id"]]
            base = _event_base(plan)
            if states[row["request_id"]] == "eligible":
                ledger.append({
                    **base,
                    "state": "recovery_authorized",
                    "attempt": 1,
                    "recovery_attempt": 2,
                    "recovery_reason": row["recovery_reason"],
                    "recovery_protocol_version": RECOVERY_PROTOCOL,
                    "authorization_version": authorization["authorization_version"],
                    "recovery_manifest_hash": authorization["recovery_manifest_hash"],
                    "timestamp_ns": time.time_ns(),
                })
            ledger.append({**base, "state": "started", "attempt": 2, "timestamp_ns": time.time_ns()})
            paid += 1
            try:
                raw = adapter.call(payload["prompt"], payload["source"])
                parsed = parse_result(raw, payload["source"], payload["prompt"])
                import hashlib
                response_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                append_jsonl(result_path, {
                    **base,
                    "attempt": 2,
                    "parsed_result": parsed,
                    "response_hash": response_hash,
                    "provider_usage": getattr(adapter, "last_usage", None),
                    "recovery_protocol_version": RECOVERY_PROTOCOL,
                })
                ledger.append({
                    **base,
                    "state": "completed",
                    "attempt": 2,
                    "response_hash": response_hash,
                    "recovery_protocol_version": RECOVERY_PROTOCOL,
                    "timestamp_ns": time.time_ns(),
                })
            except Exception as exc:
                ledger.append({
                    **base,
                    "state": "failed",
                    "attempt": 2,
                    "error_class": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                    "recovery_protocol_version": RECOVERY_PROTOCOL,
                    "timestamp_ns": time.time_ns(),
                })
    finally:
        lock.unlink(missing_ok=True)
    return paid


def preflight(
    provider: str,
    public_root: Path,
    private_root: Path,
    authorization_path: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    plan_rows = read_jsonl(public_root / "request_manifest.jsonl")
    frozen_rows = read_jsonl(public_root / "recovery" / "recovery_manifest.jsonl")
    summary = json.loads((public_root / "recovery" / "recovery_summary.json").read_text(encoding="utf-8"))
    if manifest_hash(frozen_rows) != summary["recovery_manifest_hash"]:
        raise RuntimeError("frozen recovery manifest hash drift")
    selected = [row for row in frozen_rows if row["provider"] == provider]
    if len(selected) != EXPECTED_RECOVERY[provider]:
        raise RuntimeError("provider recovery count drift")
    plan_by_id = {row["evaluation_request_id"]: row for row in plan_rows}
    payloads = {row["observation_id"]: row for row in read_jsonl(private_root / "semantic_payloads.jsonl")}
    events = read_jsonl(private_root / provider / "judge_execution_ledger.jsonl")
    results = read_jsonl(private_root / provider / "judge_results.jsonl")
    state_counts: dict[str, int] = {}
    for row in selected:
        rid = row["request_id"]
        if rid not in plan_by_id:
            raise RuntimeError(f"recovery request absent from frozen plan: {rid}")
        state = validate_recovery_state(row, plan_by_id[rid], events, results)
        state_counts[state] = state_counts.get(state, 0) + 1
    actionable = state_counts.get("eligible", 0) + state_counts.get("authorized_pending", 0)
    unauthorized = actionable
    authorization: dict[str, Any] = {}
    if authorization_path is not None:
        authorization = load_authorization(authorization_path, frozen_rows)
        unauthorized = 0
    state_counts["unauthorized"] = unauthorized
    state_counts["actionable"] = actionable
    state_counts["completed_protected"] = int(summary["completed_slots_protected"])
    return selected, authorization, plan_by_id, payloads, state_counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("deepseek", "kimi"), required=True)
    parser.add_argument("--public-root", type=Path, default=Path("results/cheap_judge_pilot_v1"))
    parser.add_argument(
        "--private-root",
        type=Path,
        default=Path("results/b4_scoring_validation/private/cheap_judge_pilot_v1"),
    )
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid-recovery-calls", action="store_true")
    args = parser.parse_args(argv)
    selected, authorization, plan_by_id, payloads, states = preflight(
        args.provider, args.public_root, args.private_root, args.authorization
    )
    print(f"PROTOCOL={PROTOCOL}")
    print(f"PROVIDER={args.provider}")
    print(f"RECOVERY_CANDIDATES={len(selected)}")
    print(f"COMPLETED_SLOTS_PROTECTED={states['completed_protected'] + states.get('recovered_completed', 0)}")
    print(f"UNAUTHORIZED_RETRIES={states['unauthorized']}")
    print(f"ESTIMATED_NEW_PAID_CALLS={states['actionable'] if authorization else 0}")
    if args.execute != args.confirm_paid_recovery_calls:
        raise SystemExit("paid recovery requires both --execute and --confirm-paid-recovery-calls")
    if not args.execute:
        print("EXECUTION_MODE=dry_run_only")
        print("PAID_CALLS_PERFORMED=0")
        return 0
    if not authorization:
        raise SystemExit("paid recovery requires --authorization")
    paid = execute_authorized(
        args.provider, selected, plan_by_id, payloads, args.private_root, authorization
    )
    print(f"PAID_CALLS_PERFORMED={paid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
