#!/usr/bin/env python3
"""Read-only forensic audit of B4 full-frontier execution failures."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_b4_private_payloads import read_jsonl

ERROR_STATES = {"provider_error", "retryable_error", "permanently_failed"}
TERMINAL_STATES = ERROR_STATES | {"completed", "parse_failed"}
SAFE_DEFINITIVE_RETRY = "SAFE_DEFINITIVE_RETRY"
UNSAFE_AMBIGUOUS = "UNSAFE_AMBIGUOUS"
NON_RETRYABLE_PARSE_OR_SCHEMA = "NON_RETRYABLE_PARSE_OR_SCHEMA"
ALREADY_COMPLETED = "ALREADY_COMPLETED"
UNKNOWN_REQUIRES_REVIEW = "UNKNOWN_REQUIRES_REVIEW"


def sanitize_error_message(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"(?:sk-ant-|sk-)[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(r"(?i)(api[_ -]?key\s*[=:]\s*)\S+", r"\1[REDACTED]", text)
    return text[:2000]


def error_family(error_class: str, message: str) -> str:
    lower = message.lower()
    if error_class == "RateLimitError" and (
        "credit_balance_exhausted" in lower
        or "insufficient_quota" in lower
        or "no credits remaining" in lower
    ):
        return "openai_http_429_credit_balance_exhausted"
    if "provider response contained no text" in lower:
        return "provider_response_no_text"
    if "rate limit" in lower or "error code: 429" in lower:
        return "http_429_rate_limit"
    if "timeout" in lower or error_class.lower().endswith("timeouterror"):
        return "timeout_or_uncertain_transport"
    if "context" in lower and ("limit" in lower or "length" in lower):
        return "context_limit"
    if "api_key" in lower or "api key" in lower or "authentication" in lower:
        return "authentication_or_local_credential"
    if "schema" in lower or "json" in lower:
        return "schema_or_json"
    return f"other:{error_class or 'unknown'}"


def _attempt_has_terminal(history: list[dict[str, Any]], attempt: int) -> bool:
    return any(event.get("attempt") == attempt and event.get("state") in TERMINAL_STATES for event in history)


def classify_slot(plan_row: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify one slot from complete append-only history."""
    latest = history[-1] if history else {}
    started_attempts = [event.get("attempt") for event in history if event.get("state") == "started" and isinstance(event.get("attempt"), int)]
    ambiguous_started = any(not _attempt_has_terminal(history, attempt) for attempt in started_attempts)
    current_state = str(latest.get("state", "missing"))
    error_class = str(latest.get("error_class", ""))
    message = sanitize_error_message(latest.get("error_message", ""))
    family = error_family(error_class, message) if current_state in ERROR_STATES | {"parse_failed"} else "none"

    submitted_evidence = "unknown"
    model_response_evidence = "unknown"
    if current_state == "completed":
        submitted_evidence = "yes"
        model_response_evidence = "yes"
        classification = ALREADY_COMPLETED
        reason = "A valid completed event is the current terminal state."
    elif ambiguous_started:
        classification = UNSAFE_AMBIGUOUS
        reason = "At least one started attempt has no definitive terminal event."
    elif current_state == "parse_failed":
        submitted_evidence = "yes"
        model_response_evidence = "yes"
        classification = NON_RETRYABLE_PARSE_OR_SCHEMA
        reason = "A model response was received but failed strict parsing/schema validation."
    elif current_state in ERROR_STATES:
        lower = message.lower()
        explicit_http_rejection = (
            ("error code: 429" in lower or "http 429" in lower)
            and ("credit_balance_exhausted" in lower or "insufficient_quota" in lower or "no credits remaining" in lower)
        )
        local_pre_submission = (
            "is required for" in lower and ("api_key" in lower or "api key" in lower)
        )
        if explicit_http_rejection:
            submitted_evidence = "yes"
            model_response_evidence = "no"
            classification = SAFE_DEFINITIVE_RETRY
            reason = "The provider returned an explicit HTTP 429 quota rejection; no semantic completion was returned."
        elif local_pre_submission:
            submitted_evidence = "no"
            model_response_evidence = "no"
            classification = SAFE_DEFINITIVE_RETRY
            reason = "A local credential preflight failure occurred before provider submission."
        elif message == "provider response contained no text":
            submitted_evidence = "yes"
            model_response_evidence = "unknown"
            classification = UNSAFE_AMBIGUOUS
            reason = "The adapter received a provider response but preserved neither raw content nor stop metadata."
        elif family == "timeout_or_uncertain_transport":
            classification = UNSAFE_AMBIGUOUS
            reason = "Transport timing does not prove whether the provider produced a completion."
        else:
            classification = UNKNOWN_REQUIRES_REVIEW
            reason = "The preserved error does not prove that no semantic completion occurred."
    else:
        classification = UNKNOWN_REQUIRES_REVIEW
        reason = f"Unsupported or missing current state: {current_state}."

    return {
        "request_slot_id": plan_row.get("request_slot_id"),
        "blinded_request_id": plan_row.get("blinded_request_id"),
        "task": plan_row.get("task"),
        "subtype": plan_row.get("subtype"),
        "judge_slot": plan_row.get("judge_slot"),
        "provider": plan_row.get("judge_provider"),
        "model": plan_row.get("judge_model"),
        "request_hash": plan_row.get("request_hash"),
        "latest_terminal_slot_state": current_state,
        "planned_event_n": sum(event.get("state") == "planned" for event in history),
        "started_event_n": sum(event.get("state") == "started" for event in history),
        "completed_event_n": sum(event.get("state") == "completed" for event in history),
        "provider_error_event_n": sum(event.get("state") in ERROR_STATES for event in history),
        "parse_failure_event_n": sum(event.get("state") == "parse_failed" for event in history),
        "error_class": error_class,
        "sanitized_error_message": message,
        "error_family": family,
        "provider_request_submitted_evidence": submitted_evidence,
        "model_response_received_evidence": model_response_evidence,
        "ambiguous_started_only_state": ambiguous_started,
        "retry_classification": classification,
        "classification_reason": reason,
    }


def _counts(records: list[dict[str, Any]], key_fn) -> dict[str, int]:
    return dict(sorted(Counter(key_fn(record) for record in records).items()))


def audit_failures(plan: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        histories[str(event.get("request_slot_id"))].append(event)
    all_records = [classify_slot(row, histories.get(str(row["request_slot_id"]), [])) for row in plan]
    current_failed = [record for record in all_records if record["latest_terminal_slot_state"] in ERROR_STATES or record["ambiguous_started_only_state"]]
    classifications = Counter(record["retry_classification"] for record in current_failed)
    return {
        "schema_version": "b4-full-frontier-failure-audit-v1",
        "mode": "read_only_offline",
        "total_plan_slots": len(plan),
        "current_completed_slots": sum(record["retry_classification"] == ALREADY_COMPLETED for record in all_records),
        "total_current_failed": len(current_failed),
        "completed_slots_in_failed_inventory": sum(record["retry_classification"] == ALREADY_COMPLETED for record in current_failed),
        "aggregates": {
            "provider_model": _counts(current_failed, lambda record: f"{record['provider']}::{record['model']}"),
            "task": _counts(current_failed, lambda record: str(record["task"])),
            "subtype": _counts(current_failed, lambda record: str(record.get("subtype") or "none")),
            "error_class": _counts(current_failed, lambda record: str(record["error_class"] or "none")),
            "error_family": _counts(current_failed, lambda record: str(record["error_family"])),
            "retry_classification": dict(sorted(classifications.items())),
            "submission_evidence": _counts(current_failed, lambda record: str(record["provider_request_submitted_evidence"])),
            "model_response_evidence": _counts(current_failed, lambda record: str(record["model_response_received_evidence"])),
        },
        "current_failed_slots": current_failed,
    }


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    aggregates = audit["aggregates"]
    lines = [
        "# B4 Full Frontier Failure Audit",
        "",
        "Read-only, offline audit. No benchmark question, context, reference answer, model answer, or judge response text is included.",
        "`SAFE_DEFINITIVE_RETRY` means the ledger proves no semantic completion was returned. It does not authorize a retry; the current runner remains fail-closed until a separate exact-slot authorization is implemented and explicitly invoked.",
        "",
        f"- total plan slots: {audit['total_plan_slots']}",
        f"- current completed slots: {audit['current_completed_slots']}",
        f"- current failed/incomplete slots: {audit['total_current_failed']}",
        f"- completed slots accidentally included: {audit['completed_slots_in_failed_inventory']}",
        "",
        "## Aggregate counts",
        "",
    ]
    for title, key in (
        ("Provider/model", "provider_model"), ("Task", "task"), ("Subtype", "subtype"),
        ("Error class", "error_class"), ("Error family", "error_family"), ("Retry classification", "retry_classification"),
        ("Submission evidence", "submission_evidence"), ("Model-response evidence", "model_response_evidence"),
    ):
        lines.extend([f"### {title}", ""])
        lines.extend(f"- `{name}`: {count}" for name, count in aggregates[key].items())
        lines.append("")
    lines.extend([
        "## Per-slot inventory",
        "",
        "The complete structured inventory is also available in `failure_audit.json`.",
        "",
        "| Request slot | Blinded request | Request hash | Task/subtype | Judge | Current state | Events P/S/C/E/F | Error | Submission/model response | Classification | Reason |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for record in audit["current_failed_slots"]:
        events = "/".join(str(record[key]) for key in (
            "planned_event_n", "started_event_n", "completed_event_n", "provider_error_event_n", "parse_failure_event_n"
        ))
        lines.append(
            "| `{request_slot_id}` | `{blinded_request_id}` | `{request_hash}` | {task}/{subtype} | {provider}/{model} ({judge_slot}) | {state} | {events} | {error_class}: {message} [{family}] | {submitted}/{response} | {classification} | {reason} |".format(
                request_slot_id=record["request_slot_id"], blinded_request_id=record["blinded_request_id"],
                request_hash=record["request_hash"],
                task=record["task"], subtype=record.get("subtype") or "none", provider=record["provider"],
                model=record["model"], judge_slot=record["judge_slot"], state=record["latest_terminal_slot_state"],
                events=events, error_class=record["error_class"],
                message=record["sanitized_error_message"].replace("|", "\\|"),
                family=record["error_family"], submitted=record["provider_request_submitted_evidence"],
                response=record["model_response_received_evidence"], classification=record["retry_classification"],
                reason=record["classification_reason"].replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, default=Path("results/b4_scoring_validation/private/full_frontier_judges"))
    args = parser.parse_args(argv)
    plan = read_jsonl(args.phase_dir / "full_validation_execution_plan.jsonl")
    events = read_jsonl(args.phase_dir / "judge_execution_ledger.jsonl")
    audit = audit_failures(plan, events)
    (args.phase_dir / "failure_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.phase_dir / "FAILURE_AUDIT.md", audit)
    aggregates = audit["aggregates"]
    provider = aggregates["provider_model"]
    task = aggregates["task"]
    classification = aggregates["retry_classification"]
    print(f"TOTAL_CURRENT_FAILED={audit['total_current_failed']}")
    print(f"OPENAI_FAILED={sum(value for key, value in provider.items() if key.startswith('openai::'))}")
    print(f"ANTHROPIC_FAILED={sum(value for key, value in provider.items() if key.startswith('anthropic::'))}")
    print(f"LME_FAILED={task.get('longmemeval', 0)}")
    print(f"MEMORA_FAILED={task.get('memora', 0)}")
    print(f"SAFE_DEFINITIVE_RETRY={classification.get(SAFE_DEFINITIVE_RETRY, 0)}")
    print(f"UNSAFE_AMBIGUOUS={classification.get(UNSAFE_AMBIGUOUS, 0)}")
    print(f"NON_RETRYABLE={classification.get(NON_RETRYABLE_PARSE_OR_SCHEMA, 0)}")
    print(f"UNKNOWN_REQUIRES_REVIEW={classification.get(UNKNOWN_REQUIRES_REVIEW, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
