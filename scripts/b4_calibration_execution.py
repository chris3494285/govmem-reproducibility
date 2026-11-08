#!/usr/bin/env python3
"""Paid-call execution primitives for the blinded B4 calibration.

This module is import-safe and performs no provider call unless ``execute_slots``
is invoked. Provider clients are created lazily after all execution gates pass.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

TERMINAL_STATES = {"completed", "parse_failed", "provider_error", "permanently_failed"}
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
RETRYABLE_NAMES = {
    "APITimeoutError", "APIConnectionError", "ConnectionError", "RateLimitError",
    "Timeout", "TimeoutError", "ServiceUnavailableError",
}
DEEPSEEK_JSON_MODE_INSTRUCTION = "Return only a valid JSON object matching the required schema."
PROVIDER_CREDENTIAL_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "kimi": "KIMI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
RECOVERY_REPAIR_VERSION = "b4-cheap-judge-json-credential-repair-v1"
MANUAL_RETRY_REPAIR_VERSION = "b4-frontier-one-slot-manual-retry-v1"
FULL_FRONTIER_OPENAI_RETRY_VERSION = "b4-full-frontier-openai-quota-retry-v1"
TWO_ANTHROPIC_NO_TEXT_RETRY_VERSION = "b4-two-anthropic-no-text-manual-retry-v1"
MANUAL_RETRY_AUTHORIZATION_STATE = "manual_retry_authorized"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def provider_request_prompt(provider: str, semantic_prompt: str) -> str:
    """Apply provider transport wording without changing the semantic rubric."""
    if provider == "deepseek" and DEEPSEEK_JSON_MODE_INSTRUCTION not in semantic_prompt:
        return semantic_prompt + "\n\n" + DEEPSEEK_JSON_MODE_INSTRUCTION
    return semantic_prompt


def missing_provider_credentials(slots: list[dict[str, Any]], environ: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Return missing credentials before any durable started event is written."""
    values = os.environ if environ is None else environ
    missing = []
    for provider in sorted({str(slot["judge_provider"]) for slot in slots}):
        variable = PROVIDER_CREDENTIAL_ENVS.get(provider)
        if variable is None:
            missing.append({"provider": provider, "environment_variable": "unsupported_provider"})
        elif not values.get(variable):
            missing.append({"provider": provider, "environment_variable": variable})
    return missing


def validate_provider_credentials(slots: list[dict[str, Any]], environ: dict[str, str] | None = None) -> None:
    missing = missing_provider_credentials(slots, environ)
    if missing:
        names = ", ".join(item["environment_variable"] for item in missing)
        raise ValueError(f"paid execution blocked before ledger start: missing credentials: {names}")


def classify_safe_recovery(slots: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify only failures proven to have produced no semantic completion."""
    current_by_logical = {
        (str(slot["blinded_request_id"]), str(slot["judge_slot"])): slot for slot in slots
    }
    histories: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        key = (str(event.get("blinded_request_id")), str(event.get("judge_slot")))
        histories.setdefault(key, []).append(event)
    safe: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    already_authorized: list[dict[str, Any]] = []
    for key, history in sorted(histories.items()):
        slot = current_by_logical.get(key)
        latest = history[-1]
        if slot is None:
            manual.append({"logical_key": key, "reason": "historical_slot_absent_from_repaired_plan"})
            continue
        if any(event.get("state") == "completed" for event in history):
            completed.append({"slot": slot, "reason": "completed_result_exists"})
            continue
        if latest.get("state") == "recovery_authorized":
            if _valid_recovery_bridge(history, slot):
                already_authorized.append({"slot": slot, "event": latest})
            else:
                manual.append({"slot": slot, "reason": "invalid_or_stale_recovery_authorization"})
            continue
        if latest.get("state") == "started":
            ambiguous.append({"slot": slot, "reason": "started_without_terminal_event"})
            continue
        if latest.get("state") == "parse_failed":
            manual.append({"slot": slot, "reason": "semantic_response_received_but_parse_failed"})
            continue
        if latest.get("state") not in {"provider_error", "permanently_failed"}:
            manual.append({"slot": slot, "reason": f"unsupported_terminal_state:{latest.get('state')}"})
            continue
        provider = str(latest.get("judge_provider"))
        error_class = str(latest.get("error_class"))
        message = str(latest.get("error_message"))
        old_hash = str(latest.get("request_hash"))
        semantic_hash = str(slot.get("semantic_payload_hash"))
        if provider == "deepseek":
            known_rejection = (
                error_class == "BadRequestError"
                and "Prompt must contain the word 'json'" in message
                and "response_format" in message
                and "json_object" in message
                and old_hash == semantic_hash
                and slot["request_hash"] != old_hash
            )
            if known_rejection:
                safe.append({"slot": slot, "terminal": latest, "reason": "deepseek_json_mode_http_400_no_completion"})
            else:
                manual.append({"slot": slot, "reason": "deepseek_failure_not_known_definitive_rejection"})
        elif provider == "kimi":
            known_local = (
                error_class == "RuntimeError"
                and message == "KIMI_API_KEY is required for kimi judge execution"
                and old_hash == slot["request_hash"]
            )
            if known_local:
                safe.append({"slot": slot, "terminal": latest, "reason": "kimi_local_missing_credential_before_provider_call"})
            else:
                manual.append({"slot": slot, "reason": "kimi_failure_not_known_local_credential_rejection"})
        else:
            manual.append({"slot": slot, "reason": "provider_not_covered_by_repair"})
    return {
        "historical_event_n": len(events),
        "historical_slot_n": len(histories),
        "safe": safe,
        "already_authorized": already_authorized,
        "completed": completed,
        "ambiguous": ambiguous,
        "manual": manual,
    }


def audit_recovery_history(slots: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate immutable incident evidence from current repaired slot state.

    Counts are slot/event invariants and intentionally do not constrain the
    total append-only ledger length.
    """
    current_by_logical = {
        (str(slot["blinded_request_id"]), str(slot["judge_slot"])): slot for slot in slots
    }
    definitive_errors: list[dict[str, Any]] = []
    deepseek_errors: list[dict[str, Any]] = []
    kimi_errors: list[dict[str, Any]] = []
    for event in events:
        key = (str(event.get("blinded_request_id")), str(event.get("judge_slot")))
        slot = current_by_logical.get(key)
        if slot is None or event.get("state") != "provider_error" or event.get("attempt") != 1:
            continue
        message = str(event.get("error_message", ""))
        if (
            event.get("judge_provider") == "deepseek"
            and event.get("error_class") == "BadRequestError"
            and "Prompt must contain the word 'json'" in message
            and "response_format" in message
            and "json_object" in message
            and event.get("request_hash") == slot.get("semantic_payload_hash")
        ):
            definitive_errors.append(event)
            deepseek_errors.append(event)
        elif (
            event.get("judge_provider") == "kimi"
            and event.get("error_class") == "RuntimeError"
            and message == "KIMI_API_KEY is required for kimi judge execution"
            and event.get("request_hash") == slot.get("semantic_payload_hash")
        ):
            definitive_errors.append(event)
            kimi_errors.append(event)

    original_ids = {str(event["request_slot_id"]) for event in definitive_errors}
    original_planned = {
        str(event["request_slot_id"])
        for event in events
        if event.get("state") == "planned" and event.get("attempt") == 0 and str(event.get("request_slot_id")) in original_ids
    }
    original_started = {
        str(event["request_slot_id"])
        for event in events
        if event.get("state") == "started" and event.get("attempt") == 1 and str(event.get("request_slot_id")) in original_ids
    }
    original_terminal = {str(event["request_slot_id"]) for event in definitive_errors}

    authorizations = [
        event for event in events
        if event.get("state") == "recovery_authorized"
        and event.get("repair_version") == RECOVERY_REPAIR_VERSION
        and str(event.get("request_slot_id")) in original_ids
    ]
    authorized_original_ids = {str(event["request_slot_id"]) for event in authorizations}
    authorized_repaired_ids = {str(event.get("repaired_request_slot_id")) for event in authorizations}

    current_ids = {str(slot["request_slot_id"]) for slot in slots}
    completed_events = [
        event for event in events
        if event.get("state") == "completed"
        and str(event.get("request_slot_id")) in current_ids
        and any(
            slot["request_slot_id"] == event.get("request_slot_id")
            and slot["request_hash"] == event.get("request_hash")
            for slot in slots
        )
    ]
    completed_ids = {str(event["request_slot_id"]) for event in completed_events}
    completed_provider_by_id = {
        str(event["request_slot_id"]): str(event.get("judge_provider")) for event in completed_events
    }
    completed_by_provider = Counter(completed_provider_by_id.values())

    terminal_states = {"completed", "provider_error", "retryable_error", "permanently_failed", "parse_failed"}
    latest_attempt_state: dict[str, tuple[int, str]] = {}
    started_attempts: dict[tuple[str, int], bool] = {}
    terminal_attempts: set[tuple[str, int]] = set()
    for event in events:
        slot_id = str(event.get("request_slot_id"))
        if slot_id not in current_ids:
            continue
        attempt = event.get("attempt")
        if not isinstance(attempt, int) or attempt < 1:
            continue
        state = str(event.get("state"))
        if state == "started":
            started_attempts[(slot_id, attempt)] = True
        if state in terminal_states:
            terminal_attempts.add((slot_id, attempt))
            previous = latest_attempt_state.get(slot_id)
            if previous is None or attempt >= previous[0]:
                latest_attempt_state[slot_id] = (attempt, state)
    ambiguous_started = {
        slot_id for slot_id, attempt in started_attempts if (slot_id, attempt) not in terminal_attempts
    }
    current_provider_error = {
        slot_id for slot_id, (_attempt, state) in latest_attempt_state.items()
        if state in {"provider_error", "retryable_error", "permanently_failed"} and slot_id not in completed_ids
    }
    current_parse_failure = {
        slot_id for slot_id, (_attempt, state) in latest_attempt_state.items()
        if state == "parse_failed" and slot_id not in completed_ids
    }

    return {
        "original_incident": {
            "planned_slot_ids": original_planned,
            "started_slot_ids": original_started,
            "provider_error_slot_ids": original_terminal,
            "deepseek_error_slot_ids": {str(event["request_slot_id"]) for event in deepseek_errors},
            "kimi_error_slot_ids": {str(event["request_slot_id"]) for event in kimi_errors},
        },
        "recovery": {
            "authorization_event_n": len(authorizations),
            "authorized_original_slot_ids": authorized_original_ids,
            "authorized_repaired_slot_ids": authorized_repaired_ids,
        },
        "current": {
            "completed_slot_ids": completed_ids,
            "completed_by_provider": dict(completed_by_provider),
            "provider_error_slot_ids": current_provider_error,
            "parse_failure_slot_ids": current_parse_failure,
            "ambiguous_started_slot_ids": ambiguous_started - completed_ids,
        },
    }


def append_recovery_authorizations(ledger: "DurableLedger", classification: dict[str, Any]) -> int:
    """Append auditable recovery bridges; never alter existing events."""
    count = 0
    for item in classification["safe"]:
        slot = item["slot"]
        terminal = item["terminal"]
        event = {
            "request_slot_id": terminal["request_slot_id"],
            "blinded_request_id": terminal["blinded_request_id"],
            "judge_slot": terminal["judge_slot"],
            "judge_provider": terminal["judge_provider"],
            "judge_model": terminal["judge_model"],
            "request_hash": terminal["request_hash"],
            "state": "recovery_authorized",
            "previous_terminal_state": terminal["state"],
            "previous_error_class": terminal.get("error_class"),
            "previous_error_message_hash": hashlib.sha256(str(terminal.get("error_message", "")).encode()).hexdigest(),
            "recovery_reason": item["reason"],
            "repair_version": RECOVERY_REPAIR_VERSION,
            "semantic_payload_hash": slot["semantic_payload_hash"],
            "repaired_request_slot_id": slot["request_slot_id"],
            "repaired_request_hash": slot["request_hash"],
            "timestamp_ns": time.time_ns(),
        }
        ledger.append(event)
        count += 1
    return count


def _manual_retry_identity(record: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(record.get("request_slot_id", "")),
        str(record.get("blinded_request_id", "")),
        str(record.get("judge_slot", "")),
        str(record.get("judge_provider", "")),
        str(record.get("judge_model", "")),
        str(record.get("request_hash", "")),
    )


def manual_retry_preflight(
    slots: list[dict[str, Any]],
    events: list[dict[str, Any]],
    expected: dict[str, str],
) -> dict[str, Any]:
    """Validate one explicitly named failed slot without broadening retry policy."""
    expected_identity = _manual_retry_identity(expected)
    matches = [slot for slot in slots if str(slot.get("request_slot_id")) == expected_identity[0]]
    if len(matches) != 1 or _manual_retry_identity(matches[0]) != expected_identity:
        raise LedgerConflictError("manual retry target identity/hash/provider/model does not match the frozen plan")
    target = matches[0]
    slot_ids = {str(slot["request_slot_id"]) for slot in slots}
    target_history = [event for event in events if str(event.get("request_slot_id")) == expected_identity[0]]
    if not target_history:
        raise LedgerConflictError("manual retry target has no ledger history")
    if any(_manual_retry_identity(event) != expected_identity for event in target_history):
        raise LedgerConflictError("manual retry target ledger identity drift")
    if any(event.get("state") == "completed" for event in target_history):
        raise LedgerConflictError("completed slots cannot be manually retried")

    latest_by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        slot_id = str(event.get("request_slot_id", ""))
        if slot_id in slot_ids:
            latest_by_id[slot_id] = event
    completed_ids = {
        slot_id for slot_id, event in latest_by_id.items()
        if event.get("state") == "completed"
    }
    latest = target_history[-1]
    candidate = latest.get("state") == "provider_error"
    authorized = latest.get("state") == MANUAL_RETRY_AUTHORIZATION_STATE
    if candidate:
        if str(latest.get("error_class")) != str(expected.get("expected_error_class")):
            raise LedgerConflictError("manual retry terminal error class does not match authorization")
        if str(latest.get("error_message")) != str(expected.get("expected_error_message")):
            raise LedgerConflictError("manual retry terminal error message does not match authorization")
    elif authorized:
        if not _valid_manual_retry_authorization(target_history, target):
            raise LedgerConflictError("manual retry authorization is invalid or stale")
    else:
        raise LedgerConflictError(f"manual retry target state is not authorizable: {latest.get('state')}")

    protected_ids = slot_ids - {expected_identity[0]}
    all_other_completed = protected_ids == completed_ids
    if not all_other_completed:
        raise LedgerConflictError("not all non-target frontier slots are completed and protected")
    return {
        "total_request_slots": len(slots),
        "completed_slots": len(completed_ids),
        "authorization_candidate_slots": 1 if candidate else 0,
        "authorized_recovery_slots": 1 if authorized else 0,
        "target_request_slot_id": target["request_slot_id"],
        "target_blinded_request_id": target["blinded_request_id"],
        "target_judge_provider": target["judge_provider"],
        "target_judge_model": target["judge_model"],
        "target_request_hash": target["request_hash"],
        "all_other_completed_slots_protected": all_other_completed,
        "maximum_new_paid_calls": 1,
    }


def append_manual_retry_authorization(
    ledger: "DurableLedger",
    slots: list[dict[str, Any]],
    expected: dict[str, str],
) -> dict[str, Any]:
    """Append one identity-bound authorization after strict preflight."""
    summary = manual_retry_preflight(slots, ledger.read(), expected)
    if summary["authorization_candidate_slots"] != 1 or summary["authorized_recovery_slots"]:
        raise LedgerConflictError("manual retry is already authorized or is not an authorization candidate")
    events = ledger.read()
    terminal = next(
        event for event in reversed(events)
        if str(event.get("request_slot_id")) == str(expected["request_slot_id"])
    )
    started_n = sum(
        event.get("state") == "started"
        for event in events
        if str(event.get("request_slot_id")) == str(expected["request_slot_id"])
    )
    event = {
        "request_slot_id": expected["request_slot_id"],
        "blinded_request_id": expected["blinded_request_id"],
        "judge_slot": expected["judge_slot"],
        "judge_provider": expected["judge_provider"],
        "judge_model": expected["judge_model"],
        "request_hash": expected["request_hash"],
        "state": MANUAL_RETRY_AUTHORIZATION_STATE,
        "previous_terminal_state": terminal.get("state"),
        "previous_terminal_attempt": terminal.get("attempt"),
        "previous_error_class": terminal.get("error_class"),
        "previous_error_message_hash": hashlib.sha256(str(terminal.get("error_message", "")).encode()).hexdigest(),
        "authorized_attempt": started_n + 1,
        "maximum_new_paid_calls": 1,
        "recovery_reason": "user_authorized_one_paid_retry_after_unpreserved_no_text_response",
        "repair_version": MANUAL_RETRY_REPAIR_VERSION,
        "timestamp_ns": time.time_ns(),
    }
    ledger.append(event)
    return event


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _parse_judge_output(raw: str, task: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("judge output must be a JSON object")
    decisions = {
        "longmemeval": {"correct", "incorrect", "abstention_correct", "abstention_incorrect", "adjudication_required"},
        "memora": {"pass", "fail", "adjudication_required"},
    }[task]
    required = {"decision", "confidence", "short_reason"} | ({"answerability_basis"} if task == "longmemeval" else set())
    if set(value) != required:
        raise ValueError(f"judge output keys must be exactly {sorted(required)}")
    if value["decision"] not in decisions:
        raise ValueError("invalid decision")
    if value["confidence"] not in {"high", "medium", "low"}:
        raise ValueError("invalid or missing confidence")
    if not isinstance(value["short_reason"], str) or not value["short_reason"].strip():
        raise ValueError("missing short_reason")
    if task == "longmemeval" and value["answerability_basis"] not in {"answerable", "insufficient_information", "ambiguous"}:
        raise ValueError("invalid answerability_basis")
    return value


def request_slot_id(
    blinded_request_id: str,
    judge_slot: str,
    provider: str,
    model: str,
    request_hash: str,
) -> str:
    """Return the stable identity of one logical paid request."""
    if judge_slot not in {"A", "B", "T"}:
        raise ValueError("judge_slot must be A, B, or T")
    identity = {
        "blinded_request_id": blinded_request_id,
        "judge_slot": judge_slot,
        "provider": provider,
        "model": model,
        "request_hash": request_hash,
    }
    return "b4-slot::" + _digest(identity)


def logical_slot_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record["blinded_request_id"]),
        str(record["judge_slot"]),
        str(record["judge_provider"]),
        str(record["judge_model"]),
    )


def estimate_input_tokens(prompt: str, model: str) -> dict[str, Any]:
    """Use a local model tokenizer when available, else a safe byte upper bound."""
    try:
        import tiktoken

        encoder = tiktoken.encoding_for_model(model)
        return {
            "estimated_tokens": len(encoder.encode(prompt)),
            "method": "exact_local_model_tokenizer",
            "approximate_tokens_char4": (len(prompt.encode("utf-8")) + 3) // 4,
        }
    except Exception:
        # A UTF-8 byte count is deliberately conservative for token-limit
        # gating. The char/4 figure is retained only as a non-gating diagnostic.
        return {
            "estimated_tokens": len(prompt.encode("utf-8")),
            "method": "utf8_bytes_conservative_upper_bound",
            "approximate_tokens_char4": (len(prompt.encode("utf-8")) + 3) // 4,
        }


def reduce_lme_native_context(native_context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Optionally retain only benchmark-native answer sessions.

    Selection depends solely on native session identifiers. It never receives
    the reference answer or model answer. Empty/inconsistent native relevance
    metadata fails safe to the full context.
    """
    context = copy.deepcopy(native_context)
    sessions = context.get("haystack_sessions")
    session_ids = context.get("haystack_session_ids")
    answer_ids = context.get("answer_session_ids")
    if not all(isinstance(value, list) for value in (sessions, session_ids, answer_ids)):
        return context, {"reduced": False, "method": "full_context_missing_native_alignment", "answerability_evidence_preserved": True}
    if not answer_ids:
        return context, {"reduced": False, "method": "full_context_no_native_relevance_ids", "answerability_evidence_preserved": True}
    if len(sessions) != len(session_ids) or not set(answer_ids).issubset(set(session_ids)):
        return context, {"reduced": False, "method": "full_context_inconsistent_native_alignment", "answerability_evidence_preserved": True}
    selected = [index for index, identifier in enumerate(session_ids) if identifier in set(answer_ids)]
    if len(selected) == len(sessions):
        return context, {"reduced": False, "method": "full_context_all_sessions_native_relevant", "answerability_evidence_preserved": True}
    context["haystack_sessions"] = [sessions[index] for index in selected]
    context["haystack_session_ids"] = [session_ids[index] for index in selected]
    dates = context.get("haystack_dates")
    if isinstance(dates, list) and len(dates) == len(sessions):
        context["haystack_dates"] = [dates[index] for index in selected]
    return context, {
        "reduced": True,
        "method": "benchmark_native_answer_session_ids",
        "answerability_evidence_preserved": set(answer_ids).issubset(set(context["haystack_session_ids"])),
        "original_session_n": len(sessions),
        "retained_session_n": len(selected),
    }


def build_slots(
    prompt_rows: list[dict[str, Any]],
    *,
    judge_a_provider: str,
    judge_a_model: str,
    judge_b_provider: str,
    judge_b_model: str,
) -> list[dict[str, Any]]:
    configs = {
        "A": (judge_a_provider, judge_a_model),
        "B": (judge_b_provider, judge_b_model),
    }
    slots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in prompt_rows:
        slot = str(row["judge_slot"])
        provider, model = configs[slot]
        semantic_prompt = str(row["prompt"])
        request_prompt = provider_request_prompt(provider, semantic_prompt)
        semantic_hash = prompt_hash(semantic_prompt)
        req_hash = prompt_hash(request_prompt)
        slot_id = request_slot_id(str(row["blinded_request_id"]), slot, provider, model, req_hash)
        if slot_id in seen:
            raise ValueError(f"duplicate request_slot_id: {slot_id}")
        seen.add(slot_id)
        slots.append({
            "request_slot_id": slot_id,
            "blinded_request_id": row["blinded_request_id"],
            "task": row["task"],
            "subtype": row.get("subtype"),
            "judge_slot": slot,
            "judge_provider": provider,
            "judge_model": model,
            "semantic_payload_hash": semantic_hash,
            "request_hash": req_hash,
            "prompt_format_version": "deepseek-json-mode-v1" if provider == "deepseek" else "semantic-prompt-v1",
            "prompt": request_prompt,
        })
    return slots


class LedgerConflictError(RuntimeError):
    pass


class DurableLedger:
    """Append-only JSONL ledger with flush/fsync durability."""

    def __init__(self, path: Path):
        self.path = path

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LedgerConflictError(f"malformed ledger line {line_number}") from exc
                if not isinstance(value, dict):
                    raise LedgerConflictError(f"ledger line {line_number} is not an object")
                records.append(value)
        return records

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def _events_by_identity(events: Iterable[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    by_logical: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        slot_id = str(event.get("request_slot_id", ""))
        if not slot_id:
            raise LedgerConflictError("ledger event lacks request_slot_id")
        by_id.setdefault(slot_id, []).append(event)
        key = (str(event["blinded_request_id"]), str(event["judge_slot"]))
        by_logical.setdefault(key, []).append(event)
    return by_id, by_logical


def _valid_recovery_bridge(events: list[dict[str, Any]], slot: dict[str, Any]) -> bool:
    return any(
        event.get("state") == "recovery_authorized"
        and event.get("repair_version") == RECOVERY_REPAIR_VERSION
        and event.get("repaired_request_slot_id") == slot["request_slot_id"]
        and event.get("repaired_request_hash") == slot["request_hash"]
        and event.get("semantic_payload_hash") == slot.get("semantic_payload_hash")
        for event in events
    )


def _valid_manual_retry_authorization(events: list[dict[str, Any]], slot: dict[str, Any]) -> bool:
    if len(events) < 2:
        return False
    authorization = events[-1]
    terminal = events[-2]
    version = authorization.get("repair_version")
    expected_terminal_state = {
        MANUAL_RETRY_REPAIR_VERSION: "provider_error",
        FULL_FRONTIER_OPENAI_RETRY_VERSION: "permanently_failed",
        TWO_ANTHROPIC_NO_TEXT_RETRY_VERSION: "provider_error",
    }.get(version)
    quota_recovery_valid = True
    if version == FULL_FRONTIER_OPENAI_RETRY_VERSION:
        message = str(terminal.get("error_message", "")).lower()
        quota_recovery_valid = (
            slot.get("judge_provider") == "openai"
            and slot.get("judge_model") == "gpt-5.6-sol"
            and terminal.get("error_class") == "RateLimitError"
            and ("credit_balance_exhausted" in message or "insufficient_quota" in message)
            and ("error code: 429" in message or "no credits remaining" in message)
        )
    elif version == TWO_ANTHROPIC_NO_TEXT_RETRY_VERSION:
        quota_recovery_valid = (
            slot.get("judge_provider") == "anthropic"
            and slot.get("judge_model") == "claude-sonnet-5"
            and slot.get("task") == "longmemeval"
            and terminal.get("error_class") == "ValueError"
            and terminal.get("error_message") == "provider response contained no text"
        )
    return (
        authorization.get("state") == MANUAL_RETRY_AUTHORIZATION_STATE
        and expected_terminal_state is not None
        and _manual_retry_identity(authorization) == _manual_retry_identity(slot)
        and authorization.get("maximum_new_paid_calls") == 1
        and authorization.get("authorized_attempt")
        == sum(event.get("state") == "started" for event in events) + 1
        and terminal.get("state") == expected_terminal_state
        and authorization.get("previous_terminal_state") == terminal.get("state")
        and authorization.get("previous_terminal_attempt") == terminal.get("attempt")
        and authorization.get("previous_error_class") == terminal.get("error_class")
        and authorization.get("previous_error_message_hash")
        == hashlib.sha256(str(terminal.get("error_message", "")).encode()).hexdigest()
        and quota_recovery_valid
    )


def authorized_manual_retry_slots(slots: list[dict[str, Any]], ledger: DurableLedger) -> list[dict[str, Any]]:
    """Return only slots whose latest event is a valid, unconsumed authorization."""
    events = ledger.read()
    by_id, _ = _events_by_identity(events)
    selected = []
    for slot in slots:
        history = by_id.get(str(slot["request_slot_id"]), [])
        if history and history[-1].get("state") == MANUAL_RETRY_AUTHORIZATION_STATE:
            if not _valid_manual_retry_authorization(history, slot):
                raise LedgerConflictError(f"invalid manual retry authorization for {slot['request_slot_id']}")
            selected.append(slot)
    return selected


def resume_plan(slots: list[dict[str, Any]], ledger: DurableLedger, max_attempts: int = 3) -> dict[str, Any]:
    events = ledger.read()
    by_id, by_logical = _events_by_identity(events)
    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for slot in slots:
        key = (str(slot["blinded_request_id"]), str(slot["judge_slot"]))
        new_assignment = (str(slot["judge_provider"]), str(slot["judge_model"]), str(slot["request_hash"]))
        logical_history = by_logical.get(key, [])
        old_assignments = {
            (str(event["judge_provider"]), str(event["judge_model"]), str(event.get("request_hash", "")))
            for event in logical_history
        }
        bridge = _valid_recovery_bridge(logical_history, slot)
        if any(assignment != new_assignment for assignment in old_assignments) and not bridge:
            raise LedgerConflictError(f"provider, model, or request hash mismatch for {key}")
        history = by_id.get(slot["request_slot_id"], [])
        if any(event.get("state") == "completed" for event in history):
            completed.append(slot)
            continue
        if any(event.get("state") == "completed" for event in logical_history):
            raise LedgerConflictError(f"completed historical slot cannot be migrated for {key}")
        latest = history[-1] if history else None
        attempts = sum(event.get("state") == "started" for event in logical_history)
        if latest and latest.get("state") == MANUAL_RETRY_AUTHORIZATION_STATE:
            if not _valid_manual_retry_authorization(history, slot):
                raise LedgerConflictError(f"invalid manual retry authorization for {key}")
            pending.append({
                **slot,
                "next_attempt": attempts + 1,
                "manual_retry_max_new_calls": 1,
            })
        elif latest and latest.get("state") == "started":
            blocked.append({**slot, "blocker": "indeterminate_started_call_requires_manual_resolution"})
        elif latest and latest.get("state") in {"parse_failed", "provider_error", "permanently_failed"}:
            blocked.append({**slot, "blocker": str(latest.get("state"))})
        elif attempts >= max_attempts:
            blocked.append({**slot, "blocker": "retry_budget_exhausted"})
        elif bridge or (latest and latest.get("state") == "recovery_authorized"):
            pending.append({**slot, "next_attempt": attempts + 1})
        else:
            pending.append({**slot, "next_attempt": attempts + 1})
    return {"completed": completed, "pending": pending, "blocked": blocked, "events": events}


def classify_provider_error(exc: BaseException) -> tuple[bool, str, str]:
    details = getattr(exc, "details", {}) or {}
    status = getattr(exc, "status_code", None) or details.get("http_status")
    name = str(details.get("provider_exception") or type(exc).__name__)
    category = str(details.get("error_category") or "")
    retryable = status in RETRYABLE_STATUS_CODES or name in RETRYABLE_NAMES or category in {"rate_limit", "timeout", "transport_error", "server_error"}
    safe_message = re.sub(r"(?:sk-ant-|sk-)[A-Za-z0-9_-]+", "[REDACTED]", str(exc))[:2000]
    return retryable, name, safe_message


def _judge_schema(task: str) -> dict[str, Any]:
    if task == "longmemeval":
        properties = {
            "decision": {"type": "string", "enum": ["correct", "incorrect", "abstention_correct", "abstention_incorrect", "adjudication_required"]},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "short_reason": {"type": "string", "minLength": 1},
            "answerability_basis": {"type": "string", "enum": ["answerable", "insufficient_information", "ambiguous"]},
        }
    else:
        properties = {
            "decision": {"type": "string", "enum": ["pass", "fail", "adjudication_required"]},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "short_reason": {"type": "string", "minLength": 1},
        }
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _memora_batch_schema(prompt: str) -> dict[str, Any]:
    """Strict transport schema for the frozen batched native Memora contract.

    B4's original ``memora`` task evaluates one criterion and intentionally
    retains its single-decision schema.  Full native evaluation batches all
    criteria for one answer, so its transport schema is derived only from the
    criterion IDs already frozen in that semantic prompt.
    """
    try:
        value = json.loads(prompt)
    except json.JSONDecodeError as exc:
        raise ValueError("batched Memora prompt is not valid JSON") from exc
    criteria = value.get("criteria") if isinstance(value, dict) else None
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("batched Memora prompt has no criteria")
    criterion_ids = [str(item.get("criterion_id")) for item in criteria if isinstance(item, dict)]
    if len(criterion_ids) != len(criteria) or any(not identifier for identifier in criterion_ids):
        raise ValueError("batched Memora prompt has an invalid criterion ID")
    if len(set(criterion_ids)) != len(criterion_ids):
        raise ValueError("batched Memora prompt has duplicate criterion IDs")
    item_properties = {
        "criterion_id": {"type": "string", "minLength": 1},
        "decision": {"type": "string", "enum": ["pass", "fail", "adjudication_required"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "short_reason": {"type": "string", "minLength": 1},
    }
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": item_properties,
                    "required": list(item_properties),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def _schema_for_request(task: str, prompt: str) -> dict[str, Any]:
    if task == "memora_batch":
        return _memora_batch_schema(prompt)
    return _judge_schema(task)


def _response_text(response: Any) -> str:
    text = response.get("output_text") if isinstance(response, dict) else getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    content = response.get("content") if isinstance(response, dict) else getattr(response, "content", None)
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if isinstance(value, str):
                parts.append(value)
    if parts:
        return "".join(parts)
    output = response.get("output") if isinstance(response, dict) else getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            blocks = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if isinstance(blocks, list):
                for block in blocks:
                    value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                    if isinstance(value, str):
                        parts.append(value)
    if not parts:
        raise ValueError("provider response contained no text")
    return "".join(parts)


_JUDGE_RAW_DIAGNOSTIC_CHARS = 256
_JUDGE_SECRET_RE = re.compile(r"(?:sk-ant-|sk-)[A-Za-z0-9_-]+")


def _redact_diagnostic(value: Any) -> str:
    return _JUDGE_SECRET_RE.sub("[REDACTED]", str(value))


def judge_response_diagnostics(
    response: Any = None, *, raw_text: str | None = None, json_parse_ok: bool | None = None
) -> dict[str, Any]:
    """Safe, bounded Anthropic/generic judge transport diagnostics.

    Captures stop_reason, usage, request id, content-block shape (including
    non-text blocks such as thinking/tool/refusal), raw-text presence/size, and
    whether the raw text parses as JSON.  The bounded raw preview is redacted
    and never used to repair or accept a payload.
    """
    if isinstance(response, dict):
        content = response.get("content")
    else:
        content = getattr(response, "content", None)
    blocks = content if isinstance(content, list) else None

    def block_type(block: Any) -> str | None:
        value = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        return str(value) if value is not None else None

    def block_text(block: Any) -> str | None:
        value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        return value if isinstance(value, str) else None

    block_types = [block_type(block) for block in (blocks or [])]
    text_blocks = [value for value in (block_text(block) for block in (blocks or [])) if value is not None]
    resolved = raw_text if raw_text is not None else ("".join(text_blocks) if text_blocks else None)

    usage = getattr(response, "usage", None) if response is not None else None

    def usage_value(name: str) -> int | None:
        if usage is None:
            return None
        raw = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        return int(raw) if raw is not None else None

    input_tokens = usage_value("input_tokens")
    output_tokens = usage_value("output_tokens")
    total_tokens = usage_value("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    request_id = None
    stop_reason = None
    if isinstance(response, dict):
        request_id = response.get("request_id")
        stop_reason = response.get("stop_reason")
    elif response is not None:
        request_id = getattr(response, "_request_id", None) or getattr(response, "request_id", None)
        stop_reason = getattr(response, "stop_reason", None)

    non_text_types = [value for value in block_types if value is not None and value != "text"]
    if json_parse_ok is None:
        if resolved is None:
            json_parse_ok = False
        else:
            try:
                json.loads(resolved)
                json_parse_ok = True
            except Exception:
                json_parse_ok = False
    diagnostics: dict[str, Any] = {
        "stop_reason": str(stop_reason) if stop_reason is not None else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "request_id": _redact_diagnostic(request_id) if request_id is not None else None,
        "content_list_present": blocks is not None,
        "content_block_types": block_types,
        "content_block_count": len(blocks or []),
        "text_block_count": len(text_blocks),
        "non_text_block_types": non_text_types,
        "has_non_text_content": bool(non_text_types),
        "has_raw_text": bool(resolved),
        "raw_text_char_count": len(resolved) if resolved is not None else 0,
        "json_parse_ok": json_parse_ok,
    }
    if resolved:
        safe = _redact_diagnostic(resolved)
        diagnostics.update({
            "raw_text_preview": safe[:_JUDGE_RAW_DIAGNOSTIC_CHARS],
            "raw_text_tail": safe[-_JUDGE_RAW_DIAGNOSTIC_CHARS:],
            "raw_text_preview_truncated": len(safe) > _JUDGE_RAW_DIAGNOSTIC_CHARS,
            "raw_text_open_braces": safe.count("{"),
            "raw_text_close_braces": safe.count("}"),
            "raw_text_last_char": safe[-1] if safe else "",
        })
    return diagnostics


_OPENAI_VISIBLE_TEXT_BLOCK_TYPES = frozenset({"output_text", "text"})


def _openai_type(value: Any) -> str | None:
    item = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    return str(item) if item is not None else None


def openai_response_diagnostics(
    response: Any = None,
    *,
    raw_text: str | None = None,
    json_parse_ok: bool | None = None,
    configured_max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Safe, bounded OpenAI Responses-API judge transport diagnostics.

    Captures structural metadata only (status, incomplete reason, ids, usage,
    output item types and content block types) plus a redacted preview of the
    *visible* output text.  Reasoning/chain-of-thought content is never read or
    persisted: only top-level item/block ``type`` strings are inspected, and
    text is extracted exclusively from visible text blocks.
    """
    def read(container: Any, key: str) -> Any:
        return container.get(key) if isinstance(container, dict) else getattr(container, key, None)

    items = read(response, "output") if response is not None else None
    items = items if isinstance(items, list) else None
    item_types: list[str | None] = []
    content_block_types: list[str | None] = []
    text_blocks: list[str] = []
    for item in (items or []):
        item_types.append(_openai_type(item))
        blocks = read(item, "content")
        if isinstance(blocks, list):
            for block in blocks:
                block_type = _openai_type(block)
                content_block_types.append(block_type)
                if block_type in _OPENAI_VISIBLE_TEXT_BLOCK_TYPES:
                    text = read(block, "text")
                    if isinstance(text, str):
                        text_blocks.append(text)
    top_text = read(response, "output_text") if response is not None else None
    resolved = raw_text
    if resolved is None:
        if isinstance(top_text, str) and top_text:
            resolved = top_text
        elif text_blocks:
            resolved = "".join(text_blocks)

    usage = read(response, "usage") if response is not None else None

    def usage_value(name: str) -> int | None:
        if usage is None:
            return None
        raw = read(usage, name)
        return int(raw) if raw is not None else None

    input_tokens = usage_value("input_tokens")
    output_tokens = usage_value("output_tokens")
    total_tokens = usage_value("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    response_id = read(response, "id") if response is not None else None
    request_id = None
    if response is not None:
        request_id = getattr(response, "_request_id", None) or read(response, "request_id")
    status = read(response, "status") if response is not None else None
    incomplete = read(response, "incomplete_details") if response is not None else None
    incomplete_reason = read(incomplete, "reason") if incomplete is not None else None
    if incomplete_reason is None and not isinstance(incomplete, (dict, type(None))):
        incomplete_reason = getattr(incomplete, "reason", None)
    if json_parse_ok is None:
        if resolved is None:
            json_parse_ok = False
        else:
            try:
                json.loads(resolved)
                json_parse_ok = True
            except Exception:
                json_parse_ok = False
    diagnostics: dict[str, Any] = {
        "provider": "openai",
        "status": str(status) if status is not None else None,
        "response_id": _redact_diagnostic(response_id) if response_id is not None else None,
        "request_id": _redact_diagnostic(request_id) if request_id is not None else None,
        "incomplete_reason": str(incomplete_reason) if incomplete_reason is not None else None,
        "incomplete_details": {"reason": str(incomplete_reason)} if incomplete_reason is not None else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "configured_max_output_tokens": configured_max_output_tokens,
        "output_item_types": item_types,
        "output_item_count": len(items or []),
        "content_block_types": content_block_types,
        "content_block_count": len(content_block_types),
        "text_block_count": len(text_blocks),
        "has_raw_text": bool(resolved),
        "raw_text_char_count": len(resolved) if resolved is not None else 0,
        "json_parse_ok": json_parse_ok,
    }
    if resolved:
        safe = _redact_diagnostic(resolved)
        diagnostics.update({
            "raw_text_preview": safe[:_JUDGE_RAW_DIAGNOSTIC_CHARS],
            "raw_text_tail": safe[-_JUDGE_RAW_DIAGNOSTIC_CHARS:],
            "raw_text_preview_truncated": len(safe) > _JUDGE_RAW_DIAGNOSTIC_CHARS,
        })
    return diagnostics


class OpenAIJudgeAdapter:
    provider_name = "openai"

    def __init__(self, model: str, client: Any | None = None, max_output_tokens: int = 1024):
        self.model = model
        self.client = client
        self.max_output_tokens = max_output_tokens
        self.last_diagnostics: dict[str, Any] | None = None
        self.last_usage_details: dict[str, int | None] | None = None

    def call(self, prompt: str, task: str) -> str:
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI()
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            text={"format": {"type": "json_schema", "name": "b4_judge_result", "strict": True, "schema": _schema_for_request(task, prompt)}},
            max_output_tokens=self.max_output_tokens,
        )
        try:
            text = _response_text(response)
        except ValueError as exc:
            # Preserve safe response diagnostics before the no-text failure is
            # surfaced; the exact error message/class is unchanged.
            self.last_diagnostics = openai_response_diagnostics(
                response, configured_max_output_tokens=self.max_output_tokens
            )
            if not isinstance(getattr(exc, "diagnostics", None), dict):
                exc.diagnostics = self.last_diagnostics
            raise
        usage = (
            response.get("usage")
            if isinstance(response, dict)
            else getattr(response, "usage", None)
        )
        details = (
            usage.get("output_tokens_details")
            if isinstance(usage, dict)
            else getattr(usage, "output_tokens_details", None)
            if usage is not None
            else None
        )
        raw_reasoning = (
            details.get("reasoning_tokens")
            if isinstance(details, dict)
            else getattr(details, "reasoning_tokens", None)
            if details is not None
            else None
        )
        self.last_usage_details = {
            "reasoning_tokens": (
                int(raw_reasoning) if raw_reasoning is not None else None
            )
        }
        self.last_diagnostics = openai_response_diagnostics(
            response, raw_text=text, configured_max_output_tokens=self.max_output_tokens
        )
        return text


class AnthropicJudgeAdapter:
    provider_name = "anthropic"

    def __init__(self, model: str, client: Any | None = None, max_output_tokens: int = 1024):
        self.model = model
        self.client = client
        self.max_output_tokens = max_output_tokens
        self.last_diagnostics: dict[str, Any] | None = None
        self.last_usage_details: dict[str, int | None] | None = None

    def call(self, prompt: str, task: str) -> str:
        if self.client is None:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": _schema_for_request(task, prompt)}},
        )
        try:
            text = _response_text(response)
        except ValueError as exc:
            # Preserve safe response diagnostics before the no-text failure is
            # surfaced; the exact error message/class is unchanged.
            self.last_diagnostics = judge_response_diagnostics(response)
            if not isinstance(getattr(exc, "diagnostics", None), dict):
                exc.diagnostics = self.last_diagnostics
            raise
        usage = (
            response.get("usage")
            if isinstance(response, dict)
            else getattr(response, "usage", None)
        )
        details = (
            usage.get("output_tokens_details")
            if isinstance(usage, dict)
            else getattr(usage, "output_tokens_details", None)
            if usage is not None
            else None
        )
        raw_thinking = (
            details.get("thinking_tokens")
            if isinstance(details, dict)
            else getattr(details, "thinking_tokens", None)
            if details is not None
            else None
        )
        self.last_usage_details = {
            "thinking_tokens": (
                int(raw_thinking) if raw_thinking is not None else None
            )
        }
        self.last_diagnostics = judge_response_diagnostics(response, raw_text=text)
        return text


def _gemini_schema(task: str, prompt: str = "") -> dict[str, Any]:
    """Convert the shared strict schema to Gemini's established schema form."""
    schema = copy.deepcopy(_schema_for_request(task, prompt))

    def convert(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("type"), str):
                value["type"] = value["type"].upper()
            # Gemini's SDK accepts this JSON-Schema keyword locally but the
            # provider endpoint rejects its serialized additional_properties
            # field. Exact-field enforcement remains authoritative in the
            # shared downstream parser.
            value.pop("additionalProperties", None)
            for child in value.values():
                convert(child)
        elif isinstance(value, list):
            for child in value:
                convert(child)

    convert(schema)
    return schema


def _gemini_response_text(response: Any) -> str:
    text = response.get("text") if isinstance(response, dict) else getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text
    candidates = response.get("candidates") if isinstance(response, dict) else getattr(response, "candidates", None)
    parts: list[str] = []
    if isinstance(candidates, list) and candidates:
        candidate = candidates[0]
        content = candidate.get("content") if isinstance(candidate, dict) else getattr(candidate, "content", None)
        blocks = content.get("parts") if isinstance(content, dict) else getattr(content, "parts", None)
        if isinstance(blocks, list):
            for block in blocks:
                value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                if isinstance(value, str):
                    parts.append(value)
    if not parts:
        raise ValueError("Gemini response did not contain output text")
    return "".join(parts)


class GeminiJudgeAdapter:
    provider_name = "gemini"

    def __init__(self, model: str, client: Any | None = None, max_output_tokens: int = 1024):
        self.model = model
        self.client = client
        self.max_output_tokens = max_output_tokens
        self.last_metadata: dict[str, Any] | None = None

    def call(self, prompt: str, task: str) -> str:
        if self.client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is required for gemini judge execution")
            from google import genai
            self.client = genai.Client(api_key=api_key)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": 0,
                "max_output_tokens": self.max_output_tokens,
                "response_mime_type": "application/json",
                "response_schema": _gemini_schema(task, prompt),
                "thinking_config": {"thinking_level": "LOW"},
            },
        )
        usage = response.get("usage_metadata") if isinstance(response, dict) else getattr(response, "usage_metadata", None)
        candidates = response.get("candidates") if isinstance(response, dict) else getattr(response, "candidates", None)
        candidate = candidates[0] if isinstance(candidates, list) and candidates else None
        finish_reason = (
            candidate.get("finish_reason") if isinstance(candidate, dict)
            else getattr(candidate, "finish_reason", None) if candidate is not None else None
        )
        def field(value: Any, name: str) -> Any:
            return value.get(name) if isinstance(value, dict) else getattr(value, name, None) if value is not None else None
        self.last_metadata = {
            "finish_reason": str(finish_reason) if finish_reason is not None else None,
            "input_tokens": field(usage, "prompt_token_count") or field(usage, "input_tokens"),
            "output_tokens": field(usage, "candidates_token_count") or field(usage, "output_tokens"),
            "thoughts_token_count": field(usage, "thoughts_token_count"),
            "total_tokens": field(usage, "total_token_count") or field(usage, "total_tokens"),
            "configured_max_output_tokens": self.max_output_tokens,
        }
        return _gemini_response_text(response)


def _chat_response_text(response: Any) -> str:
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise ValueError("provider response contained no choices")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if not isinstance(content, str) or not content:
        raise ValueError("provider response contained no message content")
    return content


class OpenAICompatibleJudgeAdapter:
    """Blinded JSON-mode judge transport for established compatible APIs."""

    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        api_key_env: str,
        base_url: str,
        client: Any | None = None,
        max_output_tokens: int = 1024,
        temperature: float,
        top_p: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        self.provider_name = provider_name
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.client = client
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.extra_body = copy.deepcopy(extra_body)
        self.last_usage: dict[str, Any] | None = None

    def call(self, prompt: str, task: str) -> str:
        # ``task`` is deliberately used only by the shared parser after this
        # transport returns. Both providers receive the exact frozen prompt.
        del task
        request_prompt = provider_request_prompt(self.provider_name, prompt)
        if self.client is None:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"{self.api_key_env} is required for {self.provider_name} judge execution")
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": request_prompt}],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if self.top_p is not None:
            request["top_p"] = self.top_p
        if self.extra_body is not None:
            request["extra_body"] = copy.deepcopy(self.extra_body)
        response = self.client.chat.completions.create(**request)
        usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
        if usage is None:
            self.last_usage = None
        elif isinstance(usage, dict):
            self.last_usage = {
                key: usage.get(key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if usage.get(key) is not None
            }
        else:
            self.last_usage = {
                key: getattr(usage, key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if getattr(usage, key, None) is not None
            }
        return _chat_response_text(response)


class DeepSeekJudgeAdapter(OpenAICompatibleJudgeAdapter):
    def __init__(self, model: str, client: Any | None = None, max_output_tokens: int = 1024, base_url: str | None = None):
        super().__init__(
            provider_name="deepseek",
            model=model,
            api_key_env="DEEPSEEK_API_KEY",
            base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            client=client,
            max_output_tokens=max_output_tokens,
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
        )


class KimiJudgeAdapter(OpenAICompatibleJudgeAdapter):
    def __init__(self, model: str, client: Any | None = None, max_output_tokens: int = 1024, base_url: str | None = None):
        super().__init__(
            provider_name="kimi",
            model=model,
            api_key_env="KIMI_API_KEY",
            # This is the endpoint recorded by the successful frozen run.
            base_url=base_url or os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
            client=client,
            max_output_tokens=max_output_tokens,
            temperature=0.6,
            top_p=0.95,
            extra_body={"thinking": {"type": "disabled"}},
        )


def make_adapter(provider: str, model: str, max_output_tokens: int = 1024) -> Any:
    if provider == "openai":
        return OpenAIJudgeAdapter(model, max_output_tokens=max_output_tokens)
    if provider == "anthropic":
        return AnthropicJudgeAdapter(model, max_output_tokens=max_output_tokens)
    if provider == "deepseek":
        return DeepSeekJudgeAdapter(model, max_output_tokens=max_output_tokens)
    if provider == "kimi":
        return KimiJudgeAdapter(model, max_output_tokens=max_output_tokens)
    if provider == "gemini":
        return GeminiJudgeAdapter(model, max_output_tokens=max_output_tokens)
    raise ValueError(f"unsupported judge provider: {provider}")


def classify_input_limits(slots: list[dict[str, Any]], limits: dict[str, int | None]) -> list[dict[str, Any]]:
    classified = []
    for slot in slots:
        measured = estimate_input_tokens(slot["prompt"], slot["judge_model"])
        limit = limits.get(slot["judge_slot"])
        classified.append({
            **slot,
            **measured,
            "configured_max_input_tokens": limit,
            "input_limit_status": "unverified" if limit is None else ("ready" if measured["estimated_tokens"] <= limit else "oversized"),
        })
    return classified


def validate_paid_input_limits(classified: list[dict[str, Any]], limits: dict[str, int | None]) -> None:
    paired_judge_gate = bool({"A", "B"} & set(limits))
    required_slots = {"A", "B"} if paired_judge_gate else {str(row["judge_slot"]) for row in classified}
    missing_slots = sorted(slot for slot in required_slots if limits.get(slot) is None)
    if missing_slots:
        requirement = "both judges" if paired_judge_gate else "judge slots"
        raise ValueError(
            f"paid execution requires verified input limits for {requirement}: "
            + ", ".join(missing_slots)
        )
    blockers = [row for row in classified if row.get("input_limit_status") != "ready"]
    if blockers:
        raise ValueError(f"paid execution blocked by {len(blockers)} oversized or unverified request slots")


def execute_slots(
    slots: list[dict[str, Any]],
    *,
    ledger: DurableLedger,
    adapters: dict[str, Any] | None = None,
    max_attempts: int = 3,
    max_output_tokens: int = 1024,
    retry_delay: Callable[[int], None] = lambda attempt: time.sleep(min(2 ** (attempt - 1), 8)),
) -> dict[str, Any]:
    """Execute preflight-approved slots; callers must enforce flags/limits first."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    plan = resume_plan(slots, ledger, max_attempts=max_attempts)
    if plan["blocked"]:
        raise LedgerConflictError(f"{len(plan['blocked'])} request slots require manual resolution")
    adapters = adapters or {}
    existing_slot_ids = {str(event.get("request_slot_id")) for event in plan["events"]}
    completed_now = 0
    for slot in plan["pending"]:
        adapter = adapters.get(slot["judge_slot"])
        if adapter is None:
            adapter = make_adapter(slot["judge_provider"], slot["judge_model"], max_output_tokens=max_output_tokens)
            adapters[slot["judge_slot"]] = adapter
        base = {key: slot[key] for key in ("request_slot_id", "blinded_request_id", "judge_slot", "judge_provider", "judge_model", "request_hash", "task")}
        if slot["request_slot_id"] not in existing_slot_ids:
            ledger.append({**base, "state": "planned", "attempt": 0, "timestamp_ns": time.time_ns()})
        attempt = int(slot["next_attempt"])
        slot_max_attempts = attempt if slot.get("manual_retry_max_new_calls") == 1 else max_attempts
        while attempt <= slot_max_attempts:
            ledger.append({**base, "state": "started", "attempt": attempt, "timestamp_ns": time.time_ns()})
            try:
                raw = adapter.call(slot["prompt"], slot["task"])
            except Exception as exc:
                retryable, error_class, message = classify_provider_error(exc)
                state = "retryable_error" if retryable and attempt < slot_max_attempts else ("permanently_failed" if retryable else "provider_error")
                ledger.append({**base, "state": state, "attempt": attempt, "error_class": error_class, "retryable": retryable, "error_message": message, "timestamp_ns": time.time_ns()})
                if state == "retryable_error":
                    retry_delay(attempt)
                    attempt += 1
                    continue
                break
            response_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            try:
                parsed = _parse_judge_output(raw, slot["task"])
            except ValueError as exc:
                ledger.append({**base, "state": "parse_failed", "attempt": attempt, "retryable": False, "error_class": type(exc).__name__, "error_message": str(exc)[:2000], "response_hash": response_hash, "raw_response": raw, "timestamp_ns": time.time_ns()})
                break
            ledger.append({**base, "state": "completed", "attempt": attempt, "retryable": False, "parsed_result": parsed, "response_hash": response_hash, "raw_response": raw, "timestamp_ns": time.time_ns()})
            completed_now += 1
            break
    final = resume_plan(slots, ledger, max_attempts=max_attempts)
    return {"completed_now": completed_now, "completed_total": len(final["completed"]), "pending": len(final["pending"]), "blocked": len(final["blocked"])}
