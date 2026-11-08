#!/usr/bin/env python3
"""Dry-run or execute provider-isolated formal 300 validation phases."""

from __future__ import annotations

import argparse, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b4_calibration_execution import (
    AnthropicJudgeAdapter, DurableLedger, GeminiJudgeAdapter, OpenAIJudgeAdapter,
    estimate_input_tokens, judge_response_diagnostics, prompt_hash, provider_request_prompt,
)
from scripts.prepare_cheap_judge_pilot import read_jsonl
from scripts.prepare_formal_300_validation import (
    EVALUATOR, LIMITS, MODELS, PROTOCOL, formal_request_id, sha256_value,
)
from scripts.run_cheap_judge_pilot import append_jsonl, parse_result


GEMINI_MEMORA_RECOVERY = "formal-300-validation-v1-gemini-batch-schema-recovery-1"
GEMINI_MEMORA_FAILURE = "Memora result must contain only decisions"
GEMINI_MEMORA_ATTEMPT3_RECOVERY = "formal-300-validation-v1-gemini-attempt3-recovery-1"
GEMINI_MEMORA_ATTEMPT4_RECOVERY = "formal-300-validation-v1-gemini-attempt4-schema-compatibility-recovery-1"
ANTHROPIC_NO_TEXT_RECOVERY = "formal-300-validation-v1-anthropic-no-text-transport-recovery-1"
ANTHROPIC_NO_TEXT_FAILURE_CLASS = "no_text"
ANTHROPIC_NO_TEXT_ERROR_CLASS = "ValueError"
ANTHROPIC_NO_TEXT_ERROR_MESSAGE = "provider response contained no text"
ANTHROPIC_ATTEMPT3_RECOVERY = "formal-300-validation-v1-anthropic-attempt3-diagnostic-retry-1"
# Frozen recovery-protocol target: this is the exact Formal-300 slot that
# consumed Anthropic attempt 3. It is a source-derived slot fingerprint and is
# intentionally NOT replaced by a synthetic value, because the attempt-3
# authorization validator (validate_attempt3_authorization) compares the frozen
# recovery manifest and the authorized request ids against it and raises on any
# drift. Changing it would silently change which frozen experiment is resumed.
ANTHROPIC_ATTEMPT3_REQUEST_ID = os.environ.get(
    "GOVMEM_FORMAL_ANTHROPIC_ATTEMPT3_REQUEST_ID", ""
).strip()
ANTHROPIC_ATTEMPT3_JUDGE_SLOT = "lme_primary_anthropic"
ANTHROPIC_ATTEMPT2_MALFORMED_JSON_MESSAGE = "malformed JSON: Unterminated string starting at: line 1 column 58 (char 57)"
ANTHROPIC_ATTEMPT4_RECOVERY = "formal-300-validation-v1-anthropic-attempt4-token-budget-retry-1"
ANTHROPIC_ATTEMPT3_OBSERVED_OUTPUT_TOKENS = 160
ANTHROPIC_TOKEN_LIMIT_RECOVERY = "formal-300-validation-v1-anthropic-token-limit-recovery-1"
ANTHROPIC_TOKEN_LIMIT_RECOVERY_OUTPUT_TOKENS = 2048
ANTHROPIC_TOKEN_LIMIT_OBSERVED_OUTPUT_TOKENS = 512
ANTHROPIC_TOKEN_LIMIT_EXPECTED_ELIGIBLE = 39
ANTHROPIC_TOKEN_LIMIT_REASON = "token_limit_truncation_512"
ANTHROPIC_TOKEN_RECOVERY_512_TO_2048 = "512-to-2048"
ANTHROPIC_TOKEN_RECOVERY_2048_TO_4096 = "2048-to-4096"
ANTHROPIC_TOKEN_RECOVERY_2236_TO_4096 = "2236-to-4096"
ANTHROPIC_TOKEN_RECOVERY_4096_TO_8192 = "4096-to-8192"
ANTHROPIC_TOKEN_RECOVERY_CLASSES = (
    ANTHROPIC_TOKEN_RECOVERY_512_TO_2048, ANTHROPIC_TOKEN_RECOVERY_2048_TO_4096,
    ANTHROPIC_TOKEN_RECOVERY_2236_TO_4096, ANTHROPIC_TOKEN_RECOVERY_4096_TO_8192,
)
ANTHROPIC_TOKEN_RECOVERY_4096_PROTOCOL = "formal-300-validation-v1-anthropic-2048-to-4096-recovery-1"
ANTHROPIC_TOKEN_RECOVERY_4096_OUTPUT_TOKENS = 4096
ANTHROPIC_TOKEN_RECOVERY_2048_OBSERVED_OUTPUT_TOKENS = 2048
ANTHROPIC_TOKEN_RECOVERY_512_POPULATION = 39
ANTHROPIC_TOKEN_RECOVERY_4096_POPULATION = 3
ANTHROPIC_TOKEN_LIMIT_REASON_2048 = "token_limit_truncation_2048"
ANTHROPIC_ESCALATION_JUDGE_SLOT = "memora_escalation_anthropic"
ANTHROPIC_ESCALATION_RECOVERY_PROTOCOL = "formal-300-validation-v1-anthropic-memora-escalation-token-recovery-1"
ANTHROPIC_ESCALATION_OUTPUT_TOKENS = 4096
ANTHROPIC_ESCALATION_OBSERVED_OUTPUT_TOKENS = 2236
ANTHROPIC_ESCALATION_POPULATION = 1
ANTHROPIC_ESCALATION_REASON = "anthropic_memora_escalation_token_limit_2236_to_4096"
ANTHROPIC_ESCALATION_8192_PROTOCOL = "formal-300-validation-v1-anthropic-memora-escalation-4096-to-8192-recovery-1"
ANTHROPIC_ESCALATION_8192_OUTPUT_TOKENS = 8192
ANTHROPIC_ESCALATION_4096_OBSERVED_OUTPUT_TOKENS = 4096
ANTHROPIC_ESCALATION_8192_POPULATION = 1
ANTHROPIC_ESCALATION_8192_REASON = "anthropic_memora_escalation_token_limit_4096_to_8192"
HUMAN_ADJUDICATION_PROTOCOL = "formal-300-validation-v1-human-adjudication"
HUMAN_ADJUDICATION_SCHEMA_VERSION = "formal-300-human-adjudication-v1"
HUMAN_ADJUDICATION_RESOLUTION_METHOD = "human_adjudication"
HUMAN_ADJUDICATION_REASON = "anthropic_escalation_thinking_only_max_tokens"
HUMAN_ADJUDICATION_OBSERVED_OUTPUT_TOKENS = 8192
HUMAN_ADJUDICATION_EXPECTED = 1
HUMAN_ADJUDICATION_ALLOWED_DECISIONS = ("pass", "fail")
HUMAN_ADJUDICATION_ALLOWED_CONFIDENCE = ("high", "medium", "low")
HUMAN_ADJUDICATION_DIRNAME = "human_adjudication"
HUMAN_ADJUDICATION_DECISIONS_FILENAME = "human_adjudication_decisions.jsonl"
OPENAI_NO_TEXT_RECOVERY = "formal-300-validation-v1-openai-no-text-transport-recovery-1"
OPENAI_NO_TEXT_RECOVERY_CLASS = "436-to-1024"
OPENAI_NO_TEXT_FAILURE_CLASS = "openai_no_text_436_to_1024"
OPENAI_NO_TEXT_ERROR_MESSAGE = "provider response contained no text"
OPENAI_NO_TEXT_JUDGE_SLOT = "memora_secondary_openai"
# Frozen recovery-protocol target: the single OpenAI no-text slot. As above it
# is a source-derived slot fingerprint that the frozen recovery selectors and
# authorization validators match exactly and fail closed on drift.
OPENAI_NO_TEXT_RECOVERY_REQUEST_ID = os.environ.get(
    "GOVMEM_FORMAL_OPENAI_NO_TEXT_REQUEST_ID", ""
).strip()
OPENAI_NO_TEXT_PLANNED_OUTPUT_TOKENS = 436
OPENAI_NO_TEXT_RECOVERY_OUTPUT_TOKENS = 1024
OPENAI_NO_TEXT_RECOVERY_POPULATION = 1
OPENAI_TOKEN_RECOVERY_2236_TO_4096 = "2236-to-4096"
OPENAI_TOKEN_RECOVERY_160_TO_512 = "160-to-512"
OPENAI_TOKEN_RECOVERY_512_TO_1024 = "512-to-1024"
OPENAI_TOKEN_RECOVERY_CLASSES = (
    OPENAI_NO_TEXT_RECOVERY_CLASS, OPENAI_TOKEN_RECOVERY_2236_TO_4096,
    OPENAI_TOKEN_RECOVERY_160_TO_512, OPENAI_TOKEN_RECOVERY_512_TO_1024,
)
OPENAI_TOKEN_RECOVERY_4096_PROTOCOL = "formal-300-validation-v1-openai-token-limit-recovery-1"
OPENAI_TOKEN_RECOVERY_4096_OUTPUT_TOKENS = 4096
OPENAI_TOKEN_RECOVERY_2236_OBSERVED_OUTPUT_TOKENS = 2236
OPENAI_TOKEN_RECOVERY_2236_POPULATION = 2
OPENAI_TOKEN_LIMIT_REASON_2236 = "openai_token_limit_truncation_2236_to_4096"
OPENAI_TOKEN_RECOVERY_512_PROTOCOL = "formal-300-validation-v1-openai-lme-token-limit-recovery-1"
OPENAI_TOKEN_RECOVERY_512_OUTPUT_TOKENS = 512
OPENAI_TOKEN_RECOVERY_160_OBSERVED_OUTPUT_TOKENS = 160
OPENAI_TOKEN_RECOVERY_160_POPULATION = 1
OPENAI_TOKEN_RECOVERY_160_JUDGE_SLOT = "lme_primary_openai"
OPENAI_TOKEN_LIMIT_REASON_160 = "openai_token_limit_truncation_160_to_512"
OPENAI_TOKEN_RECOVERY_1024_PROTOCOL = "formal-300-validation-v1-openai-lme-512-to-1024-recovery-1"
OPENAI_TOKEN_RECOVERY_1024_OUTPUT_TOKENS = 1024
OPENAI_TOKEN_RECOVERY_512_OBSERVED_OUTPUT_TOKENS = 512
OPENAI_TOKEN_RECOVERY_512_TO_1024_POPULATION = 1
OPENAI_TOKEN_LIMIT_REASON_512 = "openai_token_limit_truncation_512_to_1024"
# Transport-only reliability floor: the normal Anthropic formal-300 primary run
# hit stop_reason=max_tokens at 512 output tokens (39/39 failures), with 24/39
# exhausting the budget during thinking before emitting any text.  Future
# Anthropic judging therefore requests at least this many output tokens.
FORMAL_300_ANTHROPIC_MIN_OUTPUT_TOKENS = 2048
# Transport-only reliability floor: an observed OpenAI gpt-5.6-sol LME primary
# failure hit status=incomplete / incomplete_reason=max_output_tokens at a
# 160-token budget and succeeded when retried at 512, so future pending LME
# primary execution requests at least this many output tokens.  This is an
# execution-time floor only; the frozen plan keeps its stored 160.
FORMAL_300_OPENAI_LME_PRIMARY_MIN_OUTPUT_TOKENS = 512
# Transport-only reliability floor: an observed Anthropic Memora escalation
# failure exhausted first 2236 then a 4096 recovery budget entirely during
# thinking (no text), so future pending Memora escalation execution requests at
# least this many output tokens.
FORMAL_300_ANTHROPIC_ESCALATION_MIN_OUTPUT_TOKENS = 8192
FORMAL_LEDGER_IDENTITY_KEYS = (
    "formal_validation_request_id", "pilot_block_id", "observation_id", "source", "judge_slot",
    "provider", "model", "request_hash", "payload_hash", "protocol_version", "evaluator_methodology",
)
GEMINI_MEMORA_TRANSPORT_SCHEMA_VERSION = "formal-memora-batch-gemini-compatible-v3"
GEMINI_MEMORA_OUTPUT_BUDGET_VERSION = "formal-memora-attempt3-output-budget-v1"
ATTEMPT3_FROZEN_MANIFEST_HASH = "7d47e738693a7fa420baba2c6c1212ba94ebb721a16ea9d32455b424208c7dcf"
# Frozen recovery-protocol target: the attempt-4 schema-compatibility smoke
# slot. Selectors require exactly this single row and raise otherwise, so the
# frozen attempt-4 semantics cannot silently change if it is altered.
ATTEMPT4_SMOKE_REQUEST_ID = os.environ.get(
    "GOVMEM_FORMAL_ATTEMPT4_SMOKE_REQUEST_ID", ""
).strip()
ATTEMPT3_CLASSES = ("invalid_argument", "malformed_json")


def formal_output_budget(
    provider: str,
    planned_output_tokens: int,
    *,
    judge_slot: str | None = None,
    source: str | None = None,
    phase: str | None = None,
) -> int:
    """Effective formal-300 output budget (transport reliability floors).

    The Anthropic floor applies to all Anthropic judging.  The OpenAI floor is
    scoped precisely to OpenAI LongMemEval primary judging; other OpenAI slots
    (Memora secondary) and Gemini are returned unchanged.  Floors never change
    the prompt, schema, model, or the frozen plan's stored planned budget.
    """
    planned = int(planned_output_tokens)
    if provider == "anthropic":
        if (
            judge_slot == ANTHROPIC_ESCALATION_JUDGE_SLOT
            and source == "memora"
            and phase == "escalation"
        ):
            return max(planned, FORMAL_300_ANTHROPIC_ESCALATION_MIN_OUTPUT_TOKENS)
        return max(planned, FORMAL_300_ANTHROPIC_MIN_OUTPUT_TOKENS)
    if (
        provider == "openai"
        and judge_slot == OPENAI_TOKEN_RECOVERY_160_JUDGE_SLOT
        and source == "longmemeval"
        and phase == "primary"
    ):
        return max(planned, FORMAL_300_OPENAI_LME_PRIMARY_MIN_OUTPUT_TOKENS)
    return planned


def _token_recovery_config(recovery_class: str) -> dict[str, Any]:
    """Disjoint Anthropic token-limit recovery classes and their budgets."""
    if recovery_class == ANTHROPIC_TOKEN_RECOVERY_512_TO_2048:
        return {
            "recovery_class": recovery_class, "observed_output_tokens": ANTHROPIC_TOKEN_LIMIT_OBSERVED_OUTPUT_TOKENS,
            "output_budget": ANTHROPIC_TOKEN_LIMIT_RECOVERY_OUTPUT_TOKENS, "protocol": ANTHROPIC_TOKEN_LIMIT_RECOVERY,
            "population": ANTHROPIC_TOKEN_RECOVERY_512_POPULATION, "reason": ANTHROPIC_TOKEN_LIMIT_REASON,
            "require_zero_text": False, "judge_slot": ANTHROPIC_ATTEMPT3_JUDGE_SLOT,
            "source": "longmemeval", "phase": "primary",
        }
    if recovery_class == ANTHROPIC_TOKEN_RECOVERY_2048_TO_4096:
        return {
            "recovery_class": recovery_class, "observed_output_tokens": ANTHROPIC_TOKEN_RECOVERY_2048_OBSERVED_OUTPUT_TOKENS,
            "output_budget": ANTHROPIC_TOKEN_RECOVERY_4096_OUTPUT_TOKENS, "protocol": ANTHROPIC_TOKEN_RECOVERY_4096_PROTOCOL,
            "population": ANTHROPIC_TOKEN_RECOVERY_4096_POPULATION, "reason": ANTHROPIC_TOKEN_LIMIT_REASON_2048,
            "require_zero_text": True, "judge_slot": ANTHROPIC_ATTEMPT3_JUDGE_SLOT,
            "source": "longmemeval", "phase": "primary",
        }
    if recovery_class == ANTHROPIC_TOKEN_RECOVERY_2236_TO_4096:
        return {
            "recovery_class": recovery_class, "observed_output_tokens": ANTHROPIC_ESCALATION_OBSERVED_OUTPUT_TOKENS,
            "output_budget": ANTHROPIC_ESCALATION_OUTPUT_TOKENS, "protocol": ANTHROPIC_ESCALATION_RECOVERY_PROTOCOL,
            "population": ANTHROPIC_ESCALATION_POPULATION, "reason": ANTHROPIC_ESCALATION_REASON,
            "require_zero_text": True, "require_no_raw_text": True, "require_thinking": True,
            "required_attempt": 1, "error_message": ANTHROPIC_NO_TEXT_ERROR_MESSAGE,
            "judge_slot": ANTHROPIC_ESCALATION_JUDGE_SLOT, "source": "memora", "phase": "escalation",
        }
    if recovery_class == ANTHROPIC_TOKEN_RECOVERY_4096_TO_8192:
        return {
            "recovery_class": recovery_class, "observed_output_tokens": ANTHROPIC_ESCALATION_4096_OBSERVED_OUTPUT_TOKENS,
            "output_budget": ANTHROPIC_ESCALATION_8192_OUTPUT_TOKENS, "protocol": ANTHROPIC_ESCALATION_8192_PROTOCOL,
            "population": ANTHROPIC_ESCALATION_8192_POPULATION, "reason": ANTHROPIC_ESCALATION_8192_REASON,
            "require_zero_text": True, "require_no_raw_text": True, "require_thinking": True,
            "required_attempt": 2, "previous_recovery_class": ANTHROPIC_TOKEN_RECOVERY_2236_TO_4096,
            "judge_slot": ANTHROPIC_ESCALATION_JUDGE_SLOT, "source": "memora", "phase": "escalation",
        }
    raise RuntimeError(f"unknown Anthropic token recovery class: {recovery_class}")


def attempt3_output_budget(criterion_count: int) -> int:
    """Frozen transport-only budget: min(6144, max(1024, 256 + 80*n)).

    The 80-token allowance exceeds the observed successful maximum of 69.67
    estimated output tokens per criterion.  The 256-token framing allowance,
    1024 floor, and repository-established 6144 Gemini Flash repair ceiling
    are deterministic and do not alter the semantic contract.
    """
    if criterion_count < 1:
        raise ValueError("criterion count must be positive")
    return min(6144, max(1024, 256 + 80 * criterion_count))


def _formal_gemini_memora_plan(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    eligible = {
        str(row["formal_validation_request_id"]): row
        for row in rows
        if row.get("provider") == "gemini"
        and row.get("model") == MODELS["gemini"]
        and row.get("source") == "memora"
        and row.get("phase") == "primary"
        and row.get("judge_slot") == "memora_primary_gemini"
    }
    if len(eligible) != 450:
        raise RuntimeError(f"expected 450 frozen Gemini/Memora primary slots, found {len(eligible)}")
    return eligible


def _failure_class(event: dict[str, Any]) -> str | None:
    error_class = str(event.get("error_class", ""))
    message = str(event.get("error_message", ""))
    if error_class == "ClientError" and "INVALID_ARGUMENT" in message:
        return "invalid_argument"
    if error_class == "ValueError" and "malformed JSON" in message:
        return "malformed_json"
    return None


def derive_attempt3_candidates(
    rows: list[dict[str, Any]],
    ledger: DurableLedger,
    results: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive only latest-failed, fully consumed attempt-2 requests."""
    plan = _formal_gemini_memora_plan(rows)
    result_ids = [str(row.get("formal_validation_request_id")) for row in results]
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("duplicate durable result")
    result_set = set(result_ids)
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identity_fields = (
        "observation_id", "provider", "model", "request_hash", "payload_hash",
        "protocol_version", "evaluator_methodology",
    )
    for event in ledger.read():
        rid = str(event.get("formal_validation_request_id"))
        if rid not in plan:
            continue
        row = plan[rid]
        for key in identity_fields:
            if str(event.get(key)) != str(row.get(key)):
                raise RuntimeError(f"ledger identity drift: {rid}")
        histories[rid].append(event)

    candidates: list[dict[str, Any]] = []
    for rid, row in sorted(plan.items()):
        history = histories.get(rid, [])
        if rid in result_set or any(event.get("state") == "completed" for event in history):
            continue
        if any(event.get("attempt") == 3 or event.get("recovery_attempt") == 3 for event in history):
            continue
        authorizations = [
            event for event in history
            if event.get("state") == "recovery_authorized"
            and event.get("attempt") == 1 and event.get("recovery_attempt") == 2
        ]
        started_two = [event for event in history if event.get("state") == "started" and event.get("attempt") == 2]
        failed_two = [event for event in history if event.get("state") == "failed" and event.get("attempt") == 2]
        if len(authorizations) != 1 or len(started_two) != 1 or len(failed_two) != 1:
            continue
        if not history or history[-1] is not failed_two[0]:
            continue
        failure_class = _failure_class(failed_two[0])
        if failure_class is None:
            continue
        payload = payloads.get(str(row["observation_id"]))
        if payload is None:
            raise RuntimeError(f"missing semantic payload: {rid}")
        prompt_value = json.loads(str(payload["prompt"]))
        criteria = prompt_value.get("criteria") if isinstance(prompt_value, dict) else None
        if not isinstance(criteria, list) or not criteria:
            raise RuntimeError(f"invalid Memora criteria payload: {rid}")
        criterion_count = len(criteria)
        candidates.append({
            **row,
            "failure_class": failure_class,
            "prior_attempt_count": 2,
            "recovery_attempt": 3,
            "recovery_protocol_version": GEMINI_MEMORA_ATTEMPT3_RECOVERY,
            "transport_schema_version": GEMINI_MEMORA_TRANSPORT_SCHEMA_VERSION,
            "output_budget_policy_version": GEMINI_MEMORA_OUTPUT_BUDGET_VERSION,
            "configured_max_output_tokens": attempt3_output_budget(criterion_count),
            "criterion_count": criterion_count,
        })
    return candidates


def attempt3_public_record(row: dict[str, Any]) -> dict[str, Any]:
    """Return recovery metadata that contains no benchmark or answer text."""
    keys = (
        "formal_validation_request_id", "observation_id", "pilot_block_id", "condition_id",
        "source", "judge_slot", "provider", "model", "request_hash", "payload_hash",
        "protocol_version", "evaluator_methodology", "failure_class", "prior_attempt_count",
        "recovery_attempt", "recovery_protocol_version", "transport_schema_version",
        "output_budget_policy_version", "configured_max_output_tokens", "criterion_count",
    )
    return {key: row.get(key) for key in keys}


def attempt3_manifest_hash(rows: list[dict[str, Any]]) -> str:
    return sha256_value([attempt3_public_record(row) for row in rows])


def load_attempt3_manifest(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    ids = [str(row.get("formal_validation_request_id")) for row in rows]
    classes = Counter(str(row.get("failure_class")) for row in rows)
    if len(rows) != 51 or len(ids) != len(set(ids)):
        raise RuntimeError("frozen attempt-3 manifest must contain 51 unique requests")
    if classes != Counter({"invalid_argument": 45, "malformed_json": 6}):
        raise RuntimeError(f"frozen attempt-3 manifest class drift: {dict(classes)}")
    actual_hash = attempt3_manifest_hash(rows)
    if actual_hash != ATTEMPT3_FROZEN_MANIFEST_HASH:
        raise RuntimeError(f"frozen attempt-3 manifest hash drift: {actual_hash}")
    return rows


def validate_progressive_attempt3_state(
    frozen_rows: list[dict[str, Any]],
    current_candidates: list[dict[str, Any]],
    ledger: DurableLedger,
    results: list[dict[str, Any]],
) -> dict[str, int]:
    """Partition the immutable superset into success, failure, and unattempted."""
    frozen = {str(row["formal_validation_request_id"]): attempt3_public_record(row) for row in frozen_rows}
    if len(frozen) != 51:
        raise RuntimeError("frozen attempt-3 superset identity count drift")
    current = {
        str(row["formal_validation_request_id"]): attempt3_public_record(row)
        for row in current_candidates
    }
    if len(current) != len(current_candidates):
        raise RuntimeError("duplicate current attempt-3 eligible identity")
    new_ids = set(current) - set(frozen)
    if new_ids:
        raise RuntimeError(f"new request outside frozen attempt-3 superset: {sorted(new_ids)[0]}")
    transport_only_fields = {"transport_schema_version"}
    def comparable(row: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in row.items() if key not in transport_only_fields}
    for rid, row in current.items():
        if comparable(row) != comparable(frozen[rid]):
            raise RuntimeError(f"current attempt-3 identity or failure-class drift: {rid}")
        if row.get("transport_schema_version") != GEMINI_MEMORA_TRANSPORT_SCHEMA_VERSION:
            raise RuntimeError(f"current attempt-3 transport schema is not repaired: {rid}")

    result_ids = [str(row.get("formal_validation_request_id")) for row in results]
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("duplicate durable result")
    result_by_id = {str(row["formal_validation_request_id"]): row for row in results}
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in ledger.read():
        rid = str(event.get("formal_validation_request_id"))
        if rid in frozen:
            histories[rid].append(event)

    identity_fields = (
        "formal_validation_request_id", "observation_id", "provider", "model",
        "request_hash", "payload_hash", "protocol_version", "evaluator_methodology",
    )
    accounting = {
        "frozen_total": 51, "durable_success_direct": 0, "failed_unresolved": 0,
        "resolved_by_later_attempt": 0, "not_yet_attempted": 0,
        "final_durable_success": 0,
        "invalid_argument_frozen": 45, "invalid_argument_durable_success": 0,
        "invalid_argument_failed": 0, "invalid_argument_not_yet_attempted": 0,
        "malformed_json_frozen": 6, "malformed_json_durable_success": 0,
        "malformed_json_failed": 0, "malformed_json_not_yet_attempted": 0,
    }
    for rid, frozen_row in frozen.items():
        history = histories.get(rid, [])
        failed_two = [event for event in history if event.get("state") == "failed" and event.get("attempt") == 2]
        if len(failed_two) != 1 or _failure_class(failed_two[0]) != frozen_row["failure_class"]:
            raise RuntimeError(f"frozen attempt-3 failure-class provenance drift: {rid}")
        for event in history:
            for field in identity_fields:
                if str(event.get(field)) != str(frozen_row.get(field)):
                    raise RuntimeError(f"frozen attempt-3 ledger identity drift: {rid}:{field}")
        authorizations = [event for event in history if event.get("state") == "recovery_authorized" and event.get("recovery_attempt") == 3]
        started = [event for event in history if event.get("state") == "started" and event.get("attempt") == 3]
        terminals = [event for event in history if event.get("state") in {"completed", "failed"} and event.get("attempt") == 3]
        later_events = [
            event for event in history
            if (isinstance(event.get("attempt"), int) and event.get("attempt") > 3)
            or (isinstance(event.get("recovery_attempt"), int) and event.get("recovery_attempt") > 3)
        ]
        result = result_by_id.get(rid)
        failure_class = str(frozen_row["failure_class"])
        if not authorizations and not started and not terminals:
            if rid not in current or result is not None or later_events:
                raise RuntimeError(f"unattempted request missing from attempt-3 eligibility: {rid}")
            accounting["not_yet_attempted"] += 1
            accounting[f"{failure_class}_not_yet_attempted"] += 1
            continue
        if len(authorizations) != 1 or authorizations[0].get("attempt") != 2:
            raise RuntimeError(f"invalid attempt-3 authorization lifecycle: {rid}")
        if len(started) != 1 or len(terminals) != 1:
            raise RuntimeError(f"invalid attempt-3 execution lifecycle: {rid}")
        auth_index, start_index, terminal_index = map(history.index, (authorizations[0], started[0], terminals[0]))
        if not auth_index < start_index < terminal_index:
            raise RuntimeError(f"out-of-order attempt-3 lifecycle: {rid}")
        if rid in current:
            raise RuntimeError(f"consumed attempt-3 request re-entered eligibility: {rid}")
        terminal = terminals[0]
        if terminal.get("state") == "completed":
            if later_events:
                raise RuntimeError(f"completed attempt-3 request has an illegal later retry: {rid}")
            if result is None or result.get("attempt") != 3:
                raise RuntimeError(f"completed attempt-3 request lacks matching durable result: {rid}")
            for record in (result, terminal):
                for field in identity_fields:
                    if str(record.get(field)) != str(frozen_row.get(field)):
                        raise RuntimeError(f"completed attempt-3 identity drift: {rid}:{field}")
            accounting["durable_success_direct"] += 1
            accounting["final_durable_success"] += 1
            accounting[f"{failure_class}_durable_success"] += 1
            continue

        if not later_events:
            if result is not None:
                raise RuntimeError(f"failed attempt-3 request has a durable result without later completion: {rid}")
            accounting["failed_unresolved"] += 1
            accounting[f"{failure_class}_failed"] += 1
            continue

        # Validate attempt 4 independently; its append order may follow the
        # terminal attempt-3 failure without changing attempt 3's outcome.
        if any(
            (isinstance(event.get("attempt"), int) and event.get("attempt") > 4)
            or (isinstance(event.get("recovery_attempt"), int) and event.get("recovery_attempt") > 4)
            for event in later_events
        ):
            raise RuntimeError(f"unsupported recovery attempt after attempt 4: {rid}")
        auth4 = [event for event in history if event.get("state") == "recovery_authorized" and event.get("recovery_attempt") == 4]
        started4 = [event for event in history if event.get("state") == "started" and event.get("attempt") == 4]
        terminal4 = [event for event in history if event.get("state") in {"completed", "failed"} and event.get("attempt") == 4]
        if len(auth4) != 1 or auth4[0].get("attempt") != 3 or len(started4) != 1 or len(terminal4) != 1:
            raise RuntimeError(f"unauthorized or incomplete later retry lifecycle: {rid}")
        indexes = tuple(map(history.index, (terminal, auth4[0], started4[0], terminal4[0])))
        if not indexes[0] < indexes[1] < indexes[2] < indexes[3] or history[-1] is not terminal4[0]:
            raise RuntimeError(f"out-of-order attempt-4 lifecycle: {rid}")
        if terminal4[0].get("state") == "failed":
            if result is not None:
                raise RuntimeError(f"failed attempt-4 request has a durable result: {rid}")
            accounting["failed_unresolved"] += 1
            accounting[f"{failure_class}_failed"] += 1
            continue
        if result is None or result.get("attempt") != 4:
            raise RuntimeError(f"completed attempt-4 request lacks matching durable result: {rid}")
        for record in (result, terminal4[0]):
            for field in identity_fields:
                if str(record.get(field)) != str(frozen_row.get(field)):
                    raise RuntimeError(f"completed later-attempt identity drift: {rid}:{field}")
        accounting["resolved_by_later_attempt"] += 1
        accounting["final_durable_success"] += 1

    classes = Counter(row["failure_class"] for row in current_candidates)
    if classes["invalid_argument"] > 45 or classes["malformed_json"] > 6:
        raise RuntimeError("current attempt-3 class count exceeds frozen population")
    if accounting["not_yet_attempted"] != len(current_candidates):
        raise RuntimeError("attempt-3 unattempted accounting does not match current eligibility")
    return accounting


def validate_attempt3_authorization(
    path: Path,
    frozen_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = attempt3_manifest_hash(frozen_rows)
    if value.get("authorized") is not True:
        raise RuntimeError("attempt-3 recovery authorization is not enabled")
    expected = {
        "protocol_version": PROTOCOL,
        "evaluator_methodology": EVALUATOR,
        "recovery_protocol_version": GEMINI_MEMORA_ATTEMPT3_RECOVERY,
        "recovery_manifest_hash": expected_hash,
        "maximum_authorized_retries_per_request": 1,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"attempt-3 authorization drift: {key}")
    ids = value.get("authorized_request_ids")
    if not isinstance(ids, list) or len(ids) != len(set(map(str, ids))):
        raise RuntimeError("attempt-3 authorization request IDs are invalid")
    candidate_ids = {row["formal_validation_request_id"] for row in frozen_rows}
    authorized_ids = set(map(str, ids))
    if not authorized_ids.issubset(candidate_ids):
        raise RuntimeError("attempt-3 authorization contains an ineligible request")
    if not {row["formal_validation_request_id"] for row in selected}.issubset(authorized_ids):
        raise RuntimeError("selected attempt-3 request is not explicitly authorized")
    return value


def derive_attempt4_candidates(
    rows: list[dict[str, Any]],
    frozen_rows: list[dict[str, Any]],
    ledger: DurableLedger,
    results: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select only the known attempt-3 additional-properties transport failure."""
    unattempted = derive_attempt3_candidates(rows, ledger, results, payloads)
    accounting = validate_progressive_attempt3_state(frozen_rows, unattempted, ledger, results)
    if accounting["failed_unresolved"] not in {0, 1}:
        raise RuntimeError(f"unexpected unresolved attempt-3 failure count: {accounting['failed_unresolved']}")
    plan = _formal_gemini_memora_plan(rows)
    frozen = {str(row["formal_validation_request_id"]): row for row in frozen_rows}
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in ledger.read():
        rid = str(event.get("formal_validation_request_id"))
        if rid in frozen:
            histories[rid].append(event)
    candidates: list[dict[str, Any]] = []
    for rid, history in histories.items():
        failed_three = [event for event in history if event.get("state") == "failed" and event.get("attempt") == 3]
        if not failed_three:
            continue
        if len(failed_three) != 1:
            raise RuntimeError(f"invalid failed attempt-3 lifecycle: {rid}")
        event = failed_three[0]
        message = str(event.get("error_message", ""))
        has_attempt4 = any(item.get("attempt") == 4 or item.get("recovery_attempt") == 4 for item in history)
        if has_attempt4:
            continue
        if not (
            rid == ATTEMPT4_SMOKE_REQUEST_ID
            and event.get("error_class") == "ClientError"
            and "INVALID_ARGUMENT" in message
            and "additional_properties" in message
        ):
            raise RuntimeError(f"attempt-3 failure is not eligible for attempt 4: {rid}")
        if rid in {str(result["formal_validation_request_id"]) for result in results}:
            raise RuntimeError("attempt-4 candidate already has durable result")
        row = plan[rid]
        payload = payloads[str(row["observation_id"])]
        criteria = json.loads(str(payload["prompt"])).get("criteria")
        if not isinstance(criteria, list) or not criteria:
            raise RuntimeError("attempt-4 candidate has invalid criteria payload")
        candidates.append({
            **row,
            "failure_class": "invalid_argument",
            "prior_attempt_count": 3,
            "recovery_attempt": 4,
            "recovery_protocol_version": GEMINI_MEMORA_ATTEMPT4_RECOVERY,
            "transport_schema_version": GEMINI_MEMORA_TRANSPORT_SCHEMA_VERSION,
            "output_budget_policy_version": GEMINI_MEMORA_OUTPUT_BUDGET_VERSION,
            "configured_max_output_tokens": attempt3_output_budget(len(criteria)),
            "criterion_count": len(criteria),
        })
    if len(candidates) > 1:
        raise RuntimeError(f"expected at most one attempt-4 schema smoke, found {len(candidates)}")
    return candidates


def attempt4_manifest_hash(rows: list[dict[str, Any]]) -> str:
    return sha256_value([attempt3_public_record(row) for row in rows])


def load_attempt4_manifest(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != 1 or rows[0].get("formal_validation_request_id") != ATTEMPT4_SMOKE_REQUEST_ID:
        raise RuntimeError("frozen attempt-4 manifest must contain exactly the known smoke request")
    return rows


def validate_attempt4_authorization(
    path: Path,
    frozen_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "authorized": True,
        "protocol_version": PROTOCOL,
        "evaluator_methodology": EVALUATOR,
        "recovery_protocol_version": GEMINI_MEMORA_ATTEMPT4_RECOVERY,
        "recovery_manifest_hash": attempt4_manifest_hash(frozen_rows),
        "maximum_authorized_retries_per_request": 1,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"attempt-4 authorization drift: {key}")
    ids = value.get("authorized_request_ids")
    frozen_ids = {row["formal_validation_request_id"] for row in frozen_rows}
    if not isinstance(ids, list) or set(map(str, ids)) != frozen_ids:
        raise RuntimeError("attempt-4 authorization must bind exactly the frozen smoke request")
    if {row["formal_validation_request_id"] for row in selected} != frozen_ids:
        raise RuntimeError("selected attempt-4 request differs from authorization")
    return value


def result_index(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for provider in MODELS:
        path = root / provider / "judge_results.jsonl"
        if not path.exists(): continue
        for row in read_jsonl(path):
            key = (str(row["observation_id"]), str(row["judge_slot"]))
            if key in out: raise ValueError(f"duplicate formal semantic result: {key}")
            out[key] = row
    return out


def criterion_map(result: dict[str, Any]) -> dict[str, str] | None:
    parsed = result.get("parsed_result", {})
    decisions = parsed.get("decisions")
    if not isinstance(decisions, list): return None
    return {str(x["criterion_id"]): str(x["decision"]) for x in decisions}


def make_dynamic(payload: dict[str, Any], slot: str, provider: str, phase: str, role: str) -> dict[str, Any]:
    model = MODELS[provider]; request_prompt = provider_request_prompt(provider, payload["prompt"])
    request_hash = prompt_hash(request_prompt); measured = estimate_input_tokens(request_prompt, model)
    if measured["estimated_tokens"] > LIMITS[provider]: raise ValueError("dynamic input-limit blocker")
    rid = formal_request_id(payload["observation_id"], slot, provider, model, request_hash, payload["payload_hash"])
    return {
        "formal_validation_request_id": rid, "pilot_block_id": payload["pilot_block_id"],
        "observation_id": payload["observation_id"], "condition_id": payload["condition_id"],
        "answer_sha256": payload["answer_sha256"], "source": payload["source"], "subtype": payload.get("subtype"),
        "judge_slot": slot, "judge_role": role, "phase": phase, "provider": provider, "model": model,
        "semantic_prompt_hash": payload["semantic_prompt_hash"], "request_hash": request_hash,
        "payload_hash": payload["payload_hash"], "protocol_version": PROTOCOL, "evaluator_methodology": EVALUATOR,
        "private_payload_locator": f"semantic_payloads.jsonl#{payload['observation_id']}",
        "estimated_input_tokens": measured["estimated_tokens"], "input_estimation_method": measured["method"],
        "planned_output_tokens": 160 if payload["source"] == "longmemeval" else 40 + 36 * int(payload["criterion_n"]),
        "max_attempts": 1, "automatic_retries": False,
    }


def dynamic_plan(payloads: dict[str, dict[str, Any]], results: dict[tuple[str, str], dict[str, Any]],
                 static: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []; static_secondary = {r["observation_id"] for r in static if r["judge_slot"] == "memora_secondary_openai"}
    for oid, payload in payloads.items():
        if payload["source"] == "longmemeval":
            a = results.get((oid, "lme_primary_openai")); b = results.get((oid, "lme_primary_anthropic"))
            if a and b:
                av = a["parsed_result"].get("exploratory_binary"); bv = b["parsed_result"].get("exploratory_binary")
                if av is None or bv is None or av != bv:
                    rows.append(make_dynamic(payload, "lme_escalation_gemini", "gemini", "escalation", "binary_disagreement_third"))
        else:
            primary = results.get((oid, "memora_primary_gemini"))
            if primary:
                decisions = primary["parsed_result"].get("decisions", [])
                risky = any(x.get("confidence") == "low" or x.get("decision") == "adjudication_required" for x in decisions)
                if risky and oid not in static_secondary:
                    rows.append(make_dynamic(payload, "memora_dynamic_secondary_openai", "openai", "secondary", "dynamic_risk_secondary"))
            secondary = results.get((oid, "memora_secondary_openai")) or results.get((oid, "memora_dynamic_secondary_openai"))
            if primary and secondary:
                pm, sm = criterion_map(primary), criterion_map(secondary)
                if pm is None or sm is None or pm.keys() != sm.keys() or any(pm[k] != sm[k] for k in pm):
                    rows.append(make_dynamic(payload, "memora_escalation_anthropic", "anthropic", "escalation", "criterion_disagreement_third"))
    return sorted(rows, key=lambda r: r["formal_validation_request_id"])


def persist_dynamic(path: Path, rows: list[dict[str, Any]]) -> None:
    existing = read_jsonl(path) if path.exists() else []; by_id = {r["formal_validation_request_id"]: r for r in existing}
    if len(by_id) != len(existing): raise ValueError("duplicate dynamic request")
    for row in rows:
        rid = row["formal_validation_request_id"]
        if rid in by_id and by_id[rid] != row: raise ValueError("dynamic request identity drift")
        by_id[rid] = row
    payload = "".join(json.dumps(by_id[k], sort_keys=True) + "\n" for k in sorted(by_id))
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(payload, encoding="utf-8")


def state(rows: list[dict[str, Any]], ledger: DurableLedger, fail: bool) -> dict[str, int]:
    events = ledger.read(); expected = {r["formal_validation_request_id"]: r for r in rows}
    histories = defaultdict(list)
    for event in events:
        rid = event.get("formal_validation_request_id")
        if rid not in expected: continue  # same provider ledger may contain another selected phase
        row = expected[rid]
        for key in ("observation_id", "provider", "model", "request_hash", "payload_hash", "protocol_version", "evaluator_methodology"):
            if str(event.get(key)) != str(row.get(key)): raise RuntimeError(f"ledger identity drift: {rid}")
        histories[rid].append(event)
    counts = defaultdict(int)
    for rid in expected:
        history = histories.get(rid, [])
        label = "pending" if not history else str(history[-1].get("state"))
        counts[label] += 1
    blocked = counts["failed"] + counts["started"]
    if blocked and fail: raise RuntimeError(f"fail-closed: {blocked} formal slot(s) require recovery")
    return {"pending": counts["pending"], "completed": counts["completed"], "failed": counts["failed"], "started_only": counts["started"], "blocked": blocked}


def pending_partition_rows(rows: list[dict[str, Any]], ledger: DurableLedger) -> list[dict[str, Any]]:
    """Return only untouched (fully unattempted) rows, in frozen plan order.

    Selection never reopens a slot that already has any durable history, so
    completed requests are never rerun and failed/blocked requests are never
    silently retried.  Identity drift and duplicate request identities fail
    closed before any request could be selected.
    """
    request_ids = [str(row["formal_validation_request_id"]) for row in rows]
    if len(request_ids) != len(set(request_ids)):
        raise RuntimeError("duplicate request identity in selected partition")
    expected = {row["formal_validation_request_id"]: row for row in rows}
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in ledger.read():
        rid = str(event.get("formal_validation_request_id"))
        if rid not in expected: continue
        row = expected[rid]
        for key in ("observation_id", "provider", "model", "request_hash", "payload_hash", "protocol_version", "evaluator_methodology"):
            if str(event.get(key)) != str(row.get(key)): raise RuntimeError(f"ledger identity drift: {rid}")
        histories[rid].append(event)
    return [row for row in rows if not histories.get(row["formal_validation_request_id"])]


def adapter(provider: str, max_tokens: int) -> Any:
    if provider == "openai": return OpenAIJudgeAdapter(MODELS[provider], max_output_tokens=max_tokens)
    if provider == "anthropic": return AnthropicJudgeAdapter(MODELS[provider], max_output_tokens=max_tokens)
    return GeminiJudgeAdapter(MODELS[provider], max_output_tokens=max_tokens)


def adapter_task(source: str) -> str:
    """Select transport schema without changing the frozen semantic prompt."""
    return "memora_batch" if source == "memora" else source


def failure_diagnostics(exc: BaseException, client: Any, raw: str | None) -> dict[str, Any] | None:
    """Safe, bounded transport diagnostics attached to a failed judge slot.

    Prefers diagnostics captured by the provider adapter (stop_reason, usage,
    request id, content-block shape); falls back to a bounded, redacted shape
    diagnostic of the raw response text when only the payload is available.
    Never repairs or re-interprets the payload.
    """
    diagnostics = getattr(exc, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        diagnostics = getattr(client, "last_diagnostics", None)
    if isinstance(diagnostics, dict) and diagnostics:
        merged = dict(diagnostics)
        if raw is not None and merged.get("has_raw_text") is None:
            merged["has_raw_text"] = len(raw) > 0
            merged.setdefault("raw_text_char_count", len(raw))
        return merged
    if raw is not None:
        return judge_response_diagnostics(None, raw_text=raw)
    return None


def execute_partition(provider: str, rows: list[dict[str, Any]], payloads: dict[str, dict[str, Any]], root: Path) -> int:
    env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}[provider]
    if not os.environ.get(env): raise RuntimeError(f"{env} missing before any ledger write")
    base = root / provider; ledger = DurableLedger(base / "judge_execution_ledger.jsonl")
    status = state(rows, ledger, True); results_path = base / "judge_results.jsonl"
    existing = read_jsonl(results_path) if results_path.exists() else []
    if len({r["formal_validation_request_id"] for r in existing}) != len(existing): raise RuntimeError("duplicate durable result")
    lock = base / ".execution.lock"; base.mkdir(parents=True, exist_ok=True)
    try: fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc: raise RuntimeError("provider partition already locked") from exc
    os.close(fd); paid = 0
    # Audit note: normal execution previously built ONE adapter with the maximum
    # planned budget across the selected batch, so a request planned at 2164
    # tokens could run under another request's larger 2236-token budget.  Each
    # request now uses its own planned budget (plus any provider floor); explicit
    # recovery classes still override the budget for their own slots.
    try:
        events = ledger.read()
        completed = {e["formal_validation_request_id"] for e in events if e.get("state") == "completed"}
        for row in rows:
            rid = row["formal_validation_request_id"]
            if rid in completed: continue
            identity = {k: row[k] for k in ("formal_validation_request_id", "pilot_block_id", "observation_id", "source", "judge_slot", "provider", "model", "request_hash", "payload_hash", "protocol_version", "evaluator_methodology")}
            client = adapter(provider, formal_output_budget(
                provider, row["planned_output_tokens"],
                judge_slot=row.get("judge_slot"), source=row.get("source"), phase=row.get("phase"),
            ))
            ledger.append({**identity, "state": "planned", "attempt": 1, "timestamp_ns": time.time_ns()})
            ledger.append({**identity, "state": "started", "attempt": 1, "timestamp_ns": time.time_ns()}); paid += 1
            raw: str | None = None
            try:
                payload = payloads[row["observation_id"]]; raw = client.call(payload["prompt"], adapter_task(payload["source"]))
                parsed = parse_result(raw, payload["source"], payload["prompt"]); response_hash = sha256_value(raw)
                append_jsonl(results_path, {**identity, "attempt": 1, "parsed_result": parsed, "response_hash": response_hash})
                ledger.append({**identity, "state": "completed", "attempt": 1, "response_hash": response_hash, "timestamp_ns": time.time_ns()})
            except Exception as exc:
                event = {**identity, "state": "failed", "attempt": 1, "error_class": type(exc).__name__, "error_message": str(exc)[:1000], "timestamp_ns": time.time_ns()}
                diagnostics = failure_diagnostics(exc, client, raw)
                if diagnostics is not None: event["error_diagnostics"] = diagnostics
                ledger.append(event)
    finally: lock.unlink(missing_ok=True)
    return paid


def select_gemini_memora_recovery(
    rows: list[dict[str, Any]],
    ledger: DurableLedger,
    results: list[dict[str, Any]],
    max_requests: int,
) -> tuple[list[dict[str, Any]], int]:
    """Select only current attempt-1 failures caused by the schema mismatch."""
    if max_requests < 1:
        raise RuntimeError("max recovery requests must be at least one")
    eligible_plan = {
        row["formal_validation_request_id"]: row
        for row in rows
        if row.get("provider") == "gemini"
        and row.get("model") == MODELS["gemini"]
        and row.get("source") == "memora"
        and row.get("phase") == "primary"
        and row.get("judge_slot") == "memora_primary_gemini"
    }
    if len(eligible_plan) != 450:
        raise RuntimeError(f"expected 450 frozen Gemini/Memora primary slots, found {len(eligible_plan)}")
    result_ids = [str(row.get("formal_validation_request_id")) for row in results]
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("duplicate durable result")
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in ledger.read():
        rid = str(event.get("formal_validation_request_id"))
        if rid not in eligible_plan:
            continue
        row = eligible_plan[rid]
        for key in ("observation_id", "provider", "model", "request_hash", "payload_hash", "protocol_version", "evaluator_methodology"):
            if str(event.get(key)) != str(row.get(key)):
                raise RuntimeError(f"ledger identity drift: {rid}")
        histories[rid].append(event)
    candidates: list[dict[str, Any]] = []
    for rid, row in eligible_plan.items():
        history = histories.get(rid, [])
        if rid in result_ids or any(event.get("state") == "completed" for event in history):
            continue
        if any(event.get("attempt") == 2 or event.get("state") == "recovery_authorized" for event in history):
            continue
        attempt_one = [event for event in history if event.get("attempt") == 1]
        states = [event.get("state") for event in attempt_one]
        failures = [event for event in attempt_one if event.get("state") == "failed"]
        if states.count("started") != 1 or len(failures) != 1:
            continue
        terminal = failures[0]
        if terminal.get("error_class") != "ValueError" or terminal.get("error_message") != GEMINI_MEMORA_FAILURE:
            continue
        candidates.append(row)
    candidates.sort(key=lambda row: row["formal_validation_request_id"])
    return candidates[:max_requests], len(candidates)


def execute_gemini_batch_recovery(
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    root: Path,
) -> int:
    """Consume one explicitly bounded set of attempt-2 Gemini recoveries."""
    if not rows:
        return 0
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY missing before any recovery ledger write")
    provider_root = root / "gemini"
    ledger = DurableLedger(provider_root / "judge_execution_ledger.jsonl")
    results_path = provider_root / "judge_results.jsonl"
    results = read_jsonl(results_path) if results_path.exists() else []
    selected, _available = select_gemini_memora_recovery(all_rows, ledger, results, len(rows))
    if selected != rows:
        raise RuntimeError("selected Gemini recovery population is no longer eligible")
    if len({row["formal_validation_request_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate Gemini recovery request identity")
    lock = provider_root / ".execution.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Gemini provider partition already locked") from exc
    os.close(fd)
    paid = 0
    try:
        client = adapter("gemini", max(int(row["planned_output_tokens"]) for row in rows))
        for row in rows:
            identity = {key: row[key] for key in (
                "formal_validation_request_id", "pilot_block_id", "observation_id", "source", "judge_slot",
                "provider", "model", "request_hash", "payload_hash", "protocol_version", "evaluator_methodology",
            )}
            ledger.append({
                **identity, "state": "recovery_authorized", "attempt": 1, "recovery_attempt": 2,
                "recovery_reason": "gemini_memora_batch_transport_schema_mismatch_no_valid_semantic_result",
                "recovery_protocol_version": GEMINI_MEMORA_RECOVERY, "timestamp_ns": time.time_ns(),
            })
            ledger.append({**identity, "state": "started", "attempt": 2, "timestamp_ns": time.time_ns()})
            paid += 1
            try:
                payload = payloads[row["observation_id"]]
                raw = client.call(payload["prompt"], adapter_task(payload["source"]))
                parsed = parse_result(raw, payload["source"], payload["prompt"])
                response_hash = sha256_value(raw)
                append_jsonl(results_path, {
                    **identity, "attempt": 2, "parsed_result": parsed, "response_hash": response_hash,
                    "recovery_protocol_version": GEMINI_MEMORA_RECOVERY,
                })
                ledger.append({
                    **identity, "state": "completed", "attempt": 2, "response_hash": response_hash,
                    "recovery_protocol_version": GEMINI_MEMORA_RECOVERY, "timestamp_ns": time.time_ns(),
                })
            except Exception as exc:
                ledger.append({
                    **identity, "state": "failed", "attempt": 2, "error_class": type(exc).__name__,
                    "error_message": str(exc)[:1000], "recovery_protocol_version": GEMINI_MEMORA_RECOVERY,
                    "timestamp_ns": time.time_ns(),
                })
    finally:
        lock.unlink(missing_ok=True)
    return paid


def _anthropic_no_text_terminal(event: dict[str, Any]) -> bool:
    return (
        event.get("state") == "failed"
        and event.get("attempt") == 1
        and event.get("error_class") == ANTHROPIC_NO_TEXT_ERROR_CLASS
        and event.get("error_message") == ANTHROPIC_NO_TEXT_ERROR_MESSAGE
    )


def select_anthropic_no_text_recovery(
    rows: list[dict[str, Any]],
    ledger: DurableLedger,
    results: list[dict[str, Any]],
    max_requests: int,
) -> tuple[list[dict[str, Any]], int]:
    """Select only untouched Anthropic no-text attempt-1 terminal failures.

    Eligibility is deliberately narrow: the Anthropic formal slot's latest
    durable event must be an attempt-1 ``ValueError`` /
    ``provider response contained no text`` failure.  Completed slots, pending
    slots, other failure classes, retried slots, and non-Anthropic requests are
    never selected, and any ledger identity drift fails closed.
    """
    if max_requests < 1:
        raise RuntimeError("max recovery requests must be at least one")
    eligible_plan = {
        row["formal_validation_request_id"]: row
        for row in rows
        if row.get("provider") == "anthropic"
        and row.get("model") == MODELS["anthropic"]
        and row.get("source") == "longmemeval"
    }
    result_ids = [str(row.get("formal_validation_request_id")) for row in results]
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("duplicate durable result")
    result_set = set(result_ids)
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in ledger.read():
        rid = str(event.get("formal_validation_request_id"))
        if rid not in eligible_plan:
            continue
        row = eligible_plan[rid]
        for key in ("observation_id", "provider", "model", "request_hash", "payload_hash", "protocol_version", "evaluator_methodology"):
            if str(event.get(key)) != str(row.get(key)):
                raise RuntimeError(f"ledger identity drift: {rid}")
        histories[rid].append(event)
    candidates: list[dict[str, Any]] = []
    for rid, row in eligible_plan.items():
        history = histories.get(rid, [])
        if rid in result_set or any(event.get("state") == "completed" for event in history):
            continue
        if any(event.get("state") == "recovery_authorized" or int(event.get("attempt", 1)) > 1 for event in history):
            continue
        attempt_one = [event for event in history if int(event.get("attempt", 1)) == 1]
        failures = [event for event in attempt_one if event.get("state") == "failed"]
        if len(failures) != 1 or not history or history[-1] is not failures[0]:
            continue
        if not _anthropic_no_text_terminal(failures[0]):
            continue
        candidates.append(row)
    candidates.sort(key=lambda row: row["formal_validation_request_id"])
    return candidates[:max_requests], len(candidates)


def execute_anthropic_no_text_recovery(
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    root: Path,
) -> int:
    """Consume one explicitly bounded set of attempt-2 Anthropic no-text retries.

    The original attempt-1 failure is preserved verbatim; recovery only appends
    the transport-style attempt-2 lifecycle for the same immutable identity.
    """
    if not rows:
        return 0
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY missing before any recovery ledger write")
    provider_root = root / "anthropic"
    ledger = DurableLedger(provider_root / "judge_execution_ledger.jsonl")
    results_path = provider_root / "judge_results.jsonl"
    results = read_jsonl(results_path) if results_path.exists() else []
    selected, _available = select_anthropic_no_text_recovery(all_rows, ledger, results, len(rows))
    if selected != rows:
        raise RuntimeError("selected Anthropic no-text recovery population is no longer eligible")
    if len({row["formal_validation_request_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate Anthropic recovery request identity")
    lock = provider_root / ".execution.lock"
    provider_root.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Anthropic provider partition already locked") from exc
    os.close(fd)
    paid = 0
    try:
        client = adapter("anthropic", max(formal_output_budget("anthropic", row["planned_output_tokens"]) for row in rows))
        for row in rows:
            identity = {key: row[key] for key in FORMAL_LEDGER_IDENTITY_KEYS}
            ledger.append({
                **identity, "state": "recovery_authorized", "attempt": 1, "recovery_attempt": 2,
                "recovery_reason": "anthropic_no_text_provider_response_retry",
                "recovery_protocol_version": ANTHROPIC_NO_TEXT_RECOVERY, "timestamp_ns": time.time_ns(),
            })
            ledger.append({**identity, "state": "started", "attempt": 2, "timestamp_ns": time.time_ns()})
            paid += 1
            raw: str | None = None
            try:
                payload = payloads[row["observation_id"]]
                raw = client.call(payload["prompt"], adapter_task(payload["source"]))
                parsed = parse_result(raw, payload["source"], payload["prompt"]); response_hash = sha256_value(raw)
                append_jsonl(results_path, {
                    **identity, "attempt": 2, "parsed_result": parsed, "response_hash": response_hash,
                    "recovery_protocol_version": ANTHROPIC_NO_TEXT_RECOVERY,
                })
                ledger.append({
                    **identity, "state": "completed", "attempt": 2, "response_hash": response_hash,
                    "recovery_protocol_version": ANTHROPIC_NO_TEXT_RECOVERY, "timestamp_ns": time.time_ns(),
                })
            except Exception as exc:
                event = {
                    **identity, "state": "failed", "attempt": 2, "error_class": type(exc).__name__,
                    "error_message": str(exc)[:1000], "recovery_protocol_version": ANTHROPIC_NO_TEXT_RECOVERY,
                    "timestamp_ns": time.time_ns(),
                }
                diagnostics = failure_diagnostics(exc, client, raw)
                if diagnostics is not None: event["error_diagnostics"] = diagnostics
                ledger.append(event)
    finally:
        lock.unlink(missing_ok=True)
    return paid


def _attempt_events(history: list[dict[str, Any]], attempt: int) -> list[dict[str, Any]]:
    return [event for event in history if int(event.get("attempt", 0)) == attempt]


def select_anthropic_attempt3_candidate(
    rows: list[dict[str, Any]],
    ledger: DurableLedger,
    results: list[dict[str, Any]],
    max_requests: int,
) -> tuple[list[dict[str, Any]], int]:
    """Select the single explicitly scoped Anthropic attempt-3 diagnostic retry.

    Eligibility is intentionally narrow: exactly one frozen request id whose
    durable history is an attempt-1 no-text failure followed by an attempt-2
    malformed-JSON failure, with no completed result and fully unchanged
    identity.  Any identity mismatch fails closed; anything else is simply not
    eligible.
    """
    if max_requests != 1:
        raise RuntimeError("Anthropic attempt-3 diagnostic retry is limited to exactly one request")
    plan = {row["formal_validation_request_id"]: row for row in rows}
    target = plan.get(ANTHROPIC_ATTEMPT3_REQUEST_ID)
    if target is None:
        raise RuntimeError("exact Anthropic attempt-3 target is absent from the frozen plan")
    if target.get("provider") != "anthropic" or target.get("model") != MODELS["anthropic"]:
        raise RuntimeError("Anthropic attempt-3 target provider/model drift")
    if target.get("judge_slot") != ANTHROPIC_ATTEMPT3_JUDGE_SLOT:
        raise RuntimeError("Anthropic attempt-3 target judge slot drift")
    if any(
        str(row.get("formal_validation_request_id")) == ANTHROPIC_ATTEMPT3_REQUEST_ID
        and row.get("parsed_result") is not None
        for row in results
    ):
        return [], 0
    history = [
        event for event in ledger.read()
        if str(event.get("formal_validation_request_id")) == ANTHROPIC_ATTEMPT3_REQUEST_ID
    ]
    for event in history:
        for key in FORMAL_LEDGER_IDENTITY_KEYS:
            if str(event.get(key)) != str(target.get(key)):
                raise RuntimeError(f"Anthropic attempt-3 ledger identity drift: {key}")
    if any(event.get("state") == "completed" for event in history):
        return [], 0
    if any(event.get("recovery_attempt") == 3 for event in history):
        return [], 0
    attempt_one = [
        event for event in _attempt_events(history, 1)
        if event.get("state") != "recovery_authorized"
    ]
    attempt_two = _attempt_events(history, 2)
    attempt_one_failures = [event for event in attempt_one if event.get("state") == "failed"]
    attempt_two_failures = [event for event in attempt_two if event.get("state") == "failed"]
    if [event.get("state") for event in attempt_one] != ["planned", "started", "failed"]:
        return [], 0
    if len(attempt_one_failures) != 1 or not _anthropic_no_text_terminal(attempt_one_failures[0]):
        return [], 0
    if [event.get("state") for event in attempt_two] != ["started", "failed"]:
        return [], 0
    if len(attempt_two_failures) != 1:
        return [], 0
    attempt_two_failure = attempt_two_failures[0]
    if attempt_two_failure.get("error_class") != "ValueError" or str(
        attempt_two_failure.get("error_message", "")
    ) != ANTHROPIC_ATTEMPT2_MALFORMED_JSON_MESSAGE:
        return [], 0
    if not history or history[-1] is not attempt_two_failure:
        return [], 0
    if not any(
        event.get("state") == "recovery_authorized" and event.get("recovery_attempt") == 2
        for event in history
    ):
        return [], 0
    return [target], 1


def execute_anthropic_attempt3_recovery(
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    root: Path,
) -> int:
    """Consume exactly one scoped attempt-3 diagnostic retry for the frozen slot.

    The scientific request is byte-for-byte the original (same model, prompt,
    schema, and planned output budget); attempts 1 and 2 remain intact.
    """
    if not rows:
        return 0
    if len(rows) != 1:
        raise RuntimeError("Anthropic attempt-3 diagnostic retry is limited to exactly one request")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY missing before any recovery ledger write")
    provider_root = root / "anthropic"
    ledger = DurableLedger(provider_root / "judge_execution_ledger.jsonl")
    results_path = provider_root / "judge_results.jsonl"
    results = read_jsonl(results_path) if results_path.exists() else []
    selected, _available = select_anthropic_attempt3_candidate(all_rows, ledger, results, len(rows))
    if selected != rows:
        raise RuntimeError("selected Anthropic attempt-3 slot is no longer eligible")
    lock = provider_root / ".execution.lock"
    provider_root.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Anthropic provider partition already locked") from exc
    os.close(fd)
    paid = 0
    try:
        client = adapter("anthropic", max(formal_output_budget("anthropic", row["planned_output_tokens"]) for row in rows))
        for row in rows:
            identity = {key: row[key] for key in FORMAL_LEDGER_IDENTITY_KEYS}
            ledger.append({
                **identity, "state": "recovery_authorized", "attempt": 2, "recovery_attempt": 3,
                "recovery_reason": "anthropic_attempt3_diagnostic_retry",
                "recovery_protocol_version": ANTHROPIC_ATTEMPT3_RECOVERY, "timestamp_ns": time.time_ns(),
            })
            ledger.append({**identity, "state": "started", "attempt": 3, "timestamp_ns": time.time_ns()})
            paid += 1
            raw: str | None = None
            try:
                payload = payloads[row["observation_id"]]
                raw = client.call(payload["prompt"], adapter_task(payload["source"]))
                parsed = parse_result(raw, payload["source"], payload["prompt"]); response_hash = sha256_value(raw)
                append_jsonl(results_path, {
                    **identity, "attempt": 3, "parsed_result": parsed, "response_hash": response_hash,
                    "recovery_protocol_version": ANTHROPIC_ATTEMPT3_RECOVERY,
                })
                ledger.append({
                    **identity, "state": "completed", "attempt": 3, "response_hash": response_hash,
                    "recovery_protocol_version": ANTHROPIC_ATTEMPT3_RECOVERY, "timestamp_ns": time.time_ns(),
                })
            except Exception as exc:
                event = {
                    **identity, "state": "failed", "attempt": 3, "error_class": type(exc).__name__,
                    "error_message": str(exc)[:1000], "recovery_protocol_version": ANTHROPIC_ATTEMPT3_RECOVERY,
                    "timestamp_ns": time.time_ns(),
                }
                diagnostics = failure_diagnostics(exc, client, raw)
                if diagnostics is not None: event["error_diagnostics"] = diagnostics
                ledger.append(event)
    finally:
        lock.unlink(missing_ok=True)
    return paid


def _attempt3_token_limit_proof(event: dict[str, Any]) -> bool:
    diagnostics = event.get("error_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    return (
        diagnostics.get("stop_reason") == "max_tokens"
        and diagnostics.get("output_tokens") == ANTHROPIC_ATTEMPT3_OBSERVED_OUTPUT_TOKENS
        and diagnostics.get("json_parse_ok") is False
    )


def select_anthropic_attempt4_candidate(
    rows: list[dict[str, Any]],
    ledger: DurableLedger,
    results: list[dict[str, Any]],
    max_requests: int,
) -> tuple[list[dict[str, Any]], int]:
    """Select the single scoped Anthropic attempt-4 token-budget retry.

    Eligibility requires the full preserved history (attempt-1 no-text,
    attempt-2 malformed JSON) plus attempt-3 diagnostics proving the failure was
    ``stop_reason=max_tokens`` truncation at the old 160-token budget.  The only
    permitted change for attempt 4 is the transport output budget.
    """
    if max_requests != 1:
        raise RuntimeError("Anthropic attempt-4 diagnostic retry is limited to exactly one request")
    plan = {row["formal_validation_request_id"]: row for row in rows}
    target = plan.get(ANTHROPIC_ATTEMPT3_REQUEST_ID)
    if target is None:
        raise RuntimeError("exact Anthropic attempt-4 target is absent from the frozen plan")
    if target.get("provider") != "anthropic" or target.get("model") != MODELS["anthropic"]:
        raise RuntimeError("Anthropic attempt-4 target provider/model drift")
    if target.get("judge_slot") != ANTHROPIC_ATTEMPT3_JUDGE_SLOT:
        raise RuntimeError("Anthropic attempt-4 target judge slot drift")
    if any(
        str(row.get("formal_validation_request_id")) == ANTHROPIC_ATTEMPT3_REQUEST_ID
        and row.get("parsed_result") is not None
        for row in results
    ):
        return [], 0
    history = [
        event for event in ledger.read()
        if str(event.get("formal_validation_request_id")) == ANTHROPIC_ATTEMPT3_REQUEST_ID
    ]
    for event in history:
        for key in FORMAL_LEDGER_IDENTITY_KEYS:
            if str(event.get(key)) != str(target.get(key)):
                raise RuntimeError(f"Anthropic attempt-4 ledger identity drift: {key}")
    if any(event.get("state") == "completed" for event in history):
        return [], 0
    if any(event.get("recovery_attempt") == 4 for event in history):
        return [], 0
    attempt_one = [
        event for event in _attempt_events(history, 1) if event.get("state") != "recovery_authorized"
    ]
    attempt_two = [
        event for event in _attempt_events(history, 2) if event.get("state") != "recovery_authorized"
    ]
    attempt_three = [
        event for event in _attempt_events(history, 3) if event.get("state") != "recovery_authorized"
    ]
    attempt_one_failures = [event for event in attempt_one if event.get("state") == "failed"]
    attempt_two_failures = [event for event in attempt_two if event.get("state") == "failed"]
    attempt_three_failures = [event for event in attempt_three if event.get("state") == "failed"]
    if [event.get("state") for event in attempt_one] != ["planned", "started", "failed"]:
        return [], 0
    if len(attempt_one_failures) != 1 or not _anthropic_no_text_terminal(attempt_one_failures[0]):
        return [], 0
    if [event.get("state") for event in attempt_two] != ["started", "failed"]:
        return [], 0
    if len(attempt_two_failures) != 1 or not (
        attempt_two_failures[0].get("error_class") == "ValueError"
        and str(attempt_two_failures[0].get("error_message", "")) == ANTHROPIC_ATTEMPT2_MALFORMED_JSON_MESSAGE
    ):
        return [], 0
    if [event.get("state") for event in attempt_three] != ["started", "failed"]:
        return [], 0
    if len(attempt_three_failures) != 1:
        return [], 0
    attempt_three_failure = attempt_three_failures[0]
    if attempt_three_failure.get("error_class") != "ValueError" or not str(
        attempt_three_failure.get("error_message", "")
    ).startswith("malformed JSON"):
        return [], 0
    if not _attempt3_token_limit_proof(attempt_three_failure):
        return [], 0
    if not history or history[-1] is not attempt_three_failure:
        return [], 0
    if not any(
        event.get("state") == "recovery_authorized" and event.get("recovery_attempt") == 3
        for event in history
    ):
        return [], 0
    return [target], 1


def execute_anthropic_attempt4_recovery(
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    root: Path,
) -> int:
    """Consume exactly one attempt-4 retry, changing only the output budget."""
    if not rows:
        return 0
    if len(rows) != 1:
        raise RuntimeError("Anthropic attempt-4 diagnostic retry is limited to exactly one request")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY missing before any recovery ledger write")
    provider_root = root / "anthropic"
    ledger = DurableLedger(provider_root / "judge_execution_ledger.jsonl")
    results_path = provider_root / "judge_results.jsonl"
    results = read_jsonl(results_path) if results_path.exists() else []
    selected, _available = select_anthropic_attempt4_candidate(all_rows, ledger, results, len(rows))
    if selected != rows:
        raise RuntimeError("selected Anthropic attempt-4 slot is no longer eligible")
    budget = formal_output_budget("anthropic", rows[0]["planned_output_tokens"])
    lock = provider_root / ".execution.lock"
    provider_root.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Anthropic provider partition already locked") from exc
    os.close(fd)
    paid = 0
    try:
        client = adapter("anthropic", budget)
        for row in rows:
            identity = {key: row[key] for key in FORMAL_LEDGER_IDENTITY_KEYS}
            ledger.append({
                **identity, "state": "recovery_authorized", "attempt": 3, "recovery_attempt": 4,
                "recovery_reason": "token_limit_truncation",
                "recovery_protocol_version": ANTHROPIC_ATTEMPT4_RECOVERY,
                "output_token_budget": budget, "timestamp_ns": time.time_ns(),
            })
            ledger.append({
                **identity, "state": "started", "attempt": 4, "output_token_budget": budget,
                "timestamp_ns": time.time_ns(),
            })
            paid += 1
            raw: str | None = None
            try:
                payload = payloads[row["observation_id"]]
                raw = client.call(payload["prompt"], adapter_task(payload["source"]))
                parsed = parse_result(raw, payload["source"], payload["prompt"]); response_hash = sha256_value(raw)
                append_jsonl(results_path, {
                    **identity, "attempt": 4, "parsed_result": parsed, "response_hash": response_hash,
                    "recovery_protocol_version": ANTHROPIC_ATTEMPT4_RECOVERY,
                })
                ledger.append({
                    **identity, "state": "completed", "attempt": 4, "response_hash": response_hash,
                    "recovery_protocol_version": ANTHROPIC_ATTEMPT4_RECOVERY, "timestamp_ns": time.time_ns(),
                })
            except Exception as exc:
                event = {
                    **identity, "state": "failed", "attempt": 4, "error_class": type(exc).__name__,
                    "error_message": str(exc)[:1000], "recovery_protocol_version": ANTHROPIC_ATTEMPT4_RECOVERY,
                    "timestamp_ns": time.time_ns(),
                }
                diagnostics = failure_diagnostics(exc, client, raw)
                if diagnostics is not None: event["error_diagnostics"] = diagnostics
                ledger.append(event)
    finally:
        lock.unlink(missing_ok=True)
    return paid


def _next_transport_attempt(history: list[dict[str, Any]]) -> int:
    """Next valid attempt number for a slot, derived from its own history."""
    started_attempts = [
        int(event.get("attempt", 0)) for event in history if event.get("state") == "started"
    ]
    return (max(started_attempts) if started_attempts else 0) + 1


def _anthropic_token_recovery_terminal(event: dict[str, Any], config: dict[str, Any]) -> bool:
    """True when the slot's terminal failure matches the recovery class proof."""
    if event.get("state") != "failed" or event.get("error_class") != "ValueError":
        return False
    if config.get("error_message") is not None and event.get("error_message") != config["error_message"]:
        return False
    if config.get("required_attempt") is not None and int(event.get("attempt", 0)) != config["required_attempt"]:
        return False
    diagnostics = event.get("error_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    if diagnostics.get("stop_reason") != "max_tokens":
        return False
    if diagnostics.get("output_tokens") != config["observed_output_tokens"]:
        return False
    if diagnostics.get("json_parse_ok") is not False:
        return False
    if config.get("require_zero_text") and diagnostics.get("text_block_count") != 0:
        return False
    if config.get("require_no_raw_text") and diagnostics.get("has_raw_text") is not False:
        return False
    if config.get("require_thinking"):
        block_types = diagnostics.get("content_block_types") or []
        if "thinking" not in block_types:
            return False
    return True


def _anthropic_token_recovery_universe(
    rows: list[dict[str, Any]],
    ledger: DurableLedger,
    results: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[tuple[dict[str, Any], int]], int]:
    """Return the frozen plan, the eligible (row, next-attempt) list, and the
    count of targets already consumed by this recovery class's protocol."""
    eligible_plan = {
        row["formal_validation_request_id"]: row
        for row in rows
        if row.get("provider") == "anthropic"
        and row.get("model") == MODELS["anthropic"]
        and row.get("judge_slot") == config["judge_slot"]
        and row.get("source") == config["source"]
        and (config.get("phase") is None or row.get("phase") == config["phase"])
    }
    result_ids = [str(row.get("formal_validation_request_id")) for row in results]
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("duplicate durable result")
    result_set = set(result_ids)
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    protocol_consumed: set[str] = set()
    for event in ledger.read():
        rid = str(event.get("formal_validation_request_id"))
        if event.get("recovery_protocol_version") == config["protocol"]:
            if rid not in eligible_plan:
                raise RuntimeError(f"{config['recovery_class']} recovery protocol recorded on a non-target slot: {rid}")
            protocol_consumed.add(rid)
        if rid not in eligible_plan:
            continue
        row = eligible_plan[rid]
        for key in FORMAL_LEDGER_IDENTITY_KEYS:
            if str(event.get(key)) != str(row.get(key)):
                raise RuntimeError(f"ledger identity drift: {rid}")
        histories[rid].append(event)
    eligible: list[tuple[dict[str, Any], int]] = []
    for rid, row in eligible_plan.items():
        history = histories.get(rid, [])
        if not history or rid in result_set:
            continue
        if any(event.get("state") == "completed" for event in history):
            continue
        if rid in protocol_consumed:
            continue
        if config.get("previous_recovery_class") is not None and not any(
            event.get("recovery_class") == config["previous_recovery_class"] for event in history
        ):
            continue
        if not _anthropic_token_recovery_terminal(history[-1], config):
            continue
        eligible.append((row, _next_transport_attempt(history)))
    return eligible_plan, eligible, len(protocol_consumed)


def select_anthropic_token_recovery(
    rows: list[dict[str, Any]],
    ledger: DurableLedger,
    results: list[dict[str, Any]],
    max_requests: int,
    recovery_class: str = ANTHROPIC_TOKEN_RECOVERY_512_TO_2048,
    expected_population: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Select the current Anthropic token-truncation failures for one recovery class.

    Each class owns a frozen population: every owned slot must be either still
    eligible (terminal ``max_tokens`` failure at the class's observed budget) or
    already consumed by that class's protocol.  Any other count fails closed, so
    a partial smoke/canary run keeps the invariant while never re-selecting a
    consumed, completed, or higher-budget-failed slot.  The two classes are
    disjoint by their observed ``output_tokens`` (512 vs 2048).
    """
    if max_requests < 1:
        raise RuntimeError("max recovery requests must be at least one")
    config = _token_recovery_config(recovery_class)
    population = config["population"] if expected_population is None else expected_population
    _plan, eligible, consumed = _anthropic_token_recovery_universe(rows, ledger, results, config)
    if len(eligible) + consumed != population:
        raise RuntimeError(
            f"{recovery_class} recovery population drift: "
            f"expected {population}, found {len(eligible)} eligible + {consumed} consumed"
        )
    eligible.sort(key=lambda item: item[0]["formal_validation_request_id"])
    selected = [row for row, _next in eligible[:max_requests]]
    return selected, len(eligible)


def execute_anthropic_token_recovery(
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    root: Path,
    recovery_class: str = ANTHROPIC_TOKEN_RECOVERY_512_TO_2048,
    expected_population: int | None = None,
) -> int:
    """Execute a bounded batch of Anthropic token-truncation retries for one class.

    Every retry keeps the identical model, prompt, schema, payload, and semantic
    parser; only the transport output budget changes.  The next attempt number
    is derived per slot from its own preserved history.
    """
    if not rows:
        return 0
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY missing before any recovery ledger write")
    config = _token_recovery_config(recovery_class)
    population = config["population"] if expected_population is None else expected_population
    provider_root = root / "anthropic"
    ledger = DurableLedger(provider_root / "judge_execution_ledger.jsonl")
    results_path = provider_root / "judge_results.jsonl"
    results = read_jsonl(results_path) if results_path.exists() else []
    selected, _available = select_anthropic_token_recovery(
        all_rows, ledger, results, len(rows), recovery_class, population,
    )
    if selected != rows:
        raise RuntimeError("selected Anthropic token recovery population is no longer eligible")
    if len({row["formal_validation_request_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate Anthropic recovery request identity")
    lock = provider_root / ".execution.lock"
    provider_root.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Anthropic provider partition already locked") from exc
    os.close(fd)
    paid = 0
    try:
        client = adapter("anthropic", config["output_budget"])
        for row in rows:
            rid = row["formal_validation_request_id"]
            history = [event for event in ledger.read() if str(event.get("formal_validation_request_id")) == rid]
            next_attempt = _next_transport_attempt(history)
            identity = {key: row[key] for key in FORMAL_LEDGER_IDENTITY_KEYS}
            ledger.append({
                **identity, "state": "recovery_authorized", "attempt": next_attempt - 1,
                "recovery_attempt": next_attempt, "recovery_reason": config["reason"],
                "recovery_class": recovery_class,
                "recovery_protocol_version": config["protocol"],
                "output_token_budget": config["output_budget"],
                "timestamp_ns": time.time_ns(),
            })
            ledger.append({
                **identity, "state": "started", "attempt": next_attempt,
                "recovery_class": recovery_class,
                "output_token_budget": config["output_budget"],
                "timestamp_ns": time.time_ns(),
            })
            paid += 1
            raw: str | None = None
            try:
                payload = payloads[row["observation_id"]]
                raw = client.call(payload["prompt"], adapter_task(payload["source"]))
                parsed = parse_result(raw, payload["source"], payload["prompt"]); response_hash = sha256_value(raw)
                append_jsonl(results_path, {
                    **identity, "attempt": next_attempt, "parsed_result": parsed, "response_hash": response_hash,
                    "recovery_protocol_version": config["protocol"],
                })
                ledger.append({
                    **identity, "state": "completed", "attempt": next_attempt, "response_hash": response_hash,
                    "recovery_protocol_version": config["protocol"], "timestamp_ns": time.time_ns(),
                })
            except Exception as exc:
                event = {
                    **identity, "state": "failed", "attempt": next_attempt, "error_class": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                    "recovery_protocol_version": config["protocol"], "timestamp_ns": time.time_ns(),
                }
                diagnostics = failure_diagnostics(exc, client, raw)
                if diagnostics is not None: event["error_diagnostics"] = diagnostics
                ledger.append(event)
    finally:
        lock.unlink(missing_ok=True)
    return paid


def select_anthropic_token_limit_recovery(
    rows: list[dict[str, Any]],
    ledger: DurableLedger,
    results: list[dict[str, Any]],
    max_requests: int,
    expected_population: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Compatibility wrapper for the 512 -> 2048 token-recovery class."""
    return select_anthropic_token_recovery(
        rows, ledger, results, max_requests, ANTHROPIC_TOKEN_RECOVERY_512_TO_2048, expected_population,
    )


def execute_anthropic_token_limit_recovery(
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    root: Path,
    expected_population: int | None = None,
) -> int:
    """Compatibility wrapper for the 512 -> 2048 token-recovery class."""
    return execute_anthropic_token_recovery(
        rows, all_rows, payloads, root, ANTHROPIC_TOKEN_RECOVERY_512_TO_2048, expected_population,
    )


def select_openai_no_text_recovery(
    rows: list[dict[str, Any]],
    ledger: DurableLedger,
    results: list[dict[str, Any]],
    max_requests: int,
    expected_population: int = OPENAI_NO_TEXT_RECOVERY_POPULATION,
) -> tuple[list[dict[str, Any]], int]:
    """Select the current scoped OpenAI no-text smoke slot for a budget retry."""
    if max_requests < 1:
        raise RuntimeError("max recovery requests must be at least one")
    plan = {row["formal_validation_request_id"]: row for row in rows}
    target = plan.get(OPENAI_NO_TEXT_RECOVERY_REQUEST_ID)
    if target is None:
        raise RuntimeError("exact OpenAI no-text target is absent from the frozen plan")
    if target.get("provider") != "openai" or target.get("model") != MODELS["openai"]:
        raise RuntimeError("OpenAI no-text target provider/model drift")
    if target.get("judge_slot") != OPENAI_NO_TEXT_JUDGE_SLOT or target.get("source") != "memora":
        raise RuntimeError("OpenAI no-text target judge-slot/source drift")
    if int(target.get("planned_output_tokens", -1)) != OPENAI_NO_TEXT_PLANNED_OUTPUT_TOKENS:
        raise RuntimeError("OpenAI no-text target planned output budget drift")
    result_ids = [str(row.get("formal_validation_request_id")) for row in results]
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("duplicate durable result")
    result_set = set(result_ids)
    consumed: set[str] = set()
    history: list[dict[str, Any]] = []
    for event in ledger.read():
        rid = str(event.get("formal_validation_request_id"))
        if event.get("recovery_protocol_version") == OPENAI_NO_TEXT_RECOVERY:
            if rid != OPENAI_NO_TEXT_RECOVERY_REQUEST_ID:
                raise RuntimeError(f"OpenAI no-text recovery protocol recorded on a non-target slot: {rid}")
            consumed.add(rid)
        if rid != OPENAI_NO_TEXT_RECOVERY_REQUEST_ID:
            continue
        for key in FORMAL_LEDGER_IDENTITY_KEYS:
            if str(event.get(key)) != str(target.get(key)):
                raise RuntimeError(f"OpenAI no-text ledger identity drift: {key}")
        history.append(event)
    eligible: list[dict[str, Any]] = []
    if OPENAI_NO_TEXT_RECOVERY_REQUEST_ID not in result_set and OPENAI_NO_TEXT_RECOVERY_REQUEST_ID not in consumed:
        if history and not any(event.get("state") == "completed" for event in history):
            terminal = history[-1]
            if (
                terminal.get("state") == "failed"
                and int(terminal.get("attempt", 0)) == 1
                and terminal.get("error_class") == "ValueError"
                and terminal.get("error_message") == OPENAI_NO_TEXT_ERROR_MESSAGE
            ):
                eligible.append(target)
    if len(eligible) + len(consumed) != expected_population:
        raise RuntimeError(
            f"OpenAI no-text recovery population drift: expected {expected_population}, "
            f"found {len(eligible)} eligible + {len(consumed)} consumed"
        )
    return eligible[:max_requests], len(eligible)


def execute_openai_no_text_recovery(
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    root: Path,
) -> int:
    """Consume one scoped OpenAI no-text retry at a larger transport budget.

    Same model, prompt, schema, and semantic parser; only the output budget
    changes.  Attempt 1 remains intact and attempt 2 is appended.
    """
    if not rows:
        return 0
    if len(rows) != 1:
        raise RuntimeError("OpenAI no-text recovery is limited to exactly one request")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing before any recovery ledger write")
    provider_root = root / "openai"
    ledger = DurableLedger(provider_root / "judge_execution_ledger.jsonl")
    results_path = provider_root / "judge_results.jsonl"
    results = read_jsonl(results_path) if results_path.exists() else []
    selected, _available = select_openai_no_text_recovery(all_rows, ledger, results, len(rows))
    if selected != rows:
        raise RuntimeError("selected OpenAI no-text recovery slot is no longer eligible")
    lock = provider_root / ".execution.lock"
    provider_root.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("OpenAI provider partition already locked") from exc
    os.close(fd)
    paid = 0
    try:
        client = adapter("openai", OPENAI_NO_TEXT_RECOVERY_OUTPUT_TOKENS)
        for row in rows:
            identity = {key: row[key] for key in FORMAL_LEDGER_IDENTITY_KEYS}
            ledger.append({
                **identity, "state": "recovery_authorized", "attempt": 1, "recovery_attempt": 2,
                "recovery_reason": OPENAI_NO_TEXT_FAILURE_CLASS,
                "recovery_protocol_version": OPENAI_NO_TEXT_RECOVERY,
                "output_token_budget": OPENAI_NO_TEXT_RECOVERY_OUTPUT_TOKENS,
                "timestamp_ns": time.time_ns(),
            })
            ledger.append({
                **identity, "state": "started", "attempt": 2,
                "output_token_budget": OPENAI_NO_TEXT_RECOVERY_OUTPUT_TOKENS,
                "timestamp_ns": time.time_ns(),
            })
            paid += 1
            raw: str | None = None
            try:
                payload = payloads[row["observation_id"]]
                raw = client.call(payload["prompt"], adapter_task(payload["source"]))
                parsed = parse_result(raw, payload["source"], payload["prompt"]); response_hash = sha256_value(raw)
                append_jsonl(results_path, {
                    **identity, "attempt": 2, "parsed_result": parsed, "response_hash": response_hash,
                    "recovery_protocol_version": OPENAI_NO_TEXT_RECOVERY,
                })
                ledger.append({
                    **identity, "state": "completed", "attempt": 2, "response_hash": response_hash,
                    "recovery_protocol_version": OPENAI_NO_TEXT_RECOVERY, "timestamp_ns": time.time_ns(),
                })
            except Exception as exc:
                event = {
                    **identity, "state": "failed", "attempt": 2, "error_class": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                    "recovery_protocol_version": OPENAI_NO_TEXT_RECOVERY, "timestamp_ns": time.time_ns(),
                }
                diagnostics = failure_diagnostics(exc, client, raw)
                if diagnostics is not None: event["error_diagnostics"] = diagnostics
                ledger.append(event)
    finally:
        lock.unlink(missing_ok=True)
    return paid


def _openai_token_recovery_config(recovery_class: str) -> dict[str, Any]:
    """Disjoint OpenAI token-limit recovery classes and their budgets."""
    if recovery_class == OPENAI_TOKEN_RECOVERY_2236_TO_4096:
        return {
            "judge_slot": OPENAI_NO_TEXT_JUDGE_SLOT, "source": "memora",
            "observed_output_tokens": OPENAI_TOKEN_RECOVERY_2236_OBSERVED_OUTPUT_TOKENS,
            "output_budget": OPENAI_TOKEN_RECOVERY_4096_OUTPUT_TOKENS,
            "protocol": OPENAI_TOKEN_RECOVERY_4096_PROTOCOL, "reason": OPENAI_TOKEN_LIMIT_REASON_2236,
            "population": OPENAI_TOKEN_RECOVERY_2236_POPULATION,
        }
    if recovery_class == OPENAI_TOKEN_RECOVERY_160_TO_512:
        return {
            "judge_slot": OPENAI_TOKEN_RECOVERY_160_JUDGE_SLOT, "source": "longmemeval",
            "observed_output_tokens": OPENAI_TOKEN_RECOVERY_160_OBSERVED_OUTPUT_TOKENS,
            "output_budget": OPENAI_TOKEN_RECOVERY_512_OUTPUT_TOKENS,
            "protocol": OPENAI_TOKEN_RECOVERY_512_PROTOCOL, "reason": OPENAI_TOKEN_LIMIT_REASON_160,
            "population": OPENAI_TOKEN_RECOVERY_160_POPULATION,
        }
    if recovery_class == OPENAI_TOKEN_RECOVERY_512_TO_1024:
        return {
            "judge_slot": OPENAI_TOKEN_RECOVERY_160_JUDGE_SLOT, "source": "longmemeval",
            "observed_output_tokens": OPENAI_TOKEN_RECOVERY_512_OBSERVED_OUTPUT_TOKENS,
            "output_budget": OPENAI_TOKEN_RECOVERY_1024_OUTPUT_TOKENS,
            "protocol": OPENAI_TOKEN_RECOVERY_1024_PROTOCOL, "reason": OPENAI_TOKEN_LIMIT_REASON_512,
            "population": OPENAI_TOKEN_RECOVERY_512_TO_1024_POPULATION,
        }
    raise RuntimeError(f"unknown OpenAI token recovery class: {recovery_class}")


def _openai_token_limit_terminal(event: dict[str, Any], config: dict[str, Any]) -> bool:
    """True when the OpenAI slot's terminal failure is a max_output_tokens truncation."""
    if event.get("state") != "failed" or event.get("error_class") != "ValueError":
        return False
    diagnostics = event.get("error_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    return (
        diagnostics.get("status") == "incomplete"
        and diagnostics.get("incomplete_reason") == "max_output_tokens"
        and diagnostics.get("output_tokens") == config["observed_output_tokens"]
        and diagnostics.get("configured_max_output_tokens") == config["observed_output_tokens"]
        and diagnostics.get("json_parse_ok") is False
    )


def select_openai_token_limit_recovery(
    rows: list[dict[str, Any]],
    ledger: DurableLedger,
    results: list[dict[str, Any]],
    max_requests: int,
    recovery_class: str = OPENAI_TOKEN_RECOVERY_2236_TO_4096,
    expected_population: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Select current OpenAI max_output_tokens truncation failures for a recovery class.

    Eligibility is derived from current durable terminal state (never a
    hard-coded id).  Each class owns a frozen population and the invariant
    ``eligible + consumed == population`` keeps a partial smoke/canary run
    consistent while never re-selecting a consumed, completed, cross-phase, or
    non-truncation slot.
    """
    if max_requests < 1:
        raise RuntimeError("max recovery requests must be at least one")
    config = _openai_token_recovery_config(recovery_class)
    population = config["population"] if expected_population is None else expected_population
    eligible_plan = {
        row["formal_validation_request_id"]: row
        for row in rows
        if row.get("provider") == "openai"
        and row.get("model") == MODELS["openai"]
        and row.get("judge_slot") == config["judge_slot"]
        and row.get("source") == config["source"]
    }
    result_ids = [str(row.get("formal_validation_request_id")) for row in results]
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("duplicate durable result")
    result_set = set(result_ids)
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    consumed: set[str] = set()
    for event in ledger.read():
        rid = str(event.get("formal_validation_request_id"))
        if event.get("recovery_protocol_version") == config["protocol"]:
            if rid not in eligible_plan:
                raise RuntimeError(f"{recovery_class} recovery protocol recorded on a non-target slot: {rid}")
            consumed.add(rid)
        if rid not in eligible_plan:
            continue
        row = eligible_plan[rid]
        for key in FORMAL_LEDGER_IDENTITY_KEYS:
            if str(event.get(key)) != str(row.get(key)):
                raise RuntimeError(f"ledger identity drift: {rid}")
        histories[rid].append(event)
    eligible: list[tuple[dict[str, Any], int]] = []
    for rid, row in eligible_plan.items():
        history = histories.get(rid, [])
        if not history or rid in result_set:
            continue
        if any(event.get("state") == "completed" for event in history):
            continue
        if rid in consumed:
            continue
        if not _openai_token_limit_terminal(history[-1], config):
            continue
        eligible.append((row, _next_transport_attempt(history)))
    if len(eligible) + len(consumed) != population:
        raise RuntimeError(
            f"{recovery_class} recovery population drift: expected {population}, "
            f"found {len(eligible)} eligible + {len(consumed)} consumed"
        )
    eligible.sort(key=lambda item: item[0]["formal_validation_request_id"])
    return [row for row, _next in eligible[:max_requests]], len(eligible)


def execute_openai_token_limit_recovery(
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    root: Path,
    recovery_class: str = OPENAI_TOKEN_RECOVERY_2236_TO_4096,
) -> int:
    """Consume a bounded batch of OpenAI truncation retries for one recovery class."""
    if not rows:
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing before any recovery ledger write")
    config = _openai_token_recovery_config(recovery_class)
    provider_root = root / "openai"
    ledger = DurableLedger(provider_root / "judge_execution_ledger.jsonl")
    results_path = provider_root / "judge_results.jsonl"
    results = read_jsonl(results_path) if results_path.exists() else []
    selected, _available = select_openai_token_limit_recovery(all_rows, ledger, results, len(rows), recovery_class)
    if selected != rows:
        raise RuntimeError("selected OpenAI token-limit recovery population is no longer eligible")
    if len({row["formal_validation_request_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate OpenAI recovery request identity")
    lock = provider_root / ".execution.lock"
    provider_root.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("OpenAI provider partition already locked") from exc
    os.close(fd)
    paid = 0
    try:
        client = adapter("openai", config["output_budget"])
        for row in rows:
            rid = row["formal_validation_request_id"]
            history = [event for event in ledger.read() if str(event.get("formal_validation_request_id")) == rid]
            next_attempt = _next_transport_attempt(history)
            identity = {key: row[key] for key in FORMAL_LEDGER_IDENTITY_KEYS}
            ledger.append({
                **identity, "state": "recovery_authorized", "attempt": next_attempt - 1,
                "recovery_attempt": next_attempt, "recovery_reason": config["reason"],
                "recovery_class": recovery_class,
                "recovery_protocol_version": config["protocol"],
                "output_token_budget": config["output_budget"],
                "timestamp_ns": time.time_ns(),
            })
            ledger.append({
                **identity, "state": "started", "attempt": next_attempt,
                "recovery_class": recovery_class,
                "output_token_budget": config["output_budget"],
                "timestamp_ns": time.time_ns(),
            })
            paid += 1
            raw: str | None = None
            try:
                payload = payloads[row["observation_id"]]
                raw = client.call(payload["prompt"], adapter_task(payload["source"]))
                parsed = parse_result(raw, payload["source"], payload["prompt"]); response_hash = sha256_value(raw)
                append_jsonl(results_path, {
                    **identity, "attempt": next_attempt, "parsed_result": parsed, "response_hash": response_hash,
                    "recovery_protocol_version": config["protocol"],
                })
                ledger.append({
                    **identity, "state": "completed", "attempt": next_attempt, "response_hash": response_hash,
                    "recovery_protocol_version": config["protocol"], "timestamp_ns": time.time_ns(),
                })
            except Exception as exc:
                event = {
                    **identity, "state": "failed", "attempt": next_attempt, "error_class": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                    "recovery_protocol_version": config["protocol"], "timestamp_ns": time.time_ns(),
                }
                diagnostics = failure_diagnostics(exc, client, raw)
                if diagnostics is not None: event["error_diagnostics"] = diagnostics
                ledger.append(event)
    finally:
        lock.unlink(missing_ok=True)
    return paid


def _human_adjudication_case_path(root: Path, request_id: str) -> Path:
    return root / HUMAN_ADJUDICATION_DIRNAME / f"case_{request_id.split('::')[-1]}.json"


def _human_adjudication_decision_path(root: Path, request_id: str) -> Path:
    return root / HUMAN_ADJUDICATION_DIRNAME / f"case_{request_id.split('::')[-1]}.decision.json"


def _human_adjudication_decisions_path(root: Path) -> Path:
    return root / HUMAN_ADJUDICATION_DIRNAME / HUMAN_ADJUDICATION_DECISIONS_FILENAME


def _read_adjudication_decisions(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def write_json_frozen(path: Path, value: Any) -> None:
    """Write an artifact once; fail closed if an existing artifact would drift."""
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"human adjudication artifact drift: {path}")
    path.write_text(payload, encoding="utf-8")


def _human_adjudication_escalation_terminal(event: dict[str, Any]) -> bool:
    """True when an Anthropic escalation exhausted its budget during thinking."""
    if event.get("state") != "failed" or event.get("error_class") != "ValueError":
        return False
    if event.get("error_message") != ANTHROPIC_NO_TEXT_ERROR_MESSAGE:
        return False
    diagnostics = event.get("error_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    return (
        diagnostics.get("stop_reason") == "max_tokens"
        and diagnostics.get("output_tokens") == HUMAN_ADJUDICATION_OBSERVED_OUTPUT_TOKENS
        and diagnostics.get("text_block_count") == 0
        and diagnostics.get("has_raw_text") is False
        and "thinking" in (diagnostics.get("content_block_types") or [])
    )


def select_human_adjudication_cases(
    rows: list[dict[str, Any]],
    ledger: DurableLedger,
    results: dict[tuple[str, str], dict[str, Any]],
    decisions_path: Path,
    expected_population: int = HUMAN_ADJUDICATION_EXPECTED,
) -> tuple[list[dict[str, Any]], int]:
    """Select unresolved Anthropic Memora escalation cases for human review.

    A case is eligible only when its durable terminal state is an attempt that
    exhausted the full 8192-token budget during thinking with no text and no
    completed result.  Cases with a recorded human decision are never offered
    again.  Any identity drift or population drift fails closed.
    """
    plan = {
        row["formal_validation_request_id"]: row
        for row in rows
        if row.get("provider") == "anthropic"
        and row.get("model") == MODELS["anthropic"]
        and row.get("source") == "memora"
        and row.get("phase") == "escalation"
        and row.get("judge_slot") == ANTHROPIC_ESCALATION_JUDGE_SLOT
    }
    escalation_result_ids = {
        str(row.get("formal_validation_request_id"))
        for row in results.values()
        if row.get("judge_slot") == ANTHROPIC_ESCALATION_JUDGE_SLOT and row.get("parsed_result") is not None
    }
    resolved: set[str] = set()
    for decision in _read_adjudication_decisions(decisions_path):
        rid = str(decision.get("formal_validation_request_id"))
        if rid not in plan:
            raise RuntimeError(f"human adjudication decision recorded on a non-target slot: {rid}")
        resolved.add(rid)
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in ledger.read():
        rid = str(event.get("formal_validation_request_id"))
        if rid not in plan:
            continue
        row = plan[rid]
        for key in FORMAL_LEDGER_IDENTITY_KEYS:
            if str(event.get(key)) != str(row.get(key)):
                raise RuntimeError(f"ledger identity drift: {rid}")
        histories[rid].append(event)
    eligible: list[dict[str, Any]] = []
    for rid, row in plan.items():
        if rid in resolved:
            continue
        history = histories.get(rid, [])
        if not history or rid in escalation_result_ids:
            continue
        if any(event.get("state") == "completed" for event in history):
            continue
        if not _human_adjudication_escalation_terminal(history[-1]):
            continue
        eligible.append({**row, "_human_adjudication_diagnostics": history[-1].get("error_diagnostics")})
    if len(eligible) + len(resolved) != expected_population:
        raise RuntimeError(
            f"human adjudication population drift: expected {expected_population}, "
            f"found {len(eligible)} eligible + {len(resolved)} resolved"
        )
    eligible.sort(key=lambda row: row["formal_validation_request_id"])
    return eligible, len(eligible)


def _blinded_judge_decisions(result: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not result:
        return None
    parsed = result.get("parsed_result")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("decisions"), list):
        return None
    return [
        {
            "criterion_id": str(item.get("criterion_id")),
            "decision": str(item.get("decision")),
            "confidence": str(item.get("confidence")),
        }
        for item in parsed["decisions"]
        if isinstance(item, dict)
    ]


def build_human_adjudication_case(
    row: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
    results: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Build a blinded human-review artifact exposing only judging evidence."""
    payload = payloads[row["observation_id"]]
    prompt = json.loads(payload["prompt"])
    observation_id = row["observation_id"]
    primary = results.get((observation_id, "memora_primary_gemini"))
    secondary = results.get((observation_id, "memora_secondary_openai"))
    diagnostics = row.get("_human_adjudication_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    return {
        "schema_version": HUMAN_ADJUDICATION_SCHEMA_VERSION,
        "resolution_method": HUMAN_ADJUDICATION_RESOLUTION_METHOD,
        "formal_validation_request_id": row["formal_validation_request_id"],
        "observation_id": observation_id,
        "pilot_block_id": row["pilot_block_id"],
        "request_hash": row["request_hash"],
        "payload_hash": row["payload_hash"],
        "protocol_version": row["protocol_version"],
        "evaluator_methodology": row["evaluator_methodology"],
        "source": "memora",
        "judge_slot": row["judge_slot"],
        "unresolved_reason": HUMAN_ADJUDICATION_REASON,
        "unresolved_diagnostics": {
            "stop_reason": diagnostics.get("stop_reason"),
            "output_tokens": diagnostics.get("output_tokens"),
            "text_block_count": diagnostics.get("text_block_count"),
            "has_raw_text": diagnostics.get("has_raw_text"),
            "content_block_types": diagnostics.get("content_block_types"),
            "json_parse_ok": diagnostics.get("json_parse_ok"),
        },
        "human_review": {
            "task": prompt.get("task"),
            "question": prompt.get("question"),
            "criteria": prompt.get("criteria"),
            "existing_model_answer": prompt.get("existing_model_answer"),
            "judge_a_primary_decisions": _blinded_judge_decisions(primary),
            "judge_b_secondary_decisions": _blinded_judge_decisions(secondary),
            "allowed_decisions": {
                "decision": list(HUMAN_ADJUDICATION_ALLOWED_DECISIONS),
                "confidence": list(HUMAN_ADJUDICATION_ALLOWED_CONFIDENCE),
            },
            "decision_schema": {
                "formal_validation_request_id": "must equal the case's id",
                "observation_id": "must equal the case's observation_id",
                "adjudicator": "string identifying the human reviewer",
                "decisions": [{
                    "criterion_id": "must match exactly one case criterion",
                    "decision": "pass|fail",
                    "confidence": "high|medium|low",
                    "short_reason": "non-empty string",
                }],
            },
        },
        "decision": None,
    }


def record_human_adjudication_decision(
    root: Path,
    decision: Any,
    case: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    """Validate and append an explicit human decision; never auto-decide."""
    if not isinstance(decision, dict):
        raise RuntimeError("human adjudication decision must be a JSON object")
    if decision.get("formal_validation_request_id") != row["formal_validation_request_id"]:
        raise RuntimeError("human adjudication decision request id drift")
    if decision.get("observation_id") != row["observation_id"]:
        raise RuntimeError("human adjudication decision observation id drift")
    adjudicator = decision.get("adjudicator")
    if not isinstance(adjudicator, str) or not adjudicator.strip():
        raise RuntimeError("human adjudication decision requires an adjudicator")
    criteria = case.get("human_review", {}).get("criteria") or []
    expected_ids = [str(item.get("criterion_id")) for item in criteria]
    entries = decision.get("decisions")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("human adjudication decision requires a non-empty decisions list")
    observed_ids: list[str] = []
    normalized: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"criterion_id", "decision", "confidence", "short_reason"}:
            raise RuntimeError("human adjudication decision entry has unexpected fields")
        criterion_id = str(entry.get("criterion_id"))
        if criterion_id in observed_ids:
            raise RuntimeError("duplicate human adjudication criterion decision")
        if criterion_id not in expected_ids:
            raise RuntimeError("unknown human adjudication criterion id")
        if entry.get("decision") not in HUMAN_ADJUDICATION_ALLOWED_DECISIONS:
            raise RuntimeError("human adjudication decision must be pass or fail")
        if entry.get("confidence") not in HUMAN_ADJUDICATION_ALLOWED_CONFIDENCE:
            raise RuntimeError("human adjudication confidence is invalid")
        reason = entry.get("short_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise RuntimeError("human adjudication decision requires a short_reason")
        observed_ids.append(criterion_id)
        normalized.append({
            "criterion_id": criterion_id, "decision": entry["decision"],
            "confidence": entry["confidence"], "short_reason": reason,
        })
    if sorted(observed_ids) != sorted(expected_ids):
        raise RuntimeError("human adjudication must decide every case criterion exactly once")
    normalized.sort(key=lambda item: item["criterion_id"])
    record = {
        "schema_version": HUMAN_ADJUDICATION_SCHEMA_VERSION,
        "resolution_method": HUMAN_ADJUDICATION_RESOLUTION_METHOD,
        "formal_validation_request_id": row["formal_validation_request_id"],
        "observation_id": row["observation_id"],
        "pilot_block_id": row["pilot_block_id"],
        "request_hash": row["request_hash"],
        "payload_hash": row["payload_hash"],
        "protocol_version": row["protocol_version"],
        "evaluator_methodology": row["evaluator_methodology"],
        "source": "memora",
        "judge_slot": row["judge_slot"],
        "adjudicator": adjudicator.strip(),
        "case_hash": sha256_value(case),
        "decisions": normalized,
        "timestamp_ns": time.time_ns(),
    }
    append_jsonl(_human_adjudication_decisions_path(root), record)
    write_json_frozen(_human_adjudication_decision_path(root, row["formal_validation_request_id"]), {
        **case, "resolution_method": HUMAN_ADJUDICATION_RESOLUTION_METHOD, "decision": record,
    })
    return record


def execute_gemini_attempt3_recovery(
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    root: Path,
    authorization_path: Path,
    frozen_manifest: list[dict[str, Any]],
) -> int:
    """Execute an explicitly authorized, bounded attempt-3 recovery batch."""
    if not rows:
        return 0
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY missing before any recovery ledger write")
    provider_root = root / "gemini"
    ledger = DurableLedger(provider_root / "judge_execution_ledger.jsonl")
    results_path = provider_root / "judge_results.jsonl"
    results = read_jsonl(results_path) if results_path.exists() else []
    candidates = derive_attempt3_candidates(all_rows, ledger, results, payloads)
    validate_progressive_attempt3_state(frozen_manifest, candidates, ledger, results)
    candidate_by_id = {row["formal_validation_request_id"]: row for row in candidates}
    if any(candidate_by_id.get(row["formal_validation_request_id"]) != row for row in rows):
        raise RuntimeError("selected attempt-3 population is no longer eligible")
    validate_attempt3_authorization(authorization_path, frozen_manifest, rows)
    if len({row["formal_validation_request_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate attempt-3 recovery request identity")
    if {row["formal_validation_request_id"] for row in rows}.intersection(
        str(result["formal_validation_request_id"]) for result in results
    ):
        raise RuntimeError("completed request cannot receive attempt 3")

    lock = provider_root / ".execution.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Gemini provider partition already locked") from exc
    os.close(fd)
    paid = 0
    raw_path = provider_root / "attempt3_raw_responses.jsonl"
    try:
        for row in rows:
            identity = {key: row[key] for key in (
                "formal_validation_request_id", "pilot_block_id", "observation_id", "source", "judge_slot",
                "provider", "model", "request_hash", "payload_hash", "protocol_version", "evaluator_methodology",
            )}
            transport = {
                "recovery_protocol_version": GEMINI_MEMORA_ATTEMPT3_RECOVERY,
                "transport_schema_version": GEMINI_MEMORA_TRANSPORT_SCHEMA_VERSION,
                "output_budget_policy_version": GEMINI_MEMORA_OUTPUT_BUDGET_VERSION,
                "configured_max_output_tokens": int(row["configured_max_output_tokens"]),
                "failure_class": row["failure_class"],
            }
            ledger.append({
                **identity, **transport, "state": "recovery_authorized", "attempt": 2,
                "recovery_attempt": 3, "recovery_reason": f"transport_only_{row['failure_class']}_recovery",
                "timestamp_ns": time.time_ns(),
            })
            ledger.append({**identity, **transport, "state": "started", "attempt": 3, "timestamp_ns": time.time_ns()})
            paid += 1
            client = adapter("gemini", int(row["configured_max_output_tokens"]))
            raw: str | None = None
            try:
                payload = payloads[row["observation_id"]]
                raw = client.call(payload["prompt"], adapter_task(payload["source"]))
                response_hash = sha256_value(raw)
                provider_metadata = getattr(client, "last_metadata", None) or {}
                append_jsonl(raw_path, {
                    "formal_validation_request_id": row["formal_validation_request_id"],
                    "attempt": 3, "response_hash": response_hash, "raw_response": raw,
                    "provider_metadata": provider_metadata,
                })
                parsed = parse_result(raw, payload["source"], payload["prompt"])
                top_level = json.loads(raw)
                safe_metadata = {
                    **provider_metadata,
                    "parsed_top_level_keys": sorted(top_level) if isinstance(top_level, dict) else [],
                    "response_character_count": len(raw),
                }
                append_jsonl(results_path, {
                    **identity, **transport, "attempt": 3, "parsed_result": parsed,
                    "response_hash": response_hash, "provider_metadata": safe_metadata,
                })
                ledger.append({
                    **identity, **transport, "state": "completed", "attempt": 3,
                    "response_hash": response_hash, "provider_metadata": safe_metadata,
                    "timestamp_ns": time.time_ns(),
                })
            except Exception as exc:
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                metadata = getattr(client, "last_metadata", None) or {}
                ledger.append({
                    **identity, **transport, "state": "failed", "attempt": 3,
                    "error_class": type(exc).__name__, "error_code": str(code) if code is not None else None,
                    "error_message": str(exc)[:1000], "provider_metadata": metadata,
                    "response_received": raw is not None, "timestamp_ns": time.time_ns(),
                })
    finally:
        lock.unlink(missing_ok=True)
    return paid


def execute_gemini_attempt4_recovery(
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    root: Path,
    authorization_path: Path,
    attempt3_manifest: list[dict[str, Any]],
    attempt4_manifest: list[dict[str, Any]],
) -> int:
    """Execute only the explicitly authorized schema-compatibility smoke."""
    if len(rows) != 1:
        raise RuntimeError("attempt-4 recovery is restricted to exactly one smoke request")
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY missing before any recovery ledger write")
    provider_root = root / "gemini"
    ledger = DurableLedger(provider_root / "judge_execution_ledger.jsonl")
    results_path = provider_root / "judge_results.jsonl"
    results = read_jsonl(results_path) if results_path.exists() else []
    candidates = derive_attempt4_candidates(all_rows, attempt3_manifest, ledger, results, payloads)
    if candidates != rows or [attempt3_public_record(row) for row in candidates] != attempt4_manifest:
        raise RuntimeError("selected attempt-4 population is no longer eligible")
    validate_attempt4_authorization(authorization_path, attempt4_manifest, rows)

    row = rows[0]
    identity = {key: row[key] for key in (
        "formal_validation_request_id", "pilot_block_id", "observation_id", "source", "judge_slot",
        "provider", "model", "request_hash", "payload_hash", "protocol_version", "evaluator_methodology",
    )}
    transport = {
        "recovery_protocol_version": GEMINI_MEMORA_ATTEMPT4_RECOVERY,
        "transport_schema_version": GEMINI_MEMORA_TRANSPORT_SCHEMA_VERSION,
        "output_budget_policy_version": GEMINI_MEMORA_OUTPUT_BUDGET_VERSION,
        "configured_max_output_tokens": int(row["configured_max_output_tokens"]),
        "failure_class": row["failure_class"],
    }
    lock = provider_root / ".execution.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Gemini provider partition already locked") from exc
    os.close(fd)
    raw: str | None = None
    try:
        ledger.append({
            **identity, **transport, "state": "recovery_authorized", "attempt": 3,
            "recovery_attempt": 4,
            "recovery_reason": "remove_unsupported_gemini_additional_properties_transport_keyword",
            "timestamp_ns": time.time_ns(),
        })
        ledger.append({**identity, **transport, "state": "started", "attempt": 4, "timestamp_ns": time.time_ns()})
        client = adapter("gemini", int(row["configured_max_output_tokens"]))
        try:
            payload = payloads[row["observation_id"]]
            raw = client.call(payload["prompt"], adapter_task(payload["source"]))
            response_hash = sha256_value(raw)
            provider_metadata = getattr(client, "last_metadata", None) or {}
            append_jsonl(provider_root / "attempt4_raw_responses.jsonl", {
                "formal_validation_request_id": row["formal_validation_request_id"],
                "attempt": 4, "response_hash": response_hash, "raw_response": raw,
                "provider_metadata": provider_metadata,
            })
            parsed = parse_result(raw, payload["source"], payload["prompt"])
            top_level = json.loads(raw)
            safe_metadata = {
                **provider_metadata,
                "parsed_top_level_keys": sorted(top_level) if isinstance(top_level, dict) else [],
                "response_character_count": len(raw),
            }
            append_jsonl(results_path, {
                **identity, **transport, "attempt": 4, "parsed_result": parsed,
                "response_hash": response_hash, "provider_metadata": safe_metadata,
            })
            ledger.append({
                **identity, **transport, "state": "completed", "attempt": 4,
                "response_hash": response_hash, "provider_metadata": safe_metadata,
                "timestamp_ns": time.time_ns(),
            })
        except Exception as exc:
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            ledger.append({
                **identity, **transport, "state": "failed", "attempt": 4,
                "error_class": type(exc).__name__, "error_code": str(code) if code is not None else None,
                "error_message": str(exc)[:1000],
                "provider_metadata": getattr(client, "last_metadata", None) or {},
                "response_received": raw is not None, "timestamp_ns": time.time_ns(),
            })
    finally:
        lock.unlink(missing_ok=True)
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=("openai", "anthropic", "gemini", "all"), default="all")
    p.add_argument("--phase", choices=("primary", "secondary", "escalation", "all"), default="all")
    p.add_argument("--plan", type=Path, default=Path("results/formal_300_validation_v1/execution_plan.jsonl"))
    p.add_argument("--payloads", type=Path, default=Path("results/b4_scoring_validation/private/formal_300_validation_v1/semantic_payloads.jsonl"))
    p.add_argument("--private-root", type=Path, default=Path("results/b4_scoring_validation/private/formal_300_validation_v1"))
    p.add_argument("--recovery", action="store_true")
    p.add_argument("--recovery-attempt", type=int)
    p.add_argument("--failure-class", choices=ATTEMPT3_CLASSES)
    p.add_argument("--authorization", type=Path)
    p.add_argument(
        "--attempt3-manifest", type=Path,
        default=Path("results/formal_300_validation_v1/recovery_attempt3/attempt3_recovery_manifest.jsonl"),
    )
    p.add_argument(
        "--attempt4-manifest", type=Path,
        default=Path("results/formal_300_validation_v1/recovery_attempt4/attempt4_recovery_manifest.jsonl"),
    )
    p.add_argument("--max-recovery-requests", type=int, default=1)
    p.add_argument(
        "--recovery-token-limit", action="store_true",
        help="recover the current Anthropic 512-token-truncation failures at 2048 output tokens",
    )
    p.add_argument(
        "--anthropic-token-recovery", choices=ANTHROPIC_TOKEN_RECOVERY_CLASSES, default=None,
        help="explicit Anthropic token-limit recovery class: 512-to-2048 or 2048-to-4096",
    )
    p.add_argument(
        "--openai-token-recovery", choices=OPENAI_TOKEN_RECOVERY_CLASSES, default=None,
        help="OpenAI transport recovery class: 436-to-1024, 2236-to-4096, or 160-to-512",
    )
    p.add_argument("--export-human-adjudication", action="store_true")
    p.add_argument("--record-human-adjudication", type=Path, default=None)
    p.add_argument(
        "--max-requests", type=int, default=None,
        help="cap the number of pending normal-execution requests selected in this run",
    )
    p.add_argument("--execute", action="store_true"); p.add_argument("--confirm-paid-judge-calls", action="store_true")
    a = p.parse_args(argv)
    if a.execute != a.confirm_paid_judge_calls: raise SystemExit("paid execution requires both flags")
    if a.max_requests is not None and a.max_requests < 1:
        raise SystemExit("max requests must be at least one")
    if a.max_requests is not None and (a.recovery or a.recovery_attempt is not None or a.recovery_token_limit
                                       or a.anthropic_token_recovery is not None or a.openai_token_recovery is not None):
        raise SystemExit("--max-requests applies to normal execution; use --max-recovery-requests for recovery")
    static = read_jsonl(a.plan); payloads = {r["observation_id"]: r for r in read_jsonl(a.payloads)}
    if a.export_human_adjudication or a.record_human_adjudication is not None:
        if a.execute or a.confirm_paid_judge_calls:
            raise SystemExit("human adjudication is offline and takes no paid-execution flags")
        if a.export_human_adjudication and a.record_human_adjudication is not None:
            raise SystemExit("choose either --export-human-adjudication or --record-human-adjudication")
        results = result_index(a.private_root); dynamic = dynamic_plan(payloads, results, static)
        all_rows = static + dynamic
        rows = [
            r for r in all_rows
            if r["provider"] == "anthropic" and r["source"] == "memora" and r["phase"] == "escalation"
        ]
        ledger = DurableLedger(a.private_root / "anthropic" / "judge_execution_ledger.jsonl")
        decisions_path = _human_adjudication_decisions_path(a.private_root)
        eligible, available = select_human_adjudication_cases(rows, ledger, results, decisions_path)
        if a.export_human_adjudication:
            print(f"HUMAN_ADJUDICATION_AVAILABLE={available}")
            exported = 0
            for row in eligible:
                case = build_human_adjudication_case(row, payloads, results)
                write_json_frozen(_human_adjudication_case_path(a.private_root, row["formal_validation_request_id"]), case)
                exported += 1
            print(f"HUMAN_ADJUDICATION_EXPORTED={exported}")
            print("PAID_CALLS_PERFORMED=0")
            return 0
        decision_path = a.record_human_adjudication
        if not decision_path.exists():
            raise SystemExit(f"human adjudication decision file not found: {decision_path}")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        rid = str(decision.get("formal_validation_request_id")) if isinstance(decision, dict) else ""
        if not rid:
            raise SystemExit("human adjudication decision requires a formal_validation_request_id")
        row = next((r for r in eligible if r["formal_validation_request_id"] == rid), None)
        if row is None:
            raise SystemExit("human adjudication decision does not match a currently eligible unresolved case")
        case_path = _human_adjudication_case_path(a.private_root, rid)
        if not case_path.exists():
            raise SystemExit("export the human adjudication case before recording a decision")
        case = json.loads(case_path.read_text(encoding="utf-8"))
        record_human_adjudication_decision(a.private_root, decision, case, row)
        print("HUMAN_ADJUDICATION_RECORDED=1")
        print("HUMAN_ADJUDICATION_AVAILABLE=0")
        print("PAID_CALLS_PERFORMED=0")
        return 0
    if a.openai_token_recovery is not None:
        if a.recovery or a.recovery_attempt is not None or a.recovery_token_limit or a.anthropic_token_recovery is not None:
            raise SystemExit("choose exactly one recovery selector")
        if a.provider != "openai":
            raise SystemExit("OpenAI token recovery is restricted to --provider openai")
        if a.max_recovery_requests < 1:
            raise SystemExit("max recovery requests must be at least one")
        results = result_index(a.private_root); dynamic = dynamic_plan(payloads, results, static)
        all_rows = static + dynamic
        rows = [r for r in all_rows if r["provider"] == "openai"]
        ledger = DurableLedger(a.private_root / "openai" / "judge_execution_ledger.jsonl")
        result_path = a.private_root / "openai" / "judge_results.jsonl"
        existing = read_jsonl(result_path) if result_path.exists() else []
        if a.openai_token_recovery == OPENAI_NO_TEXT_RECOVERY_CLASS:
            selected, available = select_openai_no_text_recovery(rows, ledger, existing, a.max_recovery_requests)
            reason, budget = OPENAI_NO_TEXT_FAILURE_CLASS, OPENAI_NO_TEXT_RECOVERY_OUTPUT_TOKENS
        else:
            config = _openai_token_recovery_config(a.openai_token_recovery)
            selected, available = select_openai_token_limit_recovery(
                rows, ledger, existing, a.max_recovery_requests, a.openai_token_recovery,
            )
            reason, budget = config["reason"], config["output_budget"]
        print("RECOVERY_PROVIDER=openai")
        print(f"RECOVERY_CLASS={a.openai_token_recovery}")
        print(f"RECOVERY_REASON={reason}")
        print(f"RECOVERY_AVAILABLE={available}")
        print(f"RECOVERY_SELECTED={len(selected)}")
        print(f"RECOVERY_MAX_REQUESTS={a.max_recovery_requests}")
        if selected:
            events = ledger.read()
            next_attempts = [
                _next_transport_attempt([
                    event for event in events
                    if str(event.get("formal_validation_request_id")) == row["formal_validation_request_id"]
                ])
                for row in selected
            ]
            if len(selected) == 1:
                print(f"RECOVERY_NEXT_ATTEMPT={next_attempts[0]}")
            else:
                histogram = Counter(next_attempts)
                print(f"RECOVERY_NEXT_ATTEMPT_HISTOGRAM={json.dumps({str(k): histogram[k] for k in sorted(histogram)})}")
        print(f"RECOVERY_OUTPUT_BUDGET={budget}")
        if not a.execute:
            print("EXECUTION_MODE=dry_run_only")
            print("PAID_CALLS_PERFORMED=0")
            return 0
        if not selected:
            raise SystemExit("no eligible OpenAI recovery slot remains")
        if a.openai_token_recovery == OPENAI_NO_TEXT_RECOVERY_CLASS:
            paid = execute_openai_no_text_recovery(selected, rows, payloads, a.private_root)
        else:
            paid = execute_openai_token_limit_recovery(
                selected, rows, payloads, a.private_root, a.openai_token_recovery,
            )
        print(f"EXECUTION_MODE=paid_recovery_openai_{a.openai_token_recovery.replace('-','_')}")
        print(f"PAID_CALLS_PERFORMED={paid}")
        return 0
    if a.recovery and a.recovery_attempt is not None:
        raise SystemExit("choose either legacy --recovery or --recovery-attempt")
    recovery_class = a.anthropic_token_recovery
    if a.recovery_token_limit:
        if recovery_class is not None:
            raise SystemExit("choose exactly one recovery selector")
        recovery_class = ANTHROPIC_TOKEN_RECOVERY_512_TO_2048
    if a.recovery_attempt == 5:  # whole-population token-limit batch (not a literal per-slot attempt)
        if a.recovery or a.recovery_token_limit or recovery_class is not None:
            raise SystemExit("choose exactly one recovery selector")
        recovery_class = ANTHROPIC_TOKEN_RECOVERY_512_TO_2048; a.recovery_attempt = None
    if recovery_class is not None:
        if a.recovery or a.recovery_attempt is not None:
            raise SystemExit("choose exactly one recovery selector")
        if a.provider != "anthropic":
            raise SystemExit("Anthropic token recovery is restricted to --provider anthropic")
        if a.max_recovery_requests < 1:
            raise SystemExit("max recovery requests must be at least one")
        config = _token_recovery_config(recovery_class)
        results = result_index(a.private_root); dynamic = dynamic_plan(payloads, results, static)
        all_rows = static + dynamic
        rows = [r for r in all_rows if r["provider"] == "anthropic"]
        ledger = DurableLedger(a.private_root / "anthropic" / "judge_execution_ledger.jsonl")
        result_path = a.private_root / "anthropic" / "judge_results.jsonl"
        existing = read_jsonl(result_path) if result_path.exists() else []
        selected, available = select_anthropic_token_recovery(
            rows, ledger, existing, a.max_recovery_requests, recovery_class,
        )
        print("RECOVERY_PROVIDER=anthropic")
        print(f"RECOVERY_CLASS={recovery_class}")
        print(f"RECOVERY_REASON={config['reason']}")
        print(f"RECOVERY_AVAILABLE={available}")
        print(f"RECOVERY_SELECTED={len(selected)}")
        print(f"RECOVERY_MAX_REQUESTS={a.max_recovery_requests}")
        if selected:
            events = ledger.read()
            next_attempts = []
            for row in selected:
                history = [
                    event for event in events
                    if str(event.get("formal_validation_request_id")) == row["formal_validation_request_id"]
                ]
                next_attempts.append(_next_transport_attempt(history))
            if len(selected) == 1:
                print(f"RECOVERY_NEXT_ATTEMPT={next_attempts[0]}")
            else:
                histogram = Counter(next_attempts)
                print(f"RECOVERY_NEXT_ATTEMPT_HISTOGRAM={json.dumps({str(k): histogram[k] for k in sorted(histogram)})}")
        print(f"RECOVERY_OUTPUT_BUDGET={config['output_budget']}")
        print(f"ANTHROPIC_OUTPUT_BUDGET={config['output_budget']}")
        if not a.execute:
            print("EXECUTION_MODE=dry_run_only")
            print("PAID_CALLS_PERFORMED=0")
            return 0
        if not selected:
            raise SystemExit("no eligible Anthropic token recovery slots remain")
        paid = execute_anthropic_token_recovery(selected, rows, payloads, a.private_root, recovery_class)
        print(f"EXECUTION_MODE=paid_recovery_token_{recovery_class.replace('-','_')}")
        print(f"PAID_CALLS_PERFORMED={paid}")
        return 0
    if a.recovery_attempt is not None:
        if a.recovery_attempt not in {3, 4}:
            raise SystemExit("only explicitly frozen recovery attempts 3 and 4 are supported")
        if a.provider == "anthropic":
            if a.recovery_attempt not in {3, 4}:
                raise SystemExit("Anthropic recovery supports only --recovery-attempt 3 or 4")
            if a.max_recovery_requests != 1:
                raise SystemExit("Anthropic diagnostic retries support exactly one request")
            results = result_index(a.private_root); dynamic = dynamic_plan(payloads, results, static)
            all_rows = static + dynamic
            rows = [r for r in all_rows if r["provider"] == "anthropic" and r["phase"] == "primary"]
            ledger = DurableLedger(a.private_root / "anthropic" / "judge_execution_ledger.jsonl")
            result_path = a.private_root / "anthropic" / "judge_results.jsonl"
            existing = read_jsonl(result_path) if result_path.exists() else []
            if a.recovery_attempt == 4:
                selected, available = select_anthropic_attempt4_candidate(rows, ledger, existing, a.max_recovery_requests)
            else:
                selected, available = select_anthropic_attempt3_candidate(rows, ledger, existing, a.max_recovery_requests)
            print("RECOVERY_PROVIDER=anthropic")
            print(f"RECOVERY_ATTEMPT={a.recovery_attempt}")
            if a.recovery_attempt == 4:
                print("RECOVERY_REASON=token_limit_truncation")
            print(f"RECOVERY_AVAILABLE={available}")
            print(f"RECOVERY_SELECTED={len(selected)}")
            print(f"RECOVERY_MAX_REQUESTS={a.max_recovery_requests}")
            if a.recovery_attempt == 4:
                print(f"ANTHROPIC_OUTPUT_BUDGET={FORMAL_300_ANTHROPIC_MIN_OUTPUT_TOKENS}")
            if not a.execute:
                print("EXECUTION_MODE=dry_run_only")
                print("PAID_CALLS_PERFORMED=0")
                return 0
            if not selected:
                raise SystemExit("no eligible Anthropic recovery slot remains")
            if a.recovery_attempt == 4:
                paid = execute_anthropic_attempt4_recovery(selected, rows, payloads, a.private_root)
            else:
                paid = execute_anthropic_attempt3_recovery(selected, rows, payloads, a.private_root)
            print(f"EXECUTION_MODE=paid_recovery_attempt{a.recovery_attempt}")
            print(f"PAID_CALLS_PERFORMED={paid}")
            return 0
        if a.provider != "gemini":
            raise SystemExit("attempt-3 recovery is restricted to --provider gemini or anthropic")
        if a.failure_class is None:
            raise SystemExit("attempt-3 recovery requires --failure-class")
        if a.max_recovery_requests < 1:
            raise SystemExit("max recovery requests must be at least one")
        ledger = DurableLedger(a.private_root / "gemini" / "judge_execution_ledger.jsonl")
        result_path = a.private_root / "gemini" / "judge_results.jsonl"
        existing = read_jsonl(result_path) if result_path.exists() else []
        candidates = derive_attempt3_candidates(static, ledger, existing, payloads)
        frozen_manifest = load_attempt3_manifest(a.attempt3_manifest)
        try:
            accounting = validate_progressive_attempt3_state(frozen_manifest, candidates, ledger, existing)
        except RuntimeError as exc:
            raise SystemExit(f"attempt-3 frozen-state drift: {exc}") from exc
        print(f"ATTEMPT3_FROZEN_TOTAL={accounting['frozen_total']}")
        print(f"ATTEMPT3_DURABLE_SUCCESS_DIRECT={accounting['durable_success_direct']}")
        print(f"ATTEMPT3_FAILED_UNRESOLVED={accounting['failed_unresolved']}")
        print(f"ATTEMPT3_RESOLVED_BY_LATER_ATTEMPT={accounting['resolved_by_later_attempt']}")
        print(f"ATTEMPT3_NOT_YET_ATTEMPTED={accounting['not_yet_attempted']}")
        print(f"ATTEMPT3_INVALID_ARGUMENT_NOT_YET_ATTEMPTED={accounting['invalid_argument_not_yet_attempted']}")
        print(f"ATTEMPT3_MALFORMED_JSON_NOT_YET_ATTEMPTED={accounting['malformed_json_not_yet_attempted']}")
        print(f"FINAL_DURABLE_SUCCESS_FROM_ATTEMPT3_POPULATION={accounting['final_durable_success']}")
        if a.recovery_attempt == 4:
            if a.failure_class != "invalid_argument":
                raise SystemExit("attempt-4 smoke is restricted to --failure-class invalid_argument")
            attempt4 = derive_attempt4_candidates(static, frozen_manifest, ledger, existing, payloads)
            frozen_attempt4 = load_attempt4_manifest(a.attempt4_manifest)
            if any(attempt3_public_record(row) not in frozen_attempt4 for row in attempt4):
                raise SystemExit("attempt-4 frozen manifest identity drift")
            selected = attempt4[:a.max_recovery_requests]
            print(f"ATTEMPT4_SMOKE_AVAILABLE={len(attempt4)}")
            print(f"FAILURE_CLASS={a.failure_class}")
            print(f"RECOVERY_SELECTED={len(selected)}")
            print(f"RECOVERY_MAX_REQUESTS={a.max_recovery_requests}")
            if not a.execute:
                print("EXECUTION_MODE=dry_run_only")
                print("PAID_CALLS_PERFORMED=0")
                return 0
            if a.authorization is None:
                raise SystemExit("attempt-4 paid recovery requires --authorization")
            paid = execute_gemini_attempt4_recovery(
                selected, static, payloads, a.private_root, a.authorization,
                frozen_manifest, frozen_attempt4,
            )
            print("EXECUTION_MODE=paid_recovery_attempt4")
            print(f"PAID_CALLS_PERFORMED={paid}")
            return 0
        by_class = Counter(row["failure_class"] for row in candidates)
        matching = [row for row in candidates if row["failure_class"] == a.failure_class]
        selected = matching[:a.max_recovery_requests]
        print(f"ATTEMPT3_AVAILABLE_TOTAL={len(candidates)}")
        print(f"ATTEMPT3_INVALID_ARGUMENT_AVAILABLE={by_class['invalid_argument']}")
        print(f"ATTEMPT3_MALFORMED_JSON_AVAILABLE={by_class['malformed_json']}")
        print(f"FAILURE_CLASS={a.failure_class}")
        print(f"RECOVERY_SELECTED={len(selected)}")
        print(f"RECOVERY_MAX_REQUESTS={a.max_recovery_requests}")
        if not a.execute:
            print("EXECUTION_MODE=dry_run_only")
            print("PAID_CALLS_PERFORMED=0")
            return 0
        if a.authorization is None:
            raise SystemExit("attempt-3 paid recovery requires --authorization")
        if not selected:
            raise SystemExit("no eligible attempt-3 requests in selected failure class")
        paid = execute_gemini_attempt3_recovery(
            selected, static, payloads, a.private_root, a.authorization, frozen_manifest,
        )
        print("EXECUTION_MODE=paid_recovery_attempt3")
        print(f"PAID_CALLS_PERFORMED={paid}")
        return 0
    if a.recovery:
        if a.provider == "anthropic":
            results = result_index(a.private_root); dynamic = dynamic_plan(payloads, results, static)
            all_rows = static + dynamic
            rows = [r for r in all_rows if r["provider"] == "anthropic" and r["phase"] == "primary"]
            ledger = DurableLedger(a.private_root / "anthropic" / "judge_execution_ledger.jsonl")
            result_path = a.private_root / "anthropic" / "judge_results.jsonl"
            existing = read_jsonl(result_path) if result_path.exists() else []
            selected, available = select_anthropic_no_text_recovery(rows, ledger, existing, a.max_recovery_requests)
            print("RECOVERY_PROVIDER=anthropic")
            print(f"RECOVERY_FAILURE_CLASS={ANTHROPIC_NO_TEXT_FAILURE_CLASS}")
            print(f"RECOVERY_AVAILABLE={available}")
            print(f"RECOVERY_SELECTED={len(selected)}")
            print(f"RECOVERY_MAX_REQUESTS={a.max_recovery_requests}")
            if not a.execute:
                print("EXECUTION_MODE=dry_run_only")
                print("PAID_CALLS_PERFORMED=0")
                return 0
            if not selected:
                raise SystemExit("no eligible Anthropic no-text recovery slots remain")
            # Revalidate against the frozen plan immediately before the bounded
            # paid batch, then execute only those immutable identities.
            paid = execute_anthropic_no_text_recovery(selected, rows, payloads, a.private_root)
            print("EXECUTION_MODE=paid_recovery")
            print(f"PAID_CALLS_PERFORMED={paid}")
            return 0
        if a.provider != "gemini":
            raise SystemExit("formal recovery supports --provider gemini or anthropic")
        ledger = DurableLedger(a.private_root / "gemini" / "judge_execution_ledger.jsonl")
        result_path = a.private_root / "gemini" / "judge_results.jsonl"
        existing = read_jsonl(result_path) if result_path.exists() else []
        selected, available = select_gemini_memora_recovery(static, ledger, existing, a.max_recovery_requests)
        print(f"FAILED_REQUESTS_AVAILABLE_FOR_RECOVERY={available}")
        print(f"RECOVERY_SELECTED={len(selected)}")
        print(f"RECOVERY_MAX_REQUESTS={a.max_recovery_requests}")
        if not a.execute:
            print("EXECUTION_MODE=dry_run_only")
            print("PAID_CALLS_PERFORMED=0")
            return 0
        if not selected:
            raise SystemExit("no eligible Gemini recovery slots remain")
        # Revalidate against the complete frozen plan immediately before the
        # bounded paid batch, then execute only those immutable identities.
        paid = execute_gemini_batch_recovery(selected, static, payloads, a.private_root)
        print("EXECUTION_MODE=paid_recovery")
        print(f"PAID_CALLS_PERFORMED={paid}")
        return 0
    results = result_index(a.private_root); dynamic = dynamic_plan(payloads, results, static)
    persist_dynamic(a.private_root / "dynamic_execution_plan.jsonl", dynamic)
    all_rows = static + dynamic; providers = list(MODELS) if a.provider == "all" else [a.provider]
    phases = ("primary", "secondary", "escalation") if a.phase == "all" else (a.phase,)
    paid = 0; budget = a.max_requests
    for phase in phases:
        if a.execute and phase != "primary":
            results = result_index(a.private_root); dynamic = dynamic_plan(payloads, results, static)
            persist_dynamic(a.private_root / "dynamic_execution_plan.jsonl", dynamic); all_rows = static + dynamic
        for provider in providers:
            rows = [r for r in all_rows if r["phase"] == phase and r["provider"] == provider]
            if not rows: continue
            ledger = DurableLedger(a.private_root / provider / "judge_execution_ledger.jsonl")
            status = state(rows, ledger, a.execute)
            print(f"{phase.upper()}_{provider.upper()}={json.dumps(status, sort_keys=True)}")
            if provider == "openai" and phase == "primary":
                pending_rows = pending_partition_rows(rows, ledger)
                if pending_rows:
                    planned_dist = Counter(str(r["planned_output_tokens"]) for r in pending_rows)
                    effective_dist = Counter(str(formal_output_budget(
                        "openai", r["planned_output_tokens"],
                        judge_slot=r.get("judge_slot"), source=r.get("source"), phase=r.get("phase"),
                    )) for r in pending_rows)
                    print(f"PENDING_REQUESTS={len(pending_rows)}")
                    print("PLANNED_OUTPUT_BUDGET_DISTRIBUTION=" + json.dumps(
                        {k: planned_dist[k] for k in sorted(planned_dist, key=int)}))
                    print("EFFECTIVE_OUTPUT_BUDGET_DISTRIBUTION=" + json.dumps(
                        {k: effective_dist[k] for k in sorted(effective_dist, key=int)}))
                    print(f"OPENAI_LME_PRIMARY_OUTPUT_FLOOR={FORMAL_300_OPENAI_LME_PRIMARY_MIN_OUTPUT_TOKENS}")
            if provider == "anthropic" and phase == "escalation":
                pending_rows = [
                    r for r in pending_partition_rows(rows, ledger)
                    if r.get("judge_slot") == ANTHROPIC_ESCALATION_JUDGE_SLOT and r.get("source") == "memora"
                ]
                if pending_rows:
                    planned_dist = Counter(str(r["planned_output_tokens"]) for r in pending_rows)
                    effective_dist = Counter(str(formal_output_budget(
                        "anthropic", r["planned_output_tokens"],
                        judge_slot=r.get("judge_slot"), source=r.get("source"), phase=r.get("phase"),
                    )) for r in pending_rows)
                    print(f"PENDING_REQUESTS={len(pending_rows)}")
                    print("PLANNED_OUTPUT_BUDGET_DISTRIBUTION=" + json.dumps(
                        {k: planned_dist[k] for k in sorted(planned_dist, key=int)}))
                    print("EFFECTIVE_OUTPUT_BUDGET_DISTRIBUTION=" + json.dumps(
                        {k: effective_dist[k] for k in sorted(effective_dist, key=int)}))
                    print(f"ANTHROPIC_MEMORA_ESCALATION_OUTPUT_FLOOR={FORMAL_300_ANTHROPIC_ESCALATION_MIN_OUTPUT_TOKENS}")
                    if len(pending_rows) == 1:
                        print(f"PENDING_PLANNED_OUTPUT_TOKENS={int(pending_rows[0]['planned_output_tokens'])}")
                        print("PENDING_EFFECTIVE_OUTPUT_TOKENS=" + str(formal_output_budget(
                            "anthropic", pending_rows[0]["planned_output_tokens"],
                            judge_slot=pending_rows[0].get("judge_slot"),
                            source=pending_rows[0].get("source"), phase=pending_rows[0].get("phase"),
                        )))
            if a.max_requests is None:
                selected = rows
            else:
                pending = pending_partition_rows(rows, ledger); take = min(budget, len(pending))
                selected = pending[:take]; budget -= take
                print(f"REQUESTS_AVAILABLE={len(pending)}")
                print(f"REQUESTS_SELECTED={len(selected)}")
                print(f"REQUESTS_MAX={a.max_requests}")
            if a.execute and selected: paid += execute_partition(provider, selected, payloads, a.private_root)
    print(f"EXECUTION_MODE={'paid' if a.execute else 'dry_run_only'}")
    print(f"PAID_CALLS_PERFORMED={paid}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
