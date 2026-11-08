#!/usr/bin/env python3
"""Freeze the final B4 evaluator and prepare a label-blind holdout plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.providers.openai_provider import DEFAULT_OPENAI_MODEL
from eval.providers.gemini_provider import DEFAULT_GEMINI_MODEL
from scripts.b4_calibration_execution import estimate_input_tokens
from scripts.b4_calibration_runner import build_lme_prompt, build_memora_prompt
from scripts.prepare_final_judge_selection_calibration import digest, read_jsonl, write_json, write_jsonl


PROTOCOL_VERSION = "final-native-evaluation-v2"
FINAL_ANTHROPIC_MODEL = "claude-sonnet-5"
HOLDOUT_N = 200
LME_N = 32
MEMORA_N = 168
LIMITS = {"openai": 1_050_000, "anthropic": 1_000_000, "gemini": 1_000_000}
LIMIT_SOURCES = {
    "openai": "frozen B4 frontier input-limit configuration",
    "anthropic": "frozen B4 frontier input-limit configuration",
    "gemini": "repository-verified B4 Gemini capability configuration",
}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_payloads(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            identifier = str(row["blinded_request_id"])
            if identifier in result:
                raise ValueError(f"duplicate payload ID: {identifier}")
            result[identifier] = row
    return result


def build_base_plan(holdout_path: Path, lme_payload_path: Path, memora_payload_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    holdout = read_jsonl(holdout_path)
    if len(holdout) != HOLDOUT_N or len({row["blinded_request_id"] for row in holdout}) != HOLDOUT_N:
        raise ValueError("locked holdout must contain exactly 200 unique identities")
    if any(row.get("holdout_status") != "locked_labels_not_exported" for row in holdout):
        raise ValueError("holdout status drift")
    counts = {task: sum(row.get("task") == task for row in holdout) for task in ("longmemeval", "memora")}
    if counts != {"longmemeval": LME_N, "memora": MEMORA_N}:
        raise ValueError(f"holdout task counts drift: {counts}")
    payloads = load_payloads([lme_payload_path, memora_payload_path])
    candidates = {
        "longmemeval": [("openai", DEFAULT_OPENAI_MODEL), ("anthropic", FINAL_ANTHROPIC_MODEL)],
        "memora": [("gemini", DEFAULT_GEMINI_MODEL)],
    }
    slots = []
    for holdout_row in holdout:
        identifier = str(holdout_row["blinded_request_id"])
        payload = payloads.get(identifier)
        if payload is None or digest(payload) != holdout_row["payload_sha256"]:
            raise ValueError(f"payload hash/identity drift: {identifier}")
        prompt = build_lme_prompt(payload) if holdout_row["task"] == "longmemeval" else build_memora_prompt(payload)
        if digest(prompt) != holdout_row["prompt_sha256"]:
            raise ValueError(f"frozen prompt hash drift: {identifier}")
        for provider, model in candidates[holdout_row["task"]]:
            measured = estimate_input_tokens(prompt, model)
            limit = LIMITS[provider]
            if measured["estimated_tokens"] > limit:
                raise ValueError(f"input-limit blocker: {identifier} {provider}")
            request_hash = digest(prompt)
            slot_id = "b4-final-v2-slot::" + digest({"protocol_version": PROTOCOL_VERSION, "blinded_request_id": identifier, "provider": provider, "model": model, "request_hash": request_hash})
            slots.append({
                "request_slot_id": slot_id, "blinded_request_id": identifier,
                "task": holdout_row["task"], "subtype": holdout_row.get("subtype"),
                "provider": provider, "model": model, "protocol_version": PROTOCOL_VERSION,
                "request_hash": request_hash, "payload_sha256": holdout_row["payload_sha256"],
                "prompt_sha256": holdout_row["prompt_sha256"], "private_payload_locator": str((lme_payload_path if holdout_row["task"] == "longmemeval" else memora_payload_path)) + "#" + identifier,
                "prompt_chars": len(prompt), "estimated_input_tokens": measured["estimated_tokens"], "input_token_estimation_method": measured["method"],
                "max_input_tokens": limit, "input_limit_source": LIMIT_SOURCES[provider], "max_attempts": 1, "automatic_retries": False,
                "execution_status": "planned_not_executed", "holdout_touched": False,
            })
    if len(slots) != 232 or len({row["request_slot_id"] for row in slots}) != 232:
        raise ValueError("base holdout slot cardinality/uniqueness failure")
    metadata = {"holdout_sha256": file_hash(holdout_path), "lme_payload_sha256": file_hash(lme_payload_path), "memora_payload_sha256": file_hash(memora_payload_path), "holdout_gold_read": False}
    return sorted(slots, key=lambda row: row["request_slot_id"]), metadata


def protocol_text() -> str:
    return f"""# Final B4 evaluation protocol v2

Protocol version: `{PROTOCOL_VERSION}`.

## Frozen architecture

- OAKS remains the native deterministic evaluator.
- Memora uses `gemini-3-flash-preview` as the provisional primary. Targeted
  blinded reliability/risk checks and third-judge escalation follow the frozen
  final-native-evaluation-design-v1 semantics; unresolved cases go to human
  fallback.
- LongMemEval uses both `{DEFAULT_OPENAI_MODEL}` and `{FINAL_ANTHROPIC_MODEL}`
  as blinded independent primary judges on every case. Binary agreement resolves
  a case; binary disagreement invokes Gemini as the third judge; unresolved cases
  require blinded human adjudication.

## Holdout and gates

The pre-existing locked holdout has 200 cases: 32 LongMemEval and 168 Memora.
It is a split from the 604-case adjudicated reference, not a new external gold
set. It tests generalization of the frozen architecture, not independent
benchmark provenance. No routing or model changes are allowed after observing
holdout results.

Memora, LongMemEval, and overall agreement must each be at least 0.90;
overall 0.95 is preferred. Report kappa, confusion matrices, unresolved and
failure rates, LME dual-primary agreement, and Gemini escalation rate.

Memora Gemini self-reference was separately audited on reference-independent
calibration cases (0.9665653495440729). Gemini, DeepSeek, and Kimi all failed
the frozen single-judge LongMemEval 0.90 gate, so LongMemEval uses dual
frontier judges rather than continuing low-cost candidate search.

## Execution safety

The base plan contains 168 Gemini Memora slots and 64 LongMemEval primary slots
(32 OpenAI plus 32 Anthropic). Gemini LME escalation is conditional on binary
primary disagreement and is capped at 32 additional calls. The known base is
232 calls; maximum additional escalation is 32. Execution requires both
`--execute` and `--confirm-paid-judge-calls`, uses separate append-only ledgers,
no automatic retries, and fails closed on identity, hash, protocol, or
started-only drift. Holdout labels are not read during protocol freezing.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=Path("results/final_native_evaluation_design/judge_selection/locked_holdout_cases.jsonl"))
    parser.add_argument("--lme-payload", type=Path, default=Path("results/b4_scoring_validation/private/longmemeval_judge_execution.jsonl"))
    parser.add_argument("--memora-payload", type=Path, default=Path("results/b4_scoring_validation/private/memora_judge_execution.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/final_native_evaluation_design/final_b4_v2"))
    args = parser.parse_args(argv)
    slots, hashes = build_base_plan(args.holdout, args.lme_payload, args.memora_payload)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "locked_holdout_execution_plan.jsonl", slots)
    write_json(args.output / "final_b4_manifest.json", {"protocol_version": PROTOCOL_VERSION, "holdout_total": HOLDOUT_N, "holdout_by_task": {"longmemeval": LME_N, "memora": MEMORA_N}, "known_base_paid_calls": 232, "max_additional_escalation_calls": 32, "candidate_search_continued": False, "xai_integration_added": False, "holdout_gold_read": False, "paid_calls_performed": 0, "source_hashes": hashes, "slot_n": len(slots), "slot_hash": digest(slots)})
    (args.output / "FINAL_B4_EVALUATION_PROTOCOL.md").write_text(protocol_text(), encoding="utf-8")
    (args.output / "HOLDOUT_VALIDATION_PROTOCOL.md").write_text("# Holdout validation protocol\n\n" + protocol_text(), encoding="utf-8")
    print("FINAL_PROTOCOL_VERSION=final-native-evaluation-v2")
    print("HOLDOUT_TOTAL=200")
    print("HOLDOUT_MEMORA=168")
    print("HOLDOUT_LONGMEMEVAL=32")
    print("KNOWN_BASE_PAID_CALLS=232")
    print("MAX_ADDITIONAL_ESCALATION_CALLS=32")
    print("HOLDOUT_CASES_TOUCHED=0")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
