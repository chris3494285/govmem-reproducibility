#!/usr/bin/env python3
"""Freeze and optionally execute the 24-case B4 Gemini escalation phase."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.providers.gemini_provider import DEFAULT_GEMINI_MODEL
from scripts.b4_calibration_execution import (
    DurableLedger,
    LedgerConflictError,
    classify_input_limits,
    execute_slots,
    prompt_hash,
    request_slot_id,
    resume_plan,
    validate_paid_input_limits,
    validate_provider_credentials,
)
from scripts.prepare_b4_private_payloads import read_jsonl

PERSISTENT_FAILURE_ID = "longmemeval::gpt4_cd90e484::5784a907e475fa49"
THIRD_PROVIDER = "gemini"
THIRD_MODEL = DEFAULT_GEMINI_MODEL
EXPECTED_THIRD_MODEL = "gemini-3-flash-preview"
VERIFIED_GEMINI_MAX_INPUT_TOKENS = 1_048_576
GEMINI_LIMIT_SOURCE = "https://ai.google.dev/gemini-api/docs/models/gemini-3-flash-preview"
PROMPT_RUBRIC_VERSION = "semantic-prompt-v1"
EXPECTED_DISAGREEMENTS = 23
EXPECTED_PERSISTENT_FAILURES = 1
EXPECTED_TOTAL = 24


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise LedgerConflictError(f"frozen escalation artifact drift: {path}")
    path.write_text(payload, encoding="utf-8")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise LedgerConflictError(f"frozen escalation artifact drift: {path}")
    path.write_text(payload, encoding="utf-8")


def select_escalation_cases(
    primary_plan: list[dict[str, Any]],
    primary_prompts: list[dict[str, Any]],
    primary_events: list[dict[str, Any]],
    *,
    persistent_failure_id: str = PERSISTENT_FAILURE_ID,
    expected_disagreements: int = EXPECTED_DISAGREEMENTS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plan_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary_plan:
        plan_by_case[str(row["blinded_request_id"])].append(row)
    prompts_by_case: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for line_number, row in enumerate(primary_prompts, 1):
        prompts_by_case[str(row["blinded_request_id"])].append((line_number, row))
    latest: dict[str, dict[str, Any]] = {}
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in primary_events:
        slot_id = str(event["request_slot_id"])
        latest[slot_id] = event
        histories[slot_id].append(event)

    records: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for blinded_id, plan_rows in sorted(plan_by_case.items()):
        if len(plan_rows) != 2 or {row["judge_slot"] for row in plan_rows} != {"A", "B"}:
            raise LedgerConflictError(f"primary case does not have exactly A/B slots: {blinded_id}")
        prompt_entries = prompts_by_case.get(blinded_id, [])
        if len(prompt_entries) != 2 or {row["judge_slot"] for _, row in prompt_entries} != {"A", "B"}:
            raise LedgerConflictError(f"primary prompt plan lacks exact A/B rows: {blinded_id}")
        prompts = {str(row["prompt"]) for _, row in prompt_entries}
        if len(prompts) != 1:
            raise LedgerConflictError(f"primary judges did not receive identical semantic prompts: {blinded_id}")
        semantic_prompt = next(iter(prompts))
        decisions: dict[str, str] = {}
        for row in plan_rows:
            event = latest.get(str(row["request_slot_id"]), {})
            if event.get("state") == "completed":
                decisions[str(row["judge_slot"])] = str(event["parsed_result"]["decision"])
        disagreement = len(decisions) == 2 and decisions["A"] != decisions["B"]
        persistent = blinded_id == persistent_failure_id
        if not disagreement and not persistent:
            continue
        if disagreement and persistent:
            raise LedgerConflictError("persistent failure case unexpectedly has two disagreeing completed decisions")
        if persistent:
            if len(decisions) != 1 or "A" not in decisions:
                raise LedgerConflictError("persistent failure must retain exactly the OpenAI primary decision")
            b_row = next(row for row in plan_rows if row["judge_slot"] == "B")
            b_history = histories.get(str(b_row["request_slot_id"]), [])
            if not b_history or any(event.get("state") == "completed" for event in b_history):
                raise LedgerConflictError("persistent Claude slot unexpectedly has a completed result")
            failure = b_history[-1]
            if not (
                failure.get("state") in {"provider_error", "permanently_failed"}
                and failure.get("error_class") == "ValueError"
                and failure.get("error_message") == "provider response contained no text"
            ):
                raise LedgerConflictError("persistent Claude failure state or error has drifted")
            reason = "persistent_primary_execution_failure"
        else:
            reason = "primary_judge_disagreement"
        exemplar = plan_rows[0]
        third_hash = prompt_hash(semantic_prompt)
        third_slot_id = request_slot_id(blinded_id, "T", THIRD_PROVIDER, THIRD_MODEL, third_hash)
        records.append({
            "blinded_request_id": blinded_id,
            "task": exemplar["task"],
            "subtype": exemplar.get("subtype"),
            "primary_gpt_decision": decisions.get("A"),
            "primary_claude_decision": decisions.get("B"),
            "escalation_reason": reason,
            "semantic_prompt_source": {
                "path": "results/b4_scoring_validation/private/full_frontier_judges/full_validation_prompt_plan.jsonl",
                "line_numbers": sorted(line for line, _ in prompt_entries),
                "blinded_request_id": blinded_id,
            },
            "third_judge_prompt_hash": third_hash,
            "third_judge_request_slot_id": third_slot_id,
            "third_judge_provider": THIRD_PROVIDER,
            "third_judge_model": THIRD_MODEL,
        })
        prompt_rows.append({
            "blinded_request_id": blinded_id,
            "task": exemplar["task"],
            "subtype": exemplar.get("subtype"),
            "judge_slot": "T",
            "prompt": semantic_prompt,
            "prompt_format_version": PROMPT_RUBRIC_VERSION,
        })

    disagreement_n = sum(row["escalation_reason"] == "primary_judge_disagreement" for row in records)
    persistent_n = sum(row["escalation_reason"] == "persistent_primary_execution_failure" for row in records)
    if disagreement_n != expected_disagreements or persistent_n != 1 or len(records) != expected_disagreements + 1:
        raise LedgerConflictError(
            f"escalation cohort mismatch: disagreements={disagreement_n}, persistent={persistent_n}, total={len(records)}"
        )
    if len({row["blinded_request_id"] for row in records}) != len(records):
        raise LedgerConflictError("duplicate blinded request IDs in escalation cohort")
    return records, prompt_rows


def build_escalation_slots(prompt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slots = []
    for row in prompt_rows:
        req_hash = prompt_hash(str(row["prompt"]))
        slots.append({
            "request_slot_id": request_slot_id(str(row["blinded_request_id"]), "T", THIRD_PROVIDER, THIRD_MODEL, req_hash),
            "blinded_request_id": row["blinded_request_id"],
            "task": row["task"],
            "subtype": row.get("subtype"),
            "judge_slot": "T",
            "judge_provider": THIRD_PROVIDER,
            "judge_model": THIRD_MODEL,
            "semantic_payload_hash": req_hash,
            "request_hash": req_hash,
            "prompt_format_version": PROMPT_RUBRIC_VERSION,
            "prompt": row["prompt"],
        })
    if len({slot["request_slot_id"] for slot in slots}) != len(slots):
        raise LedgerConflictError("duplicate escalation request-slot IDs")
    return slots


def validate_escalation_input_limit(model: str, configured_limit: int | None) -> int:
    """Require the exact first-party model limit before escalation preflight passes."""
    if model != EXPECTED_THIRD_MODEL:
        raise LedgerConflictError(
            f"third-judge model drift: expected {EXPECTED_THIRD_MODEL}, found {model}"
        )
    if configured_limit is None:
        raise ValueError(
            "frontier escalation requires --max-input-tokens with the verified Gemini model limit"
        )
    if configured_limit != VERIFIED_GEMINI_MAX_INPUT_TOKENS:
        raise ValueError(
            "Gemini input-limit drift: expected verified limit "
            f"{VERIFIED_GEMINI_MAX_INPUT_TOKENS}, found {configured_limit}"
        )
    return configured_limit


def prompt_size_summary(classified: list[dict[str, Any]]) -> dict[str, Any]:
    if not classified:
        raise ValueError("cannot summarize an empty escalation cohort")
    ordered_chars = sorted(len(str(row["prompt"])) for row in classified)
    largest = max(classified, key=lambda row: len(str(row["prompt"])))
    token_estimates = [int(row["estimated_tokens"]) for row in classified]
    p90_index = max(0, math.ceil(0.90 * len(ordered_chars)) - 1)
    return {
        "request_n": len(classified),
        "characters": {
            "min": ordered_chars[0],
            "median": statistics.median(ordered_chars),
            "mean": statistics.fmean(ordered_chars),
            "p90_nearest_rank": ordered_chars[p90_index],
            "max": ordered_chars[-1],
        },
        "conservative_input_token_estimate": {
            "method": sorted({str(row["method"]) for row in classified}),
            "total": sum(token_estimates),
            "max": max(token_estimates),
        },
        "largest_request_blinded_id": largest["blinded_request_id"],
    }


def verify_frozen_execution_plan(path: Path, classified: list[dict[str, Any]]) -> None:
    """Verify immutable request identity while leaving prior preflight metadata untouched."""
    immutable = (
        "request_slot_id", "blinded_request_id", "task", "subtype", "judge_slot",
        "judge_provider", "judge_model", "semantic_payload_hash", "request_hash",
        "prompt_format_version",
    )
    expected = [{key: row.get(key) for key in immutable} for row in classified]
    if not path.exists():
        _write_jsonl(path, [{key: value for key, value in row.items() if key != "prompt"} for row in classified])
        return
    actual_rows = read_jsonl(path)
    actual = [{key: row.get(key) for key in immutable} for row in actual_rows]
    if actual != expected:
        raise LedgerConflictError(f"frozen escalation execution identity drift: {path}")


def _assert_private_output(path: Path) -> None:
    private_root = Path("results/b4_scoring_validation/private").resolve()
    try:
        path.resolve().relative_to(private_root)
    except ValueError as exc:
        raise SystemExit("Escalation output must remain under the private B4 directory") from exc
    if "results/b4_scoring_validation/private/" not in Path(".gitignore").read_text(encoding="utf-8").splitlines():
        raise SystemExit("Private B4 directory is not gitignored")


def _write_preflight(
    path: Path,
    metadata_path: Path,
    manifest: dict[str, Any],
    classified: list[dict[str, Any]],
    resume: dict[str, Any],
) -> None:
    blocker_n = sum(row["input_limit_status"] != "ready" for row in classified)
    sizes = prompt_size_summary(classified)
    paid_command = (
        "python scripts/prepare_b4_frontier_escalation.py "
        f"--max-input-tokens {VERIFIED_GEMINI_MAX_INPUT_TOKENS} "
        "--execute --confirm-paid-judge-calls"
    )
    metadata = {
        "execution": "dry_run_only",
        "total_escalation_cases": manifest["total_escalation_cases"],
        "primary_disagreement_cases": manifest["disagreement_cases"],
        "persistent_primary_failure_cases": manifest["persistent_failure_cases"],
        "third_judge_provider": manifest["third_judge_provider"],
        "third_judge_model": manifest["third_judge_model"],
        "configured_max_input_tokens": VERIFIED_GEMINI_MAX_INPUT_TOKENS,
        "input_limit_source": GEMINI_LIMIT_SOURCE,
        "input_limit_blockers": blocker_n,
        "pending_escalation_slots": len(resume["pending"]),
        "completed_escalation_slots": len(resume["completed"]),
        "blocked_escalation_slots": len(resume["blocked"]),
        "estimated_new_paid_calls": len(resume["pending"]),
        "paid_calls_performed": 0,
        "prompt_size_summary": sizes,
        "future_paid_execution_command": paid_command,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# B4 Frontier Escalation Preflight",
        "",
        "Private, offline-only preflight. It contains no benchmark or model-answer text.",
        "",
        f"- total escalation cases: {manifest['total_escalation_cases']}",
        f"- primary disagreement cases: {manifest['disagreement_cases']}",
        f"- persistent primary failure cases: {manifest['persistent_failure_cases']}",
        f"- third judge: {manifest['third_judge_provider']} / {manifest['third_judge_model']}",
        f"- completed escalation slots: {len(resume['completed'])}",
        f"- pending escalation slots: {len(resume['pending'])}",
        f"- blocked escalation slots: {len(resume['blocked'])}",
        f"- input-limit blockers/unverified: {blocker_n}",
        f"- verified input limit: {VERIFIED_GEMINI_MAX_INPUT_TOKENS:,} tokens",
        f"- input-limit source: {GEMINI_LIMIT_SOURCE}",
        f"- prompt characters min/median/mean/p90/max: {sizes['characters']['min']} / {sizes['characters']['median']} / {sizes['characters']['mean']:.2f} / {sizes['characters']['p90_nearest_rank']} / {sizes['characters']['max']}",
        f"- conservative estimated input tokens, total/max: {sizes['conservative_input_token_estimate']['total']} / {sizes['conservative_input_token_estimate']['max']}",
        f"- largest request blinded ID: `{sizes['largest_request_blinded_id']}`",
        "- maximum attempts per slot: 1",
        "- paid calls performed during preparation: 0",
        "",
        "Model choice: `gemini-3-flash-preview` is the repository's sole canonical supported Gemini model and existing final-main integration model.",
        "",
        "Future paid execution command (not run):",
        "",
        f"`{paid_command}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary() -> None:
    print("TOTAL_ESCALATION_CASES=24")
    print("PRIMARY_DISAGREEMENT_CASES=23")
    print("PERSISTENT_PRIMARY_FAILURE_CASES=1")
    print(f"THIRD_JUDGE_PROVIDER={THIRD_PROVIDER}")
    print(f"THIRD_JUDGE_MODEL={THIRD_MODEL}")
    print("INPUT_LIMIT_BLOCKERS=0")
    print("PENDING_ESCALATION_SLOTS=24")
    print("ESTIMATED_NEW_PAID_CALLS=24")
    print("PAID_CALLS_PERFORMED=0")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, default=Path("results/b4_scoring_validation/private/full_frontier_judges"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/b4_scoring_validation/private/frontier_escalation"))
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid-judge-calls", action="store_true")
    args = parser.parse_args(argv)
    if args.execute != args.confirm_paid_judge_calls:
        raise SystemExit("Paid execution requires BOTH --execute and --confirm-paid-judge-calls")
    try:
        verified_limit = validate_escalation_input_limit(THIRD_MODEL, args.max_input_tokens)
    except (LedgerConflictError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    _assert_private_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary_plan_path = args.primary_dir / "full_validation_execution_plan.jsonl"
    primary_prompt_path = args.primary_dir / "full_validation_prompt_plan.jsonl"
    primary_ledger_path = args.primary_dir / "judge_execution_ledger.jsonl"
    primary_plan = read_jsonl(primary_plan_path)
    primary_prompts = read_jsonl(primary_prompt_path)
    primary_events = read_jsonl(primary_ledger_path)
    records, prompt_rows = select_escalation_cases(primary_plan, primary_prompts, primary_events)
    slots = build_escalation_slots(prompt_rows)
    by_task = Counter(row["task"] for row in records)
    by_subtype = Counter(str(row.get("subtype") or "none") for row in records)
    manifest = {
        "schema_version": "b4-frontier-escalation-manifest-v1",
        "total_escalation_cases": len(records),
        "disagreement_cases": sum(row["escalation_reason"] == "primary_judge_disagreement" for row in records),
        "persistent_failure_cases": sum(row["escalation_reason"] == "persistent_primary_execution_failure" for row in records),
        "by_task": dict(sorted(by_task.items())),
        "by_subtype": dict(sorted(by_subtype.items())),
        "third_judge_provider": THIRD_PROVIDER,
        "third_judge_model": THIRD_MODEL,
        "prompt_rubric_version": PROMPT_RUBRIC_VERSION,
        "source_primary_phase": "full_frontier_judges",
        "source_primary_plan_sha256": file_sha256(primary_plan_path),
        "source_primary_prompt_plan_sha256": file_sha256(primary_prompt_path),
        "source_primary_ledger_sha256": file_sha256(primary_ledger_path),
        "source_cohort_sha256": digest(records),
        "cases": records,
        "semantic_decisions": "not_executed",
    }
    _write_json(args.output_dir / "escalation_manifest.json", manifest)
    _write_jsonl(args.output_dir / "escalation_prompt_plan.jsonl", prompt_rows)
    classified = classify_input_limits(slots, {"T": verified_limit})
    try:
        validate_paid_input_limits(classified, {"T": verified_limit})
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    verify_frozen_execution_plan(args.output_dir / "escalation_execution_plan.jsonl", classified)
    templates = [{
        **{key: slot[key] for key in ("request_slot_id", "blinded_request_id", "task", "subtype", "judge_slot", "judge_provider", "judge_model", "request_hash", "prompt_format_version")},
        "decision": None, "confidence": None, "short_reason": None, "parse_status": "not_executed", "response_hash": None,
    } for slot in slots]
    _write_jsonl(args.output_dir / "escalation_result_template.jsonl", templates)

    ledger = DurableLedger(args.output_dir / "judge_execution_ledger.jsonl")
    if not ledger.path.exists():
        for slot in slots:
            ledger.append({
                **{key: slot[key] for key in ("request_slot_id", "blinded_request_id", "task", "judge_slot", "judge_provider", "judge_model", "request_hash")},
                "state": "planned", "attempt": 0,
            })
    resume = resume_plan(slots, ledger, max_attempts=1)
    _write_preflight(
        args.output_dir / "ESCALATION_PREFLIGHT.md",
        args.output_dir / "escalation_preflight.json",
        manifest,
        classified,
        resume,
    )
    print_summary()
    if args.execute:
        validate_paid_input_limits(classified, {"T": verified_limit})
        validate_provider_credentials(slots)
        if resume["blocked"]:
            raise SystemExit(f"Escalation execution blocked by {len(resume['blocked'])} ledger slots")
        result = execute_slots(slots, ledger=ledger, max_attempts=1, max_output_tokens=args.max_output_tokens)
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
