#!/usr/bin/env python3
"""Prepare public-safe metadata for formal-300 Gemini attempt-3 recovery."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b4_calibration_execution import DurableLedger
from scripts.prepare_cheap_judge_pilot import read_jsonl
from scripts.prepare_formal_300_validation import EVALUATOR, PROTOCOL
from scripts.run_formal_300_validation import (
    GEMINI_MEMORA_ATTEMPT3_RECOVERY,
    GEMINI_MEMORA_OUTPUT_BUDGET_VERSION,
    GEMINI_MEMORA_TRANSPORT_SCHEMA_VERSION,
    attempt3_manifest_hash,
    attempt3_public_record,
    derive_attempt3_candidates,
    load_attempt3_manifest,
    validate_progressive_attempt3_state,
)


def write_frozen(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"attempt-3 recovery artifact drift: {path}")
    path.write_text(text, encoding="utf-8")


def prepare(public_root: Path, private_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = read_jsonl(public_root / "execution_plan.jsonl")
    payloads = {
        row["observation_id"]: row
        for row in read_jsonl(private_root / "semantic_payloads.jsonl")
    }
    result_path = private_root / "gemini" / "judge_results.jsonl"
    results = read_jsonl(result_path) if result_path.exists() else []
    ledger = DurableLedger(private_root / "gemini" / "judge_execution_ledger.jsonl")
    candidates = derive_attempt3_candidates(plan, ledger, results, payloads)
    counts = Counter(row["failure_class"] for row in candidates)
    output_root = public_root / "recovery_attempt3"
    manifest_path = output_root / "attempt3_recovery_manifest.jsonl"
    if manifest_path.exists():
        frozen = load_attempt3_manifest(manifest_path)
        validate_progressive_attempt3_state(frozen, candidates, ledger, results)
        frozen_summary = json.loads((output_root / "attempt3_recovery_summary.json").read_text())
        summary = {
            **frozen_summary,
            "attempt3_available_total": len(candidates),
            "attempt3_invalid_argument_available": counts["invalid_argument"],
            "attempt3_malformed_json_available": counts["malformed_json"],
        }
        return [attempt3_public_record(row) for row in candidates], summary
    if len(candidates) != 51 or counts != Counter({"invalid_argument": 45, "malformed_json": 6}):
        raise RuntimeError(f"attempt-3 recovery population drift: total={len(candidates)} classes={dict(counts)}")
    public_rows = [attempt3_public_record(row) for row in candidates]
    manifest_hash = attempt3_manifest_hash(candidates)
    summary = {
        "protocol_version": PROTOCOL,
        "evaluator_methodology": EVALUATOR,
        "recovery_protocol_version": GEMINI_MEMORA_ATTEMPT3_RECOVERY,
        "transport_schema_version": GEMINI_MEMORA_TRANSPORT_SCHEMA_VERSION,
        "output_budget_policy_version": GEMINI_MEMORA_OUTPUT_BUDGET_VERSION,
        "output_budget_formula": "min(6144, max(1024, 256 + 80 * criterion_count))",
        "output_budget_evidence": {
            "observed_successful_max_estimated_tokens_per_criterion": 69.67,
            "per_criterion_allowance": 80,
            "framing_allowance": 256,
            "minimum": 1024,
            "maximum": 6144,
            "maximum_source": "repository Gemini Flash repair ceiling",
        },
        "attempt3_available_total": len(candidates),
        "attempt3_invalid_argument_available": counts["invalid_argument"],
        "attempt3_malformed_json_available": counts["malformed_json"],
        "completed_results_excluded": len(results),
        "recovery_manifest_hash": manifest_hash,
        "semantic_methodology_changed": False,
        "model_changed": False,
        "answer_generation_calls": 0,
        "paid_calls_performed": 0,
    }
    authorization = {
        "authorization_version": "formal-300-attempt3-manual-authorization-v1",
        "authorized": False,
        "protocol_version": PROTOCOL,
        "evaluator_methodology": EVALUATOR,
        "recovery_protocol_version": GEMINI_MEMORA_ATTEMPT3_RECOVERY,
        "recovery_manifest_hash": manifest_hash,
        "maximum_authorized_retries_per_request": 1,
        "authorized_request_ids": [row["formal_validation_request_id"] for row in public_rows],
        "instructions": "Set authorized=true only after manual review; optionally narrow authorized_request_ids.",
    }
    write_frozen(
        output_root / "attempt3_recovery_manifest.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in public_rows),
    )
    write_frozen(
        output_root / "attempt3_recovery_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    write_frozen(
        output_root / "attempt3_authorization_template.json",
        json.dumps(authorization, indent=2, sort_keys=True) + "\n",
    )
    return public_rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=Path("results/formal_300_validation_v1"))
    parser.add_argument(
        "--private-root", type=Path,
        default=Path("results/b4_scoring_validation/private/formal_300_validation_v1"),
    )
    args = parser.parse_args(argv)
    _rows, summary = prepare(args.public_root, args.private_root)
    print(f"ATTEMPT3_AVAILABLE_TOTAL={summary['attempt3_available_total']}")
    print(f"ATTEMPT3_INVALID_ARGUMENT_AVAILABLE={summary['attempt3_invalid_argument_available']}")
    print(f"ATTEMPT3_MALFORMED_JSON_AVAILABLE={summary['attempt3_malformed_json_available']}")
    print(f"RECOVERY_MANIFEST_HASH={summary['recovery_manifest_hash']}")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
