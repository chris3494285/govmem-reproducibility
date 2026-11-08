#!/usr/bin/env python3
"""Freeze the one-request Gemini schema-compatibility attempt-4 recovery."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b4_calibration_execution import DurableLedger
from scripts.prepare_cheap_judge_pilot import read_jsonl
from scripts.prepare_formal_300_validation import EVALUATOR, PROTOCOL
from scripts.run_formal_300_validation import (
    GEMINI_MEMORA_ATTEMPT4_RECOVERY, attempt3_public_record, attempt4_manifest_hash,
    derive_attempt4_candidates, load_attempt3_manifest,
)


PUBLIC = Path("results/formal_300_validation_v1")
PRIVATE = Path("results/b4_scoring_validation/private/formal_300_validation_v1")


def freeze(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"attempt-4 recovery artifact drift: {path}")
    path.write_text(text, encoding="utf-8")


def main() -> int:
    plan = read_jsonl(PUBLIC / "execution_plan.jsonl")
    payloads = {row["observation_id"]: row for row in read_jsonl(PRIVATE / "semantic_payloads.jsonl")}
    ledger = DurableLedger(PRIVATE / "gemini/judge_execution_ledger.jsonl")
    results = read_jsonl(PRIVATE / "gemini/judge_results.jsonl")
    attempt3 = load_attempt3_manifest(PUBLIC / "recovery_attempt3/attempt3_recovery_manifest.jsonl")
    candidates = derive_attempt4_candidates(plan, attempt3, ledger, results, payloads)
    root = PUBLIC / "recovery_attempt4"
    if not candidates and (root / "attempt4_recovery_manifest.jsonl").exists():
        existing = read_jsonl(root / "attempt4_recovery_manifest.jsonl")
        print("ATTEMPT4_SMOKE_AVAILABLE=0")
        print(f"ATTEMPT4_MANIFEST_HASH={attempt4_manifest_hash(existing)}")
        print("PAID_CALLS_PERFORMED=0")
        return 0
    public_rows = [attempt3_public_record(row) for row in candidates]
    manifest_hash = attempt4_manifest_hash(candidates)
    freeze(root / "attempt4_recovery_manifest.jsonl", json.dumps(public_rows[0], sort_keys=True) + "\n")
    freeze(root / "attempt4_recovery_summary.json", json.dumps({
        "protocol_version": PROTOCOL,
        "evaluator_methodology": EVALUATOR,
        "recovery_protocol_version": GEMINI_MEMORA_ATTEMPT4_RECOVERY,
        "attempt4_smoke_available": 1,
        "recovery_manifest_hash": manifest_hash,
        "paid_calls_performed": 0,
    }, indent=2, sort_keys=True) + "\n")
    freeze(root / "attempt4_authorization_template.json", json.dumps({
        "authorization_version": "formal-300-attempt4-manual-authorization-v1",
        "authorized": False,
        "protocol_version": PROTOCOL,
        "evaluator_methodology": EVALUATOR,
        "recovery_protocol_version": GEMINI_MEMORA_ATTEMPT4_RECOVERY,
        "recovery_manifest_hash": manifest_hash,
        "maximum_authorized_retries_per_request": 1,
        "authorized_request_ids": [public_rows[0]["formal_validation_request_id"]],
    }, indent=2, sort_keys=True) + "\n")
    print("ATTEMPT4_SMOKE_AVAILABLE=1")
    print(f"ATTEMPT4_MANIFEST_HASH={manifest_hash}")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
