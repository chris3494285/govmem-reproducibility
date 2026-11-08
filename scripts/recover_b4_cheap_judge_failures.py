#!/usr/bin/env python3
"""Audit or append-authorize the known B4 cheap-judge no-completion failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b4_calibration_execution import (
    DurableLedger,
    append_recovery_authorizations,
    build_slots,
    classify_safe_recovery,
    missing_provider_credentials,
)
from scripts.prepare_b4_private_payloads import read_jsonl


def recovery_preflight(phase_dir: Path) -> tuple[dict, list[dict], DurableLedger]:
    manifest = json.loads((phase_dir / "calibration_manifest.json").read_text(encoding="utf-8"))
    config = manifest["judge_architecture"]
    prompts = read_jsonl(phase_dir / "calibration_prompt_plan.jsonl")
    slots = build_slots(prompts, **config)
    ledger = DurableLedger(phase_dir / "judge_execution_ledger.jsonl")
    classification = classify_safe_recovery(slots, ledger.read())
    provider_safe = Counter(item["slot"]["judge_provider"] for item in classification["safe"])
    summary = {
        "schema_version": "b4-cheap-judge-recovery-preflight-v1",
        "mode": "dry_run",
        "historical_event_n": classification["historical_event_n"],
        "historical_slot_n": classification["historical_slot_n"],
        "current_repaired_slot_n": len(slots),
        "safe_to_retry_n": len(classification["safe"]),
        "safe_to_retry_by_provider": dict(sorted(provider_safe.items())),
        "already_authorized_n": len(classification["already_authorized"]),
        "completed_n": len(classification["completed"]),
        "ambiguous_n": len(classification["ambiguous"]),
        "manual_review_n": len(classification["manual"]),
        "missing_credentials": missing_provider_credentials(slots),
        "frozen_counts": {
            "longmemeval": manifest["longmemeval"]["sample_n"],
            "memora": manifest["memora"]["sample_n"],
            "semantic_cases": manifest["longmemeval"]["sample_n"] + manifest["memora"]["sample_n"],
            "request_slots": len(slots),
        },
        "models": {
            "A": [config["judge_a_provider"], config["judge_a_model"]],
            "B": [config["judge_b_provider"], config["judge_b_model"]],
        },
    }
    return summary, slots, ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorize", action="store_true", help="append recovery_authorized events for strictly recognized failures")
    args = parser.parse_args(argv)
    summary, slots, ledger = recovery_preflight(args.phase_dir)
    classification = classify_safe_recovery(slots, ledger.read())
    if args.authorize:
        if classification["completed"] or classification["ambiguous"] or classification["manual"]:
            raise SystemExit("Recovery authorization blocked: unresolved, completed, or manual-review slots exist")
        if len(classification["safe"]) + len(classification["already_authorized"]) != len(slots):
            raise SystemExit("Recovery authorization blocked: not every repaired slot is accounted for")
        appended = append_recovery_authorizations(ledger, classification)
        summary["mode"] = "authorize"
        summary["recovery_authorizations_appended"] = appended
    else:
        summary["recovery_authorizations_appended"] = 0
    output = args.output or args.phase_dir / "recovery_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
