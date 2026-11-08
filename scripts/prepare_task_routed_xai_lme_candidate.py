#!/usr/bin/env python3
"""Offline capability preflight for an xAI LME candidate.

This deliberately does not add an xAI semantic-judge adapter.  The repository
has an xAI final-main transport, but the B4 judge executor does not support xAI;
the script therefore stops before creating a candidate plan.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.providers.grok_provider import DEFAULT_GROK_MODEL
from scripts.b4_calibration_execution import make_adapter


def xai_judge_support() -> tuple[bool, str]:
    try:
        make_adapter("xai", DEFAULT_GROK_MODEL)
    except ValueError as exc:
        return False, str(exc)
    return True, "xAI is registered in the B4 judge executor"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final_native_evaluation_design/judge_selection/task_routed_v3/XAI_CANDIDATE_BLOCKER.md"),
    )
    args = parser.parse_args(argv)
    supported, reason = xai_judge_support()
    if supported:
        raise SystemExit("unexpected xAI judge support; review before creating a plan")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# Task-routed v3 xAI LongMemEval candidate preflight\n\n"
        "Status: BLOCKED before execution-plan creation.\n\n"
        f"Canonical xAI model found in repository: `{DEFAULT_GROK_MODEL}`.\n"
        "An xAI final-main provider exists, but the B4 semantic-judge executor "
        f"does not register an xAI adapter (`{reason}`). Adding a speculative "
        "semantic-judge integration is outside this preparation task.\n\n"
        "No 64-case xAI plan, ledger, prompt plan, or paid request was created. "
        "The v2 DeepSeek/Kimi plans, prior candidate ledgers, frozen calibration "
        "identities, and locked holdout were not modified or read for labels.\n",
        encoding="utf-8",
    )
    print("XAI_PROVIDER_SUPPORTED=no")
    print(f"XAI_CANONICAL_MODEL={DEFAULT_GROK_MODEL}")
    print("LME_XAI_PENDING=64")
    print("MEMORA_CASES_TOUCHED=0")
    print("HOLDOUT_CASES_TOUCHED=0")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
