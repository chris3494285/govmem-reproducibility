#!/usr/bin/env python3
"""Build the NON-FROZEN / SMOKE external memory-system validation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from govmem.memory_systems.cohort import build_system_validation_smoke  # noqa: E402
from scripts.run_governance_fault_rq1 import load_manifest  # noqa: E402


DEFAULT_SOURCE = "data/governance_fault_rq1_pilot_frozen_v2.jsonl"
DEFAULT_OUTPUT = "data/system_validation_smoke_non_frozen.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    rows = load_manifest(args.source_manifest, require_full=True)
    manifest = build_system_validation_smoke(rows, source_manifest=args.source_manifest)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "selected_base_examples": manifest["selected_base_examples"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
