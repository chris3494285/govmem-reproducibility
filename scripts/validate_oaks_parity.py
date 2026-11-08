#!/usr/bin/env python3
"""Calculate GovMem-vs-native OAKS option parity from local JSONL input."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Preserve ``python -m scripts.validate_oaks_parity`` while also supporting
# direct execution from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_b4_scoring import _read_jsonl, validate_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Private/licensed OAKS validation input does not exist: {args.input}")
    summary, details, _ = validate_rows(_read_jsonl(args.input), "oaks_novel")
    report = {"schema_version": "oaks-native-parity-v1", "total": summary["validation_sample_n"], "agreements": summary["agreement_n"], "disagreements": summary["disagreement_n"], "agreement_rate": summary["agreement_rate"], "ambiguous_unscored": summary["unscored_n"] + summary["adjudication_required_n"], "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("total", "agreements", "disagreements", "agreement_rate", "ambiguous_unscored")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
