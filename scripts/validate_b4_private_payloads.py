#!/usr/bin/env python3
"""Offline validator for blinded B4 private execution payloads."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_b4_private_payloads import read_jsonl, validate_payload_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--kind", choices=("longmemeval", "memora", "memora_human"), required=True)
    args = parser.parse_args(argv)
    result = validate_payload_records(read_jsonl(args.payload), args.kind)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
