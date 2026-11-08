from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.governance_slices import write_governance_slices  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate governance stress slices from GovMem-style JSONL.")
    parser.add_argument("--input", required=True, help="Path to GovMem-style JSONL input.")
    parser.add_argument("--output", required=True, help="Path to write generated governance slices.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    write_governance_slices(args.input, args.output)


if __name__ == "__main__":
    main()

