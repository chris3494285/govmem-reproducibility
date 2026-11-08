from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.longmemeval_adapter import convert_longmemeval_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert simplified LongMemEval-style raw JSONL into GovMem demo JSONL.")
    parser.add_argument("--input", required=True, help="Path to raw LongMemEval-style JSONL input.")
    parser.add_argument("--output", required=True, help="Path to write converted GovMem-style JSONL.")
    args = parser.parse_args()

    convert_longmemeval_file(args.input, args.output)


if __name__ == "__main__":
    main()

