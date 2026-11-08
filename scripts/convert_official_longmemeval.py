from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.official_longmemeval import convert_official_file  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert official cleaned LongMemEval into GovMem JSONL.")
    parser.add_argument("--input", required=True, help="Path to official LongMemEval cleaned JSON array.")
    parser.add_argument("--output", required=True, help="Path to write GovMem-style JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of examples to convert.")
    parser.add_argument("--include-assistant", action="store_true", help="Include assistant messages as memories.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    convert_official_file(
        args.input,
        args.output,
        limit=args.limit,
        include_assistant=args.include_assistant,
    )


if __name__ == "__main__":
    main()

