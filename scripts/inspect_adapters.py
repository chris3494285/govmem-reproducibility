#!/usr/bin/env python3
"""Count adapted official/fixture examples without invoking reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.memconflict_adapter import MemConflictDatasetAdapter  # noqa: E402
from eval.memora_adapter import MemoraDatasetAdapter  # noqa: E402
from eval.reconstruction_applicability import applicable_governance_dimensions, dataset_native_task_type  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect adapted Memora or MemConflict examples.")
    parser.add_argument("--dataset", choices=("memora", "memconflict"), required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    adapter = MemoraDatasetAdapter() if args.dataset == "memora" else MemConflictDatasetAdapter()
    examples = list(adapter.load(args.input))
    counts = Counter(dataset_native_task_type(example) for example in examples)
    non_applicable = sum(not applicable_governance_dimensions(example) for example in examples)
    print(json.dumps({
        "dataset": args.dataset,
        "input": str(args.input),
        "total_adapted_examples": len(examples),
        "count_by_task_type": dict(sorted(counts.items(), key=lambda item: str(item[0]))),
        "non_applicable": non_applicable,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
