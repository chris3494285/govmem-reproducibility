from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.gatemem_adapter import GateMemDatasetAdapter  # noqa: E402
from eval.longmemeval_adapter import LongMemEvalDatasetAdapter  # noqa: E402
from eval.memconflict_adapter import MemConflictDatasetAdapter  # noqa: E402
from eval.memora_adapter import MemoraDatasetAdapter  # noqa: E402
from eval.openai_reconstruction import (  # noqa: E402
    DEFAULT_MODEL,
    OpenAIGovernanceStateReconstructor,
    ReconstructionError,
)
from eval.reconstruction_applicability import (  # noqa: E402
    applicable_governance_dimensions,
    dataset_native_task_type,
)
from eval.reconstruction import GovernanceReconstructionInput  # noqa: E402


ADAPTERS = {
    "longmemeval": LongMemEvalDatasetAdapter,
    "gatemem": GateMemDatasetAdapter,
    "memora": MemoraDatasetAdapter,
    "memconflict": MemConflictDatasetAdapter,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small OpenAI governance reconstruction smoke test.")
    parser.add_argument("--dataset", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--input", required=True, help="Dataset file or adapter-specific directory")
    parser.add_argument("--limit", type=int, default=3, help="Maximum examples to send (default: 3)")
    parser.add_argument("--model", default=None, help=f"OpenAI model (default: {DEFAULT_MODEL} or environment override)")
    parser.add_argument("--cache-dir", default=None, help="Disk cache directory")
    parser.add_argument(
        "--query-type",
        default=None,
        help="Optional dataset-native query/task type filter (for example GateMem utility, privacy, or safety)",
    )
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--streaming-chunk-tokens", type=int, default=None)
    return parser.parse_args(argv)


def reconstruct_smoke_example(example, reconstructor) -> dict:
    """Reconstruct one example, skipping non-applicable governance tasks."""
    dimensions = applicable_governance_dimensions(example)
    result = {
        "query_id": example.query.query_id,
        "query_type": example.query.metadata.get("query_type"),
        "task_type": dataset_native_task_type(example),
        "applicable_governance_dimensions": dimensions,
    }
    if not dimensions:
        result.update({
            "states": [],
            "skipped": True,
            "skipped_reason": "no_applicable_governance_dimensions",
        })
        return result

    states = reconstructor.reconstruct(GovernanceReconstructionInput.from_example(example))
    result.update({
        "states": [
            {
                "interaction_id": state.interaction_id,
                "authorized_scopes": list(state.authorized_scopes),
                "is_deleted": state.is_deleted,
                "valid_from": state.valid_from.isoformat() if state.valid_from else None,
                "valid_until": state.valid_until.isoformat() if state.valid_until else None,
                "entity_key": state.entity_key,
                "confidence": state.confidence,
                "missing_fields": list(state.missing_fields),
                "provenance": json.loads(state.provenance),
            }
            for state in states
        ],
        "skipped": False,
    })
    return result


def filter_examples_by_query_type(examples, query_type: str):
    """Filter by the adapter-normalized dataset-native task type."""
    return [example for example in examples if dataset_native_task_type(example) == query_type]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    loaded_examples = ADAPTERS[args.dataset]().load(args.input)
    if args.query_type is not None:
        loaded_examples = filter_examples_by_query_type(loaded_examples, args.query_type)
        if not loaded_examples:
            print(
                f"No {args.dataset} examples matched --query-type {args.query_type!r}",
                file=sys.stderr,
            )
            return 2
    examples = loaded_examples[: args.limit]
    reconstructor = OpenAIGovernanceStateReconstructor(
        model=args.model,
        cache_dir=args.cache_dir,
        max_input_tokens=args.max_input_tokens,
        streaming_chunk_tokens=args.streaming_chunk_tokens,
    )
    for example in examples:
        try:
            result = reconstruct_smoke_example(example, reconstructor)
        except ReconstructionError as exc:
            print(f"{example.query.query_id}: ERROR: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
