#!/usr/bin/env python3
"""Run the five-slice Phase 4C RQ1 sanity experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_reconstruction_batch import run_batch  # noqa: E402


@dataclass(frozen=True)
class SliceSpec:
    name: str
    dataset: str
    input_path: Path
    query_type: str | None
    filename: str


def build_slice_specs(args: argparse.Namespace) -> list[SliceSpec]:
    return [
        SliceSpec("gatemem_privacy", "gatemem", Path(args.gatemem_root), "privacy", "gatemem_privacy.jsonl"),
        SliceSpec("gatemem_safety", "gatemem", Path(args.gatemem_root), "safety", "gatemem_safety.jsonl"),
        # Memora's lifecycle applicability includes its native remembering and
        # forgetting task types, so this slice intentionally has no filter.
        SliceSpec("memora_lifecycle", "memora", Path(args.memora_input), "remembering", "memora_lifecycle.jsonl"),
        SliceSpec("memconflict_dynamic", "memconflict", Path(args.memconflict_input), "dynamic", "memconflict_dynamic.jsonl"),
        SliceSpec("memconflict_conditional", "memconflict", Path(args.memconflict_input), "conditional", "memconflict_conditional.jsonl"),
    ]


def _failure_record(spec: SliceSpec, error: Exception) -> dict[str, Any]:
    return {
        "slice": spec.name,
        "dataset": spec.dataset,
        "query_type": spec.query_type,
        "orchestration_error": {"type": type(error).__name__, "message": str(error)},
    }


def _write_failure(path: Path, spec: SliceSpec, error: Exception) -> None:
    path.write_text(json.dumps(_failure_record(spec, error), sort_keys=True) + "\n", encoding="utf-8")


def run_sanity(
    args: argparse.Namespace,
    *,
    batch_runner: Callable[..., dict[str, Any]] = run_batch,
    evaluator_runner: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    specs = build_slice_specs(args)
    summaries: list[dict[str, Any]] = []
    result_paths: list[Path] = []

    for spec in specs:
        output = results_dir / spec.filename
        result_paths.append(output)
        try:
            summary = batch_runner(
                dataset=spec.dataset,
                input_path=spec.input_path,
                output_path=output,
                model=args.model,
                cache_dir=args.cache_dir,
                limit=args.limit_per_slice,
                query_type=spec.query_type,
                max_input_tokens=getattr(args, "max_input_tokens", None),
                streaming_chunk_tokens=getattr(args, "streaming_chunk_tokens", None),
            )
            summaries.append({"slice": spec.name, "error": None, **summary})
        except Exception as error:  # keep the other slices and make failure visible to evaluation
            print(f"ERROR {spec.name}: {type(error).__name__}: {error}", file=sys.stderr)
            _write_failure(output, spec, error)
            summaries.append({
                "slice": spec.name,
                "error": {"type": type(error).__name__, "message": str(error)},
                "selected": 0,
                "executed": 0,
                "cache_hits": 0,
                "valid": 0,
                "invalid_or_error": 1,
            })

    summary_path = results_dir / "rq1_sanity_summary.json"
    evaluator_command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_rq1.py"),
        "--input",
        *[str(path) for path in result_paths],
        "--output",
        str(summary_path),
    ]
    if evaluator_runner is None:
        completed = subprocess.run(evaluator_command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"RQ1 evaluator failed with exit code {completed.returncode}: {' '.join(evaluator_command)}")
    else:
        evaluator_runner(evaluator_command)

    return {"slices": summaries, "summary_path": str(summary_path), "result_paths": [str(path) for path in result_paths]}, summary_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the five Phase 4C RQ1 sanity slices.")
    parser.add_argument("--gatemem-root", required=True)
    parser.add_argument("--memora-input", required=True)
    parser.add_argument("--memconflict-input", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--limit-per-slice", type=int, default=10)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--streaming-chunk-tokens", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result, _ = run_sanity(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Sanity orchestration failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print("slice\trequested\tselected\texecuted\tcache_hits\tvalid\terrors")
    for item in result["slices"]:
        print(
            f"{item['slice']}\t{args.limit_per_slice}\t{item.get('selected', 0)}\t{item.get('executed', 0)}\t"
            f"{item.get('cache_hits', 0)}\t{item.get('valid', 0)}\t{item.get('invalid_or_error', 0)}"
        )
    print(f"RQ1 summary: {result['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
