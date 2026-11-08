from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.runner import run_mode  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GovMem demo evaluation.")
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "data" / "sample_longmemeval.jsonl"),
        help="Path to a GovMem-style JSONL dataset.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(ROOT / "results"),
        help="Directory where evaluation artifacts will be written.",
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "baseline", "govmem", "compare"],
        default="demo",
        help="Evaluation mode to run.",
    )
    parser.add_argument(
        "--answer-mode",
        choices=["heuristic", "oracle", "enhanced_heuristic", "llm"],
        default="heuristic",
        help="Local answer selection mode.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Correctness judging mode for compare runs.",
    )
    parser.add_argument(
        "--govmem-selection-policy",
        choices=["default", "governance_rerank", "oracle_evidence"],
        default="default",
        help="GovMem selection policy for compare runs.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start evaluating from this dataset index after loading the dataset.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate from the selected slice.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume compare runs by skipping already completed question IDs in compare_results.json.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        results = run_mode(
            args.dataset,
            results_dir=args.results_dir,
            mode=args.mode,
            answer_mode=args.answer_mode,
            judge_mode=args.judge_mode,
            govmem_selection_policy=args.govmem_selection_policy,
            start_index=args.start_index,
            max_examples=args.max_examples,
            resume=args.resume,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "OPENAI_API_KEY" in message or message.startswith("LLM request failed") or message.startswith("LLM response did not contain"):
            print(f"Error: {message}", file=sys.stderr)
            raise SystemExit(1)
        raise

    if args.mode == "demo":
        methods = results["methods"]
        header = f"{'method':<28} {'EM':>6} {'F1':>6} {'avg_k':>8} {'policy':>8} {'stale':>8}"
        print(header)
        print("-" * len(header))
        for method in methods:
            print(
                f"{method['method']:<28} {method['exact_match']:.3f} {method['token_f1']:.3f} {method['avg_retrieved_memories']:.2f} {method['policy_violation_count']:>8} {method['stale_memory_error_count']:>8}"
            )
        print()
        print("Generated artifacts:")
        for filename in [
            "demo_results.json",
            "demo_summary.md",
            "demo_traces.json",
            "demo_trace_report.md",
        ]:
            print(f"- {Path(args.results_dir) / filename}")
    elif args.mode == "compare":
        print(
            f"compare mode: baseline accuracy {results['summary']['baseline_accuracy']:.3f}, govmem accuracy {results['summary']['govmem_accuracy']:.3f}, baseline evidence hit {results['summary']['baseline_evidence_hit_rate']:.3f}, govmem evidence hit {results['summary']['govmem_evidence_hit_rate']:.3f}, answer mode {results['summary'].get('answer_mode', args.answer_mode)}, judge mode {results['summary'].get('judge_mode', args.judge_mode)}, selection policy {results['summary'].get('govmem_selection_policy', args.govmem_selection_policy)}"
        )
        print("Generated artifacts:")
        for filename in [
            "summary.json",
            "compare_results.json",
            "examples.md",
            "report.md",
            "summary_cases.csv",
            "selection_misses.md",
            "both_correct.md",
            "govmem_only_correct.md",
            "baseline_only_correct.md",
            "both_wrong.md",
        ]:
            print(f"- {Path(args.results_dir) / filename}")
    else:
        print(f"{args.mode} mode: accuracy {results['summary']['accuracy']:.3f}")
        print("Generated artifacts:")
        for filename in [
            "summary.json",
            "examples.md",
        ]:
            print(f"- {Path(args.results_dir) / filename}")


if __name__ == "__main__":
    main()
