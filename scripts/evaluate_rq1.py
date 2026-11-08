#!/usr/bin/env python3
"""Evaluate Phase 4A reconstruction JSONL records offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.rq1_metrics import evaluate_records  # noqa: E402


def _load(paths: list[str]) -> list[dict]:
    records = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc.msg}") from exc
    return records


def _rate(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _print_table(summary: dict) -> None:
    print("RQ1 structural reconstruction metrics (applicable, non-skipped denominators)")
    print("group\tn\tvalid\tempty\tmean_states\tgrounding")
    groups = [("all", summary["structural"])]
    groups.extend((f"dataset={key}", value) for key, value in summary["by_dataset"].items())
    groups.extend((f"task={key}", value) for key, value in summary["by_task_type"].items())
    groups.extend((f"dimension={key}", value) for key, value in summary["by_governance_dimension"].items())
    for name, metrics in groups:
        print(
            f"{name}\t{metrics['n_examples']}\t{_rate(metrics['valid_prediction_rate'])}\t"
            f"{_rate(metrics['empty_reconstruction_rate'])}\t{_rate(metrics['mean_states_per_example'])}\t"
            f"{_rate(metrics['grounding_coverage'])}"
        )
    print("\nNative-supported structured state agreement (not structural accuracy)")
    for field, metrics in summary["native_supported_state_agreement"].items():
        value = _rate(metrics["accuracy"]) if metrics["available"] else "unavailable"
        print(f"{field}\tn={metrics['n']}\tagreement={value}")
    print(f"\nDownstream evaluation: {summary['downstream']['status']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 4A JSONL records for RQ1.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more batch JSONL files")
    parser.add_argument("--output", required=True, help="Machine-readable JSON summary path")
    args = parser.parse_args(argv)
    try:
        summary = evaluate_records(_load(args.input))
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    _print_table(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
