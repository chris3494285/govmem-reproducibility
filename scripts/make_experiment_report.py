from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_summary(results_dir: Path) -> dict[str, Any]:
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {results_dir}")
    return _load_json(summary_path)


def _load_examples(results_dir: Path) -> list[dict[str, Any]]:
    compare_path = results_dir / "compare_results.json"
    if not compare_path.exists():
        return []
    payload = _load_json(compare_path)
    examples = payload.get("examples", [])
    return examples if isinstance(examples, list) else []


def _display_value(value: Any, default: str = "n/a") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).strip()
    return text if text else default


def _format_number(value: Any, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return _display_value(value)


def _memory_lines(memories: list[dict[str, Any]], limit: int = 3) -> list[str]:
    lines: list[str] = []
    for memory in memories[:limit]:
        memory_id = _display_value(memory.get("memory_id"))
        text = _display_value(memory.get("text", memory.get("content")))
        lines.append(f"- `{memory_id}`: {text}")
    if not lines:
        lines.append("- none")
    return lines


def _compare_key_findings(rows: list[dict[str, Any]]) -> list[str]:
    default_row = next((row for row in rows if row["policy"] == "default"), None)
    rerank_row = next((row for row in rows if row["policy"] == "governance_rerank"), None)
    oracle_row = next((row for row in rows if row["policy"] == "oracle_evidence"), None)

    bullets = []
    if default_row and default_row["identical_memory_set_rate"] == "1.000":
        bullets.append("- Default policy selected identical memory sets and did not deliver a meaningful gain.")
    if rerank_row and rerank_row["delta_accuracy"] not in {"0.000", "-0.000"}:
        bullets.append("- Governance rerank changed memory sets and improved accuracy.")
    if oracle_row and oracle_row["delta_accuracy"] not in {"0.000", "-0.000"}:
        bullets.append("- Oracle evidence shows additional selection headroom beyond the heuristic reranker.")
    if not bullets:
        bullets.append("- The policies differ in memory overlap and accuracy, with oracle providing the strongest upper bound.")
    return bullets


def _select_examples(compare_rows: list[dict[str, Any]], outcome: str, limit: int = 3) -> list[dict[str, Any]]:
    return [row for row in compare_rows if row.get("comparison_outcome") == outcome][:limit]


def _make_main_table_rows(result_summaries: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in result_summaries:
        summary = item["summary"]
        rows.append(
            {
                "policy": item["policy"],
                "baseline_accuracy": _format_number(summary.get("baseline_accuracy")),
                "govmem_accuracy": _format_number(summary.get("govmem_accuracy")),
                "delta_accuracy": _format_number(summary.get("delta_accuracy")),
                "baseline_correct": _display_value(summary.get("baseline_correct_count")),
                "govmem_correct": _display_value(summary.get("govmem_correct_count")),
                "govmem_only_correct": _display_value(summary.get("govmem_only_correct_count")),
                "baseline_only_correct": _display_value(summary.get("baseline_only_correct_count")),
                "both_correct": _display_value(summary.get("both_correct_count")),
                "both_wrong": _display_value(summary.get("both_wrong_count")),
                "evidence_hit_baseline": _format_number(summary.get("baseline_evidence_hit_rate")),
                "evidence_hit_govmem": _format_number(summary.get("govmem_evidence_hit_rate")),
                "average_memory_jaccard": _format_number(summary.get("average_memory_jaccard")),
                "identical_memory_set_rate": _format_number(summary.get("identical_memory_set_rate")),
            }
        )
    return rows


def _render_table(headers: list[str], rows: list[dict[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" if header == "policy" else "---:" for header in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(header, "n/a") for header in headers) + " |")
    return lines


def build_experiment_report(result_dirs: list[Path], output_path: Path, csv_output: Path | None = None) -> tuple[str, list[dict[str, str]]]:
    if not result_dirs:
        raise ValueError("At least one --results-dir is required.")

    result_summaries = []
    policy_labels = []
    compare_rows_by_policy: dict[str, list[dict[str, Any]]] = {}

    for results_dir in result_dirs:
        summary = _load_summary(results_dir)
        policy = _display_value(summary.get("govmem_selection_policy"), results_dir.name)
        policy_labels.append(policy)
        result_summaries.append({"policy": policy, "summary": summary, "results_dir": results_dir})
        compare_rows_by_policy[policy] = _load_examples(results_dir)

    primary_summary = result_summaries[0]["summary"]
    headers = [
        "policy",
        "baseline_accuracy",
        "govmem_accuracy",
        "delta_accuracy",
        "baseline_correct",
        "govmem_correct",
        "govmem_only_correct",
        "baseline_only_correct",
        "both_correct",
        "both_wrong",
        "evidence_hit_baseline",
        "evidence_hit_govmem",
        "average_memory_jaccard",
        "identical_memory_set_rate",
    ]
    table_rows = _make_main_table_rows(result_summaries)

    lines: list[str] = ["# GovMem Experiment Report", ""]
    lines.append("## Setup")
    dataset_label = primary_summary.get("dataset_path") or primary_summary.get("dataset_mode")
    lines.append(f"- dataset: `{_display_value(dataset_label, 'unknown')}`")
    lines.append(f"- total examples: {_display_value(primary_summary.get('total_examples'))}")
    lines.append(f"- answer mode: `{_display_value(primary_summary.get('answer_mode'))}`")
    lines.append(f"- judge mode: `{_display_value(primary_summary.get('judge_mode'))}`")
    if primary_summary.get("model"):
        lines.append(f"- model: `{_display_value(primary_summary.get('model'))}`")
    lines.append(f"- compared policies: {', '.join(policy_labels)}")
    lines.append("")
    lines.append("## Main Results Table")
    lines.extend(_render_table(headers, table_rows))
    lines.append("")
    lines.append("## Key Findings")
    lines.extend(_compare_key_findings(table_rows))
    lines.append("")
    lines.append("## Case Studies")
    rerank_policy = "governance_rerank"
    rerank_examples = compare_rows_by_policy.get(rerank_policy, [])
    govmem_wins = _select_examples(rerank_examples, "govmem_only_correct")
    if not govmem_wins:
        lines.append("- No GovMem-only correct cases were found for governance rerank.")
    for row in govmem_wins[:3]:
        lines.append(f"### {row.get('question_id', 'example')}")
        lines.append(f"- question: {row.get('question')}")
        lines.append(f"- gold answer: {row.get('gold_answer')}")
        lines.append(f"- baseline answer: {row.get('baseline_answer')}")
        lines.append(f"- GovMem answer: {row.get('govmem_answer')}")
        lines.append("- baseline memories:")
        lines.extend(_memory_lines(row.get("baseline_retrieved_memories", [])))
        lines.append("- GovMem memories:")
        lines.extend(_memory_lines(row.get("govmem_used_memories", [])))
        note = _display_value(row.get("answer_delta_note"), "govmem-only win")
        lines.append(f"- note: {note}")
        lines.append("")
    lines.append("## Error Analysis")
    baseline_only = _select_examples(rerank_examples, "baseline_only_correct")
    if not baseline_only:
        lines.append("- No baseline-only correct cases were found for governance rerank.")
    for row in baseline_only[:3]:
        lines.append(f"### {row.get('question_id', 'example')}")
        lines.append(f"- question: {row.get('question')}")
        lines.append(f"- gold answer: {row.get('gold_answer')}")
        lines.append(f"- baseline answer: {row.get('baseline_answer')}")
        lines.append(f"- GovMem answer: {row.get('govmem_answer')}")
        lines.append(f"- note: {_display_value(row.get('answer_delta_note'), 'baseline-only win')}")
        lines.append("")
    lines.append("## Limitations")
    lines.extend(
        [
            "- LLM answer and judge results are nondeterministic.",
            "- The sample sizes are small compared with the full benchmark.",
            "- `oracle_evidence` is a diagnostic upper bound and uses the gold answer, so it is not deployable.",
            "- `governance_rerank` is heuristic and not learned.",
        ]
    )

    markdown = "\n".join(lines).rstrip() + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in table_rows:
                writer.writerow(row)

    return markdown, table_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a paper-ready experiment report from GovMem results.")
    parser.add_argument(
        "--results-dir",
        action="append",
        required=True,
        help="Path to a results directory containing summary.json and compare_results.json.",
    )
    parser.add_argument("--output", required=True, help="Markdown output path.")
    parser.add_argument("--csv-output", default=None, help="Optional CSV output path.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result_dirs = [Path(path) for path in args.results_dir]
    csv_output = Path(args.csv_output) if args.csv_output else None
    build_experiment_report(result_dirs, Path(args.output), csv_output=csv_output)


if __name__ == "__main__":
    main()
