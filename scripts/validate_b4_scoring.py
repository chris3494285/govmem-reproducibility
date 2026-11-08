#!/usr/bin/env python3
"""Offline B4 validation of GovMem scoring against supplied native decisions.

Input is a private/local JSONL cohort.  Each row must contain an identifier,
``source_dataset``, ``model_answer`` (or ``parsed_answer``), and the fields
needed by the deterministic scorer.  Native validation may be represented by
``native_correctness`` or ``native_score``.  The script never downloads data,
initializes a provider, or copies benchmark text into its reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# When invoked as ``python scripts/validate_b4_scoring.py``, Python puts the
# scripts directory (rather than the repository root) first on sys.path.
# Bootstrap only that direct-script case; normal module imports are unchanged.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.system_validation_scoring import score_task_answer_detailed

SCHEMA_VERSION = "b4-validation-v1"
JUDGE_PLAN_VERSION = "memora-native-judge-plan-v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            rows.append(value)
    return rows


def _row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("b4_validation_id") or row.get("condition_id") or row.get("example_id") or row.get("id") or f"row-{index}")


def _native_value(row: dict[str, Any]) -> tuple[bool, bool | None, str]:
    if "native_correctness" in row:
        value = row["native_correctness"]
    elif "native_score" in row:
        value = row["native_score"]
    elif "native_abstention" in row:
        value = row["native_abstention"]
    else:
        return False, None, "native_decision_missing"
    if value is None:
        return True, None, "native_unscored"
    if isinstance(value, bool):
        return True, value, "native_boolean"
    if isinstance(value, (int, float)) and value in (0, 1):
        return True, bool(value), "native_binary"
    raise ValueError(f"Native decision for {_row_id(row, 0)} must be boolean/binary/null")


def _public_scoring_details(details: dict[str, Any]) -> dict[str, Any]:
    """Keep audit counts/reasons while excluding licensed criterion text."""
    allowed = {
        "subtype", "reason", "outcome", "abstention", "positive_recall",
        "required_positive_count", "required_positive_correct_count",
        "forbidden_item_count", "forbidden_item_violation_count",
        "remaining_task_recall", "remaining_task_count", "remaining_task_correct_count",
        "false_remaining_task_count", "field_correct_count", "field_total_count",
        "field_accuracy", "required_item_correct_count", "required_item_total_count",
        "required_item_recall", "preference_compatibility", "concrete_recommendation_count",
    }
    return {key: value for key, value in details.items() if key in allowed and not isinstance(value, (list, dict))}


def validate_rows(rows: list[dict[str, Any]], source_filter: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    details = []
    queue = []
    selected = [row for row in rows if not source_filter or row.get("source_dataset") == source_filter]
    by_subtype: Counter[str] = Counter()
    for index, row in enumerate(selected):
        source = str(row.get("source_dataset") or "unknown")
        scored = score_task_answer_detailed(row, row.get("model_answer", row.get("parsed_answer", row.get("answer"))))
        native_present, native_score, native_reason = _native_value(row)
        subtype = str(row.get("subtype") or scored.get("scoring_details", {}).get("subtype") or row.get("task_type") or "unknown")
        by_subtype[subtype] += 1
        deterministic = scored.get("deterministic_score")
        status = scored.get("scoring_status")
        if status == "adjudication_required" or scored.get("adjudication_required"):
            queue.append({
                "id": _row_id(row, index),
                "source_dataset": source,
                "subtype": subtype,
                "scoring_method": scored.get("scoring_method"),
                "scoring_version": scored.get("scoring_version"),
                "reason": scored.get("scoring_details", {}).get("reason", "adjudication_required"),
                "input_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            })
        details.append({
            "id": _row_id(row, index),
            "source_dataset": source,
            "subtype": subtype,
            "scoring_method": scored.get("scoring_method"),
            "scoring_version": scored.get("scoring_version"),
            "scoring_status": status,
            "deterministic_score": deterministic,
            "native_validation_score": native_score,
            "native_decision_status": native_reason,
            "agreement": deterministic == native_score if native_present and deterministic is not None and native_score is not None else None,
            "adjudication_required": bool(scored.get("adjudication_required") or status == "adjudication_required"),
            "scoring_details": _public_scoring_details(scored.get("scoring_details", {})),
        })
    deterministic_n = sum(item["deterministic_score"] is not None for item in details)
    native_n = sum(item["native_validation_score"] is not None for item in details)
    agreement_n = sum(item["agreement"] is True for item in details)
    disagreement_n = sum(item["agreement"] is False for item in details)
    comparable_n = agreement_n + disagreement_n
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scoring_version": details[0]["scoring_version"] if details else None,
        "source_dataset": source_filter or "all",
        "validation_sample_n": len(details),
        "criterion_n": sum(int(row.get("criterion_n", 0)) for row in selected) or len(details),
        "deterministic_scored_n": deterministic_n,
        "native_scored_n": native_n,
        "agreement_n": agreement_n,
        "disagreement_n": disagreement_n,
        "agreement_rate": agreement_n / comparable_n if comparable_n else None,
        "adjudication_required_n": sum(item["adjudication_required"] for item in details),
        "unscored_n": sum(item["scoring_status"] == "unscored" for item in details),
        "native_decision_missing_n": sum(item["native_decision_status"] == "native_decision_missing" for item in details),
        "by_subtype": {},
    }
    for subtype in sorted(by_subtype):
        subset = [item for item in details if item["subtype"] == subtype]
        agrees = [item for item in subset if item["agreement"] is not None]
        summary["by_subtype"][subtype] = {
            "validation_sample_n": len(subset),
            "criterion_n": sum(int(selected[i].get("criterion_n", 0)) for i in range(len(selected)) if str(selected[i].get("subtype") or "unknown") == subtype) or len(subset),
            "deterministic_scored_n": sum(item["deterministic_score"] is not None for item in subset),
            "native_scored_n": sum(item["native_validation_score"] is not None for item in subset),
            "agreement_n": sum(item["agreement"] is True for item in subset),
            "disagreement_n": sum(item["agreement"] is False for item in subset),
            "agreement_rate": sum(item["agreement"] is True for item in agrees) / len(agrees) if agrees else None,
            "adjudication_required_n": sum(item["adjudication_required"] for item in subset),
            "unscored_n": sum(item["scoring_status"] == "unscored" for item in subset),
        }
    return summary, details, queue


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_summary_md(path: Path, summary: dict[str, Any], blocker: str | None = None) -> None:
    lines = ["# B4 scoring validation", "", f"- schema: `{summary['schema_version']}`", f"- source: `{summary['source_dataset']}`", f"- validation sample: {summary['validation_sample_n']}"]
    if blocker:
        lines += ["", "## Status", "", f"BLOCKED: {blocker}"]
    else:
        lines += [f"- deterministic scored: {summary['deterministic_scored_n']}", f"- native scored: {summary['native_scored_n']}", f"- agreement: {summary['agreement_n']} / {summary['agreement_n'] + summary['disagreement_n']}", f"- agreement rate: {summary['agreement_rate']}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="private/local validation JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("results/b4_scoring_validation"))
    parser.add_argument("--source", choices=["all", "longmemeval", "oaks_novel", "memora"], default="all")
    parser.add_argument("--dry-run-judge", action="store_true", help="write deterministic Memora judge requests; never execute them")
    parser.add_argument("--allow-paid-native-judging", action="store_true", help="required acknowledgement for any future paid judge executor; no executor is bundled")
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if not args.input:
        summary = {"schema_version": SCHEMA_VERSION, "status": "blocked", "source_dataset": args.source, "validation_sample_n": 0, "criterion_n": 0, "deterministic_scored_n": 0, "native_scored_n": 0, "agreement_n": 0, "disagreement_n": 0, "agreement_rate": None, "adjudication_required_n": 0, "unscored_n": 0, "blocker": "Private/licensed validation JSONL was not supplied; obtain local native benchmark inputs and rerun with --input."}
        _write_json(out / "b4_validation_summary.json", summary)
        _write_summary_md(out / "b4_validation_summary.md", summary, summary["blocker"])
        print(json.dumps(summary, sort_keys=True))
        return 2
    if not args.input.is_file():
        raise SystemExit(f"Private validation input does not exist: {args.input}")
    rows = _read_jsonl(args.input)
    source = None if args.source == "all" else args.source
    summary, details, queue = validate_rows(rows, source)
    summary["status"] = "validated" if summary["native_scored_n"] else "blocked"
    if not summary["native_scored_n"]:
        summary["blocker"] = "No native decisions were supplied; provide native_correctness/native_score fields from benchmark-native or human/native evaluation."
    _write_json(out / "b4_validation_summary.json", summary)
    _write_summary_md(out / "b4_validation_summary.md", summary, summary.get("blocker"))
    filename = {None: "all", "longmemeval": "longmemeval", "oaks_novel": "oaks", "memora": "memora"}[source]
    _write_json(out / f"{filename}_validation.json", {"summary": summary, "details": details})
    (out / "memora_adjudication_queue.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in queue), encoding="utf-8")
    if args.dry_run_judge:
        requests = [{"request_id": item["id"], "judge_count": 3, "purpose": "Memora native-style criterion adjudication", "input_sha256": item["input_sha256"]} for item in queue if item["source_dataset"] == "memora"]
        _write_json(out / "memora_native_judge_requests.json", {"schema_version": JUDGE_PLAN_VERSION, "execution": "dry_run_only", "judge_count": 3, "request_n": len(requests), "requests": requests})
        print(f"Memora native judge dry-run: {len(requests)} requests × 3 judges; no API calls made.")
    elif args.allow_paid_native_judging:
        raise SystemExit("Paid/native execution is not implemented in this offline repository; use the dry-run payload with an separately audited executor.")
    print(json.dumps(summary, sort_keys=True))
    # Dry-run is a successful offline planning operation even when native
    # decisions have not yet been supplied; the report still says BLOCKED.
    return 0 if summary["status"] == "validated" or args.dry_run_judge else 2


if __name__ == "__main__":
    raise SystemExit(main())
