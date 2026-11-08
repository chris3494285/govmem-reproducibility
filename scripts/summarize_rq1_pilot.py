#!/usr/bin/env python3
"""Create paper-ready JSON and Markdown summaries for RQ1 pilot results."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.rq1_metrics import evaluate_records  # noqa: E402
from scripts.build_rq1_pilot_manifest import validate_manifest  # noqa: E402
from scripts.build_rq1_pilot_preflight import FROZEN_PARENT_SHA256, validate_preflight  # noqa: E402
from scripts.build_rq1_final_manifest import (  # noqa: E402
    FROZEN_FINAL_SHA256,
    MANIFEST_VERSION as FINAL_MANIFEST_VERSION,
    validate_final_manifest,
)
from scripts.build_rq1_gpt56_smoke import (  # noqa: E402
    SMOKE_MANIFEST_VERSION,
    validate_smoke_manifest,
)


def _read_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Result file does not exist: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", ""))
            if not query_id:
                raise ValueError(f"Missing query_id in {path}:{line_number}")
            condition = str(row.get("condition") or "unspecified")
            if condition in {"baseline", "govmem"}:
                raise ValueError(
                    "Baseline/GovMem paired summaries are disabled: frozen RQ1 evaluates structural grounded reconstruction only"
                )
            if condition not in {"reconstruction", "unspecified"}:
                raise ValueError(f"Unknown result condition: {condition}")
            identity = (condition, query_id)
            if identity in seen:
                raise ValueError(f"Duplicate result identity: condition={condition}, query_id={query_id}")
            seen.add(identity)
            rows.append(row)
    return rows


def _operational(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = len(rows)
    skipped = sum(bool(row.get("skipped")) for row in rows)
    failed = sum(
        not row.get("skipped")
        and not (row.get("reconstruction_status") == "success" and row.get("validation_status") == "valid")
        for row in rows
    )
    completed = selected - skipped - failed
    eligible = selected - skipped
    provider_failures = sum(
        not row.get("skipped") and row.get("validation_status") in {"transport_error", "error"}
        for row in rows
    )
    latencies = [row["latency_ms"] for row in rows if isinstance(row.get("latency_ms"), (int, float))]
    usage_rows = [row.get("usage") for row in rows if isinstance(row.get("usage"), dict)]

    def total(field: str) -> int | None:
        values = [usage.get(field) for usage in usage_rows if isinstance(usage.get(field), int)]
        return sum(values) if values else None

    return {
        "selected_n": selected,
        "eligible_n": eligible,
        "completed_n": completed,
        "skipped_n": skipped,
        "failed_n": failed,
        "provider_failure_count": provider_failures,
        "provider_failure_rate": provider_failures / eligible if eligible else None,
        "skip_rate": skipped / selected if selected else None,
        "latency_available_n": len(latencies),
        "mean_latency_ms": statistics.mean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "usage_available_n": len(usage_rows),
        "total_input_tokens": total("input_tokens"),
        "total_output_tokens": total("output_tokens"),
        "total_tokens": total("total_tokens"),
        "mean_total_tokens": (total("total_tokens") / len(usage_rows) if total("total_tokens") is not None and usage_rows else None),
    }


def _check_completeness(
    selection: dict[str, Any], rows: list[dict[str, Any]], allow_incomplete: bool,
    dataset: str = "all",
) -> list[str]:
    selected_rows = selection.get("examples", selection.get("records", []))
    expected_ids = {
        (row["dataset"], str(row["example_id"])) for row in selected_rows
        if dataset == "all" or row["dataset"] == dataset
    }
    conditions = sorted({str(row.get("condition") or "unspecified") for row in rows})
    missing: list[str] = []
    for condition in conditions:
        actual = {(row.get("dataset"), str(row.get("query_id"))) for row in rows if (row.get("condition") or "unspecified") == condition}
        missing.extend(f"{condition}:{dataset}:{query_id}" for dataset, query_id in sorted(expected_ids - actual))
        unexpected = actual - expected_ids
        missing.extend(f"unexpected:{condition}:{dataset}:{query_id}" for dataset, query_id in sorted(unexpected))
    if missing and not allow_incomplete:
        raise ValueError(
            f"Result set does not exactly match the selected manifest for condition(s) {conditions}: "
            f"unmatched={len(missing)}; "
            "use --allow-incomplete only for explicit partial-result inspection"
        )
    return missing


def summarize(
    manifest_path: str | Path, result_paths: list[str | Path], *, allow_incomplete: bool = False,
    dataset: str = "all",
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("manifest_version") == SMOKE_MANIFEST_VERSION:
        validate_smoke_manifest(manifest)
        selection = manifest
        manifest_version = manifest["manifest_version"]
    elif manifest.get("manifest_version") == FINAL_MANIFEST_VERSION:
        validate_final_manifest(manifest)
        if hashlib.sha256(manifest_bytes).hexdigest() != FROZEN_FINAL_SHA256:
            raise ValueError("Final manifest does not match the frozen RQ1 final manifest SHA256")
        selection = manifest
        manifest_version = manifest["manifest_version"]
    elif manifest.get("preflight_version"):
        validate_preflight(manifest)
        if manifest.get("parent_manifest_sha256") != FROZEN_PARENT_SHA256:
            raise ValueError(
                "Preflight parent_manifest_sha256 does not match the frozen RQ1 parent manifest"
            )
        selection = manifest
        manifest_version = manifest["preflight_version"]
    else:
        validate_manifest(manifest)
        selection = manifest
        manifest_version = manifest["manifest_version"]
    rows = _read_rows(result_paths)
    incomplete_items = _check_completeness(selection, rows, allow_incomplete, dataset)
    by_dataset: dict[str, list[dict[str, Any]]] = {"memora": [], "memconflict": []}
    for row in rows:
        if row.get("dataset") in by_dataset:
            by_dataset[row["dataset"]].append(row)
    summary: dict[str, Any] = {
        "summary_version": "rq1-pilot-summary-v1",
        "complete": not incomplete_items,
        "incomplete_items": incomplete_items,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_version": manifest_version,
        "seed": manifest.get("seed"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conditions": {},
        "evaluation_semantics": {
            "mode": "structural_grounded_reconstruction",
            "native_structured_gold_required_for_accuracy": True,
            "native_supported_agreement_available": False,
            "native_supported_agreement_note": (
                "Native structured agreement is unavailable unless the benchmark exposes "
                "a canonical structured governance target under the current mapping."
            ),
        },
    }
    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_condition.setdefault(str(row.get("condition") or "unspecified"), []).append(row)
    for condition, condition_rows in sorted(rows_by_condition.items()):
        condition_data = {"operational": _operational(condition_rows), "datasets": {}}
        for dataset, dataset_rows in by_dataset.items():
            dataset_rows = [row for row in condition_rows if row.get("dataset") == dataset]
            metric_summary = evaluate_records(dataset_rows)
            tasks = sorted({str(row.get("task_type") or row.get("query_type") or "unknown") for row in dataset_rows})
            task_summary = {}
            for task in tasks:
                task_rows = [row for row in dataset_rows if (row.get("task_type") or row.get("query_type")) == task]
                task_summary[task] = {
                    "operational": _operational(task_rows),
                    "metrics": evaluate_records(task_rows)["structural"],
                }
            condition_data["datasets"][dataset] = {
                "operational": _operational(dataset_rows),
                "metrics": metric_summary["structural"],
                "native_supported_state_agreement": metric_summary["native_supported_state_agreement"],
                "by_task_type": task_summary,
            }
            if any(
                metric.get("available")
                for metric in metric_summary["native_supported_state_agreement"].values()
            ):
                summary["evaluation_semantics"]["native_supported_agreement_available"] = True
        summary["conditions"][condition] = condition_data

    comparison: dict[str, Any] = {
        "available": False,
        "groups": {},
        "reason": "RQ1 is structural grounded reconstruction evaluation only; downstream baseline-vs-GovMem comparison belongs to a later RQ",
    }
    summary["comparison"] = comparison
    # Preserve the prior single-condition access path for existing consumers;
    # multi-condition callers should use ``conditions`` explicitly.
    if len(summary["conditions"]) == 1:
        summary["datasets"] = next(iter(summary["conditions"].values()))["datasets"]
    else:
        summary["datasets"] = {}
    return summary


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RQ1 Official Pilot Summary",
        "",
        f"Manifest: `{summary['manifest']}`  ",
        f"Manifest SHA-256: `{summary['manifest_sha256']}`  ",
        f"Complete: `{summary['complete']}`",
        "",
        "Evaluation semantics: structural, grounded reconstruction diagnostics. "
        "Unavailable native structured agreement is not treated as zero accuracy.",
        "",
        "| Dataset | Task type | Selected | Eligible | Completed | Skipped | Failed | Valid prediction rate | Empty reconstruction rate | Provider failures | Mean latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, condition_data in summary["conditions"].items():
        for dataset, data in condition_data["datasets"].items():
          for task, task_data in data["by_task_type"].items():
            op, metrics = task_data["operational"], task_data["metrics"]
            lines.append(
                f"| {condition} / {dataset} | {task} | {op['selected_n']} | {op['eligible_n']} | {op['completed_n']} | "
                f"{op['skipped_n']} | {op['failed_n']} | {metrics.get('valid_prediction_rate')} | "
                f"{metrics.get('empty_reconstruction_rate')} | {op['provider_failure_count']} | {op['mean_latency_ms']} |"
            )
        for dataset, data in condition_data["datasets"].items():
            native = data["native_supported_state_agreement"]
            available = any(metric.get("available") for metric in native.values())
            lines.extend(["", f"**{condition} / {dataset} Native structured agreement:** "
                          f"{'available for explicitly supported fields' if available else 'unavailable'}"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", nargs="+", required=True, help="Pilot result JSONL files")
    parser.add_argument("--output", required=True, help="Summary JSON output")
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--dataset", choices=("all", "memora", "memconflict"), default="all")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = summarize(args.manifest, args.input, allow_incomplete=args.allow_incomplete, dataset=args.dataset)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        markdown_path = Path(args.markdown_output) if args.markdown_output else output.with_suffix(".md")
        markdown_path.write_text(markdown(result), encoding="utf-8")
        print(markdown(result))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
