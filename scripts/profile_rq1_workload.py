#!/usr/bin/env python3
"""Profile the frozen RQ1 cohort locally without invoking reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.openai_reconstruction import build_reconstruction_prompt, model_visible_payload  # noqa: E402
from eval.reconstruction import GovernanceReconstructionInput  # noqa: E402
from eval.reconstruction_applicability import applicable_governance_dimensions, dataset_native_task_type  # noqa: E402
from scripts.build_rq1_pilot_manifest import validate_manifest  # noqa: E402
from scripts.build_rq1_pilot_preflight import FROZEN_PARENT_SHA256  # noqa: E402
from scripts.run_reconstruction_smoke import ADAPTERS  # noqa: E402


PERCENTILES = (50, 75, 90, 95)


def _percentile(values: list[int], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def distribution(values: Iterable[int]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0] if ordered else None,
        "median": _percentile(ordered, 50),
        "p75": _percentile(ordered, 75),
        "p90": _percentile(ordered, 90),
        "p95": _percentile(ordered, 95),
        "max": ordered[-1] if ordered else None,
    }


def _profile_example(example, manifest_row: dict[str, Any]) -> dict[str, Any]:
    dimensions = tuple(applicable_governance_dimensions(example))
    input_data = GovernanceReconstructionInput.from_example(example)
    payload = model_visible_payload(input_data)
    payload_bytes = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    prompt = build_reconstruction_prompt(input_data)
    return {
        "example_id": manifest_row["example_id"],
        "query_id": example.query.query_id,
        "dataset": manifest_row["dataset"],
        "task_type": dataset_native_task_type(example),
        "applicable_governance_dimensions": list(dimensions),
        "eligible": bool(dimensions),
        "interaction_count": len(example.interactions),
        "visible_interaction_count": len(input_data.raw_interactions),
        "serialized_history_character_count": len(json.dumps(payload["raw_interactions"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)),
        "serialized_history_byte_count": len(json.dumps(payload["raw_interactions"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")),
        "serialized_model_visible_input_character_count": len(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)),
        "serialized_model_visible_input_byte_count": len(payload_bytes),
        "serialized_prompt_character_count": len(prompt),
        "serialized_prompt_byte_count": len(prompt.encode("utf-8")),
        "source_index": manifest_row.get("source_index"),
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligible"]]
    return {
        "selected_n": len(rows),
        "eligible_n": len(eligible),
        "skipped_n": len(rows) - len(eligible),
        "interaction_count": distribution(row["interaction_count"] for row in eligible),
    }


def _read_preflight_results(paths: list[str | Path], profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for path_value in paths:
        for line in Path(path_value).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", ""))
            if query_id in profiles and profiles[query_id]["eligible"]:
                rows.append({
                    "query_id": query_id,
                    "dataset": profiles[query_id]["dataset"],
                    "task_type": profiles[query_id]["task_type"],
                    "interaction_count": profiles[query_id]["interaction_count"],
                })
    return rows


def _representativeness(full: list[int], paid: list[int]) -> dict[str, Any]:
    if not paid:
        return {"status": "too_small_to_conclude", "paid_n": 0, "reason": "No eligible paid-preflight rows matched the frozen cohort."}
    full_median = statistics.median(full)
    paid_median = statistics.median(paid)
    ratio = paid_median / full_median if full_median else None
    if len(paid) < 20:
        status = "too_small_to_conclude"
    elif ratio is not None and ratio < 0.75:
        status = "systematically_lighter"
    elif ratio is not None and ratio > 1.25:
        status = "systematically_heavier"
    else:
        status = "broadly_representative"
    return {
        "status": status,
        "paid_n": len(paid),
        "full_eligible_n": len(full),
        "full_median_interactions": full_median,
        "paid_median_interactions": paid_median,
        "paid_to_full_median_ratio": ratio,
        "note": "Descriptive comparison only; the paid preflight is a small fixed cohort.",
    }


def profile(manifest_path: str | Path, memora_input: str | Path, memconflict_input: str | Path,
            *, paid_preflight_results: list[str | Path] | None = None) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    validate_manifest(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != FROZEN_PARENT_SHA256:
        raise ValueError("Manifest does not match the frozen RQ1 manifest SHA256")
    by_dataset = {"memora": str(memora_input), "memconflict": str(memconflict_input)}
    adapters = {dataset: list(ADAPTERS[dataset]().load(source)) for dataset, source in by_dataset.items()}
    examples_by_id: dict[str, Any] = {}
    for dataset, examples in adapters.items():
        for example in examples:
            query_id = str(example.query.query_id)
            if query_id in examples_by_id:
                raise ValueError(f"Duplicate adapted example ID: {query_id}")
            examples_by_id[query_id] = example
    rows = []
    for manifest_row in manifest["records"]:
        example = examples_by_id.get(manifest_row["example_id"])
        if example is None:
            raise ValueError(f"Manifest example is missing from adapted input: {manifest_row['example_id']}")
        actual_dataset = example.query.metadata.get("dataset") or example.metadata.get("dataset")
        if actual_dataset != manifest_row["dataset"]:
            raise ValueError(f"Manifest dataset mismatch for {manifest_row['example_id']}: {actual_dataset!r}")
        rows.append(_profile_example(example, manifest_row))
    rows.sort(key=lambda row: (row["dataset"], row["task_type"], row["example_id"]))
    eligible = [row for row in rows if row["eligible"]]
    datasets = {}
    for dataset in ("memora", "memconflict"):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        tasks = {}
        for task in sorted({row["task_type"] for row in dataset_rows}):
            tasks[task] = _group_summary([row for row in dataset_rows if row["task_type"] == task])
        datasets[dataset] = {"selected_n": len(dataset_rows), "eligible_n": sum(row["eligible"] for row in dataset_rows),
                             "skipped_n": sum(not row["eligible"] for row in dataset_rows), "by_task_type": tasks}
    result = {
        "profile_version": "rq1-workload-profile-v1",
        "provider_calls": 0,
        "token_counting": {
            "available": False,
            "method": "not performed; interaction and serialized-size profiling only",
        },
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_version": manifest["manifest_version"],
        "seed": manifest["seed"],
        "selected_n": len(rows),
        "eligible_n": len(eligible),
        "skipped_n": len(rows) - len(eligible),
        "datasets": datasets,
        "workload_distribution": _group_summary(rows),
        "eligible_examples": eligible,
    }
    if paid_preflight_results:
        paid = _read_preflight_results(paid_preflight_results, {row["query_id"]: row for row in rows})
        result["paid_preflight_comparison"] = _representativeness(
            [row["interaction_count"] for row in eligible], [row["interaction_count"] for row in paid]
        )
    return result


def markdown(result: dict[str, Any]) -> str:
    lines = ["# RQ1 Workload Profile", "", f"Manifest SHA-256: `{result['manifest_sha256']}`", "",
             "| Dataset | Task type | Selected | Eligible | Skipped | Min | Median | P75 | P90 | P95 | Max |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset, data in result["datasets"].items():
        for task, values in data["by_task_type"].items():
            d = values["interaction_count"]
            lines.append(f"| {dataset} | {task} | {values['selected_n']} | {values['eligible_n']} | {values['skipped_n']} | "
                         f"{d['min']} | {d['median']} | {d['p75']} | {d['p90']} | {d['p95']} | {d['max']} |")
    d = result["workload_distribution"]["interaction_count"]
    lines.extend(["", "## Overall eligible cohort", "", f"Eligible: {result['eligible_n']} / Selected: {result['selected_n']}",
                  f"Interaction counts: min={d['min']}, median={d['median']}, p75={d['p75']}, p90={d['p90']}, p95={d['p95']}, max={d['max']}"])
    if "paid_preflight_comparison" in result:
        lines.extend(["", "## Paid preflight comparison", "", json.dumps(result["paid_preflight_comparison"], sort_keys=True)])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--memora-input", required=True)
    parser.add_argument("--memconflict-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--paid-preflight-results", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = profile(args.manifest, args.memora_input, args.memconflict_input,
                         paid_preflight_results=args.paid_preflight_results)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        Path(args.markdown_output).write_text(markdown(result), encoding="utf-8")
        print(markdown(result))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
