#!/usr/bin/env python3
"""Freeze the deterministic workload-stratified 90-example RQ1 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_rq1_pilot_manifest import validate_manifest  # noqa: E402
from scripts.build_rq1_pilot_preflight import FROZEN_PARENT_SHA256  # noqa: E402
from scripts.profile_rq1_workload import distribution, profile  # noqa: E402


MANIFEST_VERSION = "rq1-final-90-v1"
FROZEN_FINAL_SHA256 = "6f119abcbc94dd22852b49613f6599e821053c36b0807ac9167334ae08257d4a"
PARENT_MANIFEST_VERSION = "rq1-official-pilot-v1"
DEFAULT_PER_STRATUM = 30
ELIGIBLE_STRATA = (
    ("memora", "remembering"),
    ("memconflict", "dynamic"),
    ("memconflict", "conditional"),
)


def even_rank_indices(n: int, count: int) -> list[int]:
    if count < 1 or n < count:
        raise ValueError(f"Cannot select {count} unique ranks from stratum of size {n}")
    if count == 1:
        return [0]
    # Endpoints are included and intermediate ranks are evenly spaced.  The
    # floor operation makes the rule integer-only and reproducible.
    return [(index * (n - 1)) // (count - 1) for index in range(count)]


def select_stratum(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (int(row["interaction_count"]), str(row["example_id"])))
    return [ordered[index] for index in even_rank_indices(len(ordered), count)]


def _validate_parent(path: str | Path) -> tuple[dict[str, Any], bytes, str]:
    payload = Path(path).read_bytes()
    parent = json.loads(payload.decode("utf-8"))
    validate_manifest(parent)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != FROZEN_PARENT_SHA256:
        raise ValueError("Parent manifest does not match the frozen RQ1 manifest SHA256")
    if parent.get("manifest_version") != PARENT_MANIFEST_VERSION:
        raise ValueError("Unsupported parent manifest version")
    return parent, payload, digest


def _diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [int(row["interaction_count"]) for row in rows]
    result = distribution(values)
    return result


def build_final_manifest(
    parent_manifest: str | Path,
    memora_input: str | Path,
    memconflict_input: str | Path,
    *,
    per_stratum: int = DEFAULT_PER_STRATUM,
) -> dict[str, Any]:
    parent, _, parent_sha256 = _validate_parent(parent_manifest)
    if per_stratum != DEFAULT_PER_STRATUM:
        raise ValueError("The frozen final RQ1 cohort requires --per-stratum 30")

    workload = profile(parent_manifest, memora_input, memconflict_input)
    workload_rows = {str(row["example_id"]): row for row in workload["eligible_examples"]}
    parent_rows = {str(row["example_id"]): row for row in parent["records"]}
    if len(parent_rows) != len(parent["records"]):
        raise ValueError("Parent manifest contains duplicate example IDs")

    selected: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for dataset, task_type in ELIGIBLE_STRATA:
        candidates = [
            {**parent_rows[example_id], "interaction_count": workload_rows[example_id]["interaction_count"]}
            for example_id in parent_rows
            if parent_rows[example_id]["dataset"] == dataset
            and parent_rows[example_id]["task_type"] == task_type
        ]
        if len(candidates) < per_stratum:
            raise ValueError(f"Insufficient eligible examples for {dataset}/{task_type}")
        chosen = select_stratum(candidates, per_stratum)
        selected.extend(chosen)
        full_stats = _diagnostic(candidates)
        selected_stats = _diagnostic(chosen)
        diagnostics[f"{dataset}:{task_type}"] = {
            "full_eligible": full_stats,
            "selected": selected_stats,
            "selected_median_over_full_median": (
                selected_stats["median"] / full_stats["median"] if full_stats["median"] else None
            ),
            "selected_p90_over_full_p90": (
                selected_stats["p90"] / full_stats["p90"] if full_stats["p90"] else None
            ),
            "selected_max_over_full_max": (
                selected_stats["max"] / full_stats["max"] if full_stats["max"] else None
            ),
        }

    selected.sort(key=lambda row: (row["dataset"], row["task_type"], row["interaction_count"], row["example_id"]))
    selected_ids = [str(row["example_id"]) for row in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Final manifest contains duplicate selected example IDs")
    if len(selected) != per_stratum * len(ELIGIBLE_STRATA):
        raise ValueError("Final manifest does not contain exactly 90 examples")
    counts = {f"{dataset}:{task}": sum(row["dataset"] == dataset and row["task_type"] == task for row in selected)
              for dataset, task in ELIGIBLE_STRATA}
    if any(value != per_stratum for value in counts.values()):
        raise ValueError(f"Unexpected final stratum counts: {counts}")
    records = [{key: value for key, value in row.items() if key in {
        "dataset", "example_id", "query_id", "task_type", "source_index", "interaction_count",
    }} for row in selected]
    return {
        "manifest_version": MANIFEST_VERSION,
        "parent_manifest_version": parent["manifest_version"],
        "parent_manifest_sha256": parent_sha256,
        "sampling_method": "deterministic_workload_stratified_even_rank",
        "per_stratum": per_stratum,
        "selected_n": len(records),
        "counts": counts,
        "records": records,
        "workload_diagnostics": diagnostics,
        "sources": {"memora": str(memora_input), "memconflict": str(memconflict_input)},
    }


def validate_final_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("Unsupported final manifest version")
    if manifest.get("parent_manifest_version") != PARENT_MANIFEST_VERSION:
        raise ValueError("Final manifest has an unsupported parent version")
    if manifest.get("parent_manifest_sha256") != FROZEN_PARENT_SHA256:
        raise ValueError("Final manifest parent SHA256 does not match the frozen RQ1 parent")
    records = manifest.get("records")
    if manifest.get("selected_n") != 90 or not isinstance(records, list) or len(records) != 90:
        raise ValueError("Final manifest must contain exactly 90 records")
    ids = [str(row.get("example_id", "")) for row in records]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Final manifest IDs must be present and unique")
    actual = {
        f"{dataset}:{task}": sum(row.get("dataset") == dataset and row.get("task_type") == task for row in records)
        for dataset, task in ELIGIBLE_STRATA
    }
    if actual != {f"{dataset}:{task}": 30 for dataset, task in ELIGIBLE_STRATA}:
        raise ValueError(f"Final manifest strata are invalid: {actual}")
    if any((row.get("dataset"), row.get("task_type")) not in ELIGIBLE_STRATA for row in records):
        raise ValueError("Final manifest contains a non-applicable stratum")


def write_manifest(manifest: dict[str, Any], output: str | Path, *, overwrite: bool = False) -> str:
    validate_final_manifest(manifest)
    path = Path(output)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Final manifest already exists; use --overwrite explicitly: {path}")
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def markdown(manifest: dict[str, Any]) -> str:
    lines = ["# Final RQ1 90-Example Cohort", "", f"Manifest SHA-256: `{hashlib.sha256((json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + chr(10)).encode()).hexdigest()}`", "",
             "| Stratum | Full n | Selected n | Full median | Selected median | Full p90 | Selected p90 | Full max | Selected max |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, value in manifest["workload_diagnostics"].items():
        full, selected = value["full_eligible"], value["selected"]
        lines.append(f"| {key} | {full['n']} | {selected['n']} | {full['median']} | {selected['median']} | {full['p90']} | {selected['p90']} | {full['max']} | {selected['max']} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--memora-input", required=True)
    parser.add_argument("--memconflict-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_final_manifest(args.parent_manifest, args.memora_input, args.memconflict_input, per_stratum=args.per_stratum)
        digest = write_manifest(manifest, args.output, overwrite=args.overwrite)
        print(markdown(manifest), end="")
        print(f"Final manifest: {args.output} (sha256={digest})")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
