"""Safely merge final-main result JSONL files without provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_results(inputs: list[Path], output: Path, *, manifest: Path | None = None, successful_only: bool = False, exclude_condition_ids: set[str] | None = None) -> int:
    """Merge result files in order, treating the first file as the baseline.

    When ``exclude_condition_ids`` is supplied, it applies only to the first
    (historical baseline) input.  Later inputs are replacement result files
    and are never filtered by that set.  With a manifest, replacement inputs
    must provide every excluded ID and no ID outside the manifest.
    """
    rows: dict[str, dict] = {}
    if not inputs:
        raise ValueError("At least one result input is required")
    replacement_ids: set[str] = set()
    for input_index, path in enumerate(inputs):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            condition_id = row.get("condition_id")
            if not condition_id:
                raise ValueError(f"Missing or duplicate condition_id while merging: {condition_id}")
            if input_index == 0 and exclude_condition_ids and condition_id in exclude_condition_ids:
                continue
            if successful_only and row.get("status") != "success":
                continue
            if condition_id in rows:
                raise ValueError(f"Missing or duplicate condition_id while merging: {condition_id}")
            rows[condition_id] = row
            if input_index > 0:
                replacement_ids.add(condition_id)
    expected = None
    if manifest is not None:
        expected = [json.loads(line)["condition_id"] for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(expected) != len(set(expected)):
            raise ValueError("Manifest contains duplicate condition_id")
        if exclude_condition_ids:
            missing_replacements = sorted(exclude_condition_ids - replacement_ids)
            unexpected_replacements = sorted(replacement_ids - exclude_condition_ids)
            if missing_replacements:
                raise ValueError(f"Replacement inputs are missing excluded condition IDs: {missing_replacements[:5]}")
            if unexpected_replacements:
                raise ValueError(f"Replacement inputs contain unexpected condition IDs: {unexpected_replacements[:5]}")
        if set(rows) != set(expected):
            raise ValueError(f"Merged results contain {len(rows)} unique IDs; expected {len(expected)}")
    order = {cid: index for index, cid in enumerate(expected)} if expected is not None else None
    ordered = sorted(rows.values(), key=lambda row: order.get(row["condition_id"], row["condition_id"]) if order is not None else row["condition_id"])
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered), encoding="utf-8")
    return len(ordered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--successful-only", action="store_true")
    parser.add_argument("--exclude-condition-ids", type=Path, help="one condition_id per line to omit from the first/base input only; later inputs are replacements")
    args = parser.parse_args(argv)
    excluded = None
    if args.exclude_condition_ids:
        excluded = {line.strip() for line in args.exclude_condition_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
    count = merge_results(args.results, args.output, manifest=args.manifest, successful_only=args.successful_only, exclude_condition_ids=excluded)
    print(json.dumps({"merged_unique_condition_ids": count, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
