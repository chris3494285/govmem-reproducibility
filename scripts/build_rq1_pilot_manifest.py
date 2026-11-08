#!/usr/bin/env python3
"""Build or validate the frozen official-data RQ1 pilot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.reconstruction_applicability import dataset_native_task_type  # noqa: E402
from scripts.run_reconstruction_smoke import ADAPTERS  # noqa: E402


MANIFEST_VERSION = "rq1-official-pilot-v1"
DEFAULT_SEED = 20260810
EXPECTED_MEMORA_COUNTS = {"remembering": 50, "reasoning": 50, "recommending": 50}
EXPECTED_MEMCONFLICT_COUNTS = {"dynamic": 2946, "static": 360, "conditional": 444}
PILOT_MEMCONFLICT_COUNTS = {"dynamic": 100, "static": 100, "conditional": 100}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_records(dataset: str, source: str, *, strict_counts: bool = True) -> list[dict[str, Any]]:
    examples = list(ADAPTERS[dataset]().load(source))
    if not examples:
        raise ValueError(f"No adapted examples loaded for {dataset}: {source}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_index, example in enumerate(examples):
        example_id = str(example.query.query_id).strip()
        if not example_id:
            raise ValueError(f"{dataset} example at source index {source_index} has no stable query_id")
        if example_id in seen:
            raise ValueError(f"Duplicate stable {dataset} example ID: {example_id}")
        seen.add(example_id)
        task_type = dataset_native_task_type(example)
        if not task_type:
            raise ValueError(f"{dataset} example {example_id} has no native task type")
        rows.append({
            "dataset": dataset,
            "example_id": example_id,
            "query_id": example_id,
            "task_type": task_type,
            "source_index": source_index,
        })
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["task_type"]] = counts.get(row["task_type"], 0) + 1
    expected = EXPECTED_MEMORA_COUNTS if dataset == "memora" else EXPECTED_MEMCONFLICT_COUNTS
    if strict_counts and counts != expected:
        raise ValueError(f"Unexpected official {dataset} task counts: observed={counts}, expected={expected}")
    return rows


def _sample(rows: list[dict[str, Any]], counts: dict[str, int], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for task_type, requested in counts.items():
        candidates = sorted((row for row in rows if row["task_type"] == task_type), key=lambda row: row["example_id"])
        if len(candidates) < requested:
            raise ValueError(
                f"Insufficient MemConflict {task_type} examples: found {len(candidates)}, need {requested}"
            )
        # Sampling is performed after native task classification and on stable
        # IDs, so source-file ordering cannot alter the selected identities.
        selected.extend(rng.sample(candidates, requested))
    return sorted(selected, key=lambda row: (row["task_type"], row["example_id"]))


def build_manifest(
    memora_input: str | Path,
    memconflict_input: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    strict_counts: bool = True,
) -> dict[str, Any]:
    memora_rows = sorted(_stable_records("memora", str(memora_input), strict_counts=strict_counts), key=lambda row: row["example_id"])
    conflict_rows = _sample(
        _stable_records("memconflict", str(memconflict_input), strict_counts=strict_counts),
        PILOT_MEMCONFLICT_COUNTS,
        seed,
    )
    records = memora_rows + conflict_rows
    return {
        "manifest_version": MANIFEST_VERSION,
        "seed": seed,
        "sampling": {
            "method": "random.Random(seed).sample(sorted_by_stable_example_id)",
            "memora": "all adapted examples",
            "memconflict": PILOT_MEMCONFLICT_COUNTS,
        },
        "expected_counts": {
            "memora": EXPECTED_MEMORA_COUNTS,
            "memconflict_source": EXPECTED_MEMCONFLICT_COUNTS,
            "memconflict_pilot": PILOT_MEMCONFLICT_COUNTS,
        },
        "sources": {"memora": str(memora_input), "memconflict": str(memconflict_input)},
        "records": records,
        "counts": {
            "total": len(records),
            "memora": len(memora_rows),
            "memconflict": len(conflict_rows),
            "memora_by_task_type": {task: sum(row["task_type"] == task for row in memora_rows) for task in EXPECTED_MEMORA_COUNTS},
            "memconflict_by_task_type": {task: sum(row["task_type"] == task for row in conflict_rows) for task in PILOT_MEMCONFLICT_COUNTS},
        },
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version: {manifest.get('manifest_version')!r}")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Manifest records must be a list")
    seen: set[tuple[str, str]] = set()
    for row in records:
        if not isinstance(row, dict) or not row.get("dataset") or not row.get("example_id") or not row.get("task_type"):
            raise ValueError("Every manifest record requires dataset, example_id, and task_type")
        key = (str(row["dataset"]), str(row["example_id"]))
        if key in seen:
            raise ValueError(f"Duplicate manifest identity: {key}")
        seen.add(key)
    counts = manifest.get("counts", {})
    if counts.get("total") != len(records):
        raise ValueError("Manifest total count does not match records")
    if len(records) != 450:
        raise ValueError(f"Pilot manifest must contain exactly 450 records, found {len(records)}")
    actual_memora = sum(row.get("dataset") == "memora" for row in records)
    actual_conflict = sum(row.get("dataset") == "memconflict" for row in records)
    if (actual_memora, actual_conflict) != (150, 300):
        raise ValueError(
            f"Manifest dataset counts must be Memora=150 and MemConflict=300; "
            f"found Memora={actual_memora}, MemConflict={actual_conflict}"
        )
    actual_memora_tasks = {
        task: sum(row.get("dataset") == "memora" and row.get("task_type") == task for row in records)
        for task in EXPECTED_MEMORA_COUNTS
    }
    actual_conflict_tasks = {
        task: sum(row.get("dataset") == "memconflict" and row.get("task_type") == task for row in records)
        for task in PILOT_MEMCONFLICT_COUNTS
    }
    if actual_memora_tasks != EXPECTED_MEMORA_COUNTS or actual_conflict_tasks != PILOT_MEMCONFLICT_COUNTS:
        raise ValueError("Manifest task strata do not match the frozen pilot counts")
    if counts.get("memconflict_by_task_type") != PILOT_MEMCONFLICT_COUNTS:
        raise ValueError("Manifest MemConflict strata are not exactly 100/100/100")
    if counts.get("memora_by_task_type") != EXPECTED_MEMORA_COUNTS:
        raise ValueError("Manifest Memora task counts are not exactly 50/50/50")


def write_manifest(manifest: dict[str, Any], output: str | Path, *, overwrite: bool = False) -> str:
    validate_manifest(manifest)
    path = Path(output)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists; use --overwrite explicitly: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    return _sha256_bytes(payload.encode("utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memora-input")
    parser.add_argument("--memconflict-input")
    parser.add_argument("--output", help="Frozen manifest JSON path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Validate an existing manifest without rebuilding it")
    parser.add_argument("--manifest", help="Existing manifest used with --validate")
    parser.add_argument("--allow-nonofficial-counts", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate:
            if not args.manifest:
                raise ValueError("--validate requires --manifest")
            validate_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
            print(f"Valid RQ1 pilot manifest: {args.manifest}")
            return 0
        if not args.memora_input or not args.memconflict_input or not args.output:
            raise ValueError("building requires --memora-input, --memconflict-input, and --output")
        manifest = build_manifest(
            args.memora_input,
            args.memconflict_input,
            seed=args.seed,
            strict_counts=not args.allow_nonofficial_counts,
        )
        digest = write_manifest(manifest, args.output, overwrite=args.overwrite)
        print(f"Wrote {args.output} ({len(manifest['records'])} records, sha256={digest})")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
