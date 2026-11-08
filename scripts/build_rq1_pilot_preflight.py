#!/usr/bin/env python3
"""Freeze the deterministic 20-example RQ1 paid-preflight selection."""

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

from scripts.build_rq1_pilot_manifest import MANIFEST_VERSION, validate_manifest  # noqa: E402


PREFLIGHT_VERSION = "rq1-official-preflight-v1"
FROZEN_PARENT_SHA256 = "8bb6665dfa0c0a8ff92f3a77eb34f2ad110b51914b865125080d57e2c41b24af"
PREFLIGHT_COUNTS = {
    "memora": {"remembering": 4, "reasoning": 3, "recommending": 3},
    "memconflict": {"dynamic": 4, "static": 3, "conditional": 3},
}


def _read_parent(path: str | Path, *, require_frozen_hash: bool = True) -> tuple[dict[str, Any], str]:
    source = Path(path)
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = json.loads(payload.decode("utf-8"))
    validate_manifest(manifest)
    if require_frozen_hash and digest != FROZEN_PARENT_SHA256:
        raise ValueError(
            f"Parent manifest SHA256 {digest} does not match frozen RQ1 manifest {FROZEN_PARENT_SHA256}"
        )
    return manifest, digest


def build_preflight_manifest(parent_path: str | Path, *, require_frozen_hash: bool = True) -> dict[str, Any]:
    parent, parent_sha256 = _read_parent(parent_path, require_frozen_hash=require_frozen_hash)
    selected: list[dict[str, Any]] = []
    for dataset, task_counts in PREFLIGHT_COUNTS.items():
        for task_type, count in task_counts.items():
            candidates = sorted(
                (row for row in parent["records"] if row["dataset"] == dataset and row["task_type"] == task_type),
                key=lambda row: row["example_id"],
            )
            if len(candidates) < count:
                raise ValueError(f"Parent manifest lacks {count} {dataset}/{task_type} examples")
            selected.extend(candidates[:count])
    selected = sorted(selected, key=lambda row: (row["dataset"], row["task_type"], row["example_id"]))
    return {
        "preflight_version": PREFLIGHT_VERSION,
        "parent_manifest_version": parent["manifest_version"],
        "parent_manifest_sha256": parent_sha256,
        "selection_method": "first_n_by_stable_example_id_within_each_manifest_stratum",
        "allocation": PREFLIGHT_COUNTS,
        "records": selected,
        "counts": {
            "total": len(selected),
            "memora": sum(row["dataset"] == "memora" for row in selected),
            "memconflict": sum(row["dataset"] == "memconflict" for row in selected),
        },
    }


def validate_preflight(preflight: dict[str, Any], parent: dict[str, Any] | None = None) -> None:
    if preflight.get("preflight_version") != PREFLIGHT_VERSION:
        raise ValueError("Unsupported preflight manifest version")
    if preflight.get("allocation") != PREFLIGHT_COUNTS:
        raise ValueError("Preflight allocation does not match the frozen 4/3/3 and 4/3/3 design")
    records = preflight.get("records")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("Preflight manifest must contain exactly 20 records")
    expected: set[tuple[str, str]] = set()
    for row in records:
        key = (row.get("dataset"), row.get("example_id"))
        if not all(key) or key in expected:
            raise ValueError(f"Invalid or duplicate preflight identity: {key}")
        expected.add(key)
    observed = {
        dataset: {
            task: sum(row.get("dataset") == dataset and row.get("task_type") == task for row in records)
            for task in tasks
        }
        for dataset, tasks in PREFLIGHT_COUNTS.items()
    }
    if observed != PREFLIGHT_COUNTS:
        raise ValueError(f"Preflight strata mismatch: {observed}")
    if parent is not None:
        parent_ids = {(row["dataset"], row["example_id"]) for row in parent["records"]}
        if not expected.issubset(parent_ids):
            raise ValueError("Preflight contains an identity outside the parent manifest")


def write_preflight(preflight: dict[str, Any], output: str | Path, *, overwrite: bool = False) -> str:
    validate_preflight(preflight)
    path = Path(output)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Preflight manifest already exists; use --overwrite explicitly: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(preflight, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Frozen 450-example parent manifest")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--preflight-manifest", default=None, help="Existing preflight manifest used with --validate")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate:
            if not args.preflight_manifest:
                raise ValueError("--validate requires --preflight-manifest")
            parent, parent_sha256 = _read_parent(args.manifest)
            payload = Path(args.preflight_manifest).read_bytes()
            preflight = json.loads(payload.decode("utf-8"))
            validate_preflight(preflight, parent)
            if preflight.get("parent_manifest_sha256") != parent_sha256:
                raise ValueError("Preflight parent SHA256 does not match the supplied parent manifest")
            print(f"Valid RQ1 preflight manifest: {args.preflight_manifest}")
            return 0
        if not args.output:
            raise ValueError("building requires --output")
        preflight = build_preflight_manifest(args.manifest)
        digest = write_preflight(preflight, args.output, overwrite=args.overwrite)
        print(f"Wrote {args.output} (20 records, sha256={digest})")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
