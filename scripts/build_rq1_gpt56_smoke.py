#!/usr/bin/env python3
"""Build the deterministic three-example GPT-5.6 Sol integration subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

FINAL_MANIFEST_VERSION = "rq1-final-90-v1"
SMOKE_MANIFEST_VERSION = "rq1-gpt56-smoke-v1"
FROZEN_FINAL_SHA256 = "6f119abcbc94dd22852b49613f6599e821053c36b0807ac9167334ae08257d4a"
SMOKE_STRATA = (("memora", "remembering"), ("memconflict", "dynamic"), ("memconflict", "conditional"))


def build_smoke_manifest(parent_path: str | Path) -> dict[str, Any]:
    payload = Path(parent_path).read_bytes()
    parent = json.loads(payload.decode("utf-8"))
    parent_sha256 = hashlib.sha256(payload).hexdigest()
    if parent_sha256 != FROZEN_FINAL_SHA256:
        raise ValueError("Parent manifest does not match the frozen final-90 SHA256")
    if parent.get("manifest_version") != FINAL_MANIFEST_VERSION or parent.get("selected_n") != 90:
        raise ValueError("Unsupported final-90 parent manifest")
    examples = []
    for dataset, task_type in SMOKE_STRATA:
        candidates = sorted(
            (row for row in parent["records"] if row.get("dataset") == dataset and row.get("task_type") == task_type),
            key=lambda row: str(row.get("example_id")),
        )
        if not candidates:
            raise ValueError(f"No parent record for {dataset}/{task_type}")
        examples.append(dict(candidates[0]))
    return {
        "manifest_version": SMOKE_MANIFEST_VERSION,
        "parent_manifest_sha256": parent_sha256,
        "selection_rule": "first_stable_sorted_example_id_per_required_stratum",
        "examples": examples,
    }


def validate_smoke_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("manifest_version") != SMOKE_MANIFEST_VERSION:
        raise ValueError("Unsupported GPT-5.6 smoke manifest version")
    if manifest.get("parent_manifest_sha256") != FROZEN_FINAL_SHA256:
        raise ValueError("GPT-5.6 smoke parent SHA256 does not match the frozen final manifest")
    examples = manifest.get("examples")
    if not isinstance(examples, list) or len(examples) != 3:
        raise ValueError("GPT-5.6 smoke manifest must contain exactly three records")
    identities = {(row.get("dataset"), row.get("task_type")) for row in examples}
    if identities != set(SMOKE_STRATA):
        raise ValueError(f"GPT-5.6 smoke strata are invalid: {identities}")
    ids = [str(row.get("example_id", "")) for row in examples]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("GPT-5.6 smoke IDs must be present and unique")


def write_smoke_manifest(manifest: dict[str, Any], output: str | Path, *, overwrite: bool = False) -> str:
    validate_smoke_manifest(manifest)
    path = Path(output)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Smoke manifest already exists; use --overwrite explicitly: {path}")
    serialized = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(serialized, encoding="utf-8")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_smoke_manifest(args.parent_manifest)
    digest = write_smoke_manifest(manifest, args.output, overwrite=args.overwrite)
    print(json.dumps({"output": str(args.output), "sha256": digest, "examples": manifest["examples"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
