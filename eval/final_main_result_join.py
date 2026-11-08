"""Strict, in-memory joins between frozen condition metadata and results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_MANIFEST_SHA256 = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"


def load_condition_manifest(
    path: Path, *, verify_hash: bool = True
) -> tuple[list[dict[str, Any]], str]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if verify_hash and digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Frozen condition manifest SHA256 mismatch: {digest}")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Condition manifest must contain JSON objects")
    ids = [row.get("condition_id") for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Condition manifest has duplicate or missing condition_id")
    return rows, digest


def join_results_to_manifest(result_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_by_id = {row["condition_id"]: row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("Condition manifest has duplicate condition_id")
    seen: set[str] = set()
    joined = []
    for result in result_rows:
        if not isinstance(result, dict):
            raise ValueError("Inference result rows must be JSON objects")
        condition_id = result.get("condition_id")
        if not condition_id or condition_id in seen:
            raise ValueError(f"Duplicate or missing result condition_id: {condition_id}")
        if condition_id not in manifest_by_id:
            raise ValueError(
                f"Unknown result condition_id (no frozen manifest match): {condition_id}"
            )
        seen.add(condition_id)
        # Result data remains untouched; metadata is layered into a new object.
        metadata = manifest_by_id[condition_id]
        joined.append({**result, **metadata, "result": result, "manifest": metadata})
    return joined


def join_result_file(result_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], str]:
    result_rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest_rows, digest = load_condition_manifest(manifest_path)
    return join_results_to_manifest(result_rows, manifest_rows), digest


__all__ = ["EXPECTED_MANIFEST_SHA256", "join_result_file", "join_results_to_manifest", "load_condition_manifest"]
