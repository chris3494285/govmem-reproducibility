"""Explicit, resumable OpenAI Batch file/batch lifecycle."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from eval.final_main_batch import read_batch_lines, write_batch_input


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _jsonable(value: Any, *, _seen: set[int] | None = None) -> Any:
    """Convert OpenAI/Pydantic SDK values into JSON-safe structured data.

    Batch status objects (notably ``request_counts`` and error objects) are
    often Pydantic models rather than plain dictionaries.  Keep their field
    structure instead of stringifying it before writing local lifecycle
    metadata.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if _seen is None:
        _seen = set()
    marker = id(value)
    if marker in _seen:
        raise TypeError("Cannot serialize cyclic Batch metadata")
    if isinstance(value, dict):
        return {str(key): _jsonable(item, _seen=_seen) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, _seen=_seen) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        _seen.add(marker)
        try:
            return _jsonable(model_dump(), _seen=_seen)
        finally:
            _seen.remove(marker)

    # Some SDK versions expose a plain object with public attributes rather
    # than model_dump().  Preserve that structure when available.
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        _seen.add(marker)
        try:
            return {
                str(key): _jsonable(item, _seen=_seen)
                for key, item in attributes.items()
                if not str(key).startswith("_")
            }
        finally:
            _seen.remove(marker)
    # This is only reached for an opaque scalar-like SDK value with no
    # structured representation; structured SDK models take the branches
    # above.
    return str(value)


def _write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Batch metadata is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Malformed Batch metadata")
    return data


def execute_batch_lifecycle(*, action: str, provider: Any, requests: list[dict[str, Any]], reverse: dict[str, str], results_dir: Path, model: str, manifest_sha: str, force_new_batch: bool = False, batch_id: str | None = None, file_poll_seconds: float = 5.0, file_timeout_seconds: float = 300.0) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    if action not in {"submit", "status", "retrieve"}:
        raise ValueError("Batch action must be submit, status, or retrieve")
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = results_dir / "batch_submission.json"
    existing = _load(metadata_path) if metadata_path.exists() else None
    if action == "submit":
        if existing and existing.get("batch_id") and not force_new_batch:
            raise ValueError("A Batch already exists in metadata; use status/retrieve or --force-new-batch")
        input_path = results_dir / "batch_input.jsonl"
        write_batch_input(requests, input_path)
        client = provider._client()
        with input_path.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        metadata = {"manifest_sha256": manifest_sha, "model": model, "input_file_id": _get(uploaded, "id"), "file_status": _get(uploaded, "status"), "file_created_at": _get(uploaded, "created_at"), "custom_id_to_condition_id": reverse, "submitted_request_count": len(requests)}
        _write(metadata_path, metadata)
        deadline = time.monotonic() + file_timeout_seconds
        while metadata.get("file_status") != "processed":
            if metadata.get("file_status") in {"error", "failed", "cancelled"}:
                raise RuntimeError(f"Batch input file did not process: {metadata.get('file_status')}")
            if time.monotonic() >= deadline:
                metadata["file_status"] = "timeout"
                _write(metadata_path, metadata)
                raise TimeoutError("Timed out waiting for Batch input file to process")
            time.sleep(file_poll_seconds)
            file_obj = client.files.retrieve(metadata["input_file_id"])
            metadata["file_status"] = _get(file_obj, "status")
            metadata["file_created_at"] = _get(file_obj, "created_at", metadata.get("file_created_at"))
            _write(metadata_path, metadata)
        batch = client.batches.create(input_file_id=metadata["input_file_id"], endpoint="/v1/responses", completion_window="24h")
        metadata.update({"batch_id": _get(batch, "id"), "batch_status": _get(batch, "status"), "created_at": _get(batch, "created_at")})
        _write(metadata_path, metadata)
        return metadata, None
    metadata = existing or {}
    chosen_batch_id = batch_id or metadata.get("batch_id")
    if not chosen_batch_id:
        raise ValueError("A batch_id is required for status/retrieve")
    if metadata and metadata.get("manifest_sha256") not in {None, manifest_sha}:
        raise ValueError("Batch metadata manifest mismatch")
    client = provider._client()
    batch = client.batches.retrieve(chosen_batch_id)
    metadata.update({"batch_id": chosen_batch_id, "batch_status": _get(batch, "status"), "output_file_id": _get(batch, "output_file_id"), "error_file_id": _get(batch, "error_file_id"), "request_counts": _get(batch, "request_counts")})
    _write(metadata_path, metadata)
    if action == "status":
        return metadata, None
    if metadata.get("batch_status") != "completed":
        raise RuntimeError(f"Batch is not completed: {metadata.get('batch_status')}")
    content = client.files.content(metadata.get("output_file_id"))
    return metadata, read_batch_lines(content)
