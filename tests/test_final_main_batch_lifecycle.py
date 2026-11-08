from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.final_main_batch_lifecycle import _write, execute_batch_lifecycle


class _ModelDump:
    def __init__(self, **fields):
        self.fields = fields

    def model_dump(self):
        return self.fields


class _Files:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return {"id": "file-1", "status": "uploaded", "created_at": 11}

    def retrieve(self, file_id):
        self.calls.append(("retrieve", file_id))
        return {"id": file_id, "status": next(self.statuses), "created_at": 11}


class _Batches:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"id": "batch-1", "status": "validating", "created_at": 12}


class _Client:
    def __init__(self, statuses):
        self.files = _Files(statuses)
        self.batches = _Batches()


class _Provider:
    def __init__(self, client):
        self.client = client

    def _client(self):
        return self.client


def _request():
    return [{"custom_id": "custom-1", "method": "POST", "url": "/v1/responses", "body": {}}]


def test_submit_waits_for_processed_file_before_creating_batch(tmp_path: Path):
    client = _Client(["uploaded", "processed"])
    metadata, lines = execute_batch_lifecycle(
        action="submit", provider=_Provider(client), requests=_request(), reverse={"custom-1": "condition-1"},
        results_dir=tmp_path, model="gpt-5.6-sol", manifest_sha="sha", file_poll_seconds=0,
    )
    assert lines is None
    assert metadata["batch_id"] == "batch-1"
    assert [call[0] for call in client.files.calls] == ["create", "retrieve", "retrieve"]
    assert len(client.batches.calls) == 1
    assert client.files.calls[-1][0] == "retrieve"
    saved = json.loads((tmp_path / "batch_submission.json").read_text())
    assert saved["input_file_id"] == "file-1"
    assert saved["batch_id"] == "batch-1"


def test_failed_file_never_creates_batch(tmp_path: Path):
    client = _Client(["failed"])
    with pytest.raises(RuntimeError, match="did not process"):
        execute_batch_lifecycle(
            action="submit", provider=_Provider(client), requests=_request(), reverse={"custom-1": "condition-1"},
            results_dir=tmp_path, model="gpt-5.6-sol", manifest_sha="sha", file_poll_seconds=0,
        )
    assert client.batches.calls == []
    saved = json.loads((tmp_path / "batch_submission.json").read_text())
    assert saved["file_status"] == "failed"


def test_existing_batch_blocks_duplicate_submission(tmp_path: Path):
    client = _Client([])
    (tmp_path / "batch_submission.json").write_text(json.dumps({"batch_id": "batch-existing"}))
    with pytest.raises(ValueError, match="already exists"):
        execute_batch_lifecycle(
            action="submit", provider=_Provider(client), requests=_request(), reverse={"custom-1": "condition-1"},
            results_dir=tmp_path, model="gpt-5.6-sol", manifest_sha="sha", file_poll_seconds=0,
        )
    assert client.files.calls == []


def test_write_serializes_nested_sdk_models_as_structured_json(tmp_path: Path):
    path = tmp_path / "batch_submission.json"
    _write(path, {
        "request_counts": _ModelDump(completed=3, failed=1, total=4),
        "errors": [_ModelDump(code="bad_request", message="invalid input")],
        "nested": {"file": _ModelDump(id="file-1", status="processed")},
    })
    saved = json.loads(path.read_text())
    assert saved["request_counts"] == {"completed": 3, "failed": 1, "total": 4}
    assert saved["errors"] == [{"code": "bad_request", "message": "invalid input"}]
    assert saved["nested"]["file"]["status"] == "processed"


def test_write_serializes_tuple_and_nested_model_dump_values(tmp_path: Path):
    path = tmp_path / "batch_submission.json"
    _write(path, {"values": (_ModelDump(value=_ModelDump(answer="ok")), 1)})
    assert json.loads(path.read_text()) == {"values": [{"value": {"answer": "ok"}}, 1]}
