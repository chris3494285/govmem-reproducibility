from __future__ import annotations

import json
from pathlib import Path

import pytest

from govmem.memory_systems.mem0 import Mem0Adapter
from scripts.run_system_validation_formal import run_formal
from scripts.run_system_validation_smoke import _native_reuse_invariant_failures, run_smoke


MANIFEST = Path("data/system_validation_frozen_v1.jsonl")
METADATA = Path("data/system_validation_frozen_v1_metadata.json")


class FormalBackend:
    def __init__(self, *, empty: bool = False, error: bool = False):
        self.by_user = {}
        self.empty = empty
        self.error = error
        self.calls = []
        self.close_calls = 0

    def reset(self):
        self.by_user.clear()

    def add(self, content, *, user_id, metadata, infer):
        assert infer is False
        self.by_user.setdefault(user_id, []).append({"id": f"{user_id}-memory", "memory": content, "metadata": metadata})

    def search(self, query, *, filters, top_k=20):
        self.calls.append((query, filters, top_k))
        if self.error:
            raise RuntimeError("native retrieval failed")
        if self.empty:
            return []
        return self.by_user.get(filters["user_id"], [])[:top_k]

    def get_all(self):
        return self.by_user

    def close(self):
        self.close_calls += 1


def _run(tmp_path, backend, **kwargs):
    return run_formal(
        memory_system="mem0",
        manifest=MANIFEST,
        metadata=METADATA,
        results_dir=tmp_path,
        retrieval_k=1,
        adapter_factory=lambda config: Mem0Adapter(backend),
        **kwargs,
    )


def test_frozen_formal_run_has_provenance_and_strict_budget(tmp_path):
    backend = FormalBackend()
    result = _run(tmp_path, backend)
    assert result["summary"]["status"] == "FROZEN / FORMAL"
    assert result["summary"]["cohort_version"] == "v1"
    assert result["summary"]["seed"] == 42
    assert result["summary"]["requested_retrieval_k"] == 1
    assert result["summary"]["max_native_retrieved_memory_count"] <= 1
    assert result["summary"]["manifest_sha256"] == json.loads(METADATA.read_text())["manifest_sha256"]
    assert len(result["records"]) == 180
    assert all(record["experiment_status"] == "FROZEN / FORMAL" for record in result["records"])
    assert all(record["cohort_version"] == "v1" for record in result["records"])
    assert all(record["requested_retrieval_k"] == 1 for record in result["records"])
    assert not (tmp_path / "system_validation_smoke.json").exists()


def test_frozen_validation_rejects_bad_metadata_before_adapter_execution(tmp_path):
    bad_metadata = tmp_path / "metadata.json"
    payload = json.loads(METADATA.read_text())
    payload["status"] = "NON-FROZEN / SMOKE"
    bad_metadata.write_text(json.dumps(payload))
    called = False

    def factory(config):
        nonlocal called
        called = True
        raise AssertionError("adapter must not be constructed")

    with pytest.raises(ValueError, match="status"):
        run_formal(memory_system="mem0", manifest=MANIFEST, metadata=bad_metadata, results_dir=tmp_path / "out", retrieval_k=1, adapter_factory=factory)
    assert called is False


def test_wrong_manifest_hash_is_rejected_before_adapter_execution(tmp_path):
    bad_metadata = tmp_path / "metadata.json"
    payload = json.loads(METADATA.read_text())
    payload["manifest_sha256"] = "0" * 64
    bad_metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA256"):
        run_formal(memory_system="mem0", manifest=MANIFEST, metadata=bad_metadata, results_dir=tmp_path / "out", retrieval_k=1, adapter_factory=lambda config: (_ for _ in ()).throw(AssertionError("not called")))


def test_non_positive_retrieval_budget_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        run_formal(memory_system="mem0", manifest=MANIFEST, metadata=METADATA, results_dir=tmp_path, retrieval_k=0, adapter_factory=lambda config: FormalBackend())


@pytest.mark.parametrize("backend", [FormalBackend(empty=True), FormalBackend(error=True)])
def test_empty_or_error_native_retrieval_fails_formal_run(tmp_path, backend):
    with pytest.raises(RuntimeError, match="retrieval"):
        _run(tmp_path, backend)


def test_formal_output_overwrite_requires_force(tmp_path):
    _run(tmp_path, FormalBackend())
    with pytest.raises(FileExistsError):
        _run(tmp_path, FormalBackend())
    _run(tmp_path, FormalBackend(), force=True)


def _records_with_native_variation(field: str):
    base = {
        "base_example_id": "base-1",
        "condition": "clean",
        "native_retrieved_memory_ids": ["m1"],
        "native_retrieved_contents": ["content"],
        "native_raw_payload": {"id": "m1"},
    }
    fault = dict(base, condition="fault")
    fault[field] = ["different"] if field != "native_raw_payload" else {"id": "different"}
    return [base, fault]


@pytest.mark.parametrize("field", ["native_retrieved_memory_ids", "native_retrieved_contents", "native_raw_payload"])
def test_native_reuse_invariant_reports_each_mismatch(field):
    failures = _native_reuse_invariant_failures(_records_with_native_variation(field))
    assert failures == [{"base_example_id": "base-1", "fields": [field]}]


def test_native_reuse_invariant_ignores_expected_governance_context_changes():
    records = _records_with_native_variation("native_raw_payload")
    records[1]["native_raw_payload"] = records[0]["native_raw_payload"]
    records[0]["final_model_visible_context"] = "[]"
    records[1]["final_model_visible_context"] = "[\"admitted\"]"
    records[0]["governance_action"] = "governance_deny"
    records[1]["governance_action"] = "governance_admit"
    assert _native_reuse_invariant_failures(records) == []


def test_smoke_non_strict_reports_native_reuse_failures_without_raising(monkeypatch, tmp_path):
    from scripts import run_system_validation_smoke as smoke

    original = smoke._condition_record

    def altered(*args, **kwargs):
        record = original(*args, **kwargs)
        if record["condition"] != "clean":
            record["native_raw_payload"] = {"altered": True}
        return record

    monkeypatch.setattr(smoke, "_condition_record", altered)
    result = run_smoke(
        manifest="data/system_validation_smoke_non_frozen.json",
        results_dir=tmp_path,
        limit_base=1,
        adapter_factory=lambda config: Mem0Adapter(FormalBackend()),
    )
    assert result["summary"]["invariant_failure_n"] == 1


def test_formal_strict_mode_raises_on_native_reuse_failure(monkeypatch, tmp_path):
    from scripts import run_system_validation_smoke as smoke

    original = smoke._condition_record

    def altered(*args, **kwargs):
        record = original(*args, **kwargs)
        if record["condition"] != "clean":
            record["native_retrieved_memory_ids"] = ["mismatch"]
        return record

    monkeypatch.setattr(smoke, "_condition_record", altered)
    backend = FormalBackend()
    adapter_holder = {}

    def factory(config):
        adapter = Mem0Adapter(backend)
        original_close = adapter.close
        adapter.close_calls = 0

        def close_once():
            adapter.close_calls += 1
            original_close()

        adapter.close = close_once
        adapter_holder["adapter"] = adapter
        return adapter

    with pytest.raises(RuntimeError, match="base_example_id|Formal sibling"):
        run_formal(
            memory_system="mem0", manifest=MANIFEST, metadata=METADATA,
            results_dir=tmp_path, retrieval_k=1, adapter_factory=factory,
        )
    assert adapter_holder["adapter"].close_calls == 1
