from __future__ import annotations

import hashlib
import json

from eval.interfaces import AdaptedExample, Query, RawInteraction
from scripts import profile_rq1_workload as profiler


def _example(query_id: str, dataset: str, task_type: str, count: int) -> AdaptedExample:
    interactions = tuple(
        RawInteraction(f"{query_id}-i{index}", "u", f"message {index}", None)
        for index in range(count)
    )
    return AdaptedExample(
        query=Query(query_id, "u", "question", None, metadata={
            "dataset": dataset, "task_type": task_type, "conflict_type": task_type,
        }),
        interactions=interactions,
        oracle_governance=(),
        metadata={"dataset": dataset},
    )


def _manifest(records):
    return {
        "manifest_version": "rq1-official-pilot-v1",
        "seed": 20260810,
        "records": records,
        "counts": {"total": len(records)},
    }


def test_profile_is_local_deterministic_and_uses_applicability(monkeypatch, tmp_path):
    examples = [
        _example("m1", "memora", "remembering", 2),
        _example("m2", "memora", "reasoning", 4),
        _example("c1", "memconflict", "dynamic", 8),
        _example("c2", "memconflict", "static", 16),
    ]
    monkeypatch.setattr(profiler, "validate_manifest", lambda manifest: None)
    monkeypatch.setattr(profiler, "ADAPTERS", {
        "memora": lambda: type("Adapter", (), {"load": lambda self, path: examples[:2]})(),
        "memconflict": lambda: type("Adapter", (), {"load": lambda self, path: examples[2:]})(),
    })
    records = [
        {"dataset": "memora", "example_id": "m1", "task_type": "remembering", "source_index": 0},
        {"dataset": "memora", "example_id": "m2", "task_type": "reasoning", "source_index": 1},
        {"dataset": "memconflict", "example_id": "c1", "task_type": "dynamic", "source_index": 0},
        {"dataset": "memconflict", "example_id": "c2", "task_type": "static", "source_index": 1},
    ]
    manifest = _manifest(records)
    manifest_path = tmp_path / "manifest.json"
    payload = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    manifest_path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(profiler, "FROZEN_PARENT_SHA256", hashlib.sha256(payload.encode()).hexdigest())

    first = profiler.profile(manifest_path, "memora", "conflict")
    second = profiler.profile(manifest_path, "memora", "conflict")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["manifest_sha256"] == hashlib.sha256(payload.encode()).hexdigest()
    assert first["selected_n"] == 4
    assert first["eligible_n"] == 2
    assert first["skipped_n"] == 2
    assert first["datasets"]["memora"]["by_task_type"]["remembering"]["eligible_n"] == 1
    assert first["datasets"]["memora"]["by_task_type"]["reasoning"]["skipped_n"] == 1
    assert first["workload_distribution"]["interaction_count"]["median"] == 5.0
    assert [row["query_id"] for row in first["eligible_examples"]] == ["c1", "m1"]


def test_percentiles_are_deterministic():
    assert profiler.distribution([1, 2, 3, 4]) == {
        "n": 4, "min": 1, "median": 2.5, "p75": 3.25,
        "p90": 3.7, "p95": 3.85, "max": 4,
    }


def test_profiler_does_not_construct_or_call_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(profiler, "validate_manifest", lambda manifest: None)
    monkeypatch.setattr(profiler, "ADAPTERS", {
        "memora": lambda: type("Adapter", (), {"load": lambda self, path: [_example("m", "memora", "remembering", 1)]})(),
        "memconflict": lambda: type("Adapter", (), {"load": lambda self, path: []})(),
    })
    records = [{"dataset": "memora", "example_id": "m", "task_type": "remembering"}]
    manifest_path = tmp_path / "manifest.json"
    payload = json.dumps(_manifest(records))
    manifest_path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(profiler, "FROZEN_PARENT_SHA256", hashlib.sha256(payload.encode()).hexdigest())
    profiler.profile(manifest_path, "memora", "conflict")
