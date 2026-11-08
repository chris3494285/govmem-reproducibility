from __future__ import annotations

import hashlib
import json

import pytest

from scripts import build_rq1_final_manifest as final_mod


def _parent_and_rows():
    rows = []
    for dataset, task in final_mod.ELIGIBLE_STRATA:
        for index in range(30):
            example_id = f"{dataset}-{task}-{index:02d}"
            rows.append({"dataset": dataset, "task_type": task, "example_id": example_id,
                         "query_id": example_id, "source_index": index})
    parent = {"manifest_version": final_mod.PARENT_MANIFEST_VERSION, "records": rows}
    return parent, rows


def _fake_profile(rows):
    return {"eligible_examples": [
        {**row, "eligible": True, "interaction_count": index + 1}
        for index, row in enumerate(rows)
    ]}


def test_final_manifest_is_exact_stratified_and_deterministic(monkeypatch):
    parent, rows = _parent_and_rows()
    digest = "frozen-parent"
    monkeypatch.setattr(final_mod, "FROZEN_PARENT_SHA256", digest)
    monkeypatch.setattr(final_mod, "_validate_parent", lambda path: (parent, b"", digest))
    monkeypatch.setattr(final_mod, "profile", lambda *args: _fake_profile(rows))
    first = final_mod.build_final_manifest("parent", "memora", "conflict")
    second = final_mod.build_final_manifest("parent", "memora", "conflict")
    assert first == second
    assert first["selected_n"] == 90
    assert first["counts"] == {"memora:remembering": 30, "memconflict:dynamic": 30, "memconflict:conditional": 30}
    assert {row["task_type"] for row in first["records"]} == {"remembering", "dynamic", "conditional"}
    assert len({row["example_id"] for row in first["records"]}) == 90
    assert min(row["interaction_count"] for row in first["records"]) == 1
    assert max(row["interaction_count"] for row in first["records"]) == 90


def test_even_rank_selection_covers_low_middle_and_high_and_ties_by_id():
    rows = [{"example_id": f"id-{index:02d}", "interaction_count": index // 2} for index in range(10)]
    chosen = final_mod.select_stratum(rows, 4)
    assert [row["example_id"] for row in chosen] == ["id-00", "id-03", "id-06", "id-09"]
    assert final_mod.even_rank_indices(100, 30)[0] == 0
    assert final_mod.even_rank_indices(100, 30)[-1] == 99


def test_parent_hash_mismatch_and_duplicate_ids_fail(tmp_path, monkeypatch):
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps({"manifest_version": final_mod.PARENT_MANIFEST_VERSION}), encoding="utf-8")
    monkeypatch.setattr(final_mod, "validate_manifest", lambda manifest: None)
    with pytest.raises(ValueError, match="frozen RQ1"):
        final_mod._validate_parent(parent_path)

    manifest = {
        "manifest_version": final_mod.MANIFEST_VERSION,
        "parent_manifest_version": final_mod.PARENT_MANIFEST_VERSION,
        "parent_manifest_sha256": final_mod.FROZEN_PARENT_SHA256,
        "selected_n": 90,
        "records": [{"dataset": "memora", "task_type": "remembering", "example_id": "same"}] * 90,
    }
    with pytest.raises(ValueError, match="unique"):
        final_mod.validate_final_manifest(manifest)


def test_final_manifest_serialization_is_byte_stable(tmp_path, monkeypatch):
    parent, rows = _parent_and_rows()
    monkeypatch.setattr(final_mod, "FROZEN_PARENT_SHA256", "frozen-parent")
    monkeypatch.setattr(final_mod, "_validate_parent", lambda path: (parent, b"", "frozen-parent"))
    monkeypatch.setattr(final_mod, "profile", lambda *args: _fake_profile(rows))
    manifest = final_mod.build_final_manifest("parent", "memora", "conflict")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_digest = final_mod.write_manifest(manifest, first_path)
    second_digest = final_mod.write_manifest(manifest, second_path)
    assert first_digest == second_digest
    assert first_path.read_bytes() == second_path.read_bytes()
