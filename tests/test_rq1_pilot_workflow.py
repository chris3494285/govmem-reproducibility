from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts import build_rq1_pilot_manifest as manifest_mod
from scripts.run_rq1_pilot import run_pilot
from scripts.run_rq1_pilot import CONDITIONS
from scripts.summarize_rq1_pilot import markdown, summarize
from scripts.build_rq1_pilot_preflight import FROZEN_PARENT_SHA256, build_preflight_manifest, validate_preflight
from scripts.run_reconstruction_batch import run_batch
from scripts.build_rq1_final_manifest import FROZEN_PARENT_SHA256 as FINAL_PARENT_SHA256, MANIFEST_VERSION as FINAL_MANIFEST_VERSION


class _BatchReconstructor:
    model = "mock"

    def reconstruct(self, input_data):
        return ()


def _rows(dataset: str, counts: dict[str, int], *, reverse: bool = False):
    rows = []
    for task_type, count in counts.items():
        for index in range(count):
            rows.append({
                "dataset": dataset,
                "example_id": f"{dataset}-{task_type}-{index:04d}",
                "query_id": f"{dataset}-{task_type}-{index:04d}",
                "task_type": task_type,
                "source_index": index,
            })
    return list(reversed(rows)) if reverse else rows


def _valid_manifest():
    return {
        "manifest_version": manifest_mod.MANIFEST_VERSION,
        "seed": 19,
        "sampling": {},
        "expected_counts": {},
        "sources": {"memora": "memora-source", "memconflict": "conflict-source"},
        "records": _rows("memora", manifest_mod.EXPECTED_MEMORA_COUNTS)
        + _rows("memconflict", manifest_mod.PILOT_MEMCONFLICT_COUNTS),
        "counts": {
            "total": 450,
            "memora": 150,
            "memconflict": 300,
            "memora_by_task_type": manifest_mod.EXPECTED_MEMORA_COUNTS,
            "memconflict_by_task_type": manifest_mod.PILOT_MEMCONFLICT_COUNTS,
        },
    }


def test_manifest_has_exact_strata_and_is_order_independent(monkeypatch):
    original = manifest_mod._stable_records

    def fake(dataset, source, *, strict_counts=True):
        counts = manifest_mod.EXPECTED_MEMORA_COUNTS if dataset == "memora" else manifest_mod.PILOT_MEMCONFLICT_COUNTS
        if dataset == "memconflict":
            counts = {"dynamic": 150, "static": 150, "conditional": 150}
        return _rows(dataset, counts, reverse=str(source).endswith("reverse"))

    monkeypatch.setattr(manifest_mod, "_stable_records", fake)
    first = manifest_mod.build_manifest("memora", "conflict", seed=123, strict_counts=True)
    second = manifest_mod.build_manifest("memora", "conflict-reverse", seed=123, strict_counts=True)
    manifest_mod.validate_manifest(first)
    assert first["counts"]["total"] == 450
    assert first["counts"]["memora_by_task_type"] == {"remembering": 50, "reasoning": 50, "recommending": 50}
    assert first["counts"]["memconflict_by_task_type"] == {"dynamic": 100, "static": 100, "conditional": 100}
    assert [(row["dataset"], row["example_id"]) for row in first["records"]] == [
        (row["dataset"], row["example_id"]) for row in second["records"]
    ]
    different = manifest_mod.build_manifest("memora", "conflict", seed=124, strict_counts=True)
    assert [row["example_id"] for row in first["records"] if row["dataset"] == "memconflict"] != [
        row["example_id"] for row in different["records"] if row["dataset"] == "memconflict"
    ]
    assert original is not None


def test_manifest_fails_duplicate_or_insufficient_strata(monkeypatch):
    duplicate = _rows("memora", {"remembering": 2})
    duplicate[1]["example_id"] = duplicate[0]["example_id"]

    class FakeAdapter:
        def load(self, source):
            return []

    monkeypatch.setitem(manifest_mod.ADAPTERS, "memora", FakeAdapter)
    with pytest.raises(ValueError, match="Duplicate stable"):
        # Exercise the production duplicate detector directly with adapter-like
        # data through a patched loader.
        class DuplicateAdapter:
            def load(self, source):
                from eval.interfaces import AdaptedExample, Query
                return [
                    AdaptedExample(Query("same", "u", "q", None, metadata={"dataset": "memora", "task_type": "remembering"}), (), ()),
                    AdaptedExample(Query("same", "u", "q2", None, metadata={"dataset": "memora", "task_type": "remembering"}), (), ()),
                ]
        monkeypatch.setitem(manifest_mod.ADAPTERS, "memora", DuplicateAdapter)
        manifest_mod._stable_records("memora", "fixture", strict_counts=False)

    with pytest.raises(ValueError, match="Insufficient"):
        manifest_mod._sample(_rows("memconflict", {"dynamic": 1, "static": 1, "conditional": 1}),
                             {"dynamic": 2, "static": 1, "conditional": 1}, 1)


def test_existing_manifest_is_not_overwritten(tmp_path):
    path = tmp_path / "pilot.json"
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        manifest_mod.write_manifest(_valid_manifest(), path)


def test_pilot_runner_uses_manifest_membership_and_applies_limit_after_selection(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")
    calls = []

    def fake_batch(**kwargs):
        calls.append(kwargs)
        return {"selected": len(kwargs["query_ids"]), "executed": 0, "cache_hits": 0, "valid": 0}

    result = run_pilot(
        manifest_path=path,
        memora_input="memora-source",
        memconflict_input="conflict-source",
        results_dir=tmp_path / "results",
        dataset="memconflict",
        limit=7,
        batch_runner=fake_batch,
    )
    assert result["selected"] == 7
    assert len(calls) == 1
    assert calls[0]["dataset"] == "memconflict"
    assert len(calls[0]["query_ids"]) == 7
    assert set(calls[0]["query_ids"]).issubset({row["example_id"] for row in _valid_manifest()["records"] if row["dataset"] == "memconflict"})
    assert calls[0]["run_metadata"]["pilot_manifest_version"] == manifest_mod.MANIFEST_VERSION


def test_preflight_is_exactly_20_and_conditions_share_membership(tmp_path):
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")
    assert FROZEN_PARENT_SHA256 == "8bb6665dfa0c0a8ff92f3a77eb34f2ad110b51914b865125080d57e2c41b24af"
    with pytest.raises(ValueError, match="does not match frozen"):
        build_preflight_manifest(parent_path)
    preflight = build_preflight_manifest(parent_path, require_frozen_hash=False)
    validate_preflight(preflight, _valid_manifest())
    assert len(preflight["records"]) == 20
    assert preflight["allocation"] == {
        "memora": {"remembering": 4, "reasoning": 3, "recommending": 3},
        "memconflict": {"dynamic": 4, "static": 3, "conditional": 3},
    }
    selection = [(row["dataset"], row["example_id"]) for row in preflight["records"]]
    calls = []

    def fake_batch(**kwargs):
        calls.append(kwargs)
        return {"selected": len(kwargs["query_ids"]), "executed": 0, "cache_hits": 0, "valid": 0}

    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    for condition in CONDITIONS:
        run_pilot(
            manifest_path=parent_path, preflight=True, preflight_manifest_path=preflight_path,
            memora_input="memora-source", memconflict_input="conflict-source",
            results_dir=tmp_path / condition, condition=condition,
            batch_runner=fake_batch, require_frozen_manifest=False,
        )
    memberships = [set(call["query_ids"]) for call in calls]
    assert len(memberships) == 2  # two datasets for the reconstruction condition
    assert len(memberships[0] | memberships[1]) == 20
    assert set().union(*memberships).issubset({row["example_id"] for row in _valid_manifest()["records"]})
    assert all(call["condition"] == "reconstruction" for call in calls)
    assert {str(call["cache_dir"]) for call in calls} == {".cache/reconstruction"}
    assert all("governance_applied" not in call["run_metadata"] for call in calls)
    assert all("governance_policy_version" not in call["run_metadata"] for call in calls)


def test_final_manifest_selects_only_frozen_90_ids_without_provider_calls(tmp_path, monkeypatch):
    import scripts.run_rq1_pilot as pilot_module
    parent = _valid_manifest()
    records = [
        row for row in parent["records"]
        if (row["dataset"], row["task_type"]) in {
            ("memora", "remembering"),
            ("memconflict", "dynamic"),
            ("memconflict", "conditional"),
        }
    ]
    records = [
        *[row for row in records if row["dataset"] == "memora"][:30],
        *[row for row in records if row["dataset"] == "memconflict" and row["task_type"] == "dynamic"][:30],
        *[row for row in records if row["dataset"] == "memconflict" and row["task_type"] == "conditional"][:30],
    ]
    final = {
        "manifest_version": FINAL_MANIFEST_VERSION,
        "parent_manifest_version": manifest_mod.MANIFEST_VERSION,
        "parent_manifest_sha256": FINAL_PARENT_SHA256,
        "selected_n": 90,
        "records": records,
        "counts": {"memora:remembering": 30, "memconflict:dynamic": 30, "memconflict:conditional": 30},
    }
    path = tmp_path / "final.json"
    path.write_text(json.dumps(final), encoding="utf-8")
    monkeypatch.setattr(pilot_module, "FROZEN_FINAL_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    calls = []

    def fake_batch(**kwargs):
        calls.append(kwargs)
        return {"selected": len(kwargs["query_ids"]), "executed": 0, "cache_hits": 0, "valid": 0}

    result = run_pilot(
        manifest_path=path, memora_input="memora-source", memconflict_input="conflict-source",
        results_dir=tmp_path / "final-results", condition="reconstruction", batch_runner=fake_batch,
        require_frozen_manifest=True,
    )
    assert result["selected"] == 90
    assert [call["condition"] for call in calls] == ["reconstruction", "reconstruction"]
    assert all(call["governance_applied"] is False for call in calls)
    assert {call["dataset"] for call in calls} == {"memora", "memconflict"}
    assert {call["dataset"]: len(call["query_ids"]) for call in calls} == {"memora": 30, "memconflict": 60}
    assert all("static" not in query_id for query_id in calls[1]["query_ids"])
    assert all("reasoning" not in query_id and "recommending" not in query_id for query_id in calls[0]["query_ids"])
    assert all("paid_preflight" not in str(call["output_path"]) for call in calls)


def test_condition_validation_and_reserved_metadata_protection(tmp_path):
    assert CONDITIONS == ("reconstruction",)
    for disabled in ("baseline", "govmem"):
        with pytest.raises(ValueError, match="disabled"):
            run_pilot(
                manifest_path=tmp_path / "missing.json", memora_input=None, memconflict_input=None,
                results_dir=tmp_path, condition=disabled,
            )
    with pytest.raises(ValueError, match="Unknown condition"):
        run_pilot(
            manifest_path=tmp_path / "missing.json", memora_input=None, memconflict_input=None,
            results_dir=tmp_path, condition="other",
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_pilot(
            manifest_path=tmp_path / "missing.json", memora_input=None, memconflict_input=None,
            results_dir=tmp_path, condition="reconstruction", preflight=True, limit=1,
        )
    with pytest.raises(ValueError, match="reserved result fields"):
        run_batch(
            dataset="gatemem", input_path="tests/fixtures/gatemem-mini", output_path=tmp_path / "rows.jsonl",
            limit=1, condition="baseline", run_metadata={"query_id": "overwrite"},
            reconstructor=_BatchReconstructor(),
        )
    output = tmp_path / "condition.jsonl"
    run_batch(
        dataset="gatemem", input_path="tests/fixtures/gatemem-mini", output_path=output,
        limit=1, condition="reconstruction", governance_applied=False,
        run_metadata={"pilot_scope": "preflight"},
        reconstructor=_BatchReconstructor(),
    )
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["condition"] == "reconstruction"
    assert row["governance_applied"] is False
    assert row["run_metadata"]["pilot_scope"] == "preflight"


def _result(query_id, dataset, task_type, *, skipped=False, failed=False):
    return {
        "query_id": query_id,
        "dataset": dataset,
        "task_type": task_type,
        "applicable_governance_dimensions": [] if skipped else ["authorization_scope"],
        "skipped": skipped,
        "reconstruction_status": "error" if failed else ("skipped" if skipped else "success"),
        "validation_status": "transport_error" if failed else ("not_applicable" if skipped else "valid"),
        "reconstructed_states": [] if skipped or failed else [{"interaction_id": "i1", "authorized_scopes": ["x"], "missing_fields": []}],
        "latency_ms": 10 if not skipped and not failed else None,
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12} if not skipped and not failed else None,
        "evaluator_only": {"model_visible_interaction_ids": ["i1"]},
    }


def test_summary_reports_denominators_and_explicit_incompleteness(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")
    result_path = tmp_path / "partial.jsonl"
    rows = [
        _result("memora-remembering-0000", "memora", "remembering"),
        _result("memora-remembering-0001", "memora", "remembering", failed=True),
        _result("memora-reasoning-0000", "memora", "reasoning", skipped=True),
    ]
    result_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    summary = summarize(manifest_path, [result_path], allow_incomplete=True)
    assert summary["complete"] is False
    assert summary["datasets"]["memora"]["by_task_type"]["remembering"]["operational"]["selected_n"] == 2
    assert summary["datasets"]["memora"]["by_task_type"]["remembering"]["operational"]["failed_n"] == 1
    assert summary["datasets"]["memora"]["by_task_type"]["reasoning"]["operational"]["skipped_n"] == 1
    with pytest.raises(ValueError, match="does not exactly match"):
        summarize(manifest_path, [result_path])


def test_summary_labels_structural_metrics_without_claiming_accuracy(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")
    result_path = tmp_path / "result.jsonl"
    result_path.write_text(json.dumps(_result("memora-remembering-0000", "memora", "remembering")) + "\n", encoding="utf-8")

    summary = summarize(manifest_path, [result_path], allow_incomplete=True)
    assert summary["evaluation_semantics"] == {
        "mode": "structural_grounded_reconstruction",
        "native_structured_gold_required_for_accuracy": True,
        "native_supported_agreement_available": False,
        "native_supported_agreement_note": (
            "Native structured agreement is unavailable unless the benchmark exposes "
            "a canonical structured governance target under the current mapping."
        ),
    }
    markdown_output = markdown(summary)
    assert "structural, grounded reconstruction diagnostics" in markdown_output
    assert "Native structured agreement:** unavailable" in markdown_output
    assert "100% accurate" not in markdown_output
    assert "Valid prediction rate" in markdown_output


def test_summary_rejects_disabled_paired_conditions(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")
    result_path = tmp_path / "conditions.jsonl"
    baseline = _result("memora-remembering-0000", "memora", "remembering")
    baseline["condition"] = "baseline"
    govmem = _result("memora-remembering-0001", "memora", "remembering")
    govmem["condition"] = "govmem"
    result_path.write_text(json.dumps(baseline) + "\n" + json.dumps(govmem) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="paired summaries are disabled"):
        summarize(manifest_path, [result_path], allow_incomplete=True)


def test_summary_requires_frozen_parent_for_preflight(tmp_path):
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(_valid_manifest()), encoding="utf-8")
    preflight = build_preflight_manifest(parent, require_frozen_hash=False)
    preflight["parent_manifest_sha256"] = FROZEN_PARENT_SHA256
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    result_path = tmp_path / "result.jsonl"
    result_path.write_text(json.dumps(_result("memora-remembering-0000", "memora", "remembering")) + "\n", encoding="utf-8")
    accepted = summarize(preflight_path, [result_path], allow_incomplete=True)
    assert accepted["manifest_version"] == "rq1-official-preflight-v1"

    preflight["parent_manifest_sha256"] = "not-the-frozen-parent"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    with pytest.raises(ValueError, match="parent_manifest_sha256"):
        summarize(preflight_path, [result_path], allow_incomplete=True)
