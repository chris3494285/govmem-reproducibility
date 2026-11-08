import hashlib
import json
from pathlib import Path

from scripts.build_final_main_governance_manifest import _select_oaks, build


def test_final_manifest_has_731_bases_and_preserves_clear_cohort():
    out = Path("/tmp/final_main_governance_test.jsonl")
    summary = build(output=out, summary_path=Path("/tmp/final_main_governance_test.json"), force=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    clear = [json.loads(line) for line in Path("data/main_governance_base_manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 731
    assert {row["base_id"] for row in rows} >= {row["base_id"] for row in clear}
    assert summary["base_count_by_source"] == {"longmemeval": 191, "memora": 380, "oaks_novel": 160}
    assert summary["oaks_count_by_primary_event_type"] == {"mixed": 110, "update": 50}


def test_oaks_selection_is_balanced_and_deterministic():
    rows = [json.loads(line) for line in Path("results/oaks_novel_main_dataset_audit/oaks_novel_candidates.jsonl").read_text().splitlines()]
    first = _select_oaks(rows)
    second = _select_oaks(rows)
    assert [row["source_example_id"] for row in first] == [row["source_example_id"] for row in second]
    counts = {}
    for row in first:
        counts[row["document_id"]] = counts.get(row["document_id"], 0) + 1
    assert set(counts) == {f"P{x:02d}" for x in (3, 6, 8, 11, 12, 13, 14, 18, 20, 21, 22, 24, 25, 27, 31, 32, 33, 34, 37)}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_support_reasons_and_no_legacy_reuse():
    rows = [json.loads(line) for line in Path("/tmp/final_main_governance_test.jsonl").read_text().splitlines()]
    legacy = {(r.get("source_dataset"), r.get("example_id")) for r in (json.loads(line) for line in Path("data/governance_fault_rq1_pilot_frozen.jsonl").read_text().splitlines())}
    assert all(row["supported_fault_operators"] for row in rows)
    assert all(set(row["supported_fault_operators"]) <= set(row["fault_support_reasons"]) for row in rows)
    assert not any((row["source_dataset"], row["source_example_id"]) in legacy for row in rows)


def test_manifest_hash_is_stable():
    path = Path("/tmp/final_main_governance_test.jsonl")
    first = hashlib.sha256(path.read_bytes()).hexdigest()
    build(output=path, summary_path=Path("/tmp/final_main_governance_test.json"), force=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first
