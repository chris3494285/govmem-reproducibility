import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_final_main_governance_manifest_v2 import build

pytestmark = pytest.mark.requires_licensed_data


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def built_manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("final-main-governance-v2")
    out = root / "manifest.jsonl"
    summary = build(
        output=out,
        summary_path=root / "summary.json",
        freeze_path=root / "freeze.json",
        force=True,
    )
    return out, root / "freeze.json", summary


def test_v2_has_mutually_exclusive_analysis_buckets(built_manifest):
    out, _freeze, summary = built_manifest
    rows = _rows(out)
    assert len(rows) == 731
    assert summary["analysis_event_type_counts"] == {"deletion": 380, "temporal": 169, "update": 182}
    assert sum(summary["analysis_event_type_counts"].values()) == 731
    assert min(summary["analysis_event_type_counts"].values()) >= 150
    assert summary["mixed_assignment_counts"] == {"temporal": 56, "update": 54}
    assert {row["analysis_event_type"] for row in rows} == {"deletion", "temporal", "update"}


def test_primary_types_and_nonmixed_reasons_are_preserved(built_manifest):
    out, _freeze, _summary = built_manifest
    rows = _rows(out)
    parent = {row["base_id"]: row for row in _rows(Path("data/final_main_governance_base_manifest.jsonl"))}
    assert {row["base_id"] for row in rows} == set(parent)
    for row in rows:
        assert row["primary_event_type"] == parent[row["base_id"]]["primary_event_type"]
        if row["primary_event_type"] != "mixed":
            assert row["analysis_event_type"] == row["primary_event_type"]
            assert row["analysis_event_assignment_reason"] == "direct_inheritance_from_primary_event_type"
    mixed = [row for row in rows if row["primary_event_type"] == "mixed"]
    assert len(mixed) == 110
    assert all(row["analysis_event_assignment_reason"] in {"evidence_preferred_temporal", "evidence_preferred_update", "deterministic_balance_tiebreak"} for row in mixed)


def test_oaks_documents_and_fault_support_are_preserved(built_manifest):
    out, _freeze, _summary = built_manifest
    rows = _rows(out)
    oaks = [row for row in rows if row["source_dataset"] == "oaks_novel"]
    assert len(oaks) == 160
    assert len({row["document_id"] for row in oaks}) == 19
    assert all(row["supported_fault_operators"] for row in rows)
    assert all(set(row["supported_fault_operators"]) <= set(row["fault_support_reasons"]) for row in rows)


def test_freeze_metadata_and_manifest_hash_match(built_manifest):
    path, freeze_path, _summary = built_manifest
    freeze = json.loads(freeze_path.read_text())
    assert freeze["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert freeze["parent_manifest_sha256"] == hashlib.sha256(Path("data/final_main_governance_base_manifest.jsonl").read_bytes()).hexdigest()
    assert "No model outputs" in freeze["statement"]
