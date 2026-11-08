import hashlib
import json
from pathlib import Path

from scripts.audit_governance_evidence import audit_base, run


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/final_main_governance_base_manifest_v2.jsonl"
COND = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audits_frozen_731_unique_bases_and_source_fields(tmp_path):
    before = (_sha(BASE), _sha(COND))
    summary = run(BASE, COND, tmp_path)
    rows = list(csv_rows(tmp_path / "governance_evidence_base_level.csv"))
    assert len(rows) == 731
    assert len({r["base_id"] for r in rows}) == 731
    assert summary["overall"]["total_bases"] == 731
    assert {r["source"] for r in rows} == {"longmemeval", "memora", "oaks_novel"}
    assert all(r["evidence_fields_used"] for r in rows)
    assert _sha(BASE) == before[0]
    assert _sha(COND) == before[1]


def test_audit_is_deterministic_and_provider_free(tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    run(BASE, COND, first)
    run(BASE, COND, second)
    for name in [
        "governance_evidence_base_level.csv", "governance_evidence_summary.json",
        "governance_evidence_by_source.csv", "governance_evidence_by_event.csv",
        "governance_evidence_by_fault.csv", "governance_evidence_spotcheck.jsonl",
        "governance_evidence_schema_notes.md",
    ]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert json.loads((first / "governance_evidence_summary.json").read_text())["api_calls_made"] == 0


def test_missing_required_evidence_is_not_marked_supported():
    bases = [json.loads(x) for x in BASE.read_text().splitlines() if x.strip()]
    conditions = [json.loads(x) for x in COND.read_text().splitlines() if x.strip()]
    base = json.loads(json.dumps(next(x for x in bases if x["source_dataset"] == "longmemeval")))
    base["source_evidence"]["evidence_session_ids"] = []
    record = audit_base(base, conditions)
    assert record["source_native_evidence_present"] is False
    assert record["fully_supported"] is False


def csv_rows(path):
    import csv
    with path.open(newline="") as f:
        yield from csv.DictReader(f)
