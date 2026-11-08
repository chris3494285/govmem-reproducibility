#!/usr/bin/env python3
"""Audit formal condition manifests for direct oracle-answer substitution.

This is intentionally a release gate.  It reports the historical fallback
without changing the frozen manifest, and exits non-zero when the fallback is
present so a release cannot silently omit the issue.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
REPAIRED_MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v3.jsonl"
DEFAULT_MANIFEST = REPAIRED_MANIFEST if REPAIRED_MANIFEST.exists() else HISTORICAL_MANIFEST
DEFAULT_OUTPUT = (ROOT / "results/answer_leakage_repair/answer_leakage_audit_v3.json") if REPAIRED_MANIFEST.exists() else (ROOT / "results/answer_leakage_audit.json")
FALLBACK_MARKER = "Benchmark-native answer-bearing evidence:"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _contains_fallback(value: Any) -> bool:
    if isinstance(value, str):
        return FALLBACK_MARKER in value
    if isinstance(value, dict):
        return any(_contains_fallback(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_fallback(child) for child in value)
    return False


def _provenance_problems(rows: list[dict[str, Any]], *, repaired: bool) -> list[str]:
    if not repaired:
        return []
    problems: list[str] = []
    for row in rows:
        spec = row.get("model_visible_context_spec") or {}
        provenance = row.get("source_provenance") or {}
        construction = " ".join(str(spec.get(key, "")) for key in ("content_source", "construction", "source"))
        if spec.get("oracle_answer_substitution") is True or provenance.get("oracle_answer_substitution") is True or "expected_answer" in construction:
            problems.append(str(row.get("condition_id")))
        if row.get("source_dataset") != "longmemeval":
            continue
        # v3 rows identify the repaired source projection explicitly. Legacy
        # unaffected rows retain their historical source-audit projection and
        # are not silently reclassified as unresolved here.
        if spec.get("content_source") != "official_longmemeval_native_evidence":
            continue
        provenance = row.get("source_provenance")
        if not isinstance(provenance, dict) or not provenance.get("native_message_locators") or provenance.get("oracle_answer_substitution") is not False:
            problems.append(str(row.get("condition_id")))
        if spec.get("oracle_answer_substitution") is not False:
            problems.append(str(row.get("condition_id")))
    return sorted(set(problems))


def audit(manifest: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    rows = _read_jsonl(manifest)
    affected = [row for row in rows if _contains_fallback(row)]
    repaired = manifest.name.endswith("_v3.jsonl")
    provenance_problems = _provenance_problems(rows, repaired=repaired)
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in affected:
        by_base[str(row.get("base_id"))].append(row)
    try:
        manifest_label = str(manifest.resolve().relative_to(ROOT))
    except ValueError:
        manifest_label = str(manifest)
    result = {
        "status": "PASS" if not affected and not provenance_problems else "FAIL",
        # A clean leakage check is necessary but not sufficient: the v3
        # provider scores and downstream paper analyses are still pending.
        "submission_ready": not repaired and not affected and not provenance_problems,
        "manifest": manifest_label,
        "total_condition_rows": len(rows),
        "affected_condition_rows": len(affected),
        "affected_base_count": len(by_base),
        "fallback_kind": "direct_oracle_answer_substitution",
        "fallback_marker": FALLBACK_MARKER,
        "affected_rows_by_condition_type": dict(sorted(Counter(str(row.get("condition_type")) for row in affected).items())),
        "affected_rows_by_source": dict(sorted(Counter(str(row.get("source_dataset")) for row in affected).items())),
        "affected_rows_by_fault_operator": dict(sorted(Counter(str(row.get("fault_operator")) for row in affected).items())),
        "affected_base_ids": sorted(by_base),
        "affected_rows": [
            {
                "condition_id": row.get("condition_id"),
                "base_id": row.get("base_id"),
                "source_dataset": row.get("source_dataset"),
                "condition_type": row.get("condition_type"),
                "fault_operator": row.get("fault_operator"),
            }
            for row in affected
        ],
        "unresolved_source_provenance_condition_ids": provenance_problems,
        "interpretation": (
            "The marker is a fallback derived directly from expected_answer, not "
            "benchmark-native evidence. Existing formal results must be rerun "
            "against native evidence before submission."
            if affected
            else "No formal condition uses direct oracle-answer substitution."
        ),
    }
    return result


def validate_no_unreported_oracle_substitution(result: dict[str, Any]) -> None:
    """Fail closed when a manifest contains the known direct-answer fallback."""
    if result.get("affected_condition_rows", 0) or result.get("unresolved_source_provenance_condition_ids"):
        raise RuntimeError(
            "Formal manifest contains unredistributable direct oracle-answer "
            f"substitution in {result['affected_condition_rows']} condition rows "
            f"across {result['affected_base_count']} bases"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-only", action="store_true", help="write the audit without applying the release gate")
    args = parser.parse_args(argv)
    result = audit(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not args.report_only:
        validate_no_unreported_oracle_substitution(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
