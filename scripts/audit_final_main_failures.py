"""Audit final provider/parse failures using authoritative frozen metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval.final_main_result_join import join_result_file
from eval.system_validation_scoring import memora_subtype


def audit(results: Path, manifest: Path) -> dict:
    joined, digest = join_result_file(results, manifest)
    failures = [row for row in joined if row.get("status") != "success" or row.get("parse_success") is not True]
    def counts(field: str) -> dict[str, int]:
        return dict(sorted(Counter("<missing>" if row.get(field) is None else str(row.get(field)) for row in failures).items()))
    subtype_counts = Counter()
    for row in failures:
        if row.get("source_dataset") == "memora":
            subtype = (row.get("task_scoring_spec") or {}).get("subtype") or memora_subtype(row.get("task_oracle_answer"))
        else:
            subtype = (row.get("task_source_metadata") or {}).get("question_type") or row.get("task_answer_type")
        subtype_counts["<missing>" if subtype is None else str(subtype)] += 1
    return {"total_result_rows": len(joined), "total_final_failures": len(failures), "failure_rate": len(failures) / len(joined) if joined else None, "failures_by_source": counts("source_dataset"), "failures_by_analysis_event_type": counts("analysis_event_type"), "failures_by_condition_type": counts("condition_type"), "failures_by_fault_operator": counts("fault_operator"), "failures_by_task_subtype": dict(sorted(subtype_counts.items())), "manifest_sha256": digest}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.results, args.manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
