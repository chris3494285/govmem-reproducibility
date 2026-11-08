"""Audit final-main transport health from stored result rows only.

This command never initializes a provider and never computes downstream
accuracy or experimental effects.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.run_final_main_smoke_deepseek import _health_fields
    from eval.system_validation_scoring import memora_subtype
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from run_final_main_smoke_deepseek import _health_fields
    from eval.system_validation_scoring import memora_subtype


def audit(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    health = _health_fields(rows)
    failures = [row for row in rows if row.get("status") == "error"]

    def count(field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row.get(field) or "unknown") for row in failures).items()))

    by_subtype = Counter()
    for row in failures:
        subtype = memora_subtype(row.get("task_oracle_answer")) if row.get("source_dataset") == "memora" else None
        by_subtype[str(subtype or "unknown")] += 1
    return {
        "total_result_rows": len(rows),
        "total_final_failures": len(failures),
        "failure_rate": len(failures) / len(rows) if rows else 0,
        "failures_by_source": count("source_dataset"),
        "failures_by_analysis_event_type": count("analysis_event_type"),
        "failures_by_condition_type": count("condition_type"),
        "failures_by_fault_operator": count("fault_operator"),
        "failures_by_memora_subtype": dict(sorted(by_subtype.items())),
        "rows_recovered_by_retry": sum(bool(row.get("recovered_by_retry")) for row in rows),
        "scorer_coverage": sum(row.get("status") == "success" for row in rows),
        **health,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.results), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
