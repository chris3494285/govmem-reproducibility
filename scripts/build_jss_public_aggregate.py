#!/usr/bin/env python3
"""Build identifier-free public JSS aggregate files from final local analyses.

Only statistical sections are retained. Provenance objects, row identities,
block fingerprints, artifact hashes, and private paths are excluded by design.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_jss_rq1_observability import write_json  # noqa: E402

DEFAULT_RQ1_SUMMARY = ROOT / "results/jss_rq1_observability_v2/observability_summary.json"
DEFAULT_SCORING = ROOT / "results/jss_public_aggregate_v1/scoring_consistency_sensitivity.json"
DEFAULT_OUTPUT = ROOT / "results/jss_public_aggregate_v1/rq1_observability_v2.json"

PUBLIC_SECTIONS = (
    "status",
    "schema_version",
    "cohort",
    "state_observability",
    "context_projection_observability",
    "downstream_output_observability",
    "statistical_design",
    "estimable_metrics",
    "not_estimable",
    "api_calls_made",
    "no_model_generation_performed",
    "no_rescoring_performed",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def build_public_rq1_summary(source: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in PUBLIC_SECTIONS if key not in source]
    if missing:
        raise ValueError(f"RQ1 summary is missing public sections: {missing}")
    result = {key: source[key] for key in PUBLIC_SECTIONS}
    expected = {
        "bases": 731,
        "condition_rows": 4891,
        "matched_fault_pairs": 2080,
        "native_resolved_blocks": 298,
        "provider_replications": 12480,
    }
    for key, value in expected.items():
        if result["cohort"].get(key) != value:
            raise ValueError(
                f"RQ1 cohort mismatch for {key}: {result['cohort'].get(key)!r} != {value}"
            )
    longmemeval = result["downstream_output_observability"]["native_style"][
        "longmemeval_binary"
    ]
    if longmemeval["response_identity_unchanged"]["numerator"] != 46:
        raise ValueError("Unexpected LongMemEval response-identity count")
    if (
        longmemeval["response_identity_unchanged_score_changed"]["numerator"]
        != 1
    ):
        raise ValueError("Unexpected LongMemEval identical-response/score-change count")
    result["public_release_scope"] = {
        "identifier_free": True,
        "row_level_inputs_included": False,
        "private_inputs_required_to_recompute": True,
        "reporting_units_are_separate": {
            "LongMemEval": "binary native-style correctness",
            "Memora": "continuous native FAMA",
        },
        "bootstrap": {
            "unit": "base_id cluster",
            "iterations": 5000,
            "seed": 20261101,
        },
    }
    result["limitations"] = [
        "The full condition manifest and row-level provider results are not redistributed.",
        "The Formal-300 panel is a fixed selected panel, not a representative cohort sample.",
        "LongMemEval binary correctness and Memora FAMA are separate score units.",
        "The final 298 analysis uses one private human adjudication; the public 297 analysis does not.",
        "RQ1 V2 uncertainty is separate from the Formal-300 block and base-cluster bootstraps.",
    ]
    return result


def validate_scoring_summary(source: dict[str, Any]) -> None:
    expected = {
        "resolved_triplets": 298,
        "response_identity_unchanged": 64,
        "response_identity_unchanged_score_changed": 1,
    }
    for key, value in expected.items():
        if source.get("counts", {}).get("overall", {}).get(key) != value:
            raise ValueError(f"Scoring consistency count mismatch for {key}")
    if source.get("reproduction", {}).get("public_release_contains_identifiers_or_answer_hashes"):
        raise ValueError("Scoring summary is not marked identifier-free")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rq1-summary", type=Path, default=DEFAULT_RQ1_SUMMARY)
    parser.add_argument("--scoring-summary", type=Path, default=DEFAULT_SCORING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        rq1 = build_public_rq1_summary(_load_object(args.rq1_summary))
        validate_scoring_summary(_load_object(args.scoring_summary))
        if args.output.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
        if args.output.exists():
            args.output.unlink()
        write_json(args.output, rq1)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"JSS_PUBLIC_AGGREGATE_BLOCKED={exc}", file=sys.stderr)
        return 2
    print(f"JSS_PUBLIC_AGGREGATE_STATUS={rq1['status']}")
    print(f"JSS_PUBLIC_AGGREGATE_OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
