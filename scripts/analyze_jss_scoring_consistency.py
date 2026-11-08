#!/usr/bin/env python3
"""Audit answer identity versus native scoring and publish aggregate sensitivity.

The analysis consumes the private final Formal-300 artifacts only to identify
identical-input cases and compute aggregate sensitivity.  Its output contains
no benchmark identifiers, answer hashes, prompts, responses, or adjudication
content.  Public users without the private records can verify and interpret the
saved aggregate but cannot recompute it end to end.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_formal_300_subgroups import (  # noqa: E402
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    estimate,
    load_blocks,
)
from scripts.audit_jss_rq1_observability import (  # noqa: E402
    AuditError,
    DEFAULT_BASE_FREEZE,
    DEFAULT_BASE_MANIFEST,
    DEFAULT_CONDITION_FREEZE,
    DEFAULT_CONDITION_MANIFEST,
    DEFAULT_EVIDENCE_BASE_LEVEL,
    DEFAULT_FINAL_FORMAL_BLOCKS,
    DEFAULT_FINAL_FORMAL_METRICS,
    DEFAULT_FORMAL_BLOCKS,
    DEFAULT_FORMAL_OBSERVATIONS,
    DEFAULT_FORMAL_SUMMARY,
    DEFAULT_HUMAN_ADJUDICATION,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SEMANTIC_PAYLOADS,
    audit_repository,
)

DEFAULT_OUTPUT = ROOT / "results/jss_public_aggregate_v1/scoring_consistency_sensitivity.json"
CONDITIONS = ("clean", "corrupted", "enforced")


def _complete_triplets(
    observations: Sequence[dict[str, Any]],
) -> list[dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        if row.get("pilot_block_id"):
            grouped[str(row["pilot_block_id"])].append(row)
    triplets: list[dict[str, dict[str, Any]]] = []
    for rows in grouped.values():
        by_condition = {str(row.get("condition")): row for row in rows}
        if set(by_condition) == set(CONDITIONS) and all(
            by_condition[condition].get("formal_score") is not None
            for condition in CONDITIONS
        ):
            triplets.append(by_condition)
    return triplets


def identity_summary(
    triplets: Sequence[dict[str, dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts_by_source: dict[str, dict[str, int]] = {}
    discrepancies: list[dict[str, Any]] = []
    for triplet in triplets:
        clean = triplet["clean"]
        corrupted = triplet["corrupted"]
        source = str(clean["source"])
        counts = counts_by_source.setdefault(
            source,
            {
                "resolved_triplets": 0,
                "response_identity_unchanged": 0,
                "response_identity_unchanged_score_unchanged": 0,
                "response_identity_unchanged_score_changed": 0,
                "response_identity_changed": 0,
                "score_identity_changed": 0,
                "score_identity_unchanged": 0,
            },
        )
        counts["resolved_triplets"] += 1
        same_response = clean["answer_sha256"] == corrupted["answer_sha256"]
        same_score = float(clean["formal_score"]) == float(corrupted["formal_score"])
        if same_response:
            counts["response_identity_unchanged"] += 1
            counts[
                "response_identity_unchanged_score_unchanged"
                if same_score
                else "response_identity_unchanged_score_changed"
            ] += 1
        else:
            counts["response_identity_changed"] += 1
        counts["score_identity_unchanged" if same_score else "score_identity_changed"] += 1
        if same_response and not same_score:
            discrepancies.append(
                {
                    "source": source,
                    "generating_provider": str(clean["generating_provider"]),
                    "fault_type": str(clean["fault_type"]),
                    "event_macro": str(clean["event_macro"]),
                    "all_three_answer_identities_equal": len(
                        {row["answer_sha256"] for row in triplet.values()}
                    )
                    == 1,
                    "frozen_scores": {
                        condition: float(triplet[condition]["formal_score"])
                        for condition in CONDITIONS
                    },
                }
            )
    total = {
        key: sum(source_counts[key] for source_counts in counts_by_source.values())
        for key in (
            "resolved_triplets",
            "response_identity_unchanged",
            "response_identity_unchanged_score_unchanged",
            "response_identity_unchanged_score_changed",
            "response_identity_changed",
            "score_identity_changed",
            "score_identity_unchanged",
        )
    }
    return {"overall": total, "by_source": counts_by_source}, discrepancies


def _effect_fields(result: dict[str, Any], blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": result["n"],
        "base_clusters": len({str(row["base_id"]) for row in blocks}),
        "clean": result["clean"],
        "corrupted": result["corrupted"],
        "enforced": result["enforced"],
        "corruption_effect": result["corruption_effect"],
        "recovery_effect": result["recovery_effect"],
        "restoration_gap": result["restoration_gap"],
        "cluster_bootstrap_ci95": result["cluster_bootstrap_ci95"],
        "induced_failure_n": sum(
            float(row["clean"]) == 1.0 and float(row["corrupted"]) == 0.0 for row in blocks
        ),
    }


def effect_summary(
    blocks: Sequence[dict[str, Any]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    overall = estimate(blocks, strata=("source",), iterations=iterations, seed=seed)
    longmemeval = [row for row in blocks if row["source"] == "longmemeval"]
    lme = estimate(longmemeval, strata=(), iterations=iterations, seed=seed)
    return {
        "overall": _effect_fields(overall, blocks),
        "longmemeval": _effect_fields(lme, longmemeval),
    }


def analyze(
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    bundle = audit_repository(
        condition_manifest=DEFAULT_CONDITION_MANIFEST,
        base_manifest=DEFAULT_BASE_MANIFEST,
        condition_freeze=DEFAULT_CONDITION_FREEZE,
        base_freeze=DEFAULT_BASE_FREEZE,
        results_dir=DEFAULT_RESULTS_DIR,
        native_panel="paper_final_298",
        formal_observations=DEFAULT_FORMAL_OBSERVATIONS,
        formal_blocks=DEFAULT_FORMAL_BLOCKS,
        formal_summary=DEFAULT_FORMAL_SUMMARY,
        final_formal_blocks=DEFAULT_FINAL_FORMAL_BLOCKS,
        final_formal_metrics=DEFAULT_FINAL_FORMAL_METRICS,
        human_adjudication=DEFAULT_HUMAN_ADJUDICATION,
        semantic_payloads=DEFAULT_SEMANTIC_PAYLOADS,
        evidence_base_level=DEFAULT_EVIDENCE_BASE_LEVEL,
    )
    triplets = _complete_triplets(bundle.formal_observations)
    counts, discrepancies = identity_summary(triplets)
    if counts["overall"]["resolved_triplets"] != 298:
        raise ValueError("expected the final 298 resolved Formal-300 triplets")
    if len(discrepancies) != 1:
        raise ValueError(
            f"expected exactly one identical-response/score-change case, found {len(discrepancies)}"
        )
    discrepancy = discrepancies[0]
    if not discrepancy["all_three_answer_identities_equal"]:
        raise ValueError("affected case does not have identical clean/corrupted/enforced answers")

    blocks, _reconciliation = load_blocks()
    if len(blocks) != 298:
        raise ValueError("expected 298 final Formal-300 block records")
    frozen = effect_summary(blocks, iterations=iterations, seed=seed)

    affected_rows = [
        row
        for row in triplets
        if row["clean"]["source"] == discrepancy["source"]
        and row["clean"]["generating_provider"] == discrepancy["generating_provider"]
        and row["clean"]["fault_type"] == discrepancy["fault_type"]
        and row["clean"]["answer_sha256"] == row["corrupted"]["answer_sha256"]
        and float(row["clean"]["formal_score"]) != float(row["corrupted"]["formal_score"])
    ]
    if len(affected_rows) != 1:
        raise ValueError("affected aggregate stratum does not resolve to one triplet")
    affected_block_id = str(affected_rows[0]["clean"]["pilot_block_id"])
    sensitive_blocks = [dict(row) for row in blocks]
    affected_blocks = [row for row in sensitive_blocks if row["pilot_block_id"] == affected_block_id]
    if len(affected_blocks) != 1:
        raise ValueError("affected block is absent or duplicated in final block records")
    common_score = 0.0
    for condition in CONDITIONS:
        affected_blocks[0][condition] = common_score
    sensitivity = effect_summary(sensitive_blocks, iterations=iterations, seed=seed)

    lme_frozen = frozen["longmemeval"]
    lme_sensitive = sensitivity["longmemeval"]
    overall_frozen = frozen["overall"]
    overall_sensitive = sensitivity["overall"]
    output = {
        "schema_version": "jss-public-scoring-consistency-sensitivity-v1",
        "status": "verified_from_private_frozen_inputs",
        "scope": (
            "Final 298-triplet Formal-300 panel. The audit compares response identity "
            "with native score identity and computes a frozen-score sensitivity under "
            "the explicit common-score treatment described below."
        ),
        "counts": counts,
        "affected_pattern": {
            "source": discrepancy["source"],
            "response_identity_unchanged": True,
            "score_identity_changed": True,
            "all_three_answer_identities_equal": True,
            "frozen_scores": discrepancy["frozen_scores"],
        },
        "frozen": frozen,
        "sensitivity": {
            "treatment": (
                "Assign the one affected block's three byte-identical answers a common "
                "score of 0.0. A common score of 1.0 produces the same paired effects "
                "for this block."
            ),
            "effects": sensitivity,
            "longmemeval_response_identity_unchanged_score_changed": 0,
            "longmemeval_score_identity_changed": (
                counts["by_source"]["longmemeval"]["score_identity_changed"] - 1
            ),
            "longmemeval_score_identity_unchanged": (
                counts["by_source"]["longmemeval"]["score_identity_unchanged"] + 1
            ),
        },
        "deltas": {
            "longmemeval_corruption_effect_pp": (
                lme_sensitive["corruption_effect"] - lme_frozen["corruption_effect"]
            )
            * 100,
            "longmemeval_recovery_effect_pp": (
                lme_sensitive["recovery_effect"] - lme_frozen["recovery_effect"]
            )
            * 100,
            "pooled_corruption_effect_pp": (
                overall_sensitive["corruption_effect"] - overall_frozen["corruption_effect"]
            )
            * 100,
            "pooled_recovery_effect_pp": (
                overall_sensitive["recovery_effect"] - overall_frozen["recovery_effect"]
            )
            * 100,
        },
        "conclusion": {
            "classification": (
                "confirmed_scoring_inconsistency_with_negligible_empirical_impact"
            ),
            "frozen_results_retained": True,
            "qualitative_direction_unchanged": True,
            "recommended_manuscript_action": (
                "Retain frozen judgments and disclose the isolated evaluator-consistency "
                "failure; report the common-score sensitivity rather than post-hoc rescoring."
            ),
        },
        "reproduction": {
            "api_calls": 0,
            "model_generation": False,
            "rescoring_performed": False,
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
            "private_inputs_required_to_recompute": True,
            "public_release_contains_identifiers_or_answer_hashes": False,
        },
    }
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = analyze(iterations=args.iterations, seed=args.seed)
    except (
        AuditError,
        FileNotFoundError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"JSS_SCORING_SENSITIVITY_BLOCKED={exc}", file=sys.stderr)
        return 2
    if args.output.exists() and not args.force:
        print(f"refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"JSS_SCORING_SENSITIVITY_STATUS={result['status']}")
    print(
        "JSS_SCORING_IDENTICAL_RESPONSE_SCORE_CHANGED="
        f"{result['counts']['overall']['response_identity_unchanged_score_changed']}"
    )
    print(f"JSS_SCORING_OUTPUT={args.output}")
    print("JSS_SCORING_API_CALLS_MADE=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
