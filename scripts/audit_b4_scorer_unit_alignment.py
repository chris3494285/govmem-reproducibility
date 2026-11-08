#!/usr/bin/env python3
"""Audit B4 scorer disagreements and validation-unit alignment offline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.system_validation_scoring import normalize_text
from scripts.analyze_b4_scorer_validation import (
    EXPECTED_CASES,
    EXPECTED_SCORER_VERSION,
    canonical_value_sha256,
    load_source_rows,
)
from scripts.b4_calibration_execution import LedgerConflictError
from scripts.prepare_b4_private_payloads import read_jsonl


AUDIT_VERSION = "b4-scorer-unit-alignment-audit-v1"
EXPECTED_LME_N = 96
EXPECTED_LME_DISAGREEMENTS = 36
EXPECTED_MEMORA_ATOMIC_N = 508


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _unique_index(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise LedgerConflictError(f"duplicate {label}: {value}")
        result[value] = row
    return result


def _source_key(provider: str, sample_id: str) -> tuple[str, str]:
    prefix = provider + "::"
    if not sample_id.startswith(prefix):
        raise LedgerConflictError(f"provider-aware sample identity drift: {sample_id}")
    return provider, sample_id[len(prefix):]


def _abstention_raw_label(reference: dict[str, Any]) -> bool:
    labels = [
        reference.get("final_reference_raw_label"),
        reference.get("primary_gpt_raw_decision"),
        reference.get("primary_claude_raw_decision"),
        reference.get("gemini_raw_decision"),
    ]
    return any(isinstance(value, str) and value.startswith("abstention_") for value in labels)


def classify_lme_disagreement(
    case: dict[str, Any],
    reference: dict[str, Any],
    execution: dict[str, Any],
    frozen_scorer_row: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Classify without changing either frozen label.

    Semantic equivalence comes only from the frozen independent reference. Text
    operations below are exact normalized diagnostics, not a new scorer.
    """
    scorer = case["deterministic_scorer_binary_label"]
    independent = case["independent_reference_binary_label"]
    if scorer == independent:
        raise LedgerConflictError("agreement passed to disagreement classifier")
    answer_norm = str((frozen_scorer_row or {}).get("normalized_model_answer") or normalize_text(execution.get("model_answer")))
    reference_norm = str((frozen_scorer_row or {}).get("normalized_oracle_answer") or normalize_text(execution.get("reference_answer")))
    oracle_substring = bool(reference_norm and reference_norm in answer_norm)
    diagnostics = {
        "answer_sha256": canonical_value_sha256(execution.get("model_answer")),
        "question_sha256": canonical_value_sha256(execution.get("question")),
        "reference_answer_sha256": canonical_value_sha256(execution.get("reference_answer")),
        "normalized_reference_is_answer_substring": oracle_substring,
        "normalized_exact_match": answer_norm == reference_norm,
        "frozen_reference_has_abstention_subtype": _abstention_raw_label(reference),
        "native_reference_declares_explicit_alternative": "also acceptable" in str(execution.get("reference_answer", "")).casefold(),
    }
    if diagnostics["frozen_reference_has_abstention_subtype"]:
        if independent == "correct":
            explanation = (
                "The independent reference accepts a justified abstention, while the frozen exact/native-"
                "alternative scorer rejects the model's non-identical abstention wording."
            )
        else:
            explanation = (
                "The independent reference rejects an abstaining/conflict response, while the frozen scorer "
                "accepts it because the normalized native answer occurs inside the longer response."
            )
        return "abstention_handling_difference", explanation, diagnostics
    if scorer == "correct" and independent == "incorrect":
        return (
            "deterministic_false_positive",
            "The frozen scorer accepts the native-answer substring inside a longer response that the independent semantic reference marks incorrect.",
            diagnostics,
        )
    if diagnostics["native_reference_declares_explicit_alternative"]:
        return (
            "normalization_or_native_alternative_gap",
            "The native reference encodes explicit acceptable surface alternatives, but the frozen deterministic matcher rejects the semantically accepted response form.",
            diagnostics,
        )
    if scorer == "incorrect" and independent == "correct":
        return (
            "semantic_paraphrase_missed_by_deterministic",
            "The frozen semantic reference accepts the answer, while the exact/native-alternative scorer rejects its non-identical wording.",
            diagnostics,
        )
    return "other_manual_review", "The frozen labels disagree without matching a narrower auditable category.", diagnostics


def audit_longmemeval(
    validation_cases: list[dict[str, Any]],
    references: list[dict[str, Any]],
    executions: list[dict[str, Any]],
    source_rows: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_by_id = _unique_index(references, "blinded_request_id", "reference case")
    execution_by_id = _unique_index(executions, "blinded_request_id", "LongMemEval execution case")
    lme = [row for row in validation_cases if row.get("task") == "longmemeval"]
    disagreements = [row for row in lme if not row.get("agreement_binary")]
    fp_n = sum(row.get("disagreement_type") == "false_positive" for row in disagreements)
    fn_n = sum(row.get("disagreement_type") == "false_negative" for row in disagreements)
    if (len(lme), len(disagreements), fp_n, fn_n) != (EXPECTED_LME_N, EXPECTED_LME_DISAGREEMENTS, 12, 24):
        raise LedgerConflictError("frozen LongMemEval disagreement accounting drift")
    audit_rows = []
    for case in disagreements:
        case_id = str(case["blinded_request_id"])
        reference = reference_by_id.get(case_id)
        execution = execution_by_id.get(case_id)
        if reference is None or execution is None:
            raise LedgerConflictError(f"missing frozen LongMemEval evidence: {case_id}")
        provider = str(case["generating_provider"])
        condition_id = str(case["immutable_provenance"]["representative_condition_id"])
        source = source_rows.get((provider, condition_id))
        if source is None:
            raise LedgerConflictError(f"missing frozen scorer row: {case_id}")
        if canonical_value_sha256(source.get("parsed_answer")) != case["immutable_provenance"]["answer_identity_sha256"]:
            raise LedgerConflictError(f"LongMemEval answer identity drift: {case_id}")
        if source.get("scoring_version") != EXPECTED_SCORER_VERSION:
            raise LedgerConflictError(f"LongMemEval scorer version drift: {case_id}")
        category, explanation, diagnostics = classify_lme_disagreement(case, reference, execution, source)
        audit_rows.append({
            "blinded_request_id": case_id,
            "scorer_label": case["deterministic_scorer_binary_label"],
            "independent_label": case["independent_reference_binary_label"],
            "independent_raw_label": case.get("independent_reference_raw_label"),
            "disagreement_type": case["disagreement_type"],
            "root_cause_category": category,
            "explanation": explanation,
            "scorer_method": source.get("scoring_method"),
            "scorer_details": source.get("scoring_details"),
            "scorer_version": source.get("scoring_version"),
            "independent_resolution_method": case.get("independent_resolution_method"),
            "provider": provider,
            "condition": case.get("experimental_condition"),
            "evidence_diagnostics": diagnostics,
            "immutable_source_locators": {
                "scorer_condition_id": condition_id,
                "scorer_provider": provider,
                "private_semantic_payload": "results/b4_scoring_validation/private/longmemeval_judge_execution.jsonl",
                "independent_reference": "results/b4_scoring_validation/private/frontier_adjudication/adjudication_cases.jsonl",
            },
        })
    audit_rows.sort(key=lambda row: row["blinded_request_id"])
    by_category: dict[str, Any] = {}
    for category in sorted({row["root_cause_category"] for row in audit_rows}):
        rows = [row for row in audit_rows if row["root_cause_category"] == category]
        by_category[category] = {
            "n": len(rows),
            "false_positive_n": sum(row["disagreement_type"] == "false_positive" for row in rows),
            "false_negative_n": sum(row["disagreement_type"] == "false_negative" for row in rows),
        }
    summary = {
        "schema_version": AUDIT_VERSION,
        "longmemeval_total_n": len(lme),
        "disagreement_n": len(audit_rows),
        "false_positive_n": fp_n,
        "false_negative_n": fn_n,
        "by_root_cause_category": by_category,
        "provider_distribution": dict(sorted(Counter(row["provider"] for row in audit_rows).items())),
        "condition_distribution": dict(sorted(Counter(row["condition"] for row in audit_rows).items())),
        "classification_policy": (
            "Frozen labels are never changed. Abstention categories use frozen raw judge subtypes; "
            "normalization diagnostics use exact deterministic normalization only; semantic acceptance "
            "comes exclusively from the frozen independent reference."
        ),
        "private_text_policy": "Question, reference-answer, and model-answer text were inspected locally but are represented in this public-safe audit only by hashes and locators.",
    }
    return audit_rows, summary


def build_memora_answer_mapping(
    validation_cases: list[dict[str, Any]],
    references: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    criteria: list[dict[str, Any]],
    source_rows: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    memora_cases = [row for row in validation_cases if row.get("task") == "memora"]
    memora_references = [row for row in references if row.get("task") == "memora"]
    if len(memora_cases) != EXPECTED_MEMORA_ATOMIC_N or len(memora_references) != EXPECTED_MEMORA_ATOMIC_N:
        raise LedgerConflictError("frozen Memora atomic population drift")
    case_by_id = _unique_index(memora_cases, "blinded_request_id", "Memora validation case")
    reference_by_id = _unique_index(memora_references, "blinded_request_id", "Memora reference")
    request_by_id = _unique_index(requests, "request_id", "Memora request")
    provenance_by_id = _unique_index(provenance, "blinded_request_id", "Memora provenance")
    criterion_by_id = _unique_index(criteria, "criterion_id", "Memora criterion")
    expected_ids = set(case_by_id)
    for mapping, label in (
        (reference_by_id, "reference"),
        (request_by_id, "request"),
        (provenance_by_id, "provenance"),
    ):
        if set(mapping) != expected_ids:
            raise LedgerConflictError(f"missing/extra Memora {label} mapping")

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    seen_assignments: set[tuple[str, str]] = set()
    for request_id in sorted(expected_ids):
        case = case_by_id[request_id]
        reference = reference_by_id[request_id]
        request = request_by_id[request_id]
        prov = provenance_by_id[request_id]
        criterion_ids = [str(value) for value in prov.get("criterion_ids", [])]
        if criterion_ids != [str(value) for value in request.get("deduplicated_criterion_ids", [])] or not criterion_ids:
            raise LedgerConflictError(f"Memora criterion identity drift: {request_id}")
        if reference.get("final_reference_binary_label") not in {"pass", "fail"}:
            raise LedgerConflictError(f"unresolved Memora reference: {request_id}")
        for criterion_id in criterion_ids:
            criterion = criterion_by_id.get(criterion_id)
            if criterion is None:
                raise LedgerConflictError(f"unknown Memora criterion: {criterion_id}")
            pair = (str(criterion["provider"]), str(criterion["original_condition_id"]))
            assignment = (request_id, criterion_id)
            if assignment in seen_assignments:
                raise LedgerConflictError(f"duplicate answer-to-criterion assignment: {assignment}")
            seen_assignments.add(assignment)
            source = source_rows.get(pair)
            if source is None:
                raise LedgerConflictError(f"missing frozen Memora scorer row: {pair}")
            if canonical_value_sha256(source.get("parsed_answer")) != request.get("answer_sha256"):
                raise LedgerConflictError(f"Memora answer identity drift: {request_id}")
            if source.get("scoring_version") != EXPECTED_SCORER_VERSION:
                raise LedgerConflictError(f"Memora scorer version drift: {request_id}")
            entry = grouped.setdefault(pair, {
                "answer_validation_id": f"{pair[0]}::{pair[1]}",
                "provider": pair[0],
                "condition": source.get("condition_type"),
                "condition_id": pair[1],
                "subtype": case.get("subtype"),
                "answer_identity_sha256": request.get("answer_sha256"),
                "deterministic_answer_label": "correct" if source.get("correctness") is True else "incorrect",
                "deterministic_scoring_method": source.get("scoring_method"),
                "deterministic_scoring_version": source.get("scoring_version"),
                "deterministic_field_total_count": (source.get("scoring_details") or {}).get("field_total_count"),
                "deterministic_field_correct_count": (source.get("scoring_details") or {}).get("field_correct_count"),
                "atomic_case_ids": [],
                "criterion_ids": [],
                "criterion_type_counts": Counter(),
                "independent_label_counts": Counter(),
            })
            immutable = (
                entry["subtype"], entry["answer_identity_sha256"], entry["deterministic_answer_label"],
                entry["deterministic_scoring_method"], entry["deterministic_scoring_version"],
            )
            current = (
                case.get("subtype"), request.get("answer_sha256"),
                "correct" if source.get("correctness") is True else "incorrect",
                source.get("scoring_method"), source.get("scoring_version"),
            )
            if immutable != current:
                raise LedgerConflictError(f"Memora grouped-answer provenance drift: {pair}")
            entry["atomic_case_ids"].append(request_id)
            entry["criterion_ids"].append(criterion_id)
            entry["criterion_type_counts"][str(criterion.get("criterion_type"))] += 1
            entry["independent_label_counts"][str(reference["final_reference_binary_label"])] += 1

    rows = []
    for pair in sorted(grouped):
        entry = grouped[pair]
        entry["atomic_case_ids"] = sorted(set(entry["atomic_case_ids"]))
        entry["criterion_ids"] = sorted(set(entry["criterion_ids"]))
        entry["criterion_type_counts"] = dict(sorted(entry["criterion_type_counts"].items()))
        entry["independent_label_counts"] = dict(sorted(entry["independent_label_counts"].items()))
        entry["atomic_case_n"] = len(entry["atomic_case_ids"])
        entry["answer_criterion_assignment_n"] = sum(entry["criterion_type_counts"].values())
        entry["all_native_criteria_pass"] = entry["independent_label_counts"].get("fail", 0) == 0
        entry["unit_alignment_status"] = "not_comparable_without_unsupported_aggregation"
        entry["immutable_source_locator"] = {
            "provider": entry["provider"],
            "condition_id": entry["condition_id"],
            "criteria_artifact": "results/b4_scoring_validation/memora_atomic_criteria.jsonl",
        }
        rows.append(entry)
    return rows


def assess_memora_alignment(mapping: list[dict[str, Any]], atomic_n: int) -> dict[str, Any]:
    if atomic_n != EXPECTED_MEMORA_ATOMIC_N:
        raise LedgerConflictError("Memora atomic denominator drift")
    unique_answers = len(mapping)
    assignment_n = sum(row["answer_criterion_assignment_n"] for row in mapping)
    field_matches = sum(
        row["deterministic_field_total_count"] == row["criterion_type_counts"].get("memory_presence", 0)
        for row in mapping
    )
    criterion_types = Counter()
    for row in mapping:
        criterion_types.update(row["criterion_type_counts"])
    return {
        "schema_version": AUDIT_VERSION,
        "atomic_semantic_case_n": atomic_n,
        "answer_criterion_assignment_n": assignment_n,
        "unique_memora_answer_n": unique_answers,
        "criteria_per_answer_distribution": dict(sorted(Counter(row["answer_criterion_assignment_n"] for row in mapping).items())),
        "atomic_cases_per_answer_distribution": dict(sorted(Counter(row["atomic_case_n"] for row in mapping).items())),
        "criterion_type_counts": dict(sorted(criterion_types.items())),
        "scorer_semantic_unit": "one generated answer; Boolean correct only when every extracted content_data leaf fact is deterministically supported",
        "independent_semantic_unit": "one native evaluation sub-question/atomic criterion; native benchmark aggregates these as accuracy components and per-question FAMA",
        "current_508_comparison_valid": False,
        "path_1_answer_level_aggregation": {
            "valid": False,
            "reason": "The native benchmark defines per-question FAMA from memory-presence and forgetting-absence accuracies, not a frozen Boolean all-criteria-pass label. GovMem's Boolean scorer covers extracted content_data facts and does not encode the native forgetting penalty, so an all-pass or any-pass projection would invent a new target.",
        },
        "path_2_criterion_level_recovery": {
            "valid": False,
            "reason": "Frozen scorer details expose content_data field paths and support bits, but B4 criteria expose native evaluation-question IDs/types and hashes without an immutable field-to-criterion mapping. The criteria also include forgetting_absence items absent from scorer field details.",
            "answers_where_scorer_field_count_equals_memory_presence_criterion_count": field_matches,
            "answers_where_counts_do_not_match": unique_answers - field_matches,
        },
        "selected_valid_alignment": "none",
        "valid_memora_n": 0,
        "valid_memora_agreement": None,
        "recommended_valid_comparison_unit": "none available from the current frozen artifacts",
        "exact_aggregation_or_recovery_rule": None,
        "evidence_sources": [
            "eval/system_validation_scoring.py: structured Memora correctness requires all extracted content_data leaf facts",
            "Memora evals/README.md: native per-question FAMA combines memory-presence and forgetting-absence accuracies",
            "frozen scorer scoring_details.fields and memora_atomic_criteria.jsonl",
        ],
        "new_judge_or_api_calls_required_for_this_audit": False,
        "scorer_rerun_required_for_this_audit": False,
        "blocker_for_valid_metric": "A predeclared common endpoint and immutable mapping between scorer fields and native criteria are absent; creating either would be new validation design, not analysis of the frozen outputs.",
    }


def write_memora_report(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Memora Unit-Alignment Audit",
        "",
        f"- Atomic semantic cases: {audit['atomic_semantic_case_n']}",
        f"- Unique generated answers: {audit['unique_memora_answer_n']}",
        f"- Answer–criterion assignments: {audit['answer_criterion_assignment_n']}",
        "- Current repeated answer-to-criterion comparison valid: no",
        "- Valid alignment from frozen artifacts: none",
        "- New judge/API calls used: no",
        "- Scorer rerun used: no",
        "",
        "## Frozen units",
        "",
        f"GovMem scorer: {audit['scorer_semantic_unit']}.",
        "",
        f"Independent reference: {audit['independent_semantic_unit']}.",
        "",
        "## Path 1 — aggregate criteria to answer level",
        "",
        audit["path_1_answer_level_aggregation"]["reason"],
        "",
        "## Path 2 — recover deterministic criterion predictions",
        "",
        audit["path_2_criterion_level_recovery"]["reason"],
        "",
        f"Only {audit['path_2_criterion_level_recovery']['answers_where_scorer_field_count_equals_memory_presence_criterion_count']} of {audit['unique_memora_answer_n']} answers even have matching scorer-field and native memory-presence criterion counts; count equality still would not prove identity.",
        "",
        "## Conclusion",
        "",
        "Path 3 applies: neither proposed alignment is defensible from the frozen artifacts. No corrected Memora accuracy or revised cross-dataset overall metric is computed. LongMemEval remains a valid standalone answer-level comparison.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-cases", type=Path, default=Path("results/b4_scoring_validation/final_scorer_validation/scorer_validation_cases.jsonl"))
    parser.add_argument("--reference", type=Path, default=Path("results/b4_scoring_validation/private/frontier_adjudication/adjudication_cases.jsonl"))
    parser.add_argument("--b4-summary", type=Path, default=Path("results/b4_scoring_validation/b4_validation_summary.json"))
    parser.add_argument("--lme-execution", type=Path, default=Path("results/b4_scoring_validation/private/longmemeval_judge_execution.jsonl"))
    parser.add_argument("--memora-requests", type=Path, default=Path("results/b4_scoring_validation/memora_judge_requests.jsonl"))
    parser.add_argument("--memora-provenance", type=Path, default=Path("results/b4_scoring_validation/private/memora_judge_provenance.jsonl"))
    parser.add_argument("--memora-criteria", type=Path, default=Path("results/b4_scoring_validation/memora_atomic_criteria.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/b4_scoring_validation/unit_alignment_audit"))
    args = parser.parse_args(argv)

    validation_cases = read_jsonl(args.validation_cases)
    references = read_jsonl(args.reference)
    if len(validation_cases) != EXPECTED_CASES or len(references) != EXPECTED_CASES:
        raise SystemExit("frozen 604-case population drift")
    b4_summary = json.loads(args.b4_summary.read_text(encoding="utf-8"))
    source_rows, _ = load_source_rows([Path(value) for value in b4_summary.get("answer_artifacts_used", [])])
    lme_rows, lme_summary = audit_longmemeval(
        validation_cases, references, read_jsonl(args.lme_execution), source_rows
    )
    memora_mapping = build_memora_answer_mapping(
        validation_cases,
        references,
        read_jsonl(args.memora_requests),
        read_jsonl(args.memora_provenance),
        read_jsonl(args.memora_criteria),
        source_rows,
    )
    memora_audit = assess_memora_alignment(memora_mapping, EXPECTED_MEMORA_ATOMIC_N)
    combined = {
        "schema_version": AUDIT_VERSION,
        "longmemeval": {
            "valid_same_unit_n": EXPECTED_LME_N,
            "agreement_n": EXPECTED_LME_N - EXPECTED_LME_DISAGREEMENTS,
            "disagreement_n": EXPECTED_LME_DISAGREEMENTS,
        },
        "memora": memora_audit,
        "revised_overall": {
            "computed": False,
            "n": None,
            "agreement_n": None,
            "reason": "Memora has no defensible same-unit binary comparison in the frozen artifacts; mixing LongMemEval answers with Memora atomic criteria is prohibited.",
        },
        "api_network_model_judge_calls": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "longmemeval_disagreement_audit.jsonl", lme_rows)
    _write_json(args.output_dir / "longmemeval_disagreement_summary.json", lme_summary)
    _write_json(args.output_dir / "memora_unit_alignment_audit.json", memora_audit)
    _write_jsonl(args.output_dir / "memora_case_mapping.jsonl", memora_mapping)
    write_memora_report(args.output_dir / "MEMORA_UNIT_ALIGNMENT_REPORT.md", memora_audit)
    _write_json(args.output_dir / "unit_alignment_audit_summary.json", combined)

    print(f"LONGMEMEVAL_TOTAL={EXPECTED_LME_N}")
    print(f"LONGMEMEVAL_DISAGREEMENTS={EXPECTED_LME_DISAGREEMENTS}")
    print("LONGMEMEVAL_FALSE_NEGATIVES=24")
    print("LONGMEMEVAL_FALSE_POSITIVES=12")
    print(f"MEMORA_ATOMIC_CASES={EXPECTED_MEMORA_ATOMIC_N}")
    print(f"MEMORA_UNIQUE_ANSWERS={memora_audit['unique_memora_answer_n']}")
    print("CURRENT_MEMORA_COMPARISON_VALID=no")
    print(f"VALID_MEMORA_ALIGNMENT={memora_audit['selected_valid_alignment']}")
    print("VALID_MEMORA_N=0")
    print("VALID_MEMORA_AGREEMENT=not_available")
    print("REVISED_OVERALL_N=not_available")
    print("REVISED_OVERALL_AGREEMENT=not_available")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
