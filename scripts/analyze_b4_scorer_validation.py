#!/usr/bin/env python3
"""Validate frozen GovMem deterministic scores against the B4 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b4_calibration_execution import LedgerConflictError
from scripts.prepare_b4_private_payloads import read_jsonl

ANALYSIS_VERSION = "b4-scorer-validation-v1"
EXPECTED_CASES = 604
EXPECTED_SCORER_VERSION = "final-main-source-aware-v1"
PROVIDER_ORDER = ("openai", "anthropic", "gemini", "deepseek", "kimi", "xai")
CONDITION_ORDER = ("clean", "corrupted", "enforced")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def deterministic_binary_label(correctness: Any, scoring_status: str) -> str:
    """Map only the frozen scorer's scored Boolean correctness contract."""
    if scoring_status != "scored":
        raise LedgerConflictError(f"deterministic result is not scored: {scoring_status}")
    if not isinstance(correctness, bool):
        raise LedgerConflictError("deterministic correctness must be Boolean")
    return "correct" if correctness else "incorrect"


def independent_binary_label(task: str, label: Any) -> str:
    if task == "longmemeval" and label in {"correct", "incorrect"}:
        return str(label)
    if task == "memora" and label in {"pass", "fail"}:
        return "correct" if label == "pass" else "incorrect"
    raise LedgerConflictError(f"invalid independent binary label for {task}: {label}")


def _source_tuple(row: dict[str, Any]) -> tuple[Any, str, str, str]:
    return (
        row.get("correctness"),
        str(row.get("scoring_status")),
        str(row.get("scoring_method")),
        str(row.get("scoring_version")),
    )


def _source_key_from_b4_id(b4_id: str, provider: str) -> tuple[str, str]:
    prefix = provider + "::"
    if not b4_id.startswith(prefix):
        raise LedgerConflictError(f"B4 validation ID/provider mismatch: {b4_id}")
    return provider, b4_id[len(prefix):]


def load_source_rows(paths: list[Path]) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    artifacts = []
    for path in paths:
        if not path.is_file():
            raise LedgerConflictError(f"frozen scorer artifact missing: {path}")
        artifact_rows = read_jsonl(path)
        providers = {str(row.get("provider")) for row in artifact_rows}
        if len(providers) != 1:
            raise LedgerConflictError(f"scorer artifact has ambiguous provider identity: {path}")
        for row in artifact_rows:
            key = (str(row.get("provider")), str(row.get("condition_id")))
            if key in rows:
                raise LedgerConflictError(f"duplicate frozen scorer row: {key}")
            rows[key] = row
        artifacts.append({
            "provider": next(iter(providers)),
            "source_path_locator": f"<external-b4-answer-root>/{path.parent.parent.name}/scored/{path.name}",
            "sha256": file_sha256(path),
            "row_n": len(artifact_rows),
        })
    versions = {str(row.get("scoring_version")) for row in rows.values()}
    if versions != {EXPECTED_SCORER_VERSION}:
        raise LedgerConflictError(f"frozen scorer version drift: {sorted(versions)}")
    return rows, artifacts


def _consistent_score(
    case_id: str,
    source_rows: list[dict[str, Any]],
) -> tuple[Any, str, str, str]:
    values = {_source_tuple(row) for row in source_rows}
    if len(values) != 1:
        raise LedgerConflictError(f"deduplicated scorer provenance is inconsistent: {case_id}")
    value = next(iter(values))
    if value[3] != EXPECTED_SCORER_VERSION:
        raise LedgerConflictError(f"scorer version mismatch for {case_id}: {value[3]}")
    return value


def extract_frozen_scorer_records(
    reference_cases: list[dict[str, Any]],
    source_rows: dict[tuple[str, str], dict[str, Any]],
    lme_provenance: list[dict[str, Any]],
    memora_provenance: list[dict[str, Any]],
    memora_requests: list[dict[str, Any]],
    memora_criteria: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reference_ids = [str(row["blinded_request_id"]) for row in reference_cases]
    if len(reference_ids) != len(set(reference_ids)):
        raise LedgerConflictError("duplicate independent reference case")
    lme_by_id = {str(row["blinded_request_id"]): row for row in lme_provenance}
    memora_by_id = {str(row["blinded_request_id"]): row for row in memora_provenance}
    request_by_id = {str(row["request_id"]): row for row in memora_requests}
    criterion_by_id = {str(row["criterion_id"]): row for row in memora_criteria}
    if len(lme_by_id) != len(lme_provenance) or len(memora_by_id) != len(memora_provenance):
        raise LedgerConflictError("duplicate B4 scorer provenance mapping")
    if len(request_by_id) != len(memora_requests) or len(criterion_by_id) != len(memora_criteria):
        raise LedgerConflictError("duplicate B4 request/criterion identity")

    scorer_records: list[dict[str, Any]] = []
    for reference in reference_cases:
        case_id = str(reference["blinded_request_id"])
        task = str(reference["task"])
        if task == "longmemeval":
            provenance = lme_by_id.get(case_id)
            if provenance is None or case_id in memora_by_id:
                raise LedgerConflictError(f"missing or ambiguous LongMemEval provenance: {case_id}")
            mapped_rows = []
            source_ids = []
            for source in provenance.get("source_provenance", []):
                provider = str(source["provider"])
                key = _source_key_from_b4_id(str(source["b4_validation_id"]), provider)
                row = source_rows.get(key)
                if row is None:
                    raise LedgerConflictError(f"missing frozen scorer row for {case_id}: {key}")
                if row.get("source_dataset") != "longmemeval":
                    raise LedgerConflictError(f"task identity drift for {case_id}")
                if canonical_value_sha256(row.get("parsed_answer")) != provenance.get("answer_hash"):
                    raise LedgerConflictError(f"answer identity drift for {case_id}: {key}")
                mapped_rows.append(row)
                source_ids.append({"provider": key[0], "condition_id": key[1]})
            if not mapped_rows:
                raise LedgerConflictError(f"LongMemEval case has empty scorer provenance: {case_id}")
            correctness, status, method, version = _consistent_score(case_id, mapped_rows)
            representative_provider = str(provenance["provider"])
            representative_condition = str(provenance["condition"])
            representative_matches = [
                item for item in source_ids
                if item["provider"] == representative_provider
                and source_rows[(item["provider"], item["condition_id"])].get("condition_type") == representative_condition
            ]
            if not representative_matches:
                raise LedgerConflictError(f"representative LongMemEval provenance drift: {case_id}")
            condition_id = representative_matches[0]["condition_id"]
            answer_hash = str(provenance["answer_hash"])
            subtype = None
            criterion_ids: list[str] = []
        elif task == "memora":
            provenance = memora_by_id.get(case_id)
            request = request_by_id.get(case_id)
            if provenance is None or request is None or case_id in lme_by_id:
                raise LedgerConflictError(f"missing or ambiguous Memora provenance: {case_id}")
            criterion_ids = [str(value) for value in provenance.get("criterion_ids", [])]
            if criterion_ids != [str(value) for value in request.get("deduplicated_criterion_ids", [])]:
                raise LedgerConflictError(f"Memora criterion provenance drift: {case_id}")
            mapped_rows = []
            source_ids = []
            criterion_subtypes = set()
            for criterion_id in criterion_ids:
                criterion = criterion_by_id.get(criterion_id)
                if criterion is None:
                    raise LedgerConflictError(f"unknown Memora criterion: {criterion_id}")
                key = (str(criterion["provider"]), str(criterion["original_condition_id"]))
                row = source_rows.get(key)
                if row is None:
                    raise LedgerConflictError(f"missing frozen scorer row for {case_id}: {key}")
                if row.get("source_dataset") != "memora":
                    raise LedgerConflictError(f"task identity drift for {case_id}")
                if canonical_value_sha256(row.get("parsed_answer")) != request.get("answer_sha256"):
                    raise LedgerConflictError(f"answer identity drift for {case_id}: {key}")
                mapped_rows.append(row)
                source_ids.append({"provider": key[0], "condition_id": key[1], "criterion_id": criterion_id})
                criterion_subtypes.add(str(criterion["subtype"]))
            if not mapped_rows or len(criterion_subtypes) != 1:
                raise LedgerConflictError(f"Memora subtype/provenance ambiguity: {case_id}")
            subtype = next(iter(criterion_subtypes))
            if subtype != reference.get("subtype") or subtype != request.get("subtype"):
                raise LedgerConflictError(f"Memora subtype identity drift: {case_id}")
            correctness, status, method, version = _consistent_score(case_id, mapped_rows)
            representative_provider = str(provenance["provider"])
            representative_condition = str(provenance["condition"])
            representative_matches = [
                item for item in source_ids
                if item["provider"] == representative_provider
                and source_rows[(item["provider"], item["condition_id"])].get("condition_type") == representative_condition
            ]
            if not representative_matches:
                raise LedgerConflictError(f"representative Memora provenance drift: {case_id}")
            condition_id = representative_matches[0]["condition_id"]
            answer_hash = str(request["answer_sha256"])
        else:
            raise LedgerConflictError(f"unknown reference task: {task}")
        scorer_records.append({
            "blinded_request_id": case_id,
            "task": task,
            "subtype": subtype,
            "generating_provider": representative_provider,
            "experimental_condition": representative_condition,
            "representative_condition_id": condition_id,
            "answer_identity_sha256": answer_hash,
            "deterministic_correctness": correctness,
            "deterministic_scoring_status": status,
            "deterministic_scoring_method": method,
            "deterministic_scoring_version": version,
            "deterministic_score_scope": "answer_level",
            "source_provenance": source_ids,
            "criterion_ids": criterion_ids,
        })
    return scorer_records


def strict_join(
    reference_cases: list[dict[str, Any]], scorer_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    scorer_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scorer_records:
        scorer_by_id[str(row["blinded_request_id"])].append(row)
    reference_ids = {str(row["blinded_request_id"]) for row in reference_cases}
    extras = set(scorer_by_id) - reference_ids
    if extras:
        raise LedgerConflictError(f"scorer mapping contains unknown cases: {sorted(extras)[:3]}")
    joined = []
    for reference in reference_cases:
        case_id = str(reference["blinded_request_id"])
        matches = scorer_by_id.get(case_id, [])
        if len(matches) != 1:
            raise LedgerConflictError(
                f"expected exactly one scorer mapping for {case_id}, found {len(matches)}"
            )
        scorer = matches[0]
        if scorer.get("task") != reference.get("task") or scorer.get("subtype") != reference.get("subtype"):
            raise LedgerConflictError(f"task/subtype identity drift: {case_id}")
        reference_label = independent_binary_label(
            str(reference["task"]), reference.get("final_reference_binary_label")
        )
        scorer_label = deterministic_binary_label(
            scorer.get("deterministic_correctness"), str(scorer.get("deterministic_scoring_status"))
        )
        agreement = reference_label == scorer_label
        disagreement_type = None
        if not agreement:
            disagreement_type = "false_positive" if scorer_label == "correct" else "false_negative"
        joined.append({
            "blinded_request_id": case_id,
            "task": reference["task"],
            "subtype": reference.get("subtype"),
            "generating_provider": scorer.get("generating_provider"),
            "experimental_condition": scorer.get("experimental_condition"),
            "independent_reference_binary_label": reference_label,
            "independent_reference_raw_label": reference.get("final_reference_raw_label"),
            "independent_resolution_method": reference.get("resolution_method"),
            "deterministic_scorer_binary_label": scorer_label,
            "deterministic_scoring_status": scorer.get("deterministic_scoring_status"),
            "deterministic_scoring_method": scorer.get("deterministic_scoring_method"),
            "deterministic_scoring_version": scorer.get("deterministic_scoring_version"),
            "deterministic_score_scope": scorer.get("deterministic_score_scope"),
            "independent_reference_scope": "answer_level" if reference["task"] == "longmemeval" else "atomic_criterion",
            "agreement_binary": agreement,
            "disagreement_type": disagreement_type,
            "immutable_provenance": {
                "representative_condition_id": scorer.get("representative_condition_id"),
                "answer_identity_sha256": scorer.get("answer_identity_sha256"),
                "criterion_ids": scorer.get("criterion_ids", []),
                "source_rows": scorer.get("source_provenance", []),
            },
            "provenance_locator": {
                "b4_mapping_artifact": (
                    "results/b4_scoring_validation/private/longmemeval_judge_provenance.jsonl"
                    if reference["task"] == "longmemeval"
                    else "results/b4_scoring_validation/private/memora_judge_provenance.jsonl"
                ),
                "blinded_request_id": case_id,
            },
        })
    return joined


def confusion_matrix(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    tp = sum(row["independent_reference_binary_label"] == "correct" and row["deterministic_scorer_binary_label"] == "correct" for row in values)
    fn = sum(row["independent_reference_binary_label"] == "correct" and row["deterministic_scorer_binary_label"] == "incorrect" for row in values)
    fp = sum(row["independent_reference_binary_label"] == "incorrect" and row["deterministic_scorer_binary_label"] == "correct" for row in values)
    tn = sum(row["independent_reference_binary_label"] == "incorrect" and row["deterministic_scorer_binary_label"] == "incorrect" for row in values)
    return {
        "orientation": "rows=independent reference; columns=deterministic scorer; positive=correct",
        "independent_correct": {"scorer_correct": tp, "scorer_incorrect": fn},
        "independent_incorrect": {"scorer_correct": fp, "scorer_incorrect": tn},
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
    }


def cohen_kappa(rows: Iterable[dict[str, Any]]) -> float | None:
    values = list(rows)
    if not values:
        return None
    refs = [row["independent_reference_binary_label"] for row in values]
    preds = [row["deterministic_scorer_binary_label"] for row in values]
    if len(set(refs)) < 2 or len(set(preds)) < 2:
        return None
    n = len(values)
    observed = sum(ref == pred for ref, pred in zip(refs, preds)) / n
    expected = sum((refs.count(label) / n) * (preds.count(label) / n) for label in ("correct", "incorrect"))
    if math.isclose(1.0 - expected, 0.0):
        return None
    return (observed - expected) / (1.0 - expected)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"confidence": 0.95, "lower": center - radius, "upper": center + radius}


def metric_block(rows: list[dict[str, Any]], *, include_ci: bool = False) -> dict[str, Any]:
    n = len(rows)
    agreement = sum(bool(row["agreement_binary"]) for row in rows)
    matrix = confusion_matrix(rows)
    kappa = cohen_kappa(rows)
    result = {
        "n": n,
        "agreement_n": agreement,
        "disagreement_n": n - agreement,
        "agreement_rate": agreement / n if n else None,
        "cohen_kappa": kappa,
        "cohen_kappa_status": "estimated" if kappa is not None else "not_estimable",
        "confusion_matrix": matrix,
        "false_positive_n": matrix["false_positive"],
        "false_negative_n": matrix["false_negative"],
    }
    if include_ci:
        result["agreement_rate_wilson_95ci"] = wilson_interval(agreement, n)
    return result


def grouped_metrics(
    rows: list[dict[str, Any]], key: str, values: Iterable[str] | None = None
) -> dict[str, dict[str, Any]]:
    groups = list(values) if values is not None else sorted({str(row.get(key)) for row in rows})
    return {group: metric_block([row for row in rows if row.get(key) == group]) for group in groups}


def summarize(joined: list[dict[str, Any]]) -> dict[str, Any]:
    task = {
        name: metric_block([row for row in joined if row["task"] == name], include_ci=True)
        for name in ("longmemeval", "memora")
    }
    memora_rows = [row for row in joined if row["task"] == "memora"]
    return {
        "schema_version": ANALYSIS_VERSION,
        "orientation": {
            "reference_positive": "independent correct/pass",
            "scorer_prediction_positive": "deterministic scorer correct",
            "false_positive": "scorer correct, independent incorrect",
            "false_negative": "scorer incorrect, independent correct",
        },
        "deterministic_binary_mapping": {
            "scoring_status=scored and correctness=true": "correct",
            "scoring_status=scored and correctness=false": "incorrect",
            "other status/value": "fail_closed_unscored",
        },
        "unit_alignment": {
            "longmemeval": "answer-level deterministic score versus answer-level independent reference",
            "memora": "answer-level deterministic score versus each deduplicated atomic-criterion independent reference; the frozen answer-level score is repeated for criteria sharing that answer",
        },
        "overall": metric_block(joined, include_ci=True),
        "by_task": task,
        "by_memora_subtype": grouped_metrics(memora_rows, "subtype", ("content_email_writeup", "content_project_proposal")),
        "by_generating_provider": grouped_metrics(joined, "generating_provider", PROVIDER_ORDER),
        "by_experimental_condition": grouped_metrics(joined, "experimental_condition", CONDITION_ORDER),
        "by_adjudication_resolution_method": grouped_metrics(joined, "independent_resolution_method"),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _fmt_rate(value: float | None) -> str:
    return "not available" if value is None else f"{100 * value:.2f}%"


def _fmt_kappa(block: dict[str, Any]) -> str:
    return "not estimable" if block["cohen_kappa"] is None else f"{block['cohen_kappa']:.6f}"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    lines = [
        "# B4 Deterministic Scorer Validation",
        "",
        "The frozen GovMem source-aware deterministic scorer is compared with the fully adjudicated independent semantic reference. No scoring or judging is executed by this analysis.",
        "",
        "## Mapping and orientation",
        "",
        "- Independent LongMemEval `correct`/`incorrect` and Memora `pass`/`fail` map to binary correct/incorrect.",
        "- Frozen deterministic `correctness=true/false` maps to correct/incorrect only when `scoring_status=scored`.",
        "- Positive means correct. FP means scorer-correct/reference-incorrect; FN means scorer-incorrect/reference-correct.",
        "- LongMemEval is answer-level versus answer-level. Memora compares the frozen answer-level deterministic score with each atomic criterion; the answer-level score is repeated for criteria sharing that answer.",
        "",
        "## Overall",
        "",
        f"- N: {overall['n']}",
        f"- agreement: {overall['agreement_n']}",
        f"- disagreement: {overall['disagreement_n']}",
        f"- agreement rate: {_fmt_rate(overall['agreement_rate'])}",
        f"- Cohen's kappa: {_fmt_kappa(overall)}",
        f"- false positives: {overall['false_positive_n']}",
        f"- false negatives: {overall['false_negative_n']}",
        f"- Wilson 95% CI: [{overall['agreement_rate_wilson_95ci']['lower']:.6f}, {overall['agreement_rate_wilson_95ci']['upper']:.6f}]",
        "",
        "## By task",
        "",
        "| Task | N | Agreement | Disagreement | Rate | Kappa | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("longmemeval", "memora"):
        block = summary["by_task"][name]
        lines.append(f"| {name} | {block['n']} | {block['agreement_n']} | {block['disagreement_n']} | {_fmt_rate(block['agreement_rate'])} | {_fmt_kappa(block)} | {block['false_positive_n']} | {block['false_negative_n']} |")
    for title, key in (
        ("Memora subtype", "by_memora_subtype"),
        ("Generating provider", "by_generating_provider"),
        ("Experimental condition", "by_experimental_condition"),
        ("Adjudication resolution method", "by_adjudication_resolution_method"),
    ):
        lines += ["", f"## By {title.lower()}", "", "| Group | N | Agreement | Disagreement | Rate |", "|---|---:|---:|---:|---:|"]
        for group, block in summary[key].items():
            lines.append(f"| {group} | {block['n']} | {block['agreement_n']} | {block['disagreement_n']} | {_fmt_rate(block['agreement_rate'])} |")
    lines += [
        "",
        "DeepSeek has N=0 because it was not part of the frozen 604-case semantic cohort; no provider identity is inferred.",
        "",
        "Disagreement records contain IDs, labels, scorer metadata, and immutable provenance only—no benchmark question, reference, or model-answer text.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("results/b4_scoring_validation/private/frontier_adjudication/adjudication_cases.jsonl"))
    parser.add_argument("--b4-summary", type=Path, default=Path("results/b4_scoring_validation/b4_validation_summary.json"))
    parser.add_argument("--lme-provenance", type=Path, default=Path("results/b4_scoring_validation/private/longmemeval_judge_provenance.jsonl"))
    parser.add_argument("--memora-provenance", type=Path, default=Path("results/b4_scoring_validation/private/memora_judge_provenance.jsonl"))
    parser.add_argument("--memora-requests", type=Path, default=Path("results/b4_scoring_validation/memora_judge_requests.jsonl"))
    parser.add_argument("--memora-criteria", type=Path, default=Path("results/b4_scoring_validation/memora_atomic_criteria.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/b4_scoring_validation/final_scorer_validation"))
    args = parser.parse_args(argv)

    reference = read_jsonl(args.reference)
    if len(reference) != EXPECTED_CASES or any(row.get("final_reference_status") != "resolved" for row in reference):
        raise SystemExit("independent reference is not the complete frozen 604-case reference")
    b4_summary = json.loads(args.b4_summary.read_text(encoding="utf-8"))
    if b4_summary.get("seed") != 20260928:
        raise SystemExit("B4 frozen cohort provenance/seed drift")
    scorer_paths = [Path(value) for value in b4_summary.get("answer_artifacts_used", [])]
    source_rows, source_artifacts = load_source_rows(scorer_paths)
    scorer_records = extract_frozen_scorer_records(
        reference,
        source_rows,
        read_jsonl(args.lme_provenance),
        read_jsonl(args.memora_provenance),
        read_jsonl(args.memora_requests),
        read_jsonl(args.memora_criteria),
    )
    joined = strict_join(reference, scorer_records)
    if len(joined) != EXPECTED_CASES:
        raise SystemExit(f"joined scorer case count mismatch: {len(joined)}")
    summary = summarize(joined)
    disagreements = [row for row in joined if not row["agreement_binary"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "scorer_validation_cases.jsonl", joined)
    _write_json(args.output_dir / "scorer_validation_summary.json", summary)
    _write_jsonl(args.output_dir / "scorer_disagreements.jsonl", disagreements)
    write_report(args.output_dir / "SCORER_VALIDATION_REPORT.md", summary)
    manifest = {
        "schema_version": ANALYSIS_VERSION,
        "analysis_script": "scripts/analyze_b4_scorer_validation.py",
        "total_cases": EXPECTED_CASES,
        "joined_cases": len(joined),
        "scorer_versions": sorted({row["deterministic_scoring_version"] for row in joined}),
        "independent_reference_sha256": file_sha256(args.reference),
        "b4_provenance_summary_sha256": file_sha256(args.b4_summary),
        "provenance_artifact_sha256": {
            "longmemeval": file_sha256(args.lme_provenance),
            "memora": file_sha256(args.memora_provenance),
            "memora_requests": file_sha256(args.memora_requests),
            "memora_criteria": file_sha256(args.memora_criteria),
        },
        "scorer_source_artifacts": source_artifacts,
        "source_selection_basis": "Exact full_results_scored.jsonl paths frozen in b4_validation_summary.json; no rescoring performed.",
        "unit_alignment": summary["unit_alignment"],
        "api_calls_made": 0,
    }
    _write_json(args.output_dir / "scorer_validation_manifest.json", manifest)

    overall = summary["overall"]
    print(f"TOTAL_REFERENCE_CASES={len(reference)}")
    print(f"JOINED_SCORER_CASES={len(joined)}")
    print(f"AGREEMENT_N={overall['agreement_n']}")
    print(f"DISAGREEMENT_N={overall['disagreement_n']}")
    print(f"AGREEMENT_RATE={overall['agreement_rate']:.6f}")
    print(f"COHEN_KAPPA={overall['cohen_kappa']:.6f}" if overall["cohen_kappa"] is not None else "COHEN_KAPPA=not_estimable")
    print(f"LONGMEMEVAL_AGREEMENT={summary['by_task']['longmemeval']['agreement_n']}/{summary['by_task']['longmemeval']['n']}")
    print(f"MEMORA_AGREEMENT={summary['by_task']['memora']['agreement_n']}/{summary['by_task']['memora']['n']}")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
