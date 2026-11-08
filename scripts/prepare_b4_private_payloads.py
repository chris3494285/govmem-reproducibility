#!/usr/bin/env python3
"""Prepare blinded, offline-only B4 semantic-judge payloads.

This module only reads local artifacts and writes private execution payloads.
It contains no provider clients and no judge executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_b4_validation_artifacts import load_lme_source, load_memora_questions

LME_DECISIONS = ["correct", "incorrect", "abstention_correct", "abstention_incorrect", "adjudication_required"]
MEMORA_DECISIONS = ["pass", "fail", "adjudication_required"]

# These are rejected recursively in judge-facing payloads.  Provenance files
# are not passed to this validator and may contain these fields.
FORBIDDEN_KEYS = {
    "provider", "model", "model_id", "model_name", "condition", "condition_type",
    "deterministic_score", "native_validation_score", "correctness", "scoring_status",
    "scoring_method", "scoring_version", "previous_decision", "previous_score",
    "govmem_decision", "govmem_score", "agreement", "agreement_rate", "judge_output",
    "other_judges", "aggregate_b4", "b4_statistics", "desired_agreement", "target_decision",
}


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}::{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _walk_keys(value: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            yield path, key_text
            yield from _walk_keys(child, f"{path}.{key_text}" if path else key_text)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{path}[{index}]")


def validate_payload_records(records: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    """Validate judge-facing records without inspecting provenance files."""
    if kind not in {"longmemeval", "memora", "memora_human"}:
        raise ValueError(f"unknown payload kind: {kind}")
    errors: list[str] = []
    required = {
        "longmemeval": ("blinded_request_id", "question", "native_context", "reference_answer", "model_answer", "decision_schema"),
        "memora": ("blinded_request_id", "native_context", "criterion", "expected_behavior", "model_answer", "subtype", "decision_schema"),
        "memora_human": ("blinded_annotation_id", "native_context", "criterion", "expected_behavior", "model_answer", "decision_schema"),
    }[kind]
    expected_schema = set(LME_DECISIONS if kind == "longmemeval" else MEMORA_DECISIONS)
    for index, record in enumerate(records):
        prefix = f"record {index}"
        for path, key in _walk_keys(record):
            if key.lower() in FORBIDDEN_KEYS:
                errors.append(f"{prefix}: forbidden key at {path}.{key}")
        for key in required:
            value = record.get(key)
            if value is None or value == "" or value == [] or value == {}:
                errors.append(f"{prefix}: required field {key!r} is empty")
        schema = record.get("decision_schema")
        if set(schema or []) != expected_schema:
            errors.append(f"{prefix}: decision_schema does not match {sorted(expected_schema)!r}")
    return {"valid": not errors, "record_count": len(records), "errors": errors}


def _native_memora_questions(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    # Re-read the native files so the private payload can retain the exact
    # evaluation-question wording, not only its hash.
    for path in root.glob("*/**/evaluation_questions_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for entries in data.get("questions", {}).values():
            if not isinstance(entries, list):
                continue
            for question in entries:
                for criterion in question.get("evaluation", {}).get("evaluation_questions", []):
                    identifier = criterion.get("evaluation_question_id")
                    if identifier:
                        result[str(identifier)] = {
                            "evaluation_question": criterion.get("evaluation_question"),
                            "expected_answer": criterion.get("expected_answer"),
                            "evaluation_type": criterion.get("evaluation_type"),
                            "source_question_id": question.get("question_id"),
                        }
    return result


def prepare_longmemeval(cohort_path: Path, request_path: Path, oracle_path: Path, out: Path) -> dict[str, Any]:
    cohort = read_jsonl(cohort_path)
    requests = read_jsonl(request_path)
    oracle = load_lme_source(oracle_path)
    private: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for request in requests:
        source = oracle.get(str(request["question_id"]))
        if not source:
            raise ValueError(f"LongMemEval question missing from oracle: {request['question_id']}")
        context = {
            "question_type": source.get("question_type"),
            "question_date": source.get("question_date"),
            "haystack_dates": source.get("haystack_dates"),
            "haystack_session_ids": source.get("haystack_session_ids"),
            "haystack_sessions": source.get("haystack_sessions"),
            "answer_session_ids": source.get("answer_session_ids"),
        }
        if not context.get("haystack_sessions"):
            raise ValueError(f"LongMemEval case lacks native haystack context: {request['question_id']}")
        private.append({
            "blinded_request_id": request["request_id"],
            "question": request["question"],
            "native_context": context,
            "reference_answer": request["reference_answer"],
            "model_answer": request["answer"],
            "native_abstention_convention": request["native_abstention_answers"],
            "decision_schema": LME_DECISIONS,
        })
        provenance.append({
            "blinded_request_id": request["request_id"],
            "question_id": request["question_id"],
            "answer_hash": request["answer_hash"],
            "provider": request.get("provider"),
            "condition": request.get("condition"),
            "source_provenance": request.get("provenance", []),
        })
    validation = validate_payload_records(private, "longmemeval")
    write_jsonl(out / "longmemeval_judge_execution.jsonl", private)
    write_jsonl(out / "longmemeval_judge_provenance.jsonl", provenance)
    return {
        "cohort_rows": len(cohort),
        "deduplicated_execution_cases": len(private),
        "records_with_context": sum(bool(x.get("native_context")) for x in private),
        "records_lacking_sufficient_context": len(private) - sum(bool(x.get("native_context")) for x in private),
        "provenance_records": len(provenance),
        "validation": validation,
    }


def prepare_memora(cohort_path: Path, request_path: Path, criteria_path: Path, native_root: Path, out: Path) -> dict[str, Any]:
    cohort = read_jsonl(cohort_path)
    requests = read_jsonl(request_path)
    criteria = read_jsonl(criteria_path)
    criteria_by_id = {str(x["criterion_id"]): x for x in criteria}
    cohort_by_id = {str(x["b4_validation_id"]): x for x in cohort}
    native = _native_memora_questions(native_root)
    private: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for request in requests:
        criterion_id = str(request["deduplicated_criterion_ids"][0])
        criterion = criteria_by_id[criterion_id]
        sample_id = str(request["sample_id"])
        sample = cohort_by_id[sample_id]
        native_criterion = native.get(str(criterion["native_question_id"]))
        if not native_criterion or not native_criterion.get("evaluation_question"):
            raise ValueError(f"Memora criterion text missing: {criterion['native_question_id']}")
        answer = sample.get("model_answer")
        if answer is None or answer == "":
            raise ValueError(f"Memora model answer missing: {sample_id}")
        private.append({
            "blinded_request_id": request["request_id"],
            "native_context": {
                "task_question": sample.get("task_question"),
                "native_task_fields": sample.get("task_oracle_answer"),
            },
            "criterion": native_criterion["evaluation_question"],
            "expected_behavior": native_criterion["expected_answer"],
            "model_answer": answer,
            "subtype": criterion["subtype"],
            "decision_schema": MEMORA_DECISIONS,
        })
        provenance.append({
            "blinded_request_id": request["request_id"],
            "criterion_ids": request["deduplicated_criterion_ids"],
            "provider": request.get("provider"),
            "condition": request.get("condition"),
            "original_condition_ids": [criteria_by_id[x]["original_condition_id"] for x in request["deduplicated_criterion_ids"]],
        })
    adjudication = [x for x in criteria if x.get("evaluation_mode") == "adjudication_required"]
    human: list[dict[str, Any]] = []
    human_provenance: list[dict[str, Any]] = []
    for criterion in adjudication:
        sample = cohort_by_id[str(criterion["sample_id"])]
        native_criterion = native.get(str(criterion["native_question_id"]))
        if not native_criterion or not native_criterion.get("evaluation_question"):
            raise ValueError(f"Memora adjudication criterion text missing: {criterion['native_question_id']}")
        human.append({
            "blinded_annotation_id": stable_id("memora-human", criterion["criterion_id"]),
            "native_context": {"task_question": sample.get("task_question"), "native_task_fields": sample.get("task_oracle_answer")},
            "criterion": native_criterion["evaluation_question"],
            "expected_behavior": native_criterion["expected_answer"],
            "model_answer": sample.get("model_answer"),
            "decision_schema": MEMORA_DECISIONS,
        })
        human_provenance.append({
            "blinded_annotation_id": stable_id("memora-human", criterion["criterion_id"]),
            "criterion_id": criterion["criterion_id"],
            "provider": criterion.get("provider"),
            "condition": sample.get("condition_type"),
            "original_condition_id": criterion.get("original_condition_id"),
        })
    validation = validate_payload_records(private, "memora")
    human_validation = validate_payload_records(human, "memora_human")
    write_jsonl(out / "memora_judge_execution.jsonl", private)
    write_jsonl(out / "memora_judge_provenance.jsonl", provenance)
    write_jsonl(out / "memora_human_adjudication.jsonl", human)
    write_jsonl(out / "memora_human_provenance.jsonl", human_provenance)
    return {
        "judge_required_before_deduplication": sum(1 for x in criteria if x.get("evaluation_mode") == "judge_required"),
        "semantic_execution_cases_after_deduplication": len(private),
        "human_adjudication_cases": len(human),
        "execution_records_with_model_answer": sum(bool(x.get("model_answer")) for x in private),
        "execution_records_with_criterion_text": sum(bool(x.get("criterion")) for x in private),
        "provenance_records": len(provenance),
        "validation": validation,
        "human_validation": human_validation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lme-cohort", type=Path, default=Path("results/b4_scoring_validation/longmemeval_cohort.jsonl"))
    parser.add_argument("--lme-requests", type=Path, default=Path("results/b4_scoring_validation/longmemeval_judge_requests.jsonl"))
    parser.add_argument("--lme-source", type=Path, required=True)
    parser.add_argument("--memora-cohort", type=Path, default=Path("results/b4_scoring_validation/memora_cohort.jsonl"))
    parser.add_argument("--memora-requests", type=Path, default=Path("results/b4_scoring_validation/memora_judge_requests.jsonl"))
    parser.add_argument("--memora-criteria", type=Path, default=Path("results/b4_scoring_validation/memora_atomic_criteria.jsonl"))
    parser.add_argument("--memora-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/b4_scoring_validation/private"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lme = prepare_longmemeval(args.lme_cohort, args.lme_requests, args.lme_source, args.output_dir)
    memora = prepare_memora(args.memora_cohort, args.memora_requests, args.memora_criteria, args.memora_source, args.output_dir)
    summary = {"schema_version": "b4-private-payload-summary-v1", "status": "ready_for_review_not_execution", "longmemeval": lme, "memora": memora, "execution": "disabled; payload preparation only"}
    (args.output_dir / "payload_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "lme_cases": lme["deduplicated_execution_cases"], "memora_cases": memora["semantic_execution_cases_after_deduplication"], "human_cases": memora["human_adjudication_cases"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
