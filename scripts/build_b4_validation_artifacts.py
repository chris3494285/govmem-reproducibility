#!/usr/bin/env python3
"""Build reproducible, offline B4 validation inputs from existing artifacts.

This script only reads existing answer files and locally supplied benchmark
sources.  It never initializes a provider or executes a judge.  Cohort JSONL
files are local validation inputs and may contain answer/oracle text; the
manifests, criteria metadata, requests, and summaries are ID/hash-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.system_validation_scoring import memora_subtype, score_task_answer_detailed
from scripts.validate_b4_scoring import _write_json, validate_rows

SEED = 20260928
SAMPLE_N = 100
PROVIDERS = (
    "final_main_openai_gpt56_sol_final_v2",
    "final_main_kimi_k2_6_nothinking_final",
    "final_main_grok_43_final",
    "final_main_gemini_3_flash_final",
    "final_main_anthropic_sonnet5_final",
)
LME_DECISIONS = ("correct", "incorrect", "abstention_correct", "abstention_incorrect", "adjudication_required")


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def b4_validation_id(row: dict[str, Any], fallback: str) -> str:
    existing = row.get("b4_validation_id")
    if existing:
        return str(existing)
    provider = str(row.get("provider") or "unknown")
    original = str(row.get("condition_id") or row.get("base_id") or fallback)
    return f"{provider}::{original}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cohort(rows: list[dict[str, Any]], n: int, seed: int, key: str, extra_group=None) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        extra = str(extra_group(row)) if extra_group else ""
        groups[(str(row.get("condition_type", "unknown")), str(row.get("provider", "unknown")), extra)].append(row)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    # Round-robin over condition/provider strata gives every available provider
    # and fault condition representation before filling the requested size.
    strata = sorted(groups)
    while len(selected) < min(n, len(rows)):
        progressed = False
        for group in strata:
            if groups[group]:
                selected.append(groups[group].pop())
                progressed = True
                if len(selected) == n:
                    break
        if not progressed:
            break
    selected.sort(key=lambda row: str(row.get(key, row.get("condition_id", ""))))
    return selected


def public_row(row: dict[str, Any], index: int, source: str) -> dict[str, Any]:
    identifier = b4_validation_id(row, f"{source}-{index}")
    return {
        "sample_id": identifier,
        "original_condition_id": row.get("original_condition_id") or row.get("condition_id"),
        "source_dataset": source,
        "provider": row.get("provider"),
        "condition_type": row.get("condition_type"),
        "base_id_sha256": sha(row.get("base_id") or identifier),
        "answer_sha256": sha(row.get("model_answer", row.get("parsed_answer"))),
        "oracle_sha256": sha(row.get("task_oracle_answer")),
    }


def load_lme_source(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["question_id"]): row for row in json.loads(path.read_text(encoding="utf-8"))}


def build_lme(rows: list[dict[str, Any]], oracle: dict[str, dict[str, Any]], out: Path) -> dict[str, Any]:
    selected = cohort(rows, SAMPLE_N, SEED, "condition_id")
    local: list[dict[str, Any]] = []
    for row in selected:
        qid = str(row.get("base_id", "")).split("::")[1]
        source = oracle.get(qid)
        if not source:
            continue
        item = {
            "sample_id": row["condition_id"],
            "b4_validation_id": b4_validation_id(row, row["condition_id"]),
            "original_condition_id": row["condition_id"],
            "source_dataset": "longmemeval",
            "provider": row.get("provider"),
            "condition_type": row.get("condition_type"),
            "question_id": qid,
            "model_answer": row.get("parsed_answer"),
            "task_oracle_answer": source.get("answer"),
            "native_abstention_answers": ["i don't know", "it cannot be determined", "cannot be determined", "information is insufficient"],
            "native_answer_is_abstention": str(source.get("answer", "")).strip().lower() in {"i don't know", "it cannot be determined", "cannot be determined", "information is insufficient"},
        }
        local.append(item)
    (out / "longmemeval_cohort.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in local), encoding="utf-8")
    (out / "longmemeval_cohort_manifest.json").write_text(json.dumps({
        "schema_version": "b4-lme-cohort-v1", "seed": SEED, "sampling": "round_robin_condition_type_provider",
        "sample_n": len(local), "source_examples": len(oracle),
        "rows": [public_row(x, i, "longmemeval") for i, x in enumerate(local)],
        "text_policy": "manifest contains hashes and IDs only; cohort JSONL is local validation input",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary, details, queue = validate_rows(local, "longmemeval")
    # Native judge labels are not inferred from the oracle. Deduplicate only
    # exact question/answer pairs; preserve all provider/condition provenance.
    cases: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for x in local:
        cases[(x["question_id"], sha(x["model_answer"]))].append(x)
    requests = []
    for (qid, answer_hash), values in sorted(cases.items()):
        representative = values[0]
        requests.append({
            "request_id": f"longmemeval::{qid}::{answer_hash[:16]}",
            "question_id": qid,
            "answer_hash": answer_hash,
            "answer": representative["model_answer"],
            "question": oracle[qid].get("question"),
            "reference_answer": representative["task_oracle_answer"],
            "native_abstention_answers": representative["native_abstention_answers"],
            "condition": representative["condition_type"],
            "provider": representative["provider"],
            "provenance": [{"provider": x["provider"], "condition": x["condition_type"], "b4_validation_id": x["b4_validation_id"]} for x in values],
            "decision": None,
            "decision_schema": list(LME_DECISIONS),
            "execution": "dry_run_only",
        })
    (out / "longmemeval_judge_requests.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in requests), encoding="utf-8")
    _write_json(out / "longmemeval_judge_request_summary.json", {"schema_version": "b4-lme-judge-plan-v2", "original_cohort_rows": len(local), "unique_judgment_cases": len(cases), "safely_deduplicated_cases": len(requests), "provider_distribution": dict(Counter(x["provider"] for x in local)), "condition_distribution": dict(Counter(x["condition_type"] for x in local)), "supported_decision_categories": list(LME_DECISIONS), "execution": "dry_run_only"})
    return {"summary": summary, "details": details, "judge_request_n": len(requests), "cohort_n": len(local), "condition_counts": dict(Counter(x["condition_type"] for x in local)), "provider_counts": dict(Counter(x["provider"] for x in local)), "unique_judgment_cases": len(cases), "decision_categories": list(LME_DECISIONS)}


def load_oaks_source(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for book in json.loads(path.read_text(encoding="utf-8")):
        for question in book["data"]["qas"].values():
            result[str(question["question_id"])] = question
    return result


def load_oaks_labels(path: Path | None) -> dict[str, str]:
    if not path or not path.is_file():
        return {}
    labels = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("eligibility") and row.get("new_answer"):
            labels[str(row["question_id"])] = str(row["new_answer"])
    return labels


def build_oaks(answer_paths: list[Path], source: dict[str, dict[str, Any]], labels: dict[str, str], out: Path) -> dict[str, Any]:
    local = []
    for path in answer_paths:
        provider_rows = read_jsonl(path)
        for row in provider_rows:
            if row.get("source_dataset") != "oaks_novel":
                continue
            parts = str(row["condition_id"]).split("::")
            qid = parts[2] if len(parts) > 2 else ""
            native = source.get(qid)
            if not native:
                continue
            item = {"sample_id": row["condition_id"], "b4_validation_id": b4_validation_id(row, row["condition_id"]), "original_condition_id": row["condition_id"], "source_dataset": "oaks_novel", "provider": row.get("provider"), "condition_type": row.get("condition_type"), "question_id": qid, "model_answer": row.get("parsed_answer"), "task_oracle_answer": row.get("task_oracle_answer"), "task_options": native.get("options", [])}
            # The native answer is the source option selected by the answer key.
            # It is available in the existing scored artifact as oracle_option;
            # source options are used for the independent resolver.
            item["task_oracle_answer"] = labels.get(qid) or row.get("scoring_details", {}).get("oracle_option") or row.get("task_oracle_answer")
            resolved = score_task_answer_detailed(item, item["model_answer"])
            item["native_correctness"] = resolved.get("deterministic_score")
            local.append(item)
    local.sort(key=lambda x: str(x["sample_id"]))
    (out / "oaks_native_validation_input.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in local), encoding="utf-8")
    validated, _, _ = validate_rows(local, "oaks_novel")
    report = {"schema_version": "oaks-native-input-v1", "source_question_n": len(source), "validation_row_n": len(local), "provider_counts": dict(Counter(x["provider"] for x in local)), "condition_counts": dict(Counter(x["condition_type"] for x in local)), "agreements": validated["agreement_n"], "disagreements": validated["disagreement_n"], "agreement_rate": validated["agreement_rate"], "ambiguous_unscored": validated["unscored_n"] + validated["adjudication_required_n"]}
    _write_json(out / "oaks_input_summary.json", report)
    return report


def load_memora_questions(root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in root.glob("*/**/evaluation_questions_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for kind in data.get("questions", {}).values():
            if isinstance(kind, list):
                for q in kind:
                    result[str(q.get("question_id"))].extend(q.get("evaluation", {}).get("evaluation_questions", []))
    return result


def build_memora(rows: list[dict[str, Any]], native_questions: dict[str, list[dict[str, Any]]], out: Path) -> dict[str, Any]:
    selected = cohort(rows, SAMPLE_N, SEED, "condition_id")
    local = []
    criteria = []
    requests = []
    for row in selected:
        item = {"sample_id": row["condition_id"], "b4_validation_id": b4_validation_id(row, row["condition_id"]), "original_condition_id": row["condition_id"], "source_dataset": "memora", "provider": row.get("provider"), "condition_type": row.get("condition_type"), "model_answer": row.get("parsed_answer"), "task_oracle_answer": row.get("task_oracle_answer"), "task_question": row.get("task_question")}
        local.append(item)
        base = str(row.get("base_id", "")).split("::")[-1]
        subtype = memora_subtype(row.get("task_oracle_answer"))
        native = native_questions.get(base, [])
        for index, nq in enumerate(native):
            etype = str(nq.get("evaluation_type", "unknown"))
            mode = "judge_required" if subtype in {"content_email_writeup", "content_meeting_notes", "content_project_proposal"} else "native_exact"
            if subtype == "preference_recommendation" and etype == "memory_presence":
                mode = "adjudication_required"
            criterion_id = f"{item['b4_validation_id']}::criterion-{index}"
            criterion = {"criterion_id": criterion_id, "sample_id": item["b4_validation_id"], "original_condition_id": item["original_condition_id"], "provider": item["provider"], "subtype": subtype, "criterion_type": etype, "expected_behavior": str(nq.get("expected_answer")), "source_field": "evaluation.evaluation_questions", "evaluation_mode": mode, "native_question_id": nq.get("evaluation_question_id"), "criterion_text_sha256": sha(nq.get("evaluation_question"))}
            criteria.append(criterion)
            if mode == "judge_required":
                requests.append({"criterion_id": criterion_id, "sample_id": item["b4_validation_id"], "original_condition_id": item["original_condition_id"], "provider": item["provider"], "condition": item["condition_type"], "subtype": subtype, "rubric": "criterion-level pass/fail; assess only the native field/fact", "answer_sha256": sha(row.get("parsed_answer")), "criterion_text_sha256": criterion["criterion_text_sha256"], "judge_count": 3, "execution": "dry_run_only"})
    (out / "memora_cohort.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in local), encoding="utf-8")
    (out / "memora_cohort_manifest.json").write_text(json.dumps({"schema_version": "b4-memora-cohort-v1", "seed": SEED, "sampling": "round_robin_condition_type_provider", "sample_n": len(local), "rows": [public_row(x, i, "memora") for i, x in enumerate(local)], "text_policy": "manifest contains hashes and IDs only; cohort JSONL is local validation input"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "memora_atomic_criteria.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in criteria), encoding="utf-8")
    # Deduplicate exact logical judge cases while retaining provenance.
    dedup: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    criterion_map = {x["criterion_id"]: x for x in criteria}
    for request in requests:
        c = criterion_map[request["criterion_id"]]
        dedup[(request["answer_sha256"], request["criterion_text_sha256"], c["subtype"], c["criterion_type"], c["expected_behavior"])].append(request)
    paid_requests = []
    for key, values in sorted(dedup.items()):
        representative = dict(values[0])
        representative["request_id"] = f"memora::judge::{sha(key)[:16]}"
        representative["deduplicated_criterion_ids"] = [x["criterion_id"] for x in values]
        representative["provenance"] = [{"sample_id": x["sample_id"], "provider": x["provider"], "condition": x["condition"]} for x in values]
        paid_requests.append(representative)
    adjudication = [x for x in criteria if x["evaluation_mode"] == "adjudication_required"]
    (out / "memora_judge_requests.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in paid_requests), encoding="utf-8")
    (out / "memora_adjudication_queue.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in adjudication), encoding="utf-8")
    counts = Counter((x["subtype"], x["evaluation_mode"]) for x in criteria)
    summary = {"schema_version": "b4-memora-criteria-v2", "sample_n": len(local), "criterion_n": len(criteria), "native_exact": sum(v for (s, m), v in counts.items() if m == "native_exact"), "judge_required_before_deduplication": len(requests), "judge_required_after_safe_deduplication": len(paid_requests), "adjudication_required": len(adjudication), "planned_judges_per_judge_required_case": 3, "total_planned_paid_judge_calls": len(paid_requests) * 3, "by_subtype": {s: {m: v for (ss, m), v in counts.items() if ss == s} for s in sorted({s for s, _ in counts})}, "execution": "dry_run_only"}
    _write_json(out / "memora_judge_request_summary.json", summary)
    scored_summary, details, queue = validate_rows(local, "memora")
    _write_json(out / "memora_judge_request_summary.json", summary)
    return {"summary": scored_summary, "criteria": summary, "judge_request_n": len(paid_requests), "cohort_n": len(local), "subtype_counts": dict(Counter(memora_subtype(x["task_oracle_answer"]) for x in local))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--lme-source", type=Path, required=True)
    parser.add_argument("--oaks-source", type=Path, required=True)
    parser.add_argument("--oaks-labels", type=Path, help="local native OAKS answer-state audit JSONL")
    parser.add_argument("--memora-source", type=Path, required=True)
    parser.add_argument("--answer-root", type=Path, required=True)
    parser.add_argument("--deepseek-answer", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/b4_scoring_validation"))
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    answer_paths = [args.answer_root / name / "scored" / "full_results_scored.jsonl" for name in PROVIDERS]
    scored_paths = [p for p in answer_paths if p.is_file()]
    oaks_paths = list(scored_paths)
    if args.deepseek_answer and args.deepseek_answer.is_file():
        oaks_paths.append(args.deepseek_answer)
    all_rows = [row for path in scored_paths for row in read_jsonl(path)]
    lme = build_lme([x for x in all_rows if x.get("source_dataset") == "longmemeval"], load_lme_source(args.lme_source), out)
    oaks = build_oaks(oaks_paths, load_oaks_source(args.oaks_source), load_oaks_labels(args.oaks_labels), out)
    memora = build_memora([x for x in all_rows if x.get("source_dataset") == "memora"], load_memora_questions(args.memora_source), out)
    summary = {"schema_version": "b4-validation-rebuild-v2", "status": "incomplete_native_judgments_not_executed", "seed": SEED, "answer_artifacts_used": [str(p) for p in scored_paths], "oaks_answer_artifacts_used": [str(p) for p in oaks_paths], "source_paths": {"longmemeval": str(args.lme_source), "oaks_novel": str(args.oaks_source), "memora": str(args.memora_source)}, "longmemeval": lme, "oaks_novel": oaks, "memora": memora, "native_claim_policy": "OAKS native option parity is computed offline; LongMemEval and Memora semantic/native judgments were not executed."}
    _write_json(out / "b4_validation_summary.json", summary)
    (out / "b4_validation_summary.md").write_text("# B4 validation rebuild\n\nStatus: **INCOMPLETE**: native/semantic judgments were not executed.\n\n- OAKS: offline native option parity prepared.\n- LongMemEval: deduplicated judge requests prepared; abstention categories are explicit.\n- Memora: native-exact criteria, paid-judge requests, and a separate human/native adjudication queue are prepared.\n\nNo answer-generation or judge calls were made.\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "lme": lme["cohort_n"], "oaks": oaks["validation_row_n"], "memora": memora["cohort_n"], "memora_criteria": memora["criteria"]["criterion_n"], "judge_calls": memora["criteria"]["total_planned_paid_judge_calls"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
