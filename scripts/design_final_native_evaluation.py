#!/usr/bin/env python3
"""Freeze an offline native/native-style evaluation design for existing answers.

This module never imports a provider client and never executes a judge.  Its
tracked outputs contain only aggregate statistics, immutable identifiers, and
hashes; benchmark text and model answers are used locally only to route cases
and estimate request sizes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DESIGN_VERSION = "final-native-evaluation-design-v1"
SEED = 20261004
LME_ROUTES = {"deterministic_native", "deterministic_abstention", "semantic_judge_required", "invalid_unscorable"}
MEMORA_ROUTES = {"deterministic_native", "deterministic_absence_check", "semantic_judge_required", "invalid_unscorable"}
ABSTENTION_RE = re.compile(
    r"^\s*(?:i\s+(?:do\s+not|don't)\s+know|(?:it\s+)?(?:cannot|can't|can\s+not)\s+be\s+determined|"
    r"(?:there\s+is\s+)?insufficient\s+information|not\s+enough\s+information|"
    r"the\s+information(?:\s+provided)?\s+is\s+not\s+enough|unable\s+to\s+determine)\b",
    re.IGNORECASE,
)
YES = {"yes", "true"}
NO = {"no", "false"}
UNIT_ALIASES = {
    "day": "day", "days": "day", "week": "week", "weeks": "week",
    "month": "month", "months": "month", "year": "year", "years": "year",
    "hour": "hour", "hours": "hour", "minute": "minute", "minutes": "minute",
    "percent": "percent", "%": "percent", "dollar": "usd", "dollars": "usd", "usd": "usd",
}


def sha256_value(value: Any) -> str:
    raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def normalize_surface(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("_", " ").replace("-", " ")
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return " ".join(text.split())


def raw_benchmark_alternatives(reference: Any) -> list[str]:
    """Return only alternatives explicitly encoded by the benchmark answer."""
    if reference is None:
        return []
    text = str(reference).strip()
    parts = re.split(r"\s+or\s+", text, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


def benchmark_alternatives(reference: Any) -> list[str]:
    return [normalize_surface(part) for part in raw_benchmark_alternatives(reference) if normalize_surface(part)]


def canonical_scalar(value: Any) -> tuple[str, str] | None:
    """Canonicalize a whole-answer numeric scalar; never search within prose."""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"^(?:the\s+answer\s+is|it\s+was|it\s+is)\s+", "", text)
    text = text.rstrip(". ")
    match = re.fullmatch(r"\$?\s*([+-]?[0-9][0-9,]*(?:\.[0-9]+)?)\s*([km])?\s*(%|[a-z]+)?", text)
    if not match:
        return None
    number = match.group(1).replace(",", "")
    numeric = float(number)
    if match.group(2) == "k":
        numeric *= 1_000
    elif match.group(2) == "m":
        numeric *= 1_000_000
    canonical_number = str(int(numeric)) if numeric.is_integer() else format(numeric, ".15g")
    unit = UNIT_ALIASES.get(match.group(3) or "", match.group(3) or "")
    return canonical_number, unit


def canonical_date(value: Any) -> str | None:
    text = str(value).strip().rstrip(".")
    for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%B %d %Y", "%b %d %Y"):
        try:
            return dt.datetime.strptime(text.replace(",", ""), pattern).date().isoformat()
        except ValueError:
            continue
    return None


def deterministic_native_equivalent(answer: str, reference: Any) -> tuple[bool, str | None]:
    answer_norm = normalize_surface(answer)
    raw_accepted = raw_benchmark_alternatives(reference)
    accepted = benchmark_alternatives(reference)
    if answer_norm and answer_norm in accepted:
        return True, "normalized_exact_or_explicit_native_alternative"
    if answer_norm in YES | NO and any(item in YES | NO and (item in YES) == (answer_norm in YES) for item in accepted):
        return True, "canonical_yes_no_equivalence"
    answer_scalar = canonical_scalar(answer)
    reference_scalars = {canonical_scalar(item) for item in raw_accepted}
    reference_scalars.discard(None)
    if answer_scalar is not None and answer_scalar in reference_scalars:
        return True, "whole_answer_numeric_unit_normalization"
    answer_date = canonical_date(answer)
    reference_dates = {canonical_date(item) for item in raw_accepted}
    reference_dates.discard(None)
    if answer_date is not None and answer_date in reference_dates:
        return True, "whole_answer_date_normalization"
    return False, None


def is_valid_answer_row(row: dict[str, Any]) -> bool:
    return row.get("status") == "success" and row.get("parse_success") is not False and isinstance(row.get("parsed_answer"), str) and bool(row["parsed_answer"].strip())


def native_lme_answerability(native: dict[str, Any]) -> str:
    qid = str(native.get("question_id", ""))
    reference = str(native.get("answer", ""))
    if qid.endswith("_abs") or ABSTENTION_RE.match(reference):
        return "insufficient_information"
    if reference.strip():
        return "answerable"
    return "unknown"


def route_longmemeval(row: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    if not is_valid_answer_row(row):
        return {"route": "invalid_unscorable", "reason": "missing_or_failed_existing_answer"}
    reference = native.get("answer")
    if not str(reference or "").strip():
        return {"route": "invalid_unscorable", "reason": "missing_native_reference"}
    answer = row["parsed_answer"]
    matched, method = deterministic_native_equivalent(answer, reference)
    if matched:
        return {"route": "deterministic_native", "reason": method}
    if ABSTENTION_RE.match(answer):
        answerability = native_lme_answerability(native)
        if answerability != "unknown":
            return {
                "route": "deterministic_abstention",
                "reason": "explicit_native_answerability_plus_canonical_abstention",
                "answerability": answerability,
                "native_decision": "abstention_correct" if answerability == "insufficient_information" else "abstention_incorrect",
            }
    return {"route": "semantic_judge_required", "reason": "not_safely_resolved_by_native_surface_rules"}


def memora_key(base_id: str) -> tuple[str, str, str]:
    parts = base_id.split("::", 3)
    if len(parts) != 4 or parts[0] != "memora":
        raise ValueError(f"invalid Memora base_id: {base_id}")
    return parts[1], parts[2], parts[3]


def load_lme_source(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {str(row["question_id"]): row for row in data}
    if len(result) != len(data):
        raise ValueError("duplicate LongMemEval question_id")
    return result


def load_memora_source(root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(root.glob("*/*/evaluation_questions_*.json")):
        period, persona = path.parts[-3], path.parts[-2]
        data = json.loads(path.read_text(encoding="utf-8"))
        for task_group, questions in data.get("questions", {}).items():
            for question in questions:
                key = (period, persona, str(question["question_id"]))
                if key in result:
                    raise ValueError(f"duplicate Memora native identity: {key}")
                result[key] = {**question, "native_task_group": task_group, "source_path": str(path)}
    return result


def load_answer_paths(repo_root: Path) -> list[Path]:
    summary_path = repo_root / "results/b4_scoring_validation/b4_validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = summary.get("answer_paths") or summary.get("inputs", {}).get("answer_paths")
    if not candidates:
        # The current summary stores the frozen paths in both its top-level
        # provenance and input section; recursively locate only explicit JSONL paths.
        def walk(value: Any) -> Iterable[str]:
            if isinstance(value, dict):
                for child in value.values():
                    yield from walk(child)
            elif isinstance(value, list):
                for child in value:
                    yield from walk(child)
            elif isinstance(value, str) and value.endswith(".jsonl") and "final_main_" in value:
                yield value
        candidates = list(dict.fromkeys(walk(summary)))
    paths = [Path(item) for item in candidates]
    existing = list(dict.fromkeys(path.resolve() for path in paths if path.exists()))
    if len(existing) != 6:
        raise ValueError(f"expected six frozen answer artifacts, found {len(existing)}: {existing}")
    return existing


def load_existing_answers(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    lme: list[dict[str, Any]] = []
    memora: list[dict[str, Any]] = []
    provenance = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        rows = read_jsonl(path)
        selected = [row for row in rows if row.get("source_dataset") in {"longmemeval", "memora"}]
        providers = {str(row.get("provider")) for row in selected}
        if len(providers) != 1:
            raise ValueError(f"answer artifact must contain exactly one provider: {path}")
        for row in selected:
            identity = (str(row.get("provider")), str(row.get("condition_id")))
            if identity in seen:
                raise ValueError(f"duplicate answer identity: {identity}")
            seen.add(identity)
            (lme if row["source_dataset"] == "longmemeval" else memora).append(row)
        run_name = path.parent.parent.name if path.parent.name == "scored" else path.parent.name
        suffix = f"scored/{path.name}" if path.parent.name == "scored" else path.name
        provenance.append({"provider": next(iter(providers)), "path_locator": f"<external-answer-root>/{run_name}/{suffix}", "sha256": sha256_file(path), "selected_row_n": len(selected)})
    return lme, memora, {"answer_artifacts": sorted(provenance, key=lambda x: x["provider"])}


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def size_summary(values: list[int]) -> dict[str, Any]:
    tokens = [math.ceil(value / 4) for value in values]
    return {
        "n": len(values), "min_chars": min(values) if values else 0,
        "median_chars": statistics.median(values) if values else 0,
        "mean_chars": statistics.mean(values) if values else 0,
        "p90_chars": percentile(values, 0.9), "max_chars": max(values) if values else 0,
        "total_chars": sum(values), "estimated_input_tokens_div4": sum(tokens),
        "estimated_token_distribution_div4": {
            "min": min(tokens) if tokens else 0, "median": statistics.median(tokens) if tokens else 0,
            "mean": statistics.mean(tokens) if tokens else 0, "p90": percentile(tokens, 0.9), "max": max(tokens) if tokens else 0,
        },
        "token_estimation_note": "conservative planning approximation ceil(UTF-8 character count / 4); not a model tokenizer",
    }


def public_case_id(row: dict[str, Any]) -> str:
    return f"{row['source_dataset']}::{row['provider']}::{sha256_value([row['condition_id'], row.get('parsed_answer')])[:20]}"


def lme_prompt(row: dict[str, Any], native: dict[str, Any]) -> str:
    return json.dumps({
        "task": "Evaluate semantic correctness using the supplied benchmark context.",
        "question": native["question"], "memory_context": native["haystack_sessions"],
        "native_reference_answer": native["answer"], "existing_model_answer": row["parsed_answer"],
        "native_answerability": native_lme_answerability(native),
        "output_schema": {"decision": ["correct", "incorrect", "abstention_correct", "abstention_incorrect", "adjudication_required"], "confidence": ["high", "medium", "low"], "short_reason": "string", "answerability_basis": ["answerable", "insufficient_information", "ambiguous"]},
        "instructions": "Judge semantic correctness, not lexical identity or style. A justified abstention is correct only when the supplied context is insufficient.",
    }, ensure_ascii=False, sort_keys=True)


def memora_prompt(row: dict[str, Any], native: dict[str, Any]) -> str:
    criteria = native["evaluation"]["evaluation_questions"]
    return json.dumps({
        "task": "Evaluate every listed native criterion against exactly one existing generated answer.",
        "question": native["question"], "existing_model_answer": row["parsed_answer"],
        "criteria": [{"criterion_id": item["evaluation_question_id"], "criterion": item["evaluation_question"], "expected_answer": item["expected_answer"], "evaluation_type": item["evaluation_type"]} for item in criteria],
        "output_schema": {"decisions": [{"criterion_id": "must match one input ID exactly once", "decision": ["pass", "fail", "adjudication_required"], "confidence": ["high", "medium", "low"], "short_reason": "string"}]},
        "instructions": "Judge each criterion independently. Return every criterion exactly once; missing, duplicate, or unknown IDs invalidate the whole batch. Do not judge writing quality.",
    }, ensure_ascii=False, sort_keys=True)


def stable_stratified_sample(records: list[dict[str, Any]], n: int, seed: int, strata: tuple[str, ...]) -> list[dict[str, Any]]:
    if len(records) < n:
        raise ValueError(f"cannot sample {n} from {len(records)} records")
    # Greedily minimize marginal and joint stratum counts.  A stable random
    # tie-break keeps the sample reproducible without letting lexicographic
    # provider order dominate small samples.
    rng = random.Random(seed)
    remaining = list(records)
    rng.shuffle(remaining)
    marginal: Counter[tuple[str, str]] = Counter()
    joint: Counter[tuple[str, ...]] = Counter()
    chosen: list[dict[str, Any]] = []
    while len(chosen) < n:
        def score(record: dict[str, Any]) -> tuple[int, int, str]:
            values = tuple(str(record.get(key, "unknown")) for key in strata)
            return (sum(marginal[(key, value)] for key, value in zip(strata, values)), joint[values], sha256_value([seed, record["case_id"]]))
        selected = min(remaining, key=score)
        remaining.remove(selected)
        values = tuple(str(selected.get(key, "unknown")) for key in strata)
        for key, value in zip(strata, values):
            marginal[(key, value)] += 1
        joint[values] += 1
        chosen.append(selected)
    return sorted(chosen, key=lambda item: item["case_id"])


def memora_subtype(question_id: str) -> str:
    for subtype in ("activity_todos", "content_email_writeup", "content_meeting_notes", "content_project_proposal"):
        if question_id.startswith(subtype + "_"):
            return subtype
    if question_id.startswith("pref_"):
        return "preference_recommendation"
    return question_id.split("_", 1)[0] if "_" in question_id else question_id


def validate_memora_batches(batches: list[dict[str, Any]], expected_assignments: set[str]) -> None:
    observed: set[str] = set()
    for batch in batches:
        ids = batch["criterion_assignment_ids"]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Memora batch has missing or duplicate criterion IDs")
        if any(not item.startswith(batch["case_id"] + "::") for item in ids):
            raise ValueError("Memora batch crosses generated-answer identity")
        overlap = observed.intersection(ids)
        if overlap:
            raise ValueError(f"duplicate criterion assignment across batches: {sorted(overlap)[:1]}")
        observed.update(ids)
    if observed != expected_assignments:
        raise ValueError(f"Memora criterion assignments incomplete: expected {len(expected_assignments)}, observed {len(observed)}")


def build_design(answer_paths: list[Path], lme_source_path: Path, memora_root: Path) -> dict[str, Any]:
    lme_rows, memora_rows, answer_provenance = load_existing_answers(answer_paths)
    lme_source = load_lme_source(lme_source_path)
    memora_source = load_memora_source(memora_root)

    lme_cases = []
    for row in lme_rows:
        qid = str(row["base_id"]).split("::")[-1]
        native = lme_source.get(qid)
        if native is None:
            raise ValueError(f"missing LongMemEval native source for {qid}")
        route = route_longmemeval(row, native)
        if route["route"] not in LME_ROUTES:
            raise AssertionError(route)
        case = {
            "case_id": public_case_id(row), "provider": row["provider"], "condition": row["condition_type"],
            "question_type": native["question_type"], "route": route["route"], "route_reason": route["reason"],
            "condition_id_hash": sha256_value(row["condition_id"]), "question_id_hash": sha256_value(qid),
        }
        if route["route"] == "semantic_judge_required":
            prompt = lme_prompt(row, native)
            case.update({"prompt_chars": len(prompt), "prompt_hash": sha256_value(prompt)})
        lme_cases.append(case)

    memora_batches = []
    memora_criterion_counts = Counter()
    expected_assignments: set[str] = set()
    for row in memora_rows:
        key = memora_key(str(row["base_id"]))
        native = memora_source.get(key)
        if native is None:
            raise ValueError(f"missing Memora native source for {key}")
        criteria = native.get("evaluation", {}).get("evaluation_questions", [])
        if not criteria:
            raise ValueError(f"Memora question has no native criteria: {key}")
        case_id = public_case_id(row)
        route = "semantic_judge_required" if is_valid_answer_row(row) else "invalid_unscorable"
        assignment_ids = []
        for criterion in criteria:
            native_id = str(criterion["evaluation_question_id"])
            assignment = f"{case_id}::{native_id}"
            assignment_ids.append(assignment)
            expected_assignments.add(assignment)
            memora_criterion_counts[(str(criterion["evaluation_type"]), route)] += 1
        batch = {
            "case_id": case_id, "provider": row["provider"], "condition": row["condition_type"],
            "subtype": memora_subtype(key[2]), "native_task_group": native["native_task_group"],
            "route": route, "criterion_assignment_ids": assignment_ids,
            "criterion_n": len(criteria), "memory_presence_n": sum(c["evaluation_type"] == "memory_presence" for c in criteria),
            "forgetting_absence_n": sum(c["evaluation_type"] == "forgetting_absence" for c in criteria),
            "condition_id_hash": sha256_value(row["condition_id"]), "native_question_key_hash": sha256_value(key),
        }
        if route == "semantic_judge_required":
            prompt = memora_prompt(row, native)
            batch.update({"prompt_chars": len(prompt), "prompt_hash": sha256_value(prompt)})
        memora_batches.append(batch)
    validate_memora_batches(memora_batches, expected_assignments)

    lme_route_counts = Counter(item["route"] for item in lme_cases)
    memora_route_counts = Counter()
    for (criterion_type, route), count in memora_criterion_counts.items():
        memora_route_counts[route] += count
    lme_semantic = [item for item in lme_cases if item["route"] == "semantic_judge_required"]
    memora_semantic = [item for item in memora_batches if item["route"] == "semantic_judge_required"]
    primary_n = len(lme_semantic) + len(memora_semantic)

    # Protocol 4 freezes a blinded dual-judge calibration of 200 cases per
    # source plus all native-insufficient LME cases still requiring semantics.
    calibration_lme = stable_stratified_sample(lme_semantic, min(200, len(lme_semantic)), SEED + 1, ("provider", "condition", "question_type"))
    calibration_memora = stable_stratified_sample(memora_semantic, min(200, len(memora_semantic)), SEED + 2, ("provider", "condition", "native_task_group"))
    secondary_ids = {item["case_id"] for item in calibration_lme + calibration_memora}
    # Native insufficient-information questions are a predeclared high-risk
    # stratum even when the existing answer itself is not an abstention.
    for row, case in zip(lme_rows, lme_cases):
        qid = str(row["base_id"]).split("::")[-1]
        if case["route"] == "semantic_judge_required" and native_lme_answerability(lme_source[qid]) == "insufficient_information":
            secondary_ids.add(case["case_id"])
    secondary_floor = len(secondary_ids)

    dry_lme = stable_stratified_sample([item for item in lme_cases if item["route"] != "invalid_unscorable"], 20, SEED, ("provider", "condition", "question_type", "route"))
    dry_memora = stable_stratified_sample(memora_semantic, 20, SEED, ("provider", "condition", "native_task_group", "route"))
    dry_rows = []
    for item in dry_lme:
        dry_rows.append({k: item[k] for k in ("case_id", "provider", "condition", "question_type", "route", "condition_id_hash")})
        dry_rows[-1].update({"source_dataset": "longmemeval", "selection_seed": SEED, "contains_private_text": False})
    for item in dry_memora:
        dry_rows.append({k: item[k] for k in ("case_id", "provider", "condition", "subtype", "native_task_group", "route", "criterion_n", "condition_id_hash")})
        dry_rows[-1].update({"source_dataset": "memora", "selection_seed": SEED, "contains_private_text": False})

    criterion_total = sum(memora_route_counts.values())
    memory_presence = sum(count for (kind, _), count in memora_criterion_counts.items() if kind == "memory_presence")
    forgetting_absence = sum(count for (kind, _), count in memora_criterion_counts.items() if kind == "forgetting_absence")
    lme_sizes = [item["prompt_chars"] for item in lme_semantic]
    memora_sizes = [item["prompt_chars"] for item in memora_semantic]
    secondary_cases = [item for item in lme_semantic + memora_semantic if item["case_id"] in secondary_ids]
    secondary_sizes = [item["prompt_chars"] for item in secondary_cases]
    output_token_estimate = len(lme_semantic) * 160 + sum(40 + 36 * item["criterion_n"] for item in memora_semantic)
    cost = {
        "pricing": {"available": False, "reason": "no repository-local pricing table is authoritative for an unspecified final judge model; dollar cost intentionally omitted"},
        "prompt_sizes": {"longmemeval": size_summary(lme_sizes), "memora_answer_batches": size_summary(memora_sizes), "combined": size_summary(lme_sizes + memora_sizes)},
        "request_volumes": {
            "primary_longmemeval": {"calls": len(lme_semantic), "estimated_input_tokens": size_summary(lme_sizes)["estimated_input_tokens_div4"]},
            "primary_memora": {"calls": len(memora_semantic), "estimated_input_tokens": size_summary(memora_sizes)["estimated_input_tokens_div4"]},
            "primary_total": {"calls": primary_n, "estimated_input_tokens": size_summary(lme_sizes + memora_sizes)["estimated_input_tokens_div4"]},
            "secondary_frozen_floor": {"calls": secondary_floor, "estimated_input_tokens": size_summary(secondary_sizes)["estimated_input_tokens_div4"]},
            "third_judge": {"calls_lower_bound": 0, "calls_upper_bound_before_results": secondary_floor, "point_estimate": None, "reason": "third calls depend on observed disagreement; no judge model is frozen, so a point rate is not assumed"},
        },
        "estimated_output_tokens": {"method": "planning allowance: 160 per LongMemEval result; 40 batch overhead plus 36 per Memora criterion", "total": output_token_estimate},
        "protocols": {
            "protocol_1_single_judge": {"primary_calls": primary_n, "secondary_calls": 0, "third_calls": 0, "total_known_calls": primary_n, "strength": "lowest", "failure_mode": "single-judge systematic bias is not independently measured"},
            "protocol_2_single_plus_flagged": {"primary_calls": primary_n, "secondary_calls": "dynamic: low-confidence, abstention, or adjudication-required outputs", "third_calls": 0, "total_known_calls": primary_n, "strength": "moderate", "failure_mode": "confident but systematic judge errors may evade review"},
            "protocol_3_two_all_plus_third": {"primary_calls": primary_n, "secondary_calls": primary_n, "third_calls": "dynamic on disagreement", "base_calls": primary_n * 2, "third_call_upper_bound": primary_n, "strength": "strongest automated design", "failure_mode": "highest cost; correlated judge bias remains possible"},
            "protocol_4_primary_calibration_targeted": {"primary_calls": primary_n, "secondary_calls_floor": secondary_floor, "secondary_calls_dynamic_additions": "all primary low-confidence or adjudication-required cases not already selected", "third_calls_initial": 0, "third_calls_dynamic": "only binary/criterion disagreement after two valid judgments", "base_known_calls": primary_n + secondary_floor, "third_call_upper_bound_before_results": secondary_floor, "strength": "recommended cost/validity balance", "failure_mode": "reliability outside calibrated strata depends on the precommitted audit design"},
        },
        "recommended_protocol": "protocol_4_primary_calibration_targeted",
    }

    lme_plan = {
        "schema_version": DESIGN_VERSION, "evaluation_unit": "one existing generated answer", "total_rows": len(lme_cases),
        "routes": dict(sorted(lme_route_counts.items())),
        "by_provider": nested_counts(lme_cases, "provider", "route"), "by_condition": nested_counts(lme_cases, "condition", "route"),
        "by_question_type": nested_counts(lme_cases, "question_type", "route"),
        "deterministic_rules": ["normalized exact match", "benchmark-encoded alternatives separated by explicit 'or'", "whole-answer numeric/unit normalization", "canonical yes/no equivalence"],
        "abstention_rule": "A canonical abstention is deterministic only when native metadata establishes answerable versus insufficient-information; otherwise it is routed to semantic judging.",
        "semantic_labels": ["correct", "incorrect", "abstention_correct", "abstention_incorrect", "adjudication_required"],
        "aggregation": "answer-level accuracy with abstention_correct projected correct and abstention_incorrect projected incorrect; raw abstention labels retained",
        "old_govmem_scorer_used_as_final_metric": False,
    }
    memora_plan = {
        "schema_version": DESIGN_VERSION, "generated_answer_n": len(memora_batches), "native_criterion_n": criterion_total,
        "memory_presence_criterion_n": memory_presence, "forgetting_absence_criterion_n": forgetting_absence,
        "routes": dict(sorted(memora_route_counts.items())), "semantic_answer_batch_n": len(memora_semantic),
        "batching": {"unit": "one generated answer", "cross_answer_batching": False, "criterion_ids_complete_and_unique": True, "whole_batch_invalid_on_missing_duplicate_or_unknown_criterion_id": True},
        "criterion_labels": ["pass", "fail", "adjudication_required"],
        "native_aggregation": {"MPA": "passed memory_presence / total memory_presence", "FAA": "passed forgetting_absence / total forgetting_absence", "lambda": "N_forget / (N_presence + N_forget)", "per_question_FAMA": "max(0, MPA - lambda * (1 - FAA))", "reported_bucket": "mean per-question FAMA * 100"},
        "deterministic_routing_note": "No successful released natural-language criterion has an immutable structural answer field that permits semantic presence/absence judgment without interpretation; all such criteria route to a judge. Failed/empty answer rows are invalid, not automatic criterion failures.",
        "old_govmem_answer_level_boolean_used_as_final_target": False,
        "by_provider_answers": nested_counts(memora_batches, "provider", "route"), "by_condition_answers": nested_counts(memora_batches, "condition", "route"),
        "by_subtype_answers": nested_counts(memora_batches, "subtype", "route"),
    }
    routing = {
        "schema_version": DESIGN_VERSION,
        "longmemeval": {"rows": len(lme_cases), "routes": dict(sorted(lme_route_counts.items()))},
        "memora": {"answers": len(memora_batches), "criteria": criterion_total, "criterion_routes": dict(sorted(memora_route_counts.items())), "batched_semantic_requests": len(memora_semantic)},
        "total_primary_semantic_requests": primary_n, "estimated_secondary_requests_floor": secondary_floor,
        "estimated_third_judge_requests": {"lower_bound": 0, "upper_bound_before_results": secondary_floor, "point_estimate": None}, "third_judge_requests_are_result_dependent": True,
    }
    manifest = {
        "schema_version": DESIGN_VERSION, "seed": SEED, "status": "protocol_frozen_offline_no_judgments_executed",
        "answer_generation_rerun_required": False, "full_regeneration_required": False, "paid_calls_performed": 0,
        "source_hashes": {"longmemeval_oracle_sha256": sha256_file(lme_source_path), "memora_native_tree_fingerprint": sha256_value(sorted((str(key), sha256_value(value.get("evaluation", {}))) for key, value in memora_source.items()))},
        **answer_provenance,
        "population": {"longmemeval_rows": len(lme_cases), "memora_answers": len(memora_batches), "memora_native_criteria": criterion_total},
        "dry_run": {"seed": SEED, "longmemeval_n": len(dry_lme), "memora_answer_n": len(dry_memora), "manifest_hash": sha256_value(dry_rows)},
        "public_output_policy": "No benchmark question/reference/context or model-answer text is written to this directory.",
    }
    return {"lme_plan": lme_plan, "memora_plan": memora_plan, "routing": routing, "cost": cost, "manifest": manifest, "dry_rows": dry_rows}


def nested_counts(records: list[dict[str, Any]], outer: str, inner: str) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counts[str(record.get(outer, "unknown"))][str(record.get(inner, "unknown"))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(counts.items())}


def protocol_markdown(design: dict[str, Any]) -> str:
    lme, mem, routing, cost = design["lme_plan"], design["memora_plan"], design["routing"], design["cost"]
    return f"""# Final native evaluation protocol (frozen offline design)

Design version: `{DESIGN_VERSION}`. Sampling seed: `{SEED}`. This protocol evaluates existing answers only; it does not rerun answer generation and made no judge/API calls while being created.

## LongMemEval

The unit is one generated answer ({lme['total_rows']:,} rows). Routing is hierarchical and exclusive: normalized exact/native alternatives and conservative scalar/yes-no equivalence first; canonical abstentions second, but only when the native source establishes answerability; semantic judging next; invalid or missing artifacts last. Substring and fuzzy semantic matching are prohibited. The frozen route counts are `{json.dumps(lme['routes'], sort_keys=True)}`.

Semantic outputs are `correct`, `incorrect`, `abstention_correct`, `abstention_incorrect`, or `adjudication_required`. Answer-level accuracy projects the two `*_correct` states to their binary correctness while retaining raw abstention labels. `adjudication_required` is excluded from accuracy until resolved.

## Memora

The evaluation unit is the released native atomic criterion, not GovMem's old answer-level Boolean. The {mem['generated_answer_n']:,} existing answers expand to {mem['native_criterion_n']:,} criterion assignments: {mem['memory_presence_criterion_n']:,} `memory_presence` and {mem['forgetting_absence_criterion_n']:,} `forgetting_absence`. Natural-language semantic presence and absence are not replaced by keyword containment. Frozen criterion routes are `{json.dumps(mem['routes'], sort_keys=True)}`.

Criteria for one answer may share one request, producing exactly one `pass`, `fail`, or `adjudication_required` record per criterion ID. Batches never cross answers; missing, duplicate, or unknown IDs invalidate a batch. Native aggregation is preserved:

`FAMA = max(0, MPA - lambda * (1 - FAA))`, where `MPA` and `FAA` are criterion accuracies and `lambda = N_forget / (N_presence + N_forget)`. Reported cells are means of per-question FAMA, multiplied by 100.

## Frozen semantic-judge protocol

The selected design is Protocol 4: one strong primary judge for all {routing['total_primary_semantic_requests']:,} routed requests, plus a precommitted blinded secondary calibration/risk stratum of at least {routing['estimated_secondary_requests_floor']:,}. Any primary `low` confidence or `adjudication_required` output not already selected also receives a second judgment. A third independent judge is invoked only for disagreement after two usable judgments; unresolved cases go to blinded human review. Judge identity is not mandatory, but judges must be strong general-purpose models from different provider families when two or three are used.

The four evaluated designs are recorded in `estimated_costs.json`. Protocol 1 is cheapest but leaves systematic single-judge bias unmeasured. Protocol 2 catches declared uncertainty but misses confident bias. Protocol 3 is strongest and most expensive. Protocol 4 gives a frozen reliability estimate and targets high-risk cases without paying for universal duplication.

Judge prompts hide generating provider, model, condition, GovMem score, other judge outputs, and desired agreement. LongMemEval uses the native question, full benchmark context, native answer, answer, and abstention convention. Memora uses one answer, its question, and its complete native criterion set. Strict structured parsing fails closed.

## Cost and execution status

Prompt-size/token estimates are based on the actual frozen local inputs but expose no text. Token counts use a documented characters/4 planning approximation because no final judge model is mandated. No dollar estimate is given because no authoritative local pricing table applies. Execution remains disabled; paid calls performed: 0.

## Reproducibility and scope

`dry_run_case_manifest.jsonl` freezes 20 LongMemEval and 20 Memora answer identities using seed {SEED}. It contains IDs/hashes and strata only. OAKS is outside this redesign because its 6,720 rows already have native option parity. The old GovMem source-aware scorer is not an evaluation target in this protocol.
"""


def write_outputs(output: Path, design: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "evaluation_design_manifest.json", design["manifest"])
    write_json(output / "longmemeval_evaluation_plan.json", design["lme_plan"])
    write_json(output / "memora_evaluation_plan.json", design["memora_plan"])
    write_json(output / "case_routing_summary.json", design["routing"])
    write_json(output / "estimated_costs.json", design["cost"])
    (output / "dry_run_case_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in design["dry_rows"]), encoding="utf-8")
    (output / "EVALUATION_PROTOCOL.md").write_text(protocol_markdown(design), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--longmemeval-source", type=Path, default=Path("$LONGMEMEVAL_SOURCE/longmemeval-cleaned/longmemeval_oracle.json"))
    parser.add_argument("--memora-source", type=Path, default=Path("$MEMORA_SOURCE"))
    parser.add_argument("--output", type=Path, default=Path("results/final_native_evaluation_design"))
    args = parser.parse_args(argv)
    paths = load_answer_paths(args.repo_root)
    design = build_design(paths, args.longmemeval_source, args.memora_source)
    write_outputs(args.output, design)
    routing = design["routing"]
    lme = routing["longmemeval"]; mem = routing["memora"]
    print(f"LONGMEMEVAL_TOTAL_ROWS={lme['rows']}")
    print(f"LONGMEMEVAL_DETERMINISTIC_ROWS={lme['routes'].get('deterministic_native', 0) + lme['routes'].get('deterministic_abstention', 0)}")
    print(f"LONGMEMEVAL_SEMANTIC_JUDGE_ROWS={lme['routes'].get('semantic_judge_required', 0)}")
    print(f"MEMORA_TOTAL_ANSWERS={mem['answers']}")
    print(f"MEMORA_TOTAL_CRITERIA={mem['criteria']}")
    print(f"MEMORA_DETERMINISTIC_CRITERIA={mem['criterion_routes'].get('deterministic_native', 0) + mem['criterion_routes'].get('deterministic_absence_check', 0)}")
    print(f"MEMORA_SEMANTIC_CRITERIA={mem['criterion_routes'].get('semantic_judge_required', 0)}")
    print(f"MEMORA_BATCHED_SEMANTIC_REQUESTS={mem['batched_semantic_requests']}")
    print(f"TOTAL_PRIMARY_SEMANTIC_REQUESTS={routing['total_primary_semantic_requests']}")
    print(f"ESTIMATED_SECONDARY_REQUESTS={routing['estimated_secondary_requests_floor']}")
    third = routing["estimated_third_judge_requests"]
    print(f"ESTIMATED_THIRD_JUDGE_REQUESTS={third['lower_bound']}_to_{third['upper_bound_before_results']}")
    print("FULL_REGENERATION_REQUIRED=no")
    print("ANSWER_GENERATION_RERUN_REQUIRED=no")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
