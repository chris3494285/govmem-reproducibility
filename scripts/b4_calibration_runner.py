#!/usr/bin/env python3
"""B4 calibration planner and explicitly gated paid-execution entry point.

Dry-run is the default. Provider clients are imported lazily only after both
paid-call flags, explicit input limits, prompt checks, and ledger checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_b4_private_payloads import read_jsonl, validate_payload_records

SEED = 20260929
LME_N = 20
MEMORA_N = 40
LME_DECISIONS = {"correct", "incorrect", "abstention_correct", "abstention_incorrect", "adjudication_required"}
MEMORA_DECISIONS = {"pass", "fail", "adjudication_required"}
CONFIDENCES = {"high", "medium", "low"}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def memora_semantic_fingerprint(record: dict[str, Any]) -> str:
    """Identity of judge-visible criterion semantics, independent of ID namespace."""
    return digest({
        "native_context": record.get("native_context"),
        "criterion": record.get("criterion"),
        "expected_behavior": record.get("expected_behavior"),
        "model_answer": record.get("model_answer"),
    })


def _distribution(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0, "median": 0, "mean": 0, "p90": 0, "max": 0, "total": 0}
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": round(statistics.fmean(ordered), 3),
        "p90": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
        "max": ordered[-1],
        "total": sum(ordered),
    }


def summarize_prompt_sizes(classified: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for provider in sorted({str(row["judge_provider"]) for row in classified}):
        rows = [row for row in classified if row["judge_provider"] == provider]
        result[provider] = {
            "request_n": len(rows),
            "characters": _distribution([len(row["prompt"]) for row in rows]),
            "utf8_bytes": _distribution([len(row["prompt"].encode("utf-8")) for row in rows]),
            "estimated_tokens": _distribution([int(row["estimated_tokens"]) for row in rows]),
            "estimation_methods": sorted({str(row["method"]) for row in rows}),
        }
    return result


def _stable_seed(seed: int, label: str) -> int:
    return int(hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()[:16], 16)


def sample_lme(records: list[dict[str, Any]], n: int = LME_N, seed: int = SEED) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda x: str(x["blinded_request_id"]))
    if n > len(ordered):
        raise ValueError(f"LongMemEval sample size {n} exceeds population {len(ordered)}")
    return sorted(random.Random(seed).sample(ordered, n), key=lambda x: str(x["blinded_request_id"]))


def sample_memora(records: list[dict[str, Any]], n: int = MEMORA_N, seed: int = SEED) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda x: str(x["blinded_request_id"]))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ordered:
        groups[str(record.get("subtype", "unknown"))].append(record)
    if n > len(ordered):
        raise ValueError(f"Memora sample size {n} exceeds population {len(ordered)}")
    if len(groups) < 2:
        raise ValueError("Memora calibration requires both semantic subtypes when available")
    # Largest-remainder allocation with a minimum of three per subtype.
    labels = sorted(groups)
    if n < 3 * len(labels):
        raise ValueError("Memora calibration size is too small for subtype minimums")
    total = len(ordered)
    allocation = {label: 3 for label in labels}
    remaining = n - 3 * len(labels)
    shares = {label: (len(groups[label]) / total) * remaining for label in labels}
    for label in sorted(labels, key=lambda x: (-shares[x] % 1, x)):
        add = min(int(shares[label]), remaining)
        allocation[label] += add
        remaining -= add
    for label in labels:
        if remaining <= 0:
            break
        allocation[label] += 1
        remaining -= 1
    selected: list[dict[str, Any]] = []
    for label in labels:
        selected.extend(random.Random(_stable_seed(seed, label)).sample(groups[label], allocation[label]))
    return sorted(selected, key=lambda x: str(x["blinded_request_id"]))


def build_lme_prompt(record: dict[str, Any]) -> str:
    return json.dumps({
        "task": "Evaluate semantic correctness using the supplied benchmark context.",
        "question": record["question"],
        "memory_context": record["native_context"],
        "native_reference_answer": record["reference_answer"],
        "existing_model_answer": record["model_answer"],
        "abstention_convention": record.get("native_abstention_convention", []),
        "output_schema": {
            "decision": sorted(LME_DECISIONS),
            "confidence": sorted(CONFIDENCES),
            "short_reason": "brief explanation",
            "answerability_basis": ["answerable", "insufficient_information", "ambiguous"],
        },
        "instructions": "Judge meaning, not lexical identity or style. Do not infer information absent from the supplied context.",
    }, ensure_ascii=False, sort_keys=True)


def build_memora_prompt(record: dict[str, Any]) -> str:
    return json.dumps({
        "task": "Evaluate exactly one native atomic criterion.",
        "native_context": record["native_context"],
        "atomic_criterion": record["criterion"],
        "expected_behavior": record["expected_behavior"],
        "existing_model_answer": record["model_answer"],
        "output_schema": {
            "decision": sorted(MEMORA_DECISIONS),
            "confidence": sorted(CONFIDENCES),
            "short_reason": "brief criterion-level explanation",
        },
        "instructions": "Judge only the stated criterion. Ignore general writing quality, verbosity, and harmless paraphrase.",
    }, ensure_ascii=False, sort_keys=True)


def parse_judge_output(raw: str | dict[str, Any], task: str) -> dict[str, Any]:
    allowed = (LME_DECISIONS if task == "longmemeval" else MEMORA_DECISIONS)
    required = {"decision", "confidence", "short_reason"}
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON: {exc}") from exc
    else:
        value = raw
    if not isinstance(value, dict):
        raise ValueError("judge output must be a JSON object")
    expected = required | ({"answerability_basis"} if task == "longmemeval" else set())
    if set(value) != expected:
        raise ValueError(f"judge output keys must be exactly {sorted(expected)}")
    if value["decision"] not in allowed:
        raise ValueError("invalid decision")
    if value["confidence"] not in CONFIDENCES:
        raise ValueError("invalid or missing confidence")
    if not isinstance(value["short_reason"], str) or not value["short_reason"].strip():
        raise ValueError("missing short_reason")
    if task == "longmemeval" and value["answerability_basis"] not in {"answerable", "insufficient_information", "ambiguous"}:
        raise ValueError("invalid answerability_basis")
    return value


def analyze_results(records: list[dict[str, Any]], task: str) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        slot = record.get("judge_slot")
        if slot not in {"A", "B"}:
            raise ValueError("judge_slot must be A or B")
        grouped[str(record["blinded_request_id"])][slot] = record
    decisions = []
    for request_id, slots in grouped.items():
        a, b = slots.get("A"), slots.get("B")
        parsed = {}
        parse_failure = False
        for slot, record in (("A", a), ("B", b)):
            if not record:
                parse_failure = True
                continue
            try:
                parsed[slot] = parse_judge_output(record["raw_output"], task)
            except (KeyError, ValueError):
                parse_failure = True
        if "A" in parsed and "B" in parsed:
            exact = parsed["A"]["decision"] == parsed["B"]["decision"]
            adjudication = any(x["decision"] == "adjudication_required" for x in parsed.values())
            low_confidence = any(x["confidence"] == "low" for x in parsed.values())
            disagreement = not exact
        else:
            exact = disagreement = False
            adjudication = low_confidence = False
        escalation = parse_failure or disagreement or adjudication or low_confidence
        decisions.append({"blinded_request_id": request_id, "exact_agreement": exact, "disagreement": disagreement, "adjudication_required": adjudication, "low_confidence": low_confidence, "parse_failure": parse_failure, "escalation": escalation, "parsed": parsed})
    summary: dict[str, Any] = {
        "task": task,
        "total_calibration_cases": len(grouped),
        "exact_decision_agreement_n": sum(x["exact_agreement"] for x in decisions),
        "disagreement_n": sum(x["disagreement"] for x in decisions),
        "adjudication_required_n": sum(x["adjudication_required"] for x in decisions),
        "low_confidence_n": sum(x["low_confidence"] for x in decisions),
        "parse_failure_n": sum(x["parse_failure"] for x in decisions),
        "escalation_n": sum(x["escalation"] for x in decisions),
    }
    if task == "longmemeval":
        summary["abstention_decision_counts"] = Counter(
            parsed["decision"] for x in decisions for parsed in x["parsed"].values() if parsed["decision"].startswith("abstention_")
        )
    else:
        summary["pass_fail_exact_agreement_n"] = sum(
            x["exact_agreement"] and all(p["decision"] in {"pass", "fail"} for p in x["parsed"].values()) for x in decisions
        )
        summary["by_subtype"] = {}
        for record in records:
            summary["by_subtype"].setdefault(record.get("subtype", "unknown"), 0)
            summary["by_subtype"][record.get("subtype", "unknown")] += 1
    return summary


def _provider_config(args: argparse.Namespace) -> dict[str, str]:
    a = (args.judge_a_provider, args.judge_a_model)
    b = (args.judge_b_provider, args.judge_b_model)
    if a == b:
        raise SystemExit("Judge A and Judge B cannot use the same provider+model pair")
    if args.judge_a_provider == args.judge_b_provider:
        warnings.warn("Judge A and Judge B use the same provider family; heterogeneous providers are preferred")
    return {"judge_a_provider": a[0], "judge_a_model": a[1], "judge_b_provider": b[0], "judge_b_model": b[1]}


def _verify_frozen_selection(path: Path, lme: list[dict[str, Any]], memora: list[dict[str, Any]], seed: int, config: dict[str, str]) -> None:
    """Refuse to overwrite an existing calibration manifest with a new cohort."""
    if not path.exists():
        return
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("seed") != seed:
        raise SystemExit("Frozen calibration seed mismatch")
    if frozen.get("judge_architecture") not in (None, config):
        raise SystemExit("Frozen calibration judge provider/model mismatch")
    for key, records in (("longmemeval", lme), ("memora", memora)):
        expected_ids = [x["blinded_request_id"] for x in records]
        expected_hashes = [digest(x) for x in records]
        section = frozen.get(key, {})
        if section.get("ids") != expected_ids or section.get("request_hashes") != expected_hashes:
            raise SystemExit(f"Frozen calibration selection/hash mismatch for {key}")


def _verify_frozen_prompt_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    if not path.exists():
        return
    existing = read_jsonl(path)
    if existing != rows:
        raise SystemExit("Frozen calibration prompt plan mismatch")


def _load_frozen_records(
    manifest_path: Path,
    lme_population: list[dict[str, Any]],
    memora_population: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected: list[list[dict[str, Any]]] = []
    for key, population in (("longmemeval", lme_population), ("memora", memora_population)):
        section = frozen.get(key, {})
        ids = section.get("ids")
        hashes = section.get("request_hashes")
        if not isinstance(ids, list) or not isinstance(hashes, list) or len(ids) != len(hashes):
            raise SystemExit(f"Invalid frozen calibration manifest section: {key}")
        by_id = {str(record["blinded_request_id"]): record for record in population}
        try:
            records = [by_id[str(identifier)] for identifier in ids]
        except KeyError as exc:
            raise SystemExit(f"Frozen {key} ID is absent from the private population: {exc.args[0]}") from exc
        if [digest(record) for record in records] != hashes:
            raise SystemExit(f"Frozen {key} request hashes do not match private payloads")
        if len(records) != section.get("sample_n"):
            raise SystemExit(f"Frozen {key} sample count mismatch")
        selected.append(records)
    return selected[0], selected[1]


def _write_context_audit(
    path: Path,
    rows: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    limits: dict[str, int | None],
) -> None:
    lines = [
        "# LongMemEval Calibration Context Audit",
        "",
        "This private, ID-only report contains no benchmark text. Token gates use a local model tokenizer when available and otherwise a conservative UTF-8 byte upper bound. No context is silently truncated.",
        "",
        f"- calibration cases: {len(rows)}",
        f"- safely reduced cases: {sum(bool(x['reduction']['reduced']) for x in audits)}",
        f"- Judge A configured limit: {limits.get('A') if limits.get('A') is not None else 'not supplied'}",
        f"- Judge B configured limit: {limits.get('B') if limits.get('B') is not None else 'not supplied'}",
        "",
        "| Blinded ID | Original chars | Original approx tokens (bytes/4) | Reduced chars | Reduction method | Evidence preserved | Judge A status | Judge B status |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for audit in audits:
        lines.append(
            "| {id} | {original_chars} | {approx} | {reduced_chars} | {method} | {preserved} | {a} | {b} |".format(
                id=audit["blinded_request_id"],
                original_chars=audit["original_chars"],
                approx=audit["original_approximate_tokens"],
                reduced_chars=audit["reduced_chars"],
                method=audit["reduction"]["method"],
                preserved=str(audit["reduction"]["answerability_evidence_preserved"]).lower(),
                a=audit["limit_status"]["A"],
                b=audit["limit_status"]["B"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _assert_private_execution_output(path: Path) -> None:
    private_root = (Path.cwd() / "results/b4_scoring_validation/private").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise SystemExit(f"Paid execution output must remain under {private_root}") from exc
    ignore = Path(".gitignore")
    if not ignore.exists() or "results/b4_scoring_validation/private/" not in ignore.read_text(encoding="utf-8").splitlines():
        raise SystemExit("Private B4 output path is not protected by .gitignore")


def select_execution_slots(slots: list[dict[str, Any]], case_order: list[str], smoke_cases: int | None) -> list[dict[str, Any]]:
    if smoke_cases is None:
        return slots
    if smoke_cases < 1 or smoke_cases > len(case_order):
        raise ValueError(f"smoke case count must be between 1 and {len(case_order)}")
    selected = set(case_order[:smoke_cases])
    return [slot for slot in slots if slot["blinded_request_id"] in selected]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-a-provider", required=True)
    parser.add_argument("--judge-a-model", required=True)
    parser.add_argument("--judge-b-provider", required=True)
    parser.add_argument("--judge-b-model", required=True)
    parser.add_argument("--lme-input", type=Path, default=Path("results/b4_scoring_validation/private/longmemeval_judge_execution.jsonl"))
    parser.add_argument("--memora-input", type=Path, default=Path("results/b4_scoring_validation/private/memora_judge_execution.jsonl"))
    parser.add_argument("--memora-human-input", type=Path, default=Path("results/b4_scoring_validation/private/memora_human_adjudication.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/b4_scoring_validation/private/calibration"))
    parser.add_argument("--frozen-calibration-manifest", type=Path)
    parser.add_argument("--frozen-prompt-plan", type=Path)
    parser.add_argument("--full-population", action="store_true", help="use every semantic execution case; never sample")
    parser.add_argument("--expected-lme-cases", type=int)
    parser.add_argument("--expected-memora-cases", type=int)
    parser.add_argument("--phase", default="generic_calibration")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--judge-a-max-input-tokens", type=int)
    parser.add_argument("--judge-b-max-input-tokens", type=int)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--reduce-lme-context", action="store_true")
    parser.add_argument("--smoke-cases", type=int, help="execute the first N frozen semantic cases through both judges")
    parser.add_argument("--authorized-retries-only", action="store_true", help="select only latest valid manual-retry authorizations")
    parser.add_argument("--expected-authorized-retry-slots", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid-judge-calls", action="store_true")
    args = parser.parse_args(argv)
    config = _provider_config(args)
    if args.execute != args.confirm_paid_judge_calls:
        raise SystemExit("Paid execution requires BOTH --execute and --confirm-paid-judge-calls")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be positive")
    if args.smoke_cases is not None and args.smoke_cases < 1:
        raise SystemExit("--smoke-cases must be positive")
    if args.authorized_retries_only and args.smoke_cases is not None:
        raise SystemExit("--authorized-retries-only cannot be combined with --smoke-cases")
    lme_population = read_jsonl(args.lme_input)
    memora_population = read_jsonl(args.memora_input)
    if not validate_payload_records(lme_population, "longmemeval")["valid"]:
        raise SystemExit("LongMemEval private payload failed blindness validation")
    if not validate_payload_records(memora_population, "memora")["valid"]:
        raise SystemExit("Memora private payload failed blindness validation")
    if args.full_population and args.frozen_calibration_manifest is not None:
        raise SystemExit("Full-population mode cannot load a calibration sample manifest")
    if args.full_population:
        if args.expected_lme_cases is None or args.expected_memora_cases is None:
            raise SystemExit("Full-population mode requires explicit expected population counts")
        lme = sorted(lme_population, key=lambda row: str(row["blinded_request_id"]))
        memora = sorted(memora_population, key=lambda row: str(row["blinded_request_id"]))
        if len(lme) != args.expected_lme_cases or len(memora) != args.expected_memora_cases:
            raise SystemExit("Full semantic population count differs from explicit expectation")
        if len({row["blinded_request_id"] for row in lme}) != len(lme):
            raise SystemExit("Duplicate LongMemEval semantic case IDs")
        if len({row["blinded_request_id"] for row in memora}) != len(memora):
            raise SystemExit("Duplicate Memora semantic case IDs")
        human = read_jsonl(args.memora_human_input)
        if not validate_payload_records(human, "memora_human")["valid"]:
            raise SystemExit("Memora human-adjudication payload failed blindness validation")
        semantic_fingerprints = {memora_semantic_fingerprint(row) for row in memora}
        human_fingerprints = {memora_semantic_fingerprint(row) for row in human}
        if semantic_fingerprints & human_fingerprints:
            raise SystemExit("Memora semantic judge pool overlaps the human-adjudication pool")
    elif args.frozen_calibration_manifest is not None:
        lme, memora = _load_frozen_records(args.frozen_calibration_manifest, lme_population, memora_population)
    else:
        lme = sample_lme(lme_population, seed=args.seed)
        memora = sample_memora(memora_population, seed=args.seed)
    # The sampler consumes only the semantic execution population.  The
    # separate human-adjudication payload is intentionally never read.
    if args.full_population:
        _assert_private_execution_output(args.output_dir)
        forbidden_outputs = {
            Path("results/b4_scoring_validation/private/calibration").resolve(),
            Path("results/b4_scoring_validation/private/calibration/frontier_judges").resolve(),
        }
        if args.output_dir.resolve() in forbidden_outputs:
            raise SystemExit("Full validation must use a separate output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "full_validation" if args.full_population else "calibration"
    manifest_path = args.output_dir / f"{stem}_manifest.json"
    prompt_plan_path = args.output_dir / f"{stem}_prompt_plan.jsonl"
    execution_plan_path = args.output_dir / f"{stem}_execution_plan.jsonl"
    result_template_path = args.output_dir / f"{stem}_result_template.jsonl"
    _verify_frozen_selection(manifest_path, lme, memora, args.seed, config)
    write_jsonl(args.output_dir / f"longmemeval_{stem}.jsonl", lme)
    write_jsonl(args.output_dir / f"memora_{stem}.jsonl", memora)
    manifest = {
        "schema_version": "b4-full-validation-manifest-v1" if args.full_population else "b4-calibration-manifest-v1", "seed": args.seed, "phase": args.phase,
        "validation_scope": "full_semantic_population" if args.full_population else "calibration_sample",
        "frozen_source_manifest": str(args.frozen_calibration_manifest) if args.frozen_calibration_manifest else None,
        "judge_architecture": config, "execution": "paid" if args.execute else "dry_run_only",
        "longmemeval": {"source_population_n": len(lme_population), "sample_n": len(lme), "ids": [x["blinded_request_id"] for x in lme], "request_hashes": [digest(x) for x in lme]},
        "memora": {"source_population_n": len(memora_population), "sample_n": len(memora), "subtype_counts": dict(Counter(x["subtype"] for x in memora)), "ids": [x["blinded_request_id"] for x in memora], "request_hashes": [digest(x) for x in memora]},
        "semantic_decisions": "none",
        "human_adjudication_excluded_n": len(human) if args.full_population else None,
        "human_semantic_fingerprint_overlap_n": 0 if args.full_population else None,
        "planned_calls": {"longmemeval": len(lme) * 2, "memora": len(memora) * 2, "total": len(lme) * 2 + len(memora) * 2},
        "projected_full_run_base_calls": {"longmemeval": len(lme_population) * 2, "memora": len(memora_population) * 2},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from scripts.b4_calibration_execution import (
        DurableLedger,
        authorized_manual_retry_slots,
        build_slots,
        classify_input_limits,
        estimate_input_tokens,
        execute_slots,
        reduce_lme_native_context,
        resume_plan,
        validate_paid_input_limits,
        validate_provider_credentials,
    )

    prompt_rows = []
    context_audits = []
    for record in lme:
        original_prompt = build_lme_prompt(record)
        reduced_context, reduction = reduce_lme_native_context(record["native_context"])
        execution_record = dict(record)
        if args.reduce_lme_context and reduction["reduced"]:
            execution_record["native_context"] = reduced_context
        execution_prompt = build_lme_prompt(execution_record)
        prompt_rows.extend({"task": "longmemeval", "blinded_request_id": record["blinded_request_id"], "judge_slot": slot, "prompt": execution_prompt} for slot in ("A", "B"))
        statuses = {}
        for slot, model, limit in (
            ("A", args.judge_a_model, args.judge_a_max_input_tokens),
            ("B", args.judge_b_model, args.judge_b_max_input_tokens),
        ):
            measured = estimate_input_tokens(execution_prompt, model)
            statuses[slot] = "unverified" if limit is None else ("ready" if measured["estimated_tokens"] <= limit else "oversized")
        context_audits.append({
            "blinded_request_id": record["blinded_request_id"],
            "original_chars": len(original_prompt),
            "original_approximate_tokens": (len(original_prompt.encode("utf-8")) + 3) // 4,
            "reduced_chars": len(execution_prompt),
            "reduction": reduction,
            "limit_status": statuses,
        })
    for record in memora:
        prompt_rows.extend({"task": "memora", "blinded_request_id": record["blinded_request_id"], "subtype": record["subtype"], "judge_slot": slot, "prompt": build_memora_prompt(record)} for slot in ("A", "B"))
    if args.frozen_prompt_plan is not None:
        _verify_frozen_prompt_plan(args.frozen_prompt_plan, prompt_rows)
    _verify_frozen_prompt_plan(prompt_plan_path, prompt_rows)
    write_jsonl(prompt_plan_path, prompt_rows)
    slots = build_slots(prompt_rows, **config)
    templates = [{key: row[key] for key in ("request_slot_id", "blinded_request_id", "judge_slot", "judge_provider", "judge_model", "semantic_payload_hash", "request_hash", "prompt_format_version", "task")} | {"decision": None, "confidence": None, "short_reason": None, "parse_status": "not_executed", "response_hash": None} for row in slots]
    write_jsonl(result_template_path, templates)
    write_jsonl(execution_plan_path, [{key: value for key, value in row.items() if key != "prompt"} for row in classify_input_limits(slots, {"A": args.judge_a_max_input_tokens, "B": args.judge_b_max_input_tokens})])
    _write_context_audit(args.output_dir / "LONGMEMEVAL_CONTEXT_AUDIT.md", lme, context_audits, {"A": args.judge_a_max_input_tokens, "B": args.judge_b_max_input_tokens})

    ledger = DurableLedger(args.output_dir / "judge_execution_ledger.jsonl")
    case_order = [record["blinded_request_id"] for record in lme] + [record["blinded_request_id"] for record in memora]
    try:
        execution_slots = select_execution_slots(slots, case_order, args.smoke_cases)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.authorized_retries_only:
        execution_slots = authorized_manual_retry_slots(slots, ledger)
        if args.expected_authorized_retry_slots is None:
            raise SystemExit("--authorized-retries-only requires --expected-authorized-retry-slots")
        if len(execution_slots) != args.expected_authorized_retry_slots:
            raise SystemExit(
                f"authorized retry count mismatch: expected {args.expected_authorized_retry_slots}, found {len(execution_slots)}"
            )
    resume = resume_plan(execution_slots, ledger, max_attempts=args.max_attempts)
    classified = classify_input_limits(slots, {"A": args.judge_a_max_input_tokens, "B": args.judge_b_max_input_tokens})
    blockers = [row for row in classified if row["input_limit_status"] != "ready"]
    size_summary = summarize_prompt_sizes(classified)
    largest_by_task = {}
    for task in ("longmemeval", "memora"):
        rows = [row for row in classified if row["task"] == task]
        largest = max(rows, key=lambda row: len(row["prompt"]))
        largest_by_task[task] = {
            "blinded_request_id": largest["blinded_request_id"],
            "characters": len(largest["prompt"]),
            "utf8_bytes": len(largest["prompt"].encode("utf-8")),
            "estimated_tokens": largest["estimated_tokens"],
            "estimation_method": largest["method"],
        }
    preflight = {
        "execution": "paid" if args.execute else "dry_run_only",
        "judge_a": [args.judge_a_provider, args.judge_a_model],
        "judge_b": [args.judge_b_provider, args.judge_b_model],
        "lme_cases": len(lme), "memora_cases": len(memora),
        "stable_request_slots": len(slots),
        "selected_semantic_cases": len({slot["blinded_request_id"] for slot in execution_slots}),
        "selected_request_slots": len(execution_slots),
        "completed_slots": len(resume["completed"]),
        "skipped_slots": len(resume["completed"]),
        "remaining_slots": len(resume["pending"]),
        "estimated_new_paid_calls": len(resume["pending"]),
        "configured_input_limits": {"A": args.judge_a_max_input_tokens, "B": args.judge_b_max_input_tokens},
        "input_limit_blockers": len(blockers),
        "prompt_size_by_provider": size_summary,
        "largest_request_by_task": largest_by_task,
        "ledger": str(ledger.path),
    }
    if args.full_population:
        (args.output_dir / "full_validation_preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(preflight, sort_keys=True))
    if args.execute:
        _assert_private_execution_output(args.output_dir)
        if args.phase == "cheap_judges":
            expected = (args.judge_a_provider, args.judge_a_model, args.judge_b_provider, args.judge_b_model)
            if expected != ("deepseek", "deepseek-v4-pro", "kimi", "kimi-k2.6"):
                raise SystemExit("Cheap-judge execution requires DeepSeek deepseek-v4-pro and Kimi kimi-k2.6")
            if (args.judge_a_max_input_tokens, args.judge_b_max_input_tokens) != (1048576, 262144):
                raise SystemExit("Cheap-judge execution requires verified input limits 1048576 and 262144")
        if args.full_population:
            expected = (args.judge_a_provider, args.judge_a_model, args.judge_b_provider, args.judge_b_model)
            if expected != ("openai", "gpt-5.6-sol", "anthropic", "claude-sonnet-5"):
                raise SystemExit("Full frontier execution requires OpenAI gpt-5.6-sol and Anthropic claude-sonnet-5")
            if (args.judge_a_max_input_tokens, args.judge_b_max_input_tokens) != (1050000, 1000000):
                raise SystemExit("Full frontier execution requires verified input limits 1050000 and 1000000")
        try:
            selected_classified = [row for row in classified if row["request_slot_id"] in {slot["request_slot_id"] for slot in execution_slots}]
            validate_paid_input_limits(selected_classified, {"A": args.judge_a_max_input_tokens, "B": args.judge_b_max_input_tokens})
            validate_provider_credentials(execution_slots)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if resume["blocked"]:
            raise SystemExit(f"Paid execution blocked: {len(resume['blocked'])} ledger slots require manual resolution")
        result = execute_slots(execution_slots, ledger=ledger, max_attempts=args.max_attempts, max_output_tokens=args.max_output_tokens)
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
