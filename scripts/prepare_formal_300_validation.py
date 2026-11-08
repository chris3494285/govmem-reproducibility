#!/usr/bin/env python3
"""Prepare the frozen formal evaluator on the locked 300-block sample."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b4_calibration_execution import estimate_input_tokens, prompt_hash, provider_request_prompt
from scripts.design_final_native_evaluation import (
    SEED, is_valid_answer_row, lme_prompt, load_answer_paths, load_existing_answers,
    load_lme_source, load_memora_source, memora_key, memora_prompt, memora_subtype,
    native_lme_answerability, public_case_id, route_longmemeval, sha256_file,
    sha256_value, stable_stratified_sample,
)
from scripts.prepare_cheap_judge_pilot import read_jsonl

PROTOCOL = "formal-300-validation-v1"
EVALUATOR = "final-native-evaluation-v2"
LOCKED_FINGERPRINT = os.environ.get("GOVMEM_FORMAL_BLOCK_FINGERPRINT", "").strip()
MODELS = {"openai": "gpt-5.6-sol", "anthropic": "claude-sonnet-5", "gemini": "gemini-3-flash-preview"}
LIMITS = {"openai": 1_050_000, "anthropic": 1_000_000, "gemini": 1_048_576}
CONDITIONS = ("clean", "corrupted", "enforced")


def freeze(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError(f"frozen formal artifact drift: {path}")
    path.write_text(payload, encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    freeze(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    freeze(path, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def formal_request_id(observation_id: str, slot: str, provider: str, model: str, request_hash: str, payload_hash: str) -> str:
    return "formal-300-slot::" + sha256_value({
        "protocol": PROTOCOL, "evaluator": EVALUATOR, "observation_id": observation_id,
        "judge_slot": slot, "provider": provider, "model": model,
        "request_hash": request_hash, "payload_hash": payload_hash,
    })


def validate_lock(blocks: list[dict[str, Any]], answers: list[dict[str, Any]]) -> dict[str, Any]:
    fingerprint = sha256_value(blocks)
    if not LOCKED_FINGERPRINT:
        raise ValueError(
            "GOVMEM_FORMAL_BLOCK_FINGERPRINT is required for private formal-300 preparation"
        )
    if fingerprint != LOCKED_FINGERPRINT:
        raise ValueError("selected-block fingerprint drift")
    if len(blocks) != 300 or len({r["pilot_block_id"] for r in blocks}) != 300:
        raise ValueError("locked block count/identity drift")
    if len(answers) != 900 or len({r["observation_id"] for r in answers}) != 900:
        raise ValueError("locked observation count/identity drift")
    grouped = Counter(r["pilot_block_id"] for r in answers)
    if set(grouped.values()) != {3}:
        raise ValueError("locked triplet cardinality drift")
    by_source = Counter(r["source"] for r in blocks)
    by_provider = Counter(r["generating_provider"] for r in blocks)
    if by_source != Counter({"longmemeval": 150, "memora": 150}):
        raise ValueError("locked source split drift")
    if by_provider != Counter({p: 50 for p in ("anthropic", "deepseek", "gemini", "kimi", "openai", "xai")}):
        raise ValueError("locked provider split drift")
    return {"fingerprint": fingerprint, "by_source": dict(sorted(by_source.items())), "by_provider": dict(sorted(by_provider.items()))}


def prepare(repo: Path, blocks_path: Path, answers_path: Path, lme_path: Path, memora_root: Path,
            output: Path, private: Path) -> dict[str, Any]:
    blocks, locked_answers = read_jsonl(blocks_path), read_jsonl(answers_path)
    lock = validate_lock(blocks, locked_answers)
    answer_paths = load_answer_paths(repo)
    lme_rows, memora_rows, provenance = load_existing_answers(answer_paths)
    canonical = {(str(r["provider"]), str(r["condition_id"])): r for r in lme_rows + memora_rows}
    if len(canonical) != len(lme_rows) + len(memora_rows):
        raise ValueError("canonical answer identity duplication")
    lme_source, memora_source = load_lme_source(lme_path), load_memora_source(memora_root)

    # Reconstruct the already-frozen global Memora 200-case risk sample.
    full_memora = []
    for row in memora_rows:
        key = memora_key(str(row["base_id"])); native = memora_source.get(key)
        if native is None or not is_valid_answer_row(row):
            continue
        full_memora.append({
            "case_id": public_case_id(row), "generating_provider": row["provider"],
            "condition": row["condition_type"], "native_task_group": native["native_task_group"],
        })
    frozen_secondary_ids = {
        r["case_id"] for r in stable_stratified_sample(
            full_memora, 200, SEED + 2, ("generating_provider", "condition", "native_task_group")
        )
    }

    payloads, deterministic, plan, manifest = [], [], [], []
    selected_public = []
    counts = Counter()

    def add_slot(case: dict[str, Any], slot: str, provider: str, phase: str, role: str) -> None:
        model = MODELS[provider]
        request_prompt = provider_request_prompt(provider, case["prompt"])
        request_hash = prompt_hash(request_prompt)
        measured = estimate_input_tokens(request_prompt, model)
        if measured["estimated_tokens"] > LIMITS[provider]:
            raise ValueError(f"input-limit blocker: {case['observation_id']}/{provider}")
        rid = formal_request_id(case["observation_id"], slot, provider, model, request_hash, case["payload_hash"])
        row = {
            "formal_validation_request_id": rid, "pilot_block_id": case["pilot_block_id"],
            "observation_id": case["observation_id"], "condition_id": case["condition_id"],
            "answer_sha256": case["answer_sha256"], "source": case["source"],
            "subtype": case.get("subtype"), "judge_slot": slot, "judge_role": role,
            "phase": phase, "provider": provider, "model": model,
            "semantic_prompt_hash": case["semantic_prompt_hash"], "request_hash": request_hash,
            "payload_hash": case["payload_hash"], "protocol_version": PROTOCOL,
            "evaluator_methodology": EVALUATOR,
            "private_payload_locator": f"semantic_payloads.jsonl#{case['observation_id']}",
            "estimated_input_tokens": measured["estimated_tokens"], "input_estimation_method": measured["method"],
            "planned_output_tokens": 160 if case["source"] == "longmemeval" else 40 + 36 * case["criterion_n"],
            "max_attempts": 1, "automatic_retries": False,
            "ledger_path": f"<private-root>/{provider}/judge_execution_ledger.jsonl",
        }
        manifest.append(row)
        plan.append({**row, "initial_state": "planned", "resume_policy": "completed_skip; failed_or_started_only_fail_closed; explicit_recovery_required"})

    for locked in locked_answers:
        key = (str(locked["generating_provider"]), str(locked["condition_id"]))
        row = canonical.get(key)
        if row is None or sha256_value(row.get("parsed_answer")) != locked["answer_sha256"]:
            raise ValueError(f"canonical selected-answer drift: {key}")
        source = str(locked["source"]); case_id = public_case_id(row)
        common = {
            "pilot_block_id": locked["pilot_block_id"], "observation_id": locked["observation_id"],
            "source": source, "condition_id": row["condition_id"], "answer_sha256": locked["answer_sha256"],
            "generating_provider": row["provider"], "condition": row["condition_type"],
            "fault_type": locked["fault_type"], "event_macro": locked["event_macro"], "case_id": case_id,
        }
        if source == "longmemeval":
            counts["lme_total"] += 1
            qid = str(row["base_id"]).split("::")[-1]; native = lme_source.get(qid)
            if native is None: raise ValueError(f"missing LME native row: {qid}")
            route = route_longmemeval(row, native)
            if route["route"] == "invalid_unscorable": raise ValueError("locked LME observation became invalid")
            if route["route"] != "semantic_judge_required":
                counts["lme_deterministic"] += 1
                decision = "correct" if route["route"] == "deterministic_native" else route["native_decision"]
                deterministic.append({**common, "evaluator_type": "native_deterministic", "raw_decision": decision,
                                      "binary_correct": decision in {"correct", "abstention_correct"},
                                      "method": route["reason"], "evaluator_methodology": EVALUATOR})
                selected_public.append({**common, "route": route["route"], "paid_slot_n": 0})
                continue
            counts["lme_semantic"] += 1
            prompt = lme_prompt(row, native); subtype = native.get("question_type"); criterion_n = 0
        else:
            counts["memora_total"] += 1
            native_key = memora_key(str(row["base_id"])); native = memora_source.get(native_key)
            if native is None or not is_valid_answer_row(row): raise ValueError("locked Memora observation became invalid")
            prompt = memora_prompt(row, native); subtype = memora_subtype(native_key[2])
            criterion_n = len(native["evaluation"]["evaluation_questions"])
            counts["memora_primary"] += 1
        semantic_hash = prompt_hash(prompt)
        payload_hash = sha256_value({"protocol": PROTOCOL, "evaluator": EVALUATOR,
                                     "observation_id": locked["observation_id"], "prompt": prompt})
        case = {**common, "subtype": subtype, "criterion_n": criterion_n, "prompt": prompt,
                "semantic_prompt_hash": semantic_hash, "payload_hash": payload_hash,
                "protocol_version": PROTOCOL, "evaluator_methodology": EVALUATOR}
        payloads.append(case)
        if source == "longmemeval":
            add_slot(case, "lme_primary_openai", "openai", "primary", "dual_primary")
            add_slot(case, "lme_primary_anthropic", "anthropic", "primary", "dual_primary")
            paid_n = 2
        else:
            add_slot(case, "memora_primary_gemini", "gemini", "primary", "primary")
            paid_n = 1
            if case_id in frozen_secondary_ids:
                add_slot(case, "memora_secondary_openai", "openai", "secondary", "frozen_risk_secondary")
                counts["memora_secondary"] += 1; paid_n += 1
        selected_public.append({**common, "subtype": subtype, "route": "semantic_judge_required", "paid_slot_n": paid_n})

    if counts["lme_total"] != 450 or counts["memora_total"] != 450:
        raise ValueError(f"source observation drift: {counts}")
    for rows in (payloads, deterministic, plan, manifest, selected_public):
        rows.sort(key=lambda x: (x.get("formal_validation_request_id", x.get("observation_id", "")), x.get("condition_id", "")))
    if len({r["formal_validation_request_id"] for r in plan}) != len(plan):
        raise ValueError("duplicate formal request ID")
    write_jsonl(private / "semantic_payloads.jsonl", payloads)
    write_jsonl(output / "execution_plan.jsonl", plan)
    write_jsonl(output / "request_manifest.jsonl", manifest)
    write_jsonl(output / "deterministic_results.jsonl", deterministic)
    write_jsonl(output / "selected_observations.jsonl", selected_public)

    by_provider = Counter(r["provider"] for r in manifest)
    token_by_provider = Counter(); output_by_provider = Counter()
    phase_provider = Counter()
    for r in manifest:
        token_by_provider[r["provider"]] += r["estimated_input_tokens"]
        output_by_provider[r["provider"]] += r["planned_output_tokens"]
        phase_provider[(r["provider"], r["phase"])] += 1
    lock_doc = {
        "protocol_version": PROTOCOL, "evaluator_methodology": EVALUATOR,
        "source_selected_blocks": str(blocks_path), "source_selected_answers": str(answers_path),
        "block_n": 300, "observation_n": 900,
        "complete_matched_triplets": 300, "by_source": lock["by_source"], "by_generating_provider": lock["by_provider"],
        "resampling_performed": False, "answer_generation_calls": 0,
        "answer_artifacts": provenance["answer_artifacts"],
        "native_source_hashes": {"longmemeval": sha256_file(lme_path), "memora": sha256_value(sorted(map(str, memora_source)))},
    }
    write_json(output / "sampling_lock.json", lock_doc)
    summary = {
        "protocol": PROTOCOL, "evaluator": EVALUATOR, "blocks": 300, "observations": 900,
        "lme_total_observations": counts["lme_total"], "lme_deterministic_observations": counts["lme_deterministic"],
        "lme_semantic_observations": counts["lme_semantic"],
        "lme_openai_primary_calls": phase_provider[("openai", "primary")],
        "lme_anthropic_primary_calls": phase_provider[("anthropic", "primary")],
        "memora_total_observations": counts["memora_total"], "memora_gemini_primary_calls": phase_provider[("gemini", "primary")],
        "memora_frozen_secondary_calls": phase_provider[("openai", "secondary")],
        "memora_dynamic_secondary_baseline": 0, "lme_gemini_escalation_baseline": 0,
        "base_paid_calls": len(plan), "calls_by_provider": dict(sorted(by_provider.items())),
        "estimated_input_tokens_by_provider": dict(sorted(token_by_provider.items())),
        "estimated_output_token_budget_by_provider": dict(sorted(output_by_provider.items())),
        "estimated_input_tokens": sum(token_by_provider.values()),
        "estimated_output_token_budget": sum(output_by_provider.values()),
        "usd_cost": None, "usd_cost_reason": "no authoritative repository-local pricing table covers all exact formal judge models",
        "answer_generation_calls": 0, "paid_calls_performed": 0,
    }
    write_json(output / "dry_run_summary.json", summary)
    protocol = f"""# Formal 300-block validation protocol

Protocol `{PROTOCOL}` applies the frozen `{EVALUATOR}` evaluator to exactly the locked 300-block sample. The private block identity is supplied separately and is not redistributed. It performs no resampling and reuses existing answers only.

LongMemEval uses frozen deterministic/native routing where safe. Each of the {counts['lme_semantic']} semantic observations receives independent blinded OpenAI `{MODELS['openai']}` and Anthropic `{MODELS['anthropic']}` judgments; binary disagreement creates a result-dependent Gemini `{MODELS['gemini']}` escalation. Memora batches all native criteria for one answer through Gemini, preserving native FAMA. The intersection with the globally frozen 200-case risk sample creates {counts['memora_secondary']} OpenAI secondary requests. Low-confidence/adjudication outputs create result-dependent secondary requests; criterion disagreement creates an Anthropic third judgment. Unresolved cases require human review.

Prompts contain no generating provider, condition, GovMem score, cheap-judge result, expected effect, paper result, or other judge output. Dynamic requests are materialized only from frozen triggers and retain the same private semantic payload. Provider ledgers/results are separate and single-writer. Completed slots skip; failed and started-only slots fail closed; no automatic retries.

Analysis uses complete matched clean/corrupted/enforced triplets and a fixed-seed paired bootstrap over provider/base blocks. LongMemEval contributes binary correctness; Memora contributes native per-answer FAMA. This formal plan excludes OAKS because its frozen native deterministic evaluator remains unchanged.
"""
    freeze(output / "FORMAL_300_VALIDATION_PROTOCOL.md", protocol)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--blocks", type=Path, default=Path("results/cheap_judge_pilot_v1/selected_blocks.jsonl"))
    p.add_argument("--answers", type=Path, default=Path("results/cheap_judge_pilot_v1/selected_answers.jsonl"))
    p.add_argument("--longmemeval-source", type=Path, default=Path("$LONGMEMEVAL_SOURCE/longmemeval-cleaned/longmemeval_oracle.json"))
    p.add_argument("--memora-source", type=Path, default=Path("$MEMORA_SOURCE"))
    p.add_argument("--output", type=Path, default=Path("results/formal_300_validation_v1"))
    p.add_argument("--private", type=Path, default=Path("results/b4_scoring_validation/private/formal_300_validation_v1"))
    a = p.parse_args(argv)
    s = prepare(a.repo_root, a.blocks, a.answers, a.longmemeval_source, a.memora_source, a.output, a.private)
    for key in ("lme_total_observations", "lme_deterministic_observations", "lme_semantic_observations",
                "lme_openai_primary_calls", "lme_anthropic_primary_calls", "memora_total_observations",
                "memora_gemini_primary_calls", "memora_frozen_secondary_calls", "memora_dynamic_secondary_baseline",
                "lme_gemini_escalation_baseline", "base_paid_calls", "estimated_input_tokens", "estimated_output_token_budget"):
        print(f"{key.upper()}={s[key]}")
    print(f"PROTOCOL={PROTOCOL}")
    print(f"EVALUATOR={EVALUATOR}")
    print("ANSWER_GENERATION_CALLS=0")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__": raise SystemExit(main())
