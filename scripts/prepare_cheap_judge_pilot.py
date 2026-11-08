#!/usr/bin/env python3
"""Prepare the offline cheap-judge exploratory pilot over existing answers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b4_calibration_execution import estimate_input_tokens, prompt_hash, provider_request_prompt
from scripts.design_final_native_evaluation import (
    DESIGN_VERSION, is_valid_answer_row, lme_prompt, load_answer_paths,
    load_existing_answers, load_lme_source, load_memora_source, memora_key,
    memora_prompt, memora_subtype, native_lme_answerability,
    route_longmemeval, sha256_file, sha256_value,
)

PROTOCOL = "cheap-judge-pilot-v1"
B4_EVALUATOR = "final-native-evaluation-v2"
PILOT_SEED = 20261010
BLOCK_N = 300
CONDITIONS = ("clean", "corrupted", "enforced")
PROVIDERS = ("openai", "anthropic", "deepseek", "gemini", "kimi", "xai")
SOURCES = ("longmemeval", "memora")
JUDGES = {
    "deepseek": {"model": "deepseek-v4-pro", "limit": 1_048_576},
    "kimi": {"model": "kimi-k2.6", "limit": 262_144},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _freeze(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError(f"frozen pilot artifact drift: {path}")
    path.write_text(payload, encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    _freeze(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _freeze(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def stable_key(*parts: Any) -> str:
    return sha256_value([PILOT_SEED, *parts])


def build_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_provider_base: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("source_dataset") in SOURCES:
            by_provider_base[(str(row["provider"]), str(row["base_id"]))].append(row)
    candidates = []
    for (provider, base_id), group in by_provider_base.items():
        clean = [r for r in group if r.get("condition_type") == "clean"]
        if len(clean) != 1:
            continue
        pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in group:
            if row.get("condition_type") in {"corrupted", "enforced"} and row.get("pair_id"):
                pairs[str(row["pair_id"])][str(row["condition_type"])] = row
        for pair_id, pair in pairs.items():
            if set(pair) != {"corrupted", "enforced"}:
                continue
            triplet = {"clean": clean[0], **pair}
            if any(not is_valid_answer_row(r) for r in triplet.values()):
                continue
            source = str(clean[0]["source_dataset"])
            fault = str(pair["corrupted"].get("fault_operator") or pair_id.rsplit("::", 1)[-1])
            event = str(pair["corrupted"].get("analysis_event_type") or pair["corrupted"].get("primary_event_type") or "unknown")
            candidates.append({
                "provider": provider, "base_id": base_id, "source": source,
                "pair_id": pair_id, "fault_type": fault, "event_macro": event,
                "rows": triplet,
            })
    return candidates


def select_blocks(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select 25 unique bases per source/provider with balanced fault/event use."""
    selected: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        for source in SOURCES:
            pool = [c for c in candidates if c["provider"] == provider and c["source"] == source]
            used_bases: set[str] = set()
            fault_counts: Counter[str] = Counter()
            event_counts: Counter[str] = Counter()
            for _ in range(25):
                eligible = [c for c in pool if c["base_id"] not in used_bases]
                if not eligible:
                    raise ValueError(f"insufficient unique bases for {provider}/{source}")
                chosen = min(
                    eligible,
                    key=lambda c: (
                        fault_counts[c["fault_type"]], event_counts[c["event_macro"]],
                        stable_key(provider, source, c["base_id"], c["pair_id"]),
                    ),
                )
                selected.append(chosen)
                used_bases.add(chosen["base_id"])
                fault_counts[chosen["fault_type"]] += 1
                event_counts[chosen["event_macro"]] += 1
    if len(selected) != BLOCK_N:
        raise ValueError(f"expected {BLOCK_N} blocks, found {len(selected)}")
    keys = {(c["provider"], c["base_id"]) for c in selected}
    if len(keys) != BLOCK_N:
        raise ValueError("provider/base blocks are not unique")
    return sorted(selected, key=lambda c: (c["provider"], c["source"], c["base_id"]))


def pilot_block_id(block: dict[str, Any]) -> str:
    return "pilot-block::" + sha256_value([PROTOCOL, block["provider"], block["base_id"], block["pair_id"]])


def request_id(observation_id: str, provider: str, model: str, request_hash: str, payload_hash: str) -> str:
    return "cheap-pilot-slot::" + sha256_value({
        "protocol": PROTOCOL, "observation_id": observation_id,
        "judge_provider": provider, "judge_model": model,
        "request_hash": request_hash, "payload_hash": payload_hash,
    })


def prepare(repo: Path, lme_path: Path, memora_root: Path, output: Path, private: Path) -> dict[str, Any]:
    answer_paths = load_answer_paths(repo)
    lme_rows, memora_rows, provenance = load_existing_answers(answer_paths)
    all_rows = lme_rows + memora_rows
    lme_source = load_lme_source(lme_path)
    memora_source = load_memora_source(memora_root)
    selected = select_blocks(build_candidates(all_rows))

    block_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    native_n = semantic_n = 0

    for block in selected:
        bid = pilot_block_id(block)
        block_rows.append({
            "pilot_block_id": bid, "source": block["source"],
            "generating_provider": block["provider"], "base_id": block["base_id"],
            "base_id_sha256": sha256_value(block["base_id"]),
            "pair_id": block["pair_id"], "fault_type": block["fault_type"],
            "event_macro": block["event_macro"], "conditions": list(CONDITIONS),
        })
        for condition in CONDITIONS:
            row = block["rows"][condition]
            observation_id = "pilot-observation::" + sha256_value([bid, row["condition_id"]])
            route: dict[str, Any]
            prompt: str | None = None
            criterion_n = 0
            subtype = None
            if block["source"] == "longmemeval":
                qid = str(row["base_id"]).split("::")[-1]
                native = lme_source.get(qid)
                if native is None:
                    raise ValueError(f"missing LME source: {qid}")
                route = route_longmemeval(row, native)
                if route["route"] == "invalid_unscorable":
                    raise ValueError(f"selected invalid LME observation: {row['condition_id']}")
                if route["route"] == "semantic_judge_required":
                    prompt = lme_prompt(row, native)
                native_result = route.get("native_decision")
                if route["route"] == "deterministic_native":
                    native_result = "correct"
            else:
                key = memora_key(str(row["base_id"]))
                native = memora_source.get(key)
                if native is None:
                    raise ValueError(f"missing Memora source: {key}")
                criteria = native.get("evaluation", {}).get("evaluation_questions", [])
                if not criteria:
                    raise ValueError(f"missing Memora criteria: {key}")
                route = {"route": "semantic_judge_required", "reason": "native_atomic_criteria_require_semantic_evaluation"}
                prompt = memora_prompt(row, native)
                criterion_n = len(criteria)
                subtype = memora_subtype(key[2])
                native_result = None
            evaluator_type = "native_deterministic" if prompt is None else "cheap_dual_semantic"
            native_n += evaluator_type == "native_deterministic"
            semantic_n += evaluator_type == "cheap_dual_semantic"
            answer_public = {
                "pilot_block_id": bid, "observation_id": observation_id,
                "source": block["source"], "subtype": subtype,
                "generating_provider": block["provider"], "base_id": block["base_id"],
                "condition": condition, "condition_id": row["condition_id"],
                "answer_sha256": sha256_value(row["parsed_answer"]),
                "fault_type": block["fault_type"], "event_macro": block["event_macro"],
                "evaluator_type": evaluator_type, "route": route["route"],
                "native_result": native_result, "criterion_n": criterion_n,
            }
            answer_rows.append(answer_public)
            if prompt is None:
                continue
            semantic_hash = prompt_hash(prompt)
            payload_hash = sha256_value({"protocol": PROTOCOL, "observation_id": observation_id, "prompt": prompt})
            payload_rows.append({
                "pilot_block_id": bid, "observation_id": observation_id,
                "source": block["source"], "subtype": subtype,
                "condition_id": row["condition_id"], "answer_sha256": answer_public["answer_sha256"],
                "criterion_n": criterion_n, "prompt": prompt,
                "semantic_prompt_hash": semantic_hash, "payload_hash": payload_hash,
                "protocol_version": PROTOCOL,
            })
            for judge_provider, config in JUDGES.items():
                request_prompt = provider_request_prompt(judge_provider, prompt)
                request_hash = prompt_hash(request_prompt)
                measured = estimate_input_tokens(request_prompt, config["model"])
                if measured["estimated_tokens"] > config["limit"]:
                    raise ValueError(f"input-limit blocker: {observation_id}/{judge_provider}")
                rid = request_id(observation_id, judge_provider, config["model"], request_hash, payload_hash)
                request_rows.append({
                    "evaluation_request_id": rid, "pilot_block_id": bid,
                    "observation_id": observation_id, "source": block["source"],
                    "condition_id": row["condition_id"], "answer_sha256": answer_public["answer_sha256"],
                    "judge_provider": judge_provider, "judge_model": config["model"],
                    "judge_slot": "A" if judge_provider == "deepseek" else "B",
                    "semantic_prompt_hash": semantic_hash, "request_hash": request_hash,
                    "payload_hash": payload_hash, "protocol_version": PROTOCOL,
                    "private_payload_locator": f"semantic_payloads.jsonl#{observation_id}",
                    "estimated_input_tokens": measured["estimated_tokens"],
                    "input_estimation_method": measured["method"],
                    "planned_output_tokens": 160 if block["source"] == "longmemeval" else 40 + 36 * criterion_n,
                    "max_attempts": 1, "automatic_retries": False,
                })

    if len(block_rows) != 300 or len(answer_rows) != 900:
        raise ValueError("pilot population invariant failed")
    if len({r["observation_id"] for r in answer_rows}) != 900:
        raise ValueError("duplicate pilot observation")
    if len({r["evaluation_request_id"] for r in request_rows}) != len(request_rows):
        raise ValueError("duplicate paid request identity")
    if any(r["judge_provider"] not in JUDGES for r in request_rows):
        raise ValueError("non-cheap provider entered pilot")

    block_rows.sort(key=lambda r: r["pilot_block_id"])
    answer_rows.sort(key=lambda r: (r["pilot_block_id"], CONDITIONS.index(r["condition"])))
    payload_rows.sort(key=lambda r: r["observation_id"])
    request_rows.sort(key=lambda r: r["evaluation_request_id"])
    write_jsonl(output / "selected_blocks.jsonl", block_rows)
    write_jsonl(output / "selected_answers.jsonl", answer_rows)
    write_jsonl(output / "request_manifest.jsonl", request_rows)
    write_jsonl(private / "semantic_payloads.jsonl", payload_rows)

    by_source = dict(sorted(Counter(r["source"] for r in block_rows).items()))
    by_provider = dict(sorted(Counter(r["generating_provider"] for r in block_rows).items()))
    by_fault = dict(sorted(Counter(r["fault_type"] for r in block_rows).items()))
    by_event = dict(sorted(Counter(r["event_macro"] for r in block_rows).items()))
    allocation = {
        "protocol_version": PROTOCOL, "seed": PILOT_SEED, "block_n": 300,
        "observation_n": 900, "conditions": list(CONDITIONS),
        "sampling_unit": "unique generating-provider/base with one matched fault pair",
        "by_source": by_source, "by_generating_provider": by_provider,
        "by_fault_type": by_fault, "by_event_macro": by_event,
        "source_provider_cells": {
            f"{s}/{p}": sum(r["source"] == s and r["generating_provider"] == p for r in block_rows)
            for s in SOURCES for p in PROVIDERS
        },
        "oaks_policy": "excluded: frozen native deterministic parity already complete; no semantic judgment required",
        "answer_artifacts": provenance["answer_artifacts"],
        "source_hashes": {"longmemeval": sha256_file(lme_path), "memora": sha256_value(sorted(map(str, memora_source)))},
        "selected_block_fingerprint": sha256_value(block_rows),
    }
    write_json(output / "sampling_plan.json", allocation)
    calls = Counter(r["judge_provider"] for r in request_rows)
    tokens = Counter()
    outputs = 0
    for row in request_rows:
        tokens[row["judge_provider"]] += row["estimated_input_tokens"]
        outputs += row["planned_output_tokens"]
    dry = {
        "protocol": PROTOCOL, "scientific_status": "exploratory_only_failed_formal_selection_gate",
        "pilot_blocks": 300, "expected_observations": 900,
        "complete_matched_triplets": 300, "by_source": by_source,
        "by_generating_provider": by_provider, "by_fault_type": by_fault,
        "native_deterministic_observations": native_n,
        "semantic_observations": semantic_n,
        "deepseek_expected_calls": calls["deepseek"], "kimi_expected_calls": calls["kimi"],
        "total_expected_paid_calls": len(request_rows),
        "estimated_deepseek_input_tokens": tokens["deepseek"],
        "estimated_kimi_input_tokens": tokens["kimi"],
        "estimated_output_token_budget": outputs,
        "answer_generation_calls": 0, "other_provider_calls": 0,
        "concurrency_safe": True,
        "concurrency_note": "separate provider ledgers/results plus one exclusive process lock per provider; one writer per partition",
        "paid_calls_performed": 0,
        "usd_cost": None,
        "usd_cost_reason": "no authoritative repository-local pricing table covers both exact judge models",
    }
    write_json(output / "dry_run_summary.json", dry)
    protocol_doc = f"""# Cheap-judge exploratory pilot v1

Protocol: `{PROTOCOL}`. Seed: `{PILOT_SEED}`. Scientific status: **exploratory only**. DeepSeek `{JUDGES['deepseek']['model']}` and Kimi `{JUDGES['kimi']['model']}` failed the formal B4 evaluator-selection gate, so this pilot cannot replace or modify `final-native-evaluation-v2`, final results, or paper claims.

## Frozen sample

The sampling unit is one unique generating-provider/base with one matched fault pair: the base's clean answer and that pair's corrupted and enforced answers. The 300 blocks contain 900 observations, split equally between LongMemEval and Memora and exactly 50 blocks per generating provider. OAKS is excluded because its native deterministic parity is already established and it requires no semantic pilot judging. Full source/provider/fault/event allocation is in `sampling_plan.json`.

## Evaluation

Conservative LongMemEval native matches and answerability-established abstentions are resolved without paid calls. Every other selected observation is independently blinded to DeepSeek and Kimi. The prompts omit generating provider, condition, GovMem score, expected direction, and the other judge's output. LongMemEval retains the frozen abstention-aware labels and projects them to binary correctness.

Memora retains every native `memory_presence` and `forgetting_absence` criterion and computes native per-answer FAMA for each judge. For this exploratory binary analysis only, an answer projects to `correct` when its native FAMA is exactly 1.0; FAMA itself is preserved. This full-credit projection does not replace native FAMA or create a new final B4 metric. Any `adjudication_required`, missing output, or binary disagreement remains unresolved. No frontier or human escalation is allowed.

## Analysis and execution safety

Effects are paired within complete provider/base triplets. Ninety-five-percent intervals use 10,000 fixed-seed bootstrap resamples of matched blocks. The precommitted signal labels are implemented in `analyze_cheap_judge_pilot.py`; small resolved samples or cheap-judge agreement below 0.75 are inconclusive.

DeepSeek and Kimi use separate append-only ledgers, result files, and exclusive provider-partition locks. One writer per provider is allowed; completed slots skip, failed slots do not retry automatically, and started-only slots fail closed. Default execution is dry-run. Paid execution requires both `--execute` and `--confirm-paid-judge-calls`.
"""
    _freeze(output / "PILOT_PROTOCOL.md", protocol_doc)
    return dry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--longmemeval-source", type=Path, default=Path("$LONGMEMEVAL_SOURCE/longmemeval-cleaned/longmemeval_oracle.json"))
    parser.add_argument("--memora-source", type=Path, default=Path("$MEMORA_SOURCE"))
    parser.add_argument("--output", type=Path, default=Path("results/cheap_judge_pilot_v1"))
    parser.add_argument("--private", type=Path, default=Path("results/b4_scoring_validation/private/cheap_judge_pilot_v1"))
    args = parser.parse_args(argv)
    x = prepare(args.repo_root, args.longmemeval_source, args.memora_source, args.output, args.private)
    print(f"PROTOCOL={PROTOCOL}")
    print(f"PILOT_BLOCKS={x['pilot_blocks']}")
    print(f"EXPECTED_OBSERVATIONS={x['expected_observations']}")
    print(f"COMPLETE_MATCHED_TRIPLETS={x['complete_matched_triplets']}")
    print(f"BY_SOURCE={json.dumps(x['by_source'], sort_keys=True)}")
    print(f"BY_GENERATING_PROVIDER={json.dumps(x['by_generating_provider'], sort_keys=True)}")
    print(f"BY_FAULT_TYPE={json.dumps(x['by_fault_type'], sort_keys=True)}")
    print(f"NATIVE_DETERMINISTIC_OBSERVATIONS={x['native_deterministic_observations']}")
    print(f"SEMANTIC_OBSERVATIONS={x['semantic_observations']}")
    print(f"DEEPSEEK_EXPECTED_CALLS={x['deepseek_expected_calls']}")
    print(f"KIMI_EXPECTED_CALLS={x['kimi_expected_calls']}")
    print(f"TOTAL_EXPECTED_PAID_CALLS={x['total_expected_paid_calls']}")
    print(f"ESTIMATED_DEEPSEEK_INPUT_TOKENS={x['estimated_deepseek_input_tokens']}")
    print(f"ESTIMATED_KIMI_INPUT_TOKENS={x['estimated_kimi_input_tokens']}")
    print(f"ESTIMATED_OUTPUT_TOKEN_BUDGET={x['estimated_output_token_budget']}")
    print("ANSWER_GENERATION_CALLS=0")
    print("OTHER_PROVIDER_CALLS=0")
    print("CONCURRENCY_SAFE=yes")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
