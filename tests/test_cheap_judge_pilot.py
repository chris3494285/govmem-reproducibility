from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.analyze_cheap_judge_pilot import analyze, paired_bootstrap, paired_metrics
from scripts.prepare_cheap_judge_pilot import (
    JUDGES, PROTOCOL, request_id, select_blocks, sha256_value,
)
from scripts.run_cheap_judge_pilot import parse_result, provider_preflight
from scripts.b4_calibration_execution import DurableLedger


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results/cheap_judge_pilot_v1"
PRIVATE = ROOT / "results/b4_scoring_validation/private/cheap_judge_pilot_v1"


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_frozen_sample_has_exact_matched_population():
    blocks = rows(PUBLIC / "selected_blocks.jsonl")
    answers = rows(PUBLIC / "selected_answers.jsonl")
    assert len(blocks) == 300
    assert len({(r["generating_provider"], r["base_id"]) for r in blocks}) == 300
    assert len(answers) == 900
    by_block = {}
    for row in answers:
        by_block.setdefault(row["pilot_block_id"], []).append(row)
    assert set(by_block) == {r["pilot_block_id"] for r in blocks}
    assert all({r["condition"] for r in group} == {"clean", "corrupted", "enforced"} for group in by_block.values())
    assert Counter(r["generating_provider"] for r in blocks) == Counter({p: 50 for p in ("openai", "anthropic", "deepseek", "gemini", "kimi", "xai")})
    assert Counter(r["source"] for r in blocks) == Counter({"longmemeval": 150, "memora": 150})


def test_sample_and_request_hashes_are_deterministic():
    plan = json.loads((PUBLIC / "sampling_plan.json").read_text())
    blocks = rows(PUBLIC / "selected_blocks.jsonl")
    assert plan["selected_block_fingerprint"] == sha256_value(blocks)
    requests = rows(PUBLIC / "request_manifest.jsonl")
    for row in requests:
        expected = request_id(row["observation_id"], row["judge_provider"], row["judge_model"], row["request_hash"], row["payload_hash"])
        assert row["evaluation_request_id"] == expected
    assert len({r["evaluation_request_id"] for r in requests}) == len(requests)


def test_only_deepseek_and_kimi_slots_and_native_routes_cost_zero():
    answers = rows(PUBLIC / "selected_answers.jsonl")
    requests = rows(PUBLIC / "request_manifest.jsonl")
    assert {r["judge_provider"] for r in requests} == {"deepseek", "kimi"}
    assert {r["judge_model"] for r in requests} == {"deepseek-v4-pro", "kimi-k2.6"}
    requested = Counter(r["observation_id"] for r in requests)
    for row in answers:
        assert requested[row["observation_id"]] == (0 if row["evaluator_type"] == "native_deterministic" else 2)
    assert not any(r["judge_provider"] in {"openai", "anthropic", "gemini"} for r in requests)


def test_prompts_are_blind_and_judge_slots_independent():
    payloads = rows(PRIVATE / "semantic_payloads.jsonl")
    requests = rows(PUBLIC / "request_manifest.jsonl")
    forbidden_keys = {"provider", "generating_provider", "condition", "condition_id", "correctness", "govmem", "other_judge"}
    for payload in payloads:
        prompt = json.loads(payload["prompt"])
        assert not forbidden_keys.intersection(prompt)
    slots = Counter((r["observation_id"], r["judge_provider"]) for r in requests)
    assert all(value == 1 for value in slots.values())
    assert all({p for obs, p in slots if obs == payload["observation_id"]} == set(JUDGES) for payload in payloads)


def test_no_answer_generation_or_paid_calls_in_dry_run():
    summary = json.loads((PUBLIC / "dry_run_summary.json").read_text())
    assert summary["answer_generation_calls"] == 0
    assert summary["other_provider_calls"] == 0
    assert summary["paid_calls_performed"] == 0
    assert summary["protocol"] == PROTOCOL


def test_strict_lme_parser_and_memora_native_fama_projection():
    raw = json.dumps({"decision": "abstention_incorrect", "confidence": "high", "short_reason": "answerable", "answerability_basis": "answerable"})
    parsed = parse_result(raw, "longmemeval", "{}")
    assert parsed["exploratory_binary"] is False
    prompt = json.dumps({"criteria": [
        {"criterion_id": "mp", "evaluation_type": "memory_presence"},
        {"criterion_id": "fa", "evaluation_type": "forgetting_absence"},
    ]})
    raw = json.dumps({"decisions": [
        {"criterion_id": "mp", "decision": "pass", "confidence": "high", "short_reason": "present"},
        {"criterion_id": "fa", "decision": "pass", "confidence": "high", "short_reason": "absent"},
    ]})
    parsed = parse_result(raw, "memora", prompt)
    assert parsed["native_fama"] == 1.0
    assert parsed["exploratory_binary"] is True
    with pytest.raises(ValueError):
        parse_result('{"decision":"correct"}', "longmemeval", "{}")


def test_disagreement_remains_unresolved_and_analysis_is_paired():
    answers = []
    results = {}
    for index, condition in enumerate(("clean", "corrupted", "enforced")):
        oid = f"o{index}"
        answers.append({"pilot_block_id": "b", "observation_id": oid, "source": "longmemeval", "subtype": None,
                        "generating_provider": "p", "condition": condition, "fault_type": "stale",
                        "evaluator_type": "cheap_dual_semantic", "native_result": None})
        results[(oid, "deepseek")] = {"parsed_result": {"exploratory_binary": condition != "corrupted"}}
        results[(oid, "kimi")] = {"parsed_result": {"exploratory_binary": condition != "corrupted"}}
    results[("o2", "kimi")]["parsed_result"]["exploratory_binary"] = False
    output = analyze(answers, results, iterations=100)
    assert output["summary"]["complete_matched_triplets"] == 0
    assert output["summary"]["cheap_judge_disagreement_n"] == 1


def test_paired_bootstrap_is_fixed_seed_deterministic():
    blocks = [
        {"results": {"clean": True, "corrupted": False, "enforced": True}},
        {"results": {"clean": True, "corrupted": True, "enforced": True}},
    ]
    assert paired_metrics(blocks)["clean_minus_corrupted"] == 0.5
    assert paired_bootstrap(blocks, seed=123, iterations=250) == paired_bootstrap(blocks, seed=123, iterations=250)


def test_started_only_fails_closed_without_provider_call(tmp_path):
    ledger = DurableLedger(tmp_path / "ledger.jsonl")
    row = {"evaluation_request_id": "r", "judge_provider": "deepseek"}
    ledger.append({**row, "state": "started"})
    with pytest.raises(RuntimeError, match="fail-closed"):
        provider_preflight([row], ledger)
    audit = provider_preflight([row], ledger, fail_on_blocked=False)
    assert audit == {"pending": 0, "completed": 0, "failed": 0, "started_only": 1, "blocked": 1}


def test_completed_slots_skip_and_failed_slots_need_recovery(tmp_path):
    completed = {"evaluation_request_id": "done", "observation_id": "o1", "judge_provider": "deepseek",
                 "judge_model": "m", "request_hash": "r1", "payload_hash": "p1", "protocol_version": PROTOCOL}
    failed = {**completed, "evaluation_request_id": "bad", "observation_id": "o2", "request_hash": "r2", "payload_hash": "p2"}
    ledger = DurableLedger(tmp_path / "ledger.jsonl")
    ledger.append({**completed, "state": "started"})
    ledger.append({**completed, "state": "completed"})
    ledger.append({**failed, "state": "started"})
    ledger.append({**failed, "state": "failed"})
    audit = provider_preflight([completed, failed], ledger, fail_on_blocked=False)
    assert audit["completed"] == 1
    assert audit["failed"] == 1
    assert audit["pending"] == 0
    with pytest.raises(RuntimeError, match="fail-closed"):
        provider_preflight([completed, failed], ledger)
