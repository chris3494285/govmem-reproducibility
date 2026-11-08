#!/usr/bin/env python3
"""Preflight or execute the isolated DeepSeek/Kimi exploratory pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b4_calibration_execution import DeepSeekJudgeAdapter, DurableLedger, KimiJudgeAdapter
from scripts.b4_calibration_execution import prompt_hash, provider_request_prompt
from scripts.prepare_cheap_judge_pilot import JUDGES, PROTOCOL, read_jsonl, request_id, sha256_value


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_result(raw: str, source: str, prompt: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("result must be an object")
    if source == "longmemeval":
        required = {"decision", "confidence", "short_reason", "answerability_basis"}
        if set(value) != required:
            raise ValueError("invalid LongMemEval result fields")
        if value["decision"] not in {"correct", "incorrect", "abstention_correct", "abstention_incorrect", "adjudication_required"}:
            raise ValueError("invalid LongMemEval decision")
        if value["answerability_basis"] not in {"answerable", "insufficient_information", "ambiguous"}:
            raise ValueError("invalid answerability_basis")
        binary = {"correct": True, "abstention_correct": True, "incorrect": False, "abstention_incorrect": False}.get(value["decision"])
        result = {**value, "exploratory_binary": binary, "native_fama": None}
    else:
        if set(value) != {"decisions"} or not isinstance(value["decisions"], list):
            raise ValueError("Memora result must contain only decisions")
        prompt_value = json.loads(prompt)
        criteria = prompt_value["criteria"]
        expected = {str(c["criterion_id"]): str(c["evaluation_type"]) for c in criteria}
        observed: dict[str, dict[str, Any]] = {}
        for item in value["decisions"]:
            if not isinstance(item, dict) or set(item) != {"criterion_id", "decision", "confidence", "short_reason"}:
                raise ValueError("invalid Memora criterion result fields")
            cid = str(item["criterion_id"])
            if cid not in expected or cid in observed:
                raise ValueError("unknown or duplicate Memora criterion ID")
            if item["decision"] not in {"pass", "fail", "adjudication_required"}:
                raise ValueError("invalid Memora decision")
            observed[cid] = item
        if set(observed) != set(expected):
            raise ValueError("missing Memora criterion result")
        unresolved = any(item["decision"] == "adjudication_required" for item in observed.values())
        presence = [cid for cid, kind in expected.items() if kind == "memory_presence"]
        absence = [cid for cid, kind in expected.items() if kind == "forgetting_absence"]
        mpa = sum(observed[cid]["decision"] == "pass" for cid in presence) / len(presence) if presence else 1.0
        faa = sum(observed[cid]["decision"] == "pass" for cid in absence) / len(absence) if absence else 1.0
        lam = len(absence) / len(expected)
        fama = None if unresolved else max(0.0, mpa - lam * (1.0 - faa))
        # Exploratory full-credit projection; native FAMA remains preserved.
        binary = None if fama is None else abs(fama - 1.0) < 1e-12
        result = {"decisions": list(observed.values()), "native_fama": fama, "exploratory_binary": binary}
    confidences = [result.get("confidence")] if source == "longmemeval" else [x["confidence"] for x in result["decisions"]]
    if any(c not in {"high", "medium", "low"} for c in confidences):
        raise ValueError("invalid confidence")
    reasons = [result.get("short_reason")] if source == "longmemeval" else [x["short_reason"] for x in result["decisions"]]
    if any(not isinstance(x, str) or not x.strip() for x in reasons):
        raise ValueError("missing short_reason")
    return result


def current_state(events: list[dict[str, Any]], request_id: str) -> str:
    history = [e for e in events if e.get("evaluation_request_id") == request_id]
    if not history:
        return "pending"
    states = [str(e.get("state")) for e in history]
    if states.count("completed") == 1 and states[-1] == "completed":
        return "completed"
    if "started" in states and not any(s in {"completed", "failed"} for s in states[states.index("started") + 1:]):
        return "started_only"
    return states[-1]


def provider_preflight(rows: list[dict[str, Any]], ledger: DurableLedger, *, fail_on_blocked: bool = True) -> dict[str, int]:
    events = ledger.read()
    expected = {r["evaluation_request_id"]: r for r in rows}
    if len(expected) != len(rows):
        raise RuntimeError("duplicate request identity in frozen manifest")
    identity_fields = ("observation_id", "judge_provider", "judge_model", "request_hash", "payload_hash", "protocol_version")
    for event in events:
        rid = event.get("evaluation_request_id")
        if rid not in expected:
            raise RuntimeError("ledger contains a request outside the frozen provider partition")
        if any(str(event.get(key)) != str(expected[rid].get(key)) for key in identity_fields):
            raise RuntimeError(f"ledger identity drift: {rid}")
    states = [current_state(events, r["evaluation_request_id"]) for r in rows]
    failed = states.count("failed")
    started_only = states.count("started_only")
    blocked = sum(s not in {"pending", "completed"} for s in states)
    if blocked and fail_on_blocked:
        raise RuntimeError(f"fail-closed: {blocked} slot(s) require explicit recovery")
    return {"pending": states.count("pending"), "completed": states.count("completed"),
            "failed": failed, "started_only": started_only, "blocked": blocked}


def validate_plan(rows: list[dict[str, Any]], payloads: dict[str, dict[str, Any]]) -> None:
    for row in rows:
        if row.get("protocol_version") != PROTOCOL:
            raise RuntimeError("protocol drift")
        provider = str(row["judge_provider"])
        config = JUDGES.get(provider)
        if config is None or row.get("judge_model") != config["model"]:
            raise RuntimeError("provider/model drift")
        payload = payloads.get(str(row["observation_id"]))
        if payload is None:
            raise RuntimeError("missing private payload")
        expected_payload_hash = sha256_value({"protocol": PROTOCOL, "observation_id": row["observation_id"], "prompt": payload["prompt"]})
        expected_request_hash = prompt_hash(provider_request_prompt(provider, payload["prompt"]))
        expected_id = request_id(row["observation_id"], provider, config["model"], expected_request_hash, expected_payload_hash)
        if (row.get("payload_hash"), row.get("request_hash"), row.get("evaluation_request_id")) != (expected_payload_hash, expected_request_hash, expected_id):
            raise RuntimeError("prompt/payload/request identity drift")


def execute_provider(provider: str, rows: list[dict[str, Any]], payloads: dict[str, dict[str, Any]], root: Path) -> int:
    config = JUDGES[provider]
    env_name = "DEEPSEEK_API_KEY" if provider == "deepseek" else "KIMI_API_KEY"
    if not os.environ.get(env_name):
        raise RuntimeError(f"{env_name} is required before any ledger event is written")
    provider_root = root / provider
    ledger = DurableLedger(provider_root / "judge_execution_ledger.jsonl")
    state = provider_preflight(rows, ledger)
    result_path = provider_root / "judge_results.jsonl"
    existing_results = read_jsonl(result_path) if result_path.exists() else []
    result_ids = [str(r.get("evaluation_request_id")) for r in existing_results]
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("duplicate durable result identity")
    completed_ids = {str(e.get("evaluation_request_id")) for e in ledger.read() if e.get("state") == "completed"}
    if set(result_ids) != completed_ids:
        raise RuntimeError("ledger/result completion mismatch")
    if not state["pending"]:
        return 0
    lock = provider_root / ".execution.lock"
    provider_root.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"provider partition is already locked: {provider}") from exc
    os.close(fd)
    max_output_tokens = max(int(row["planned_output_tokens"]) for row in rows)
    adapter = DeepSeekJudgeAdapter(config["model"], max_output_tokens=max_output_tokens) if provider == "deepseek" else KimiJudgeAdapter(config["model"], max_output_tokens=max_output_tokens)
    paid = 0
    try:
        events = ledger.read()
        for row in rows:
            if current_state(events, row["evaluation_request_id"]) == "completed":
                continue
            payload = payloads[row["observation_id"]]
            base = {k: row[k] for k in ("evaluation_request_id", "pilot_block_id", "observation_id", "source", "judge_provider", "judge_model", "request_hash", "payload_hash", "protocol_version")}
            ledger.append({**base, "state": "planned", "attempt": 1, "timestamp_ns": time.time_ns()})
            ledger.append({**base, "state": "started", "attempt": 1, "timestamp_ns": time.time_ns()})
            paid += 1
            try:
                raw = adapter.call(payload["prompt"], payload["source"])
                parsed = parse_result(raw, payload["source"], payload["prompt"])
                response_hash = sha256_value(raw)
                result = {**base, "attempt": 1, "parsed_result": parsed, "response_hash": response_hash,
                          "provider_usage": getattr(adapter, "last_usage", None)}
                append_jsonl(result_path, result)
                ledger.append({**base, "state": "completed", "attempt": 1, "response_hash": response_hash, "timestamp_ns": time.time_ns()})
            except Exception as exc:
                ledger.append({**base, "state": "failed", "attempt": 1, "error_class": type(exc).__name__, "error_message": str(exc)[:1000], "timestamp_ns": time.time_ns()})
            events = ledger.read()
    finally:
        lock.unlink(missing_ok=True)
    return paid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/cheap_judge_pilot_v1/request_manifest.jsonl"))
    parser.add_argument("--payloads", type=Path, default=Path("results/b4_scoring_validation/private/cheap_judge_pilot_v1/semantic_payloads.jsonl"))
    parser.add_argument("--private-root", type=Path, default=Path("results/b4_scoring_validation/private/cheap_judge_pilot_v1"))
    parser.add_argument("--provider", choices=("deepseek", "kimi", "both"), default="both")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid-judge-calls", action="store_true")
    args = parser.parse_args(argv)
    rows = read_jsonl(args.manifest)
    payloads = {r["observation_id"]: r for r in read_jsonl(args.payloads)}
    validate_plan(rows, payloads)
    providers = list(JUDGES) if args.provider == "both" else [args.provider]
    planned = 0
    summaries = {}
    for provider in providers:
        selected = [r for r in rows if r["judge_provider"] == provider]
        state = provider_preflight(
            selected,
            DurableLedger(args.private_root / provider / "judge_execution_ledger.jsonl"),
            fail_on_blocked=args.execute,
        )
        summaries[provider] = state
        planned += state["pending"]
        print(f"{provider.upper()}_PENDING={state['pending']}")
        print(f"{provider.upper()}_COMPLETED={state['completed']}")
        print(f"{provider.upper()}_FAILED={state['failed']}")
        print(f"{provider.upper()}_STARTED_ONLY={state['started_only']}")
        print(f"{provider.upper()}_BLOCKED={state['blocked']}")
    print(f"ESTIMATED_NEW_PAID_CALLS={planned}")
    print("CONCURRENCY_SAFE=yes")
    if args.execute != args.confirm_paid_judge_calls:
        raise SystemExit("paid execution requires both --execute and --confirm-paid-judge-calls")
    if not args.execute:
        print("EXECUTION_MODE=dry_run_only")
        print("PAID_CALLS_PERFORMED=0")
        return 0
    paid = 0
    for provider in providers:
        paid += execute_provider(provider, [r for r in rows if r["judge_provider"] == provider], payloads, args.private_root)
    print(f"PAID_CALLS_PERFORMED={paid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
