#!/usr/bin/env python3
"""Prepare a blinded, offline judge-selection calibration from frozen B4 gold.

No provider client is imported or invoked.  Judge-visible benchmark text stays
in the existing gitignored private payload files; public plans contain only
identifiers, hashes, sizes, schema metadata, and calibration gold labels.
Locked holdout labels are deliberately omitted from every generated artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b4_calibration_runner import build_lme_prompt, build_memora_prompt


SCHEMA_VERSION = "final-judge-selection-calibration-v1"
SOURCE_PROTOCOL_VERSION = "final-native-evaluation-design-v1"
SEED = 20261004
EXPECTED_GOLD_N = 604
CALIBRATION_TARGET_N = 404
SUPPORTED_PROVIDER_FAMILIES = ["anthropic", "deepseek", "gemini", "kimi", "openai"]
FORBIDDEN_PROMPT_KEYS = {
    "provider", "model", "condition", "condition_type", "correctness",
    "deterministic_score", "scoring_status", "scoring_method", "scoring_version",
    "govmem", "agreement", "gold", "reference_label", "other_judge",
}


def digest(value: Any) -> str:
    payload = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def load_gold(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != EXPECTED_GOLD_N:
        raise ValueError(f"expected {EXPECTED_GOLD_N} frozen gold cases, found {len(rows)}")
    ids = [str(row.get("blinded_request_id")) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate blinded_request_id in frozen gold")
    for row in rows:
        if row.get("final_reference_status") != "resolved" or row.get("human_review_required") is not False:
            raise ValueError(f"gold case is not fully resolved: {row.get('blinded_request_id')}")
        task = row.get("task")
        binary = row.get("final_reference_binary_label")
        if task == "longmemeval" and binary not in {"correct", "incorrect"}:
            raise ValueError("invalid LongMemEval binary gold label")
        if task == "memora" and binary not in {"pass", "fail"}:
            raise ValueError("invalid Memora gold label")
        if task not in {"longmemeval", "memora"}:
            raise ValueError(f"unknown gold task: {task}")
    return sorted(rows, key=lambda row: str(row["blinded_request_id"]))


def raw_abstention_class(row: dict[str, Any]) -> str:
    if row["task"] != "longmemeval":
        return "not_applicable"
    raw = row.get("final_reference_raw_label")
    if raw in {"abstention_correct", "abstention_incorrect"}:
        return str(raw)
    if raw in {"correct", "incorrect"}:
        return "non_abstention"
    return "raw_subtype_unresolved"


def split_stratum(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row["task"]),
        str(row.get("subtype") or "none"),
        str(row["final_reference_binary_label"]),
        str(row["resolution_method"]),
        raw_abstention_class(row),
    )


def stratified_split(
    rows: list[dict[str, Any]],
    calibration_n: int = CALIBRATION_TARGET_N,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Allocate exactly calibration_n using deterministic largest remainders."""
    if not 0 < calibration_n < len(rows):
        raise ValueError("calibration size must leave a non-empty holdout")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[split_stratum(row)].append(row)
    ratio = calibration_n / len(rows)
    allocation = {key: math.floor(len(values) * ratio) for key, values in groups.items()}
    remaining = calibration_n - sum(allocation.values())
    ranked = sorted(
        groups,
        key=lambda key: (-(len(groups[key]) * ratio - allocation[key]), digest([seed, key])),
    )
    for key in ranked[:remaining]:
        allocation[key] += 1
    if sum(allocation.values()) != calibration_n:
        raise AssertionError("stratified allocation did not reach target")

    calibration: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    exact_accounting = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda row: digest([seed, row["blinded_request_id"]]))
        cut = allocation[key]
        calibration.extend(ordered[:cut])
        holdout.extend(ordered[cut:])
        exact_accounting.append({"stratum": list(key), "total": len(ordered), "calibration": cut, "holdout": len(ordered) - cut})
    calibration.sort(key=lambda row: str(row["blinded_request_id"]))
    holdout.sort(key=lambda row: str(row["blinded_request_id"]))
    if len(calibration) != calibration_n or len(holdout) != len(rows) - calibration_n:
        raise AssertionError("split cardinality drift")
    if {row["blinded_request_id"] for row in calibration} & {row["blinded_request_id"] for row in holdout}:
        raise AssertionError("calibration/holdout overlap")
    # Only a fingerprint leaves this script. Exact label-bearing stratum
    # accounting remains reproducible from the private frozen gold source.
    accounting = {
        "stratum_n": len(exact_accounting),
        "exact_accounting_sha256": digest(exact_accounting),
        "calibration_n": len(calibration),
        "holdout_n": len(holdout),
    }
    return calibration, holdout, accounting


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key).casefold())
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def validate_blind_payload(record: dict[str, Any]) -> None:
    offending = sorted({key for key in _walk_keys(record) if key in FORBIDDEN_PROMPT_KEYS or key.startswith("primary_") or key.startswith("gemini_")})
    if offending:
        raise ValueError(f"judge payload contains forbidden leakage keys: {offending}")
    required = (
        {"question", "native_context", "reference_answer", "model_answer"}
        if str(record.get("blinded_request_id", "")).startswith("longmemeval::")
        else {"native_context", "criterion", "expected_behavior", "model_answer", "subtype"}
    )
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise ValueError(f"judge payload missing semantic fields: {missing}")


def load_payloads(lme_path: Path, memora_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    index: dict[str, dict[str, Any]] = {}
    locators: dict[str, str] = {}
    for task, path in (("longmemeval", lme_path), ("memora", memora_path)):
        for row in read_jsonl(path):
            validate_blind_payload(row)
            identifier = str(row["blinded_request_id"])
            if identifier in index:
                raise ValueError(f"duplicate private payload ID: {identifier}")
            index[identifier] = row
            locators[identifier] = f"{path.as_posix()}#{identifier}"
    return index, locators


def build_prompt(task: str, payload: dict[str, Any]) -> str:
    prompt = build_lme_prompt(payload) if task == "longmemeval" else build_memora_prompt(payload)
    parsed = json.loads(prompt)
    prompt_keys = set(_walk_keys(parsed))
    forbidden = {key for key in prompt_keys if key in FORBIDDEN_PROMPT_KEYS or key.startswith("primary_") or key.startswith("gemini_")}
    if forbidden:
        raise ValueError(f"candidate prompt leaks forbidden metadata: {sorted(forbidden)}")
    return prompt


def materialize_cases(
    gold: list[dict[str, Any]], payload_index: dict[str, dict[str, Any]], locators: dict[str, str]
) -> dict[str, dict[str, Any]]:
    if set(payload_index) != {str(row["blinded_request_id"]) for row in gold}:
        missing = {str(row["blinded_request_id"]) for row in gold} - set(payload_index)
        extra = set(payload_index) - {str(row["blinded_request_id"]) for row in gold}
        raise ValueError(f"gold/payload identity mismatch: missing={len(missing)} extra={len(extra)}")
    result = {}
    for row in gold:
        identifier = str(row["blinded_request_id"])
        prompt = build_prompt(str(row["task"]), payload_index[identifier])
        result[identifier] = {
            "blinded_request_id": identifier,
            "task": row["task"],
            "subtype": row.get("subtype"),
            "payload_locator": locators[identifier],
            "payload_sha256": digest(payload_index[identifier]),
            "prompt_sha256": digest(prompt),
            "prompt_chars": len(prompt),
            "estimated_input_tokens_div4": math.ceil(len(prompt) / 4),
        }
    return result


def distribution(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0, "median": 0, "mean": 0, "p90": 0, "max": 0, "total": 0}
    return {
        "min": ordered[0], "median": statistics.median(ordered),
        "mean": round(statistics.fmean(ordered), 3),
        "p90": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
        "max": ordered[-1], "total": sum(ordered),
    }


def task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["task"]) for row in rows).items()))


def subtype_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("subtype") or "none") for row in rows).items()))


def selection_rules() -> dict[str, Any]:
    return {
        "eligibility_thresholds": {"overall_agreement_min": 0.90, "longmemeval_agreement_min": 0.90, "memora_agreement_min": 0.90},
        "preferred_overall_agreement": 0.95,
        "required_reports": ["cohen_kappa", "confusion_matrix_and_fp_fn", "longmemeval_abstention_specific_accuracy", "adjudication_required_rate", "parse_failure_rate", "provider_api_failure_rate"],
        "selection_principle": "A candidate must pass every eligibility threshold; cost alone cannot select a judge. Prefer >=0.95 overall among eligible candidates, then consider reliability, failures, independence, and token/call cost.",
        "holdout_rule": "Holdout labels remain locked during selection. Evaluating or revealing them requires a separate explicit command after one primary candidate is selected and frozen.",
    }


def protocol_markdown(manifest: dict[str, Any]) -> str:
    c = manifest["split"]["calibration"]
    h = manifest["split"]["locked_holdout"]
    costs = manifest["cost_scale"]
    return f"""# Judge-selection calibration protocol

Schema: `{SCHEMA_VERSION}`. Source evaluation protocol: `{SOURCE_PROTOCOL_VERSION}`. Split seed: `{SEED}`.

## Frozen population and split

The only gold source is the fully adjudicated 604-case B4 reference. Labels are not changed or re-adjudicated. The deterministic stratified split contains {c['total']} calibration cases ({c['by_task'].get('longmemeval', 0)} LongMemEval, {c['by_task'].get('memora', 0)} Memora) and {h['total']} locked holdout identities ({h['by_task'].get('longmemeval', 0)} LongMemEval, {h['by_task'].get('memora', 0)} Memora).

Strata use task, Memora subtype, independent binary/pass-fail label, adjudication resolution method, and LongMemEval abstention class. They never use generating provider, condition, or the GovMem scorer. Exact stratum accounting is fingerprinted in the manifest; holdout label strata are not published because subtraction would leak holdout labels.

`calibration_cases.jsonl` contains calibration gold labels. `locked_holdout_cases.jsonl` contains identities, task/subtype, and immutable prompt/payload hashes only. It contains no gold label, resolution method, abstention class, or primary-judge output. `candidate_execution_plan.jsonl` is also gold-free.

## Candidate prompt and output contract

Candidate prompts are reconstructed at execution from existing private B4 payload locators. No benchmark text or model answer is copied into these public-safe plans. LongMemEval uses the frozen semantic labels `correct`, `incorrect`, `abstention_correct`, `abstention_incorrect`, and `adjudication_required`, plus confidence and answerability basis. Memora evaluates exactly one atomic criterion as `pass`, `fail`, or `adjudication_required`, plus confidence.

Prompts hide the answer-generating provider/model, experiment condition, GovMem score, desired result, aggregate statistics, gold label, and every other judge output. Candidate families supported by the existing executor are: {', '.join(SUPPORTED_PROVIDER_FAMILIES)}. No provider or model is selected by this protocol.

## Precommitted selection rule

A candidate is eligible only if calibration agreement is at least 0.90 overall, 0.90 on LongMemEval, and 0.90 on Memora. Overall agreement of at least 0.95 is preferred when achievable. Selection also reports Cohen's kappa, confusion/FP/FN, LongMemEval abstention-specific accuracy, adjudication-required rate, strict parse-failure rate, and provider/API-failure rate. Lowest price alone cannot select a candidate.

LongMemEval scoring projects `correct` and `abstention_correct` to binary correct, and `incorrect` and `abstention_incorrect` to binary incorrect, while retaining raw abstention labels. The six gold cases whose final raw subtype is unresolved remain usable for binary agreement but are excluded from raw abstention-subtype accuracy. Memora compares pass/fail directly.

## Locked holdout

Candidate selection cannot read holdout labels. This preparation script implements no holdout execution or unlock path. After one candidate is selected and its provider/model/configuration are frozen, a separate explicit command and artifact version are required to evaluate the {h['total']}-case holdout exactly once.

## Execution and durability

Each candidate receives exactly one paid call per calibration case ({c['total']} calls). A future slot ID binds blinded request ID, candidate provider, candidate model, prompt hash, and this protocol version. Candidate ledgers must be append-only under the existing gitignored `results/b4_scoring_validation/private/final_judge_selection/<candidate-id>/` tree. Valid completed slots are skipped; started-only states fail closed; provider/model/hash drift fails closed. `max_attempts=1`; there are no automatic retries.

## Cost scale

Per candidate: {costs['calibration']['calls']} calls, approximately {costs['calibration']['estimated_input_tokens']} input tokens and {costs['calibration']['estimated_output_tokens']} output tokens. Locked holdout after explicit unlock: {costs['holdout']['calls']} calls, approximately {costs['holdout']['estimated_input_tokens']} input tokens and {costs['holdout']['estimated_output_tokens']} output tokens. Tokens use the documented characters/4 planning estimate; no dollar prices are asserted.

No API, network, model, or judge call is made by this preparation script.
"""


def prepare(
    reference_path: Path,
    lme_payload_path: Path,
    memora_payload_path: Path,
    design_manifest_path: Path,
) -> dict[str, Any]:
    design_manifest = json.loads(design_manifest_path.read_text(encoding="utf-8"))
    if design_manifest.get("schema_version") != SOURCE_PROTOCOL_VERSION:
        raise ValueError("final native evaluation protocol version drift")
    gold = load_gold(reference_path)
    payloads, locators = load_payloads(lme_payload_path, memora_payload_path)
    material = materialize_cases(gold, payloads, locators)
    calibration, holdout, accounting = stratified_split(gold)

    calibration_cases = []
    for row in calibration:
        identifier = str(row["blinded_request_id"])
        calibration_cases.append({
            "blinded_request_id": identifier, "task": row["task"], "subtype": row.get("subtype"),
            "gold_binary_label": row["final_reference_binary_label"],
            "gold_raw_label": row.get("final_reference_raw_label"),
            "gold_raw_label_status": "resolved" if row.get("final_reference_raw_label") is not None else "binary_only_raw_subtype_unresolved",
            "resolution_method": row["resolution_method"],
            "prompt_sha256": material[identifier]["prompt_sha256"],
            "payload_sha256": material[identifier]["payload_sha256"],
        })

    locked_holdout = []
    for row in holdout:
        identifier = str(row["blinded_request_id"])
        locked_holdout.append({
            "blinded_request_id": identifier, "task": row["task"], "subtype": row.get("subtype"),
            "prompt_sha256": material[identifier]["prompt_sha256"],
            "payload_sha256": material[identifier]["payload_sha256"],
            "holdout_status": "locked_labels_not_exported",
        })

    execution_plan = []
    for row in calibration:
        identifier = str(row["blinded_request_id"])
        item = material[identifier]
        execution_plan.append({
            "blinded_request_id": identifier, "task": row["task"], "subtype": row.get("subtype"),
            "private_payload_locator": item["payload_locator"], "payload_sha256": item["payload_sha256"],
            "semantic_prompt_sha256": item["prompt_sha256"], "prompt_chars": item["prompt_chars"],
            "estimated_input_tokens_div4": item["estimated_input_tokens_div4"],
            "candidate_binding": {"provider": "REQUIRED_AT_EXECUTION", "model": "REQUIRED_AT_EXECUTION"},
            "request_slot_id_formula": "sha256(blinded_request_id,candidate_provider,candidate_model,semantic_prompt_sha256,protocol_version)",
            "max_attempts": 1, "automatic_retries": False, "execution_status": "planned_not_executed",
        })

    def token_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = [material[str(row["blinded_request_id"])]["estimated_input_tokens_div4"] for row in rows]
        outputs = [160 if row["task"] == "longmemeval" else 96 for row in rows]
        return {
            "calls": len(rows), "estimated_input_tokens": sum(inputs), "estimated_output_tokens": sum(outputs),
            "input_token_distribution": distribution(inputs), "output_estimation": "160 tokens per LME result; 96 per Memora criterion result",
        }

    calibration_ids = [str(row["blinded_request_id"]) for row in calibration]
    holdout_ids = [str(row["blinded_request_id"]) for row in holdout]
    manifest = {
        "schema_version": SCHEMA_VERSION, "source_protocol_version": SOURCE_PROTOCOL_VERSION, "seed": SEED,
        "status": "prepared_offline_candidates_not_selected_holdout_locked", "paid_calls_performed": 0,
        "gold_source": {"path": str(reference_path), "sha256": file_digest(reference_path), "resolved_case_n": len(gold)},
        "private_payload_sources": {
            "longmemeval": {"path": str(lme_payload_path), "sha256": file_digest(lme_payload_path), "case_n": 96},
            "memora": {"path": str(memora_payload_path), "sha256": file_digest(memora_payload_path), "case_n": 508},
        },
        "split": {
            "target": {"calibration": CALIBRATION_TARGET_N, "holdout": EXPECTED_GOLD_N - CALIBRATION_TARGET_N},
            "calibration": {"total": len(calibration), "by_task": task_counts(calibration), "by_subtype": subtype_counts(calibration), "ids_sha256": digest(calibration_ids)},
            "locked_holdout": {"total": len(holdout), "by_task": task_counts(holdout), "by_subtype": subtype_counts(holdout), "ids_sha256": digest(holdout_ids), "gold_labels_exported": False},
            "overlap_n": 0, "stratification": ["task", "memora_subtype", "independent_binary_or_pass_fail_label", "resolution_method", "longmemeval_abstention_class"],
            "excluded_stratifiers": ["generating_provider", "experimental_condition", "govmem_scorer"],
            "accounting": accounting,
        },
        "selection_rules": selection_rules(),
        "candidate_execution": {
            "supported_provider_families": SUPPORTED_PROVIDER_FAMILIES, "provider_model_frozen": False,
            "calls_per_candidate": len(calibration), "one_call_per_case_per_candidate": True,
            "automatic_retries": False, "max_attempts": 1,
            "ledger_policy": "append-only; completed skip; started-only fail closed; provider/model/hash drift fail closed",
            "private_ledger_root": "results/b4_scoring_validation/private/final_judge_selection/<candidate-id>/",
        },
        "cost_scale": {"calibration": token_cost(calibration), "holdout": token_cost(holdout), "pricing": "not estimated; no frozen authoritative model price applies"},
        "artifact_hashes": {
            "calibration_cases_sha256": digest(calibration_cases), "locked_holdout_cases_sha256": digest(locked_holdout),
            "candidate_execution_plan_sha256": digest(execution_plan), "source_design_manifest_sha256": file_digest(design_manifest_path),
        },
        "safety": {"benchmark_text_exported": False, "model_answers_exported": False, "holdout_gold_exported": False, "api_network_model_judge_calls": 0},
    }
    return {"manifest": manifest, "calibration_cases": calibration_cases, "holdout_cases": locked_holdout, "execution_plan": execution_plan}


def write_outputs(output: Path, prepared: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "calibration_cases.jsonl", prepared["calibration_cases"])
    write_jsonl(output / "locked_holdout_cases.jsonl", prepared["holdout_cases"])
    write_jsonl(output / "candidate_execution_plan.jsonl", prepared["execution_plan"])
    write_json(output / "calibration_manifest.json", prepared["manifest"])
    (output / "JUDGE_SELECTION_PROTOCOL.md").write_text(protocol_markdown(prepared["manifest"]), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("results/b4_scoring_validation/private/frontier_adjudication/adjudication_cases.jsonl"))
    parser.add_argument("--longmemeval-payload", type=Path, default=Path("results/b4_scoring_validation/private/longmemeval_judge_execution.jsonl"))
    parser.add_argument("--memora-payload", type=Path, default=Path("results/b4_scoring_validation/private/memora_judge_execution.jsonl"))
    parser.add_argument("--design-manifest", type=Path, default=Path("results/final_native_evaluation_design/evaluation_design_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/final_native_evaluation_design/judge_selection"))
    args = parser.parse_args(argv)
    prepared = prepare(args.reference, args.longmemeval_payload, args.memora_payload, args.design_manifest)
    write_outputs(args.output, prepared)
    split = prepared["manifest"]["split"]
    print(f"TOTAL_GOLD_CASES={split['calibration']['total'] + split['locked_holdout']['total']}")
    print(f"CALIBRATION_CASES={split['calibration']['total']}")
    print(f"LOCKED_HOLDOUT_CASES={split['locked_holdout']['total']}")
    print(f"CALIBRATION_LONGMEMEVAL={split['calibration']['by_task'].get('longmemeval', 0)}")
    print(f"CALIBRATION_MEMORA={split['calibration']['by_task'].get('memora', 0)}")
    print(f"HOLDOUT_LONGMEMEVAL={split['locked_holdout']['by_task'].get('longmemeval', 0)}")
    print(f"HOLDOUT_MEMORA={split['locked_holdout']['by_task'].get('memora', 0)}")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
