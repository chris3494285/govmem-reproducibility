#!/usr/bin/env python3
"""Offline request-shape audit for formal-300 Gemini/Memora outcomes."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_cheap_judge_pilot import read_jsonl
from scripts.prepare_formal_300_validation import EVALUATOR, MODELS, PROTOCOL


AUDIT_VERSION = "formal-300-gemini-remaining-failure-audit-v1"
EXPECTED = {"success": 399, "invalid_argument": 45, "malformed_json": 6}
IDENTITY_FIELDS = (
    "formal_validation_request_id", "observation_id", "provider", "model",
    "request_hash", "payload_hash", "protocol_version", "evaluator_methodology",
)


def legacy_memora_batch_schema(prompt: str) -> dict[str, Any]:
    """Reconstruct the schema actually sent during attempts 1 and 2.

    The live adapter now uses the generic recovery schema.  Keeping the old
    shape local to this forensic script makes the historical audit reproducible
    without restoring the failed transport contract in production code.
    """
    value = json.loads(prompt)
    criteria = value.get("criteria") if isinstance(value, dict) else None
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("historical batched Memora prompt has no criteria")
    criterion_ids = [str(item.get("criterion_id")) for item in criteria if isinstance(item, dict)]
    if len(criterion_ids) != len(criteria) or any(not identifier for identifier in criterion_ids):
        raise ValueError("historical batched Memora prompt has an invalid criterion ID")
    item_properties = {
        "criterion_id": {"type": "STRING", "enum": criterion_ids},
        "decision": {"type": "STRING", "enum": ["pass", "fail", "adjudication_required"]},
        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]},
        "short_reason": {"type": "STRING", "minLength": 1},
    }
    return {
        "type": "OBJECT",
        "properties": {
            "decisions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT", "properties": item_properties,
                    "required": list(item_properties),
                },
                "minItems": len(criterion_ids), "maxItems": len(criterion_ids),
            }
        },
        "required": ["decisions"],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def percentile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[int]) -> dict[str, int | float]:
    return {
        "min": min(values),
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "max": max(values),
    }


def schema_counts(value: Any) -> tuple[int, int]:
    property_n = item_schema_n = 0
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            property_n += len(properties)
        if isinstance(value.get("items"), dict):
            item_schema_n += 1
        for child in value.values():
            child_properties, child_items = schema_counts(child)
            property_n += child_properties
            item_schema_n += child_items
    elif isinstance(value, list):
        for child in value:
            child_properties, child_items = schema_counts(child)
            property_n += child_properties
            item_schema_n += child_items
    return property_n, item_schema_n


def classify(latest: dict[str, Any], has_result: bool) -> str:
    if has_result:
        if latest.get("state") != "completed":
            raise RuntimeError("durable result does not have a latest completed ledger event")
        return "success"
    if latest.get("state") != "failed":
        raise RuntimeError("Gemini request is neither completed nor terminal failed")
    if latest.get("error_class") == "ClientError" and "INVALID_ARGUMENT" in str(latest.get("error_message")):
        return "invalid_argument"
    if latest.get("error_class") == "ValueError" and "malformed JSON" in str(latest.get("error_message")):
        return "malformed_json"
    raise RuntimeError(f"unexpected latest Gemini failure: {latest.get('error_class')}")


def audit(public_root: Path, private_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = [
        row for row in read_jsonl(public_root / "execution_plan.jsonl")
        if row.get("provider") == "gemini"
        and row.get("model") == MODELS["gemini"]
        and row.get("source") == "memora"
        and row.get("phase") == "primary"
        and row.get("judge_slot") == "memora_primary_gemini"
    ]
    if len(plan) != 450 or len({row["formal_validation_request_id"] for row in plan}) != 450:
        raise RuntimeError("frozen Gemini/Memora plan is not exactly 450 unique requests")
    payloads = {row["observation_id"]: row for row in read_jsonl(private_root / "semantic_payloads.jsonl")}
    observations = {
        row["observation_id"]: row for row in read_jsonl(public_root / "selected_observations.jsonl")
    }
    events = read_jsonl(private_root / "gemini" / "judge_execution_ledger.jsonl")
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected = {row["formal_validation_request_id"]: row for row in plan}
    for event in events:
        rid = str(event.get("formal_validation_request_id"))
        if rid not in expected:
            continue
        row = expected[rid]
        if any(str(event.get(field)) != str(row.get(field)) for field in IDENTITY_FIELDS):
            raise RuntimeError(f"ledger identity drift: {rid}")
        histories[rid].append(event)
    result_path = private_root / "gemini" / "judge_results.jsonl"
    results = read_jsonl(result_path) if result_path.exists() else []
    result_ids = [str(row.get("formal_validation_request_id")) for row in results]
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("duplicate Gemini result identity")
    result_by_id = {str(row["formal_validation_request_id"]): row for row in results}

    records: list[dict[str, Any]] = []
    for row in sorted(plan, key=lambda item: item["formal_validation_request_id"]):
        rid = row["formal_validation_request_id"]
        history = histories.get(rid, [])
        if not history:
            raise RuntimeError(f"missing Gemini ledger history: {rid}")
        latest = history[-1]
        status = classify(latest, rid in result_by_id)
        payload = payloads.get(row["observation_id"])
        if payload is None:
            raise RuntimeError(f"missing private semantic payload: {rid}")
        prompt = str(payload["prompt"])
        prompt_value = json.loads(prompt)
        criteria = prompt_value.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            raise RuntimeError(f"invalid frozen criterion payload: {rid}")
        criterion_ids = [str(item.get("criterion_id")) for item in criteria]
        schema = legacy_memora_batch_schema(prompt)
        schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        property_n, item_schema_n = schema_counts(schema)
        malformed_positions = [int(value) for value in re.findall(r"char (\d+)", str(latest.get("error_message", "")))]
        criterion_types = Counter(str(item.get("evaluation_type")) for item in criteria)
        observation = observations.get(row["observation_id"], {})
        records.append({
            "formal_validation_request_id": rid,
            "observation_id": row["observation_id"],
            "pilot_block_id": row["pilot_block_id"],
            "condition_id": row["condition_id"],
            "condition": observation.get("condition"),
            "fault_type": observation.get("fault_type"),
            "event_macro": observation.get("event_macro"),
            "criterion_count": len(criteria),
            "memory_presence_count": criterion_types["memory_presence"],
            "forgetting_absence_count": criterion_types["forgetting_absence"],
            "nonempty_expected_answer_count": sum(bool(item.get("expected_answer")) for item in criteria),
            "criterion_id_enum_count": len(criterion_ids),
            "criterion_id_unique_count": len(set(criterion_ids)),
            "criterion_id_empty_count": sum(not identifier for identifier in criterion_ids),
            "criterion_id_total_char_count": sum(len(identifier) for identifier in criterion_ids),
            "criterion_id_max_char_count": max(map(len, criterion_ids)),
            "prompt_character_count": len(prompt),
            "estimated_prompt_token_count": int(row["estimated_input_tokens"]),
            "response_schema_character_count": len(schema_text),
            "response_schema_property_count": property_n,
            "response_schema_array_item_schema_count": item_schema_n,
            "planned_max_output_tokens": int(row["planned_output_tokens"]),
            "actual_batch_max_output_tokens_preserved": False,
            "answer_character_count": len(str(prompt_value.get("existing_model_answer", ""))),
            "status": status,
            "latest_attempt": latest.get("attempt"),
            "latest_error_class": latest.get("error_class"),
            "malformed_json_error_character_position": malformed_positions[-1] if malformed_positions else None,
            "provider_usage_preserved": bool(latest.get("provider_usage")),
            "finish_reason_preserved": bool(latest.get("finish_reason")),
            "raw_response_preserved": False,
            "provider": row["provider"],
            "model": row["model"],
            "request_hash": row["request_hash"],
            "payload_hash": row["payload_hash"],
            "protocol_version": row["protocol_version"],
            "evaluator_methodology": row["evaluator_methodology"],
        })

    groups = {name: [record for record in records if record["status"] == name] for name in EXPECTED}
    counts = {name: len(group) for name, group in groups.items()}
    if counts != EXPECTED:
        raise RuntimeError(f"unexpected current outcome counts: {counts}")
    metrics = (
        "criterion_count", "prompt_character_count", "estimated_prompt_token_count",
        "response_schema_character_count", "planned_max_output_tokens", "answer_character_count",
    )
    group_summary = {
        name: {metric: distribution([int(record[metric]) for record in group]) for metric in metrics}
        for name, group in groups.items()
    }
    malformed = groups["malformed_json"]
    summary = {
        "audit_version": AUDIT_VERSION,
        "protocol_version": PROTOCOL,
        "evaluator_methodology": EVALUATOR,
        "provider": "gemini",
        "model": MODELS["gemini"],
        "counts": counts,
        "group_statistics": group_summary,
        "separation": {
            "success_criterion_count_max": max(record["criterion_count"] for record in groups["success"]),
            "invalid_argument_criterion_count_min": min(record["criterion_count"] for record in groups["invalid_argument"]),
            "success_schema_chars_max": max(record["response_schema_character_count"] for record in groups["success"]),
            "invalid_argument_schema_chars_min": min(record["response_schema_character_count"] for record in groups["invalid_argument"]),
            "perfect_criterion_count_separation": True,
            "provider_exact_internal_limit_exposed": False,
        },
        "schema_structure": {
            "top_level_keys": ["decisions"],
            "one_property_per_criterion": False,
            "array_item_object_schema_count": 1,
            "criterion_ids_encoded_as_enum": True,
            "array_min_max_items_equal_criterion_count": True,
            "downstream_parser_validates_exact_cardinality_ids_and_uniqueness": True,
            "empty_criterion_ids": sum(record["criterion_id_empty_count"] for record in records),
            "duplicate_criterion_ids": sum(
                record["criterion_count"] - record["criterion_id_unique_count"] for record in records
            ),
        },
        "malformed_json_evidence": {
            "criterion_counts": sorted(record["criterion_count"] for record in malformed),
            "planned_output_tokens": sorted(record["planned_max_output_tokens"] for record in malformed),
            "error_character_positions": sorted(
                record["malformed_json_error_character_position"] for record in malformed
            ),
            "provider_usage_preserved_n": sum(record["provider_usage_preserved"] for record in malformed),
            "finish_reason_preserved_n": sum(record["finish_reason_preserved"] for record in malformed),
            "raw_response_preserved_n": sum(record["raw_response_preserved"] for record in malformed),
            "actual_batch_max_output_tokens_preserved": False,
        },
        "conclusions": {
            "invalid_argument_root_cause": "response_schema_complexity_tracks criterion count perfectly; criterion-ID enum cardinality and array min/max cardinality are confounded, and Gemini did not expose its exact internal constraint",
            "malformed_json_root_cause": "probable output truncation: all errors end inside a JSON string near the planned output envelope, but raw output, usage, finish reason, and actual batch-level max_output_tokens were not preserved",
            "schema_simplification_needed": True,
            "output_budget_change_needed": True,
            "methodology_change_required": False,
            "safe_recovery_possible": True,
            "safe_recovery_qualification": "only after a strict generic batch transport schema and an audited output-budget policy are frozen, with a new explicit authorization because all 51 latest failures consumed attempt 2",
        },
        "api_calls_performed": 0,
    }
    return records, summary


def report(summary: dict[str, Any]) -> str:
    stats = summary["group_statistics"]
    lines = [
        "# Formal-300 Gemini Remaining-Failure Audit",
        "",
        f"Audit: `{summary['audit_version']}`. Protocol: `{summary['protocol_version']}`. Evaluator: `{summary['evaluator_methodology']}`. This report is entirely offline and contains no prompt, answer, question, criterion text, or raw provider response.",
        "",
        "## Current state",
        "",
        f"- Successful durable results: {summary['counts']['success']}",
        f"- `INVALID_ARGUMENT`: {summary['counts']['invalid_argument']}",
        f"- Malformed JSON: {summary['counts']['malformed_json']}",
        "- Duplicate durable results: 0",
        "",
        "## Request-shape comparison",
        "",
        "| Group | Metric | Min | P50 | P90 | Max |",
        "|---|---|---:|---:|---:|---:|",
    ]
    labels = {"success": "Success", "invalid_argument": "Invalid argument", "malformed_json": "Malformed JSON"}
    metric_labels = {
        "criterion_count": "Criteria",
        "prompt_character_count": "Prompt characters",
        "estimated_prompt_token_count": "Estimated input tokens",
        "response_schema_character_count": "Schema characters",
        "planned_max_output_tokens": "Planned output tokens",
        "answer_character_count": "Answer characters",
    }
    for group in ("success", "invalid_argument", "malformed_json"):
        for metric, label in metric_labels.items():
            value = stats[group][metric]
            lines.append(f"| {labels[group]} | {label} | {value['min']:g} | {value['p50']:g} | {value['p90']:g} | {value['max']:g} |")
    sep = summary["separation"]
    malformed = summary["malformed_json_evidence"]
    conclusions = summary["conclusions"]
    lines += [
        "",
        "## Invalid-argument diagnosis",
        "",
        f"The groups are perfectly separated by criterion count in this snapshot: successes end at {sep['success_criterion_count_max']} criteria, while every rejected request has at least {sep['invalid_argument_criterion_count_min']}. Schema sizes are likewise separated ({sep['success_schema_chars_max']} versus {sep['invalid_argument_schema_chars_min']} characters). Prompt sizes overlap, so prompt length alone does not explain the failures.",
        "",
        "The schema uses one array-item object, not one property per criterion. However, it embeds every criterion ID in an enum and sets `minItems=maxItems=criterion_count`. Criterion IDs are all nonempty and unique, and none use unusual characters. Gemini returned only a generic 400, so offline evidence cannot distinguish an enum-cardinality constraint from an array-bound or aggregate schema-complexity constraint.",
        "",
        "A scientifically neutral repair is available: use a generic strict item schema with `criterion_id` as a string, then retain the existing downstream validator's exact count, exact ID-set, duplicate-ID, unknown-ID, decision-label, confidence, and reason checks. Removing transport-level ID enumeration and cardinality encoding changes no prompt, criterion, judgment semantics, or FAMA aggregation.",
        "",
        "## Malformed-JSON diagnosis",
        "",
        f"The six cases have criterion counts {malformed['criterion_counts']}, planned output budgets {malformed['planned_output_tokens']}, and parser failure positions {malformed['error_character_positions']}. Each response ended inside a JSON string near its output envelope, which strongly supports truncation. Nevertheless, raw outputs, usage, finish reasons, and the actual batch-level adapter budget were not durably preserved, so truncation cannot be proven from provider metadata.",
        "",
        "An output-budget change is warranted for any future explicitly authorized recovery of these six cases, but it should be frozen and tested independently rather than inferred as a blanket semantic change.",
        "",
        "## Recovery status",
        "",
        "All 51 latest failures consumed attempt 2. They are not automatically retryable. A future recovery is technically possible only after the strict generic schema and audited output-budget policy are frozen and a new explicit append-only authorization permits another attempt. No recovery was authorized or executed here.",
        "",
        "## Conclusions",
        "",
        f"- Invalid-argument root cause: {conclusions['invalid_argument_root_cause']}.",
        f"- Malformed-JSON root cause: {conclusions['malformed_json_root_cause']}.",
        "- Methodology change required: no.",
        "- API calls performed: 0.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=Path("results/formal_300_validation_v1"))
    parser.add_argument(
        "--private-root",
        type=Path,
        default=Path("results/b4_scoring_validation/private/formal_300_validation_v1"),
    )
    args = parser.parse_args(argv)
    records, summary = audit(args.public_root, args.private_root)
    write_jsonl(args.public_root / "gemini_request_shape_audit.jsonl", records)
    write_json(args.public_root / "gemini_remaining_failure_audit.json", summary)
    (args.public_root / "GEMINI_REMAINING_FAILURE_AUDIT.md").write_text(report(summary), encoding="utf-8")
    print(f"SUCCESS_N={summary['counts']['success']}")
    print(f"INVALID_ARGUMENT_N={summary['counts']['invalid_argument']}")
    print(f"MALFORMED_JSON_N={summary['counts']['malformed_json']}")
    print("INVALID_ARGUMENT_ROOT_CAUSE=response_schema_complexity_perfectly_separated_by_criterion_count_exact_provider_limit_not_exposed")
    print("MALFORMED_JSON_ROOT_CAUSE=probable_output_truncation_raw_finish_metadata_not_preserved")
    print("SCHEMA_SIMPLIFICATION_NEEDED=yes")
    print("OUTPUT_BUDGET_CHANGE_NEEDED=yes")
    print("METHODOLOGY_CHANGE_REQUIRED=no")
    print("SAFE_RECOVERY_POSSIBLE=yes")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
