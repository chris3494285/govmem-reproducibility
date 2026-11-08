#!/usr/bin/env python3
"""Combine frozen B4 primary and third-judge decisions entirely offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b4_calibration_execution import LedgerConflictError, prompt_hash
from scripts.prepare_b4_frontier_escalation import PERSISTENT_FAILURE_ID
from scripts.prepare_b4_private_payloads import read_jsonl

EXPECTED_TOTAL = 604
EXPECTED_LME = 96
EXPECTED_MEMORA = 508
EXPECTED_ESCALATION = 24
EXPECTED_DISAGREEMENTS = 23
EXPECTED_PERSISTENT = 1

LME_BINARY_PROJECTION = {
    "correct": "correct",
    "abstention_correct": "correct",
    "incorrect": "incorrect",
    "abstention_incorrect": "incorrect",
    "adjudication_required": None,
    None: None,
}
MEMORA_BINARY_PROJECTION = {
    "pass": "pass",
    "fail": "fail",
    "adjudication_required": None,
    None: None,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_binary(task: str, raw_decision: str | None) -> str | None:
    if task == "longmemeval":
        if raw_decision not in LME_BINARY_PROJECTION:
            raise ValueError(f"unknown LongMemEval decision: {raw_decision}")
        return LME_BINARY_PROJECTION[raw_decision]
    if task == "memora":
        if raw_decision not in MEMORA_BINARY_PROJECTION:
            raise ValueError(f"unknown Memora decision: {raw_decision}")
        return MEMORA_BINARY_PROJECTION[raw_decision]
    raise ValueError(f"unknown task: {task}")


def _raw_majority(*labels: str | None) -> str | None:
    counts = Counter(label for label in labels if label is not None)
    return counts.most_common(1)[0][0] if counts and counts.most_common(1)[0][1] >= 2 else None


def adjudicate_case(
    *,
    blinded_request_id: str,
    task: str,
    subtype: str | None,
    gpt_raw: str | None,
    claude_raw: str | None,
    gemini_raw: str | None,
    escalation_reason: str | None,
) -> dict[str, Any]:
    """Apply the predeclared label-combination rules without semantic reinterpretation."""
    gpt_binary = project_binary(task, gpt_raw)
    claude_binary = project_binary(task, claude_raw)
    gemini_binary = project_binary(task, gemini_raw)
    status = "resolved"
    binary: str | None = None
    raw: str | None = None
    method: str | None = None
    human = False
    reason: str

    if gpt_raw is not None and claude_raw is not None and gpt_raw == claude_raw:
        if gpt_binary is None:
            status, human = "unresolved_pending_human", True
            reason = "primary_exact_label_has_no_binary_projection"
        else:
            binary, raw = gpt_binary, gpt_raw
            method = "primary_exact_agreement"
            reason = "primary_raw_labels_identical_and_binary_projectable"
    elif gpt_raw is not None and claude_raw is not None:
        if task == "longmemeval" and gpt_binary is not None and gpt_binary == claude_binary:
            binary = gpt_binary
            method = "primary_binary_agreement_raw_subtype_disagreement"
            reason = "primary_raw_labels_differ_but_binary_correctness_projection_matches"
        elif gpt_binary is not None and claude_binary is not None and gpt_binary != claude_binary:
            if gemini_binary in {gpt_binary, claude_binary}:
                binary = gemini_binary
                raw = _raw_majority(gpt_raw, claude_raw, gemini_raw)
                method = "third_judge_binary_majority" if task == "longmemeval" else "third_judge_majority"
                reason = "third_judge_projection_matches_exactly_one_primary_projection"
            else:
                status, human = "unresolved_pending_human", True
                reason = "primary_binary_disagreement_without_usable_third_judge_majority"
        else:
            status, human = "unresolved_pending_human", True
            reason = "primary_disagreement_contains_unprojectable_label"
    else:
        surviving_raw = gpt_raw if gpt_raw is not None else claude_raw
        surviving_binary = gpt_binary if gpt_raw is not None else claude_binary
        if surviving_binary is not None and gemini_binary is not None and surviving_binary == gemini_binary:
            binary = surviving_binary
            raw = surviving_raw if surviving_raw == gemini_raw else None
            method = "surviving_primary_plus_third_agreement"
            reason = "available_primary_and_third_judge_binary_projections_match"
        else:
            status, human = "unresolved_pending_human", True
            reason = "incomplete_primary_pair_not_resolved_by_available_primary_and_third_judge"

    if human:
        binary = None
        raw = None
        method = "human_review_required"

    comparison = None
    if escalation_reason is not None:
        comparison = {
            "gemini_agrees_with_gpt_raw": gemini_raw is not None and gemini_raw == gpt_raw,
            "gemini_agrees_with_claude_raw": gemini_raw is not None and gemini_raw == claude_raw,
            "gemini_agrees_with_gpt_binary": gemini_binary is not None and gemini_binary == gpt_binary,
            "gemini_agrees_with_claude_binary": gemini_binary is not None and gemini_binary == claude_binary,
            "neither_or_unresolved": gemini_binary is None or (
                gemini_binary != gpt_binary and gemini_binary != claude_binary
            ),
        }

    return {
        "blinded_request_id": blinded_request_id,
        "task": task,
        "subtype": subtype,
        "primary_gpt_raw_decision": gpt_raw,
        "primary_claude_raw_decision": claude_raw,
        "gemini_raw_decision": gemini_raw,
        "primary_gpt_binary_projection": gpt_binary,
        "primary_claude_binary_projection": claude_binary,
        "gemini_binary_projection": gemini_binary,
        "escalation_reason": escalation_reason,
        "final_reference_status": status,
        "final_reference_binary_label": binary,
        "final_reference_raw_label": raw,
        "resolution_method": method,
        "human_review_required": human,
        "reason": reason,
        "escalation_comparison": comparison,
    }


def _latest_by_slot(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        latest[str(event["request_slot_id"])] = event
    return latest


def _completed_decision(event: dict[str, Any] | None) -> str | None:
    if not event or event.get("state") != "completed":
        return None
    result = event.get("parsed_result")
    if not isinstance(result, dict):
        raise LedgerConflictError("completed judge event lacks parsed_result")
    decision = result.get("decision")
    if not isinstance(decision, str):
        raise LedgerConflictError("completed judge event lacks decision")
    return decision


def verify_prompt_plan(
    execution_plan: list[dict[str, Any]], prompt_plan: list[dict[str, Any]]
) -> None:
    if len(execution_plan) != len(prompt_plan):
        raise LedgerConflictError("execution/prompt plan row count mismatch")
    by_key = {
        (str(row["blinded_request_id"]), str(row["judge_slot"])): row
        for row in execution_plan
    }
    if len(by_key) != len(execution_plan):
        raise LedgerConflictError("duplicate semantic request slots in execution plan")
    for prompt_row in prompt_plan:
        key = (str(prompt_row["blinded_request_id"]), str(prompt_row["judge_slot"]))
        plan_row = by_key.get(key)
        if plan_row is None:
            raise LedgerConflictError(f"prompt has no execution slot: {key}")
        if prompt_hash(str(prompt_row["prompt"])) != plan_row.get("request_hash"):
            raise LedgerConflictError(f"frozen prompt hash drift: {key}")


def analyze_adjudication(
    primary_plan: list[dict[str, Any]],
    primary_events: list[dict[str, Any]],
    escalation_manifest: dict[str, Any],
    escalation_plan: list[dict[str, Any]],
    escalation_events: list[dict[str, Any]],
    *,
    enforce_frozen_counts: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_latest = _latest_by_slot(primary_events)
    escalation_latest = _latest_by_slot(escalation_events)
    primary_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary_plan:
        primary_by_case[str(row["blinded_request_id"])].append(row)
    escalation_by_case: dict[str, dict[str, Any]] = {}
    for row in escalation_plan:
        case_id = str(row["blinded_request_id"])
        if case_id in escalation_by_case:
            raise LedgerConflictError(f"duplicate escalation case: {case_id}")
        escalation_by_case[case_id] = row
    manifest_cases = {
        str(row["blinded_request_id"]): row for row in escalation_manifest.get("cases", [])
    }
    if set(manifest_cases) != set(escalation_by_case):
        raise LedgerConflictError("escalation manifest/plan cohort mismatch")

    cases: list[dict[str, Any]] = []
    for case_id, slots in sorted(primary_by_case.items()):
        if len(slots) != 2 or {row["judge_slot"] for row in slots} != {"A", "B"}:
            raise LedgerConflictError(f"primary case lacks exact A/B slots: {case_id}")
        by_judge = {str(row["judge_slot"]): row for row in slots}
        gpt = _completed_decision(primary_latest.get(str(by_judge["A"]["request_slot_id"])))
        claude = _completed_decision(primary_latest.get(str(by_judge["B"]["request_slot_id"])))
        escalation_row = escalation_by_case.get(case_id)
        gemini = None
        escalation_reason = None
        if escalation_row is not None:
            if escalation_row.get("judge_provider") != "gemini" or escalation_row.get("judge_model") != "gemini-3-flash-preview":
                raise LedgerConflictError(f"third-judge provider/model drift: {case_id}")
            gemini = _completed_decision(escalation_latest.get(str(escalation_row["request_slot_id"])))
            escalation_reason = manifest_cases[case_id].get("escalation_reason")
        case = adjudicate_case(
            blinded_request_id=case_id,
            task=str(slots[0]["task"]),
            subtype=slots[0].get("subtype"),
            gpt_raw=gpt,
            claude_raw=claude,
            gemini_raw=gemini,
            escalation_reason=escalation_reason,
        )
        cases.append(case)

    escalation_case_ids = {row["blinded_request_id"] for row in cases if row["escalation_reason"] is not None}
    if escalation_case_ids != set(escalation_by_case):
        raise LedgerConflictError("escalation cases are not a subset of the primary cohort")
    summary = summarize_cases(cases)
    if enforce_frozen_counts:
        expected = {
            "total_cases": EXPECTED_TOTAL,
            "longmemeval": EXPECTED_LME,
            "memora": EXPECTED_MEMORA,
            "escalation": EXPECTED_ESCALATION,
            "disagreements": EXPECTED_DISAGREEMENTS,
            "persistent": EXPECTED_PERSISTENT,
        }
        observed = {
            "total_cases": len(cases),
            "longmemeval": sum(row["task"] == "longmemeval" for row in cases),
            "memora": sum(row["task"] == "memora" for row in cases),
            "escalation": len(escalation_case_ids),
            "disagreements": sum(row["escalation_reason"] == "primary_judge_disagreement" for row in cases),
            "persistent": sum(row["escalation_reason"] == "persistent_primary_execution_failure" for row in cases),
        }
        if observed != expected:
            raise LedgerConflictError(f"frozen adjudication population mismatch: {observed}")
        persistent = next(row for row in cases if row["blinded_request_id"] == PERSISTENT_FAILURE_ID)
        if not persistent["human_review_required"]:
            raise LedgerConflictError("known persistent primary/Gemini conflict must require human review")
    return cases, summary


def apply_human_adjudications(
    cases: list[dict[str, Any]],
    adjudications: list[dict[str, Any]],
    *,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """Apply explicit human decisions only to cases already requiring review."""
    if expected_count is not None and len(adjudications) != expected_count:
        raise LedgerConflictError(
            f"expected exactly {expected_count} human adjudication(s), found {len(adjudications)}"
        )
    ids = [str(row.get("blinded_request_id", "")) for row in adjudications]
    if len(ids) != len(set(ids)):
        raise LedgerConflictError("duplicate human adjudication")
    by_id = {str(row["blinded_request_id"]): row for row in cases}
    updated = [dict(row) for row in cases]
    updated_by_id = {str(row["blinded_request_id"]): row for row in updated}
    required = {
        "blinded_request_id", "task", "final_reference_binary_label",
        "final_reference_raw_label", "resolution_method", "adjudicator_type",
        "adjudication_version", "reason", "evidence_locator",
        "primary_gpt_raw_decision", "primary_claude_raw_decision", "gemini_raw_decision",
    }
    for adjudication in adjudications:
        missing = sorted(required - set(adjudication))
        if missing:
            raise LedgerConflictError(f"human adjudication missing fields: {missing}")
        case_id = str(adjudication["blinded_request_id"])
        original = by_id.get(case_id)
        if original is None:
            raise LedgerConflictError(f"human adjudication references unknown case: {case_id}")
        if not original.get("human_review_required"):
            raise LedgerConflictError(f"human adjudication targets already-resolved case: {case_id}")
        if adjudication["task"] != original["task"]:
            raise LedgerConflictError(f"human adjudication task drift: {case_id}")
        if adjudication["adjudicator_type"] != "human" or adjudication["resolution_method"] != "human_adjudication":
            raise LedgerConflictError(f"invalid human adjudication provenance: {case_id}")
        for source_field in (
            "primary_gpt_raw_decision", "primary_claude_raw_decision", "gemini_raw_decision"
        ):
            if adjudication[source_field] != original[source_field]:
                raise LedgerConflictError(f"human adjudication raw-label drift for {source_field}: {case_id}")
        raw_label = adjudication["final_reference_raw_label"]
        binary_label = adjudication["final_reference_binary_label"]
        try:
            projected = project_binary(str(original["task"]), raw_label)
        except ValueError as exc:
            raise LedgerConflictError(f"invalid human raw label: {case_id}") from exc
        if projected is None or projected != binary_label:
            raise LedgerConflictError(f"incompatible human binary/raw labels: {case_id}")
        if not isinstance(adjudication["reason"], str) or not adjudication["reason"].strip():
            raise LedgerConflictError(f"human adjudication reason is empty: {case_id}")
        if not adjudication["evidence_locator"]:
            raise LedgerConflictError(f"human adjudication evidence locator is empty: {case_id}")
        target = updated_by_id[case_id]
        target["pre_human_review_reason"] = target["reason"]
        target["final_reference_status"] = "resolved"
        target["final_reference_binary_label"] = binary_label
        target["final_reference_raw_label"] = raw_label
        target["resolution_method"] = "human_adjudication"
        target["human_review_required"] = False
        target["reason"] = adjudication["reason"]
        target["human_adjudication"] = {
            "adjudicator_type": adjudication["adjudicator_type"],
            "adjudication_version": adjudication["adjudication_version"],
            "evidence_locator": adjudication["evidence_locator"],
        }
    return updated


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    methods = Counter(row["resolution_method"] for row in cases if row["resolution_method"])
    primary_exact = methods["primary_exact_agreement"]
    binary_only = methods["primary_binary_agreement_raw_subtype_disagreement"]
    third_majority = methods["third_judge_majority"] + methods["third_judge_binary_majority"]
    surviving = methods["surviving_primary_plus_third_agreement"]
    human_adjudication = methods["human_adjudication"]
    human = sum(bool(row["human_review_required"]) for row in cases)

    def task_summary(task: str) -> dict[str, Any]:
        rows = [row for row in cases if row["task"] == task]
        exact = sum(row["resolution_method"] == "primary_exact_agreement" for row in rows)
        disagreements = sum(
            row["primary_gpt_raw_decision"] is not None
            and row["primary_claude_raw_decision"] is not None
            and row["primary_gpt_raw_decision"] != row["primary_claude_raw_decision"]
            for row in rows
        )
        base = {
            "total": len(rows),
            "primary_exact_agreements": exact,
            "primary_raw_disagreements": disagreements,
            "human_review_cases": sum(bool(row["human_review_required"]) for row in rows),
        }
        if task == "longmemeval":
            base.update({
                "binary_only_agreements_despite_raw_disagreement": sum(
                    row["resolution_method"] == "primary_binary_agreement_raw_subtype_disagreement" for row in rows
                ),
                "true_binary_disagreements_requiring_gemini": sum(
                    row["escalation_reason"] == "primary_judge_disagreement"
                    and row["primary_gpt_binary_projection"] != row["primary_claude_binary_projection"]
                    for row in rows
                ),
                "gemini_majority_resolutions": sum(row["resolution_method"] == "third_judge_binary_majority" for row in rows),
                "persistent_failure_cases": sum(row["escalation_reason"] == "persistent_primary_execution_failure" for row in rows),
            })
        else:
            base.update({
                "gemini_majority_resolutions": sum(row["resolution_method"] == "third_judge_majority" for row in rows),
                "subtype_counts": dict(sorted(Counter(str(row.get("subtype") or "none") for row in rows).items())),
            })
        return base

    escalated = [row for row in cases if row["escalation_reason"] is not None]
    comparisons = {
        key: sum(bool(row["escalation_comparison"][key]) for row in escalated)
        for key in (
            "gemini_agrees_with_gpt_raw",
            "gemini_agrees_with_claude_raw",
            "gemini_agrees_with_gpt_binary",
            "gemini_agrees_with_claude_binary",
            "neither_or_unresolved",
        )
    }
    return {
        "schema_version": "b4-frontier-adjudication-summary-v1",
        "overall": {
            "total_cases": len(cases),
            "resolved_binary_cases": len(cases) - human,
            "unresolved_human_review_cases": human,
            "primary_exact_agreement_cases": primary_exact,
            "primary_binary_agreement_raw_subtype_disagreement_cases": binary_only,
            "third_judge_majority_cases": third_majority,
            "surviving_primary_plus_third_agreement_cases": surviving,
            "human_adjudication_cases": human_adjudication,
            "human_review_required_cases": human,
        },
        "longmemeval": task_summary("longmemeval"),
        "memora": task_summary("memora"),
        "escalation_comparison_counts": comparisons,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    lme = summary["longmemeval"]
    memora = summary["memora"]
    comparisons = summary["escalation_comparison_counts"]
    lines = [
        "# B4 Frontier Adjudication Report",
        "",
        "The automated layer combines frozen judge labels only and does not inspect answer text or benchmark context. Any human decision is loaded from the separate provenance artifact.",
        "",
        "## Overall",
        "",
        f"- total cases: {overall['total_cases']}",
        f"- resolved binary cases: {overall['resolved_binary_cases']}",
        f"- human review required: {overall['human_review_required_cases']}",
        f"- primary exact agreements: {overall['primary_exact_agreement_cases']}",
        f"- primary binary-only agreements with raw subtype disagreement: {overall['primary_binary_agreement_raw_subtype_disagreement_cases']}",
        f"- third-judge majority resolutions: {overall['third_judge_majority_cases']}",
        f"- surviving-primary plus third-judge agreements: {overall['surviving_primary_plus_third_agreement_cases']}",
        f"- human adjudications: {overall['human_adjudication_cases']}",
        "",
        "## LongMemEval",
        "",
        f"- total: {lme['total']}",
        f"- primary exact agreements: {lme['primary_exact_agreements']}",
        f"- raw disagreements: {lme['primary_raw_disagreements']}",
        f"- binary-only agreements despite raw disagreement: {lme['binary_only_agreements_despite_raw_disagreement']}",
        f"- true binary disagreements requiring Gemini: {lme['true_binary_disagreements_requiring_gemini']}",
        f"- Gemini majority resolutions: {lme['gemini_majority_resolutions']}",
        f"- persistent primary failures: {lme['persistent_failure_cases']}",
        f"- human-review cases: {lme['human_review_cases']}",
        "",
        "LongMemEval binary projection: `correct` and `abstention_correct` map to correct; `incorrect` and `abstention_incorrect` map to incorrect. Raw labels remain preserved.",
        "",
        "## Memora",
        "",
        f"- total: {memora['total']}",
        f"- primary agreements: {memora['primary_exact_agreements']}",
        f"- primary disagreements: {memora['primary_raw_disagreements']}",
        f"- Gemini majority resolutions: {memora['gemini_majority_resolutions']}",
        f"- human-review cases: {memora['human_review_cases']}",
        "",
        "## Escalation descriptive comparisons",
        "",
        *[f"- {key}: {value}" for key, value in comparisons.items()],
        "",
        "These descriptive counts do not override the adjudication rules.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, default=Path("results/b4_scoring_validation/private/full_frontier_judges"))
    parser.add_argument("--escalation-dir", type=Path, default=Path("results/b4_scoring_validation/private/frontier_escalation"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/b4_scoring_validation/private/frontier_adjudication"))
    parser.add_argument("--human-adjudications", type=Path)
    args = parser.parse_args(argv)

    primary_manifest_path = args.primary_dir / "full_validation_manifest.json"
    primary_prompt_path = args.primary_dir / "full_validation_prompt_plan.jsonl"
    primary_plan_path = args.primary_dir / "full_validation_execution_plan.jsonl"
    primary_ledger_path = args.primary_dir / "judge_execution_ledger.jsonl"
    escalation_manifest_path = args.escalation_dir / "escalation_manifest.json"
    escalation_prompt_path = args.escalation_dir / "escalation_prompt_plan.jsonl"
    escalation_plan_path = args.escalation_dir / "escalation_execution_plan.jsonl"
    escalation_ledger_path = args.escalation_dir / "judge_execution_ledger.jsonl"

    primary_manifest = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    primary_plan = read_jsonl(primary_plan_path)
    primary_prompts = read_jsonl(primary_prompt_path)
    primary_events = read_jsonl(primary_ledger_path)
    escalation_manifest = json.loads(escalation_manifest_path.read_text(encoding="utf-8"))
    escalation_plan = read_jsonl(escalation_plan_path)
    escalation_prompts = read_jsonl(escalation_prompt_path)
    escalation_events = read_jsonl(escalation_ledger_path)
    verify_prompt_plan(primary_plan, primary_prompts)
    verify_prompt_plan(escalation_plan, escalation_prompts)

    if primary_manifest.get("phase") != "full_frontier_judges":
        raise SystemExit("primary phase drift")
    if escalation_manifest.get("third_judge_provider") != "gemini" or escalation_manifest.get("third_judge_model") != "gemini-3-flash-preview":
        raise SystemExit("escalation provider/model drift")
    cases, summary = analyze_adjudication(
        primary_plan, primary_events, escalation_manifest, escalation_plan, escalation_events,
        enforce_frozen_counts=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    human_adjudication_path = args.human_adjudications or args.output_dir / "human_adjudications.jsonl"
    human_adjudications = read_jsonl(human_adjudication_path) if human_adjudication_path.exists() else []
    if human_adjudication_path.exists():
        cases = apply_human_adjudications(cases, human_adjudications, expected_count=1)
        summary = summarize_cases(cases)
    manifest = {
        "schema_version": "b4-frontier-adjudication-manifest-v1",
        "status": "frozen_complete_independent_reference" if not summary["overall"]["human_review_required_cases"] else "frozen_offline_reference_with_human_review_pending",
        "source_primary_phase": "full_frontier_judges",
        "source_escalation_phase": "frontier_escalation",
        "source_fingerprints": {
            "primary_manifest_sha256": file_sha256(primary_manifest_path),
            "primary_prompt_plan_sha256": file_sha256(primary_prompt_path),
            "primary_execution_plan_sha256": file_sha256(primary_plan_path),
            "primary_ledger_sha256": file_sha256(primary_ledger_path),
            "escalation_manifest_sha256": file_sha256(escalation_manifest_path),
            "escalation_prompt_plan_sha256": file_sha256(escalation_prompt_path),
            "escalation_execution_plan_sha256": file_sha256(escalation_plan_path),
            "escalation_ledger_sha256": file_sha256(escalation_ledger_path),
            "human_adjudications_sha256": file_sha256(human_adjudication_path) if human_adjudications else None,
        },
        "binary_projection": {
            "longmemeval": {
                "correct": "correct",
                "abstention_correct": "correct",
                "incorrect": "incorrect",
                "abstention_incorrect": "incorrect",
                "adjudication_required": "unresolved",
                "missing/provider_error": "unresolved",
            },
            "memora": {
                "pass": "pass",
                "fail": "fail",
                "adjudication_required": "unresolved",
                "missing/provider_error": "unresolved",
            },
        },
        "total_cases": len(cases),
        "human_review_required": summary["overall"]["human_review_required_cases"],
        "human_adjudication_count": summary["overall"]["human_adjudication_cases"],
        "automated_analysis_inspected_semantic_text": False,
        "api_calls_made": 0,
    }
    _write_json(args.output_dir / "adjudication_manifest.json", manifest)
    _write_jsonl(args.output_dir / "adjudication_cases.jsonl", cases)
    _write_json(args.output_dir / "adjudication_summary.json", summary)
    _write_report(args.output_dir / "ADJUDICATION_REPORT.md", summary)
    human_rows = [{
        "blinded_request_id": row["blinded_request_id"],
        "task": row["task"],
        "subtype": row["subtype"],
        "primary_gpt_raw_decision": row["primary_gpt_raw_decision"],
        "primary_claude_raw_decision": row["primary_claude_raw_decision"],
        "gemini_raw_decision": row["gemini_raw_decision"],
        "reason": row["reason"],
        "primary_payload_locator": f"{primary_prompt_path}::{row['blinded_request_id']}",
        "escalation_payload_locator": f"{escalation_prompt_path}::{row['blinded_request_id']}",
    } for row in cases if row["human_review_required"]]
    _write_jsonl(args.output_dir / "human_review_cases.jsonl", human_rows)

    overall = summary["overall"]
    print(f"TOTAL_CASES={overall['total_cases']}")
    print(f"RESOLVED_CASES={overall['resolved_binary_cases']}")
    print(f"HUMAN_REVIEW_REQUIRED={overall['human_review_required_cases']}")
    print(f"PRIMARY_EXACT_AGREEMENT={overall['primary_exact_agreement_cases']}")
    print(f"PRIMARY_BINARY_ONLY_AGREEMENT={overall['primary_binary_agreement_raw_subtype_disagreement_cases']}")
    print(f"THIRD_JUDGE_MAJORITY={overall['third_judge_majority_cases']}")
    print(f"HUMAN_ADJUDICATION={overall['human_adjudication_cases']}")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
