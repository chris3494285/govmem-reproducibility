#!/usr/bin/env python3
"""Audit which GovMem paper claims depend on the frozen B4-target scorer."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


AUDIT_VERSION = "b4-paper-scorer-impact-v1"
VERDICT = "MAJOR_RERUN_REQUIRED"
SMALLEST_REMEDIATION = "A"
DEPENDENT = {"scorer_dependent_primary", "scorer_dependent_secondary", "mixed", "unclear"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _claim(
    claim_id: str,
    section: str,
    summary: str,
    rq: str,
    table: str | None,
    metric: str,
    category: str,
    role: str,
    reason: str,
    provenance: list[str],
    threatened: bool,
    action: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "section": section,
        "exact_claim_summary": summary,
        "rq": rq,
        "table_or_figure": table,
        "metric_or_artifact": metric,
        "scorer_dependency_category": category,
        "scientific_role": role,
        "reason": reason,
        "source_code_artifact_provenance": provenance,
        "b4_finding_threatens_claim": threatened,
        "recommended_action": action,
    }


def build_claims() -> list[dict[str, Any]]:
    final_summary = "results/answer_leakage_repair/final_v3_analysis/cross_provider_summary.json"
    propagation = "results/answer_leakage_repair/final_v3_analysis/control_plane_propagation/mechanism_summary.json"
    matched = "results/answer_leakage_repair/final_v3_analysis/matched_fault_support/matched_fault_summary.json"
    evidence = "results/answer_leakage_repair/final_v3_analysis/governance_evidence_audit/governance_evidence_summary.json"
    claims = [
        _claim("cohort-731", "Methods / cohort", "The final cohort contains 731 bases: 191 LongMemEval, 380 Memora, and 160 OAKS Novel.", "RQ-FM", "Cohort table", "base and source counts", "scorer_independent", "primary", "Counts come from the frozen manifest and do not use answer correctness.", ["ARTIFACTS.md", "data/final_main_governance_base_manifest_v2.freeze.json"], False, "Retain; describe as design/provenance, not performance."),
        _claim("conditions-4891", "Methods / cohort", "Each provider has 731 clean, 2,080 corrupted, and 2,080 enforced rows (4,891 total).", "RQ-FM", "Cohort table", "condition counts", "scorer_independent", "primary", "Condition membership and fault support are manifest properties.", ["ARTIFACTS.md", "data/final_main_governance_condition_manifest_v3.freeze.json"], False, "Retain."),
        _claim("governance-evidence-731", "Validity / governance evidence", "All 731 bases have source-supported clean states and assigned faults.", "RQ-FM", "Governance evidence table", "731/731 fully supported", "scorer_independent", "secondary", "The audit explicitly uses frozen source evidence and no provider result.", ["scripts/audit_governance_evidence.py", evidence], False, "Retain with its exact scope: governance evidence, not answer correctness."),
        _claim("admission-change", "Results / control plane", "Fault injection changes the reconstructed admission state in 12,012/12,480 provider-pairs (96.25%).", "RQ-FM", "Control-plane propagation table", "admission_change", "scorer_independent", "primary", "Admission change compares persisted admitted memory-item projections and never reads correctness.", ["scripts/analyze_control_plane_propagation.py", propagation], False, "Retain as a control-plane transition result."),
        _claim("admission-restoration", "Results / control plane", "Enforcement restores admission state in 12,480/12,480 provider-pairs (100%).", "RQ-FM", "Control-plane propagation table", "admission_restoration", "scorer_independent", "primary", "Admission restoration compares manifest-derived/persisted state only.", ["scripts/analyze_control_plane_propagation.py", propagation], False, "Retain; do not describe it as restored answer correctness."),
        _claim("context-change-restoration", "Results / control plane", "Model-visible context changes in 96.25% of pairs and is restored in 100%.", "RQ-FM", "Control-plane propagation table", "context_change/context_restoration", "scorer_independent", "primary", "Canonical model-visible memory/context is compared directly, without answer scoring.", ["scripts/analyze_control_plane_propagation.py", propagation], False, "Retain as context-state evidence."),
        _claim("event-macro-corruption", "Abstract / main results", "Governance faults reduce downstream deterministic correctness by 27.55 percentage points on the event-macro analysis.", "RQ-FM", "Final-main effects table", "event_macro_corruption", "scorer_dependent_primary", "primary", "The effect is corrupted minus clean over primary_score/correctness from the B4-target source-aware scorer.", ["scripts/analyze_final_main_results.py", final_summary], True, "Do not report as semantic correctness until affected outputs receive valid native evaluation."),
        _claim("event-macro-recovery", "Abstract / main results", "Enforcement recovers 27.51 percentage points of downstream deterministic correctness.", "RQ-FM", "Enforcement recovery table", "event_macro_recovery", "scorer_dependent_primary", "primary", "The recovery delta is enforced minus corrupted using the same frozen correctness labels.", ["scripts/analyze_final_main_results.py", final_summary], True, "Remove/qualify, or recompute with valid native evaluation."),
        _claim("condition-accuracy", "Results / model comparison", "Clean, corrupted, and enforced answer accuracy is reported for every provider.", "RQ-FM", "Cross-provider accuracy table", "clean/corrupted/enforced accuracy", "scorer_dependent_primary", "primary", "Every numerator is the frozen Boolean correctness score.", ["scripts/analyze_final_main_results.py", final_summary], True, "Withdraw as semantic accuracy pending source-valid evaluation."),
        _claim("context-stratified-effects", "Results / mechanism", "Corruption is −26.04 pp when context changes and −1.28 pp when context is unchanged.", "RQ-FM", "Control-plane propagation table", "corruption_effect_by_context_change", "mixed", "primary", "The stratum is scorer-independent context state, but the outcome is scorer-dependent correctness delta.", ["scripts/analyze_control_plane_propagation.py", final_summary], True, "Retain context-change counts; remove or reevaluate correctness deltas."),
        _claim("behavioral-propagation", "Abstract / introduction / conclusion", "Governance faults propagate through admission and context construction into downstream answer errors.", "RQ-FM", "Propagation figure/table", "A→C→Y/full propagation", "mixed", "primary", "A and C are directly observed state transitions; Y and full propagation depend on frozen correctness.", ["scripts/analyze_control_plane_propagation.py", propagation, "README.md"], True, "Split the claim: retain A→C; suspend C→incorrect-answer and full A→C→Y claims."),
        _claim("enforcement-correct-behavior", "Abstract / conclusion", "Enforcement restores correct downstream behavior.", "RQ-FM", "Enforcement recovery table", "enforced correctness/recovery", "scorer_dependent_primary", "primary", "Correct downstream behavior is operationalized by the B4-target score, not by admission/context restoration.", ["scripts/analyze_final_main_results.py", final_summary], True, "Replace with 'restores intended admission/context state' unless native answer evaluation is rerun."),
        _claim("fault-breakdowns", "Results / robustness", "Correctness degradation and recovery differ by source, event, and fault operator.", "RQ-FM", "Source/event/fault effect tables", "grouped corruption/recovery", "scorer_dependent_secondary", "secondary", "All grouped effects are functions of frozen correctness labels.", ["scripts/analyze_final_main_results.py", final_summary], True, "Remove or regenerate after valid evaluation."),
        _claim("matched-fault-severity", "Results / robustness", "Matched-fault comparisons cover 3,534 instances, including stale/conflicting 238, stale/missing 351, and stale/flipped 0.", "RQ-FM robustness", "Matched-fault robustness table", "matched correctness-delta contrasts", "mixed", "secondary", "Overlap denominators are manifest-based, but severity, wins, means, and CIs compare scorer-derived correctness harms.", ["scripts/analyze_matched_fault_support.py", matched], True, "Retain overlap/support counts only; withdraw severity/effect conclusions."),
        _claim("provider-performance", "Results / providers", "Six providers differ in clean accuracy and fault-induced correctness effects.", "RQ-FM", "Cross-provider performance table", "provider accuracy/effects", "scorer_dependent_secondary", "secondary", "Provider rankings and differences are computed from frozen correctness.", ["scripts/analyze_final_main_results.py", final_summary], True, "Do not interpret as semantic performance until reevaluated."),
        _claim("scorer-validity", "Methods / validity / B4", "The deterministic source-aware scorer is a valid proxy for semantic/native correctness.", "RQ-B4", None, "B4 scorer agreement", "scorer_dependent_primary", "primary", "LongMemEval agreement is 62.5%; Memora has no valid same-unit estimate; only OAKS resolved parity is established.", ["results/b4_scoring_validation/unit_alignment_audit/", "results/b4_scoring_validation/oaks_validation.json"], True, "State source-specific evidence and explicitly deny a general validity claim."),
        _claim("rq1-pilot", "RQ1 results", "RQ1 reports paired governance-violation effects on a separate frozen pilot cohort.", "RQ1", "RQ1 permissive effects table", "governance_violation", "scorer_independent", "primary", "The RQ1 artifact uses score_governance_fault_rq1.py and separate manifests/results, not the final-main source-aware scorer.", ["scripts/score_governance_fault_rq1.py", "scripts/build_rq1_paper_table.py"], False, "Retain as separate-evaluator evidence; B4 does not validate that evaluator."),
        _claim("rq2-pilot", "RQ2 results", "RQ2 compares matched fault severity using separate RQ1 governance-violation/task-utility outputs.", "RQ2", "RQ2 fault severity tables", "governance_violation/task_utility", "scorer_independent", "primary", "RQ2 consumes separate RQ1 scored artifacts and not final-main correctness.", ["scripts/analyze_rq2_fault_severity.py", "scripts/score_governance_fault_rq1.py"], False, "Retain with a separate evaluator-validity caveat."),
        _claim("rq3-pilot", "RQ3 results", "RQ3 traces admission and context changes to downstream failures in the separate RQ1 pilot.", "RQ3", "RQ3 propagation table", "admission/context/downstream_failure", "scorer_independent", "primary", "The downstream component uses the separate RQ1 governance evaluator, not system_validation_scoring.py.", ["scripts/analyze_rq3_error_propagation.py", "scripts/score_governance_fault_rq1.py"], False, "Retain as outside B4 scope; do not claim B4 validates its downstream evaluator."),
        _claim("rq4-control-plane", "RQ4 results", "RQ4 compares deterministic control-plane enforcement-policy tradeoffs.", "RQ4", "RQ4 policy/tradeoff tables", "unsafe exposure/false rejection/deferred access", "scorer_independent", "primary", "RQ4 evaluates manifest/control-plane policy outcomes without final-main answer correctness.", ["scripts/evaluate_rq4_enforcement_policies.py"], False, "Retain."),
        _claim("system-validation", "System validation", "Cross-system governance and task-utility effects are reported for Mem0, LangMem, and A-Mem.", "System validation", "Cross-system figure", "governance_violation/task_utility_success", "scorer_independent", "secondary", "These use a separate frozen cohort and score_governance_fault_rq1-derived evaluator, not the B4-target scorer.", ["scripts/analyze_system_validation_downstream.py", "scripts/analyze_system_validation_statistics.py"], False, "Retain as separate evidence; do not generalize B4 conclusions to this evaluator."),
        _claim("external-manuscript-wording", "All manuscript sections", "Exact abstract, introduction, hypothesis, results, and conclusion wording.", "All", None, "external manuscript", "unclear", "secondary", "No manuscript/Prism/LaTeX source exists in this repository; exact sentence-level completeness cannot be verified.", ["FINALIZATION_REPORT.md", "results/answer_leakage_repair/final_v3_analysis/paper_update_checklist.md"], True, "Run the generated claim inventory against the external manuscript before submission."),
    ]
    if len({row["claim_id"] for row in claims}) != len(claims):
        raise ValueError("duplicate paper-impact claim ID")
    return claims


def affected_rqs() -> dict[str, Any]:
    rows = [
        {"rq": "RQ-FM", "main_metric": "final-main corruption/recovery and condition accuracy", "uses_deterministic_source_aware_scorer": True, "b4_impact": "Core downstream-quality answer is unsupported across LongMemEval/Memora; control-plane A/C findings remain valid.", "action": "Native reevaluation or remove downstream correctness claims.", "affected": True},
        {"rq": "RQ1", "main_metric": "paired governance-violation delta", "uses_deterministic_source_aware_scorer": False, "b4_impact": "No direct B4 impact; separate deterministic evaluator.", "action": "Keep provenance separation explicit.", "affected": False},
        {"rq": "RQ2", "main_metric": "matched GVR/task-utility fault severity", "uses_deterministic_source_aware_scorer": False, "b4_impact": "No direct B4 impact; separate RQ1 scored artifacts.", "action": "Do not claim B4 validates this evaluator.", "affected": False},
        {"rq": "RQ3", "main_metric": "admission/context/downstream propagation in RQ1 pilot", "uses_deterministic_source_aware_scorer": False, "b4_impact": "No direct B4 impact; downstream metric is from a different scorer.", "action": "Keep separate from final-main A→C→Y claims.", "affected": False},
        {"rq": "RQ4", "main_metric": "control-plane policy tradeoffs", "uses_deterministic_source_aware_scorer": False, "b4_impact": "No direct B4 impact.", "action": "Retain.", "affected": False},
        {"rq": "RQ-B4", "main_metric": "agreement with native/native-style semantic reference", "uses_deterministic_source_aware_scorer": True, "b4_impact": "LongMemEval is weak and Memora is not unit-aligned; a global scorer-validity claim fails.", "action": "Report source-specific evidence only.", "affected": True},
    ]
    return {"schema_version": AUDIT_VERSION, "affected_rq_n": sum(row["affected"] for row in rows), "rows": rows}


def affected_tables_figures() -> dict[str, Any]:
    rows = [
        {"artifact": "Final-main cross-provider accuracy table", "category": "scorer_dependent_primary", "affected": True, "action": "Withdraw/recompute accuracy columns."},
        {"artifact": "Final-main source/event/fault effect tables", "category": "scorer_dependent_primary", "affected": True, "action": "Recompute all correctness deltas."},
        {"artifact": "Enforcement recovery table", "category": "scorer_dependent_primary", "affected": True, "action": "Separate state restoration from correctness recovery."},
        {"artifact": "Control-plane propagation table", "category": "mixed", "affected": True, "action": "Retain admission/context columns; remove/recompute Y and correctness-delta columns."},
        {"artifact": "Matched-fault robustness table", "category": "mixed", "affected": True, "action": "Retain overlap Ns only; remove/recompute score-based severity."},
        {"artifact": "External final-main correctness figures", "category": "unclear", "affected": True, "action": "External manuscript audit required; no generator/source is present."},
        {"artifact": "Governance evidence table", "category": "scorer_independent", "affected": False, "action": "Retain."},
        {"artifact": "RQ1/RQ2/RQ3/RQ4 pilot tables", "category": "scorer_independent", "affected": False, "action": "Retain with separate-evaluator provenance."},
        {"artifact": "Cross-system validation figure", "category": "scorer_independent", "affected": False, "action": "Retain; separate evaluator and experiment."},
    ]
    return {"schema_version": AUDIT_VERSION, "affected_table_figure_n": sum(row["affected"] for row in rows), "rows": rows}


def remediation_options() -> dict[str, Any]:
    return {
        "schema_version": AUDIT_VERSION,
        "smallest_scientifically_defensible": "A",
        "smallest_note": "Option A is the smallest operational remedy but narrows the paper to control-plane/state evidence and removes central downstream-correctness conclusions. To preserve the current downstream-quality thesis, Option D is recommended.",
        "recommended_to_preserve_current_thesis": "D",
        "options": {
            "A": {"name": "Remove or qualify scorer-dependent claims", "sections": ["abstract", "introduction", "methods/scoring", "final-main results", "robustness", "conclusion", "limitations"], "main_experiment_rerun": False, "api_cost_required": False, "affected_outputs_or_cases": "At least 6 affected table/figure families and all final-main correctness claims; retain 12,480 control-plane state comparisons.", "pros": ["No fabricated metric", "Preserves verified cohort/governance/control-plane evidence"], "cons": ["Removes the central downstream answer-quality and correctness-recovery conclusions", "Requires substantial paper reframing"]},
            "B": {"name": "Use the existing semantic validation cohort only", "sections": ["methods/scoring", "B4 validity", "limitations"], "main_experiment_rerun": False, "api_cost_required": False, "affected_outputs_or_cases": "604 semantic cases, but only LongMemEval N=96 is currently same-unit; Memora remains unaligned.", "pros": ["Uses frozen independent judgments", "No new answer generation"], "cons": ["Cannot repair the 29,346-row main accuracy analysis", "Cannot produce a valid Memora accuracy estimate"]},
            "C": {"name": "Design/freeze scorer v3 and rerun affected scoring", "sections": ["methods", "all final-main correctness results", "robustness", "conclusion"], "main_experiment_rerun": "Scoring/analysis only; existing answers can be reused", "api_cost_required": "Potentially no, only if a defensible deterministic scorer can be specified", "affected_outputs_or_cases": "22,626 LongMemEval+Memora condition answers across six providers; OAKS 6,720 rows need not change", "pros": ["Could preserve paired full-cohort design", "No answer-generation rerun"], "cons": ["A defensible deterministic Memora criterion mapping is currently absent", "Must be frozen and independently validated before seeing full results"]},
            "D": {"name": "Use benchmark-native/native-style evaluation for affected datasets", "sections": ["methods", "all final-main correctness results", "robustness", "conclusion", "B4"], "main_experiment_rerun": "Evaluation/analysis only; no answer-generation rerun", "api_cost_required": "Likely for semantic LongMemEval/Memora judging unless human/native tooling is used", "affected_outputs_or_cases": "22,626 existing LongMemEval+Memora answers (6×[1,111+2,660]); retain OAKS native parity", "pros": ["Strongest construct validity", "Directly addresses paraphrase, abstention, and Memora native criteria"], "cons": ["Substantial evaluation cost and adjudication workload", "Requires frozen aggregation and quality-control protocol"]},
        },
    }


def verify_artifacts(root: Path) -> dict[str, Any]:
    propagation = _read_json(root / "results/answer_leakage_repair/final_v3_analysis/control_plane_propagation/mechanism_summary.json")
    matched = _read_json(root / "results/answer_leakage_repair/final_v3_analysis/matched_fault_support/matched_fault_summary.json")
    evidence = _read_json(root / "results/answer_leakage_repair/final_v3_analysis/governance_evidence_audit/governance_evidence_summary.json")
    units = _read_json(root / "results/b4_scoring_validation/unit_alignment_audit/unit_alignment_audit_summary.json")
    overall = propagation["overall"]
    if propagation.get("base_count") != 731 or propagation.get("fault_pair_count") != 2080 or propagation.get("provider_rows") != 12480:
        raise ValueError("control-plane cohort/count drift")
    if overall["admission_change"] != {"n": 12012, "denominator": 12480, "rate": 0.9625}:
        raise ValueError("admission-change result drift")
    if overall["admission_restoration"] != {"n": 12480, "denominator": 12480, "rate": 1.0}:
        raise ValueError("admission-restoration result drift")
    if matched.get("matched_instance_count") != 3534:
        raise ValueError("matched-fault count drift")
    overlaps = {frozenset((row["fault_a"], row["fault_b"])): row["matched_base_count"] for row in matched["overlap_counts"] if "breakdown" not in row}
    if (overlaps[frozenset(("stale", "conflicting"))], overlaps[frozenset(("stale", "flipped"))], overlaps[frozenset(("stale", "missing"))]) != (238, 0, 351):
        raise ValueError("matched-fault overlap drift")
    if evidence["overall"].get("fully_supported_bases") != 731 or evidence["overall"].get("unsupported_bases") != 0:
        raise ValueError("governance evidence result drift")
    if units["longmemeval"] != {"valid_same_unit_n": 96, "agreement_n": 60, "disagreement_n": 36} or units["memora"].get("selected_valid_alignment") != "none":
        raise ValueError("B4 unit-alignment evidence drift")
    provider_counts: dict[str, dict[str, int]] = {}
    for provider in ("openai", "anthropic", "deepseek", "gemini", "kimi", "grok"):
        counts = Counter()
        # Prefer the Tier-B pseudonymous release files (identifier-free); fall
        # back to the original sanitized files on the author's local tree.
        path = root / f"results/final_main_tier_b/{provider}.jsonl"
        if not path.is_file():
            path = root / f"results/final_main_sanitized/{provider}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[str(json.loads(line).get("source_dataset"))] += 1
        if counts != {"longmemeval": 1111, "memora": 2660, "oaks_novel": 1120}:
            raise ValueError(f"source row count drift: {provider}")
        provider_counts[provider] = dict(counts)
    manuscript_tex = [str(path.relative_to(root)) for path in root.rglob("*.tex")]
    return {
        "verified_control_plane": {"bases": 731, "fault_pairs": 2080, "denominator": 12480, "admission_change": 0.9625, "restoration": 1.0},
        "verified_matched_fault": {"instances": 3534, "stale_vs_conflicting": 238, "stale_vs_flipped": 0, "stale_vs_missing": 351},
        "verified_governance_evidence": {"fully_supported": 731, "unsupported": 0},
        "provider_source_row_counts": provider_counts,
        "affected_native_evaluation_rows": 6 * (1111 + 2660),
        "oaks_rows_with_native_parity": 6 * 1120,
        "manuscript_tex_files_present": manuscript_tex,
        "manuscript_available": bool(manuscript_tex),
    }


def write_report(path: Path, claims: list[dict[str, Any]], verification: dict[str, Any], rqs: dict[str, Any], tables: dict[str, Any], remediation: dict[str, Any]) -> None:
    categories = Counter(row["scorer_dependency_category"] for row in claims)
    lines = [
        "# B4 Paper Scorer-Impact Audit",
        "",
        f"**Verdict: `{VERDICT}`.** The current paper synopsis makes downstream answer quality and correctness recovery central, and those results materially depend on a scorer that has weak LongMemEval agreement and no valid Memora same-unit estimate.",
        "",
        "## Scope limitation",
        "",
        "No current manuscript/Prism/LaTeX source is present in this repository. `FINALIZATION_REPORT.md` independently records that omission. This audit therefore covers every substantive claim mapped by `ARTIFACTS.md`, the v3 paper-update checklist, README synopsis/headlines, and generating analyses. Exact external-paper sentence coverage remains an explicit blocker.",
        "",
        "## What survives B4",
        "",
        "- Cohort, source, condition, and fault-support counts.",
        "- Governance evidence: 731/731 supported.",
        "- Admission and model-visible context change: 12,012/12,480 (96.25%).",
        "- Admission and context restoration: 12,480/12,480 (100%).",
        "- Separate RQ1–RQ4 pilot artifacts, which do not consume the B4-target final-main scorer (their own evaluator validity is a separate question).",
        "",
        "## What does not survive as semantic correctness evidence",
        "",
        "- Event-macro corruption (−27.55 pp), recovery (+27.51 pp), condition accuracy, provider comparisons, grouped correctness effects, and correctness-based matched-fault severity.",
        "- The Y/behavioral leg of A→C→Y propagation and claims that enforcement restores correct answers. State restoration alone does not establish correctness restoration.",
        "",
        "## RQ-level impact",
        "",
        "| RQ | Main metric | Uses source-aware scorer? | B4 impact | Action |",
        "|---|---|---:|---|---|",
    ]
    for row in rqs["rows"]:
        lines.append(f"| {row['rq']} | {row['main_metric']} | {'yes' if row['uses_deterministic_source_aware_scorer'] else 'no'} | {row['b4_impact']} | {row['action']} |")
    lines += [
        "",
        "## Classification totals",
        "",
        *[f"- {name}: {categories.get(name, 0)}" for name in ("scorer_independent", "scorer_dependent_primary", "scorer_dependent_secondary", "mixed", "unclear")],
        "",
        "## Remediation",
        "",
        "Option A is the smallest scientifically defensible operation only if the paper is narrowed to control-plane/state-transition evidence. It removes central downstream-correctness conclusions. To preserve the existing downstream-quality thesis, Option D is required: reevaluate the 22,626 existing LongMemEval and Memora answer rows with frozen native/native-style protocols; no answer-generation rerun is needed. OAKS's 6,720 rows retain native option parity.",
        "",
        "## Safety",
        "",
        "This audit is read-only with respect to experimental inputs and made zero API, network, model, judge, scoring, or answer-generation calls.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(root: Path, output_dir: Path) -> dict[str, Any]:
    verification = verify_artifacts(root)
    claims = build_claims()
    rqs = affected_rqs()
    tables = affected_tables_figures()
    remediation = remediation_options()
    independent = [row for row in claims if row["scorer_dependency_category"] == "scorer_independent"]
    dependent = [row for row in claims if row["scorer_dependency_category"] in DEPENDENT]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "scorer_independent_claims.jsonl", independent)
    _write_jsonl(output_dir / "scorer_dependent_claims.jsonl", dependent)
    _write_json(output_dir / "affected_tables_figures.json", tables)
    _write_json(output_dir / "affected_rqs.json", rqs)
    _write_json(output_dir / "remediation_options.json", remediation)
    write_report(output_dir / "PAPER_SCORER_IMPACT_AUDIT.md", claims, verification, rqs, tables, remediation)
    counts = Counter((row["scorer_dependency_category"], row["scientific_role"]) for row in claims)
    result = {
        "paper_verdict": VERDICT,
        "scorer_independent_primary_claims": counts[("scorer_independent", "primary")],
        "scorer_dependent_primary_claims": counts[("scorer_dependent_primary", "primary")],
        "scorer_dependent_secondary_claims": counts[("scorer_dependent_secondary", "secondary")],
        "affected_rqs": rqs["affected_rq_n"],
        "affected_tables_figures": tables["affected_table_figure_n"],
        "smallest_defensible_remediation": SMALLEST_REMEDIATION,
        "verification": verification,
        "claim_category_counts": dict(sorted(Counter(row["scorer_dependency_category"] for row in claims).items())),
        "api_network_model_judge_calls": 0,
    }
    _write_json(output_dir / "paper_impact_summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("results/b4_scoring_validation/paper_impact_audit"))
    args = parser.parse_args(argv)
    result = run_audit(args.repo_root.resolve(), args.output_dir)
    print(f"PAPER_VERDICT={result['paper_verdict']}")
    print(f"SCORER_INDEPENDENT_PRIMARY_CLAIMS={result['scorer_independent_primary_claims']}")
    print(f"SCORER_DEPENDENT_PRIMARY_CLAIMS={result['scorer_dependent_primary_claims']}")
    print(f"SCORER_DEPENDENT_SECONDARY_CLAIMS={result['scorer_dependent_secondary_claims']}")
    print(f"AFFECTED_RQS={result['affected_rqs']}")
    print(f"AFFECTED_TABLES_FIGURES={result['affected_tables_figures']}")
    print(f"SMALLEST_DEFENSIBLE_REMEDIATION={result['smallest_defensible_remediation']}")
    print("PAID_CALLS_PERFORMED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
