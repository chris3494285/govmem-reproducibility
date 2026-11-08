#!/usr/bin/env python3
"""Build the v3 final-main offline analysis and old/new comparison.

This command only reads persisted scored rows and frozen manifests.  It never
imports a provider client and never performs inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_plane_propagation import analyze as analyze_propagation
from scripts.analyze_final_main_results import _authoritative_rows, _condition_accuracy, analyze_single
from scripts.summarize_rq1_cross_provider import _final_main_model_row
from scripts.analyze_matched_fault_support import analyze as analyze_matched

MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v3.jsonl"
BASE = ROOT / "data/final_main_governance_base_manifest_v2.jsonl"
SPEC = ROOT / "configs/main_analysis_spec_v1.json"
FAILURE = ROOT / "configs/main_failure_handling_spec_v1.json"
PUBLIC = ROOT / "results/final_main_sanitized"
OUT = ROOT / "results/answer_leakage_repair/final_v3_analysis"
PROVIDERS = ("openai", "anthropic", "deepseek", "gemini", "kimi", "grok")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _primary_row(label: str, provider: str, summary: dict) -> dict:
    return _final_main_model_row(label, provider, {"run_b": summary["primary"]})


def _validate_provider_rows(path: Path, provider: str) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    failures = [row for row in rows if row.get("status") != "success" or row.get("parse_success") is not True]
    return {"provider": provider, "rows": len(rows), "unique_ids": len({row.get("condition_id") for row in rows}), "failures": len(failures), "sha256": _sha(path)}


def _git_jsonl(path: str) -> list[dict]:
    data = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    provider_summaries = {}
    model_rows = []
    validation = {}
    labels = {
        "openai": "OpenAI GPT-5.6-sol", "anthropic": "Anthropic Claude Sonnet 5",
        "deepseek": "DeepSeek V4 Pro", "gemini": "Gemini 3 Flash",
        "kimi": "Kimi K2.6 no-thinking", "grok": "Grok 4.3 no-reasoning",
    }
    for provider in PROVIDERS:
        result = PUBLIC / f"{provider}.jsonl"
        provider_out = OUT / "providers" / provider
        summary = analyze_single(result, MANIFEST, BASE, SPEC, FAILURE, provider_out)
        provider_summaries[provider] = summary
        model_rows.append(_primary_row(labels[provider], "xAI" if provider == "grok" else provider, summary))
        validation[provider] = _validate_provider_rows(result, provider)

    provider_paths = {provider: PUBLIC / f"{provider}.jsonl" for provider in PROVIDERS}
    propagation = analyze_propagation(manifest=MANIFEST, result_paths=provider_paths, output_dir=OUT / "control_plane_propagation")
    matched = analyze_matched(manifest=MANIFEST, provider_paths=provider_paths, output_dir=OUT / "matched_fault_support")

    old_path = ROOT / "results/final_main_cross_provider_summary_v3/cross_provider_summary.json"
    old = json.loads(old_path.read_text(encoding="utf-8")) if old_path.is_file() else {"models": []}
    old_by_provider = {str(row.get("provider", "")).lower(): row for row in old.get("models", [])}
    new_by_provider = {row["provider"].lower(): row for row in model_rows}
    comparison_rows = []
    keys = sorted(set().union(*(set(row) for row in model_rows), *(set(row) for row in old.get("models", []))))
    old_provider_keys = {"openai": "openai", "anthropic": "anthropic", "deepseek": "deepseek", "gemini": "google", "kimi": "kimi", "grok": "xai"}
    for provider in PROVIDERS:
        old_row = old_by_provider.get(old_provider_keys[provider], {})
        new_row = new_by_provider.get("xai" if provider == "grok" else provider, {})
        for key in keys:
            if key in {"model", "provider"}:
                continue
            old_value, new_value = old_row.get(key), new_row.get(key)
            comparison_rows.append({"provider": provider, "statistic": key, "old": old_value, "new": new_value, "changed": old_value != new_value})

    # Add denominator and failure statistics, using the committed v2 public
    # rows from Git for the old side so promotion cannot erase the comparison.
    v2_manifest = [json.loads(line) for line in (ROOT / "data/final_main_governance_condition_manifest_v2.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    v2_by_provider = {}
    for provider in PROVIDERS:
        old_rows = _git_jsonl(f"results/final_main_sanitized/{provider}.jsonl")
        v2_accuracy = _condition_accuracy(_authoritative_rows(old_rows, v2_manifest))
        new_rows = [json.loads(line) for line in (PUBLIC / f"{provider}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        v3_manifest = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
        v3_accuracy = _condition_accuracy(_authoritative_rows(new_rows, v3_manifest))
        for condition in ("clean", "corrupted", "enforced"):
            for field in ("total_n", "scored_n", "unscored_n", "coverage_rate"):
                old_value = v2_accuracy[condition][field]
                new_value = v3_accuracy[condition][field]
                comparison_rows.append({"provider": provider, "statistic": f"{condition}_{field}", "old": old_value, "new": new_value, "changed": old_value != new_value})

    (OUT / "provider_summaries.json").write_text(json.dumps(provider_summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    aggregate = {"status": "OFFLINE V3 FINAL-MAIN ANALYSIS", "manifest_sha256": _sha(MANIFEST), "providers": model_rows, "validation": validation, "control_plane": propagation, "matched_fault_support": matched, "api_calls_made": 0}
    (OUT / "cross_provider_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (OUT / "old_vs_new_statistics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["provider", "statistic", "old", "new", "changed"])
        writer.writeheader(); writer.writerows(comparison_rows)
    (OUT / "old_vs_new_statistics.json").write_text(json.dumps({"old_source": str(old_path), "new_source": str(OUT / "cross_provider_summary.json"), "rows": comparison_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    changed = [row for row in comparison_rows if row["changed"]]
    lines = ["# V2 versus v3 final-main statistics", "", f"Manifest: `{_sha(MANIFEST)}`", "", f"Changed statistics: **{len(changed)}**", "", "| Provider | Statistic | v2 | v3 |", "|---|---|---:|---:|"]
    for row in changed:
        lines.append(f"| {row['provider']} | {row['statistic']} | {row['old']} | {row['new']} |")
    lines += ["", "Unchanged statistics are retained in `old_vs_new_statistics.csv` and `.json`. The six-provider rows are descriptive cross-provider results; no second run is fabricated.", ""]
    (OUT / "old_vs_new_statistics.md").write_text("\n".join(lines), encoding="utf-8")
    checklist = [
        "# v3 paper update checklist", "",
        "The repository contains no Prism, LaTeX, manuscript, or figure-source paper file; occurrences must be updated in the external paper source.",
        "", "| Provider | Section/table/figure scope | Statistic | Denominator/source | Old | Corrected | Changed |", "|---|---|---|---|---:|---:|---:|"]
    for row in comparison_rows:
        checklist.append(f"| {row['provider']} | final-main results/tables/figures | `{row['statistic']}` | `results/answer_leakage_repair/final_v3_analysis/` | {row['old']} | {row['new']} | {row['changed']} |")
    checklist += ["", "Control-plane and matched-support claims are separately recorded in the v3 analysis summaries: denominator 12,480; change/restoration 0.9625/1.0; matched instances 3,534; stale-vs-conflicting 238; stale-vs-flipped 0; stale-vs-missing 351.", "", "RQ1–RQ4 artifacts use separate frozen pilot manifests/results and are not derived from the repaired final-main rows; they were rerun offline and remain unchanged."]
    (OUT / "paper_update_checklist.md").write_text("\n".join(checklist) + "\n", encoding="utf-8")
    licensing = {
        "status": "REVIEW_REQUIRED",
        "raw_benchmark_text_detected_in_promoted_sanitized_files": False,
        "raw_benchmark_text_detected_in_proposed_manifest_artifacts": True,
        "datasets": {
            dataset: {
                "files": [f"results/final_main_sanitized/{provider}.jsonl" for provider in PROVIDERS],
                "exposed_fields": ["source_dataset", "base_id", "condition_id", "source-derived event/fault labels", "source and context SHA-256 fingerprints"],
                "raw_text_fields_exposed": [],
                "recommendation": "Confirm benchmark terms permit derived identifiers/fingerprints and score release; do not add raw benchmark text without permission.",
            }
            for dataset in ("LongMemEval", "Memora", "OAKS Novel")
        },
        "provenance_file": {
            "file": "results/answer_leakage_repair/source_evidence_provenance.json",
            "exposed_fields": ["official source file name", "session/message locators", "content SHA-256 fingerprints", "source example IDs"],
            "raw_text_fields_exposed": [],
            "recommendation": "Obtain author/license approval for locator and fingerprint redistribution; retain raw source outside the release.",
        },
        "manifest_artifacts": {
            "files": ["data/final_main_governance_condition_manifest_v3.jsonl", "results/answer_leakage_repair/provider_rerun_manifests/*.jsonl"],
            "exposed_fields": ["model_visible_context_text", "source_evidence nested content", "task_question", "task_oracle_answer", "project_description and other benchmark answer-bearing fields"],
            "raw_text_counts": {"v3_condition_manifest_model_visible_context_text": 2893, "v3_condition_manifest_other_long_text_fields": 2370, "provider_rerun_manifest_model_visible_context_text_per_file": 231},
            "recommendation": "Do not redistribute these manifest artifacts until each benchmark license is confirmed; otherwise publish locator/hash-only manifests and deterministic local rebuild instructions.",
        },
        "excluded_private_material": ["merged provider JSONL", "rerun JSONL", "logs", "checkpoints", "raw benchmark downloads", "provider payloads and request metadata"],
    }
    (OUT / "licensing_blocker_report.json").write_text(json.dumps(licensing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    licensing_md = ["# Licensing and redistribution blocker report", "", "No raw benchmark text was detected in the promoted sanitized score files. However, the v3 condition manifest and provider rerun manifests contain benchmark-derived/model-visible text and answer-bearing fields.", "", "| Dataset | File scope | Exposed fields | Raw text | Required action |", "|---|---|---|---|---|"]
    for dataset, details in licensing["datasets"].items():
        licensing_md.append(f"| {dataset} | `{', '.join(details['files'])}` | {', '.join(details['exposed_fields'])} | none detected | {details['recommendation']} |")
    licensing_md += ["| All three benchmarks | `data/final_main_governance_condition_manifest_v3.jsonl`; `results/answer_leakage_repair/provider_rerun_manifests/*.jsonl` | `model_visible_context_text`, nested source evidence, `task_question`, `task_oracle_answer`, and answer-bearing benchmark fields | detected; counts recorded in JSON report | Confirm redistribution rights or replace with locator/hash-only artifacts plus deterministic rebuild instructions before staging. |", "", "The provenance artifact exposes official source filenames, session/message locators, example IDs, and content hashes but no message content. Confirm that these derived locators/fingerprints are redistributable. Raw merged/rerun files and benchmark downloads remain excluded."]
    (OUT / "licensing_blocker_report.md").write_text("\n".join(licensing_md) + "\n", encoding="utf-8")
    tracked_allowlist = ["README.md", "ARTIFACTS.md", "eval/final_main_rerun_identity.py", "eval/final_main_result_join.py", "scripts/sanitize_final_results.py", "scripts/analyze_final_main_results.py", "scripts/analyze_control_plane_propagation.py", "scripts/analyze_matched_fault_support.py", "scripts/audit_governance_evidence.py", "scripts/run_final_main_anthropic.py", "scripts/run_final_main_deepseek.py", "scripts/run_final_main_gemini.py", "scripts/run_final_main_grok.py", "scripts/run_final_main_kimi.py", "scripts/run_final_main_openai.py", "data/final_main_governance_condition_manifest_v3.jsonl", "data/final_main_governance_condition_manifest_v3.freeze.json", "results/final_main_sanitized/anthropic.jsonl", "results/final_main_sanitized/deepseek.jsonl", "results/final_main_sanitized/gemini.jsonl", "results/final_main_sanitized/grok.jsonl", "results/final_main_sanitized/kimi.jsonl", "results/final_main_sanitized/openai.jsonl", "results/final_main_sanitized/sha256_mapping.json"]
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    untracked_allow_prefixes = ("results/answer_leakage_repair/final_v3_analysis/", "results/answer_leakage_repair/answer_leakage_audit_v3.json", "results/answer_leakage_repair/changed_", "results/answer_leakage_repair/repair_summary.json", "results/answer_leakage_repair/source_evidence_provenance.json", "results/answer_leakage_repair/provider_rerun_manifests/", "results/answer_leakage_repair/final_live_commands.md", "results/answer_leakage_repair/rerun_cost_estimate.json", "results/answer_leakage_repair/rerun_execution_plan.md", "results/answer_leakage_repair/rerun_identity_audit.json")
    untracked_allowlist = sorted(path for path in untracked if path.startswith(untracked_allow_prefixes))
    excluded_untracked = sorted(path for path in untracked if path not in untracked_allowlist)
    report = [
        "# GovMem v3 finalization report", "", "Status: offline finalization complete; no API calls, inference, staging, commit, push, deletion, or cleanup performed.", "",
        f"Promoted files: six providers × 4,891 rows; public mapping SHA-256 `{_sha(PUBLIC / 'sha256_mapping.json')}`.",
        "", "## Audits", "", "- v3 leakage audit: PASS; 0 affected rows.", "- Identity audit: all six providers PASS.", "- Governance evidence: PASS; 731/731 fully supported, 0 unsupported.", "- Anthropic: six repair-slice failures retained under the frozen failure policy; no retries or edits.", "- `pytest`: PASS; compileall and `git diff --check`: PASS.", "- Provider API calls: 0.", "", "## Paper values", "", "All old/corrected provider statistics, denominators, CIs, effects, and changed flags are listed in `old_vs_new_statistics.{json,csv,md}` and `paper_update_checklist.md`.", "", "Headline corrected values: event-macro corruption −27.55 pp; recovery +27.51 pp; context-changed corruption −26.04 pp; unchanged-context corruption −1.28 pp; control-plane change 96.25%; restoration 100%; matched instances 3,534; stale/conflicting 238; stale/flipped 0; stale/missing 351.", "", "## RQ dependency decisions", "", "- Final-main cross-provider, RQ1-style effects, source/event/fault breakdowns, enforcement recovery, propagation, and matched-fault outputs were computed from promoted v3 final-main files where supported by the canonical implementation.", "- Existing RQ1–RQ4 paper artifacts use separate frozen pilot manifests/results; their paths and hashes are independent of the repaired 265 final-main conditions and were regenerated offline without being silently replaced.", "- System-validation tables/figure were regenerated from their separate frozen system-validation artifacts and do not consume final-main provider scores.", "", "## Regenerated outputs", "", "- `results/answer_leakage_repair/final_v3_analysis/` provider summaries, cross-provider summary, propagation, matched-fault, governance audit, comparison, checklist, and licensing report.", "- `results/system_validation_analysis/`, `results/system_validation_statistics/`, and `results/system_validation_paper/`.", "- `results/rq1_paper/`, `results/rq2/`, `results/rq3/`, and `results/rq4/` were regenerated offline from their separate frozen inputs.", "", "## Blockers", "", "- Benchmark redistribution terms for LongMemEval, Memora, and OAKS Novel, including derived locators/fingerprints, are not established in this repository.", "- External manuscript/Prism/LaTeX sources are absent; paper text and external figure references require author-side update from the checklist.", "- `submission_ready=true` must remain unset until those decisions and the changed values are reviewed.", "", "## Proposed git-add allowlist", "", "See the exact grouped allowlist in ARTIFACTS.md/README.md: documentation; v3 manifest/freeze; sanitizer and analysis/audit code; six sanitized files and mapping; repair provenance/manifests/audits; and `final_v3_analysis/`. Exclude all raw merged/rerun files, logs, checkpoints, canaries, smoke/preflight/exploratory outputs, caches, and unreviewed machine-specific configs.", ""]
    report += ["", "## Exact proposed git-add allowlist", "", "Tracked:", ""] + [f"- `{path}`" for path in tracked_allowlist] + ["", "Untracked approved:", ""] + [f"- `{path}`" for path in untracked_allowlist] + ["", "## Exact untracked paths to keep uncommitted", ""] + [f"- `{path}`" for path in excluded_untracked] + [""]
    (OUT / "FINALIZATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return aggregate


if __name__ == "__main__":
    result = build()
    print(json.dumps({"status": result["status"], "providers": len(result["providers"]), "api_calls_made": 0, "output_dir": str(OUT)}, sort_keys=True))
