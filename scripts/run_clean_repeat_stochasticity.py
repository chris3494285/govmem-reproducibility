#!/usr/bin/env python3
"""CLEAN-A / CLEAN-B repeated-generation stochasticity control.

Phase 1 (preflight/repair) builds, validates, and costs the repeat-generation
experiment.  Phase 2 (generate) produces the CLEAN-B answers.  Phase 3 (judge)
scores only the CLEAN-B answers with the frozen formal-300 evaluator routes and
adapters.

Paid phases (generate, judge) are always fail-closed: they require ``--execute``,
judging additionally requires ``--confirm-paid-judge-calls``, and a bounded
``--max-requests`` count.  ``--dry-run`` never performs a paid call.  No phase
ever regenerates CLEAN-A, regenerates CLEAN-B, resamples the frozen blocks, or
mutates frozen formal-300 artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.finalize_formal_300_validation import human_decision_map, resolve
from scripts.prepare_cheap_judge_pilot import read_jsonl
from scripts.run_formal_300_validation import result_index

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results/formal_300_validation_v1"
PRIVATE = ROOT / "results/b4_scoring_validation/private/formal_300_validation_v1"
SANITIZED = ROOT / "results/final_main_sanitized"
OUT = ROOT / "results/stochasticity_control/clean_repeat_v1"
LEDGER_NAME = "generation_execution_ledger.jsonl"
RESULTS_NAME = "generation_results.jsonl"
DIAGNOSTICS_NAME = "generation_diagnostics.jsonl"
JUDGING_DIRNAME = "judging"
JUDGE_MODELS = {"openai": "gpt-5.6-sol", "anthropic": "claude-sonnet-5", "gemini": "gemini-3-flash-preview"}
# Frozen formal-300 planned output budgets: LongMemEval primary judging plans
# 160 output tokens; Memora plans ``40 + 36 * criterion_n`` for the single
# batched criterion request.  These are the *planned* (stored) budgets; the
# frozen transport reliability floors are applied separately at execution.
JUDGE_LME_PLANNED_OUTPUT_TOKENS = 160
JUDGE_MEMORA_PLANNED_OUTPUT_BASE = 40
JUDGE_MEMORA_PLANNED_OUTPUT_PER_CRITERION = 36
JUDGE_OUTPUT_BUDGET_MODES = ("formal300", "planned")
# Explicit, transport-only output-budget overrides.  ``none`` keeps the frozen
# formal-300 plan/floor budgets; ``gemini_memora_transport`` raises the Gemini
# Memora budget to the frozen formal-300 transport recovery budget
# ``attempt3_output_budget`` (= min(6144, max(1024, 256 + 80*n))).  This never
# changes the prompt, criteria, schema, model, or request identity.
JUDGE_OUTPUT_BUDGET_OVERRIDES = ("none", "gemini_memora_transport")
# Static base judge plan: LongMemEval dual primaries, Memora Gemini primary.
JUDGE_SLOTS = {
    "longmemeval": (("lme_primary_openai", "openai"), ("lme_primary_anthropic", "anthropic")),
    "memora": (("memora_primary_gemini", "gemini"),),
}
JUDGE_PROTOCOL = "clean-repeat-v1"
# Ledger identity fields that must be identical across every event of a request.
JUDGE_IDENTITY_FIELDS = (
    "judge_request_id", "repeat_block_id", "judge_slot", "provider", "model", "source",
    "prompt_sha256", "clean_A_observation_id", "formal300_block_id", "clean_B_answer_sha256",
)
JUDGE_TERMINAL_STATES = ("completed", "failed")
JUDGE_RECONCILIATION_NAME = "judge_reconciliation.json"
# Optional offline recovery artifact: one JSONL row per failed request carrying
# the raw model text, so a malformed/empty failure can be re-parsed without a
# new paid call when the transport already returned a usable payload.
JUDGE_RECOVERY_ARTIFACT_NAME = "judge_recovery_candidates.jsonl"
# Frozen formal-300 LongMemEval route classes.  Only ``semantic_judge_required``
# observations receive paid semantic judging; every other class is resolved
# offline by the frozen deterministic routing implementation.
LME_SEMANTIC_ROUTE = "semantic_judge_required"
OFFLINE_DETERMINISTIC_RESULTS_NAME = "deterministic_abstention_results.jsonl"
# Frozen LongMemEval native source used by formal-300.  Only referenced to rebuild
# the frozen judge prompt for observations that have no stored semantic payload
# (the frozen native-deterministic route); see ``clean_b_judge_prompt``.
LME_SOURCE_DEFAULT = Path("$LONGMEMEVAL_SOURCE/longmemeval-cleaned/longmemeval_oracle.json")
_LME_SOURCE_CACHE: dict[str, dict[str, Any]] = {}

SEED = 20261013
PER_PROVIDER = 10
PER_SOURCE_PER_PROVIDER = 5
PROVIDERS = ("openai", "anthropic", "gemini", "kimi", "xai", "deepseek")
SOURCES = ("longmemeval", "memora")
SANITIZED_FILE = {"openai": "openai", "anthropic": "anthropic", "gemini": "gemini",
                  "kimi": "kimi", "xai": "grok", "deepseek": "deepseek"}
EVALUATOR_ROUTE = {
    "longmemeval": "dual_primary:openai_gpt-5.6-sol+anthropic_claude-sonnet-5;"
                   "binary_agreement_resolves;gemini_escalation_on_disagreement_or_nonbinary",
    "memora": "gemini_criterion_primary;openai_secondary_when_frozen_protocol_requires;"
              "anthropic_escalation_when_required;human_adjudication_only_if_protocol_reaches_it",
}
FROZEN_HASHES = {
    "selected_observations": "results/formal_300_validation_v1/selected_observations.jsonl",
    "deterministic_results": "results/formal_300_validation_v1/deterministic_results.jsonl",
    "execution_plan": "results/formal_300_validation_v1/execution_plan.jsonl",
    "semantic_payloads": "results/b4_scoring_validation/private/formal_300_validation_v1/semantic_payloads.jsonl",
}


def _resolved_observations() -> list[dict[str, Any]]:
    """Resolve formal-300 observations with the frozen final evaluator (offline)."""
    observations = read_jsonl(PUBLIC / "selected_observations.jsonl")
    deterministic = read_jsonl(PUBLIC / "deterministic_results.jsonl")
    plan = read_jsonl(PUBLIC / "execution_plan.jsonl")
    dynamic = PRIVATE / "dynamic_execution_plan.jsonl"
    if dynamic.exists():
        plan = plan + read_jsonl(dynamic)
    payloads = {r["observation_id"]: r for r in read_jsonl(PRIVATE / "semantic_payloads.jsonl")}
    resolved, _counters = resolve(
        observations, deterministic, payloads, result_index(PRIVATE), plan, human_decision_map(PRIVATE)
    )
    return resolved


def select_blocks(resolved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic stratified selection that never reads the correctness value.

    Only the resolution *status* (score present) and the frozen semantic route
    are used; correctness True/False never influences the choice.
    """
    eligible = [
        r for r in resolved
        if r["condition"] == "clean"
        and r["route"] == "semantic_judge_required"
        and r["formal_score"] is not None
    ]
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_stratum[(row["generating_provider"], row["source"])].append(row)
    selected: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        for source in SOURCES:
            pool = sorted(by_stratum.get((provider, source), []),
                          key=lambda r: (r["condition_id"], r["pilot_block_id"]))
            rng = random.Random(f"{SEED}:{provider}:{source}")
            rng.shuffle(pool)
            selected.extend(pool[:PER_SOURCE_PER_PROVIDER])
    return selected


def _sanitized_index() -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for provider, filename in SANITIZED_FILE.items():
        for row in read_jsonl(SANITIZED / f"{filename}.jsonl"):
            index[(provider, row["condition_id"])] = row
    return index


def build_manifest(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join frozen references and clean-A correctness (after selection)."""
    payloads = {r["observation_id"]: r for r in read_jsonl(PRIVATE / "semantic_payloads.jsonl")}
    conditions = _condition_rows()
    sanitized = _sanitized_index()
    manifest = []
    for index, row in enumerate(selected, 1):
        observation_id = row["observation_id"]
        payload = payloads.get(observation_id)
        condition = conditions.get(row["condition_id"], {})
        frozen_row = sanitized.get((row["generating_provider"], row["condition_id"]), {})
        prompt = json.loads(payload["prompt"]) if payload else {}
        try:
            reconstructed = frozen_prompt(row["condition_id"], conditions)
            verifiable = _prompt_hash(reconstructed) == frozen_row.get("prompt_sha256")
            block_reason = None if verifiable else "frozen_prompt_hash_mismatch"
        except RuntimeError as exc:
            verifiable, block_reason = False, str(exc)
        manifest.append({
            "repeat_block_id": f"clean-repeat-v1::{row['generating_provider']}::{row['source']}::{index:02d}",
            "formal300_block_id": row["pilot_block_id"],
            "source": row["source"],
            "base_id": condition.get("base_id"),
            "selected_fault": row.get("fault_type"),
            "generating_provider": row["generating_provider"],
            "generating_model": frozen_row.get("model"),
            "clean_condition_id": row["condition_id"],
            "clean_A_observation_id": observation_id,
            "clean_A_route": row["route"],
            "clean_A_generation_ref": {
                "artifact": FROZEN_HASHES["semantic_payloads"] if row["route"] == "semantic_judge_required"
                            else FROZEN_HASHES["deterministic_results"],
                "observation_id": observation_id,
                "field": "prompt.existing_model_answer (blinded clean-A answer seen by the evaluator)"
                         if row["route"] == "semantic_judge_required"
                         else "native-deterministic route: answer text not stored; correctness from deterministic_results.jsonl",
                "answer_sha256": row.get("answer_sha256"),
            },
            "clean_A_correctness": row["formal_score"],
            "clean_A_correctness_joined_after_selection": True,
            "clean_input_ref": {
                "condition_manifest": "data/final_main_governance_condition_manifest_v2.jsonl",
                "condition_id": row["condition_id"],
                "model_visible_context_text_sha256": frozen_row.get("model_visible_context_sha256"),
                "generation_prompt_sha256": frozen_row.get("prompt_sha256"),
                "manifest_sha256": frozen_row.get("manifest_sha256"),
            },
            "inference_config_ref": {
                "provider_runner": f"scripts/run_final_main_{SANITIZED_FILE[row['generating_provider']]}.py",
                "model_parameters": frozen_row.get("model_parameters"),
                "clean_A_input_tokens": frozen_row.get("input_tokens"),
                "clean_A_output_tokens": frozen_row.get("output_tokens"),
            },
            "clean_B_status": "pending",
            "clean_B_input_verifiable": verifiable,
            "clean_B_block_reason": block_reason,
            "evaluator_route_expected": EVALUATOR_ROUTE[row["source"]],
        })
    return manifest


def expected_calls_and_cost(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    lme = [m for m in manifest if m["source"] == "longmemeval"]
    memora = [m for m in manifest if m["source"] == "memora"]
    # Observed frozen rates used for expected case loads (no arbitrary thresholds).
    lme_escalation_rate = 9 / 218
    memora_secondary_rate = 6 / 450
    memora_escalation_rate = 2 / 450
    generation_calls = len(manifest)
    gen_input = sum(int(m["inference_config_ref"]["clean_A_input_tokens"] or 0) for m in manifest)
    gen_output = sum(int(m["inference_config_ref"]["clean_A_output_tokens"] or 0) for m in manifest)
    judge_base = 2 * len(lme) + len(memora)
    judge_expected = (
        judge_base
        + round(len(lme) * lme_escalation_rate)
        + round(len(memora) * memora_secondary_rate)
        + round(len(memora) * memora_escalation_rate)
    )
    judge_max = judge_base + len(lme) + 2 * len(memora)

    # Artifact-derived judge token estimates: judge input prompt = frozen semantic
    # payload prompt with the CLEAN-B answer substituted; judge output = the frozen
    # execution-plan output budget for the corresponding judge slot.
    payloads = {r["observation_id"]: r for r in read_jsonl(PRIVATE / "semantic_payloads.jsonl")}
    plan = read_jsonl(PUBLIC / "execution_plan.jsonl")
    dynamic = PRIVATE / "dynamic_execution_plan.jsonl"
    if dynamic.exists():
        plan = plan + read_jsonl(dynamic)
    planned_out = {(r["observation_id"], r["judge_slot"]): int(r["planned_output_tokens"]) for r in plan}
    judge_input = 0
    judge_output = 0.0
    for m in manifest:
        prompt = payloads.get(m["clean_A_observation_id"], {}).get("prompt", "")
        per_call_in = math.ceil(len(prompt) / 4)
        if m["source"] == "longmemeval":
            calls = 2 + lme_escalation_rate
            out = 160 * calls
        else:
            calls = 1 + memora_secondary_rate + memora_escalation_rate
            out = planned_out.get((m["clean_A_observation_id"], "memora_primary_gemini"), 256) * calls
        judge_input += per_call_in * calls
        judge_output += out
    return {
        "generation_calls": generation_calls,
        "generation_calls_by_provider": dict(sorted(Counter(m["generating_provider"] for m in manifest).items())),
        "expected_generation_input_tokens": gen_input,
        "expected_generation_output_tokens": gen_output,
        "generation_output_token_conservative": sum(
            int((m["inference_config_ref"]["model_parameters"] or {}).get("max_output_tokens") or 0) for m in manifest
        ),
        "judge_calls_base": judge_base,
        "judge_calls_expected": judge_expected,
        "judge_calls_max": judge_max,
        "judge_call_rates_used": {
            "lme_gemini_escalation": lme_escalation_rate,
            "memora_openai_secondary": memora_secondary_rate,
            "memora_anthropic_escalation": memora_escalation_rate,
            "source": "frozen formal-300 routing counts",
        },
        "judge_input_token_estimate_method": "ceil(frozen CLEAN-A semantic payload prompt chars / 4) per expected judge call",
        "judge_output_token_estimate_method": "frozen execution-plan planned_output_tokens per expected judge call (LME 160; Memora Gemini primary per plan)",
        "judge_input_tokens_expected": int(round(judge_input)),
        "judge_output_tokens_expected": int(round(judge_output)),
        "pricing": {
            "available_locally": False,
            "note": "configs/deepseek_v4_flash_pricing_v1.json has null price fields; no usable local pricing metadata, so dollar costs are not invented.",
        },
        "api_calls_performed": 0,
    }


def write_preflight(manifest: list[dict[str, Any]], cost: dict[str, Any], resolved: list[dict[str, Any]],
                    out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(m, sort_keys=True) + "\n" for m in manifest), encoding="utf-8")
    summary = manifest_summary(manifest, resolved)
    (out / "manifest_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "expected_calls_and_cost.json").write_text(json.dumps(cost, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "preflight_report.md").write_text(_preflight_md(summary, cost, manifest), encoding="utf-8")
    (out / "analysis_plan.md").write_text(_analysis_plan_md(), encoding="utf-8")


def manifest_summary(manifest: list[dict[str, Any]], resolved: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter(m["source"] for m in manifest)
    by_provider = Counter(m["generating_provider"] for m in manifest)
    by_stratum = Counter((m["generating_provider"], m["source"]) for m in manifest)
    clean_all = [r for r in resolved if r["condition"] == "clean"]
    unresolved_clean = sum(1 for r in clean_all if r["formal_score"] is None)
    deterministic_clean = sum(1 for r in clean_all if r["route"] != "semantic_judge_required")
    return {
        "seed": SEED,
        "blocks": len(manifest),
        "longmemeval": by_source["longmemeval"],
        "memora": by_source["memora"],
        "by_generating_provider": dict(sorted(by_provider.items())),
        "by_provider_source": {f"{p}::{s}": n for (p, s), n in sorted(by_stratum.items())},
        "eligible_clean_resolved_semantic": len(
            [r for r in clean_all if r["route"] == "semantic_judge_required" and r["formal_score"] is not None]),
        "clean_observations_total": len(clean_all),
        "clean_observations_resolved": len(clean_all) - unresolved_clean,
        "clean_observations_deterministic_route": deterministic_clean,
        "selection_conditioned_on_correctness": False,
        "duplicate_formal300_blocks": len(manifest) - len({m["formal300_block_id"] for m in manifest}),
        "unique_clean_condition_ids": len({m["clean_condition_id"] for m in manifest}),
        "clean_B_input_verifiable": sum(1 for m in manifest if m.get("clean_B_input_verifiable")),
        "clean_B_input_blocked_unverifiable": sum(1 for m in manifest if not m.get("clean_B_input_verifiable")),
        "api_calls_performed": 0,
    }


def _replacement_rank(provider: str, row: dict[str, Any]) -> str:
    """Deterministic, seed-derived ordering key (independent of Python hashing)."""
    key = f"{SEED}:replacement:{provider}:{row['condition_id']}:{row['pilot_block_id']}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def repair_manifest(out: Path, resolved: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace only the fail-closed rows with verifiable same-stratum alternates.

    Selection uses frozen identifiers and a seed-derived rank only; the
    correctness value is never read during selection.
    """
    manifest_path = out / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit("manifest missing; run --phase preflight first")
    manifest = read_jsonl(manifest_path)
    ledger = out / LEDGER_NAME
    completed = {e["repeat_block_id"] for e in (read_jsonl(ledger) if ledger.exists() else [])
                 if e.get("state") == "completed"}
    blocked = [m for m in manifest if not m.get("clean_B_input_verifiable")]
    audit: list[dict[str, Any]] = []
    if not blocked:
        return {"manifest": manifest, "audit": audit, "completed_preserved": sorted(completed)}
    conditions = _condition_rows()
    sanitized = _sanitized_index()
    clean = [
        r for r in resolved
        if r["condition"] == "clean" and r["formal_score"] is not None
    ]
    used = {m["formal300_block_id"] for m in manifest}
    repaired: list[dict[str, Any]] = []
    for row in manifest:
        if row.get("clean_B_input_verifiable"):
            repaired.append(row)
            continue
        provider = row["generating_provider"]
        route = row.get("clean_A_route")
        semantic_pool, any_pool = [], []
        for candidate in clean:
            if candidate["generating_provider"] != provider or candidate["source"] != "longmemeval":
                continue
            if candidate["pilot_block_id"] in used:
                continue
            frozen = sanitized.get((provider, candidate["condition_id"]))
            if not frozen:
                continue
            try:
                prompt = frozen_prompt(candidate["condition_id"], conditions)
            except RuntimeError:
                continue
            if _prompt_hash(prompt) != frozen.get("prompt_sha256"):
                continue
            any_pool.append(candidate)
            if candidate["route"] == route:
                semantic_pool.append(candidate)
        # Prefer a candidate whose frozen evaluator route matches the blocked slot;
        # fall back to the remaining verifiable same-provider LongMemEval universe
        # only when the same-route pool is exhausted by the frozen sample structure.
        pool = semantic_pool or any_pool
        if not pool:
            raise RuntimeError(f"no verifiable replacement available for {row['repeat_block_id']}")
        pool.sort(key=lambda c: (_replacement_rank(provider, c), c["condition_id"], c["pilot_block_id"]))
        chosen = pool[0]
        used.add(chosen["pilot_block_id"])
        new_row = build_manifest([chosen])[0]
        new_row["repeat_block_id"] = row["repeat_block_id"]
        audit.append({
            "original_repeat_block_id": row["repeat_block_id"],
            "original_formal300_block_id": row["formal300_block_id"],
            "original_clean_condition_id": row["clean_condition_id"],
            "block_reason": row.get("clean_B_block_reason"),
            "replacement_formal300_block_id": chosen["pilot_block_id"],
            "replacement_clean_condition_id": chosen["condition_id"],
            "provider": provider,
            "source": "longmemeval",
            "replacement_evaluator_route": chosen["route"],
            "same_route_pool_used": bool(semantic_pool),
            "selection_rank": 0,
            "selection_rank_definition": (
                "0-based index in the pool ordered by sha256('{seed}:replacement:{provider}:{condition_id}:{pilot_block_id}') "
                "with (condition_id, pilot_block_id) tie-break; no correctness input"
            ),
            "correctness_joined_after_selection": True,
        })
        repaired.append(new_row)
    return {"manifest": repaired, "audit": audit, "completed_preserved": sorted(completed)}


def _preflight_md(summary: dict[str, Any], cost: dict[str, Any], manifest: list[dict[str, Any]]) -> str:
    lines = [
        "# CLEAN-A / CLEAN-B stochasticity control — preflight (PHASE 1)",
        "",
        "Offline only. No generations, no judge calls, no changes to frozen formal-300 artifacts.",
        "",
        "## Sample",
        "",
        f"- Blocks: {summary['blocks']} ({summary['longmemeval']} LongMemEval, {summary['memora']} Memora).",
        f"- Per generating provider: {summary['by_generating_provider']}.",
        f"- Seed: {summary['seed']} (deterministic stratified selection).",
        f"- Duplicate formal-300 blocks: {summary['duplicate_formal300_blocks']}.",
        f"- Unique clean condition IDs: {summary['unique_clean_condition_ids']}.",
        "",
        "## Eligibility and non-conditioning on correctness",
        "",
        f"- Clean observations in the frozen sample: {summary['clean_observations_total']}; "
        f"resolved: {summary['clean_observations_resolved']}; "
        f"deterministic-route (excluded by route): {summary['clean_observations_deterministic_route']}; "
        f"eligible semantic-resolved: {summary['eligible_clean_resolved_semantic']}.",
        "- Selection uses only the frozen block/condition identifiers, the resolution *status*, and the frozen "
        "semantic route. The correctness value (True/False) is never read during selection; "
        f"`clean_A_correctness` is joined in the manifest only after selection "
        f"(flag: {all(m['clean_A_correctness_joined_after_selection'] for m in manifest)}).",
        "- Blocks whose clean-A observation was scored by the frozen native-deterministic route are excluded so "
        "that CLEAN-A and CLEAN-B use the identical evaluator route; this exclusion is by route, not by correctness.",
        f"- CLEAN-B input verification: {summary['clean_B_input_verifiable']} of {summary['blocks']} blocks reconstruct "
        f"the frozen generation prompt hash exactly; {summary['clean_B_input_blocked_unverifiable']} blocks are "
        "fail-closed because the v3-corrected prompt text is not stored in the repository (only its hash). "
        "Generation is blocked for those slots.",
        "",
        "## Frozen-input verification per block",
        "",
        "Each manifest row stores the frozen clean condition ID, the model-visible-context hash, the generation "
        "prompt hash, the provider runner, and the inference configuration. CLEAN-B must reproduce the same "
        "`model_visible_context_sha256` and `prompt_sha256` before a generation call is allowed.",
        "",
        "## Expected calls",
        "",
        f"- Generation calls: {cost['generation_calls']} by provider {cost['generation_calls_by_provider']}.",
        f"- Judge calls: base {cost['judge_calls_base']}, expected {cost['judge_calls_expected']}, max {cost['judge_calls_max']}.",
        f"- Expected generation input tokens (frozen CLEAN-A): {cost['expected_generation_input_tokens']}.",
        f"- Expected generation output tokens (frozen CLEAN-A): {cost['expected_generation_output_tokens']}.",
        f"- Pricing: {cost['pricing']['note']}",
        "",
        "## Selected blocks",
        "",
        "| repeat_block_id | provider | source | clean_A_correctness | evaluator route |",
        "| --- | --- | --- | --- | --- |",
    ]
    for m in manifest:
        route = "LME dual-primary + Gemini escalation" if m["source"] == "longmemeval" else \
            "Memora Gemini primary + secondary/escalation"
        lines.append(
            f"| {m['repeat_block_id']} | {m['generating_provider']} | {m['source']} | "
            f"{m['clean_A_correctness']} | {route} |")
    lines.append("")
    return "\n".join(lines)


def _analysis_plan_md() -> str:
    return "\n".join([
        "# Pre-registered analysis plan — CLEAN-A / CLEAN-B stochasticity control",
        "",
        "Frozen before execution. CLEAN-B is a fresh generation under the identical frozen clean input and",
        "provider configuration; the only difference is the generation request itself.",
        "",
        "## Primary metrics over the 60 paired blocks",
        "",
        "A. Clean repeat correctness disagreement rate = mean(correctness_A != correctness_B).",
        "B. Clean repeat paired accuracy difference = mean(correctness_A - correctness_B), in percentage points.",
        "C. Absolute clean repeat accuracy gap = |mean(correctness_A) - mean(correctness_B)|, in percentage points.",
        "D. Transition counts: correct-correct, correct-incorrect, incorrect-correct, incorrect-incorrect.",
        "E. Agreement rate = mean(correctness_A == correctness_B).",
        "F. 95% paired bootstrap CI for correctness_A - correctness_B (paired over the 60 blocks, deterministic seed).",
        "",
        "## Secondary (descriptive only)",
        "",
        "By source and by generating provider; n is small (30 and 10), so these are descriptive.",
        "",
        "## Comparison rule (pre-registered, descriptive)",
        "",
        "Compare the repeat-generation variation with the frozen formal-300 corruption effect",
        "(+35.54 pp; 95% CI 29.78-41.12). No significance threshold is invented after seeing results.",
        "If repeat variation is much smaller than 35.54 pp, this strengthens the argument that the corruption",
        "effect is not plausibly explained by ordinary repeat-generation stochasticity alone; if it is large,",
        "that is reported honestly.",
        "",
        "## Scoring protocol (unchanged from formal-300)",
        "",
        "- LongMemEval: OpenAI gpt-5.6-sol + Anthropic claude-sonnet-5 primaries; agreement resolves;",
        "  disagreement/non-binary routes to Gemini escalation.",
        "- Memora: Gemini criterion-level primary; OpenAI secondary where the frozen protocol requires;",
        "  Anthropic escalation if required; human adjudication only if the protocol reaches that state.",
        "- The obsolete unified deterministic scorer is not used.",
        "",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "repair", "generate", "judge", "finalize"), default="preflight")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid-judge-calls", action="store_true")
    parser.add_argument("--provider", choices=PROVIDERS + ("all",), default="all")
    parser.add_argument("--judge-provider", choices=("openai", "anthropic", "gemini", "all"), default="all")
    parser.add_argument("--canary-base", action="store_true")
    parser.add_argument("--recover-failed", action="store_true",
                        help="explicitly recover previously failed judge requests (paid; requires bounds + confirmation)")
    parser.add_argument("--judge-output-budget", choices=JUDGE_OUTPUT_BUDGET_MODES, default="formal300",
                        help="effective output-token budget: frozen formal-300 floors (default) or the stored planned budget")
    parser.add_argument("--output-budget-override", choices=JUDGE_OUTPUT_BUDGET_OVERRIDES, default="none",
                        help="explicit transport-only budget override (default none); "
                             "gemini_memora_transport uses the frozen formal-300 Gemini Memora recovery budget")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-requests-per-provider", type=int, default=None)
    parser.add_argument("--lme-source", type=Path, default=None,
                        help="frozen LongMemEval native source (defaults to the formal-300 path)")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)

    if args.phase == "preflight":
        resolved = _resolved_observations()
        selected = select_blocks(resolved)
        manifest = build_manifest(selected)
        cost = expected_calls_and_cost(manifest)
        write_preflight(manifest, cost, resolved, args.output)
        _print_summary(manifest, cost)
        return 0

    if args.phase == "generate":
        return _generate_phase(args)

    if args.phase == "judge":
        return _judge_phase(args)

    if args.phase == "repair":
        resolved = _resolved_observations()
        result = repair_manifest(args.output, resolved)
        manifest, audit = result["manifest"], result["audit"]
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "manifest.jsonl").write_text(
            "".join(json.dumps(m, sort_keys=True) + "\n" for m in manifest), encoding="utf-8")
        summary = manifest_summary(manifest, resolved)
        (args.output / "manifest_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # preserve prior audit history, append-only
        audit_path = args.output / "manifest_replacement_audit.json"
        history = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {"replacements": []}
        merged: dict[str, dict[str, Any]] = {
            entry["original_repeat_block_id"]: entry for entry in history.get("replacements", [])
        }
        for entry in audit:
            merged[entry["original_repeat_block_id"]] = entry
        history["replacements"] = [merged[k] for k in sorted(merged)]
        history.pop("replacement_runs", None)
        history["seed"] = SEED
        history["completed_canaries_preserved"] = result["completed_preserved"]
        history["selection_conditioned_on_correctness"] = False
        history["method"] = (
            "blocked rows replaced by the highest-ranked verifiable same-provider LongMemEval candidate "
            "from the remaining frozen formal-300 clean semantic-resolved universe (excluding rows already "
            "in the manifest); rank = sha256(seed:replacement:provider:condition_id:pilot_block_id)"
        )
        audit_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"REPLACED_BLOCKED_ROWS={len(audit)}")
        print(f"MANIFEST_TOTAL={summary['blocks']}")
        print(f"INPUT_VERIFIABLE_TOTAL={summary['clean_B_input_verifiable']}")
        print(f"GENERATION_BLOCKED_UNVERIFIABLE={summary['clean_B_input_blocked_unverifiable']}")
        print("PAID_CALLS_PERFORMED=0")
        return 0

    if args.phase == "finalize":
        # Offline finalization; requires CLEAN-B correctness already present.
        manifest_path = args.output / "manifest.jsonl"
        if not manifest_path.exists():
            raise SystemExit("preflight manifest missing; run --phase preflight first")
        rows = read_jsonl(manifest_path)
        if any(r.get("clean_B_correctness") is None for r in rows):
            print("CLEAN_B_RESULTS=pending")
            print("PAID_CALLS_PERFORMED=0")
            return 0
        print("CLEAN_B_RESULTS=present")
        print("PAID_CALLS_PERFORMED=0")
        return 0

    if not (args.execute and args.confirm_paid_judge_calls):
        raise SystemExit(f"phase '{args.phase}' is a paid phase and requires --execute --confirm-paid-judge-calls")
    raise SystemExit(
        "paid judging is not enabled in this build; CLEAN-B generation must be executed first and "
        "the judge executor wired under the frozen evaluator protocol"
    )


def _condition_rows() -> dict[str, dict[str, Any]]:
    rows = {r["condition_id"]: r for r in read_jsonl(ROOT / "data/final_main_governance_condition_manifest_v2.jsonl")}
    changed = ROOT / "results/answer_leakage_repair/changed_conditions.jsonl"
    for row in read_jsonl(changed) if changed.exists() else []:
        rows.setdefault(row["condition_id"], row)
    return rows


def frozen_prompt(condition_id: str, conditions: dict[str, dict[str, Any]]) -> str:
    if condition_id not in conditions:
        raise RuntimeError(f"frozen condition row missing: {condition_id}")
    from scripts.run_final_main_smoke_deepseek import render_prompt
    return render_prompt(conditions[condition_id])


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# Test/backend injection point.  The default performs a real provider call.
_GENERATION_BACKEND = None


def default_generation_backend(provider: str, prompt: str, model: str, max_tokens: int | None,
                               temperature: float | None, extra: dict[str, Any]) -> dict[str, Any]:
    """Call the frozen provider adapter once; returns answer text and usage."""
    import inspect
    if provider == "openai":
        from eval.providers.openai_provider import OpenAIDownstreamProvider as Cls
    elif provider == "anthropic":
        from eval.providers.anthropic_provider import AnthropicMessagesProvider as Cls
    elif provider == "gemini":
        from eval.providers.gemini_provider import GeminiGenerativeProvider as Cls
    elif provider == "kimi":
        from eval.providers.kimi_provider import KimiDownstreamProvider as Cls
    elif provider == "deepseek":
        from eval.providers.deepseek_provider import DeepSeekDownstreamProvider as Cls
    elif provider == "xai":
        from eval.providers.grok_provider import GrokDownstreamProvider as Cls
    else:
        raise RuntimeError(f"unknown provider: {provider}")
    kwargs: dict[str, Any] = {"model": model}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    for key, value in extra.items():
        if value is not None:
            kwargs[key] = value
    accepted = set(inspect.signature(Cls.__init__).parameters)
    kwargs = {k: v for k, v in kwargs.items() if k in accepted or k == "self"}
    adapter = Cls(**kwargs)
    response = adapter.request_downstream(prompt, max_tokens=max_tokens)
    parsed = getattr(response, "parsed", None)
    answer = parsed.get("answer") if isinstance(parsed, dict) else getattr(parsed, "answer", None)
    if not isinstance(answer, str):
        raise RuntimeError(f"{provider} response lacks a string answer field")
    telemetry = getattr(response, "telemetry", None)
    return {
        "answer": answer,
        "usage": {
            "input_tokens": getattr(telemetry, "input_tokens", None),
            "output_tokens": getattr(telemetry, "output_tokens", None),
            "total_tokens": getattr(telemetry, "total_tokens", None),
            "request_id": getattr(telemetry, "request_id", None),
        },
    }


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def plan_generation(manifest: list[dict[str, Any]], conditions: dict[str, dict[str, Any]],
                    providers: tuple[str, ...], max_requests: int | None,
                    max_per_provider: int | None, completed: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (runnable, blocked) slots using only frozen, hash-verified inputs."""
    runnable: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for row in sorted(manifest, key=lambda r: r["repeat_block_id"]):
        if row["generating_provider"] not in providers:
            continue
        if row["repeat_block_id"] in completed:
            continue
        try:
            prompt = frozen_prompt(row["clean_condition_id"], conditions)
        except RuntimeError as exc:
            blocked.append({**row, "block_reason": str(exc)})
            continue
        if _prompt_hash(prompt) != row["clean_input_ref"]["generation_prompt_sha256"]:
            blocked.append({**row, "block_reason": "frozen_prompt_hash_mismatch"})
            continue
        runnable.append({**row, "clean_prompt": prompt})
    if max_per_provider is not None:
        selected: list[dict[str, Any]] = []
        for provider in PROVIDERS:
            selected.extend([r for r in runnable if r["generating_provider"] == provider][:max_per_provider])
        runnable = selected
    elif max_requests is not None:
        runnable = runnable[:max_requests]
    return runnable, blocked


def _sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _judging_dir(out: Path) -> Path:
    path = out / JUDGING_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_frozen_lme_source(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the frozen LongMemEval native source (cached, read-only).

    The frozen formal-300 semantic payload is preferred for every observation,
    so this source is only needed to rebuild the frozen judge prompt for the one
    CLEAN-B observation whose CLEAN-A observation was resolved by the frozen
    native-deterministic route and therefore has no stored semantic payload.
    """
    key = str(path or LME_SOURCE_DEFAULT)
    if key not in _LME_SOURCE_CACHE:
        from scripts.design_final_native_evaluation import load_lme_source
        _LME_SOURCE_CACHE[key] = load_lme_source(Path(key))
    return _LME_SOURCE_CACHE[key]


def clean_b_judge_prompt(block: dict[str, Any], generation_results: dict[str, dict[str, Any]],
                         payloads: dict[str, dict[str, Any]]) -> str:
    """Reproduce the frozen formal-300 judge prompt with the CLEAN-B answer.

    Only ``semantic_judge_required`` observations reach paid semantic judging,
    and every one of them has a frozen formal-300 semantic payload (the exact
    prompt the frozen evaluator saw for the CLEAN-A answer).  It is reused
    verbatim and only the candidate answer field is replaced, so evaluator
    semantics are not reimplemented.
    """
    rid = block["repeat_block_id"]
    frozen = payloads.get(block["clean_A_observation_id"])
    if frozen is None:
        raise RuntimeError(f"frozen semantic payload missing for {rid}")
    value = json.loads(frozen["prompt"])
    value["existing_model_answer"] = generation_results[rid]["response_text"]
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def judge_request_identity(repeat_block_id: str, judge_slot: str, provider: str, model: str,
                           prompt: str, source: str) -> dict[str, Any]:
    prompt_sha = _prompt_hash(prompt)
    request_id = "clean-repeat-judge::" + _sha256({
        "protocol": JUDGE_PROTOCOL, "repeat_block_id": repeat_block_id, "judge_slot": judge_slot,
        "provider": provider, "model": model, "prompt_sha256": prompt_sha,
    })
    return {"judge_request_id": request_id, "repeat_block_id": repeat_block_id, "judge_slot": judge_slot,
            "provider": provider, "model": model, "source": source, "prompt_sha256": prompt_sha}


def planned_output_tokens(source: str, prompt: str) -> int:
    """Frozen formal-300 *planned* output budget for a single judge request."""
    if source == "longmemeval":
        return JUDGE_LME_PLANNED_OUTPUT_TOKENS
    return JUDGE_MEMORA_PLANNED_OUTPUT_BASE + JUDGE_MEMORA_PLANNED_OUTPUT_PER_CRITERION * len(
        json.loads(prompt)["criteria"])


def judge_output_budget(row: dict[str, Any], mode: str = "formal300") -> int:
    """Effective output-token budget for a judge request.

    ``planned`` uses only the stored planned budget.  ``formal300`` additionally
    applies the frozen formal-300 transport reliability floors via
    ``scripts.run_formal_300_validation.formal_output_budget`` (Anthropic always,
    OpenAI LongMemEval primaries, Anthropic Memora escalations).  Floors never
    change the model, prompt, schema, or the stored planned budget.
    """
    planned = int(row["planned_output_tokens"])
    if mode == "planned":
        return planned
    from scripts.run_formal_300_validation import formal_output_budget
    return formal_output_budget(row["provider"], planned, judge_slot=row.get("judge_slot"),
                                source=row.get("source"), phase=row.get("phase"))


def _criterion_count(row: dict[str, Any]) -> int | None:
    if row.get("source") != "memora":
        return None
    if row.get("criterion_count") is not None:
        return int(row["criterion_count"])
    prompt = row.get("prompt")
    if not isinstance(prompt, str):
        return None
    try:
        return len(json.loads(prompt)["criteria"])
    except Exception:
        return None


def gemini_memora_transport_budget(criterion_count: int) -> int:
    """Frozen formal-300 transport-only Gemini Memora budget (attempt-3 policy)."""
    from scripts.run_formal_300_validation import attempt3_output_budget
    return attempt3_output_budget(int(criterion_count))


def output_budget_record(row: dict[str, Any], mode: str = "formal300",
                         override: str = "none") -> dict[str, Any]:
    """Per-attempt output-budget metadata: frozen plan, effective value, policy.

    ``planned_output_tokens`` is always the frozen formal-300 *planned* budget so
    protocol differences stay auditable.  ``effective_output_tokens`` is what is
    actually requested.  Overrides never touch prompt/schema/model/identity.
    """
    planned = int(row["planned_output_tokens"])
    effective, policy = planned, "formal300_planned"
    if mode != "planned":
        effective = judge_output_budget(row, mode)
        policy = "formal300_floors"
    if override == "gemini_memora_transport" and row.get("provider") == "gemini" \
            and row.get("judge_slot") == "memora_primary_gemini":
        criteria = _criterion_count(row)
        if criteria is None:
            raise RuntimeError(f"cannot apply gemini_memora_transport override without criterion count: "
                               f"{row.get('judge_request_id')}")
        overridden = gemini_memora_transport_budget(criteria)
        if overridden > effective:
            effective, policy = overridden, "formal300_recovery_256_80n"
    return {
        "planned_output_tokens": planned,
        "effective_output_tokens": effective,
        "output_budget_policy": policy,
        "output_budget_override": override,
        "criterion_count": _criterion_count(row),
    }


def build_judge_plan(manifest: list[dict[str, Any]], generation_results: dict[str, dict[str, Any]],
                     payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Static paid judge plan built from the frozen formal-300 evaluator routes.

    A paid semantic slot is created only for a CLEAN-B observation whose frozen
    CLEAN-A route class requires semantic judging.  LongMemEval
    ``semantic_judge_required`` rows contribute two primary requests
    (OpenAI + Anthropic); Memora contributes one Gemini criterion-level primary
    per observation.  LongMemEval rows on the frozen deterministic route are
    scored offline and never receive paid semantic judging.  Result-dependent
    escalations are never part of this plan; they are materialized only when the
    frozen protocol requires them.
    """
    rows: list[dict[str, Any]] = []
    for block in sorted(manifest, key=lambda b: b["repeat_block_id"]):
        rid = block["repeat_block_id"]
        if rid not in generation_results:
            continue
        if block["source"] == "longmemeval" and block.get("clean_A_route") != LME_SEMANTIC_ROUTE:
            continue  # frozen deterministic route: resolved offline, never paid
        prompt = clean_b_judge_prompt(block, generation_results, payloads)
        for slot, provider in JUDGE_SLOTS[block["source"]]:
            rows.append({
                **judge_request_identity(rid, slot, provider, JUDGE_MODELS[provider], prompt, block["source"]),
                "judge_role": "primary", "phase": "primary",
                "clean_A_route": block.get("clean_A_route"),
                "clean_B_answer_sha256": generation_results[rid].get("response_sha256"),
                "clean_A_observation_id": block["clean_A_observation_id"],
                "formal300_block_id": block["formal300_block_id"], "prompt": prompt,
                "planned_output_tokens": planned_output_tokens(block["source"], prompt),
                "criterion_count": (len(json.loads(prompt)["criteria"])
                                    if block["source"] == "memora" else None),
            })
    rows.sort(key=lambda r: r["judge_request_id"])
    return rows


def _frozen_deterministic_row(observation_id: str) -> dict[str, Any] | None:
    """The frozen formal-300 deterministic result for a CLEAN-A observation."""
    for row in read_jsonl(PUBLIC / "deterministic_results.jsonl"):
        if row.get("observation_id") == observation_id:
            return {k: row.get(k) for k in
                    ("raw_decision", "binary_correct", "method", "evaluator_type", "evaluator_methodology", "answer_sha256")}
    return None


def _native_lme_routing_row(block: dict[str, Any],
                            lme_source: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    """The frozen routing inputs ({question_id, answer}) for a LongMemEval block.

    Prefers the frozen formal-300 LongMemEval native source; falls back to the
    in-repo frozen governance condition manifest, which carries the same native
    reference answer and question identifier, so the frozen routing
    implementation can be reused without reimplementation.
    """
    qid = str(block["base_id"]).split("::")[-1]
    if lme_source is not None and qid in lme_source:
        native = lme_source[qid]
        return ({"question_id": native.get("question_id", qid), "answer": native.get("answer")},
                f"{LME_SOURCE_DEFAULT} (frozen formal-300 native source)")
    condition = _condition_rows().get(block.get("clean_condition_id"))
    if condition is None:
        raise RuntimeError(f"frozen condition row missing for {block['repeat_block_id']}")
    return ({"question_id": condition.get("task_source_metadata", {}).get("question_id", qid),
             "answer": condition.get("task_oracle_answer")},
            "data/final_main_governance_condition_manifest_v2.jsonl (frozen condition manifest)")


def offline_deterministic_result(block: dict[str, Any], generation_results: dict[str, dict[str, Any]],
                                 lme_source: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve a non-semantic LongMemEval CLEAN-B answer offline.

    Reuses the exact frozen routing implementation
    (``scripts.design_final_native_evaluation.route_longmemeval``) and the exact
    frozen mapping into correctness used by
    ``scripts.prepare_formal_300_validation.prepare``
    (``binary_correct = decision in {"correct", "abstention_correct"}``).  No
    paid judge call is made and no score is invented when the frozen route does
    not itself produce a decision.
    """
    from scripts.design_final_native_evaluation import route_longmemeval
    rid = block["repeat_block_id"]
    native, native_source = _native_lme_routing_row(block, lme_source)
    answer = generation_results[rid]["response_text"]
    row = {"status": "success", "parse_success": True, "parsed_answer": answer, "base_id": block["base_id"]}
    route = route_longmemeval(row, native)
    clean_b_route = route["route"]
    if clean_b_route == "deterministic_native":
        decision = "correct"
    elif clean_b_route == "deterministic_abstention":
        decision = route["native_decision"]
    else:
        decision = None  # frozen route did not itself produce a decision: do not invent one
    binary_correct = None if decision is None else decision in {"correct", "abstention_correct"}
    return {
        "repeat_block_id": rid,
        "formal300_block_id": block["formal300_block_id"],
        "base_id": block["base_id"],
        "clean_A_observation_id": block["clean_A_observation_id"],
        "clean_A_route": block.get("clean_A_route"),
        "clean_B_route": clean_b_route,
        "clean_B_route_matches_clean_A_route": clean_b_route == block.get("clean_A_route"),
        "clean_A_correctness": block.get("clean_A_correctness"),
        "clean_A_frozen_deterministic_row": _frozen_deterministic_row(block["clean_A_observation_id"]),
        "clean_B_decision": decision,
        "clean_B_binary_correct": binary_correct,
        "clean_B_correctness": None if binary_correct is None else float(binary_correct),
        "clean_B_answer_sha256": generation_results[rid].get("response_sha256"),
        "paid_call": False,
        "resolution_method": "native_deterministic_route_offline",
        "evaluator_methodology": "final-native-evaluation-v2",
        "scorer_implementation": "scripts.design_final_native_evaluation.route_longmemeval",
        "scorer_reason": route.get("reason"),
        "scorer_answerability": route.get("answerability"),
        "correctness_mapping": "scripts.prepare_formal_300_validation.prepare: "
                               "binary_correct = decision in {correct, abstention_correct}",
        "source_artifacts": {
            "routing_implementation": "scripts/design_final_native_evaluation.py::route_longmemeval",
            "correctness_mapping": "scripts/prepare_formal_300_validation.py::prepare",
            "frozen_clean_a_artifact": "results/formal_300_validation_v1/deterministic_results.jsonl",
            "native_routing_source": native_source,
        },
    }


def build_offline_deterministic_results(manifest: list[dict[str, Any]],
                                        generation_results: dict[str, dict[str, Any]],
                                        lme_source: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Offline results for every LongMemEval CLEAN-B block on a non-semantic route."""
    rows: list[dict[str, Any]] = []
    for block in sorted(manifest, key=lambda b: b["repeat_block_id"]):
        if block["source"] != "longmemeval" or block.get("clean_A_route") == LME_SEMANTIC_ROUTE:
            continue
        if block["repeat_block_id"] not in generation_results:
            continue
        rows.append(offline_deterministic_result(block, generation_results, lme_source))
    return rows


def adapter_task_for(source: str) -> str:
    from scripts.run_formal_300_validation import adapter_task
    return adapter_task(source)


_JUDGE_BACKEND = None


def default_judge_backend(provider: str, model: str, prompt: str, source: str,
                          max_output_tokens: int) -> dict[str, Any]:
    """Call the frozen formal-300 judge adapter and parse with the frozen parser.

    This is the only place a paid judge request is issued.  It reuses the exact
    formal-300 transport adapters and the frozen downstream result parser so no
    evaluator semantics are reimplemented.
    """
    from scripts.b4_calibration_execution import AnthropicJudgeAdapter, GeminiJudgeAdapter, OpenAIJudgeAdapter
    from scripts.run_cheap_judge_pilot import parse_result
    cls = {"openai": OpenAIJudgeAdapter, "anthropic": AnthropicJudgeAdapter, "gemini": GeminiJudgeAdapter}[provider]
    adapter = cls(model, max_output_tokens=max_output_tokens)
    raw: str | None = None
    try:
        raw = adapter.call(prompt, adapter_task_for(source))
        parsed = parse_result(raw, source, prompt)
    except Exception as exc:  # transport, truncation, or malformed-output failure
        try:
            setattr(exc, "judge_diagnostics", judge_failure_diagnostics(
                exc, provider=provider, model=model, adapter=adapter, raw=raw,
                configured_max_output_tokens=max_output_tokens))
        except Exception:  # diagnostics must never mask the original failure
            pass
        raise
    diagnostics = getattr(adapter, "last_diagnostics", None) or getattr(adapter, "last_metadata", None)
    return {"raw": raw, "parsed": parsed, "diagnostics": diagnostics}


def _classify_judge_error(exc: BaseException) -> str:
    if isinstance(exc, json.JSONDecodeError) or "malformed JSON" in str(exc):
        return "malformed_json"
    if "no text" in str(exc).lower():
        return "empty_text"
    return "transport_error"


_TRUNCATION_FINISH_REASONS = {
    "max_tokens", "max_output_tokens", "length", "max_output_tokens_reached",
    "token_limit", "content_length",
}


def _appears_truncated(diagnostics: dict[str, Any]) -> bool:
    """Whether the response structure or finish reason indicates truncation."""
    finish = str(
        diagnostics.get("stop_reason")
        or diagnostics.get("finish_reason")
        or diagnostics.get("incomplete_reason")
        or ""
    ).lower()
    if finish in _TRUNCATION_FINISH_REASONS:
        return True
    if diagnostics.get("json_parse_ok") is not False:
        return False
    unbalanced = (
        diagnostics.get("raw_text_open_braces") is not None
        and diagnostics.get("raw_text_open_braces") != diagnostics.get("raw_text_close_braces")
    )
    return bool(unbalanced)


def judge_failure_diagnostics(exc: BaseException, *, provider: str, model: str, adapter: Any,
                              raw: str | None, configured_max_output_tokens: int) -> dict[str, Any]:
    """Safe, bounded, non-sensitive failure diagnostics for a judge request.

    Reuses the frozen formal-300 ``failure_diagnostics`` helper (adapter-captured
    transport diagnostics, else a bounded, redacted shape diagnostic) and adds
    provider/model, the requested output budget, the parsing error class, and a
    truncation signal.  API keys, authorization headers, raw prompts, and
    reasoning/chain-of-thought content are never captured.
    """
    from scripts.run_formal_300_validation import failure_diagnostics
    base = failure_diagnostics(exc, adapter, raw) or {}
    metadata = getattr(adapter, "last_metadata", None)  # Gemini exposes last_metadata
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            base.setdefault(key, value)
    diagnostics: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "error_class": type(exc).__name__,
        "error_kind": _classify_judge_error(exc),
        "configured_max_output_tokens": configured_max_output_tokens,
        "response_text_length": len(raw) if isinstance(raw, str) else None,
    }
    for key, value in base.items():
        diagnostics.setdefault(key, value)
    diagnostics["appears_truncated"] = _appears_truncated(diagnostics)
    return diagnostics


def _judge_ledger_events(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def read_jsonl_tolerant(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read a JSONL artifact, skipping (but reporting) blank/corrupt lines."""
    rows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    if not path.exists():
        return rows, malformed
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            malformed.append({"line_number": index, "error": "json_decode_error"})
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            malformed.append({"line_number": index, "error": "not_a_json_object"})
    return rows, malformed


def _judge_identity(event: dict[str, Any]) -> tuple:
    return tuple(str(event.get(key)) for key in JUDGE_IDENTITY_FIELDS)


def reconcile_judge_state(known_ids: set[str], ledger_events: list[dict[str, Any]],
                          result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Explicit ledger/result state machine keyed by request ID and attempt.

    Distinguishes, per request: completed-with-durable-result, failed-with-
    terminal-failure, started-without-terminal-event, terminal-completion-
    without-result, and invalid/conflicting ledger states (identity drift,
    duplicate completions, terminal ordering anomalies, orphan results).
    Ambiguous/interrupted attempts are fail-closed; a properly recorded failure
    is a terminal failure, never an interrupted attempt.
    """
    result_counts: dict[str, int] = defaultdict(int)
    for row in result_rows:
        result_counts[str(row.get("judge_request_id"))] += 1

    events_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in ledger_events:
        rid = event.get("judge_request_id")
        if rid is not None:
            events_by_id[str(rid)].append(event)

    requests: dict[str, dict[str, Any]] = {}
    for rid in sorted(set(events_by_id) | set(result_counts) | set(known_ids)):
        events = events_by_id.get(rid, [])
        attempts = [int(event.get("attempt") or 1) for event in events]
        started = [e for e in events if e.get("state") == "started"]
        completed = [e for e in events if e.get("state") == "completed"]
        failed = [e for e in events if e.get("state") == "failed"]
        completed_attempts = [int(e.get("attempt") or 1) for e in completed]
        has_result = result_counts.get(rid, 0) > 0
        identity_drift = len({_judge_identity(e) for e in events}) > 1
        duplicate_completed = len(completed_attempts) != len(set(completed_attempts))
        # every terminal event must be preceded (same attempt) by a started event
        started_attempts = {int(e.get("attempt") or 1) for e in started}
        terminal_without_start = any(
            int(e.get("attempt") or 1) not in started_attempts for e in completed + failed)
        last_state = events[-1].get("state") if events else None
        in_flight = last_state == "started"

        if identity_drift:
            status, fail_closed, recoverable = "conflicting_identity", True, False
        elif duplicate_completed or len(completed) > 1:
            status, fail_closed, recoverable = "duplicate_completed_events", True, False
        elif terminal_without_start:
            status, fail_closed, recoverable = "terminal_without_started", True, False
        elif in_flight:
            status, fail_closed, recoverable = "started_without_terminal", True, False
        elif last_state == "completed":
            if has_result:
                status, fail_closed, recoverable = "completed_with_result", False, False
            else:
                status, fail_closed, recoverable = "completed_without_result", True, False
        elif last_state == "failed":
            if has_result:
                status, fail_closed, recoverable = "failed_with_result", True, False
            else:
                status, fail_closed, recoverable = "failed", False, True
        elif has_result:
            status, fail_closed, recoverable = "result_without_terminal", True, False
        else:
            status, fail_closed, recoverable = "unattempted", False, False

        requests[rid] = {
            "judge_request_id": rid,
            "status": status,
            "attempts": max(attempts) if attempts else 0,
            "started_events": len(started),
            "completed_events": len(completed),
            "failed_events": len(failed),
            "has_result": has_result,
            "fail_closed": fail_closed,
            "recoverable": recoverable,
            "recovered": bool(completed) and bool(failed),
        }
    counts = Counter(state["status"] for state in requests.values())
    return {
        "requests": requests,
        "counts": dict(sorted(counts.items())),
        "fail_closed": sorted(rid for rid, s in requests.items() if s["fail_closed"]),
        "recoverable": sorted(rid for rid, s in requests.items() if s["recoverable"]),
    }


def _criterion_map(result: dict[str, Any] | None) -> dict[str, str] | None:
    parsed = (result or {}).get("parsed_result") or {}
    decisions = parsed.get("decisions")
    if not isinstance(decisions, list):
        return None
    return {str(item["criterion_id"]): str(item["decision"]) for item in decisions}


def _dynamic_row(repeat_block_id: str, slot: str, provider: str, phase: str,
                 group: dict[str, Any]) -> dict[str, Any]:
    prompt = group["prompt"]
    return {
        **judge_request_identity(repeat_block_id, slot, provider, JUDGE_MODELS[provider], prompt, group["source"]),
        "judge_role": phase, "phase": phase, "clean_A_route": group.get("clean_A_route"),
        "clean_B_answer_sha256": group.get("clean_B_answer_sha256"),
        "clean_A_observation_id": group.get("clean_A_observation_id"),
        "formal300_block_id": group.get("formal300_block_id"),
        "prompt": prompt, "planned_output_tokens": planned_output_tokens(group["source"], prompt),
        "criterion_count": (len(json.loads(prompt)["criteria"]) if group["source"] == "memora" else None),
    }


def dynamic_judge_requests(plan: list[dict[str, Any]],
                           results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Materialize only the frozen-protocol escalations currently required.

    Nothing is materialized until the frozen trigger fires on durable primary
    results: LongMemEval binary disagreement/non-binary -> Gemini escalation;
    Memora low-confidence or ``adjudication_required`` -> OpenAI secondary; a
    criterion disagreement between the Memora primary and secondary -> Anthropic
    escalation.  Escalations are never pre-run.
    """
    groups: dict[str, dict[str, Any]] = {}
    for row in plan:
        group = groups.setdefault(row["repeat_block_id"], {
            "source": row["source"], "prompt": row.get("prompt"),
            "formal300_block_id": row.get("formal300_block_id"),
            "clean_A_observation_id": row.get("clean_A_observation_id"),
            "clean_B_answer_sha256": row.get("clean_B_answer_sha256"),
            "clean_A_route": row.get("clean_A_route"), "slots": {},
        })
        result = results.get(row["judge_request_id"])
        if result is not None:
            group["slots"][row["judge_slot"]] = result
    rows: list[dict[str, Any]] = []
    for rid, group in sorted(groups.items()):
        if group["source"] == "longmemeval":
            openai = group["slots"].get("lme_primary_openai")
            anthropic = group["slots"].get("lme_primary_anthropic")
            if openai is None or anthropic is None:
                continue
            a = (openai.get("parsed_result") or {}).get("exploratory_binary")
            b = (anthropic.get("parsed_result") or {}).get("exploratory_binary")
            if a is None or b is None or a != b:
                rows.append(_dynamic_row(rid, "lme_escalation_gemini", "gemini", "escalation", group))
        else:
            primary = group["slots"].get("memora_primary_gemini")
            if primary is None:
                continue
            decisions = (primary.get("parsed_result") or {}).get("decisions", [])
            risky = any(d.get("confidence") == "low" or d.get("decision") == "adjudication_required"
                        for d in decisions)
            if risky:
                rows.append(_dynamic_row(rid, "memora_dynamic_secondary_openai", "openai", "secondary", group))
            secondary = group["slots"].get("memora_dynamic_secondary_openai") \
                or group["slots"].get("memora_secondary_openai")
            if secondary is not None:
                pm, sm = _criterion_map(primary), _criterion_map(secondary)
                if pm is None or sm is None or set(pm) != set(sm) or any(pm[k] != sm[k] for k in pm):
                    rows.append(_dynamic_row(rid, "memora_escalation_anthropic", "anthropic", "escalation", group))
    rows.sort(key=lambda r: r["judge_request_id"])
    return rows


def _persist_dynamic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append-only persistence of materialized dynamic escalations (no prompts)."""
    existing = read_jsonl(path) if path.exists() else []
    by_id = {row["judge_request_id"]: row for row in existing}
    if len(by_id) != len(existing):
        raise SystemExit("duplicate dynamic judge request identity")
    for row in rows:
        clean = {k: v for k, v in row.items() if k != "prompt"}
        rid = row["judge_request_id"]
        if rid in by_id and by_id[rid] != clean:
            raise SystemExit("dynamic judge request identity drift")
        by_id[rid] = clean
    path.write_text("".join(json.dumps(by_id[k], sort_keys=True) + "\n" for k in sorted(by_id)), encoding="utf-8")


def _canary_selection(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exactly three base-adapter requests covering all three primary adapters.

    One deterministic LongMemEval CLEAN-B observation (its OpenAI and Anthropic
    primaries, same observation) plus one deterministic Memora CLEAN-B
    observation (its Gemini primary).  Selection uses the smallest repeat-block
    identifier per source and never reads CLEAN-A/CLEAN-B correctness or answer
    quality.
    """
    lme_blocks = sorted({r["repeat_block_id"] for r in plan if r["source"] == "longmemeval"})
    memora_blocks = sorted({r["repeat_block_id"] for r in plan if r["source"] == "memora"})
    if not lme_blocks or not memora_blocks:
        raise SystemExit("canary requires at least one LongMemEval and one Memora judge request")
    selected = sorted((r for r in plan if r["repeat_block_id"] == lme_blocks[0]), key=lambda r: r["judge_slot"])
    selected.append(next(r for r in plan if r["repeat_block_id"] == memora_blocks[0]))
    return selected


def _load_recovery_artifacts(path: Path) -> dict[str, str]:
    """request_id -> stored raw model text for offline recovery (if any)."""
    rows = read_jsonl(path) if path.exists() else []
    return {str(r["judge_request_id"]): r.get("raw_text") for r in rows if r.get("judge_request_id")}


def _offline_recover(row: dict[str, Any], artifact_raw: Any) -> dict[str, Any] | None:
    """Re-parse a stored raw response without any paid call; ``None`` if unusable."""
    from scripts.run_cheap_judge_pilot import parse_result
    if not isinstance(artifact_raw, str) or not artifact_raw.strip():
        return None
    try:
        return parse_result(artifact_raw, row["source"], row["prompt"])
    except Exception:
        return None


def _validate_selected_row(row: dict[str, Any], plan_by_id: dict[str, dict[str, Any]],
                           dynamic_by_id: dict[str, dict[str, Any]]) -> None:
    """Fail closed unless the selected request matches a frozen/dynamic plan row."""
    rid = row["judge_request_id"]
    reference = plan_by_id.get(rid) or dynamic_by_id.get(rid)
    if reference is None:
        raise SystemExit(f"fail-closed: selected request {rid} is not in the frozen judge plan")
    for key in ("judge_request_id", "repeat_block_id", "judge_slot", "provider", "model", "source", "prompt_sha256"):
        if str(reference.get(key)) != str(row.get(key)):
            raise SystemExit(f"fail-closed: selected request {rid} identity drift on '{key}'")


def _judge_phase(args: argparse.Namespace) -> int:
    manifest = read_jsonl(args.output / "manifest.jsonl")
    if not (args.output / RESULTS_NAME).exists():
        raise SystemExit("CLEAN-B generation results missing; finish --phase generate before judging")
    generation_results = {r["repeat_block_id"]: r for r in read_jsonl(args.output / RESULTS_NAME)}
    missing = [b["repeat_block_id"] for b in manifest if b["repeat_block_id"] not in generation_results]
    if missing:
        raise SystemExit(f"fail-closed: {len(missing)} CLEAN-B block(s) lack a durable generation result")
    payloads = {r["observation_id"]: r for r in read_jsonl(PRIVATE / "semantic_payloads.jsonl")}
    try:
        lme_source = load_frozen_lme_source(args.lme_source)
    except (FileNotFoundError, OSError):
        # Fallback routing inputs come from the in-repo frozen condition manifest.
        lme_source = None
    plan = build_judge_plan(manifest, generation_results, payloads)
    try:
        offline_results = build_offline_deterministic_results(manifest, generation_results, lme_source)
    except RuntimeError as exc:
        raise SystemExit(f"fail-closed: cannot resolve the deterministic-route CLEAN-B block(s): {exc}")
    judging = _judging_dir(args.output)
    plan_path = judging / "judge_execution_plan.jsonl"
    plan_path.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "prompt"}, sort_keys=True) + "\n"
                                 for r in plan), encoding="utf-8")
    (judging / OFFLINE_DETERMINISTIC_RESULTS_NAME).write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in offline_results), encoding="utf-8")
    ledger = judging / "judge_execution_ledger.jsonl"
    results_path = judging / "judge_results.jsonl"
    diagnostics = judging / "judge_diagnostics.jsonl"
    dynamic_path = judging / "dynamic_judge_plan.jsonl"
    if not dynamic_path.exists():
        dynamic_path.write_text("", encoding="utf-8")
    ledger_events, ledger_malformed = read_jsonl_tolerant(ledger)
    result_rows, results_malformed = read_jsonl_tolerant(results_path)
    results: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        results[str(row.get("judge_request_id"))] = row  # last durable result per request wins
    dynamic_rows = dynamic_judge_requests(plan, results)
    _persist_dynamic(dynamic_path, dynamic_rows)

    plan_by_id = {r["judge_request_id"]: r for r in plan}
    dynamic_by_id = {r["judge_request_id"]: r for r in dynamic_rows}
    known = set(plan_by_id) | set(dynamic_by_id) | set(results)
    reconciliation = reconcile_judge_state(known, ledger_events, result_rows)
    state = reconciliation["requests"]
    # a durable result whose identity disagrees with the frozen/dynamic plan is a conflict
    identity_conflicts = sorted(
        rid for rid, row in results.items()
        if rid in (plan_by_id | dynamic_by_id)
        and any(str((plan_by_id | dynamic_by_id)[rid].get(k)) != str(row.get(k))
                for k in ("judge_slot", "provider", "model", "source", "prompt_sha256")))
    fail_closed = sorted(set(reconciliation["fail_closed"]) | set(identity_conflicts))
    completed = {rid for rid, s in state.items() if s["status"] == "completed_with_result"}
    failed = sorted(rid for rid, s in state.items() if s["status"] == "failed")
    interrupted = sorted(rid for rid, s in state.items() if s["status"] == "started_without_terminal")

    pool = plan + [r for r in dynamic_rows if r["judge_request_id"] not in plan_by_id]
    providers = tuple(JUDGE_MODELS) if args.judge_provider == "all" else (args.judge_provider,)
    if args.recover_failed:
        candidates = [r for r in pool if r["judge_request_id"] in set(failed)]
    elif args.canary_base:
        candidates = _canary_selection(plan)
    else:
        candidates = [r for r in pool if state.get(r["judge_request_id"], {}).get("status") == "unattempted"]
    selected = [r for r in candidates if r["provider"] in providers]
    if args.max_requests is not None:
        selected = selected[:args.max_requests]

    print(f"JUDGE_STATIC_TOTAL={len(plan)}")
    print(f"JUDGE_COMPLETED_TOTAL={sum(1 for r in plan if r['judge_request_id'] in completed)}")
    print(f"JUDGE_PENDING_TOTAL={sum(1 for r in plan if r['judge_request_id'] not in completed)}")
    print(f"DYNAMIC_ESCALATIONS_CURRENTLY_REQUIRED={len(dynamic_rows)}")
    print(f"OFFLINE_DETERMINISTIC_BLOCKS={len(offline_results)}")
    print(f"JUDGE_FAILED_TOTAL={len(failed)}")
    print(f"JUDGE_RECOVERABLE_TOTAL={len(reconciliation['recoverable'])}")
    print(f"JUDGE_INVALID_TOTAL={len(fail_closed)}")
    print(f"JUDGE_RECOVERY_MODE={'yes' if args.recover_failed else 'no'}")
    print(f"OUTPUT_BUDGET_OVERRIDE={args.output_budget_override}")
    print(f"JUDGE_SELECTED_REQUESTS={len(selected)}")
    print(f"SELECTED_OPENAI={sum(1 for r in selected if r['provider'] == 'openai')}")
    print(f"SELECTED_ANTHROPIC={sum(1 for r in selected if r['provider'] == 'anthropic')}")
    print(f"SELECTED_GEMINI={sum(1 for r in selected if r['provider'] == 'gemini')}")
    print(f"JUDGE_STARTED_ONLY_FAIL_CLOSED={len(interrupted)}")
    rec_path = judging / JUDGE_RECONCILIATION_NAME
    rec_path.write_text(json.dumps({
        "protocol": JUDGE_PROTOCOL,
        "counts": reconciliation["counts"],
        "completed": sorted(completed),
        "failed": failed,
        "recoverable": reconciliation["recoverable"],
        "interrupted": interrupted,
        "fail_closed": fail_closed,
        "ledger_malformed_lines": ledger_malformed,
        "results_malformed_lines": results_malformed,
        "result_identity_conflicts": identity_conflicts,
        "output_budget_mode": args.judge_output_budget,
        "output_budget_override": args.output_budget_override,
        "paid_calls_performed": 0,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.dry_run:
        for row in selected:
            print(f"WOULD_JUDGE {row['judge_request_id']} {row['provider']} {row['judge_slot']}")
        print("DRY_RUN_CALLS=0")
        print("PAID_CALLS_PERFORMED=0")
        return 0
    if not args.execute:
        raise SystemExit("judge is a paid phase and requires --execute")
    if not args.confirm_paid_judge_calls:
        raise SystemExit("judge is a paid phase and requires --confirm-paid-judge-calls")
    if args.max_requests is None:
        raise SystemExit("judge requires a bounded request count: --max-requests N")
    if args.max_requests < 1:
        raise SystemExit("--max-requests must be at least one")
    if fail_closed:
        raise SystemExit(
            f"fail-closed: {len(fail_closed)} judge request(s) have interrupted or conflicting ledger state")
    backend = _JUDGE_BACKEND or default_judge_backend
    recovery_artifacts = _load_recovery_artifacts(judging / JUDGE_RECOVERY_ARTIFACT_NAME)
    paid = 0
    for row in selected:
        _validate_selected_row(row, plan_by_id, dynamic_by_id)
        rid = row["judge_request_id"]
        identity = {k: row[k] for k in JUDGE_IDENTITY_FIELDS}
        attempt = int(state.get(rid, {}).get("attempts", 0)) + 1
        budget_record = output_budget_record(row, args.judge_output_budget, args.output_budget_override)
        budget = budget_record["effective_output_tokens"]
        # Prefer offline recovery of an already-returned response over a paid call.
        recovered = _offline_recover(row, recovery_artifacts.get(rid)) if rid in recovery_artifacts else None
        if recovered is not None:
            raw_artifact = recovery_artifacts[rid]
            _append(results_path, {**identity, "parsed_result": recovered,
                                   "response_hash": _prompt_hash(raw_artifact), "attempt": attempt,
                                   **budget_record,
                                   "recovery": "offline_from_stored_response",
                                   "timestamp_ns": time.time_ns()})
            _append(ledger, {**identity, "state": "completed", "attempt": attempt, **budget_record,
                             "recovery": "offline_from_stored_response", "timestamp_ns": time.time_ns()})
            _append(diagnostics, {**identity, "state": "completed", "attempt": attempt, **budget_record,
                                  "recovery": "offline_from_stored_response"})
            continue
        _append(ledger, {**identity, "state": "started", "attempt": attempt, **budget_record,
                         "timestamp_ns": time.time_ns()})
        paid += 1
        try:
            output = backend(row["provider"], row["model"], row["prompt"], row["source"], budget)
            raw = output["raw"]
            _append(results_path, {**identity, "parsed_result": output["parsed"], "response_hash": _prompt_hash(raw),
                                   "attempt": attempt, **budget_record, "timestamp_ns": time.time_ns()})
            _append(ledger, {**identity, "state": "completed", "attempt": attempt, **budget_record,
                             "response_hash": _prompt_hash(raw), "timestamp_ns": time.time_ns()})
            _append(diagnostics, {**identity, "state": "completed", "attempt": attempt, **budget_record,
                                  "diagnostics": output.get("diagnostics")})
        except Exception as exc:
            diags = getattr(exc, "judge_diagnostics", None)
            _append(ledger, {**identity, "state": "failed", "attempt": attempt, **budget_record,
                             "error_class": type(exc).__name__, "error_message": str(exc)[:1000],
                             "timestamp_ns": time.time_ns()})
            _append(diagnostics, {**identity, "state": "failed", "attempt": attempt, **budget_record,
                                  "error_class": type(exc).__name__,
                                  "error_message": str(exc)[:1000], "diagnostics": diags})
    print(f"JUDGE_CALLS_PERFORMED={paid}")
    print(f"PAID_CALLS_PERFORMED={paid}")
    return 0


def _generate_phase(args: argparse.Namespace) -> int:
    manifest_path = args.output / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit("preflight manifest missing; run --phase preflight first")
    manifest = read_jsonl(manifest_path)
    providers = tuple(PROVIDERS) if args.provider == "all" else (args.provider,)
    if args.max_requests is None and args.max_requests_per_provider is None:
        raise SystemExit("generate requires a bounded request count: --max-requests N or --max-requests-per-provider N")
    for value in (args.max_requests, args.max_requests_per_provider):
        if value is not None and value < 1:
            raise SystemExit("request bounds must be at least one")

    ledger = args.output / LEDGER_NAME
    results = args.output / RESULTS_NAME
    diagnostics = args.output / DIAGNOSTICS_NAME
    completed = {
        r["repeat_block_id"] for r in (read_jsonl(ledger) if ledger.exists() else [])
        if r.get("state") == "completed"
    }
    conditions = _condition_rows()
    runnable, blocked = plan_generation(
        manifest, conditions, providers, args.max_requests, args.max_requests_per_provider, completed)

    print(f"GENERATE_PROVIDERS={','.join(providers)}")
    print(f"GENERATE_PENDING_RUNNABLE={len(runnable)}")
    print(f"GENERATE_BLOCKED_UNVERIFIABLE={len(blocked)}")
    print(f"GENERATE_ALREADY_COMPLETED={len(completed)}")
    if args.dry_run:
        for row in runnable:
            print(f"WOULD_GENERATE {row['repeat_block_id']} {row['generating_provider']} {row['generating_model']}")
        print("DRY_RUN_CALLS=0")
        print("PAID_CALLS_PERFORMED=0")
        return 0
    if not args.execute:
        raise SystemExit("generate is a paid phase and requires --execute")
    backend = _GENERATION_BACKEND or default_generation_backend
    paid = 0
    for row in runnable:
        identity = {
            "repeat_block_id": row["repeat_block_id"], "formal300_block_id": row["formal300_block_id"],
            "source": row["source"], "base_id": row["base_id"], "generating_provider": row["generating_provider"],
            "generating_model": row["generating_model"], "clean_condition_id": row["clean_condition_id"],
            "clean_A_observation_id": row["clean_A_observation_id"],
            "generation_prompt_sha256": row["clean_input_ref"]["generation_prompt_sha256"],
        }
        _append(ledger, {**identity, "state": "planned", "attempt": 1})
        _append(ledger, {**identity, "state": "started", "attempt": 1})
        paid += 1
        config = row["inference_config_ref"].get("model_parameters") or {}
        start = time.perf_counter()
        try:
            output = backend(
                row["generating_provider"], row["clean_prompt"], row["generating_model"],
                config.get("max_output_tokens"), config.get("temperature"),
                {"top_p": config.get("top_p"), "reasoning_effort": config.get("reasoning_effort"),
                 "thinking": config.get("thinking")},
            )
            latency = round((time.perf_counter() - start) * 1000, 3)
            response_sha = hashlib.sha256(output["answer"].encode("utf-8")).hexdigest()
            _append(results, {**identity, "provider": row["generating_provider"], "model": row["generating_model"],
                              "response_text": output["answer"], "response_sha256": response_sha,
                              "usage": output.get("usage") or {}, "inference_config": config,
                              "timestamp_ns": time.time_ns()})
            _append(ledger, {**identity, "state": "completed", "attempt": 1, "response_sha256": response_sha})
            _append(diagnostics, {**identity, "state": "completed", "latency_ms": latency,
                                  **(output.get("usage") or {})})
        except Exception as exc:  # fail closed; the slot stays retryable and is never marked complete
            _append(ledger, {**identity, "state": "failed", "attempt": 1,
                             "error_class": type(exc).__name__, "error_message": str(exc)[:1000]})
            _append(diagnostics, {**identity, "state": "failed", "error_class": type(exc).__name__,
                                  "error_message": str(exc)[:1000]})
    for row in blocked:
        _append(diagnostics, {"repeat_block_id": row["repeat_block_id"],
                              "generating_provider": row["generating_provider"],
                              "clean_condition_id": row["clean_condition_id"],
                              "state": "blocked_unverifiable_input", "reason": row["block_reason"]})
    print(f"GENERATION_CALLS_PERFORMED={paid}")
    print("PAID_CALLS_PERFORMED=0" if paid == 0 else f"PAID_CALLS_PERFORMED={paid}")
    return 0


def _print_summary(manifest: list[dict[str, Any]], cost: dict[str, Any]) -> None:
    by_source = Counter(m["source"] for m in manifest)
    by_provider = Counter(m["generating_provider"] for m in manifest)
    print(f"CLEAN_REPEAT_BLOCKS={len(manifest)}")
    print(f"CLEAN_REPEAT_LONGMEMEVAL={by_source['longmemeval']}")
    print(f"CLEAN_REPEAT_MEMORA={by_source['memora']}")
    print(f"CLEAN_REPEAT_OPENAI={by_provider['openai']}")
    print(f"CLEAN_REPEAT_ANTHROPIC={by_provider['anthropic']}")
    print(f"CLEAN_REPEAT_GEMINI={by_provider['gemini']}")
    print(f"CLEAN_REPEAT_KIMI={by_provider['kimi']}")
    print(f"CLEAN_REPEAT_XAI={by_provider['xai']}")
    print(f"CLEAN_REPEAT_DEEPSEEK={by_provider['deepseek']}")
    print(f"SAMPLE_SEED={SEED}")
    print("SELECTION_CONDITIONED_ON_CORRECTNESS=no")
    print(f"NEW_GENERATION_CALLS_EXPECTED={cost['generation_calls']}")
    print(f"JUDGE_CALLS_BASE_EXPECTED={cost['judge_calls_base']}")
    print(f"JUDGE_CALLS_MAX_EXPECTED={cost['judge_calls_max']}")
    print(f"EXPECTED_GENERATION_INPUT_TOKENS={cost['expected_generation_input_tokens']}")
    print(f"EXPECTED_GENERATION_OUTPUT_TOKENS={cost['expected_generation_output_tokens']}")
    print(f"EXPECTED_JUDGE_INPUT_TOKENS={cost['judge_input_tokens_expected']}")
    print(f"EXPECTED_JUDGE_OUTPUT_TOKENS={cost['judge_output_tokens_expected']}")
    print("ESTIMATED_COST_LOW=unavailable_no_local_pricing")
    print("ESTIMATED_COST_EXPECTED=unavailable_no_local_pricing")
    print("ESTIMATED_COST_HIGH=unavailable_no_local_pricing")
    print("PAID_CALLS_PERFORMED=0")
    print("READY_FOR_PAID_EXECUTION=yes")


if __name__ == "__main__":
    raise SystemExit(main())
