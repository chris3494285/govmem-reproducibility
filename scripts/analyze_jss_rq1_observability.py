#!/usr/bin/env python3
"""Offline RQ1 Context-Output observability analysis for GovMem.

This supplementary analysis asks which governance-to-behavior anomalies are
visible at three levels:

1. structural governance-state constraints;
2. the model-visible context projection;
3. downstream output outcomes.

The frozen corruptions are realised in the projection while
``internal_governance_state`` remains a copy of the oracle state.  State-level
detector performance is therefore not estimable and is retained only as a
representation limitation.  Downstream metrics remain separated by score unit:
frozen binary proxy correctness, LongMemEval binary native correctness, and
Memora FAMA.

No network, provider API, model generation, or scoring is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_jss_rq1_observability import (  # noqa: E402
    AuditBundle,
    AuditError,
    DEFAULT_BASE_FREEZE,
    DEFAULT_BASE_MANIFEST,
    DEFAULT_CONDITION_FREEZE,
    DEFAULT_CONDITION_MANIFEST,
    DEFAULT_EVIDENCE_BASE_LEVEL,
    DEFAULT_FINAL_FORMAL_BLOCKS,
    DEFAULT_FINAL_FORMAL_METRICS,
    DEFAULT_FORMAL_BLOCKS,
    DEFAULT_FORMAL_OBSERVATIONS,
    DEFAULT_FORMAL_SUMMARY,
    DEFAULT_HUMAN_ADJUDICATION,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SEMANTIC_PAYLOADS,
    NATIVE_PANEL_MODES,
    _state_issues,
    audit_repository,
    canonical_items,
    context_membership,
    effective_context_text,
    parse_proxy_outcome,
    write_json,
)

DEFAULT_OUTPUT = ROOT / "results/jss_rq1_observability_v2"
BOOTSTRAP_SEED = 20261101
OUTPUT_FILES = (
    "observability_summary.json",
    "observability_by_source.csv",
    "observability_by_fault.csv",
    "observability_by_event.csv",
    "observability_by_provider.csv",
    "observability_overlap.csv",
    "observability_limitations.md",
)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take a percentile of no values")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def cluster_bootstrap_rate(
    records: Sequence[dict[str, Any]],
    indicator: str,
    *,
    iterations: int = 5000,
    seed: int = BOOTSTRAP_SEED,
    cluster_field: str = "base_id",
) -> dict[str, Any]:
    """Cluster-bootstrap a binary indicator, resampling base clusters."""
    eligible = [row for row in records if row.get(indicator) is not None]
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        clusters[str(row.get(cluster_field))].append(row)
    if not eligible:
        return {"numerator": 0, "denominator": 0, "estimate": None, "ci95": None, "base_clusters": 0}
    numerator = sum(bool(row[indicator]) for row in eligible)
    estimate = numerator / len(eligible)
    keys = sorted(clusters)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        sampled: list[dict[str, Any]] = []
        for _ in keys:
            sampled.extend(clusters[keys[rng.randrange(len(keys))]])
        samples.append(sum(bool(row[indicator]) for row in sampled) / len(sampled))
    return {
        "numerator": numerator,
        "denominator": len(eligible),
        "estimate": estimate,
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "base_clusters": len(clusters),
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
        "bootstrap_unit": "base_id cluster",
    }


def cluster_bootstrap_mean(
    records: Sequence[dict[str, Any]],
    value_field: str,
    *,
    iterations: int = 5000,
    seed: int = BOOTSTRAP_SEED,
    cluster_field: str = "base_id",
) -> dict[str, Any]:
    """Cluster-bootstrap a paired continuous value, resampling bases."""
    eligible = [row for row in records if row.get(value_field) is not None]
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        clusters[str(row.get(cluster_field))].append(row)
    if not eligible:
        return {"numerator": None, "denominator": 0, "estimate": None, "ci95": None, "base_clusters": 0}
    values = [float(row[value_field]) for row in eligible]
    keys = sorted(clusters)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        sampled: list[float] = []
        for _ in keys:
            sampled.extend(float(row[value_field]) for row in clusters[keys[rng.randrange(len(keys))]])
        samples.append(statistics.fmean(sampled))
    return {
        "numerator": None,
        "denominator": len(eligible),
        "estimate": statistics.fmean(values),
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "base_clusters": len(clusters),
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
        "bootstrap_unit": "base_id cluster",
    }


def build_context_records(bundle: AuditBundle) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for triplet in bundle.triplets:
        clean = triplet["conditions"]["clean"]
        corrupted = triplet["conditions"]["corrupted"]
        enforced = triplet["conditions"]["enforced"]
        source = triplet["source_dataset"]
        counts = context_membership(clean, corrupted)
        items_changed = canonical_items(clean) != canonical_items(corrupted)
        text_changed = effective_context_text(clean) != effective_context_text(corrupted)
        changed = items_changed or text_changed
        added_relative_to_clean = counts["added_item_instances"] > 0
        removed_relative_to_clean = counts["removed_item_instances"] > 0
        restored = canonical_items(enforced) == canonical_items(clean) and effective_context_text(
            enforced
        ) == effective_context_text(clean)
        state_changed = (
            corrupted.get("internal_governance_state") != clean.get("internal_governance_state")
        )
        state_issues = _state_issues(corrupted.get("internal_governance_state"), source)
        records.append(
            {
                "base_id": triplet["base_id"],
                "pair_id": triplet["pair_id"],
                "source_dataset": source,
                "analysis_event_type": triplet["analysis_event_type"],
                "fault_operator": triplet["fault_operator"],
                "evidence_supported": bool(bundle.evidence_support.get(triplet["base_id"])),
                "state_internal_changed": state_changed,
                "state_independent_constraint_anomaly": bool(state_issues),
                "state_oracle_comparison_anomaly": (
                    corrupted.get("internal_governance_state")
                    != corrupted.get("oracle_state")
                ),
                "state_constraint_issue_types": sorted(state_issues),
                "injected_fault_label_present": bool(
                    (corrupted.get("corruption_specification") or {}).get("operator")
                ),
                "context_changed": changed,
                "context_items_changed": items_changed,
                "context_text_changed": text_changed,
                "effective_context_fallback_used": bool(
                    not str(clean.get("model_visible_context_text") or "").strip()
                    or not str(corrupted.get("model_visible_context_text") or "").strip()
                ),
                "context_added_item_relative_to_clean": added_relative_to_clean,
                "context_removed_item_relative_to_clean": removed_relative_to_clean,
                "other_context_change": (
                    changed and not added_relative_to_clean and not removed_relative_to_clean
                ),
                "enforced_context_restored": restored,
                "added_item_instances": counts["added_item_instances"],
                "removed_item_instances": counts["removed_item_instances"],
                "context_classification_reference": "evidence-supported clean oracle projection",
            }
        )
    return records


def _context_by_pair(context_records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["pair_id"]): row for row in context_records}


def build_proxy_records(
    context_records: Sequence[dict[str, Any]],
    provider_results: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for context in context_records:
        condition_ids = {
            "clean": f"{context['base_id']}::clean",
            "corrupted": f"{context['pair_id']}::corrupted",
            "enforced": f"{context['pair_id']}::enforced",
        }
        for provider, rows in sorted(provider_results.items()):
            outcomes: dict[str, bool | None] = {}
            outcome_states: dict[str, str | None] = {}
            reasons: dict[str, str | None] = {}
            for condition, condition_id in condition_ids.items():
                state, score, reason = parse_proxy_outcome(rows.get(condition_id))
                outcome_states[condition] = state if state != "missing" else None
                outcomes[condition] = score
                reasons[condition] = reason
            clean_score = outcomes["clean"]
            corrupt_score = outcomes["corrupted"]
            enforced_score = outcomes["enforced"]
            corruption_pair_scoreable = clean_score is not None and corrupt_score is not None
            recovery_pair_scoreable = corrupt_score is not None and enforced_score is not None
            full_triplet_scoreable = corruption_pair_scoreable and enforced_score is not None
            correctness_changed = (
                (clean_score != corrupt_score) if corruption_pair_scoreable else None
            )
            correctness_unchanged = (
                (clean_score == corrupt_score) if corruption_pair_scoreable else None
            )
            induced_failure = (
                bool(clean_score is True and corrupt_score is False)
                if corruption_pair_scoreable
                else None
            )
            correctness_improved = (
                bool(clean_score is False and corrupt_score is True)
                if corruption_pair_scoreable
                else None
            )
            recovery_success = (
                bool(corrupt_score is False and enforced_score is True)
                if recovery_pair_scoreable
                else None
            )
            records.append(
                {
                    **context,
                    "provider": provider,
                    "condition_ids": condition_ids,
                    "proxy_outcome_states": outcome_states,
                    "proxy_unresolved_reasons": reasons,
                    "proxy_pair_scoreable": corruption_pair_scoreable,
                    "proxy_pair_unresolved": not corruption_pair_scoreable,
                    "proxy_recovery_pair_scoreable": recovery_pair_scoreable,
                    "proxy_recovery_pair_unresolved": not recovery_pair_scoreable,
                    "proxy_full_triplet_scoreable": full_triplet_scoreable,
                    "proxy_enforced_unresolved": enforced_score is None,
                    "proxy_clean_correct": clean_score,
                    "proxy_corrupted_correct": corrupt_score,
                    "proxy_enforced_correct": enforced_score,
                    "proxy_correctness_outcome_changed": correctness_changed,
                    "proxy_correctness_outcome_unchanged": correctness_unchanged,
                    "proxy_corruption_induced_failure": induced_failure,
                    "proxy_correctness_improved": correctness_improved,
                    "proxy_recovery_success": recovery_success,
                    "proxy_context_changed_correctness_unchanged": (
                        bool(context["context_changed"] and correctness_unchanged)
                        if correctness_unchanged is not None
                        else None
                    ),
                    "proxy_context_unchanged_correctness_changed": (
                        bool(not context["context_changed"] and correctness_changed)
                        if correctness_changed is not None
                        else None
                    ),
                }
            )
    return records


def _formal_observation_groups(observations: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        groups[str(row["pilot_block_id"])].append(row)
    return groups


def build_native_records(bundle: AuditBundle, context_records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    context_by_pair = _context_by_pair(context_records)
    resolved_by_block = {
        str(row["pilot_block_id"]): row for row in bundle.formal_blocks
    }
    groups = _formal_observation_groups(bundle.formal_observations)
    records: list[dict[str, Any]] = []
    for block_id, observations in sorted(groups.items()):
        by_condition = {str(row["condition"]): row for row in observations}
        first = observations[0]
        base_id = resolved_by_block[block_id]["base_id"] if block_id in resolved_by_block else None
        fault_operator = str(first.get("fault_type"))
        if block_id in resolved_by_block:
            block = resolved_by_block[block_id]
            base_id = block["base_id"]
            fault_operator = block["fault_operator"]
            pair_id = f"{base_id}::{fault_operator}"
            context = context_by_pair.get(pair_id)
            scores = {condition: block["scores"].get(condition) for condition in ("clean", "corrupted", "enforced")}
            answer_hashes = block["answer_sha256"]
            source = block["source"]
            provider = block["generating_provider"]
            event = block["analysis_event_type"]
        else:
            pair_id = None
            context = None
            scores = {condition: by_condition[condition].get("formal_score") for condition in ("clean", "corrupted", "enforced")}
            answer_hashes = {condition: by_condition[condition].get("answer_sha256") for condition in ("clean", "corrupted", "enforced")}
            source = str(first.get("source"))
            provider = str(first.get("generating_provider"))
            event = first.get("event_macro")
        resolved = all(value is not None for value in scores.values())
        response_identity_evaluable = all(
            isinstance(answer_hashes[condition], str) and answer_hashes[condition]
            for condition in ("clean", "corrupted")
        )
        response_identity_changed = (
            answer_hashes["clean"] != answer_hashes["corrupted"]
            if response_identity_evaluable
            else None
        )
        delta = (
            float(scores["corrupted"]) - float(scores["clean"])
            if resolved
            else None
        )
        score_identity_changed: bool | None = None
        if source == "longmemeval":
            unit = "binary_correctness"
            corruption_observation = (
                bool(scores["clean"] == 1 and scores["corrupted"] == 0)
                if resolved
                else None
            )
            if resolved:
                score_identity_changed = scores["clean"] != scores["corrupted"]
        elif source == "memora":
            unit = "fama_continuous_0_to_1"
            corruption_observation = bool(delta < 0) if resolved else None
            score_identity_changed = (
                float(scores["clean"]) != float(scores["corrupted"])
                if resolved
                else None
            )
        else:
            unit = "unsupported_native_unit"
            corruption_observation = None
            score_identity_changed = None
        records.append(
            {
                "pilot_block_id": block_id,
                "base_id": base_id,
                "pair_id": pair_id,
                "source_dataset": source,
                "analysis_event_type": event,
                "fault_operator": fault_operator,
                "generating_provider": provider,
                "condition_ids": {
                    condition: by_condition[condition].get("condition_id") for condition in ("clean", "corrupted", "enforced")
                },
                "native_resolved": resolved,
                "native_score_unit": unit,
                "native_clean_score": scores["clean"],
                "native_corrupted_score": scores["corrupted"],
                "native_enforced_score": scores["enforced"],
                "native_corruption_score_delta": delta,
                "native_longmemeval_induced_failure": (
                    corruption_observation if unit == "binary_correctness" else None
                ),
                "native_memora_score_decreased": (
                    corruption_observation if unit == "fama_continuous_0_to_1" else None
                ),
                "native_response_identity_evaluable": response_identity_evaluable,
                "native_response_identity_unavailable": not response_identity_evaluable,
                "native_response_identity_changed": response_identity_changed,
                "native_response_identity_unchanged": (
                    not response_identity_changed
                    if response_identity_changed is not None
                    else None
                ),
                "native_score_identity_changed": score_identity_changed,
                "native_score_identity_unchanged": (
                    not score_identity_changed
                    if score_identity_changed is not None
                    else None
                ),
                "native_response_identity_unchanged_score_changed": (
                    bool(not response_identity_changed and score_identity_changed)
                    if response_identity_changed is not None and score_identity_changed is not None
                    else None
                ),
                "native_context_changed_response_identity_unchanged": (
                    bool(context["context_changed"] and not response_identity_changed)
                    if context is not None and response_identity_changed is not None
                    else None
                ),
                "native_context_unchanged_response_identity_changed": (
                    bool(not context["context_changed"] and response_identity_changed)
                    if context is not None and response_identity_changed is not None
                    else None
                ),
                "context_changed": context["context_changed"] if context else None,
            }
        )
    return records


def _metric(
    *,
    breakdown: str,
    breakdown_value: str,
    metric_family: str,
    metric: str,
    provider: str,
    analysis_unit: str,
    numerator: int | float | None,
    denominator: int,
    estimate: float | None,
    estimate_unit: str,
    base_clusters: int,
    denominator_definition: str,
    iterations: int,
) -> dict[str, Any]:
    return {
        "breakdown": breakdown,
        "breakdown_value": breakdown_value,
        "metric_family": metric_family,
        "metric": metric,
        "provider": provider,
        "analysis_unit": analysis_unit,
        "numerator": numerator,
        "denominator": denominator,
        "estimate": estimate,
        "estimate_unit": estimate_unit,
        "base_clusters": base_clusters,
        "denominator_definition": denominator_definition,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": iterations,
        "bootstrap_unit": "base_id cluster",
    }


def _rate_row(
    records: Sequence[dict[str, Any]],
    *,
    indicator: str,
    breakdown: str,
    breakdown_value: str,
    metric_family: str,
    provider: str,
    analysis_unit: str,
    denominator_definition: str,
    denominator_records: Sequence[dict[str, Any]] | None = None,
    iterations: int,
) -> dict[str, Any] | None:
    eligible = [row for row in records if row.get(indicator) is not None]
    denominator = denominator_records if denominator_records is not None else eligible
    if not denominator:
        return None
    numerator = sum(bool(row.get(indicator)) for row in eligible)
    return _metric(
        breakdown=breakdown,
        breakdown_value=breakdown_value,
        metric_family=metric_family,
        metric=indicator,
        provider=provider,
        analysis_unit=analysis_unit,
        numerator=numerator,
        denominator=len(denominator),
        estimate=numerator / len(denominator),
        estimate_unit="proportion",
        base_clusters=len({row.get("base_id") for row in denominator if row.get("base_id")}),
        denominator_definition=denominator_definition,
        iterations=iterations,
    )


def _mean_row(
    records: Sequence[dict[str, Any]],
    *,
    value_field: str,
    breakdown: str,
    breakdown_value: str,
    metric_family: str,
    provider: str,
    analysis_unit: str,
    denominator_definition: str,
    iterations: int,
) -> dict[str, Any] | None:
    eligible = [row for row in records if row.get(value_field) is not None]
    if not eligible:
        return None
    return _metric(
        breakdown=breakdown,
        breakdown_value=breakdown_value,
        metric_family=metric_family,
        metric=value_field,
        provider=provider,
        analysis_unit=analysis_unit,
        numerator=None,
        denominator=len(eligible),
        estimate=statistics.fmean(float(row[value_field]) for row in eligible),
        estimate_unit="score_points",
        base_clusters=len({row.get("base_id") for row in eligible if row.get("base_id")}),
        denominator_definition=denominator_definition,
        iterations=iterations,
    )


def _context_metric_rows(
    records: Sequence[dict[str, Any]],
    breakdown: str,
    value: str,
    iterations: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = (
        ("context_changed", "context_projection", "all matched pairs"),
        (
            "context_added_item_relative_to_clean",
            "context_projection",
            "all matched pairs; added item relative to the evidence-supported clean projection",
        ),
        (
            "context_removed_item_relative_to_clean",
            "context_projection",
            "all matched pairs; removed item relative to the evidence-supported clean projection",
        ),
        ("other_context_change", "context_projection", "all matched pairs"),
        ("enforced_context_restored", "context_projection", "all matched pairs"),
    )
    for indicator, family, denominator_definition in metrics:
        row = _rate_row(
            records,
            indicator=indicator,
            breakdown=breakdown,
            breakdown_value=value,
            metric_family=family,
            provider="ALL",
            analysis_unit="matched_fault_pair",
            denominator_definition=denominator_definition,
            iterations=iterations,
        )
        if row is not None:
            rows.append(row)
    return rows


def _proxy_metric_rows(
    records: Sequence[dict[str, Any]],
    breakdown: str,
    value: str,
    provider: str,
    iterations: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scoreable = [row for row in records if row["proxy_pair_scoreable"]]
    clean_correct = [row for row in scoreable if row["proxy_clean_correct"] is True]
    unresolved = _rate_row(
        records,
        indicator="proxy_pair_unresolved",
        breakdown=breakdown,
        breakdown_value=value,
        metric_family="frozen_proxy_binary",
        provider=provider,
        analysis_unit="provider_matched_pair",
        denominator_definition="all provider-replicated matched pairs",
        iterations=iterations,
    )
    if unresolved is not None:
        rows.append(unresolved)
    enforced_unresolved = _rate_row(
        records,
        indicator="proxy_enforced_unresolved",
        breakdown=breakdown,
        breakdown_value=value,
        metric_family="frozen_proxy_binary",
        provider=provider,
        analysis_unit="provider_matched_pair",
        denominator_definition="all provider-replicated matched pairs; enforced output unavailable",
        iterations=iterations,
    )
    if enforced_unresolved is not None:
        rows.append(enforced_unresolved)
    recovery_unresolved = _rate_row(
        records,
        indicator="proxy_recovery_pair_unresolved",
        breakdown=breakdown,
        breakdown_value=value,
        metric_family="frozen_proxy_binary",
        provider=provider,
        analysis_unit="provider_matched_pair",
        denominator_definition="all provider-replicated matched pairs; corrupted or enforced score unavailable",
        iterations=iterations,
    )
    if recovery_unresolved is not None:
        rows.append(recovery_unresolved)
    for indicator in (
        "proxy_clean_correct",
        "proxy_corruption_induced_failure",
        "proxy_correctness_outcome_changed",
        "proxy_correctness_improved",
        "proxy_context_changed_correctness_unchanged",
        "proxy_context_unchanged_correctness_changed",
    ):
        row = _rate_row(
            scoreable,
            indicator=indicator,
            breakdown=breakdown,
            breakdown_value=value,
            metric_family="frozen_proxy_binary",
            provider=provider,
            analysis_unit="provider_matched_pair",
            denominator_definition="provider pairs with scoreable clean and corrupted binary outcomes",
            iterations=iterations,
        )
        if row is not None:
            rows.append(row)
    row = _rate_row(
        clean_correct,
        indicator="proxy_corruption_induced_failure",
        breakdown=breakdown,
        breakdown_value=value,
        metric_family="frozen_proxy_binary",
        provider=provider,
        analysis_unit="provider_matched_pair",
        denominator_definition="provider pairs correct under clean with scoreable corrupted outcome",
        iterations=iterations,
    )
    if row is not None:
        row["metric"] = "proxy_induced_failure_given_clean_correct"
        rows.append(row)
    recovery_records = [row for row in records if row["proxy_recovery_pair_scoreable"]]
    recovery_success = _rate_row(
        recovery_records,
        indicator="proxy_recovery_success",
        breakdown=breakdown,
        breakdown_value=value,
        metric_family="frozen_proxy_binary",
        provider=provider,
        analysis_unit="provider_matched_pair",
        denominator_definition="provider pairs with scoreable corrupted and enforced binary outcomes",
        iterations=iterations,
    )
    if recovery_success is not None:
        rows.append(recovery_success)
    return rows


def _native_metric_rows(
    records: Sequence[dict[str, Any]],
    breakdown: str,
    value: str,
    provider: str,
    iterations: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resolved = [row for row in records if row["native_resolved"]]
    lme = [row for row in resolved if row["native_score_unit"] == "binary_correctness"]
    fama = [row for row in resolved if row["native_score_unit"] == "fama_continuous_0_to_1"]
    for indicator, subset, family, definition in (
        (
            "native_longmemeval_induced_failure",
            lme,
            "native_longmemeval_binary",
            "resolved LongMemEval matched blocks; clean correct and corrupted incorrect",
        ),
        (
            "native_score_identity_changed",
            lme,
            "native_longmemeval_binary",
            "resolved LongMemEval blocks; clean and corrupted binary correctness differ",
        ),
        (
            "native_score_identity_unchanged",
            lme,
            "native_longmemeval_binary",
            "resolved LongMemEval blocks; clean and corrupted binary correctness agree",
        ),
        (
            "native_response_identity_unavailable",
            lme,
            "native_longmemeval_response_identity",
            "resolved LongMemEval blocks; one or both clean/corrupted answer hashes unavailable",
        ),
        (
            "native_response_identity_changed",
            lme,
            "native_longmemeval_response_identity",
            "resolved LongMemEval blocks with both clean and corrupted answer hashes",
        ),
        (
            "native_response_identity_unchanged",
            lme,
            "native_longmemeval_response_identity",
            "resolved LongMemEval blocks with both clean and corrupted answer hashes",
        ),
        (
            "native_context_changed_response_identity_unchanged",
            lme,
            "native_longmemeval_response_identity",
            "resolved LongMemEval blocks with response identity and operative context",
        ),
        (
            "native_context_unchanged_response_identity_changed",
            lme,
            "native_longmemeval_response_identity",
            "resolved LongMemEval blocks with response identity and operative context",
        ),
        (
            "native_response_identity_unchanged_score_changed",
            lme,
            "native_longmemeval_response_identity",
            "resolved LongMemEval blocks with clean/corrupted response identity and binary correctness",
        ),
        (
            "native_memora_score_decreased",
            fama,
            "native_memora_fama",
            "resolved Memora matched blocks; corrupted FAMA below clean FAMA",
        ),
        (
            "native_score_identity_changed",
            fama,
            "native_memora_fama",
            "resolved Memora blocks; clean and corrupted FAMA differ",
        ),
        (
            "native_score_identity_unchanged",
            fama,
            "native_memora_fama",
            "resolved Memora blocks; clean and corrupted FAMA agree",
        ),
        (
            "native_response_identity_unavailable",
            fama,
            "native_memora_response_identity",
            "resolved Memora blocks; one or both clean/corrupted answer hashes unavailable",
        ),
        (
            "native_response_identity_changed",
            fama,
            "native_memora_response_identity",
            "resolved Memora blocks with both clean and corrupted answer hashes",
        ),
        (
            "native_response_identity_unchanged",
            fama,
            "native_memora_response_identity",
            "resolved Memora blocks with both clean and corrupted answer hashes",
        ),
        (
            "native_context_changed_response_identity_unchanged",
            fama,
            "native_memora_response_identity",
            "resolved Memora blocks with response identity and operative context",
        ),
        (
            "native_context_unchanged_response_identity_changed",
            fama,
            "native_memora_response_identity",
            "resolved Memora blocks with response identity and operative context",
        ),
        (
            "native_response_identity_unchanged_score_changed",
            fama,
            "native_memora_response_identity",
            "resolved Memora blocks with clean/corrupted response identity and FAMA",
        ),
    ):
        row = _rate_row(
            subset,
            indicator=indicator,
            breakdown=breakdown,
            breakdown_value=value,
            metric_family=family,
            provider=provider,
            analysis_unit="native_matched_block",
            denominator_definition=definition,
            iterations=iterations,
        )
        if row is not None:
            rows.append(row)
    mean_delta = _mean_row(
        fama,
        value_field="native_corruption_score_delta",
        breakdown=breakdown,
        breakdown_value=value,
        metric_family="native_memora_fama",
        provider=provider,
        analysis_unit="native_matched_block",
        denominator_definition="resolved Memora matched blocks",
        iterations=iterations,
    )
    if mean_delta is not None:
        mean_delta["metric"] = "native_memora_corruption_score_delta"
        rows.append(mean_delta)
    return rows


def _breakdown_values(
    contexts: Sequence[dict[str, Any]],
    proxies: Sequence[dict[str, Any]],
    natives: Sequence[dict[str, Any]],
    field: str,
) -> list[str]:
    values = {
        str(row[field])
        for row in (*contexts, *proxies, *natives)
        if row.get(field) not in (None, "")
    }
    return sorted(values)


def build_stratified_rows(
    contexts: Sequence[dict[str, Any]],
    proxies: Sequence[dict[str, Any]],
    natives: Sequence[dict[str, Any]],
    *,
    breakdown: str,
    field: str,
    iterations: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in _breakdown_values(contexts, proxies, natives, field):
        context_subset = [row for row in contexts if str(row[field]) == value]
        proxy_subset = [row for row in proxies if str(row[field]) == value]
        native_subset = [row for row in natives if str(row[field]) == value]
        rows.extend(_context_metric_rows(context_subset, breakdown, value, iterations))
        for provider in sorted({str(row["provider"]) for row in proxy_subset}):
            rows.extend(
                _proxy_metric_rows(
                    [row for row in proxy_subset if row["provider"] == provider],
                    breakdown,
                    value,
                    provider,
                    iterations,
                )
            )
        for provider in sorted({str(row["generating_provider"]) for row in native_subset}):
            provider_subset = [
                row for row in native_subset if row["generating_provider"] == provider
            ]
            rows.extend(
                _native_metric_rows(provider_subset, breakdown, value, provider, iterations)
            )
            for source in sorted({str(row["source_dataset"]) for row in provider_subset}):
                source_records = [
                    row for row in provider_subset if row["source_dataset"] == source
                ]
                unresolved = sum(not row["native_resolved"] for row in source_records)
                if unresolved:
                    rows.append(
                        _metric(
                            breakdown=breakdown,
                            breakdown_value=value,
                            metric_family=f"native_{source}",
                            metric="native_unresolved_block",
                            provider=provider,
                            analysis_unit="native_block",
                            numerator=unresolved,
                            denominator=len(source_records),
                            estimate=unresolved / len(source_records),
                            estimate_unit="proportion",
                            base_clusters=len(
                                {
                                    row["base_id"]
                                    for row in source_records
                                    if row.get("base_id")
                                }
                            ),
                            denominator_definition="all observed native blocks in this selected panel",
                            iterations=iterations,
                        )
                    )
    return rows


def build_provider_rows(
    contexts: Sequence[dict[str, Any]],
    proxies: Sequence[dict[str, Any]],
    natives: Sequence[dict[str, Any]],
    iterations: int,
) -> list[dict[str, Any]]:
    rows = _context_metric_rows(contexts, "provider", "ALL", iterations)
    for provider in sorted({str(row["provider"]) for row in proxies}):
        subset = [row for row in proxies if row["provider"] == provider]
        rows.extend(_proxy_metric_rows(subset, "provider", provider, provider, iterations))
    for provider in sorted({str(row["generating_provider"]) for row in natives}):
        subset = [row for row in natives if row["generating_provider"] == provider]
        rows.extend(_native_metric_rows(subset, "provider", provider, provider, iterations))
        for source in sorted({str(row["source_dataset"]) for row in subset}):
            source_records = [row for row in subset if row["source_dataset"] == source]
            unresolved = sum(not row["native_resolved"] for row in source_records)
            if unresolved:
                rows.append(
                    _metric(
                        breakdown="provider",
                        breakdown_value=provider,
                        metric_family=f"native_{source}",
                        metric="native_unresolved_block",
                        provider=provider,
                        analysis_unit="native_block",
                        numerator=unresolved,
                        denominator=len(source_records),
                        estimate=unresolved / len(source_records),
                        estimate_unit="proportion",
                        base_clusters=len(
                            {
                                row["base_id"]
                                for row in source_records
                                if row.get("base_id")
                            }
                        ),
                        denominator_definition="all observed native blocks in this selected panel",
                        iterations=iterations,
                    )
                )
    return rows


def _overlap_counts(
    records: Sequence[dict[str, Any]],
    *,
    context_indicator: str,
    output_indicator: str,
) -> dict[str, int] | None:
    eligible = [
        row
        for row in records
        if row.get(context_indicator) is not None and row.get(output_indicator) is not None
    ]
    if not eligible:
        return None
    both = sum(bool(row[context_indicator]) and bool(row[output_indicator]) for row in eligible)
    context_only = sum(bool(row[context_indicator]) and not bool(row[output_indicator]) for row in eligible)
    output_only = sum(not bool(row[context_indicator]) and bool(row[output_indicator]) for row in eligible)
    neither = len(eligible) - both - context_only - output_only
    return {
        "eligible": len(eligible),
        "both": both,
        "context_only": context_only,
        "output_only": output_only,
        "neither": neither,
    }


def build_overlap_rows(
    proxies: Sequence[dict[str, Any]],
    natives: Sequence[dict[str, Any]],
    iterations: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider in sorted({str(row["provider"]) for row in proxies}):
        records = [
            row
            for row in proxies
            if row["provider"] == provider and row["proxy_pair_scoreable"]
        ]
        counts = _overlap_counts(
            records,
            context_indicator="context_changed",
            output_indicator="proxy_corruption_induced_failure",
        )
        if counts:
            category_names = {
                "both": "context_changed_with_proxy_induced_failure",
                "context_only": "context_changed_without_proxy_induced_failure",
                "output_only": "context_unchanged_with_proxy_induced_failure",
                "neither": "context_unchanged_without_proxy_induced_failure",
            }
            for category, metric in category_names.items():
                rows.append(
                    _metric(
                        breakdown="provider",
                        breakdown_value=provider,
                        metric_family="frozen_proxy_binary",
                        metric=metric,
                        provider=provider,
                        analysis_unit="provider_matched_pair",
                        numerator=counts[category],
                        denominator=counts["eligible"],
                        estimate=counts[category] / counts["eligible"],
                        estimate_unit="proportion",
                        base_clusters=len({row["base_id"] for row in records}),
                        denominator_definition=(
                            "provider pairs with scoreable clean and corrupted outcomes; "
                            "selected event is clean-correct -> corrupted-incorrect"
                        ),
                        iterations=iterations,
                    )
                )

    for source, output_indicator, family in (
        ("longmemeval", "native_longmemeval_induced_failure", "native_longmemeval_binary"),
        ("memora", "native_memora_score_decreased", "native_memora_fama"),
    ):
        for provider in sorted({str(row["generating_provider"]) for row in natives}):
            records = [
                row
                for row in natives
                if row["source_dataset"] == source
                and row["generating_provider"] == provider
                and row["native_resolved"]
            ]
            counts = _overlap_counts(
                records,
                context_indicator="context_changed",
                output_indicator=output_indicator,
            )
            if not counts:
                continue
            event_name = "lme_binary_failure" if source == "longmemeval" else "fama_decrease"
            category_names = {
                "both": f"context_changed_with_{event_name}",
                "context_only": f"context_changed_without_{event_name}",
                "output_only": f"context_unchanged_with_{event_name}",
                "neither": f"context_unchanged_without_{event_name}",
            }
            for category, metric in category_names.items():
                rows.append(
                    _metric(
                        breakdown="provider",
                        breakdown_value=provider,
                        metric_family=family,
                        metric=metric,
                        provider=provider,
                        analysis_unit="native_matched_block",
                        numerator=counts[category],
                        denominator=counts["eligible"],
                        estimate=counts[category] / counts["eligible"],
                        estimate_unit="proportion",
                        base_clusters=len({row["base_id"] for row in records if row.get("base_id")}),
                        denominator_definition=(
                            f"resolved {source} blocks in the selected native panel; "
                            + (
                                "output event is clean correct -> corrupted incorrect"
                                if source == "longmemeval"
                                else "output event is a negative FAMA delta"
                            )
                        ),
                        iterations=iterations,
                    )
                )
    return rows


def _source_summary(records: Sequence[dict[str, Any]], source: str, iterations: int) -> dict[str, Any]:
    subset = [row for row in records if row["source_dataset"] == source]
    if source == "longmemeval":
        event = "native_longmemeval_induced_failure"
    else:
        event = "native_memora_score_decreased"
    resolved = [row for row in subset if row["native_resolved"]]
    return {
        "resolved_blocks": sum(row["native_resolved"] for row in subset),
        "observed_blocks": len(subset),
        "unresolved_blocks": sum(not row["native_resolved"] for row in subset),
        "output_observation": event,
        "output_observation_rate": cluster_bootstrap_rate(
            resolved, event, iterations=iterations
        ),
        "score_identity_changed": cluster_bootstrap_rate(
            resolved, "native_score_identity_changed", iterations=iterations
        ),
        "score_identity_unchanged": cluster_bootstrap_rate(
            resolved, "native_score_identity_unchanged", iterations=iterations
        ),
        "response_identity_changed": cluster_bootstrap_rate(
            resolved, "native_response_identity_changed", iterations=iterations
        ),
        "response_identity_unchanged": cluster_bootstrap_rate(
            resolved, "native_response_identity_unchanged", iterations=iterations
        ),
        "response_identity_unavailable": cluster_bootstrap_rate(
            resolved, "native_response_identity_unavailable", iterations=iterations
        ),
        "context_changed_response_identity_unchanged": cluster_bootstrap_rate(
            resolved,
            "native_context_changed_response_identity_unchanged",
            iterations=iterations,
        ),
        "context_unchanged_response_identity_changed": cluster_bootstrap_rate(
            resolved,
            "native_context_unchanged_response_identity_changed",
            iterations=iterations,
        ),
        "response_identity_unchanged_score_changed": cluster_bootstrap_rate(
            resolved,
            "native_response_identity_unchanged_score_changed",
            iterations=iterations,
        ),
    }


def summarize(
    bundle: AuditBundle,
    contexts: Sequence[dict[str, Any]],
    proxies: Sequence[dict[str, Any]],
    natives: Sequence[dict[str, Any]],
    *,
    iterations: int,
) -> dict[str, Any]:
    provider_summaries: dict[str, Any] = {}
    for provider in sorted({str(row["provider"]) for row in proxies}):
        subset = [row for row in proxies if row["provider"] == provider]
        scoreable = [row for row in subset if row["proxy_pair_scoreable"]]
        provider_summaries[provider] = {
            "provider_pairs": len(subset),
            "scoreable_pairs": len(scoreable),
            "unresolved_pairs": len(subset) - len(scoreable),
            "unresolved_condition_rows": sum(
                value is not None
                for row in subset
                for value in row["proxy_unresolved_reasons"].values()
            ),
            "clean_correct": cluster_bootstrap_rate(scoreable, "proxy_clean_correct", iterations=iterations),
            "corruption_induced_failure": cluster_bootstrap_rate(
                scoreable, "proxy_corruption_induced_failure", iterations=iterations
            ),
            "induced_failure_given_clean_correct": cluster_bootstrap_rate(
                [row for row in scoreable if row["proxy_clean_correct"] is True],
                "proxy_corruption_induced_failure",
                iterations=iterations,
            ),
            "correctness_outcome_changed": cluster_bootstrap_rate(
                scoreable, "proxy_correctness_outcome_changed", iterations=iterations
            ),
        }

    lme = [
        row
        for row in natives
        if row["source_dataset"] == "longmemeval" and row["native_resolved"]
    ]
    fama = [
        row
        for row in natives
        if row["source_dataset"] == "memora" and row["native_resolved"]
    ]
    native_summary = {
        "panel_mode": bundle.report["formal_panel"]["mode"],
        "selected_panel_not_random": True,
        "observed_blocks": len(natives),
        "resolved_blocks": len(lme) + len(fama),
        "unresolved_blocks": sum(not row["native_resolved"] for row in natives),
        "unresolved_observations": bundle.report.get("formal_panel", {}).get(
            "unresolved_observations", 0
        ),
        "longmemeval_binary": _source_summary(natives, "longmemeval", iterations),
        "memora_fama": {
            **_source_summary(natives, "memora", iterations),
            "corruption_score_delta": cluster_bootstrap_mean(
                fama, "native_corruption_score_delta", iterations=iterations
            ),
        },
    }
    return {
        "status": "offline_rq1_observability_analysis",
        "schema_version": "jss-rq1-observability-v2",
        "api_calls_made": 0,
        "no_model_generation_performed": True,
        "no_rescoring_performed": True,
        "provenance": bundle.report,
        "cohort": {
            "bases": bundle.report["condition_manifest"]["counts"]["base_count"],
            "condition_rows": bundle.report["condition_manifest"]["counts"]["condition_count"],
            "matched_fault_pairs": len(contexts),
            "provider_replications": len(proxies),
            "native_resolved_blocks": native_summary["resolved_blocks"],
        },
        "state_observability": {
            "status": "not_estimable",
            "state_detector_implemented": False,
            "state_faulted_record_count": 0,
            "diagnostic_internal_state_unchanged": sum(
                not row["state_internal_changed"] for row in contexts
            ),
            "diagnostic_pair_count": len(contexts),
            "labels_used_as_independent_detections": False,
            "precision_recall_fpr_fnr_estimable": False,
            "anomaly_prevalence_estimable": False,
            "reason_not_estimable": (
                "The frozen corruption changes the model-visible projection; internal_governance_state "
                "is unchanged and copied from oracle_state, so there is no independent population of "
                "faulted state records or state-detector positives."
            ),
        },
        "context_projection_observability": {
            "context_changed": cluster_bootstrap_rate(contexts, "context_changed", iterations=iterations),
            "context_added_item_relative_to_clean": cluster_bootstrap_rate(
                contexts, "context_added_item_relative_to_clean", iterations=iterations
            ),
            "context_removed_item_relative_to_clean": cluster_bootstrap_rate(
                contexts, "context_removed_item_relative_to_clean", iterations=iterations
            ),
            "other_context_change": cluster_bootstrap_rate(contexts, "other_context_change", iterations=iterations),
            "enforced_context_restored": cluster_bootstrap_rate(
                contexts, "enforced_context_restored", iterations=iterations
            ),
            "classification_reference": (
                "Frozen clean oracle projection, whose source evidence is recorded as fully supported "
                "by the governance-evidence audit. Addition/removal counts are relative to this "
                "reference and are not, by themselves, absolute policy-violation detections."
            ),
        },
        "downstream_output_observability": {
            "score_units_are_separate": True,
            "frozen_proxy_binary": {
                "score_unit": "binary_correctness",
                "output_text_identity_estimable": False,
                "output_text_identity_reason": (
                    "Sanitized frozen provider files expose scored outcomes and context hashes, "
                    "not raw response text or response hashes."
                ),
                "providers": provider_summaries,
            },
            "native_style": native_summary,
            "overlap_definitions": {
                "frozen_proxy": "context state x clean-correct-to-corrupted-incorrect event",
                "longmemeval": "context state x LongMemEval binary clean-correct-to-corrupted-incorrect event",
                "memora": "context state x negative continuous FAMA delta",
            },
        },
        "estimable_metrics": [
            "context-projection change count",
            "context item-addition count relative to the supported clean oracle projection",
            "context item-removal count relative to the supported clean oracle projection",
            "frozen binary proxy correctness identity, improvement, induced failure, and recovery",
            "LongMemEval native binary corruption-induced failure count",
            "Memora native FAMA decrease count and mean paired score difference",
            "native clean/corrupted response identity where both hashes exist",
            "context-versus-selected-output-event overlap within each score-unit family",
            "source, fault, event, and provider stratified counts with base-cluster uncertainty",
        ],
        "not_estimable": {
            "state_detector_precision_recall_fpr_fnr": (
                "No independently faulted internal-state observations and no negative-state detector population."
            ),
            "state_to_output_detection_overlap": (
                "The state record does not carry the injected anomaly; using labels would be circular."
            ),
            "frozen_proxy_output_text_identity": (
                "Raw text/answer hashes are absent from the sanitized provider artifacts."
            ),
            "combined_longmemeval_memora_accuracy": (
                "LongMemEval binary correctness and Memora FAMA are incompatible score units."
            ),
            "native_representative_population_effect": (
                "Formal-300 is a fixed selected validation panel, not a random sample of the full cohort."
            ),
            "native_response_identity_with_missing_hashes": (
                "Response identity is unavailable whenever either clean or corrupted answer hash is absent."
            ),
            "provider_detection_accuracy_ranking": (
                "Provider outcomes have different score surfaces and repeated base observations; "
                "the analysis reports strata rather than a common detection-accuracy ranking."
            ),
            "causal_generalization": (
                "The deterministic matched interventions support design-specific contrasts, "
                "not population or causal claims outside the frozen cohort."
            ),
        },
        "statistical_design": {
            "uncertainty": "base-cluster bootstrap",
            "base_cluster_field": "base_id",
            "repeated_observations_accounted_for": True,
            "paired_conditions": ["clean", "corrupted", "enforced"],
            "iterations": iterations,
            "seed": BOOTSTRAP_SEED,
            "provider_rows_pooled_into_common_accuracy": False,
        },
    }


def limitations_markdown(summary: dict[str, Any]) -> str:
    context = summary["context_projection_observability"]
    native = summary["downstream_output_observability"]["native_style"]
    return f"""# RQ1 observability limitations

This supplementary analysis is an offline reanalysis of frozen GovMem
artifacts. No provider API was called, no model output was generated, and no
answer was re-scored.

## Verified scope

- {summary['cohort']['bases']} base examples.
- {summary['cohort']['matched_fault_pairs']} matched clean/corrupted/enforced fault triplets.
- {summary['cohort']['provider_replications']} provider-replicated pair observations.
- Native panel `{native['panel_mode']}`: {native['resolved_blocks']} resolved Formal-300 triplets and {native['unresolved_blocks']} unresolved selected-panel blocks.

## State-level boundary

The frozen corruption is realised in `model_visible_memory_items` /
`model_visible_context_text`. `internal_governance_state` is unchanged and is a
copy of `oracle_state`. Consequently the injected fault label cannot serve as
an independent state detection. State anomaly prevalence and state-detector
precision, recall, false-positive rate, and false-negative rate are not
estimable. The zero internal-state-change count is retained only as a
representation diagnostic.

## Context-level interpretation

There are {context['context_changed']['numerator']} context changes among
{context['context_changed']['denominator']} matched pairs. Item additions and
removals are counted relative to the evidence-supported clean projection. A
context change is not, by itself, interpreted as an absolute policy violation.

## Downstream score boundaries

Frozen proxy outcomes, LongMemEval binary native correctness, and Memora FAMA
remain separate. They are not combined into one accuracy ranking. The sanitized
provider results do not contain response text or response hashes, so actual
output identity under frozen proxy scoring is not estimable. Formal-300 does
contain answer hashes, enabling clean-versus-corrupted response-identity counts
only for resolved selected-panel blocks where both hashes exist. Response
identity is never treated as correctness identity.

The Formal-300 panel is a fixed selected validation panel. It is not a random
or representative sample of the 731-base cohort, and subgroup comparisons are
descriptive for that panel. Native panel results are not available for OAKS
Novel, so source-specific native comparisons are limited to LongMemEval and
Memora.

## Inference

Uncertainty uses a base-cluster bootstrap because multiple fault pairs share a
base. Provider-replicated observations are not pooled as independent data.
The manuscript's Formal-300 matched-block bootstrap is not replaced by this
supplementary analysis.
The matched deterministic design supports design-specific observational
contrasts only; it does not establish general causal effects or performance
rankings across different detection targets.
"""


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    if not rows:
        raise ValueError(f"Refusing to create an empty table: {path.name}")
    fieldnames = list(rows[0])
    for row in rows:
        if set(row) != set(fieldnames):
            raise ValueError(f"Inconsistent row schema for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def write_outputs(output_dir: Path, summary: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    targets = [output_dir / name for name in OUTPUT_FILES]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing analysis artifact(s): "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "observability_summary.json", summary)
    for name, rows in tables.items():
        _write_csv(output_dir / name, rows)
    limitations_path = output_dir / "observability_limitations.md"
    limitations_path.write_text(limitations_markdown(summary), encoding="utf-8")


def analyze(
    *,
    condition_manifest: Path,
    base_manifest: Path,
    condition_freeze: Path,
    base_freeze: Path,
    results_dir: Path,
    native_panel: str,
    formal_observations: Path,
    formal_blocks: Path,
    formal_summary: Path,
    final_formal_blocks: Path,
    final_formal_metrics: Path,
    human_adjudication: Path,
    semantic_payloads: Path,
    evidence_base_level: Path,
    output_dir: Path,
    iterations: int = 5000,
) -> dict[str, Any]:
    bundle = audit_repository(
        condition_manifest=condition_manifest,
        base_manifest=base_manifest,
        condition_freeze=condition_freeze,
        base_freeze=base_freeze,
        results_dir=results_dir,
        native_panel=native_panel,
        formal_observations=formal_observations,
        formal_blocks=formal_blocks,
        formal_summary=formal_summary,
        final_formal_blocks=final_formal_blocks,
        final_formal_metrics=final_formal_metrics,
        human_adjudication=human_adjudication,
        semantic_payloads=semantic_payloads,
        evidence_base_level=evidence_base_level,
    )
    contexts = build_context_records(bundle)
    proxies = build_proxy_records(contexts, bundle.provider_results)
    natives = build_native_records(bundle, contexts)
    summary = summarize(bundle, contexts, proxies, natives, iterations=iterations)
    tables = {
        "observability_by_source.csv": build_stratified_rows(
            contexts,
            proxies,
            natives,
            breakdown="source",
            field="source_dataset",
            iterations=iterations,
        ),
        "observability_by_fault.csv": build_stratified_rows(
            contexts,
            proxies,
            natives,
            breakdown="fault",
            field="fault_operator",
            iterations=iterations,
        ),
        "observability_by_event.csv": build_stratified_rows(
            contexts,
            proxies,
            natives,
            breakdown="event",
            field="analysis_event_type",
            iterations=iterations,
        ),
        "observability_by_provider.csv": build_provider_rows(
            contexts, proxies, natives, iterations=iterations
        ),
        "observability_overlap.csv": build_overlap_rows(
            proxies, natives, iterations=iterations
        ),
    }
    write_outputs(output_dir, summary, tables)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-manifest", type=Path, default=DEFAULT_CONDITION_MANIFEST)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--condition-freeze", type=Path, default=DEFAULT_CONDITION_FREEZE)
    parser.add_argument("--base-freeze", type=Path, default=DEFAULT_BASE_FREEZE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--native-panel", choices=NATIVE_PANEL_MODES, default="paper_final_298")
    parser.add_argument("--formal-observations", type=Path, default=DEFAULT_FORMAL_OBSERVATIONS)
    parser.add_argument("--formal-blocks", type=Path, default=DEFAULT_FORMAL_BLOCKS)
    parser.add_argument("--formal-summary", type=Path, default=DEFAULT_FORMAL_SUMMARY)
    parser.add_argument("--final-formal-blocks", type=Path, default=DEFAULT_FINAL_FORMAL_BLOCKS)
    parser.add_argument("--final-formal-metrics", type=Path, default=DEFAULT_FINAL_FORMAL_METRICS)
    parser.add_argument("--human-adjudication", type=Path, default=DEFAULT_HUMAN_ADJUDICATION)
    parser.add_argument("--semantic-payloads", type=Path, default=DEFAULT_SEMANTIC_PAYLOADS)
    parser.add_argument("--evidence-base-level", type=Path, default=DEFAULT_EVIDENCE_BASE_LEVEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.bootstrap_iterations < 1:
        print("--bootstrap-iterations must be positive", file=sys.stderr)
        return 2
    try:
        summary = analyze(
            condition_manifest=args.condition_manifest,
            base_manifest=args.base_manifest,
            condition_freeze=args.condition_freeze,
            base_freeze=args.base_freeze,
            results_dir=args.results_dir,
            native_panel=args.native_panel,
            formal_observations=args.formal_observations,
            formal_blocks=args.formal_blocks,
            formal_summary=args.formal_summary,
            final_formal_blocks=args.final_formal_blocks,
            final_formal_metrics=args.final_formal_metrics,
            human_adjudication=args.human_adjudication,
            semantic_payloads=args.semantic_payloads,
            evidence_base_level=args.evidence_base_level,
            output_dir=args.output_dir,
            iterations=args.bootstrap_iterations,
        )
    except (AuditError, FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as exc:
        print(f"JSS_RQ1_ANALYSIS_BLOCKED={exc}", file=sys.stderr)
        return 2
    print(f"JSS_RQ1_BASES={summary['cohort']['bases']}")
    print(f"JSS_RQ1_FAULT_PAIRS={summary['cohort']['matched_fault_pairs']}")
    print(
        "JSS_RQ1_NATIVE_PANEL="
        f"{summary['downstream_output_observability']['native_style']['panel_mode']}"
    )
    print(f"JSS_RQ1_PROVIDER_REPLICATIONS={summary['cohort']['provider_replications']}")
    print(f"JSS_RQ1_NATIVE_RESOLVED_BLOCKS={summary['cohort']['native_resolved_blocks']}")
    print("JSS_RQ1_API_CALLS_PERFORMED=0")
    print("JSS_RQ1_MODEL_GENERATIONS_PERFORMED=0")
    print(f"JSS_RQ1_OUTPUT_DIR={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
