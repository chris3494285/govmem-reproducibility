"""Focused tests for the offline JSS RQ1 observability analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.analyze_jss_rq1_observability import (
    OUTPUT_FILES,
    _proxy_metric_rows,
    build_context_records,
    build_native_records,
    build_overlap_rows,
    build_provider_rows,
    build_proxy_records,
    build_stratified_rows,
    cluster_bootstrap_rate,
    summarize,
    write_outputs,
)
from scripts.audit_jss_rq1_observability import (
    AuditBundle,
    AuditError,
    _state_issues,
    audit_formal_panel,
    audit_repository,
    build_triplets,
    parse_proxy_outcome,
)


def _condition(base_id: str, condition_type: str, *, items: list[str], fault: str | None = None):
    pair_id = f"{base_id}::{fault}" if fault else None
    condition_id = f"{base_id}::clean" if condition_type == "clean" else f"{pair_id}::{condition_type}"
    state = {
        "current_status": "valid",
        "ordered_evidence": [{"ordinal": 0, "session_id": "s0"}],
        "relevant_session_ids": ["s0"],
        "temporal_relation": "knowledge-update",
    }
    return {
        "base_id": base_id,
        "condition_id": condition_id,
        "condition_type": condition_type,
        "pair_id": pair_id,
        "fault_operator": fault,
        "source_dataset": "fixture_source",
        "analysis_event_type": "update",
        "internal_governance_state": state,
        "oracle_state": state,
        "model_visible_memory_items": items,
        "model_visible_context_spec": {
            "content_source": "fixture",
            "item_count": len(items),
            "projection": "admitted_evidence",
        },
        "model_visible_context_text": "\n".join(items),
        "corruption_specification": (
            {
                "operator": fault,
                "deterministic": True,
                "source_oracle_fields": ["current_status"],
            }
            if fault
            else None
        ),
        "enforcement_specification": (
            {"corrects_operator": fault, "deterministic": True}
            if condition_type == "enforced"
            else None
        ),
    }


def _rows(base_id: str = "fixture::b1", fault: str = "missing"):
    return [
        _condition(base_id, "clean", items=["current"]),
        _condition(base_id, "corrupted", items=[], fault=fault),
        _condition(base_id, "enforced", items=["current"], fault=fault),
    ]


def _triplet(base_id: str, fault: str = "missing", *, source: str = "longmemeval", event: str = "update"):
    clean, corrupted, enforced = _rows(base_id, fault)
    clean["source_dataset"] = source
    corrupted["source_dataset"] = source
    enforced["source_dataset"] = source
    clean["analysis_event_type"] = event
    corrupted["analysis_event_type"] = event
    enforced["analysis_event_type"] = event
    return {
        "base_id": base_id,
        "pair_id": f"{base_id}::{fault}",
        "fault_operator": fault,
        "source_dataset": source,
        "analysis_event_type": event,
        "conditions": {"clean": clean, "corrupted": corrupted, "enforced": enforced},
    }


def _bundle(triplets: list[dict]) -> AuditBundle:
    report = {
        "condition_manifest": {
            "counts": {
                "base_count": len({row["base_id"] for row in triplets}),
                "condition_count": len(triplets) * 3,
            }
        },
        "formal_panel": {
            "mode": "public_297",
            "unresolved_observations": 0,
        },
    }
    return AuditBundle(
        report=report,
        triplets=triplets,
        provider_results={},
        formal_observations=[],
        formal_blocks=[],
        evidence_support={row["base_id"]: True for row in triplets},
    )


def _result(score: bool | None, *, status: str = "success", scoring_status: str = "scored"):
    return {
        "status": status,
        "scoring_status": scoring_status,
        "correctness": score,
        "primary_correctness": None,
        "primary_score": None if score is None else int(score),
        "error_type": None if score is not None else "fixture_failure",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _formal_panel_fixture(tmp_path: Path) -> dict[str, object]:
    triplets = [
        _triplet("fixture::resolved", source="longmemeval", event="update"),
        _triplet("fixture::human", source="memora", event="deletion"),
        _triplet("fixture::unresolved-clean", source="longmemeval", event="temporal"),
        _triplet("fixture::unresolved-enforced", source="longmemeval", event="update"),
    ]
    block_ids = {
        "fixture::resolved": "block-resolved",
        "fixture::human": "block-human",
        "fixture::unresolved-clean": "block-unresolved-clean",
        "fixture::unresolved-enforced": "block-unresolved-enforced",
    }
    providers = {
        "fixture::resolved": "openai",
        "fixture::human": "kimi",
        "fixture::unresolved-clean": "gemini",
        "fixture::unresolved-enforced": "deepseek",
    }
    condition_rows = [
        triplet["conditions"][condition]
        for triplet in triplets
        for condition in ("clean", "corrupted", "enforced")
    ]
    observations = []
    for triplet in triplets:
        base_id = triplet["base_id"]
        block_id = block_ids[base_id]
        scores = {
            "clean": 1.0 if triplet["source_dataset"] == "memora" else 0.0,
            "corrupted": 1.0 if triplet["source_dataset"] == "memora" else 0.0,
            "enforced": 1.0 if triplet["source_dataset"] == "memora" else 0.0,
        }
        if base_id == "fixture::resolved":
            scores["clean"] = 1.0
            scores["corrupted"] = 0.0
        if base_id == "fixture::unresolved-clean":
            scores["corrupted"] = None
        if base_id == "fixture::unresolved-enforced":
            scores["enforced"] = None
        if base_id == "fixture::human":
            scores["enforced"] = None
        for condition in ("clean", "corrupted", "enforced"):
            observations.append(
                {
                    "pilot_block_id": block_id,
                    "observation_id": f"{block_id}-{condition}",
                    "condition": condition,
                    "condition_id": triplet["conditions"][condition]["condition_id"],
                    "formal_score": scores[condition],
                    "answer_sha256": f"{block_id}-{condition}-sha",
                    "source": triplet["source_dataset"],
                    "generating_provider": providers[base_id],
                    "fault_type": triplet["fault_operator"],
                    "event_macro": triplet["analysis_event_type"],
                    "resolution_method": "fixture",
                }
            )
    observations_path = tmp_path / "observations.jsonl"
    _write_jsonl(observations_path, observations)

    public_blocks_path = tmp_path / "public_blocks.jsonl"
    _write_jsonl(
        public_blocks_path,
        [
            {
                "pilot_block_id": "block-resolved",
                "source": "longmemeval",
                "generating_provider": "openai",
                "fault_type": "missing",
                "scores": {"clean": 1.0, "corrupted": 0.0, "enforced": 0.0},
            }
        ],
    )
    public_summary_path = tmp_path / "public_summary.json"
    public_summary_path.write_text(
        json.dumps(
            {
                "resolved_matched_triplets": 1,
                "unresolved_observations": 3,
            }
        ),
        encoding="utf-8",
    )

    final_blocks_path = tmp_path / "final_blocks.csv"
    final_blocks_path.write_text(
        "pilot_block_id,source,generating_provider,fault_type,clean_score,corrupted_score,enforced_score\n"
        "block-resolved,longmemeval,openai,missing,1.0,0.0,0.0\n"
        "block-human,memora,kimi,missing,1.0,1.0,0.5\n",
        encoding="utf-8",
    )
    final_metrics_path = tmp_path / "final_metrics.json"
    final_metrics_path.write_text(
        json.dumps(
            {
                "resolved_matched_triplets": 2,
                "unresolved_observations": 2,
                "human_adjudicated_observations": 1,
                "unresolved_details": [
                    {"observation_id": "block-unresolved-clean-corrupted"},
                    {"observation_id": "block-unresolved-enforced-enforced"},
                ],
            }
        ),
        encoding="utf-8",
    )
    human_path = tmp_path / "human.jsonl"
    _write_jsonl(
        human_path,
        [
            {
                "observation_id": "block-human-enforced",
                "pilot_block_id": "block-human",
                "resolution_method": "human_adjudication",
                "decisions": [
                    {"criterion_id": "present", "decision": "pass"},
                    {"criterion_id": "absent", "decision": "fail"},
                ],
            }
        ],
    )
    payload_path = tmp_path / "payloads.jsonl"
    _write_jsonl(
        payload_path,
        [
            {
                "observation_id": "block-human-enforced",
                "condition": "enforced",
                "prompt": json.dumps(
                    {
                        "criteria": [
                            {
                                "criterion_id": "present",
                                "evaluation_type": "memory_presence",
                            },
                            {
                                "criterion_id": "absent",
                                "evaluation_type": "forgetting_absence",
                            },
                        ]
                    }
                ),
            }
        ],
    )
    return {
        "observations": observations_path,
        "public_blocks": public_blocks_path,
        "public_summary": public_summary_path,
        "final_blocks": final_blocks_path,
        "final_metrics": final_metrics_path,
        "human": human_path,
        "payloads": payload_path,
        "conditions": condition_rows,
        "block_ids": block_ids,
    }


def test_identifier_join_reconstructs_one_complete_triplet():
    triplets, integrity = build_triplets(_rows())
    assert integrity["complete_triplets"] == 1
    assert triplets[0]["pair_id"] == "fixture::b1::missing"
    assert triplets[0]["conditions"]["corrupted"]["condition_id"].endswith("::corrupted")
    assert triplets[0]["conditions"]["enforced"]["condition_id"].endswith("::enforced")


def test_duplicate_condition_identifier_is_rejected():
    rows = _rows()
    rows[1]["condition_id"] = rows[0]["condition_id"]
    with pytest.raises(AuditError, match="identifier failure"):
        build_triplets(rows)


def test_authoritative_and_public_formal_panel_modes(tmp_path):
    fixture = _formal_panel_fixture(tmp_path)
    common = {
        "observations_path": fixture["observations"],
        "public_blocks_path": fixture["public_blocks"],
        "public_summary_path": fixture["public_summary"],
        "final_blocks_path": fixture["final_blocks"],
        "final_metrics_path": fixture["final_metrics"],
        "human_adjudication_path": fixture["human"],
        "semantic_payloads_path": fixture["payloads"],
        "condition_rows": fixture["conditions"],
    }
    final_blocks, final_observations, final_report = audit_formal_panel(
        native_panel="paper_final_298", **common
    )
    assert final_report["resolved_matched_triplets"] == 2
    assert set(final_report["unresolved_block_ids"]) == {
        "block-unresolved-clean",
        "block-unresolved-enforced",
    }
    assert final_report["unresolved_observations"] == 2
    assert final_report["human_adjudication"]["observation_id"] == "block-human-enforced"
    assert final_report["human_adjudication"]["score"] == pytest.approx(0.5)
    human_enforced = next(
        row
        for row in final_observations
        if row["observation_id"] == "block-human-enforced"
    )
    assert human_enforced["formal_score"] == pytest.approx(0.5)
    assert human_enforced["resolution_method"] == "human_adjudication"
    human_block = next(row for row in final_blocks if row["pilot_block_id"] == "block-human")
    assert human_block["scores"]["enforced"] == pytest.approx(0.5)
    assert set(final_report["source_artifacts"]) == {
        "observations",
        "final_blocks",
        "final_metrics",
        "human_adjudication",
        "semantic_payloads",
    }
    assert all(len(value["sha256"]) == 64 for value in final_report["source_artifacts"].values())

    public_blocks, public_observations, public_report = audit_formal_panel(
        native_panel="public_297", **common
    )
    assert len(public_blocks) == 1
    assert public_report["resolved_matched_triplets"] == 1
    assert public_report["unresolved_observations"] == 3
    assert public_report["human_adjudication"] is None
    assert set(public_report["unresolved_block_ids"]) == {
        "block-human",
        "block-unresolved-clean",
        "block-unresolved-enforced",
    }
    public_human_enforced = next(
        row
        for row in public_observations
        if row["observation_id"] == "block-human-enforced"
    )
    assert public_human_enforced["formal_score"] is None


def test_missing_or_error_output_is_preserved_as_unresolved():
    assert parse_proxy_outcome(None) == ("missing", None, "missing_condition_row")
    state, score, reason = parse_proxy_outcome(
        _result(None, status="error", scoring_status="unscored")
    )
    assert (state, score) == ("unresolved", None)
    assert reason == "fixture_failure"
    assert parse_proxy_outcome(_result(True)) == ("scored", True, None)


def test_corruption_denominator_does_not_require_enforced_output():
    context = build_context_records(_bundle([_triplet("fixture::b1")]))[0]
    condition_ids = {
        "clean": "fixture::b1::clean",
        "corrupted": "fixture::b1::missing::corrupted",
        "enforced": "fixture::b1::missing::enforced",
    }
    rows = {
        condition_ids["clean"]: _result(True),
        condition_ids["corrupted"]: _result(False),
        condition_ids["enforced"]: _result(None, status="error", scoring_status="unscored"),
    }
    proxy = build_proxy_records([context], {"fixture_provider": rows})[0]
    assert proxy["proxy_pair_scoreable"] is True
    assert proxy["proxy_full_triplet_scoreable"] is False
    assert proxy["proxy_recovery_pair_scoreable"] is False
    assert proxy["proxy_enforced_unresolved"] is True
    metrics = _proxy_metric_rows(
        [proxy], "source", "fixture_source", "fixture_provider", 20
    )
    induced = next(row for row in metrics if row["metric"] == "proxy_corruption_induced_failure")
    enforced_missing = next(row for row in metrics if row["metric"] == "proxy_enforced_unresolved")
    recovery_missing = next(
        row for row in metrics if row["metric"] == "proxy_recovery_pair_unresolved"
    )
    assert (induced["numerator"], induced["denominator"]) == (1, 1)
    assert (enforced_missing["numerator"], enforced_missing["denominator"]) == (1, 1)
    assert (recovery_missing["numerator"], recovery_missing["denominator"]) == (1, 1)


def test_correctness_change_is_not_treated_as_directional_failure():
    context = build_context_records(_bundle([_triplet("fixture::b1")]))[0]
    rows = {
        "fixture::b1::clean": _result(False),
        "fixture::b1::missing::corrupted": _result(True),
        "fixture::b1::missing::enforced": _result(True),
    }
    proxy = build_proxy_records([context], {"fixture_provider": rows})[0]
    assert proxy["proxy_correctness_outcome_changed"] is True
    assert proxy["proxy_corruption_induced_failure"] is False
    assert proxy["proxy_correctness_improved"] is True


def test_recovery_denominator_does_not_require_a_clean_score():
    context = build_context_records(_bundle([_triplet("fixture::b1")]))[0]
    rows = {
        "fixture::b1::clean": _result(None, status="error", scoring_status="unscored"),
        "fixture::b1::missing::corrupted": _result(False),
        "fixture::b1::missing::enforced": _result(True),
    }
    proxy = build_proxy_records([context], {"fixture_provider": rows})[0]
    assert proxy["proxy_pair_scoreable"] is False
    assert proxy["proxy_recovery_pair_scoreable"] is True
    assert proxy["proxy_recovery_success"] is True


def test_state_label_is_not_counted_as_an_independent_detector():
    triplet = _triplet("fixture::b1")
    context = build_context_records(_bundle([triplet]))[0]
    assert context["injected_fault_label_present"] is True
    assert context["state_internal_changed"] is False
    assert context["state_independent_constraint_anomaly"] is False
    assert _state_issues(triplet["conditions"]["corrupted"]["internal_governance_state"], "longmemeval") == []


def test_context_classification_uses_membership_not_fault_label():
    triplet = _triplet("fixture::b1")
    triplet["conditions"]["clean"]["model_visible_memory_items"] = ["current", "shared"]
    triplet["conditions"]["clean"]["model_visible_context_text"] = "current\nshared"
    corrupt = triplet["conditions"]["corrupted"]
    corrupt["model_visible_memory_items"] = ["shared", "superseded"]
    corrupt["model_visible_context_text"] = "shared\nsuperseded"
    context = build_context_records(_bundle([triplet]))[0]
    assert context["context_changed"] is True
    assert context["context_added_item_relative_to_clean"] is True
    assert context["context_removed_item_relative_to_clean"] is True
    assert context["added_item_instances"] == 1
    assert context["removed_item_instances"] == 1


def test_score_units_remain_separate_in_native_records():
    contexts = [
        build_context_records(_bundle([_triplet("fixture::lme", source="longmemeval")]))[0],
        build_context_records(_bundle([_triplet("fixture::mem", source="memora", event="deletion")]))[0],
    ]
    bundle = _bundle(
        [
            _triplet("fixture::lme", source="longmemeval"),
            _triplet("fixture::mem", source="memora", event="deletion"),
        ]
    )
    bundle.formal_observations = [
        {
            "pilot_block_id": "block-lme",
            "condition": condition,
            "condition_id": f"fixture::lme::{condition}" if condition == "clean" else f"fixture::lme::missing::{condition}",
            "formal_score": score,
            "answer_sha256": f"hash-{score}",
            "source": "longmemeval",
            "generating_provider": "openai",
            "fault_type": "missing",
            "event_macro": "update",
        }
        for condition, score in (("clean", 1.0), ("corrupted", 0.0), ("enforced", 1.0))
    ] + [
        {
            "pilot_block_id": "block-mem",
            "condition": condition,
            "condition_id": f"fixture::mem::{condition}" if condition == "clean" else f"fixture::mem::missing::{condition}",
            "formal_score": score,
            "answer_sha256": f"hash-{score}",
            "source": "memora",
            "generating_provider": "openai",
            "fault_type": "missing",
            "event_macro": "deletion",
        }
        for condition, score in (("clean", 0.9), ("corrupted", 0.4), ("enforced", 0.8))
    ]
    bundle.formal_blocks = [
        {
            "pilot_block_id": "block-lme",
            "base_id": "fixture::lme",
            "source": "longmemeval",
            "generating_provider": "openai",
            "fault_operator": "missing",
            "analysis_event_type": "update",
            "scores": {"clean": 1.0, "corrupted": 0.0, "enforced": 1.0},
            "answer_sha256": {"clean": "hash-1", "corrupted": "hash-0", "enforced": "hash-1"},
        },
        {
            "pilot_block_id": "block-mem",
            "base_id": "fixture::mem",
            "source": "memora",
            "generating_provider": "openai",
            "fault_operator": "missing",
            "analysis_event_type": "deletion",
            "scores": {"clean": 0.9, "corrupted": 0.4, "enforced": 0.8},
            "answer_sha256": {"clean": "hash-9", "corrupted": "hash-4", "enforced": "hash-8"},
        },
    ]
    native = build_native_records(bundle, contexts)
    lme = next(row for row in native if row["source_dataset"] == "longmemeval")
    memora = next(row for row in native if row["source_dataset"] == "memora")
    assert lme["native_score_unit"] == "binary_correctness"
    assert lme["native_longmemeval_induced_failure"] is True
    assert lme["native_memora_score_decreased"] is None
    assert lme["native_response_identity_changed"] is True
    assert lme["native_score_identity_changed"] is True
    assert memora["native_score_unit"] == "fama_continuous_0_to_1"
    assert memora["native_memora_score_decreased"] is True
    assert memora["native_longmemeval_induced_failure"] is None
    assert memora["native_corruption_score_delta"] == pytest.approx(-0.5)
    assert memora["native_response_identity_changed"] is True
    assert memora["native_score_identity_changed"] is True


def test_response_identity_uses_only_clean_and_corrupted_hashes():
    contexts = [build_context_records(_bundle([_triplet("fixture::lme")]))[0]]
    bundle = _bundle([_triplet("fixture::lme")])
    bundle.formal_observations = [
        {
            "pilot_block_id": "block-lme",
            "condition": condition,
            "condition_id": f"fixture::lme::{condition}" if condition == "clean" else f"fixture::lme::missing::{condition}",
            "formal_score": score,
            "answer_sha256": answer_hash,
        }
        for condition, score, answer_hash in (
            ("clean", 1.0, "same-clean-corrupted"),
            ("corrupted", 0.0, "same-clean-corrupted"),
            ("enforced", 1.0, "different-enforced"),
        )
    ]
    bundle.formal_blocks = [
        {
            "pilot_block_id": "block-lme",
            "base_id": "fixture::lme",
            "source": "longmemeval",
            "generating_provider": "openai",
            "fault_operator": "missing",
            "analysis_event_type": "update",
            "scores": {"clean": 1.0, "corrupted": 0.0, "enforced": 1.0},
            "answer_sha256": {
                "clean": "same-clean-corrupted",
                "corrupted": "same-clean-corrupted",
                "enforced": "different-enforced",
            },
        }
    ]
    record = build_native_records(bundle, contexts)[0]
    assert record["native_response_identity_evaluable"] is True
    assert record["native_response_identity_changed"] is False
    assert record["native_score_identity_changed"] is True
    assert record["native_response_identity_unchanged_score_changed"] is True


def test_missing_response_hash_marks_identity_unavailable_without_imputation():
    contexts = [build_context_records(_bundle([_triplet("fixture::lme")]))[0]]
    bundle = _bundle([_triplet("fixture::lme")])
    bundle.formal_observations = [
        {
            "pilot_block_id": "block-lme",
            "condition": condition,
            "condition_id": f"fixture::lme::{condition}" if condition == "clean" else f"fixture::lme::missing::{condition}",
            "formal_score": 1.0,
            "answer_sha256": None if condition == "clean" else "corrupted-hash",
        }
        for condition in ("clean", "corrupted", "enforced")
    ]
    bundle.formal_blocks = [
        {
            "pilot_block_id": "block-lme",
            "base_id": "fixture::lme",
            "source": "longmemeval",
            "generating_provider": "openai",
            "fault_operator": "missing",
            "analysis_event_type": "update",
            "scores": {"clean": 1.0, "corrupted": 1.0, "enforced": 1.0},
            "answer_sha256": {
                "clean": None,
                "corrupted": "corrupted-hash",
                "enforced": "enforced-hash",
            },
        }
    ]
    record = build_native_records(bundle, contexts)[0]
    assert record["native_response_identity_evaluable"] is False
    assert record["native_response_identity_unavailable"] is True
    assert record["native_response_identity_changed"] is None
    assert record["native_response_identity_unchanged_score_changed"] is None


def test_base_cluster_bootstrap_is_deterministic_and_uses_base_clusters():
    records = [
        {"base_id": "b1", "flag": True},
        {"base_id": "b1", "flag": False},
        {"base_id": "b2", "flag": True},
    ]
    first = cluster_bootstrap_rate(records, "flag", iterations=50)
    second = cluster_bootstrap_rate(records, "flag", iterations=50)
    assert first == second
    assert first["base_clusters"] == 2
    assert first["denominator"] == 3
    assert first["estimate"] == pytest.approx(2 / 3)


def test_correct_stratified_and_overlap_denominators():
    triplets = [
        _triplet("fixture::b1", source="longmemeval"),
        _triplet("fixture::b2", source="longmemeval"),
    ]
    contexts = build_context_records(_bundle(triplets))
    provider_rows = {}
    for base in ("fixture::b1", "fixture::b2"):
        provider_rows[f"{base}::clean"] = _result(True)
        provider_rows[f"{base}::missing::corrupted"] = _result(base.endswith("b1") is False)
        provider_rows[f"{base}::missing::enforced"] = _result(True)
    proxies = build_proxy_records(contexts, {"openai": provider_rows})
    tables = build_stratified_rows(
        contexts,
        proxies,
        [],
        breakdown="source",
        field="source_dataset",
        iterations=20,
    )
    induced = next(
        row
        for row in tables
        if row["metric"] == "proxy_corruption_induced_failure"
    )
    assert (induced["numerator"], induced["denominator"]) == (1, 2)
    overlap = build_overlap_rows(proxies, [], iterations=20)
    both = next(
        row
        for row in overlap
        if row["metric"] == "context_changed_with_proxy_induced_failure"
    )
    assert (both["numerator"], both["denominator"]) == (1, 2)


def test_overlap_categories_name_selected_event_and_partition_denominator():
    contexts = [build_context_records(_bundle([_triplet("fixture::b1")]))[0]]
    provider_rows = {
        "fixture::b1::clean": _result(True),
        "fixture::b1::missing::corrupted": _result(False),
        "fixture::b1::missing::enforced": _result(True),
    }
    proxies = build_proxy_records(contexts, {"openai": provider_rows})
    overlap = build_overlap_rows(proxies, [], iterations=20)
    names = {row["metric"] for row in overlap}
    assert names == {
        "context_changed_with_proxy_induced_failure",
        "context_changed_without_proxy_induced_failure",
        "context_unchanged_with_proxy_induced_failure",
        "context_unchanged_without_proxy_induced_failure",
    }
    assert sum(int(row["numerator"]) for row in overlap) == int(overlap[0]["denominator"])
    assert all("output unchanged" not in row["metric"] for row in overlap)


def test_state_observability_is_not_reported_as_a_detector_metric():
    triplets = [_triplet("fixture::b1")]
    bundle = _bundle(triplets)
    contexts = build_context_records(bundle)
    summary = summarize(bundle, contexts, [], [], iterations=20)
    state = summary["state_observability"]
    assert state["status"] == "not_estimable"
    assert state["state_detector_implemented"] is False
    assert state["anomaly_prevalence_estimable"] is False
    assert state["precision_recall_fpr_fnr_estimable"] is False
    assert all("state" not in metric for metric in summary["estimable_metrics"])


def test_offline_outputs_are_reproducible_and_non_empty(tmp_path):
    triplets = [
        _triplet("fixture::b1", source="longmemeval"),
        _triplet("fixture::b2", source="memora", event="deletion"),
    ]
    bundle = _bundle(triplets)
    contexts = build_context_records(bundle)
    provider_rows = {}
    for triplet in triplets:
        base_id = triplet["base_id"]
        provider_rows[f"{base_id}::clean"] = _result(True)
        provider_rows[f"{base_id}::missing::corrupted"] = _result(False)
        provider_rows[f"{base_id}::missing::enforced"] = _result(True)
    proxies = build_proxy_records(contexts, {"openai": provider_rows})
    bundle.formal_observations = [
        {
            "pilot_block_id": "block-b1",
            "condition": condition,
            "formal_score": score,
            "answer_sha256": f"{condition}-hash",
            "source": "longmemeval",
            "generating_provider": "openai",
            "fault_type": "missing",
            "event_macro": "update",
        }
        for condition, score in (("clean", 1.0), ("corrupted", 0.0), ("enforced", 1.0))
    ]
    bundle.formal_blocks = [
        {
            "pilot_block_id": "block-b1",
            "base_id": "fixture::b1",
            "source": "longmemeval",
            "generating_provider": "openai",
            "fault_operator": "missing",
            "analysis_event_type": "update",
            "scores": {"clean": 1.0, "corrupted": 0.0, "enforced": 1.0},
            "answer_sha256": {
                "clean": "clean-hash",
                "corrupted": "corrupted-hash",
                "enforced": "enforced-hash",
            },
        }
    ]
    natives = build_native_records(bundle, contexts)
    summary = summarize(bundle, contexts, proxies, natives, iterations=20)
    tables = {
        "observability_by_source.csv": build_stratified_rows(
            contexts,
            proxies,
            natives,
            breakdown="source",
            field="source_dataset",
            iterations=20,
        ),
        "observability_by_fault.csv": build_stratified_rows(
            contexts,
            proxies,
            natives,
            breakdown="fault",
            field="fault_operator",
            iterations=20,
        ),
        "observability_by_event.csv": build_stratified_rows(
            contexts,
            proxies,
            natives,
            breakdown="event",
            field="analysis_event_type",
            iterations=20,
        ),
        "observability_by_provider.csv": build_provider_rows(
            contexts, proxies, natives, iterations=20
        ),
        "observability_overlap.csv": build_overlap_rows(
            proxies, natives, iterations=20
        ),
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_outputs(first, summary, tables)
    write_outputs(second, summary, tables)
    for name in OUTPUT_FILES:
        assert (first / name).exists()
        assert (first / name).read_bytes() == (second / name).read_bytes()
    with (first / "observability_by_source.csv").open(newline="", encoding="utf-8") as handle:
        table_rows = list(csv.DictReader(handle))
        assert table_rows
        assert {
            "bootstrap_seed",
            "bootstrap_iterations",
            "bootstrap_unit",
            "denominator",
        } <= set(table_rows[0])
    with pytest.raises(FileExistsError):
        write_outputs(first, summary, tables)


@pytest.mark.requires_private_artifacts
def test_actual_297_and_298_response_identity_regressions():
    condition_manifest = Path(
        "<PRIVATE_REPO_ROOT>/data/"
        "final_main_governance_condition_manifest_v3.jsonl"
    )
    if not condition_manifest.is_file():
        pytest.skip("developer condition manifest is unavailable")
    common = {
        "condition_manifest": condition_manifest,
        "base_manifest": Path("data/final_main_governance_base_manifest_v2.jsonl"),
        "condition_freeze": Path(
            "data/final_main_governance_condition_manifest_v3.freeze.json"
        ),
        "base_freeze": Path(
            "data/final_main_governance_base_manifest_v2.freeze.json"
        ),
        "results_dir": Path("results/final_main_sanitized"),
        "formal_observations": Path(
            "results/formal_300_validation_v1/analysis/resolved_observations.jsonl"
        ),
        "formal_blocks": Path(
            "results/formal_300_validation_v1/analysis/resolved_blocks.jsonl"
        ),
        "formal_summary": Path(
            "results/formal_300_validation_v1/analysis/formal_validation_summary.json"
        ),
        "final_formal_blocks": Path(
            "results/b4_scoring_validation/private/formal_300_validation_v1/"
            "final_analysis/formal_300_block_level_results.csv"
        ),
        "final_formal_metrics": Path(
            "results/b4_scoring_validation/private/formal_300_validation_v1/"
            "final_analysis/formal_300_final_metrics.json"
        ),
        "human_adjudication": Path(
            "results/b4_scoring_validation/private/formal_300_validation_v1/"
            "human_adjudication/human_adjudication_decisions.jsonl"
        ),
        "semantic_payloads": Path(
            "results/b4_scoring_validation/private/formal_300_validation_v1/"
            "semantic_payloads.jsonl"
        ),
        "evidence_base_level": Path(
            "results/answer_leakage_repair/final_v3_analysis/"
            "governance_evidence_audit/governance_evidence_base_level.csv"
        ),
    }
    observed = {}
    for mode in ("public_297", "paper_final_298"):
        bundle = audit_repository(native_panel=mode, **common)
        assert bundle.report["formal_panel"]["mode"] == mode
        contexts = build_context_records(bundle)
        natives = build_native_records(bundle, contexts)
        resolved = [row for row in natives if row["native_resolved"]]
        observed[mode] = {
            "resolved": len(resolved),
            "unresolved": len(natives) - len(resolved),
            "longmemeval_identity_unchanged": sum(
                bool(row["native_context_changed_response_identity_unchanged"])
                for row in resolved
                if row["source_dataset"] == "longmemeval"
            ),
            "memora_identity_unchanged": sum(
                bool(row["native_context_changed_response_identity_unchanged"])
                for row in resolved
                if row["source_dataset"] == "memora"
            ),
        }
    assert observed["public_297"] == {
        "resolved": 297,
        "unresolved": 3,
        "longmemeval_identity_unchanged": 17,
        "memora_identity_unchanged": 18,
    }
    assert observed["paper_final_298"] == {
        "resolved": 298,
        "unresolved": 2,
        "longmemeval_identity_unchanged": 17,
        "memora_identity_unchanged": 18,
    }
    final_bundle = audit_repository(native_panel="paper_final_298", **common)
    unresolved_observation_ids = {
        str(row["observation_id"])
        for row in json.loads(common["final_formal_metrics"].read_text())["unresolved_details"]
    }
    observed_unresolved_blocks = {
        str(row["pilot_block_id"])
        for row in final_bundle.formal_observations
        if str(row["observation_id"]) in unresolved_observation_ids
    }
    reported_unresolved_blocks = set(
        final_bundle.report["formal_panel"]["unresolved_block_ids"]
    )
    assert len(reported_unresolved_blocks) == 2
    assert reported_unresolved_blocks == observed_unresolved_blocks
    assert all(value.startswith("pilot-block::") for value in reported_unresolved_blocks)
    assert final_bundle.report["formal_panel"]["human_adjudication"]["score"] == pytest.approx(
        0.5408320493066255
    )
