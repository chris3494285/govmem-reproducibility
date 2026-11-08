#!/usr/bin/env python3
"""Audit deterministic candidates for a restrictive governance-fault supplement."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.gatemem_adapter import GateMemDatasetAdapter
from eval.memconflict_adapter import MemConflictDatasetAdapter
from eval.memora_adapter import MemoraDatasetAdapter
from govmem.governance.oracle import derive_base_examples
from govmem.governance.schema import AuthorizationState, DeletionState, GovernanceState


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    return value


def _base_fields(base: Any) -> dict[str, Any]:
    return {
        "candidate_id": base.example_id,
        "source_dataset": base.source_dataset,
        "dimension": base.dimension,
        "task_query_id": base.source_example_id,
        "task": base.task,
        "memory": base.memory,
        "gold_governance_state": base.gold_governance_state,
        "gold_admission_decision": base.gold_admission_decision,
        "utility_target": base.utility_target,
        "source_metadata": base.source_metadata,
    }


def _transition_id(base: Any, metadata: dict[str, Any]) -> str:
    current = metadata.get("current_session") or {}
    session_id = current.get("session_id", current.get("Session_ID", "unknown"))
    return f"{metadata.get('source_record_id', base.source_example_id)}::s{session_id}::{metadata.get('attribute')}"


def _unsupported_reason(example: Any, dataset: str) -> str:
    if dataset == "gatemem":
        action = (example.evaluator_metadata.get("gold", {}) or {}).get("expected_action")
        return "no_unambiguous_native_allow_privacy_case" if example.query.metadata.get("query_type") == "privacy" else "not_privacy_checkpoint"
    if dataset == "memora":
        if example.query.metadata.get("task_type") != "remembering":
            return "not_remembering_lifecycle_task"
        return "no_unique_active_oracle_item"
    if dataset == "memconflict":
        return "no_supported_dynamic_updated_attributes_transition"
    return "unsupported_dataset"


def audit_examples(examples_by_dataset: dict[str, list[Any]]) -> dict[str, Any]:
    loaded = {dataset: len(examples) for dataset, examples in examples_by_dataset.items()}
    unsupported: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    for dataset, examples in examples_by_dataset.items():
        for example in examples:
            derivations = derive_base_examples(example)
            if not derivations:
                unsupported[_unsupported_reason(example, dataset)] += 1
                continue
            for derivation in derivations:
                base = derivation.example
                state = base.gold_governance_state
                candidate: dict[str, Any] | None = None
                if dataset == "gatemem":
                    auth = state.authorization
                    if auth is None or auth.decision != "allow":
                        unsupported["gatemem_native_privacy_gold_is_not_allow"] += 1
                        continue
                    observed = GovernanceState(authorization=AuthorizationState("deny", auth.scope_id, auth.version))
                    candidate = {
                        **_base_fields(base),
                        "candidate_cluster_id": base.source_example_id,
                        "gold_authorization_decision": "allow",
                        "restrictive_fault_supported": True,
                        "restrictive_fault_reason": "Native privacy expected_action explicitly allows access",
                        "observed_governance_state": observed,
                    }
                elif dataset == "memora":
                    deletion = state.deletion
                    if deletion is None or deletion.status != "active":
                        unsupported["memora_item_not_active_at_evaluation"] += 1
                        continue
                    lifecycle = base.source_metadata.get("lifecycle", [])
                    add_sessions = [item.get("session_id") for item in lifecycle if item.get("operation") == "add"]
                    if not add_sessions:
                        unsupported["memora_active_item_lacks_add_provenance"] += 1
                        continue
                    observed = GovernanceState(deletion=DeletionState("deleted", deletion.epoch))
                    candidate = {
                        **_base_fields(base),
                        "candidate_cluster_id": base.source_example_id,
                        "item": base.source_metadata.get("item"),
                        "gold_deletion_status": "active",
                        "add_session": add_sessions[0],
                        "latest_relevant_lifecycle_session": max(item.get("session_id", 0) for item in lifecycle),
                        "restrictive_fault_supported": True,
                        "restrictive_fault_reason": "Native item remains active with traceable add lifecycle",
                        "observed_governance_state": observed,
                    }
                elif dataset == "memconflict":
                    temporal = state.temporal
                    if temporal is None or temporal.status != "valid":
                        unsupported["memconflict_current_assertion_not_valid"] += 1
                        continue
                    metadata = base.source_metadata
                    observed = replace(temporal, status="expired")
                    transition_id = _transition_id(base, metadata)
                    relevant_attribute = {
                        "Attribute": metadata.get("attribute"),
                        "Before": metadata.get("previous_value"),
                        "After": metadata.get("current_value"),
                    }
                    candidate = {
                        **_base_fields(base),
                        "transition_id": transition_id,
                        "attribute": metadata.get("attribute"),
                        "relevant_updated_attribute": relevant_attribute,
                        "current_value": metadata.get("current_value"),
                        "previous_value": metadata.get("previous_value"),
                        "current_session": metadata.get("current_session"),
                        "gold_temporal_status": "valid",
                        "gold_admission_decision": "allow",
                        "restrictive_fault_supported": True,
                        "restrictive_fault_reason": "Current valid assertion can be deterministically marked expired without changing memory",
                        "observed_governance_state": GovernanceState(temporal=observed),
                    }
                if candidate is not None:
                    candidates.append(_jsonable(candidate))
    by_dataset = Counter(item["source_dataset"] for item in candidates)
    by_dimension = Counter(item["dimension"] for item in candidates)
    return {
        "source_examples_loaded": loaded,
        "restrictive_candidates_by_dataset": dict(by_dataset),
        "restrictive_candidates_by_dimension": dict(by_dimension),
        "restrictive_candidates_total": len(candidates),
        "restrictive_candidates_raw": len(candidates),
        "memora_unique_query_clusters": len({
            item["candidate_cluster_id"] for item in candidates
            if item["source_dataset"] == "memora"
        }),
        "memconflict_unique_transition_clusters": len({
            item["transition_id"] for item in candidates
            if item["source_dataset"] == "memconflict"
        }),
        "unsupported_by_reason": dict(unsupported),
        "candidates": candidates,
    }


def audit_inputs(gatemem_input: str | None, memora_input: str | None, memconflict_input: str | None) -> dict[str, Any]:
    # Audit each large official source in a short-lived process.  This keeps
    # parser allocations from one benchmark from inflating the next parser's
    # RSS when all three official inputs are supplied together.
    reports = []
    for dataset, path in (
        ("gatemem", gatemem_input),
        ("memora", memora_input),
        ("memconflict", memconflict_input),
    ):
        if path:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
                report_path = Path(handle.name)
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--_single-dataset", dataset,
                        "--_single-input", str(path),
                        "--output", str(report_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            finally:
                report_path.unlink(missing_ok=True)
    candidates = [candidate for report in reports for candidate in report["candidates"]]
    loaded = {dataset: 0 for dataset in ("gatemem", "memora", "memconflict")}
    unsupported: Counter[str] = Counter()
    for report in reports:
        loaded.update(report["source_examples_loaded"])
        unsupported.update(report["unsupported_by_reason"])
    return {
        "source_examples_loaded": loaded,
        "restrictive_candidates_by_dataset": dict(Counter(item["source_dataset"] for item in candidates)),
        "restrictive_candidates_by_dimension": dict(Counter(item["dimension"] for item in candidates)),
        "restrictive_candidates_total": len(candidates),
        "restrictive_candidates_raw": len(candidates),
        "memora_unique_query_clusters": len({
            item["candidate_cluster_id"] for item in candidates
            if item["source_dataset"] == "memora"
        }),
        "memconflict_unique_transition_clusters": len({
            item["transition_id"] for item in candidates
            if item["source_dataset"] == "memconflict"
        }),
        "unsupported_by_reason": dict(unsupported),
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gatemem-input")
    parser.add_argument("--memora-input")
    parser.add_argument("--memconflict-input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--_single-dataset", choices=("gatemem", "memora", "memconflict"), help=argparse.SUPPRESS)
    parser.add_argument("--_single-input", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._single_dataset:
        adapters = {
            "gatemem": GateMemDatasetAdapter(),
            "memora": MemoraDatasetAdapter(),
            "memconflict": MemConflictDatasetAdapter(),
        }
        report = audit_examples({args._single_dataset: list(adapters[args._single_dataset].load(args._single_input))})
    else:
        report = audit_inputs(args.gatemem_input, args.memora_input, args.memconflict_input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "source_examples_loaded", "restrictive_candidates_by_dataset", "restrictive_candidates_by_dimension",
        "restrictive_candidates_total", "restrictive_candidates_raw", "memora_unique_query_clusters",
        "memconflict_unique_transition_clusters", "unsupported_by_reason",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
