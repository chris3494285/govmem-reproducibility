#!/usr/bin/env python3
"""Build a deterministic governance-state fault RQ1 JSONL manifest.

Only deterministic oracle-supported examples are emitted.  Unsupported source
examples are counted and reported rather than being assigned synthetic gold.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from eval.gatemem_adapter import GateMemDatasetAdapter
from eval.memconflict_adapter import MemConflictDatasetAdapter
from eval.memora_adapter import MemoraDatasetAdapter
from govmem.governance.faults import FaultInjectionError, fault_spec_for, inject_fault
from govmem.governance.oracle import derive_base_examples
from govmem.governance.schema import BaseGovernanceExample, GovernanceState, RunnableGovernanceCondition
from govmem.governance.validation import validate_manifest


TARGETS = {
    "gatemem": ("authorization", 30),
    "memora": ("deletion", 25),
    "memconflict": ("temporal", 45),
}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def row_to_dict(row: RunnableGovernanceCondition) -> dict[str, Any]:
    return _jsonable(row)


def _history_for(base: BaseGovernanceExample) -> tuple[GovernanceState, ...]:
    metadata = base.source_metadata
    states: list[GovernanceState] = []
    for item in metadata.get("historical_temporal_states", ()):
        states.append(GovernanceState(temporal=item))
    for item in metadata.get("historical_deletion_states", ()):
        states.append(GovernanceState(deletion=item))
    return tuple(states)


def _make_rows(base: BaseGovernanceExample, operators: Iterable[str]) -> tuple[list[RunnableGovernanceCondition], list[str]]:
    rows = [RunnableGovernanceCondition(
        run_id=f"{base.example_id}::clean",
        example_id=base.example_id,
        source_dataset=base.source_dataset,
        dimension=base.dimension,
        condition_type="clean",
        fault_spec=None,
        gold_governance_state=base.gold_governance_state,
        observed_governance_state=base.gold_governance_state,
        gold_admission_decision=base.gold_admission_decision,
        task=base.task,
        memory=base.memory,
        utility_target=base.utility_target,
        source_metadata=base.source_metadata,
    )]
    skipped: list[str] = []
    history = _history_for(base)
    for operator in operators:
        try:
            observed = inject_fault(base.gold_governance_state, base.dimension, operator, history)
            spec = fault_spec_for(base.gold_governance_state, base.dimension, operator)
        except FaultInjectionError as exc:
            skipped.append(f"{operator}: {exc}")
            continue
        rows.append(RunnableGovernanceCondition(
            run_id=f"{base.example_id}::fault::{operator}",
            example_id=base.example_id,
            source_dataset=base.source_dataset,
            dimension=base.dimension,
            condition_type="fault",
            fault_spec=spec,
            gold_governance_state=base.gold_governance_state,
            observed_governance_state=observed,
            gold_admission_decision=base.gold_admission_decision,
            task=base.task,
            memory=base.memory,
            utility_target=base.utility_target,
            source_metadata=base.source_metadata,
        ))
    return rows, skipped


def _valid_faults(base: BaseGovernanceExample) -> dict[str, tuple[Any, Any]]:
    """Return all valid observed/spec pairs before candidate selection."""
    history = _history_for(base)
    options: dict[str, tuple[Any, Any]] = {}
    for operator in ("missing", "stale", "flipped", "conflicting"):
        try:
            options[operator] = (
                inject_fault(base.gold_governance_state, base.dimension, operator, history),
                fault_spec_for(base.gold_governance_state, base.dimension, operator),
            )
        except FaultInjectionError:
            continue
    return options


def build_conditions(examples: Iterable[BaseGovernanceExample], seed: int = 42, source_stats: dict[str, Any] | None = None) -> tuple[list[RunnableGovernanceCondition], dict[str, Any]]:
    """Select and fault a deterministic set of base examples."""
    rng = random.Random(seed)
    candidates: dict[tuple[str, str], list[BaseGovernanceExample]] = {}
    for example in examples:
        candidates.setdefault((example.source_dataset, example.dimension), []).append(example)
    rows: list[RunnableGovernanceCondition] = []
    skipped: Counter[str] = Counter()
    insufficient = Counter()
    selected: Counter[str] = Counter()
    for dataset, (dimension, requested) in TARGETS.items():
        eligible_pool = []
        for item in candidates.get((dataset, dimension), ()):
            if len(_valid_faults(item)) < 2:
                insufficient[f"{dataset}/{dimension}"] += 1
            else:
                eligible_pool.append(item)
        pool = sorted(eligible_pool, key=lambda item: item.example_id)
        rng.shuffle(pool)
        pool = sorted(pool[:requested], key=lambda item: item.example_id)
        selected[dataset] = len(pool)
        for index, base in enumerate(pool):
            if dimension == "authorization":
                available = ["missing", "flipped", "conflicting"]
            elif dimension == "deletion":
                available = ["missing", "stale", "flipped"]
            else:
                available = ["missing", "stale", "conflicting"]
            available = [operator for operator in available if operator in _valid_faults(base)]
            rng.shuffle(available)
            generated, reasons = _make_rows(base, available[:2])
            rows.extend(generated)
            skipped.update(f"{dataset}: {reason}" for reason in reasons)
    validate_manifest(rows)
    info = {
        "selected_base_examples": dict(selected),
        "skipped_faults": dict(skipped),
        "skipped_base_examples_insufficient_faults": dict(insufficient),
        "seed": seed,
    }
    if source_stats:
        info.update(source_stats)
    return rows, info


def _adapt(path: str | None, adapter: Any) -> list[Any]:
    return list(adapter().load(path)) if path else []


def build_from_inputs(gatemem_input: str | None, memora_input: str | None, memconflict_input: str | None, seed: int) -> tuple[list[RunnableGovernanceCondition], dict[str, Any]]:
    adapted: list[BaseGovernanceExample] = []
    loaded = Counter()
    unsupported = Counter()
    for path, adapter in ((gatemem_input, GateMemDatasetAdapter), (memora_input, MemoraDatasetAdapter), (memconflict_input, MemConflictDatasetAdapter)):
        for example in _adapt(path, adapter):
            dataset = str(example.metadata.get("dataset") or example.query.metadata.get("dataset"))
            loaded[dataset] += 1
            results = derive_base_examples(example)
            if not results:
                # The oracle functions intentionally do not guess; retain a
                # stable reason even when a dataset has no supported mapping.
                unsupported[f"{dataset}:unsupported_or_ambiguous"] += 1
            for result in results:
                if result.supported and result.example is not None:
                    adapted.append(result.example)
    return build_conditions(adapted, seed, {
        "source_examples_loaded": dict(loaded),
        "oracle_supported": len(adapted),
        "oracle_unsupported": sum(unsupported.values()),
        "oracle_unsupported_by_reason": dict(unsupported),
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gatemem-input")
    parser.add_argument("--memora-input")
    parser.add_argument("--memconflict-input")
    parser.add_argument("--output", default="data/governance_fault_rq1_pilot.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not any((args.gatemem_input, args.memora_input, args.memconflict_input)):
        parser.error("at least one dataset input is required")
    rows, info = build_from_inputs(args.gatemem_input, args.memora_input, args.memconflict_input, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row_to_dict(row), sort_keys=True) + "\n")
    by_dataset = Counter(row.source_dataset for row in rows)
    by_dimension = Counter({dimension: len({row.example_id for row in rows if row.dimension == dimension}) for dimension in {row.dimension for row in rows}})
    by_operator = Counter(row.fault_spec.operator for row in rows if row.fault_spec)
    by_direction = Counter(row.fault_spec.direction for row in rows if row.fault_spec)
    print(json.dumps({
        "output": str(output),
        "base_examples_total": len({row.example_id for row in rows}),
        "conditions_total": len(rows),
        "base_examples_by_dataset": info["selected_base_examples"],
        "conditions_by_dataset": dict(by_dataset),
        "base_examples_by_dimension": dict(by_dimension),
        "conditions_by_fault_operator": dict(by_operator),
        "conditions_by_fault_direction": dict(by_direction),
        "skipped_faults_by_reason": info["skipped_faults"],
        "source_examples_loaded": info.get("source_examples_loaded", {}),
        "oracle_supported": info.get("oracle_supported", 0),
        "oracle_unsupported": info.get("oracle_unsupported", 0),
        "oracle_unsupported_by_reason": info.get("oracle_unsupported_by_reason", {}),
        "skipped_base_examples_insufficient_faults": info.get("skipped_base_examples_insufficient_faults", {}),
        "validation_status": "valid",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
