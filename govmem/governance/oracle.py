"""Conservative, item-level deterministic governance oracle construction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from eval.interfaces import AdaptedExample

from .schema import AuthorizationState, BaseGovernanceExample, DeletionState, GovernanceState, TemporalState


@dataclass(frozen=True)
class OracleDerivation:
    supported: bool
    example: BaseGovernanceExample | None = None
    reason: str | None = None


def _base(
    example: AdaptedExample,
    dimension: str,
    state: GovernanceState,
    decision: str,
    memory: tuple[dict[str, Any], ...],
    metadata: dict[str, Any],
    example_suffix: str = "",
) -> BaseGovernanceExample:
    dataset = str(example.metadata.get("dataset") or example.query.metadata.get("dataset"))
    source_id = example.query.query_id
    return BaseGovernanceExample(
        example_id=f"{dataset}::{source_id}::{dimension}{example_suffix}",
        source_dataset=dataset,
        source_example_id=source_id,
        dimension=dimension,  # type: ignore[arg-type]
        task=example.query.text,
        memory=memory,
        gold_governance_state=state,
        gold_admission_decision=decision,  # type: ignore[arg-type]
        utility_target=example.query.gold_answer,
        source_metadata={
            "dataset": dataset,
            "query_id": source_id,
            "task_type": example.query.metadata.get("task_type"),
            **metadata,
        },
    )


def derive_gatemem_authorization(example: AdaptedExample) -> OracleDerivation:
    """Use GateMem's native privacy action, never relationship inference."""
    if example.query.metadata.get("query_type") != "privacy":
        return OracleDerivation(False, reason="GateMem checkpoint is not a privacy task")
    gold = dict(example.evaluator_metadata.get("gold", {}))
    action = str(gold.get("expected_action", "")).strip().lower()
    if action in {"refuse", "deny", "reject", "no_memory"}:
        decision = "deny"
    elif action in {"answer", "allow", "admit"}:
        decision = "allow"
    else:
        return OracleDerivation(False, reason=f"GateMem privacy action is not an explicit allow/deny action: {action!r}")
    leak_targets = tuple(str(item) for item in (gold.get("leak_targets") or ()))
    relevant = []
    for interaction in example.interactions:
        if leak_targets and any(target.lower() in interaction.content.lower() for target in leak_targets):
            relevant.append({
                "interaction_id": interaction.interaction_id,
                "content": interaction.content,
                "timestamp": interaction.timestamp.isoformat() if interaction.timestamp else None,
            })
    metadata = {
        "oracle_rule": "GateMem privacy expected_action",
        "asker": dict(example.evaluator_metadata.get("asker", {})),
        "relationships": tuple(dict(item) for item in example.metadata.get("relationships", ())),
        "attack_type": gold.get("attack_type"),
        "leak_targets": leak_targets,
        "checkpoint_id": example.evaluator_metadata.get("checkpoint_id"),
        "episode_id": example.evaluator_metadata.get("episode_id"),
    }
    state = GovernanceState(authorization=AuthorizationState(decision))
    return OracleDerivation(True, _base(example, "authorization", state, decision, tuple(relevant), metadata))


def _item_key(value: Any) -> str:
    if isinstance(value, dict):
        return str(sorted((str(k), _item_key(v)) for k, v in value.items()))
    return str(value)


def _operation_item(operation: dict[str, Any]) -> Any:
    details = operation.get("operation_details") or {}
    return details.get("item")


def _same_item(left: Any, right: Any) -> bool:
    if left == right:
        return True
    if isinstance(left, dict) and not isinstance(right, dict):
        return str(right) in {str(value) for value in left.values()}
    if isinstance(right, dict) and not isinstance(left, dict):
        return str(left) in {str(value) for value in right.values()}
    return False


def _operation_is_visible_at_question(operation: dict[str, Any], example: AdaptedExample) -> bool:
    """Limit lifecycle evidence to the benchmark's evaluation boundary."""
    question_time = example.query.current_time
    if question_time is None or not operation.get("date"):
        return True
    try:
        operation_time = datetime.fromisoformat(str(operation["date"]).replace("Z", "+00:00"))
    except ValueError:
        return True
    if operation_time.tzinfo is None and question_time.tzinfo is not None:
        operation_time = operation_time.replace(tzinfo=question_time.tzinfo)
    if question_time.tzinfo is None and operation_time.tzinfo is not None:
        question_time = question_time.replace(tzinfo=operation_time.tzinfo)
    return operation_time <= question_time


def _memora_candidates(example: AdaptedExample) -> list[tuple[str, Any, dict[str, Any], DeletionState, tuple[DeletionState, ...]]]:
    if example.query.metadata.get("task_type") != "remembering":
        return []
    operations = list(example.evaluator_metadata.get("session_operations", ()))
    adds = [item for item in operations if item.get("operation") == "add" and _operation_item(item) is not None]
    deletes = [item for item in operations if item.get("operation") in {"delete", "remove", "forget"} and _operation_item(item) is not None]
    question_gold = example.evaluator_metadata.get("gold", {})
    memory_evidence = question_gold.get("memory_evidence", {}) or {}
    active_items = memory_evidence.get("active_memory")
    # Official Memora uses ``remaining_tasks`` directly under memory_evidence
    # for active lifecycle items; retain support for the mini fixture's
    # active_memory wrapper as well.
    if active_items is None and isinstance(memory_evidence, dict):
        active_items = memory_evidence
    forgotten_items = (question_gold.get("forgetting_evidence", {}) or {}).get("forgotten_items") or []
    candidates = []
    seen: set[str] = set()

    def add_candidate(value: Any, status: str, session_id: Any, operation: dict[str, Any], predecessor: tuple[DeletionState, ...]) -> None:
        key = _item_key(value)
        if key in seen:
            return
        seen.add(key)
        lifecycle = [{"operation": "add", "session_id": session_id, "item": value}]
        lifecycle.append({"operation": operation.get("operation"), "session_id": operation.get("session_id"), "item": _operation_item(operation)})
        memory = ({"item": value, "lifecycle": lifecycle},)
        metadata = {
            "oracle_rule": "Memora explicit add/delete session operations",
            "item": value,
            "lifecycle": lifecycle,
            "historical_deletion_states": predecessor,
        }
        candidates.append((key, value, metadata, DeletionState(status, int(operation.get("session_id")) if status == "deleted" and isinstance(operation.get("session_id"), int) else None), predecessor))

    for forgotten in forgotten_items:
        value = forgotten.get("value") if isinstance(forgotten, dict) else forgotten
        matches = [op for op in deletes if _same_item(_operation_item(op), value)]
        if len(matches) != 1:
            continue
        delete = matches[0]
        adds_for_item = [op for op in adds if _same_item(_operation_item(op), _operation_item(delete)) and op.get("session_id", 0) <= delete.get("session_id", 0)]
        if len(adds_for_item) != 1:
            continue
        add = adds_for_item[0]
        add_state = DeletionState("active", int(add["session_id"]) if isinstance(add.get("session_id"), int) else None)
        add_candidate = {"operation": "add", "session_id": add.get("session_id"), "item": _operation_item(add)}
        delete_candidate = {"operation": "delete", "session_id": delete.get("session_id"), "item": _operation_item(delete)}
        # Preserve both concrete lifecycle events in the compact payload.
        key = _item_key(value)
        if key not in seen:
            seen.add(key)
            metadata = {"oracle_rule": "Memora explicit add/delete session operations", "item": value, "lifecycle": [add_candidate, delete_candidate], "historical_deletion_states": (add_state,)}
            candidates.append((key, value, metadata, DeletionState("deleted", int(delete["session_id"]) if isinstance(delete.get("session_id"), int) else None), (add_state,)))

    active_values: list[Any] = []
    if isinstance(active_items, dict):
        if "value" in active_items:
            active_values.append(active_items["value"])
        for key in ("remaining_tasks", "active_items"):
            if isinstance(active_items.get(key), list):
                active_values.extend(
                    item.get("value") if isinstance(item, dict) and "value" in item else item
                    for item in active_items[key]
                )
    for value in active_values:
        matches = [op for op in adds if _same_item(_operation_item(op), value)]
        if len(matches) == 1:
            add = matches[0]
            add_session = add.get("session_id", 0)
            later_deletes = [
                op for op in deletes
                if _same_item(_operation_item(op), _operation_item(add))
                and op.get("session_id", 0) > add_session
                and _operation_is_visible_at_question(op, example)
            ]
            if later_deletes:
                continue
            metadata = {"oracle_rule": "Memora explicit add session operation", "item": value, "lifecycle": [{"operation": "add", "session_id": add.get("session_id"), "item": _operation_item(add)}], "historical_deletion_states": ()}
            key = _item_key(value)
            if key not in seen:
                candidates.append((key, value, metadata, DeletionState("active", int(add["session_id"]) if isinstance(add.get("session_id"), int) else None), ()))
    return candidates


def derive_memora_deletion_items(example: AdaptedExample) -> list[OracleDerivation]:
    results = []
    for index, (_, value, metadata, state, predecessor) in enumerate(_memora_candidates(example)):
        results.append(OracleDerivation(True, _base(example, "deletion", GovernanceState(deletion=state), "allow" if state.status == "active" else "deny", ({"item": value, "lifecycle": metadata["lifecycle"]},), metadata, f"::item{index}")))
    return results


def derive_memora_deletion(example: AdaptedExample) -> OracleDerivation:
    results = derive_memora_deletion_items(example)
    return results[0] if results else OracleDerivation(False, reason="No remembering item has a unique explicit lifecycle")


def _attribute_matches_question(attribute: str, question: str) -> bool:
    phrase = re.sub(r"[_-]+", " ", attribute).lower()
    return phrase in question.lower()


def derive_memconflict_temporal(example: AdaptedExample) -> OracleDerivation:
    if example.query.metadata.get("conflict_type") != "dynamic":
        return OracleDerivation(False, reason="Only dynamic MemConflict records are eligible")
    updates = list((example.evaluator_metadata.get("gold", {}) or {}).get("updated_attributes", ()))
    candidates = []
    for update in updates:
        for attribute in update.get("attributes", ()):
            name = str(attribute.get("Attribute", ""))
            if name and _attribute_matches_question(name, example.query.text):
                candidates.append((update, attribute))
    if len(candidates) != 1:
        return OracleDerivation(False, reason="Dynamic question does not link to exactly one Updated_Attributes transition")
    update, attribute = candidates[0]
    # Re-bind the name from the selected transition.  The loop variable may
    # otherwise retain the last attribute in a multi-attribute session,
    # producing a mismatched attribute name with the selected Before/After
    # values.
    name = str(attribute.get("Attribute", ""))
    session_history = list((example.evaluator_metadata.get("gold", {}) or {}).get("session_history", ()))
    previous_session = next((item for item in reversed(session_history) if item.get("session_id") != update.get("session_id")), None)
    current_time = datetime.fromisoformat(str(update["date"])) if update.get("date") else example.query.current_time
    previous_time = datetime.fromisoformat(str(previous_session["date"])) if previous_session and previous_session.get("date") else None
    if current_time is None or previous_time is None:
        return OracleDerivation(False, reason="Updated_Attributes transition lacks two explicit session dates")
    current = TemporalState("valid", valid_from=current_time, evaluated_at=current_time, version=int(update.get("session_id")) if isinstance(update.get("session_id"), int) else None)
    previous = TemporalState("expired", valid_from=previous_time, valid_until=current_time, evaluated_at=current_time, version=int(previous_session.get("session_id")) if isinstance(previous_session, dict) and isinstance(previous_session.get("session_id"), int) else None)
    metadata = {
        "oracle_rule": "MemConflict Updated_Attributes Before/After transition",
        "source_record_id": example.query.metadata.get("source_record_id"),
        "attribute": name,
        "previous_value": attribute.get("Before"),
        "current_value": attribute.get("After"),
        "previous_session": previous_session,
        "current_session": update,
        "historical_temporal_states": (previous,),
    }
    memory = ({"attribute": name, "previous_value": attribute.get("Before"), "current_value": attribute.get("After"), "previous_session": previous_session, "current_session": update},)
    state = GovernanceState(temporal=current)
    return OracleDerivation(True, _base(example, "temporal", state, "allow", memory, metadata))


def derive_base_examples(example: AdaptedExample) -> list[OracleDerivation]:
    dataset = example.metadata.get("dataset") or example.query.metadata.get("dataset")
    if dataset == "gatemem":
        result = derive_gatemem_authorization(example)
        return [result] if result.supported else []
    if dataset == "memora":
        return derive_memora_deletion_items(example)
    if dataset == "memconflict":
        result = derive_memconflict_temporal(example)
        return [result] if result.supported else []
    return []


def derive_base_example(example: AdaptedExample) -> OracleDerivation:
    results = derive_base_examples(example)
    return results[0] if results else OracleDerivation(False, reason="No deterministic governance oracle available")


__all__ = ["OracleDerivation", "derive_base_example", "derive_base_examples", "derive_gatemem_authorization", "derive_memora_deletion", "derive_memora_deletion_items", "derive_memconflict_temporal"]
