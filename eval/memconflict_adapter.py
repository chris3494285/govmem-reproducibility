"""Adapter for public MemConflict conflict-instance records."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .interfaces import AdaptedExample, Query, RawInteraction


MEMCONFLICT_GOLD_SOURCE = "dataset_native_gold"
_VALIDITY_DIMENSIONS = {
    "dynamic": "temporal_validity",
    "static": "factual_validity",
    "conditional": "contextual_applicability",
}


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        files = sorted((*source.glob("*.jsonl"), *source.glob("*.json")))
        if len(files) != 1:
            raise ValueError("MemConflict input must be one JSONL file, JSON file, or directory containing exactly one")
        source = files[0]
    if source.suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("examples"), list):
        return value["examples"]
    return [value]


def _first_present(record: dict[str, Any], *fields: str) -> Any:
    for field in fields:
        if field in record:
            return record[field]
    return None


def _session_turns(session: dict[str, Any]) -> Iterable[dict[str, Any]]:
    turns = _first_present(session, "turns", "messages", "conversation", "dialogue")
    if isinstance(turns, list):
        return turns
    return ()


class MemConflictDatasetAdapter:
    """Normalize legacy records and official Step4_4 question records.

    Official Step4_4 has no top-level conflict type.  Its native
    ``Full_Session_Chain[*].Session_Questions[*].conflict_type`` field is the
    sole task-type source; the adapter never classifies from dialogue text.
    """

    name = "memconflict"

    def load(self, path: str | Path) -> list[AdaptedExample]:
        examples: list[AdaptedExample] = []
        for record in _read_records(path):
            if "Full_Session_Chain" in record:
                examples.extend(self._official_questions(record))
            else:
                examples.append(self.adapt_example(record))
        return examples

    def adapt_example(self, record: dict[str, Any]) -> AdaptedExample:
        if "Full_Session_Chain" in record:
            first = next(iter(self._official_questions(record)), None)
            if first is None:
                raise ValueError("Official MemConflict record contains no Session_Questions")
            return first
        conflict_type = record["conflict_type"]
        if conflict_type not in _VALIDITY_DIMENSIONS:
            raise ValueError(f"Unsupported MemConflict conflict_type: {conflict_type!r}")

        example_id = str(_first_present(record, "example_id", "id", "query_id"))
        user_id = str(_first_present(record, "user_id", "user", "persona") or "")
        sessions = _first_present(record, "sessions", "history", "interaction_history") or []
        interactions: list[RawInteraction] = []
        for session_index, session in enumerate(sessions):
            session_id = str(_first_present(session, "session_id", "id") or session_index)
            session_timestamp = _parse_timestamp(_first_present(session, "timestamp", "date"))
            for turn_index, turn in enumerate(_session_turns(session)):
                content = _first_present(turn, "text", "content", "message")
                if content is None:
                    continue
                timestamp = _parse_timestamp(_first_present(turn, "timestamp", "date")) or session_timestamp
                interactions.append(
                    RawInteraction(
                        interaction_id=f"{example_id}::{session_id}::{turn_index}",
                        user_id=user_id,
                        content=str(content),
                        timestamp=timestamp,
                        role=str(_first_present(turn, "role", "speaker") or "user"),
                        session_id=session_id,
                        metadata={
                            "dataset": self.name,
                            "session_order": session_index,
                            "turn_order": turn_index,
                        },
                    )
                )

        latest_timestamp = interactions[-1].timestamp if interactions else None
        nested_gold = record.get("gold") if isinstance(record.get("gold"), dict) else {}
        gold_fields = (
            "conflict_type",
            "gold_answer",
            "answer",
            "supporting_memory",
            "gold_memory",
            "correct_memory",
            "dynamic_state",
            "conditional_context",
            "query_context",
            "timestamps",
            "session_order",
            "updated_attributes",
        )
        gold = {
            field: _first_present(record, field) if field in record else nested_gold[field]
            for field in gold_fields
            if field in record or field in nested_gold
        }
        gold_answer = _first_present(record, "gold_answer", "answer")
        if gold_answer is None:
            gold_answer = _first_present(nested_gold, "gold_answer", "answer")
        query_text = _first_present(record, "query", "question")
        return AdaptedExample(
            query=Query(
                query_id=example_id,
                user_id=user_id,
                text=str(query_text),
                current_time=_parse_timestamp(_first_present(record, "query_timestamp", "query_time"))
                or latest_timestamp,
                gold_answer=str(gold_answer) if gold_answer is not None else None,
                metadata={
                    "dataset": self.name,
                    "conflict_type": conflict_type,
                    "task_type": conflict_type,
                    "validity_dimension": _VALIDITY_DIMENSIONS[conflict_type],
                },
            ),
            interactions=tuple(interactions),
            oracle_governance=(),
            metadata={
                "dataset": self.name,
                "conflict_type": conflict_type,
                "validity_dimension": _VALIDITY_DIMENSIONS[conflict_type],
                "governance_source": MEMCONFLICT_GOLD_SOURCE,
            },
            evaluator_metadata={
                "governance_source": MEMCONFLICT_GOLD_SOURCE,
                "example_id": example_id,
                "gold": gold,
            },
        )

    def _adapt_official_question(self, record: dict[str, Any]) -> AdaptedExample:
        chain = record.get("Full_Session_Chain")
        if not isinstance(chain, list):
            raise ValueError("Official MemConflict record Full_Session_Chain must be a list")
        record_id = str(record.get("ID", ""))
        question_entry = record.get("_question_entry")
        if not isinstance(question_entry, tuple) or len(question_entry) != 2:
            raise ValueError("Official MemConflict question entry is missing")
        session_index, question = question_entry
        native_type = question.get("conflict_type")
        normalized_type = {
            "dynamic_conflict": "dynamic",
            "static_conflict": "static",
            "conditional_conflict": "conditional",
        }.get(native_type)
        if normalized_type is None:
            raise ValueError(f"Unsupported official MemConflict question conflict_type: {native_type!r}")
        validity_dimension = {
            "dynamic": "temporal_validity",
            "static": "factual_validity",
            "conditional": "contextual_applicability",
        }[normalized_type]

        interactions: list[RawInteraction] = []
        for session in chain[: session_index + 1]:
            session_id = str(session.get("Session_ID"))
            timestamp = _parse_timestamp(str(session.get("Date")) if session.get("Date") else None)
            dialogue = session.get("Session_Dialogue") or {}
            dialogue_items = []
            for key, messages in dialogue.items():
                match = re.search(r"(\d+)$", str(key))
                order = int(match.group(1)) if match else len(dialogue_items)
                if isinstance(messages, list):
                    dialogue_items.append((order, str(key), messages))
            for _, turn_key, messages in sorted(dialogue_items, key=lambda item: (item[0], item[1])):
                for message_index, message in enumerate(messages):
                    if not isinstance(message, dict) or message.get("content") is None:
                        continue
                    interactions.append(RawInteraction(
                        interaction_id=f"{record_id}::s{session_id}::{turn_key}::{message_index}",
                        user_id=record_id,
                        content=str(message["content"]),
                        timestamp=timestamp,
                        role=str(message.get("role", "user")),
                        session_id=session_id,
                        metadata={"dataset": self.name, "session_id": session_id},
                    ))

        question_id = str(question.get("question_id"))
        query_id = f"{record_id}::s{chain[session_index].get('Session_ID')}::{question_id}"
        gold = {
            "question_id": question_id,
            "answer": question.get("answer"),
            "conflict_type": native_type,
            "ability_target": question.get("ability_target"),
            "difficulty": question.get("difficulty"),
            "updated_attributes": [
                {
                    "session_id": session.get("Session_ID"),
                    "date": session.get("Date"),
                    "attributes": session.get("Updated_Attributes") or [],
                }
                for session in chain[: session_index + 1]
                if session.get("Updated_Attributes")
            ],
            "session_history": [
                {"session_id": session.get("Session_ID"), "date": session.get("Date")}
                for session in chain[: session_index + 1]
            ],
        }
        return AdaptedExample(
            query=Query(
                query_id=query_id,
                user_id=record_id,
                text=str(question.get("question", "")),
                current_time=_parse_timestamp(str(chain[session_index].get("Date")))
                if chain[session_index].get("Date") else None,
                gold_answer=None,
                metadata={
                    "dataset": self.name,
                    "conflict_type": normalized_type,
                    "task_type": normalized_type,
                    "native_conflict_type": native_type,
                    "validity_dimension": validity_dimension,
                    "source_record_id": record_id,
                    "source_session_id": chain[session_index].get("Session_ID"),
                },
            ),
            interactions=tuple(interactions),
            oracle_governance=(),
            metadata={
                "dataset": self.name,
                "conflict_type": normalized_type,
                "native_conflict_type": native_type,
                "validity_dimension": validity_dimension,
                "source_record_id": record_id,
                "governance_source": MEMCONFLICT_GOLD_SOURCE,
            },
            evaluator_metadata={
                "governance_source": MEMCONFLICT_GOLD_SOURCE,
                "source_record_id": record_id,
                "source_session_id": chain[session_index].get("Session_ID"),
                "gold": gold,
            },
        )

    def _official_questions(self, record: dict[str, Any]) -> Iterable[AdaptedExample]:
        chain = record.get("Full_Session_Chain") or []
        for session_index, session in enumerate(chain):
            for question in session.get("Session_Questions") or []:
                prepared = dict(record)
                prepared["_question_entry"] = (session_index, question)
                yield self._adapt_official_question(prepared)
