from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .interfaces import AdaptedExample, OracleGovernanceState, Query, RawInteraction


_LEGACY_GOVERNANCE_SOURCE = "legacy_demo_annotation"


def load_longmemeval_raw(path: str) -> list[dict]:
    raw_path = Path(path)
    examples: list[dict] = []
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def _entity_key_from_content(content: str) -> str:
    lowered = content.lower()
    if "live" in lowered or "moved" in lowered:
        return "profile:location"
    if "prefer" in lowered or "favorite" in lowered:
        return "preference:general"
    if "work" in lowered:
        return "profile:work"
    return "memory:general"


def convert_longmemeval_example(raw: dict) -> dict:
    memories: list[dict[str, Any]] = []
    for session in raw.get("sessions", []):
        session_id = session["session_id"]
        timestamp = session["timestamp"]
        messages = session.get("messages", [])
        for message_index, message in enumerate(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            memories.append(
                {
                    "memory_id": f"{raw['id']}::{session_id}::{message_index}",
                    "user_id": raw["user_id"],
                    "content": content,
                    "timestamp": timestamp,
                    "source": "user_message",
                    "scope": "default",
                    "valid_from": timestamp,
                    "valid_until": None,
                    "is_deleted": False,
                    "confidence": 1.0,
                    "memory_type": "episodic",
                    "entity_key": _entity_key_from_content(content),
                }
            )

    return {
        "id": raw["id"],
        "user_id": raw["user_id"],
        "question": raw["question"],
        "answer": raw["answer"],
        "current_time": raw["question_time"],
        "allowed_scopes": ["default"],
        "memories": memories,
    }


def convert_longmemeval_file(input_path: str, output_path: str) -> None:
    raw_examples = load_longmemeval_raw(input_path)
    output_path_obj = Path(output_path)
    with output_path_obj.open("w", encoding="utf-8") as handle:
        for raw in raw_examples:
            converted = convert_longmemeval_example(raw)
            handle.write(json.dumps(converted, sort_keys=True))
            handle.write("\n")


class LongMemEvalDatasetAdapter:
    """Adapter for the repository's simplified LongMemEval-style JSONL."""

    name = "longmemeval"

    def load(self, path: str | Path) -> list[AdaptedExample]:
        examples = load_longmemeval_raw(str(path))
        adapted: list[AdaptedExample] = []
        for index, example in enumerate(examples, start=1):
            if self._is_converted_schema(example):
                adapted.append(self.from_converted_example(example))
            elif self._is_raw_schema(example):
                adapted.append(self.adapt_example(example))
            else:
                raise ValueError(
                    "Unsupported LongMemEval record schema at record "
                    f"{index}: expected raw records with sessions/question_time "
                    "or converted records with memories/current_time"
                )
        return adapted

    @staticmethod
    def _is_raw_schema(example: dict[str, Any]) -> bool:
        return "sessions" in example and "question_time" in example and "question" in example

    @staticmethod
    def _is_converted_schema(example: dict[str, Any]) -> bool:
        return "memories" in example and "current_time" in example and "question" in example

    def adapt_example(self, raw: dict[str, Any]) -> AdaptedExample:
        converted = convert_longmemeval_example(raw)
        return self.from_converted_example(converted)

    def from_converted_example(self, example: dict[str, Any]) -> AdaptedExample:
        def parse(value: str | None) -> datetime | None:
            return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None

        interactions: list[RawInteraction] = []
        oracle: list[OracleGovernanceState] = []
        for memory in example["memories"]:
            interaction_id = memory["memory_id"]
            interactions.append(
                RawInteraction(
                    interaction_id=interaction_id,
                    user_id=memory["user_id"],
                    content=memory["content"],
                    timestamp=parse(memory.get("timestamp")),
                    role=memory.get("role", "user"),
                    session_id=memory.get("session_id"),
                    metadata=dict(memory),
                )
            )
            oracle.append(
                OracleGovernanceState(
                    interaction_id=interaction_id,
                    authorized_scopes=(memory["scope"],),
                    is_deleted=bool(memory["is_deleted"]),
                    valid_from=parse(memory.get("valid_from")),
                    valid_until=parse(memory.get("valid_until")),
                    entity_key=memory.get("entity_key"),
                    metadata={**memory, "governance_source": _LEGACY_GOVERNANCE_SOURCE},
                )
            )

        query = Query(
            query_id=example["id"],
            user_id=example["user_id"],
            text=example["question"],
            current_time=parse(example.get("current_time")),
            allowed_scopes=tuple(example.get("allowed_scopes", ())),
            gold_answer=example.get("answer"),
            metadata={"dataset": self.name, "dataset_mode": example.get("dataset_mode")},
        )
        return AdaptedExample(
            query=query,
            interactions=tuple(interactions),
            oracle_governance=tuple(oracle),
            metadata={
                "dataset": self.name,
                "dataset_mode": example.get("dataset_mode"),
                "governance_source": _LEGACY_GOVERNANCE_SOURCE,
            },
        )
