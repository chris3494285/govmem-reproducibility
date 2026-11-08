"""Adapter for the public Memora long-term memory benchmark format."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .interfaces import AdaptedExample, Query, RawInteraction


MEMORA_GOLD_SOURCE = "dataset_native_gold"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class MemoraDatasetAdapter:
    """Normalize a Memora persona-period scope or an explicitly supplied parent.

    A direct persona directory loads one persona. A period directory loads its
    immediate persona directories; a ``data`` parent explicitly loads all
    period/persona directories in sorted order. Conversation messages are
    model-visible; lifecycle operations and evaluation evidence remain in
    ``AdaptedExample.evaluator_metadata``.
    """

    name = "memora"

    def load(self, path: str | Path) -> list[AdaptedExample]:
        root = Path(path)
        if (root / "conversations").is_dir():
            return self._load_persona(root)
        persona_dirs = sorted(
            child for child in root.iterdir() if child.is_dir() and (child / "conversations").is_dir()
        ) if root.is_dir() else []
        if persona_dirs:
            return [example for persona in persona_dirs for example in self._load_persona(persona, prefix=persona.name)]
        period_dirs = sorted(
            child for child in root.iterdir()
            if child.is_dir() and any((grandchild / "conversations").is_dir() for grandchild in child.iterdir())
        ) if root.is_dir() else []
        if period_dirs:
            return [
                example
                for period in period_dirs
                for persona in sorted(child for child in period.iterdir() if child.is_dir() and (child / "conversations").is_dir())
                for example in self._load_persona(persona, prefix=f"{period.name}::{persona.name}")
            ]
        raise ValueError(
            "Memora input must be a persona directory, a period directory, "
            "or an explicitly supplied data parent containing period/persona directories"
        )

    def _load_persona(self, root: Path, prefix: str | None = None) -> list[AdaptedExample]:
        question_files = sorted(root.glob("evaluation_questions_*.json"))
        if len(question_files) != 1:
            raise ValueError("Memora input must contain exactly one evaluation_questions_<persona>.json file")
        question_file = question_files[0]
        question_data = _load_json(question_file)
        sessions = [_load_json(session_path) for session_path in sorted((root / "conversations").glob("session_*.json"))]
        return [self.adapt_question(question, question_data, sessions, query_id_prefix=prefix) for question in self._questions(question_data)]

    @staticmethod
    def _questions(question_data: dict[str, Any]) -> list[dict[str, Any]]:
        questions: list[dict[str, Any]] = []
        for task_type, task_questions in question_data.get("questions", {}).items():
            for question in task_questions:
                questions.append({**question, "task_type": task_type})
        return questions

    def adapt_question(
        self,
        question: dict[str, Any],
        question_data: dict[str, Any],
        sessions: list[dict[str, Any]],
        query_id_prefix: str | None = None,
    ) -> AdaptedExample:
        persona = question_data["persona"]
        interactions: list[RawInteraction] = []
        for session in sessions:
            session_id = str(session["session_id"])
            timestamp = _parse_date(session.get("date"))
            for turn in session.get("conversation", []):
                interactions.append(
                    RawInteraction(
                        interaction_id=f"{persona}::{session_id}::{turn['turn']}",
                        user_id=persona,
                        content=turn["message"],
                        timestamp=timestamp,
                        role=turn.get("speaker", "user"),
                        session_id=session_id,
                        metadata={
                            "dataset": self.name,
                            "persona": persona,
                            "turn": turn["turn"],
                            "date": session.get("date"),
                        },
                    )
                )

        question_date = question.get("question_date")
        gold = {
            "memory_evidence": question.get("memory_evidence", {}),
            "forgetting_evidence": question.get("forgetting_evidence", {}),
            "evaluation": question.get("evaluation", {}),
        }
        session_operations = [
            {
                "session_id": session.get("session_id"),
                "session_type": session.get("session_type"),
                "operation": session.get("operation"),
                "operation_details": session.get("operation_details"),
                "date": session.get("date"),
            }
            for session in sessions
        ]
        session_turn_annotations = [
            {
                "session_id": session.get("session_id"),
                "turn": turn.get("turn"),
                "share_memory": turn.get("share_memory"),
            }
            for session in sessions
            for turn in session.get("conversation", [])
            if "share_memory" in turn
        ]
        return AdaptedExample(
            query=Query(
                query_id=(f"{query_id_prefix}::{question['question_id']}" if query_id_prefix else question["question_id"]),
                user_id=persona,
                text=question["question"],
                current_time=_parse_date(question_date),
                metadata={
                    "dataset": self.name,
                    "persona": persona,
                    "task_type": question.get("task_type"),
                    "native_task_type": question.get("task_type"),
                    "lifecycle_applicable": question.get("task_type") == "remembering",
                    "question_date": question_date,
                },
            ),
            interactions=tuple(interactions),
            oracle_governance=(),
            metadata={
                "dataset": self.name,
                "persona": persona,
                "date_range": dict(question_data.get("date_range", {})),
                "governance_source": MEMORA_GOLD_SOURCE,
            },
            evaluator_metadata={
                "governance_source": MEMORA_GOLD_SOURCE,
                "question_id": question["question_id"],
                "persona": persona,
                "task_type": question.get("task_type"),
                "native_task_type": question.get("task_type"),
                "lifecycle_applicable": question.get("task_type") == "remembering",
                "question_date": question_date,
                "gold": gold,
                "session_operations": session_operations,
                "session_turn_annotations": session_turn_annotations,
            },
        )
