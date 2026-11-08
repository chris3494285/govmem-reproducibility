"""Adapter for the public GateMem paired episode/checkpoint format."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .interfaces import AdaptedExample, Query, RawInteraction


GATEMEM_GOLD_SOURCE = "dataset_native_gold"
_GOLD_FIELDS = (
    "query_type",
    "attack_type",
    "expected_action",
    "answer",
    "answer_redacted",
    "refuse",
    "no_memory",
    "judge_spec",
    "leak_targets",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _checkpoint_paths(path: str | Path) -> tuple[Path, Path]:
    root = Path(path)
    if root.is_file():
        raise ValueError("GateMem input must be a domain directory containing episodes.jsonl and checkpoints.jsonl")
    return root / "episodes.jsonl", root / "checkpoints.jsonl"


class GateMemDatasetAdapter:
    """Normalize a GateMem domain directory into checkpoint-level examples.

    Model-visible interactions contain only episode turns available through
    ``as_of_turn_id``. GateMem's checkpoint annotations are retained under
    ``AdaptedExample.evaluator_metadata`` and never copied into interaction
    content or query text.
    """

    name = "gatemem"

    def load(self, path: str | Path) -> list[AdaptedExample]:
        episodes_path, checkpoints_path = _checkpoint_paths(path)
        episodes = {episode["episode_id"]: episode for episode in _load_jsonl(episodes_path)}
        checkpoints = _load_jsonl(checkpoints_path)
        return [self.adapt_example(checkpoint, episodes[checkpoint["episode_id"]]) for checkpoint in checkpoints]

    def adapt_example(self, checkpoint: dict[str, Any], episode: dict[str, Any]) -> AdaptedExample:
        episode_id = checkpoint["episode_id"]
        if episode.get("episode_id") != episode_id:
            raise ValueError(f"Episode mismatch for checkpoint {checkpoint.get('checkpoint_id')!r}")

        boundary = checkpoint["as_of_turn_id"]
        turns = episode.get("turns", [])
        boundary_index = next((index for index, turn in enumerate(turns) if turn["turn_id"] == boundary), None)
        if boundary_index is None:
            raise ValueError(f"Unknown as_of_turn_id {boundary!r} in episode {episode_id!r}")

        interactions: list[RawInteraction] = []
        for turn in turns[: boundary_index + 1]:
            speaker = turn["speaker"]
            interactions.append(
                RawInteraction(
                    interaction_id=f"{episode_id}::{turn['turn_id']}",
                    user_id=speaker["principal_id"],
                    content=turn["text"],
                    timestamp=_parse_timestamp(turn.get("timestamp")),
                    role=speaker.get("role", "user"),
                    metadata={
                        "turn_id": turn["turn_id"],
                        "turn_kind": turn.get("turn_kind"),
                        "episode_id": episode_id,
                        "domain": episode.get("domain"),
                    },
                )
            )

        asker = checkpoint["asker"]
        query = Query(
            query_id=checkpoint["checkpoint_id"],
            user_id=asker["principal_id"],
            text=checkpoint["query_text"],
            current_time=interactions[-1].timestamp if interactions else None,
            requester_role=asker.get("role"),
            metadata={
                "dataset": self.name,
                "episode_id": episode_id,
                "checkpoint_position": boundary,
                "domain": episode.get("domain"),
                "query_type": checkpoint.get("query_type"),
                "task_type": checkpoint.get("query_type"),
            },
        )

        gold = {field: checkpoint[field] for field in _GOLD_FIELDS if field in checkpoint}
        return AdaptedExample(
            query=query,
            interactions=tuple(interactions),
            oracle_governance=(),
            metadata={
                "dataset": self.name,
                "episode_id": episode_id,
                "domain": episode.get("domain"),
                "principals": tuple(dict(principal) for principal in episode.get("entities", {}).get("principals", [])),
                "relationships": tuple(dict(relationship) for relationship in episode.get("entities", {}).get("relationships", [])),
                "checkpoint_position": boundary,
                "governance_source": GATEMEM_GOLD_SOURCE,
            },
            evaluator_metadata={
                "governance_source": GATEMEM_GOLD_SOURCE,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "episode_id": episode_id,
                "as_of_turn_id": boundary,
                "asker": dict(asker),
                "gold": gold,
            },
        )
