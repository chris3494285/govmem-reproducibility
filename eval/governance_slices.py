from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


def load_governance_source(path: str | Path) -> list[dict]:
    source_path = Path(path)
    examples: list[dict] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _format_dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _has_answer_pattern(text: str) -> bool:
    return bool(re.search(r"answer:\s*([^\n.;]+)", text, flags=re.IGNORECASE))


def _extract_answer(text: str) -> str | None:
    match = re.search(r"answer:\s*([^\n.;]+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _is_valid_memory(memory: dict, current_time: datetime) -> bool:
    if memory.get("is_deleted"):
        return False
    valid_from = _parse_dt(memory.get("valid_from"))
    valid_until = _parse_dt(memory.get("valid_until"))
    if valid_from is not None and valid_from > current_time:
        return False
    if valid_until is not None and valid_until < current_time:
        return False
    return True


def _answer_memories(example: dict) -> list[dict]:
    current_time = _parse_dt(example["current_time"])
    assert current_time is not None
    return [memory for memory in example["memories"] if _has_answer_pattern(memory["content"]) and _is_valid_memory(memory, current_time)]


def _latest_answer_memory(example: dict) -> dict | None:
    answer_memories = _answer_memories(example)
    if not answer_memories:
        return None
    return max(answer_memories, key=lambda memory: (_parse_dt(memory["timestamp"]) or datetime.min, memory["memory_id"]))


def _wrong_answer(answer: str) -> str:
    if not answer:
        return "old value"
    lowered = answer.lower()
    if lowered.startswith("old "):
        return f"former {answer}"
    return f"old {answer}"


def _copy_memory(memory: dict, *, memory_id: str | None = None) -> dict:
    copied = dict(memory)
    if memory_id is not None:
        copied["memory_id"] = memory_id
    return copied


def _unsafe_memory_template(example: dict, answer: str, suffix: str, *, scope: str = "default") -> dict:
    current_time = _parse_dt(example["current_time"])
    assert current_time is not None
    question = example["question"]
    wrong_answer = _wrong_answer(answer)
    return {
        "memory_id": f"{example['id']}::{suffix}",
        "user_id": example["user_id"],
        "content": f"{question} {suffix.replace('_', ' ')}. answer: {wrong_answer}",
        "timestamp": _format_dt(current_time - timedelta(days=90)),
        "source": "user_message",
        "scope": scope,
        "valid_from": _format_dt(current_time - timedelta(days=120)),
        "valid_until": None,
        "is_deleted": False,
        "confidence": 0.99,
        "memory_type": "episodic",
        "entity_key": "governance:stress",
    }


def _build_slice_base(example: dict, slice_id: str, *, answer: str) -> dict:
    return {
        "id": f"{example['id']}::{slice_id}",
        "user_id": example["user_id"],
        "question": example["question"],
        "answer": answer,
        "current_time": example["current_time"],
        "allowed_scopes": list(example["allowed_scopes"]),
        "memories": [],
    }


def _generate_conflict_slice(example: dict) -> dict | None:
    target = _latest_answer_memory(example)
    if target is None:
        return None
    current_answer = example["answer"]
    slice_example = _build_slice_base(example, "conflict", answer=current_answer)
    memories = [_copy_memory(memory) for memory in example["memories"]]
    old_memory = _unsafe_memory_template(example, current_answer, "conflict")
    old_memory["timestamp"] = _format_dt(_parse_dt(example["current_time"]) - timedelta(days=365))
    old_memory["valid_from"] = old_memory["timestamp"]
    old_memory["entity_key"] = target["entity_key"]
    memories.append(old_memory)
    slice_example["memories"] = memories
    return slice_example


def _generate_expired_slice(example: dict) -> dict | None:
    target = _latest_answer_memory(example)
    if target is None:
        return None
    current_answer = example["answer"]
    current_time = _parse_dt(example["current_time"])
    assert current_time is not None
    slice_example = _build_slice_base(example, "expired", answer=current_answer)
    memories = [_copy_memory(memory) for memory in example["memories"]]
    expired_memory = _unsafe_memory_template(example, current_answer, "expired")
    expired_memory["timestamp"] = _format_dt(current_time - timedelta(days=180))
    expired_memory["valid_from"] = _format_dt(current_time - timedelta(days=365))
    expired_memory["valid_until"] = _format_dt(current_time - timedelta(days=1))
    expired_memory["entity_key"] = target["entity_key"]
    memories.append(expired_memory)
    slice_example["memories"] = memories
    return slice_example


def _generate_deleted_slice(example: dict) -> dict | None:
    target = _latest_answer_memory(example)
    if target is None:
        return None
    current_answer = example["answer"]
    slice_example = _build_slice_base(example, "deleted", answer=current_answer)
    memories = [_copy_memory(memory) for memory in example["memories"]]
    deleted_memory = _unsafe_memory_template(example, current_answer, "deleted")
    deleted_memory["is_deleted"] = True
    deleted_memory["entity_key"] = target["entity_key"]
    memories.append(deleted_memory)
    slice_example["memories"] = memories
    return slice_example


def _generate_unauthorized_slice(example: dict) -> dict | None:
    target = _latest_answer_memory(example)
    if target is None:
        return None
    current_answer = example["answer"]
    slice_example = _build_slice_base(example, "unauthorized", answer=current_answer)
    memories = [_copy_memory(memory) for memory in example["memories"]]
    private_memory = _unsafe_memory_template(example, current_answer, "unauthorized", scope="private")
    private_memory["entity_key"] = target["entity_key"]
    memories.append(private_memory)
    slice_example["memories"] = memories
    return slice_example


def _invalidate_answer_memory(example: dict, memory: dict, index: int) -> dict:
    current_time = _parse_dt(example["current_time"])
    assert current_time is not None
    mutated = _copy_memory(memory, memory_id=f"{memory['memory_id']}::abstain::{index}")
    answer = _extract_answer(memory["content"]) or example["answer"]
    mutated["content"] = f"{example['question']} abstain stress case {index}. answer: {_wrong_answer(answer)}"
    if index % 3 == 0:
        mutated["is_deleted"] = True
    elif index % 3 == 1:
        mutated["scope"] = "private"
    else:
        mutated["valid_until"] = _format_dt(current_time - timedelta(days=1))
    return mutated


def _generate_abstain_slice(example: dict) -> dict:
    slice_example = _build_slice_base(example, "abstain", answer="unknown")
    memories: list[dict] = []
    answer_index = 0
    for memory in example["memories"]:
        if _has_answer_pattern(memory["content"]):
            memories.append(_invalidate_answer_memory(example, memory, answer_index))
            answer_index += 1
        else:
            memories.append(_copy_memory(memory))
    slice_example["memories"] = memories
    return slice_example


def generate_governance_slices(examples: Iterable[dict]) -> list[dict]:
    slices: list[dict] = []
    for example in examples:
        for builder in (
            _generate_conflict_slice,
            _generate_expired_slice,
            _generate_deleted_slice,
            _generate_unauthorized_slice,
        ):
            derived = builder(example)
            if derived is not None:
                slices.append(derived)
        slices.append(_generate_abstain_slice(example))
    return slices


def write_governance_slices(input_path: str | Path, output_path: str | Path) -> None:
    examples = load_governance_source(input_path)
    slices = generate_governance_slices(examples)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in slices:
            handle.write(json.dumps(example, sort_keys=True))
            handle.write("\n")

