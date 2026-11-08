from __future__ import annotations

import json
from datetime import datetime
from json import JSONDecoder, JSONDecodeError
from pathlib import Path
from typing import Iterator


_OFFICIAL_DATETIME_FORMAT = "%Y/%m/%d (%a) %H:%M"


def _parse_official_datetime(value: str) -> datetime:
    try:
        return datetime.strptime(value, _OFFICIAL_DATETIME_FORMAT)
    except ValueError as exc:
        raise ValueError(f"Unsupported LongMemEval datetime format: {value!r}") from exc


def _format_iso_datetime(value: str) -> str:
    return _parse_official_datetime(value).isoformat()


def _iter_json_array_items(path: Path) -> Iterator[dict]:
    decoder = JSONDecoder()
    buffer = ""
    started = False

    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk and not buffer:
                break
            buffer += chunk

            while True:
                if not started:
                    stripped = buffer.lstrip()
                    if not stripped:
                        break
                    if not stripped.startswith("["):
                        raise ValueError(f"{path} is not a JSON array")
                    buffer = stripped[1:]
                    started = True
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer.startswith("]"):
                    return
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                try:
                    obj, idx = decoder.raw_decode(buffer)
                except JSONDecodeError:
                    break
                yield obj
                buffer = buffer[idx:]

            if not chunk:
                break

    buffer = buffer.lstrip()
    if buffer and not buffer.startswith("]"):
        raise ValueError(f"{path} ended before the JSON array was fully parsed")


def load_official_longmemeval(path: str, limit: int | None = None) -> list[dict]:
    source = Path(path)
    examples: list[dict] = []
    for item in _iter_json_array_items(source):
        examples.append(item)
        if limit is not None and len(examples) >= limit:
            break
    return examples


def convert_official_example(raw: dict, include_assistant: bool = False) -> dict:
    question_id = raw["question_id"]
    question_type = raw.get("question_type")
    current_time = _format_iso_datetime(raw["question_date"])
    user_id = f"official_longmemeval::{question_id}"
    memories: list[dict] = []

    haystack_sessions = raw.get("haystack_sessions", [])
    haystack_session_ids = raw.get("haystack_session_ids", [])
    haystack_dates = raw.get("haystack_dates", [])

    for session_index, session in enumerate(haystack_sessions):
        session_id = haystack_session_ids[session_index] if session_index < len(haystack_session_ids) else f"session_{session_index}"
        session_timestamp = haystack_dates[session_index] if session_index < len(haystack_dates) else raw["question_date"]
        session_timestamp_iso = _format_iso_datetime(session_timestamp)

        for message_index, message in enumerate(session):
            role = message.get("role", "unknown")
            if not include_assistant and role != "user":
                continue

            memory = {
                "memory_id": f"{question_id}::{session_id}::{message_index}",
                "user_id": user_id,
                "content": message.get("content", ""),
                "timestamp": session_timestamp_iso,
                "source": "user_message" if role == "user" else "assistant_message" if role == "assistant" else "unknown",
                "scope": "default",
                # Official cleaned data keeps the session timestamp as metadata, but
                # we do not turn it into a hard validity window. That keeps the raw
                # official ingestion path focused on retrieval/evidence evaluation
                # rather than temporal hard-filtering.
                "valid_from": None,
                "valid_until": None,
                "is_deleted": False,
                "confidence": 1.0,
                "memory_type": "episodic",
                "entity_key": (
                    f"official:{question_type}:{session_id}:{message_index}"
                    if question_type
                    else f"official:memory:{session_id}:{message_index}"
                ),
                "official_effective_time": session_timestamp_iso,
                "official_session_id": session_id,
                "official_question_type": question_type,
                "role": role,
            }
            if "has_answer" in message:
                memory["has_answer"] = message["has_answer"]
            memories.append(memory)

    return {
        "id": question_id,
        "user_id": user_id,
        "question": raw["question"],
        "answer": raw["answer"],
        "current_time": current_time,
        "allowed_scopes": ["default"],
        "dataset_mode": "official_longmemeval",
        "memories": memories,
    }


def convert_official_file(
    input_path: str,
    output_path: str,
    limit: int | None = None,
    include_assistant: bool = False,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, raw in enumerate(_iter_json_array_items(Path(input_path))):
            if limit is not None and index >= limit:
                break
            converted = convert_official_example(raw, include_assistant=include_assistant)
            handle.write(json.dumps(converted, sort_keys=True))
            handle.write("\n")
