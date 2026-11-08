from __future__ import annotations

import re
from typing import Iterable

from .schema import MemoryRecord


def _extract_answer(content: str) -> str | None:
    match = re.search(r"answer:\s*([^.;\n]+)", content, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


class DeterministicMemoryAgent:
    def answer(self, question: str, retrieved_records: Iterable[MemoryRecord]) -> str:
        records = list(retrieved_records)
        if not records:
            return "unknown"

        for record in records:
            extracted = _extract_answer(record.content)
            if extracted:
                return extracted

        return "unknown"
