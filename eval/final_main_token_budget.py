"""Provider-neutral, versioned output-token budget resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FROZEN_BUDGETS = {
    "longmemeval": 256,
    "memora_email_writeup": 512,
    "memora_preference": 256,
    "memora_project_proposal": 768,
    "memora_meeting_notes": 256,
    "memora_todos": 256,
    "oaks_novel": 128,
}

GEMINI_BUDGETS = {
    "longmemeval": 1024,
    "memora_email_writeup": 2048,
    "memora_meeting_notes": 1536,
    "memora_preference": 1024,
    "memora_project_proposal": 4096,
    "memora_todos": 1024,
    "oaks_novel": 768,
}

GEMINI_FLASH_BUDGETS = {
    "longmemeval": 1536,
    "memora_email_writeup": 3072,
    "memora_meeting_notes": 1024,
    "memora_preference": 1536,
    "memora_project_proposal": 3072,
    "memora_todos": 3072,
    "oaks_novel": 1024,
}

GEMINI_FLASH_REPAIR_BUDGETS = {
    "longmemeval": 3072,
    "memora_email_writeup": 6144,
    "memora_meeting_notes": 3072,
    "memora_preference": 3072,
    "memora_project_proposal": 6144,
    "memora_todos": 6144,
    "oaks_novel": 3072,
}

GEMINI_FLASH_REPAIR_V2_BUDGETS = {
    "longmemeval": 3072,
    "memora_email_writeup": 6144,
    "memora_meeting_notes": 6144,
    "memora_preference": 3072,
    "memora_project_proposal": 6144,
    "memora_todos": 6144,
    "oaks_novel": 3072,
}

QWEN_BUDGETS = {
    "longmemeval": 1536,
    "memora_email_writeup": 3072,
    "memora_meeting_notes": 1024,
    "memora_preference": 1536,
    "memora_project_proposal": 3072,
    "memora_todos": 3072,
    "oaks_novel": 1024,
}

KIMI_BUDGETS = {
    "longmemeval": 1536,
    "memora_email_writeup": 3072,
    "memora_meeting_notes": 1024,
    "memora_preference": 1536,
    "memora_project_proposal": 3072,
    "memora_todos": 3072,
    "oaks_novel": 1024,
}

KIMI_V2_BUDGETS = {
    "longmemeval": 1536,
    "memora_email_writeup": 4096,
    "memora_meeting_notes": 2048,
    "memora_preference": 1536,
    "memora_project_proposal": 6144,
    "memora_todos": 3072,
    "oaks_novel": 2048,
}

KIMI_V3_BUDGETS = {
    "longmemeval": 1536,
    "memora_email_writeup": 4096,
    "memora_meeting_notes": 2048,
    "memora_preference": 1536,
    "memora_project_proposal": 6144,
    "memora_todos": 3072,
    "oaks_novel": 4096,
}

KIMI_V4_BUDGETS = {
    "longmemeval": 3072,
    "memora_email_writeup": 8192,
    "memora_meeting_notes": 4096,
    "memora_preference": 3072,
    "memora_project_proposal": 12288,
    "memora_todos": 3072,
    "oaks_novel": 8192,
}

KIMI_REPAIR_V1_BUDGETS = {
    "longmemeval": 3072,
    "memora_email_writeup": 8192,
    "memora_meeting_notes": 4096,
    "memora_preference": 6144,
    "memora_project_proposal": 12288,
    "memora_todos": 3072,
    "oaks_novel": 8192,
}

GROK_V1_BUDGETS = {
    "longmemeval": 1024,
    "memora_email_writeup": 2048,
    "memora_meeting_notes": 1024,
    "memora_preference": 1024,
    "memora_project_proposal": 3072,
    "memora_todos": 1536,
    "oaks_novel": 1024,
}


def memora_task_subtype(row: dict[str, Any]) -> str:
    """Resolve Memora subtype from the authoritative frozen source ID.

    The historical task_scoring_spec was broad enough to label meeting-note
    rows as project proposals.  Source question IDs are the authoritative
    subtype discriminator for the frozen manifest.
    """
    if row.get("source_dataset") != "memora":
        return ""
    metadata = row.get("task_source_metadata") or {}
    candidates = [row.get("source_example_id")]
    if isinstance(metadata, dict):
        candidates.extend((metadata.get("question_id"), metadata.get("source_example_id")))
    text = " ".join(str(value).lower() for value in candidates if value)
    if "content_project_proposal" in text:
        return "content_project_proposal"
    if "content_meeting_notes" in text:
        return "content_meeting_notes"
    if "content_email_writeup" in text or "email_writeup" in text:
        return "content_email_writeup"
    if "activity_todos" in text:
        return "activity_todos"
    if "pref_" in text or "preference" in text:
        return "preference_recommendation"
    return "memora_unresolved"


def budget_class(row: dict[str, Any]) -> str:
    source = row.get("source_dataset")
    if source == "longmemeval":
        return "longmemeval"
    if source == "oaks_novel":
        return "oaks_novel"
    subtype = memora_task_subtype(row)
    return {
        "content_project_proposal": "memora_project_proposal",
        "content_meeting_notes": "memora_meeting_notes",
        "content_email_writeup": "memora_email_writeup",
        "preference_recommendation": "memora_preference",
        "activity_todos": "memora_todos",
    }.get(subtype, "memora_unresolved")


def resolve_max_output_tokens(row: dict[str, Any], config: dict[str, Any] | None = None) -> int:
    budgets = FROZEN_BUDGETS if config is None else config.get("budgets", {})
    key = budget_class(row)
    # The legacy v1 smoke manifest predates task/oracle payload fields.  Its
    # historical runner used the generic 256-token fallback; retain that
    # behavior when this optional resolver is used without a task class.
    if key == "memora_unresolved":
        return 256
    if key not in budgets:
        raise ValueError(f"No token budget configured for class {key}")
    return int(budgets[key])


def load_budget_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "final-main-token-budget-v2":
        raise ValueError("Unsupported final-main token-budget configuration")
    if config.get("budgets", {}).get("memora_project_proposal") != 1536:
        raise ValueError("Versioned proposal budget must be 1536")
    for key, value in FROZEN_BUDGETS.items():
        if key != "memora_project_proposal" and config.get("budgets", {}).get(key) != value:
            raise ValueError(f"Versioned token budget changed frozen class {key}")
    return config


def load_gemini_budget_config(path: str | Path) -> dict[str, Any]:
    """Load Gemini-only measurement ceilings; never changes frozen defaults."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "final-main-token-budget-gemini-v1":
        raise ValueError("Unsupported Gemini token-budget configuration")
    budgets = config.get("budgets")
    if not isinstance(budgets, dict) or set(budgets) != set(GEMINI_BUDGETS):
        raise ValueError("Gemini configuration must contain exactly all known budget classes")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in budgets.values()):
        raise ValueError("Gemini token budgets must be positive integers")
    if budgets != GEMINI_BUDGETS:
        raise ValueError("Gemini token budgets do not match approved measurement ceilings")
    return config


def load_gemini_flash_budget_config(path: str | Path) -> dict[str, Any]:
    """Load only the Gemini Flash measurement-ceiling configuration."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "final-main-token-budget-gemini-flash-v1":
        raise ValueError("Unsupported Gemini Flash token-budget configuration")
    if config.get("provider") != "gemini" or config.get("model") != "gemini-3-flash-preview":
        raise ValueError("Gemini Flash configuration has invalid provider/model metadata")
    budgets = config.get("budgets")
    if not isinstance(budgets, dict) or set(budgets) != set(GEMINI_FLASH_BUDGETS):
        raise ValueError("Gemini Flash configuration must contain exactly all known budget classes")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in budgets.values()):
        raise ValueError("Gemini Flash token budgets must be positive integers")
    if budgets != GEMINI_FLASH_BUDGETS:
        raise ValueError("Gemini Flash token budgets do not match approved measurement ceilings")
    return config


def load_gemini_flash_repair_budget_config(path: str | Path) -> dict[str, Any]:
    """Load explicit Gemini Flash selective-repair measurement ceilings."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "final-main-token-budget-gemini-flash-repair-v1":
        raise ValueError("Unsupported Gemini Flash repair token-budget configuration")
    if config.get("provider") != "gemini" or config.get("model") != "gemini-3-flash-preview":
        raise ValueError("Gemini Flash repair configuration has invalid provider/model metadata")
    budgets = config.get("budgets")
    if not isinstance(budgets, dict) or set(budgets) != set(GEMINI_FLASH_REPAIR_BUDGETS):
        raise ValueError("Gemini Flash repair configuration must contain exactly all known budget classes")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in budgets.values()):
        raise ValueError("Gemini Flash repair token budgets must be positive integers")
    if budgets != GEMINI_FLASH_REPAIR_BUDGETS:
        raise ValueError("Gemini Flash repair token budgets do not match approved repair ceilings")
    return config


def load_gemini_flash_repair_v2_budget_config(path: str | Path) -> dict[str, Any]:
    """Load the second-stage Gemini Flash selective-repair ceilings."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "final-main-token-budget-gemini-flash-repair-v2":
        raise ValueError("Unsupported Gemini Flash repair-v2 token-budget configuration")
    if config.get("provider") != "gemini" or config.get("model") != "gemini-3-flash-preview":
        raise ValueError("Gemini Flash repair-v2 configuration has invalid provider/model metadata")
    budgets = config.get("budgets")
    if not isinstance(budgets, dict) or set(budgets) != set(GEMINI_FLASH_REPAIR_V2_BUDGETS):
        raise ValueError("Gemini Flash repair-v2 configuration must contain exactly all known budget classes")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in budgets.values()):
        raise ValueError("Gemini Flash repair-v2 token budgets must be positive integers")
    if budgets != GEMINI_FLASH_REPAIR_V2_BUDGETS:
        raise ValueError("Gemini Flash repair-v2 token budgets do not match approved repair-v2 ceilings")
    return config


def load_qwen_budget_config(path: str | Path) -> dict[str, Any]:
    """Load Qwen-only measurement ceilings without changing frozen defaults."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "final-main-token-budget-qwen-v1":
        raise ValueError("Unsupported Qwen token-budget configuration")
    if config.get("provider") != "qwen" or config.get("model") != "qwen3.7-plus-2026-05-26":
        raise ValueError("Qwen configuration has invalid provider/model metadata")
    budgets = config.get("budgets")
    if not isinstance(budgets, dict) or set(budgets) != set(QWEN_BUDGETS):
        raise ValueError("Qwen configuration must contain exactly all known budget classes")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in budgets.values()):
        raise ValueError("Qwen token budgets must be positive integers")
    if budgets != QWEN_BUDGETS:
        raise ValueError("Qwen token budgets do not match approved measurement ceilings")
    return config


def load_kimi_budget_config(path: str | Path) -> dict[str, Any]:
    """Load Kimi-only measurement ceilings without changing frozen defaults."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "final-main-token-budget-kimi-v1":
        raise ValueError("Unsupported Kimi token-budget configuration")
    if config.get("provider") != "kimi" or config.get("model") != "kimi-k2.6":
        raise ValueError("Kimi configuration has invalid provider/model metadata")
    budgets = config.get("budgets")
    if not isinstance(budgets, dict) or set(budgets) != set(KIMI_BUDGETS):
        raise ValueError("Kimi configuration must contain exactly all known budget classes")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in budgets.values()):
        raise ValueError("Kimi token budgets must be positive integers")
    if budgets != KIMI_BUDGETS:
        raise ValueError("Kimi token budgets do not match approved measurement ceilings")
    return config


def load_kimi_v2_budget_config(path: str | Path) -> dict[str, Any]:
    """Load the strict Kimi K2.6 v2 measurement ceilings."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "final-main-token-budget-kimi-v2":
        raise ValueError("Unsupported Kimi v2 token-budget configuration")
    if config.get("provider") != "kimi" or config.get("model") != "kimi-k2.6":
        raise ValueError("Kimi v2 configuration has invalid provider/model metadata")
    budgets = config.get("budgets")
    if not isinstance(budgets, dict) or set(budgets) != set(KIMI_V2_BUDGETS):
        raise ValueError("Kimi v2 configuration must contain exactly all known budget classes")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in budgets.values()):
        raise ValueError("Kimi v2 token budgets must be positive integers")
    if budgets != KIMI_V2_BUDGETS:
        raise ValueError("Kimi v2 token budgets do not match approved measurement ceilings")
    return config


def load_kimi_v3_budget_config(path: str | Path) -> dict[str, Any]:
    """Load the strict Kimi K2.6 v3 measurement ceilings."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "final-main-token-budget-kimi-v3":
        raise ValueError("Unsupported Kimi v3 token-budget configuration")
    if config.get("provider") != "kimi" or config.get("model") != "kimi-k2.6":
        raise ValueError("Kimi v3 configuration has invalid provider/model metadata")
    budgets = config.get("budgets")
    if not isinstance(budgets, dict) or set(budgets) != set(KIMI_V3_BUDGETS):
        raise ValueError("Kimi v3 configuration must contain exactly all known budget classes")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in budgets.values()):
        raise ValueError("Kimi v3 token budgets must be positive integers")
    if budgets != KIMI_V3_BUDGETS:
        raise ValueError("Kimi v3 token budgets do not match approved measurement ceilings")
    return config


def load_kimi_v4_budget_config(path: str | Path) -> dict[str, Any]:
    """Load the strict Kimi K2.6 v4 measurement ceilings."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "final-main-token-budget-kimi-v4":
        raise ValueError("Unsupported Kimi v4 token-budget configuration")
    if config.get("provider") != "kimi" or config.get("model") != "kimi-k2.6":
        raise ValueError("Kimi v4 configuration has invalid provider/model metadata")
    budgets = config.get("budgets")
    if not isinstance(budgets, dict) or set(budgets) != set(KIMI_V4_BUDGETS):
        raise ValueError("Kimi v4 configuration must contain exactly all known budget classes")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in budgets.values()):
        raise ValueError("Kimi v4 token budgets must be positive integers")
    if budgets != KIMI_V4_BUDGETS:
        raise ValueError("Kimi v4 token budgets do not match approved measurement ceilings")
    return config


def load_kimi_repair_v1_budget_config(path: str | Path) -> dict[str, Any]:
    """Load the strict Kimi selective-repair v1 measurement ceilings."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "final-main-token-budget-kimi-repair-v1":
        raise ValueError("Unsupported Kimi repair v1 token-budget configuration")
    if config.get("provider") != "kimi" or config.get("model") != "kimi-k2.6":
        raise ValueError("Kimi repair v1 configuration has invalid provider/model metadata")
    budgets = config.get("budgets")
    if not isinstance(budgets, dict) or set(budgets) != set(KIMI_REPAIR_V1_BUDGETS):
        raise ValueError("Kimi repair v1 configuration must contain exactly all known budget classes")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in budgets.values()):
        raise ValueError("Kimi repair v1 token budgets must be positive integers")
    if budgets != KIMI_REPAIR_V1_BUDGETS:
        raise ValueError("Kimi repair v1 token budgets do not match approved measurement ceilings")
    return config


def load_grok_v1_budget_config(path: str | Path) -> dict[str, Any]:
    """Load strict xAI Grok v1 measurement ceilings."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "final-main-token-budget-grok-v1":
        raise ValueError("Unsupported Grok v1 token-budget configuration")
    if config.get("provider") != "xai" or config.get("model") != "grok-4.3":
        raise ValueError("Grok v1 configuration has invalid provider/model metadata")
    budgets = config.get("budgets")
    if not isinstance(budgets, dict) or set(budgets) != set(GROK_V1_BUDGETS):
        raise ValueError("Grok v1 configuration must contain exactly all known budget classes")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in budgets.values()):
        raise ValueError("Grok v1 token budgets must be positive integers")
    if budgets != GROK_V1_BUDGETS:
        raise ValueError("Grok v1 token budgets do not match approved measurement ceilings")
    return config


REPAIR_BUDGETS = {
    "longmemeval": 512,
    "memora_email_writeup": 1024,
    "memora_meeting_notes": 1024,
    "memora_preference": 768,
    "memora_project_proposal": 1536,
    "memora_todos": 512,
    "oaks_novel": 256,
}


def load_repair_budget_config(path: str | Path) -> dict[str, Any]:
    """Load only the explicit Anthropic selective-repair budget contract."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "final-main-token-budget-anthropic-repair-v1":
        raise ValueError("Unsupported Anthropic repair token-budget configuration")
    budgets = config.get("budgets")
    if not isinstance(budgets, dict) or set(budgets) != set(REPAIR_BUDGETS):
        raise ValueError("Anthropic repair configuration must contain exactly all known budget classes")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in budgets.values()):
        raise ValueError("Anthropic repair token budgets must be positive integers")
    if budgets != REPAIR_BUDGETS:
        raise ValueError("Anthropic repair token budgets do not match the approved repair configuration")
    return config


def project_proposal_condition_ids(rows: list[dict[str, Any]]) -> set[str]:
    selected = {row["condition_id"] for row in rows if budget_class(row) == "memora_project_proposal"}
    if not selected:
        raise ValueError("No authoritative Memora content_project_proposal conditions found")
    return selected
