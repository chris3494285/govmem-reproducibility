"""Deterministic, source-aware scoring for main-condition smoke responses.

This module never calls a provider.  It is deliberately conservative for
open-ended Memora generation tasks: without a benchmark-native exact or
structured criterion, the result is explicitly unscored.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


def normalize_text(value: Any) -> str:
    """Normalize surface variation without applying fuzzy semantic matching."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("_", " ").replace("-", " ")
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return " ".join(text.split())


def _accepted(expected: Any) -> list[str]:
    if expected is None:
        return []
    text = str(expected)
    # LongMemEval occasionally encodes accepted alternatives in one native
    # answer string; split only the explicit benchmark convention.
    parts = re.split(r"\s+or\s+", text, flags=re.IGNORECASE)
    return [normalize_text(part).strip(" .") for part in parts if normalize_text(part).strip(" .")]


def scoring_spec(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("source_dataset")
    expected = row.get("task_oracle_answer")
    if source == "longmemeval":
        accepted = _accepted(expected)
        return {"method": "normalized_exact_or_native_alternative", "deterministic": True, "exact_match_appropriate": True, "accepted_answers": accepted, "native_abstention_answers": sorted(_native_abstention_set(row)), "native_answer_is_abstention": any(value in _native_abstention_set(row) for value in accepted)}
    if source == "oaks_novel":
        return {"method": "normalized_native_option_match", "deterministic": True, "exact_match_appropriate": True, "accepted_answers": _accepted(expected), "options": row.get("task_options") or []}
    if source == "memora":
        evidence = expected if isinstance(expected, dict) else {}
        subtype = memora_subtype(evidence)
        if subtype == "preference_recommendation":
            return {"method": "memora_preference_constraints_v1", "subtype": subtype, "deterministic": True, "exact_match_appropriate": False}
        if subtype == "activity_todos":
            return {"method": "memora_activity_todos_v1", "subtype": subtype, "deterministic": True, "exact_match_appropriate": False}
        values = []
        def collect(value: Any, *, in_content_data: bool = False) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in {"session_id", "session_history", "relevant_source_session_ids"}:
                        continue
                    if key in {"value", "item"} and isinstance(child, str) and child.strip():
                        values.append(child)
                    elif key not in {"session_ids"}:
                        collect(child, in_content_data=in_content_data or key in {"content_data", "memory_items"})
            elif isinstance(value, list):
                for child in value:
                    collect(child, in_content_data=in_content_data)
            elif in_content_data and isinstance(value, str) and value.strip():
                # Memora generation tasks expose structured content_data rather
                # than a scalar answer.  Its leaf facts are the native scoring
                # target; session/history bookkeeping is deliberately ignored.
                values.append(value)
            elif in_content_data and isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(value)
        collect(evidence)
        values = list(dict.fromkeys(values))
        if values:
            return {"method": "required_native_fact_presence", "subtype": subtype, "deterministic": True, "exact_match_appropriate": False, "required_facts": values}
        return {"method": "unscored_no_native_criterion", "deterministic": True, "exact_match_appropriate": False}
    return {"method": "unscored_unknown_source", "deterministic": True, "exact_match_appropriate": False}


def score_task_answer(row: dict[str, Any], answer: Any) -> tuple[bool | None, str]:
    if not isinstance(answer, str) or not answer.strip():
        return None, "unscored_empty_answer"
    if row.get("source_dataset") == "longmemeval":
        outcome = score_longmemeval_answer(row, answer)
        return outcome["correctness"], str(outcome["reason"])
    spec = row.get("task_scoring_spec") or scoring_spec(row)
    answer_norm = normalize_text(answer).strip(" .")
    method = spec.get("method")
    if method in {"normalized_exact_or_native_alternative", "normalized_native_option_match"}:
        accepted = [normalize_text(value).strip(" .") for value in spec.get("accepted_answers", [])]
        return (any(value and (answer_norm == value or value in answer_norm) for value in accepted), "scored_native_exact") if accepted else (None, "insufficient_gold")
    if method == "required_native_fact_presence":
        facts = [normalize_text(value).strip(" .") for value in spec.get("required_facts", []) if normalize_text(value).strip(" .")]
        if not facts:
            return None, "insufficient_gold"
        return (all(fact in answer_norm for fact in facts), "scored_required_native_facts")
    return None, str(method or "unscored_no_native_criterion")


SCORING_VERSION = "final-main-source-aware-v2"

# These are intentionally exact normalized strings.  They are the answer
# forms used by LongMemEval's insufficient-information convention, not a
# semantic matcher for arbitrary answers.
LONGMEMEVAL_ABSTENTION_ANSWERS = frozenset({
    "i don't know",
    "i do not know",
    "unknown",
    "information insufficient",
    "insufficient information",
    "we cannot answer this question at this point",
})


def _native_abstention_set(row: dict[str, Any]) -> set[str]:
    values = row.get("native_abstention_answers")
    if values is None:
        values = row.get("task_native_abstention_answers")
    if values is None:
        values = LONGMEMEVAL_ABSTENTION_ANSWERS
    if isinstance(values, str):
        values = [values]
    return {normalize_text(value).strip(" .") for value in values if str(value).strip()}


def score_longmemeval_answer(row: dict[str, Any], answer: Any) -> dict[str, Any]:
    """Compare a LongMemEval answer using exact native answer conventions.

    Abstention is a separate outcome because native LongMemEval questions can
    make "information insufficient" a correct answer.  No semantic or fuzzy
    equivalence is attempted here.
    """
    answer_norm = normalize_text(answer).strip(" .") if isinstance(answer, str) else ""
    accepted = [normalize_text(value).strip(" .") for value in _accepted(row.get("task_oracle_answer"))]
    native_abstention = answer_norm in _native_abstention_set(row)
    expected_abstention = bool(row.get("native_answer_is_abstention") or row.get("task_native_answer_is_abstention"))
    expected_abstention = expected_abstention or any(value in _native_abstention_set(row) for value in accepted)
    if native_abstention:
        return {
            "correctness": expected_abstention,
            "outcome": "native_abstention",
            "abstention": True,
            "reason": "native_abstention_answer" if expected_abstention else "abstention_for_answerable_item",
        }
    if not accepted:
        return {"correctness": None, "outcome": "unscored", "abstention": False, "reason": "insufficient_gold"}
    if len(set(accepted)) != len(accepted):
        return {"correctness": None, "outcome": "ambiguous", "abstention": False, "reason": "duplicate_native_answers"}
    return {
        "correctness": answer_norm in set(accepted),
        "outcome": "answer",
        "abstention": False,
        "reason": "scored_native_exact_or_alternative",
    }


def _numeric_values(value: Any) -> set[int | float]:
    """Return deterministic numeric interpretations, including k/m suffixes."""
    text = str(value).casefold().replace(",", "")
    found: set[int | float] = set()
    for match in re.finditer(r"(?<![a-z0-9])\$?([0-9]+(?:\.[0-9]+)?)\s*([km])?(?![a-z0-9])", text):
        number = float(match.group(1))
        suffix = match.group(2)
        if suffix == "k":
            number *= 1_000
        elif suffix == "m":
            number *= 1_000_000
        found.add(int(number) if number.is_integer() else number)
    return found


def _fact_supported(fact: Any, answer: str) -> bool:
    fact_norm = normalize_text(fact)
    answer_norm = normalize_text(answer)
    if not fact_norm:
        return False
    if fact_norm in answer_norm:
        return True
    expected_numbers = _numeric_values(fact)
    return bool(expected_numbers & _numeric_values(answer))


_RECOMMENDATION_CUES = re.compile(
    r"\b(recommend(?:s|ed|ation)?|suggest(?:s|ed|ion)?|try|choose|pick|go with|would be a good|consider)\b",
    re.IGNORECASE,
)


def _recommendation_evidence(item: Any, answer: str) -> bool:
    """Require an item plus an explicit recommendation/completion cue."""
    if not _fact_supported(item, answer):
        return False
    # Native Memora answers often use a compact comma/and-separated list of
    # concrete recommendations.  Accept that form, but do not mistake a
    # preference restatement ("I like X") for task completion.
    preference_only = re.search(r"\b(i\s+like|i\s+love|i\s+prefer|my\s+(?:favorite|favourite))\b", answer, re.IGNORECASE)
    if not _RECOMMENDATION_CUES.search(answer) and preference_only:
        return False
    item_norm = normalize_text(item)
    answer_norm = normalize_text(answer)
    # A negated recommendation is not a recommendation of the item.
    for match in re.finditer(re.escape(item_norm), answer_norm):
        prefix = answer_norm[max(0, match.start() - 24):match.start()]
        if re.search(r"\b(do not|don't|dont|not|avoid|never)\s*$", prefix):
            continue
        return True
    return False


def _memora_facts(value: Any, path: str = "") -> list[dict[str, Any]]:
    """Flatten native content_data into auditable field/item requirements."""
    facts: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            facts.extend(_memora_facts(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            facts.extend(_memora_facts(child, f"{path}[{index}]"))
    elif value not in (None, ""):
        facts.append({"path": path, "expected": value})
    return facts


def memora_subtype(oracle: Any) -> str:
    if not isinstance(oracle, dict):
        return "unknown"
    if "remaining_tasks" in oracle:
        return "activity_todos"
    if "specific_items" in oracle or "subcategory_data" in oracle or "memory_items" in oracle:
        return "preference_recommendation"
    content = oracle.get("content_data")
    if isinstance(content, dict):
        if "email_purpose" in content or "call_to_action" in content:
            return "content_email_writeup"
        return "content_project_proposal"
    return "unknown"


def memora_token_budget(oracle: Any) -> int | None:
    return {
        "preference_recommendation": 256,
        "activity_todos": 256,
        "content_email_writeup": 512,
        "content_project_proposal": 768,
    }.get(memora_subtype(oracle))


def max_output_tokens_for_row(row: dict[str, Any]) -> int:
    source = row.get("source_dataset")
    if source == "longmemeval":
        return 256
    if source == "oaks_novel":
        return 128
    if source == "memora":
        return memora_token_budget(row.get("task_oracle_answer")) or 256
    return 256


def _native_items(oracle: dict[str, Any], key: str) -> list[str]:
    values: list[str] = []
    value = oracle.get(key)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                candidate = item.get("item", item.get("value"))
                if isinstance(candidate, str) and candidate.strip():
                    values.append(candidate)
            elif isinstance(item, str) and item.strip():
                values.append(item)
    return list(dict.fromkeys(values))


def _preference_constraints(oracle: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    data = oracle.get("subcategory_data") or {}
    likes = _native_items(data, "likes")
    dislikes = _native_items(data, "dislikes")
    specific = _native_items(oracle, "specific_items")
    memory_items = oracle.get("memory_items") or {}
    visited = []
    if isinstance(memory_items, dict):
        for category, constraints in memory_items.items():
            if not isinstance(constraints, dict):
                continue
            category_likes = _native_items(constraints, "likes")
            category_dislikes = _native_items(constraints, "dislikes")
            if category == "already_visited_list":
                visited.extend(category_likes)
            else:
                likes.extend(category_likes)
                dislikes.extend(category_dislikes)
    if not likes:
        likes = specific
    likes = list(dict.fromkeys(likes))
    dislikes = list(dict.fromkeys(dislikes))
    visited = list(dict.fromkeys(visited))
    return likes, dislikes, visited


def resolve_oaks_option(row: dict[str, Any], answer: Any) -> str | None:
    """Resolve exactly one native option; never infer an option index/letter."""
    if not isinstance(answer, str) or not answer.strip():
        return None
    answer_norm = normalize_text(answer)
    options = [option for option in row.get("task_options") or [] if isinstance(option, str) and option.strip()]
    valid_labels = "".join(chr(97 + index) for index in range(len(options)))
    if len(answer_norm) == 1 and answer_norm in valid_labels:
        return options[ord(answer_norm) - ord("a")]
    exact = [option for option in options if normalize_text(option) == answer_norm]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None
    contained = [option for option in options if normalize_text(option) in answer_norm]
    return contained[0] if len(contained) == 1 else None


def score_task_answer_detailed(row: dict[str, Any], answer: Any) -> dict[str, Any]:
    """Return the stable common scoring contract plus source-specific details."""
    result: dict[str, Any] = {
        "source_dataset": row.get("source_dataset"),
        "subtype": None,
        "correctness": None,
        "scoring_status": "unscored",
        "scoring_method": None,
        "scoring_version": SCORING_VERSION,
        "normalized_model_answer": normalize_text(answer) if isinstance(answer, str) else None,
        "normalized_oracle_answer": None,
        "scoring_details": {},
        "deterministic_score": None,
        "native_validation_score": None,
        "adjudication_required": False,
    }
    spec = row.get("task_scoring_spec") or scoring_spec(row)
    result["scoring_method"] = spec.get("method")
    if "native_correctness" in row:
        result["native_validation_score"] = row.get("native_correctness")
    elif "native_score" in row:
        result["native_validation_score"] = row.get("native_score")
    source = row.get("source_dataset")
    result["subtype"] = memora_subtype(row.get("task_oracle_answer")) if source == "memora" else None
    if source == "oaks_novel":
        resolved = resolve_oaks_option(row, answer)
        oracle = row.get("task_oracle_answer")
        result["normalized_oracle_answer"] = normalize_text(oracle) if oracle is not None else None
        result["scoring_details"] = {"resolved_option": resolved, "oracle_option": oracle}
        if resolved is None:
            result["scoring_details"]["reason"] = "oaks_zero_or_ambiguous_option_matches"
            result["scoring_details"]["ambiguous_or_unscored"] = True
            return result
        result.update({"correctness": normalize_text(resolved) == normalize_text(oracle), "scoring_status": "scored", "deterministic_score": normalize_text(resolved) == normalize_text(oracle)})
        return result
    if source == "longmemeval":
        native = score_longmemeval_answer(row, answer)
        result["correctness"] = native["correctness"]
        result["scoring_status"] = "scored" if native["correctness"] is not None else ("adjudication_required" if native["outcome"] == "ambiguous" else "unscored")
        result["deterministic_score"] = native["correctness"]
        result["scoring_details"] = native
        return result
    if source == "memora":
        expected = row.get("task_oracle_answer")
        subtype = memora_subtype(expected)
        if isinstance(expected, dict) and (expected.get("adjudication_required") or expected.get("native_evaluation_required")):
            result["adjudication_required"] = True
            result["scoring_status"] = "adjudication_required"
            result["scoring_details"] = {"subtype": subtype, "reason": "native_evaluation_required"}
            return result
        if subtype == "preference_recommendation":
            likes, dislikes, visited = _preference_constraints(expected)
            positive_supported = [item for item in likes if _recommendation_evidence(item, answer)]
            forbidden = list(dict.fromkeys(dislikes + visited))
            violations = [item for item in forbidden if _recommendation_evidence(item, answer)]
            result["scoring_method"] = "memora_preference_constraints_v1"
            result["scoring_details"] = {
                "subtype": subtype,
                "required_positive_count": len(likes),
                "required_positive_correct_count": len(positive_supported),
                "positive_recall": len(positive_supported) / len(likes) if likes else None,
                "forbidden_item_count": len(forbidden),
                "forbidden_item_violation_count": len(violations),
                "forbidden_items_found": violations,
                "already_visited_items": visited,
                "preference_compatibility": bool(not violations),
                "concrete_recommendation_count": len(positive_supported),
            }
            if not likes:
                # For travel schemas with no positive preferences, explicit
                # negative/visited constraints are still a deterministic
                # native criterion: positive compatibility is vacuous, while
                # recommending a forbidden or already-visited item is not.
                if forbidden:
                    result["scoring_details"]["reason"] = "scored_native_negative_constraints_only"
                    result.update({"correctness": not violations, "scoring_status": "scored", "deterministic_score": not violations})
                    return result
                result["scoring_details"]["reason"] = "insufficient_native_positive_preferences_or_recommendation_target"
                result["adjudication_required"] = True
                result["scoring_status"] = "adjudication_required"
                return result
            result.update({"correctness": len(positive_supported) == len(likes) and not violations, "scoring_status": "scored", "deterministic_score": len(positive_supported) == len(likes) and not violations})
            return result
        if subtype == "activity_todos":
            remaining = _native_items(expected, "remaining_tasks")
            negative: list[str] = []
            for key in ("completed_tasks", "completed", "not_remaining", "done_tasks"):
                negative.extend(_native_items(expected, key))
            covered = [item for item in remaining if _fact_supported(item, answer)]
            false_remaining = [item for item in dict.fromkeys(negative) if _fact_supported(item, answer)]
            result["scoring_method"] = "memora_activity_todos_v1"
            result["scoring_details"] = {
                "subtype": subtype,
                "remaining_task_recall": len(covered) / len(remaining) if remaining else None,
                "remaining_task_count": len(remaining),
                "remaining_task_correct_count": len(covered),
                "false_remaining_task_count": len(false_remaining),
                "false_remaining_tasks": false_remaining,
            }
            if not remaining:
                result["scoring_details"]["reason"] = "insufficient_native_remaining_tasks"
                return result
            result.update({"correctness": len(covered) == len(remaining) and not false_remaining, "scoring_status": "scored", "deterministic_score": len(covered) == len(remaining) and not false_remaining})
            return result
        content_data = expected.get("content_data") if isinstance(expected, dict) else None
        facts = _memora_facts(content_data) if content_data is not None else []
        covered = [{**fact, "supported": _fact_supported(fact["expected"], answer)} for fact in facts]
        supported_n = sum(bool(fact["supported"]) for fact in covered)
        result["scoring_details"] = {
            "fields": covered,
            "field_correct_count": supported_n,
            "field_total_count": len(covered),
            "field_accuracy": supported_n / len(covered) if covered else None,
            "required_item_correct_count": supported_n,
            "required_item_total_count": len(covered),
            "required_item_recall": supported_n / len(covered) if covered else None,
        }
        result["normalized_oracle_answer"] = json.dumps(content_data, ensure_ascii=False, sort_keys=True) if content_data is not None else None
        if not covered:
            result["scoring_details"]["reason"] = "insufficient_native_structured_criterion"
            return result
        result.update({"correctness": supported_n == len(covered), "scoring_status": "scored", "deterministic_score": supported_n == len(covered)})
        return result
    score, reason = score_task_answer(row, answer)
    result.update({"correctness": score, "scoring_status": "scored" if score is not None else "unscored", "deterministic_score": score})
    result["scoring_details"] = {"reason": reason}
    oracle = row.get("task_oracle_answer")
    result["normalized_oracle_answer"] = normalize_text(oracle) if oracle is not None else None
    return result


__all__ = ["SCORING_VERSION", "LONGMEMEVAL_ABSTENTION_ANSWERS", "max_output_tokens_for_row", "memora_subtype", "normalize_text", "resolve_oaks_option", "score_longmemeval_answer", "score_task_answer", "score_task_answer_detailed", "scoring_spec"]
