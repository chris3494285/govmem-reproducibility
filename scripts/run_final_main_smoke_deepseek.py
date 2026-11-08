"""Run the frozen main-condition smoke cohort through DeepSeek.

The default mode is a zero-call request-construction preflight.  The real
mode is intentionally a thin downstream transport loop; condition semantics
come exclusively from the frozen condition manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import tempfile
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from eval.final_main_token_budget import (
    budget_class as authoritative_budget_class,
    resolve_max_output_tokens,
)
from eval.system_validation_scoring import max_output_tokens_for_row, memora_subtype, score_task_answer, score_task_answer_detailed, scoring_spec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/final_main_governance_smoke_condition_manifest_v2.jsonl"
DEFAULT_RESULTS = ROOT / "results/final_main_smoke_deepseek_v4_flash"
EXPECTED_V1_MANIFEST_SHA = "7fbc728ec8cd7336ecc76fc8c2e3a7299c3dfdc9d9b9190658582e31cb93a638"
EXPECTED_V2_MANIFEST_SHA = "af4daeefd35bfd8a19c89133f20a77cf531dcf00582f9658fcdac9a05073535e"
# Backward-compatible name used by the v1 smoke tests.
EXPECTED_MANIFEST_SHA = EXPECTED_V1_MANIFEST_SHA
DEFAULT_MODEL = "deepseek-v4-flash"
PROVIDER = "deepseek"
TEMPERATURE = 0
MAX_OUTPUT_TOKENS = 256
SCHEMA_VERSION = "final-main-smoke-deepseek-v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen smoke condition manifest is missing: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Smoke manifest must contain JSON objects")
    return rows


def validate_manifest(rows: list[dict[str, Any]], manifest_sha: str, *, expected_sha256: set[str] | None = None, expected_counts: dict[str, int] | None = None, expected_base_count: int | None = None) -> None:
    accepted_hashes = expected_sha256 or {EXPECTED_V1_MANIFEST_SHA, EXPECTED_V2_MANIFEST_SHA}
    if manifest_sha not in accepted_hashes:
        raise ValueError(f"Frozen smoke manifest SHA256 mismatch: {manifest_sha}")
    expected_total = sum(expected_counts.values()) if expected_counts else 160
    if len(rows) != expected_total:
        raise ValueError(f"Expected {expected_total} frozen conditions, found {len(rows)}")
    if len({row.get("condition_id") for row in rows}) != expected_total:
        raise ValueError("Duplicate or missing condition_id in smoke manifest")
    by_base: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for field in ("base_id", "source_dataset", "analysis_event_type", "condition_type", "oracle_state"):
            if field not in row:
                raise ValueError(f"Condition {row.get('condition_id')} lacks {field}")
        if row["condition_type"] not in {"clean", "corrupted", "enforced"}:
            raise ValueError(f"Unknown condition type: {row['condition_type']}")
        by_base.setdefault(row["base_id"], []).append(row)
    if expected_base_count is None:
        expected_base_count = 24
    if len(by_base) != expected_base_count or sum(row["condition_type"] == "clean" for row in rows) != expected_base_count:
        raise ValueError(f"Frozen cohort must contain {expected_base_count} bases and clean rows")
    for base_id, group in by_base.items():
        if sum(row["condition_type"] == "clean" for row in group) != 1:
            raise ValueError(f"Base {base_id} does not have exactly one clean condition")
        corrupt = {row["pair_id"] for row in group if row["condition_type"] == "corrupted"}
        enforced = {row["pair_id"] for row in group if row["condition_type"] == "enforced"}
        if corrupt != enforced:
            raise ValueError(f"Incomplete corrupted/enforced pairs for {base_id}")
    counts = Counter(row["condition_type"] for row in rows)
    if expected_counts is None:
        expected_counts = {"clean": 24, "corrupted": 68, "enforced": 68}
    if counts != Counter(expected_counts):
        raise ValueError(f"Unexpected condition composition: {counts}")


def _canonical(value: Any) -> str:
    return " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())


def _visible_state(row: dict[str, Any]) -> dict[str, Any]:
    """Render neutral projected state; experiment labels never enter prompts."""
    if row["condition_type"] == "clean":
        return {"governance_state": row["oracle_state"]}
    if row["condition_type"] == "enforced":
        return {"governance_state": row["oracle_state"], "state_resolution": "validated"}
    operation = (row.get("corruption_specification") or {}).get("operator")
    projected = deepcopy(row["oracle_state"])
    if operation == "missing":
        projected["current_status"] = "unknown"
    elif operation == "stale":
        projected["current_status"] = "historical"
    elif operation == "flipped":
        if projected.get("current_status") == "valid":
            projected["current_status"] = "invalid"
        elif projected.get("current_status") == "invalid":
            projected["current_status"] = "valid"
    elif operation == "conflicting":
        projected["current_status"] = "ambiguous"
    return {"governance_state": projected}


def _context_text(row: dict[str, Any]) -> str:
    text = row.get("model_visible_context_text")
    if isinstance(text, str) and text.strip():
        return text
    # A legacy row may lack the v2 projection, but internal governance state
    # must never be serialized into a downstream prompt. Preserve the old
    # smoke test's distinction with neutral evidence text only; v2 rows always
    # provide their actual benchmark-native projection above.
    if row.get("condition_type") == "clean":
        return "No benchmark-native memory evidence is available in the admitted context."
    return "No benchmark-native memory evidence is available in the admitted context.\nAdditional admitted evidence is unavailable."


def render_prompt(row: dict[str, Any]) -> str:
    query = row.get("task_question") or row.get("question") or row.get("task") or row.get("query")
    if not query:
        query = "Answer using only the governed context provided below."
    context = _context_text(row)
    options = ""
    if row.get("task_answer_type") == "multiple_choice" and row.get("task_options"):
        labels = [chr(65 + index) for index in range(len(row["task_options"]))]
        options = "\n\nOptions:\n" + "\n".join(
            f"{labels[index]}. {option}" for index, option in enumerate(row["task_options"])
        )
    if row.get("task_answer_type") == "multiple_choice" and row.get("task_options"):
        labels = ", ".join(chr(65 + index) for index in range(len(row["task_options"])))
        instruction = (
            "Answer the user query using only the provided context. If the context "
            "does not support an answer, select the applicable option. Return only "
            "valid JSON with exactly one string field named \"answer\". The value "
            f"of \"answer\" must be exactly one of the option labels {labels}. Do not "
            "provide an explanation or option text."
        )
    else:
        instruction = (
            "Answer the user query using only the provided context. If the context "
            "does not support an answer, say that it cannot be determined. Do not "
            "invent facts. Return only valid JSON with one string field named answer."
        )
    return f"{instruction}\n\nUser query:\n{query}\n\nContext:\n{context}{options}"


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def _parse_answer(response: Any) -> str:
    parsed = getattr(response, "parsed", response)
    if isinstance(parsed, dict):
        answer = parsed.get("answer")
    else:
        answer = getattr(parsed, "answer", None)
    if not isinstance(answer, str):
        raise ValueError("DeepSeek response lacks string answer field")
    return answer


def _safe_raw(response: Any) -> Any:
    raw = getattr(response, "raw", None)
    if raw is None:
        return None
    if isinstance(raw, (str, int, float, bool, list, dict)):
        return raw
    # Keep diagnostics JSON-safe without serializing client internals.
    return {"type": type(raw).__name__, "repr": str(raw)[:4000]}


def _score(answer: str, row: dict[str, Any]) -> bool | None:
    score, _reason = score_task_answer(row, answer)
    return score


def _budget_class(row: dict[str, Any]) -> str:
    return authoritative_budget_class(row)


SUPPORTED_SCORING_METHODS = {
    "normalized_exact_or_native_alternative",
    "normalized_native_option_match",
    "required_native_fact_presence",
    "memora_preference_constraints_v1",
    "memora_activity_todos_v1",
}


def validate_scoring_preflight(rows: list[dict[str, Any]], *, strict: bool = True) -> dict[str, Any]:
    """Validate that every v2 condition has a deterministic scoring contract."""
    coverage = Counter()
    unresolved = []
    subtypes = Counter()
    for row in rows:
        spec = row.get("task_scoring_spec") or scoring_spec(row)
        method = spec.get("method")
        source = row.get("source_dataset")
        if method in SUPPORTED_SCORING_METHODS:
            coverage[source] += 1
        else:
            unresolved.append({"condition_id": row.get("condition_id"), "method": method or "missing"})
        if source == "memora":
            subtype = memora_subtype(row.get("task_oracle_answer"))
            subtypes[subtype] += 1
    if strict and unresolved:
        examples = ", ".join(f"{item['condition_id']}={item['method']}" for item in unresolved[:5])
        raise ValueError(f"Deterministic scoring preflight failed for {len(unresolved)} conditions: {examples}")
    return {"coverage_by_source": dict(coverage), "total_supported": sum(coverage.values()), "total": len(rows), "unresolved": unresolved, "memora_subtypes": dict(subtypes)}


def _identity(row: dict[str, Any], *, model: str, manifest_sha: str, prompt_sha: str) -> dict[str, Any]:
    return {"condition_id": row["condition_id"], "manifest_sha256": manifest_sha, "model": model, "model_parameters": {"temperature": TEMPERATURE, "max_output_tokens": MAX_OUTPUT_TOKENS, "thinking": "disabled", "tools": None}, "prompt_sha256": prompt_sha}


def _model_parameters(resolved_tokens: int, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    parameters = {
        "temperature": TEMPERATURE,
        "max_output_tokens": resolved_tokens,
        "thinking": "disabled",
        "tools": None,
    }
    if overrides:
        parameters.update(overrides)
    return parameters


def _load_cache(
    path: Path,
    *,
    model: str,
    manifest_sha: str,
    prompt_hashes: dict[str, str],
    token_budgets: dict[str, int],
    model_parameters: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("condition_id")
        if not cid or cid in cache:
            raise ValueError("Malformed or duplicate cached condition result")
        if row.get("manifest_sha256") != manifest_sha or row.get("model") != model:
            raise ValueError(f"Cached result identity mismatch for {cid}")
        if prompt_hashes.get(cid) != row.get("prompt_sha256"):
            raise ValueError(f"cached prompt identity mismatch for {cid}")
        expected_parameters = _model_parameters(
            token_budgets.get(cid, MAX_OUTPUT_TOKENS), model_parameters
        )
        if row.get("model_parameters") != expected_parameters:
            raise ValueError(f"cached model-settings identity mismatch for {cid}")
        cache[cid] = row
    return cache


def _provider_error(exc: Exception) -> tuple[str, str]:
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        return str(details.get("provider_exception", type(exc).__name__)), str(details.get("provider_message", exc))
    return type(exc).__name__, str(exc)


def _retryable_provider_error(exc: Exception) -> bool:
    """Retry only malformed JSON and provider/transport failures."""
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return True
    details = getattr(exc, "details", {})
    if isinstance(details, dict):
        if details.get("retryable") is False:
            return False
        if details.get("error_category") == "token_limit_truncation" or details.get("token_limit_truncation") is True:
            return False
        status = details.get("http_status")
        if status == 400:
            return False
        if status == 429:
            return True
        if isinstance(status, int) and 400 <= status < 500:
            return False
        if details.get("error_category") == "structural_invalid":
            return True
        if details.get("provider_exception") == "ConfigurationError":
            return False
        if status in {500, 502, 503}:
            return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "badrequest" in name or "invalid schema" in text or "unsupported parameter" in text or "invalid model parameter" in text:
        return False
    return "transport" in name or "connection" in name or "timeout" in name or "rate" in name or "provider" in name or any(term in text for term in ("connection", "timed out", "timeout", "temporar", "rate limit", "503", "502", "500"))


def _is_structural_failure(exc: Exception) -> bool:
    details = getattr(exc, "details", {})
    return isinstance(exc, (json.JSONDecodeError, ValueError)) or (
        isinstance(details, dict) and details.get("error_category") == "structural_invalid"
    )


def _retry_backoff_seconds(attempt_number: int, random_fn: Callable[[], float]) -> float:
    return (0.5 if attempt_number == 1 else 1.0) + (
        0.5 if attempt_number == 1 else 1.0
    ) * random_fn()


def _explicit_token_limit_truncation(row: dict[str, Any]) -> bool:
    """Recognize truncation only from explicit provider termination metadata."""
    values: list[Any] = [
        row.get("incomplete_details"),
        row.get("termination_reason"),
        row.get("finish_reason"),
        row.get("truncated_output"),
    ]
    for attempt in row.get("attempt_errors") or []:
        if isinstance(attempt, dict):
            if attempt.get("error_category") == "token_limit_truncation" or attempt.get("token_limit_truncation") is True:
                return True
            values.extend(
                (
                    attempt.get("incomplete_details"),
                    attempt.get("termination_reason"),
                    attempt.get("finish_reason"),
                    attempt.get("truncated_output"),
                )
            )
    for value in values:
        if value is True:
            return True
        if isinstance(value, dict):
            reason = str(value.get("reason") or value.get("type") or "").lower()
            if any(token in reason for token in ("max_output_tokens", "max_tokens", "length", "token_limit")):
                return True
        elif isinstance(value, str) and any(
            token in value.lower()
            for token in ("max_output_tokens", "max_tokens", "token_limit")
        ):
            return True
    return False


def _execute_condition(
    row: dict[str, Any],
    prompt: str,
    provider: Any,
    *,
    model: str,
    manifest_sha: str,
    budget_config: dict[str, Any] | None = None,
    model_parameters: dict[str, Any] | None = None,
    sleep_fn: Callable[[float], Any] = time.sleep,
    random_fn: Callable[[], float] = random.random,
) -> dict[str, Any]:
    """Run one condition with the frozen per-condition retry contract."""
    resolved_tokens = resolve_max_output_tokens(row, budget_config)
    result: dict[str, Any] = {
        "condition_id": row["condition_id"],
        "base_id": row["base_id"],
        "pair_id": row.get("pair_id"),
        "fault_operator": row.get("fault_operator"),
        "condition_type": row["condition_type"],
        "source_dataset": row["source_dataset"],
        "analysis_event_type": row["analysis_event_type"],
        "task_question": row.get("task_question", row.get("question", row.get("task"))),
        "task_oracle_answer": row.get("task_oracle_answer", row.get("oracle_answer", row.get("expected_answer"))),
        "provider": getattr(provider, "provider_name", PROVIDER),
        "model": model,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "model_visible_context_sha256": hashlib.sha256(_context_text(row).encode()).hexdigest(),
        "manifest_sha256": manifest_sha,
        "oracle_answer": row.get("task_oracle_answer", row.get("oracle_answer", row.get("expected_answer"))),
        "model_parameters": _model_parameters(resolved_tokens, model_parameters),
        "resolved_max_output_tokens": resolved_tokens,
        "attempt_count": 0,
        "retry_count": 0,
        "attempt_errors": [],
        "recovered_by_retry": False,
        "truncated_output": False,
        "truncation_reason": None,
        "response_status": None,
        "incomplete_details": None,
        "raw_output_text": None,
        "raw_response": None,
        "downstream_response": None,
        "parsed_answer": None,
        "parse_success": False,
        "correctness": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cache_hit_tokens": None,
        "latency_ms": None,
        "error_type": None,
        "error_message": None,
    }
    result["budget_class"] = _budget_class(row)
    started = time.perf_counter()
    for attempt_number in range(1, 4):
        result["attempt_count"] = attempt_number
        try:
            request_method = provider.request_downstream
            try:
                import inspect

                accepts_budget = "max_tokens" in inspect.signature(request_method).parameters
            except (TypeError, ValueError):
                accepts_budget = False
            response = (
                request_method(prompt, max_tokens=resolved_tokens)
                if accepts_budget
                else request_method(prompt)
            )
            answer = _parse_answer(response)
            telemetry = getattr(response, "telemetry", None)
            result.update(
                {
                    "raw_response": _safe_raw(response),
                    "downstream_response": answer,
                    "parsed_answer": answer,
                    "parse_success": True,
                    **score_task_answer_detailed(row, answer),
                    "input_tokens": getattr(telemetry, "input_tokens", None),
                    "output_tokens": getattr(telemetry, "output_tokens", None),
                    "total_tokens": getattr(telemetry, "total_tokens", None),
                    "thoughts_tokens": getattr(telemetry, "thoughts_tokens", None),
                    "cached_input_tokens": getattr(telemetry, "cached_input_tokens", None),
                    "cache_hit_tokens": getattr(telemetry, "cached_input_tokens", None),
                    "latency_ms": getattr(telemetry, "latency_ms", round((time.perf_counter() - started) * 1000, 3)),
                    "request_id": getattr(telemetry, "request_id", None),
                    "finish_reason": getattr(telemetry, "stop_reason", None),
                    "retry_count": attempt_number - 1,
                    "recovered_by_retry": attempt_number > 1,
                    "status": "success",
                    "error_type": None,
                    "error_message": None,
                }
            )
            return result
        except Exception as exc:
            error_type, error_message = _provider_error(exc)
            details = getattr(exc, "details", {})
            if isinstance(details, dict):
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "raw_output_text",
                    "response_status",
                    "incomplete_details",
                ):
                    if key in details:
                        result[key] = details[key]
                if details.get("error_category") == "token_limit_truncation" or details.get("token_limit_truncation") is True:
                    result["truncated_output"] = True
                    result["truncation_reason"] = "max_output_tokens"
            attempt_error = {
                "attempt": attempt_number,
                "error_type": error_type,
                "error_message": error_message,
                "http_status": details.get("http_status") if isinstance(details, dict) else None,
            }
            if isinstance(details, dict):
                for key in (
                    "error_category",
                    "token_limit_truncation",
                    "response_status",
                    "raw_output_text",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "incomplete_details",
                    "termination_reason",
                    "finish_reason",
                    "truncated_output",
                ):
                    if key in details:
                        attempt_error[key] = details[key]
            result["attempt_errors"].append(attempt_error)
            result.update(
                {
                    "error_type": error_type,
                    "error_message": error_message,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "retry_count": attempt_number - 1,
                }
            )
            if attempt_number >= 3 or not _retryable_provider_error(exc):
                result["status"] = "error"
                return result
            if not _is_structural_failure(exc):
                wait_seconds = _retry_backoff_seconds(attempt_number, random_fn)
                result["attempt_errors"][-1]["retry_wait_seconds"] = wait_seconds
                sleep_fn(wait_seconds)
    return result


def run(
    *,
    manifest: Path = DEFAULT_MANIFEST,
    results_dir: Path = DEFAULT_RESULTS,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    limit: int | None = None,
    force: bool = False,
    resume: bool = False,
    provider_factory: Callable[..., Any] | None = None,
    expected_sha256: set[str] | None = None,
    expected_counts: dict[str, int] | None = None,
    expected_base_count: int | None = None,
    output_filename: str = "smoke_results.jsonl",
    strict_scoring: bool | None = None,
    workers: int = 1,
    provider_name: str = PROVIDER,
    selected_condition_ids: set[str] | None = None,
    budget_config: dict[str, Any] | None = None,
    model_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    manifest_sha = sha256_file(manifest)
    rows = _read_manifest(manifest)
    validate_manifest(rows, manifest_sha, expected_sha256=expected_sha256, expected_counts=expected_counts, expected_base_count=expected_base_count)
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    if selected_condition_ids is not None:
        available = {row["condition_id"] for row in rows}
        missing = sorted(set(selected_condition_ids) - available)
        if missing:
            raise ValueError(
                "Requested condition_id not found in frozen manifest: "
                + ", ".join(missing)
            )
        rows = [row for row in rows if row["condition_id"] in selected_condition_ids]
    rows = rows if limit is None else rows[:limit]
    rendered = [(row, render_prompt(row)) for row in rows]
    prompt_hashes = [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for _, prompt in rendered]
    if any(not row.get("oracle_state") for row, _ in rendered):
        raise ValueError("Every smoke condition must have a non-empty oracle_state")
    scoring_preflight = validate_scoring_preflight(rows, strict=(manifest_sha == EXPECTED_V2_MANIFEST_SHA) if strict_scoring is None else strict_scoring)
    token_budgets = [resolve_max_output_tokens(row, budget_config) for row in rows]
    token_budget_classes = {
        name: resolve_max_output_tokens(
            next(row for row in rows if _budget_class(row) == name), budget_config
        )
        for name in sorted({_budget_class(row) for row in rows})
    }
    dry_rows = []
    for row, prompt in rendered:
        resolved_tokens = resolve_max_output_tokens(row, budget_config)
        dry_rows.append({"condition_id": row["condition_id"], "base_id": row["base_id"], "pair_id": row.get("pair_id"), "condition_type": row["condition_type"], "fault_operator": row.get("fault_operator"), "source_dataset": row["source_dataset"], "analysis_event_type": row["analysis_event_type"], "task_question": row.get("task_question"), "model": model, "provider": provider_name, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "model_visible_context_sha256": hashlib.sha256(_context_text(row).encode()).hexdigest(), "prompt_character_count": len(prompt), "context_character_count": len(_context_text(row)), "estimated_input_tokens": _estimate_tokens(prompt), "manifest_sha256": manifest_sha, "model_parameters": _model_parameters(resolved_tokens, model_parameters), "status": "dry_run"})
    if dry_run:
        lengths = sorted(r["prompt_character_count"] for r in dry_rows)
        p95 = lengths[min(len(lengths) - 1, max(0, int(len(lengths) * 0.95) - 1))]
        return {"rows": dry_rows, "summary": {"status": "DRY-RUN", "manifest_sha256": manifest_sha, "model": model, "provider": provider_name, "requested_rows": len(rows), "total_rows": len(rows), "condition_counts": dict(Counter(row["condition_type"] for row in rows)), "fault_counts": dict(Counter(row.get("fault_operator") for row in rows if row.get("fault_operator"))), "min_prompt_length": min(lengths), "median_prompt_length": statistics.median(lengths), "p95_prompt_length": p95, "max_prompt_length": max(lengths), "estimated_total_input_tokens": sum(r["estimated_input_tokens"] for r in dry_rows), "estimated_total_max_output_tokens": sum(token_budgets), "resolved_max_output_tokens": token_budget_classes, "resolved_token_budget_class_counts": dict(Counter(_budget_class(row) for row in rows)), "scoring_preflight": scoring_preflight, "api_calls_made": 0, "validation_errors": len(scoring_preflight["unresolved"]), "model_parameters": dry_rows[0]["model_parameters"] if dry_rows else {}}}
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / output_filename
    if output.exists() and not (force or resume):
        raise FileExistsError(f"Results exist; pass --resume or --force: {output}")
    cache = {} if force else _load_cache(output, model=model, manifest_sha=manifest_sha, prompt_hashes={row["condition_id"]: prompt_hash for row, prompt_hash in zip(rows, prompt_hashes)}, token_budgets={row["condition_id"]: resolve_max_output_tokens(row, budget_config) for row in rows}, model_parameters=model_parameters) if resume else {}
    provider = None
    attempted_condition_ids: set[str] = set()
    results: dict[str, dict[str, Any]] = dict(cache)
    try:
        if provider_factory:
            try:
                provider = provider_factory(model, temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS)
            except TypeError:
                provider = provider_factory(model)
        else:
            from eval.providers.deepseek_provider import DeepSeekDownstreamProvider
            provider = DeepSeekDownstreamProvider(model=model, temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS)
        for row, prompt in rendered:
            cid = row["condition_id"]
            cached = results.get(cid)
            if cached and cached.get("status") == "success" and (cached.get("downstream_response") or cached.get("parsed_answer")) and cached.get("error") is None:
                continue
            attempted_condition_ids.add(cid)
            result = _execute_condition(
                row,
                prompt,
                provider,
                model=model,
                manifest_sha=manifest_sha,
                budget_config=budget_config,
                model_parameters=model_parameters,
            )
            result["provider"] = provider_name
            results[cid] = result
            # Rewrite atomically after each condition so an interrupted run is resumable.
            fd, temp = tempfile.mkstemp(prefix=".smoke-results.", dir=results_dir, text=True)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    for item in sorted(results.values(), key=lambda x: x["condition_id"]):
                        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                os.replace(temp, output)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
    finally:
        if provider is not None and callable(getattr(provider, "close", None)):
            provider.close()
    ordered = [results[row["condition_id"]] for row, _ in rendered if row["condition_id"] in results]
    attempted_n = len(attempted_condition_ids)
    successful_n = sum(r.get("status") == "success" for r in ordered)
    failed_n = sum(r.get("status") == "error" for r in ordered)
    summary = {"status": "COMPLETED", "manifest_sha256": manifest_sha, "model": model, "provider": provider_name, "attempted_rows": attempted_n, "successful_rows": successful_n, "failed_rows": failed_n, "rows_succeeded_first_attempt": sum(r.get("status") == "success" and r.get("attempt_count", 0) == 1 for r in ordered), "rows_recovered_by_retry": sum(r.get("recovered_by_retry") is True for r in ordered), "rows_failed_after_retries": failed_n, "total_provider_attempts": sum(r.get("attempt_count", 0) for r in ordered if r.get("condition_id") in attempted_condition_ids), "retry_rate": sum(r.get("retry_count", 0) > 0 for r in ordered if r.get("condition_id") in attempted_condition_ids) / attempted_n if attempted_n else 0, "parse_failures_final": sum(r.get("parse_success") is not True and any("JSON" in str(e.get("error_type", "")) for e in r.get("attempt_errors", [])) for r in ordered), "parse_failures": sum(r.get("parse_success") is not True for r in ordered), "total_input_tokens": sum(r.get("input_tokens") or 0 for r in ordered), "total_output_tokens": sum(r.get("output_tokens") or 0 for r in ordered), "total_latency_ms": sum(r.get("latency_ms") or 0 for r in ordered), "resolved_max_output_tokens": token_budget_classes, "scoring_preflight": scoring_preflight, "model_parameters": _model_parameters(MAX_OUTPUT_TOKENS, model_parameters)}
    summary["latency_ms_p50"] = statistics.median([r["latency_ms"] for r in ordered if r.get("latency_ms") is not None]) if any(r.get("latency_ms") is not None for r in ordered) else None
    latencies = sorted(r["latency_ms"] for r in ordered if r.get("latency_ms") is not None)
    summary["latency_ms_p95"] = latencies[min(len(latencies) - 1, max(0, int(len(latencies) * 0.95) - 1))] if latencies else None
    (results_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"rows": ordered, "summary": summary}


def write_dry_run(result: dict[str, Any], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "dry_run_requests.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in result["rows"]), encoding="utf-8")
    (results_dir / "dry_run_summary.json").write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run(manifest=args.manifest, results_dir=args.results_dir, model=args.model, dry_run=args.dry_run, limit=args.limit, force=args.force, resume=args.resume)
    if args.dry_run:
        write_dry_run(result, args.results_dir)
        # Print representative request bodies for human preflight auditing.
        manifest_rows = _read_manifest(args.manifest)
        seen_sources = set()
        for item in result["rows"]:
            source = item["source_dataset"]
            if source in seen_sources:
                continue
            base_id = item["base_id"]
            siblings = [row for row in manifest_rows if row["base_id"] == base_id]
            if not {row["condition_type"] for row in siblings} >= {"clean", "corrupted", "enforced"}:
                continue
            seen_sources.add(source)
            print(f"PROMPT SAMPLE ({source})")
            for sibling in sorted(siblings, key=lambda row: (row["condition_type"] != "clean", row["condition_id"])):
                print(render_prompt(sibling))
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
