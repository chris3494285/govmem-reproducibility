"""Run the frozen full GovMem cohort through OpenAI Responses.

This is a provider-specific entry point over the same final-main execution
loop used by the validated DeepSeek runner.  It deliberately does not define
new prompts, scoring, retry, or cache semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from eval.final_main_token_budget import load_budget_config, memora_task_subtype, project_proposal_condition_ids, resolve_max_output_tokens
from eval.final_main_batch import batch_cost_estimate, batch_custom_id, build_batch_requests, read_batch_lines, validate_batch_output, write_batch_input
from eval.final_main_batch_lifecycle import execute_batch_lifecycle
from eval.providers.openai_provider import OpenAIDownstreamProvider
from eval.final_main_rerun_identity import assert_selective_identity, atomic_write_json
from eval.system_validation_scoring import score_task_answer_detailed

try:
    from scripts.run_final_main_smoke_deepseek import run as run_shared
    from scripts.run_final_main_smoke_deepseek import render_prompt
    from scripts.run_final_main_smoke_deepseek import _context_text, _parse_answer, _safe_raw
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from run_final_main_smoke_deepseek import run as run_shared
    from run_final_main_smoke_deepseek import render_prompt
    from run_final_main_smoke_deepseek import _context_text, _parse_answer, _safe_raw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
DEFAULT_RESULTS = ROOT / "results/final_main_openai_gpt56_sol"
EXPECTED_MANIFEST_SHA = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"
EXPECTED_V3_MANIFEST_SHA = "4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb"
ACCEPTED_MANIFEST_SHAS = {EXPECTED_MANIFEST_SHA, EXPECTED_V3_MANIFEST_SHA}
EXPECTED_COUNTS = {"clean": 731, "corrupted": 2080, "enforced": 2080}
EXPECTED_BASE_COUNT = 731
DEFAULT_MODEL = "gpt-5.6-sol"
PROVIDER = "openai"


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary_fields(rows: list[dict[str, Any]], budget_config: dict[str, Any] | None = None) -> dict[str, Any]:
    def grouped(fields: tuple[str, ...], *, include_none: bool = True) -> dict[str, int]:
        counts = Counter()
        for row in rows:
            values = tuple(row.get(field) for field in fields)
            if not include_none and any(value is None for value in values):
                continue
            counts["|".join("" if value is None else str(value) for value in values)] += 1
        return dict(sorted(counts.items()))

    budget_classes = Counter()
    budget_values: dict[str, int] = {}
    for row in rows:
        source = row.get("source_dataset")
        if source == "longmemeval":
            name = "longmemeval"
        elif source == "oaks_novel":
            name = "oaks_novel"
        else:
            subtype = memora_task_subtype(row)
            name = {
                "content_project_proposal": "memora_project_proposal",
                "content_meeting_notes": "memora_meeting_notes",
                "content_email_writeup": "memora_email_writeup",
                "preference_recommendation": "memora_preference",
                "activity_todos": "memora_todos",
            }.get(subtype, "memora_unresolved")
        budget_classes[name] += 1
        budget_values[name] = resolve_max_output_tokens(row, budget_config)
    return {
        "total_rows": len(rows),
        "condition_counts": dict(Counter(row.get("condition_type") for row in rows)),
        "source_counts": dict(Counter(row.get("source_dataset") for row in rows)),
        "analysis_event_type_counts": dict(Counter(row.get("analysis_event_type") for row in rows)),
        "fault_operator_counts": dict(Counter(row.get("fault_operator") for row in rows if row.get("fault_operator"))),
        "source_event_counts": grouped(("source_dataset", "analysis_event_type")),
        "source_fault_counts": grouped(("source_dataset", "fault_operator"), include_none=False),
        "resolved_token_budget_class_counts": dict(sorted(budget_classes.items())),
        "resolved_max_output_tokens": dict(sorted(budget_values.items())),
    }


def _provider_factory(model: str, *, temperature: int = 0, max_tokens: int | None = None) -> OpenAIDownstreamProvider:
    return OpenAIDownstreamProvider(
        model=model,
        temperature=temperature,
        reasoning_effort="none",
        max_tokens=max_tokens,
    )


def select_stratified_validation_ids(rows: list[dict[str, Any]], *, limit: int = 60) -> list[str]:
    """Return deterministic validation IDs spanning source/task/condition strata.

    This helper is intentionally selection-only.  It never consults model
    outputs and does not alter the full-run default, which remains manifest
    order.  Callers can use the returned IDs to prepare a separate validation
    manifest/result directory.
    """
    if limit <= 0:
        raise ValueError("validation limit must be positive")
    ordered = sorted(rows, key=lambda row: row["condition_id"])
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in ordered:
        subtype = memora_task_subtype(row)
        key = (str(row.get("source_dataset", "")), str(subtype), str(row.get("condition_type", "")))
        strata.setdefault(key, []).append(row)
    chosen: list[dict[str, Any]] = []
    # Round-robin strata gives each available source/task/condition a chance.
    while len(chosen) < min(limit, len(ordered)):
        progressed = False
        for key in sorted(strata):
            if strata[key]:
                chosen.append(strata[key].pop(0))
                progressed = True
                if len(chosen) == limit:
                    break
        if not progressed:
            break
    return [row["condition_id"] for row in chosen]


def select_long_output_validation_ids(rows: list[dict[str, Any]], *, base_limit: int = 30, seed: int = 42) -> list[str]:
    """Select 30 project-proposal bases and all three conditions per base.

    Selection is entirely pre-inference: candidates are grouped by base, then
    ranked by a stable risk key using prompt length, persona/source metadata,
    and a seed-bound SHA256 tie-break.  Every selected base must contribute
    clean, corrupted, and enforced rows, yielding exactly 90 conditions.
    """
    if base_limit != 30:
        raise ValueError("long-output validation requires exactly 30 bases")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        subtype = memora_task_subtype(row)
        if row.get("source_dataset") != "memora" or subtype != "content_project_proposal":
            continue
        groups.setdefault(row["base_id"], []).append(row)
    candidates = []
    for base_id, group in groups.items():
        if {row.get("condition_type") for row in group} != {"clean", "corrupted", "enforced"}:
            continue
        metadata = group[0].get("task_source_metadata") or group[0].get("source_metadata") or {}
        persona = str(metadata.get("persona", "")) if isinstance(metadata, dict) else ""
        max_prompt_length = max(len(render_prompt(row)) for row in group)
        tie = hashlib.sha256(f"{seed}:{base_id}".encode()).hexdigest()
        candidates.append((persona, -max_prompt_length, tie, base_id))
    if len(candidates) < base_limit:
        raise ValueError(f"Only {len(candidates)} complete Memora project-proposal bases available; need {base_limit}")
    # Round-robin personas first, then consume the longest remaining prompts.
    by_persona: dict[str, list[tuple[str, int, str, str]]] = {}
    for item in sorted(candidates):
        by_persona.setdefault(item[0], []).append(item)
    chosen: list[tuple[str, int, str, str]] = []
    while len(chosen) < base_limit:
        progressed = False
        for persona in sorted(by_persona):
            if by_persona[persona]:
                chosen.append(by_persona[persona].pop(0))
                progressed = True
                if len(chosen) == base_limit:
                    break
        if not progressed:
            break
    chosen_ids = {item[3] for item in chosen}
    remaining = sorted((item for item in candidates if item[3] not in chosen_ids), key=lambda item: (item[1], item[2], item[3]))
    chosen.extend(remaining[: max(0, base_limit - len(chosen))])
    selected_bases = {item[3] for item in chosen}
    selected = []
    # A base can support several fault operators.  Stage B intentionally
    # chooses one deterministic pair per base so its 30/30/30 design remains
    # exactly 90 rows while preserving the frozen pair semantics.
    groups_by_base = {base_id: group for base_id, group in groups.items()}
    for index, base_id in enumerate(sorted(selected_bases)):
        group = groups_by_base[base_id]
        selected.append(next(row for row in group if row.get("condition_type") == "clean"))
        faults = sorted({row.get("fault_operator") for row in group if row.get("condition_type") == "corrupted" and row.get("fault_operator")})
        fault = faults[index % len(faults)]
        selected.extend(row for row in group if row.get("fault_operator") == fault and row.get("condition_type") in {"corrupted", "enforced"})
    selected.sort(key=lambda row: row["condition_id"])
    if len(selected) != 90:
        raise ValueError(f"Long-output validation selected {len(selected)} rows instead of 90")
    return [row["condition_id"] for row in selected]


def _batch_result_row(row: dict[str, Any], prompt: str, provider: OpenAIDownstreamProvider, line: dict[str, Any], manifest_sha: str, budget: int) -> dict[str, Any]:
    """Convert one Batch output line to the sync-compatible raw result shape."""
    condition_id = row["condition_id"]
    result = {
        "condition_id": condition_id, "base_id": row["base_id"], "pair_id": row.get("pair_id"),
        "fault_operator": row.get("fault_operator"), "condition_type": row["condition_type"],
        "source_dataset": row["source_dataset"], "analysis_event_type": row["analysis_event_type"],
        "task_question": row.get("task_question"), "task_oracle_answer": row.get("task_oracle_answer"),
        "provider": "openai", "model": provider.model,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "model_visible_context_sha256": hashlib.sha256(_context_text(row).encode()).hexdigest(),
        "manifest_sha256": manifest_sha, "oracle_answer": row.get("task_oracle_answer"),
        "model_parameters": {"temperature": provider.requested_temperature, "max_output_tokens": budget, "thinking": "disabled", "tools": None},
        "resolved_max_output_tokens": budget, "attempt_count": 1, "retry_count": 0,
        "attempt_errors": [], "recovered_by_retry": False, "truncated_output": False,
        "raw_response": None, "raw_output_text": None, "downstream_response": None,
        "parsed_answer": None, "parse_success": False, "correctness": None,
        "input_tokens": None, "output_tokens": None, "total_tokens": None, "cached_input_tokens": None,
        "cache_hit_tokens": None, "latency_ms": None, "error_type": None, "error_message": None,
        "status": "error",
    }
    error = line.get("error")
    if error:
        result["error_type"] = str(error.get("code") or "BatchError") if isinstance(error, dict) else "BatchError"
        result["error_message"] = str(error.get("message") or error) if isinstance(error, dict) else str(error)
        result["attempt_errors"] = [{"attempt": 1, "error_type": result["error_type"], "error_message": result["error_message"]}]
        return result
    response_envelope = line.get("response") or {}
    body = response_envelope.get("body") if isinstance(response_envelope, dict) else None
    try:
        parsed_response = provider.parse_raw_response(body or {})
        answer = _parse_answer(parsed_response)
        telemetry = parsed_response.telemetry
        result.update({"raw_response": _safe_raw(parsed_response), "raw_output_text": provider._output_text(body or {}), "response_status": provider._field(body or {}, "status"), "downstream_response": answer, "parsed_answer": answer, "parse_success": True, "status": "success", "input_tokens": telemetry.input_tokens, "output_tokens": telemetry.output_tokens, "total_tokens": telemetry.total_tokens, "cached_input_tokens": telemetry.cached_input_tokens, "cache_hit_tokens": telemetry.cached_input_tokens, "request_id": telemetry.request_id, "latency_ms": telemetry.latency_ms, **score_task_answer_detailed(row, answer)})
    except Exception as exc:
        details = getattr(exc, "details", {})
        result["error_type"] = str(details.get("provider_exception") if isinstance(details, dict) else type(exc).__name__)
        result["error_message"] = str(exc)
        if isinstance(details, dict):
            for key in ("raw_output_text", "response_status", "incomplete_details", "input_tokens", "output_tokens", "total_tokens"):
                if key in details:
                    result[key] = details[key]
            if details.get("error_category") == "token_limit_truncation":
                result["truncated_output"] = True
        result["attempt_errors"] = [{"attempt": 1, "error_type": result["error_type"], "error_message": result["error_message"], **(details if isinstance(details, dict) else {})}]
    return result


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sorted(rows, key=lambda item: item["condition_id"])), encoding="utf-8")


def _add_cost_estimate(summary: dict[str, Any], rows: list[dict[str, Any]], *, budget_config: dict[str, Any] | None, input_price: float | None, cached_input_price: float | None, output_price: float | None, batch_discount: float) -> None:
    input_tokens = sum(max(1, (len(render_prompt(row).encode("utf-8")) + 3) // 4) for row in rows)
    output_tokens = sum(resolve_max_output_tokens(row, budget_config) for row in rows)
    summary["cost_estimate"] = batch_cost_estimate(input_tokens=input_tokens, max_output_tokens=output_tokens, input_price_per_million=input_price, cached_input_price_per_million=cached_input_price, output_price_per_million=output_price, batch_discount=batch_discount)


def _run_batch(*, rows: list[dict[str, Any]], results_dir: Path, model: str, manifest_sha: str, budget_config: dict[str, Any] | None, force: bool, resume: bool, provider_factory: Any, dry_run: bool, input_price: float | None, cached_input_price: float | None, output_price: float | None, batch_discount: float) -> dict[str, Any]:
    prompts = {row["condition_id"]: render_prompt(row) for row in rows}
    budgets = {row["condition_id"]: resolve_max_output_tokens(row, budget_config) for row in rows}
    # A provider instance is only needed to build the provider-identical body.
    provider = (provider_factory or _provider_factory)(model, temperature=0, max_tokens=256)
    # A dry-run test double may only record construction and return None; use
    # the real provider object solely for deterministic request serialization.
    if provider is None:
        provider = OpenAIDownstreamProvider(model=model, temperature=0, reasoning_effort="none", max_tokens=256)
    requests, reverse = build_batch_requests(rows, provider, prompts, budgets)
    results_dir.mkdir(parents=True, exist_ok=True)
    input_path = results_dir / "batch_input.jsonl"
    write_batch_input(requests, input_path)
    if dry_run:
        estimate = batch_cost_estimate(input_tokens=sum(max(1, (len(prompts[cid].encode("utf-8")) + 3) // 4) for cid in prompts), max_output_tokens=sum(budgets.values()), input_price_per_million=input_price, cached_input_price_per_million=cached_input_price, output_price_per_million=output_price, batch_discount=batch_discount)
        summary = {"status": "FROZEN FULL / DRY-RUN", "execution_mode": "batch", "batch_id": None, "batch_status": None, "submitted_rows": 0, "requested_rows": len(rows), "completed_rows": 0, "failed_rows": 0, "missing_rows": 0, "duplicate_custom_ids": 0, "total_rows": len(rows), "api_calls_made": 0, "manifest_sha256": manifest_sha, "model": model, "cost_estimate": estimate}
        return {"rows": [{"condition_id": row["condition_id"], "custom_id": batch_custom_id(row["condition_id"]), "prompt_sha256": hashlib.sha256(prompts[row["condition_id"]].encode()).hexdigest(), "resolved_max_output_tokens": budgets[row["condition_id"]], "model": model, "provider": "openai"} for row in rows], "summary": summary}
    output = results_dir / "full_results.jsonl"
    if output.exists() and not (force or resume):
        raise FileExistsError(f"Results exist; pass --resume or --force: {output}")
    cached_rows: dict[str, dict[str, Any]] = {}
    if resume and output.exists() and not force:
        for line in output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            cid = item.get("condition_id")
            if not cid or cid in cached_rows:
                raise ValueError("Malformed or duplicate cached Batch condition result")
            if item.get("manifest_sha256") != manifest_sha or item.get("model") != model:
                raise ValueError(f"Cached Batch result identity mismatch for {cid}")
            cached_rows[cid] = item
    pending = [row for row in rows if row["condition_id"] not in cached_rows or cached_rows[row["condition_id"]].get("status") != "success"]
    if not pending:
        ordered_cached = [cached_rows[row["condition_id"]] for row in rows]
        return {"rows": ordered_cached, "summary": {"status": "RESUMED", "execution_mode": "batch", "batch_id": None, "batch_status": "completed", "submitted_rows": 0, "completed_rows": len(ordered_cached), "failed_rows": 0, "missing_rows": 0, "duplicate_custom_ids": 0, "total_rows": len(rows), "manifest_sha256": manifest_sha, "model": model}}
    if pending != rows:
        prompts = {row["condition_id"]: render_prompt(row) for row in pending}
        budgets = {row["condition_id"]: resolve_max_output_tokens(row, budget_config) for row in pending}
        requests, reverse = build_batch_requests(pending, provider, prompts, budgets)
        write_batch_input(requests, input_path)
    client = provider._client()
    uploaded = client.files.create(file=input_path.open("rb"), purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/responses", completion_window="24h")
    while getattr(batch, "status", None) not in {"completed", "failed", "expired", "cancelled"}:
        time.sleep(30)
        batch = client.batches.retrieve(batch.id)
    if getattr(batch, "status", None) != "completed":
        raise RuntimeError(f"OpenAI Batch did not complete: {getattr(batch, 'status', None)}")
    content = client.files.content(batch.output_file_id)
    lines = read_batch_lines(content)
    mapped = validate_batch_output(lines, reverse)
    if len(mapped) != len(pending):
        raise RuntimeError(f"OpenAI Batch missing {len(pending) - len(mapped)} output rows")
    result_rows = [_batch_result_row(row, prompts[row["condition_id"]], provider, mapped[batch_custom_id(row["condition_id"])], manifest_sha, budgets[row["condition_id"]]) for row in pending]
    result_rows = [cached_rows[row["condition_id"]] for row in rows if row["condition_id"] in cached_rows] + result_rows
    result_rows.sort(key=lambda item: item["condition_id"])
    _write_rows(result_rows, output)
    if callable(getattr(provider, "close", None)):
        provider.close()
    summary = {"status": "COMPLETED", "execution_mode": "batch", "batch_id": batch.id, "batch_status": batch.status, "submitted_rows": len(pending), "completed_rows": sum(row.get("status") == "success" for row in result_rows), "failed_rows": sum(row.get("status") == "error" for row in result_rows), "missing_rows": 0, "duplicate_custom_ids": 0, "total_rows": len(rows), "manifest_sha256": manifest_sha, "model": model}
    return {"rows": result_rows, "summary": summary}


def _run_batch_explicit(*, rows: list[dict[str, Any]], results_dir: Path, model: str, manifest_sha: str, budget_config: dict[str, Any] | None, provider_factory: Any, action: str, batch_id: str | None, force_new_batch: bool, file_poll_seconds: float, file_timeout_seconds: float) -> dict[str, Any]:
    prompts = {row["condition_id"]: render_prompt(row) for row in rows}
    budgets = {row["condition_id"]: resolve_max_output_tokens(row, budget_config) for row in rows}
    provider = provider_factory(model, temperature=0, max_tokens=256)
    if provider is None:
        provider = OpenAIDownstreamProvider(model=model, temperature=0, reasoning_effort="none", max_tokens=256)
    try:
        requests, reverse = build_batch_requests(rows, provider, prompts, budgets)
        metadata, lines = execute_batch_lifecycle(action=action, provider=provider, requests=requests, reverse=reverse, results_dir=results_dir, model=model, manifest_sha=manifest_sha, force_new_batch=force_new_batch, batch_id=batch_id, file_poll_seconds=file_poll_seconds, file_timeout_seconds=file_timeout_seconds)
        summary = {"status": "SUBMITTED" if action == "submit" else "STATUS", "execution_mode": "batch", "batch_action": action, "batch_id": metadata.get("batch_id"), "batch_status": metadata.get("batch_status"), "submitted_rows": len(requests) if action == "submit" else metadata.get("submitted_request_count", 0), "completed_rows": 0, "failed_rows": 0, "missing_rows": 0, "duplicate_custom_ids": 0, "total_rows": len(rows), "manifest_sha256": manifest_sha, "model": model}
        result_rows: list[dict[str, Any]] = []
        if action == "retrieve":
            mapped = validate_batch_output(lines or [], reverse)
            if len(mapped) != len(rows):
                raise RuntimeError(f"Batch missing {len(rows) - len(mapped)} output rows")
            result_rows = [_batch_result_row(row, prompts[row["condition_id"]], provider, mapped[batch_custom_id(row["condition_id"])], manifest_sha, budgets[row["condition_id"]]) for row in rows]
            output = results_dir / "full_results.jsonl"
            if output.exists() and not force_new_batch:
                raise FileExistsError(f"Results exist; use a new results directory or --force-new-batch: {output}")
            _write_rows(result_rows, output)
            summary.update({"status": "COMPLETED", "completed_rows": sum(item.get("status") == "success" for item in result_rows), "failed_rows": sum(item.get("status") == "error" for item in result_rows)})
        return {"rows": result_rows, "summary": summary}
    finally:
        if callable(getattr(provider, "close", None)):
            provider.close()


def run(
    *,
    manifest: Path = DEFAULT_MANIFEST,
    results_dir: Path = DEFAULT_RESULTS,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    limit: int | None = None,
    force: bool = False,
    resume: bool = False,
    provider_factory: Any = None,
    workers: int = 1,
    stratified_validation: bool = False,
    long_output_validation: bool = False,
    selected_condition_ids: set[str] | None = None,
    proposal_only: bool = False,
    budget_config_path: Path | None = None,
    execution_mode: str = "sync",
    input_price_per_million: float | None = None,
    cached_input_price_per_million: float | None = None,
    output_price_per_million: float | None = None,
    batch_discount: float = 0.5,
    batch_action: str = "submit",
    batch_id: str | None = None,
    force_new_batch: bool = False,
    file_poll_seconds: float = 5.0,
    file_timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    if execution_mode not in {"sync", "batch"}:
        raise ValueError("--execution-mode must be sync or batch")
    rows = _read(manifest)
    if selected_condition_ids is not None and (stratified_validation or long_output_validation or proposal_only):
        raise ValueError("targeted condition selection cannot be combined with a fixed validation subset")
    if selected_condition_ids is not None and limit is not None:
        raise ValueError("targeted condition selection cannot be combined with --limit")
    available_ids = {row["condition_id"] for row in rows}
    if selected_condition_ids is not None:
        missing = sorted(set(selected_condition_ids) - available_ids)
        if missing:
            raise ValueError(f"Requested condition_id not found in frozen manifest: {', '.join(missing)}")
    if sum(bool(value) for value in (stratified_validation, long_output_validation, proposal_only)) > 1:
        raise ValueError("choose only one validation subset")
    budget_config = load_budget_config(budget_config_path) if budget_config_path else None
    selected_ids = set(selected_condition_ids) if selected_condition_ids is not None else None
    if stratified_validation:
        selected_ids = set(select_stratified_validation_ids(rows, limit=60))
    elif long_output_validation:
        selected_ids = set(select_long_output_validation_ids(rows))
    elif proposal_only:
        selected_ids = project_proposal_condition_ids(rows)
    selected_rows = [row for row in rows if selected_ids is None or row["condition_id"] in selected_ids]
    selected_rows = selected_rows if limit is None else selected_rows[:limit]
    if selected_ids is not None:
        assert_selective_identity(provider="openai", model=model, model_parameters={"temperature": 0, "thinking": "disabled", "tools": None}, manifest=manifest, selected_rows=selected_rows, expected_ids=selected_ids)
    if execution_mode == "batch":
        preflight = run_shared(
            manifest=manifest, results_dir=results_dir, model=model, dry_run=True,
            limit=limit, force=False, resume=False, provider_factory=None,
            provider_name=PROVIDER, expected_sha256=ACCEPTED_MANIFEST_SHAS,
            expected_counts=EXPECTED_COUNTS, expected_base_count=EXPECTED_BASE_COUNT,
            output_filename="full_results.jsonl", strict_scoring=True, workers=1,
            selected_condition_ids=selected_ids, budget_config=budget_config,
        )
        if dry_run:
            batch_result = _run_batch(
                rows=selected_rows, results_dir=results_dir, model=model,
                manifest_sha=sha256_file(manifest), budget_config=budget_config,
                force=force, resume=resume, provider_factory=provider_factory or _provider_factory,
                dry_run=True, input_price=input_price_per_million,
                cached_input_price=cached_input_price_per_million,
                output_price=output_price_per_million, batch_discount=batch_discount,
            )
        else:
            batch_result = _run_batch_explicit(
                rows=selected_rows, results_dir=results_dir, model=model,
                manifest_sha=sha256_file(manifest), budget_config=budget_config,
                provider_factory=provider_factory or _provider_factory, action=batch_action,
                batch_id=batch_id, force_new_batch=force_new_batch,
                file_poll_seconds=file_poll_seconds, file_timeout_seconds=file_timeout_seconds,
            )
        batch_result["summary"]["scoring_preflight"] = preflight["summary"].get("scoring_preflight")
        batch_result["summary"].update(_summary_fields(selected_rows, budget_config))
        _add_cost_estimate(batch_result["summary"], selected_rows, budget_config=budget_config, input_price=input_price_per_million, cached_input_price=cached_input_price_per_million, output_price=output_price_per_million, batch_discount=batch_discount)
        batch_result["summary"]["provider"] = PROVIDER
        batch_result["summary"]["execution_mode"] = "batch"
        (results_dir / ("full_dry_run_summary.json" if dry_run else "run_summary.json")).write_text(json.dumps(batch_result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return batch_result
    result = run_shared(
        manifest=manifest,
        results_dir=results_dir,
        model=model,
        dry_run=dry_run,
        limit=limit,
        force=force,
        resume=resume,
        provider_factory=provider_factory or _provider_factory,
        provider_name=PROVIDER,
        expected_sha256=ACCEPTED_MANIFEST_SHAS,
        expected_counts=EXPECTED_COUNTS,
        expected_base_count=EXPECTED_BASE_COUNT,
        output_filename="full_results.jsonl",
        strict_scoring=True,
        workers=workers,
        selected_condition_ids=selected_ids,
        budget_config=budget_config,
    )
    result["summary"].update(_summary_fields(selected_rows, budget_config))
    result["summary"]["provider"] = PROVIDER
    result["summary"]["model"] = model
    result["summary"]["reasoning_effort"] = "none"
    _add_cost_estimate(result["summary"], selected_rows, budget_config=budget_config, input_price=input_price_per_million, cached_input_price=cached_input_price_per_million, output_price=output_price_per_million, batch_discount=batch_discount)
    results_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        result["summary"]["status"] = "FROZEN FULL / DRY-RUN"
        result["summary"]["validation_errors"] = len(result["summary"]["scoring_preflight"]["unresolved"])
        (results_dir / "full_dry_run_requests.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in result["rows"]),
            encoding="utf-8",
        )
        (results_dir / "full_dry_run_summary.json").write_text(
            json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        atomic_write_json(results_dir / "run_summary.json", result["summary"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stratified-validation", action="store_true", help="run the deterministic 60-row source/task/condition validation subset")
    parser.add_argument("--long-output-validation", action="store_true", help="run the deterministic 90-row Memora project-proposal validation subset")
    parser.add_argument("--proposal-only", action="store_true", help="select all authoritative frozen Memora content_project_proposal conditions")
    parser.add_argument("--token-budget-config", type=Path, help="versioned provider-neutral token-budget configuration")
    parser.add_argument("--execution-mode", choices=("sync", "batch"), default="sync")
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--cached-input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--batch-discount", type=float, default=0.5)
    parser.add_argument("--batch-action", choices=("submit", "status", "retrieve"), default="submit")
    parser.add_argument("--batch-id")
    parser.add_argument("--force-new-batch", action="store_true")
    parser.add_argument("--file-poll-seconds", type=float, default=5.0)
    parser.add_argument("--file-timeout-seconds", type=float, default=300.0)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--condition-id", help="run exactly one frozen condition by condition_id")
    target_group.add_argument("--condition-ids-file", type=Path, help="run frozen condition IDs listed one per non-empty line")
    args = parser.parse_args(argv)
    targeted_ids: set[str] | None = None
    if args.condition_id or args.condition_ids_file:
        if args.stratified_validation or args.long_output_validation or args.proposal_only:
            parser.error("targeted condition selection cannot be combined with --stratified-validation or --long-output-validation")
        if args.limit is not None:
            parser.error("targeted condition selection cannot be combined with --limit")
        if args.condition_id:
            targeted_ids = {args.condition_id}
        else:
            try:
                requested = [line.strip() for line in args.condition_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            except OSError as exc:
                parser.error(f"cannot read --condition-ids-file: {exc}")
            if len(requested) != len(set(requested)):
                parser.error("--condition-ids-file contains duplicate condition IDs")
            targeted_ids = set(requested)
            if not targeted_ids:
                parser.error("--condition-ids-file contains no non-empty condition IDs")
    result = run(
        manifest=args.manifest,
        results_dir=args.results_dir,
        model=args.model,
        dry_run=args.dry_run,
        limit=args.limit,
        force=args.force,
        resume=args.resume,
        workers=args.workers,
        stratified_validation=args.stratified_validation,
        long_output_validation=args.long_output_validation,
        selected_condition_ids=targeted_ids,
        proposal_only=args.proposal_only,
        budget_config_path=args.token_budget_config,
        execution_mode=args.execution_mode,
        input_price_per_million=args.input_price_per_million,
        cached_input_price_per_million=args.cached_input_price_per_million,
        output_price_per_million=args.output_price_per_million,
        batch_discount=args.batch_discount,
        batch_action=args.batch_action,
        batch_id=args.batch_id,
        force_new_batch=args.force_new_batch,
        file_poll_seconds=args.file_poll_seconds,
        file_timeout_seconds=args.file_timeout_seconds,
    )
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
