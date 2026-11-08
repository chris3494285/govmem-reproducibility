#!/usr/bin/env python3
"""Run downstream inference for the separate governance-state-fault RQ1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from eval.providers.deepseek_provider import DEFAULT_DEEPSEEK_MODEL, DeepSeekDownstreamProvider
from eval.providers.anthropic_provider import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicMessagesProvider,
    anthropic_supports_temperature,
)
from eval.providers.openai_provider import DEFAULT_OPENAI_MODEL, OpenAIDownstreamProvider
from eval.reconstruction_provider import ProviderResponse, ProviderTransportError
from govmem.governance.enforcement import AdmissionResult, decide_admission, select_memory


DEFAULT_MANIFEST = "data/governance_fault_rq1_pilot_frozen.jsonl"
RQ1B_MANIFEST_VERSION = "governance-fault-rq1-restrictive-v1"
RQ1A_MANIFEST_VERSIONS = frozenset({None, "governance-fault-rq1-v2-corrected"})
NEUTRAL_CONTEXT_INSTRUCTION = (
    "If the admitted context does not support the requested information, state that it cannot be determined "
    "from the available context. Do not fabricate missing facts."
)


def _condition_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["example_id"]), []).append(row)
    return groups


def _manifest_cohort(rows: list[dict[str, Any]]) -> str:
    """Return the cohort contract encoded by the manifest rows.

    The version is authoritative when present.  The historical RQ1-A artifact
    predates per-row manifest version metadata, so an all-unversioned manifest
    is the backward-compatible RQ1-A contract.
    """
    versions = {row.get("manifest_version") for row in rows if row.get("manifest_version") is not None}
    if len(versions) > 1:
        raise ValueError(f"Manifest mixes multiple manifest versions: {sorted(versions)}")
    version = next(iter(versions), None)
    if version == RQ1B_MANIFEST_VERSION:
        return "rq1b_restrictive"
    if version in RQ1A_MANIFEST_VERSIONS:
        return "rq1a"
    raise ValueError(f"Unsupported governance fault manifest version: {version!r}")


def validate_manifest_rows(rows: list[dict[str, Any]], *, require_full: bool = True) -> None:
    cohort = _manifest_cohort(rows)
    expected_rows = 60 if cohort == "rq1b_restrictive" else 300
    expected_conditions = 2 if cohort == "rq1b_restrictive" else 3
    expected_faults = 1 if cohort == "rq1b_restrictive" else 2
    if require_full and len(rows) != expected_rows:
        label = "RQ1-B restrictive" if cohort == "rq1b_restrictive" else "RQ1-A"
        raise ValueError(f"Expected {expected_rows} {label} condition rows in the frozen manifest, found {len(rows)}")
    groups = _condition_groups(rows)
    for example_id, group in groups.items():
        if len(group) != expected_conditions:
            raise ValueError(
                f"Example {example_id} does not contain exactly {expected_conditions} conditions "
                f"for {cohort}"
            )
        clean = [row for row in group if row.get("condition_type") == "clean"]
        faults = [row for row in group if row.get("condition_type") == "fault"]
        if len(clean) != 1 or len(faults) != expected_faults:
            raise ValueError(
                f"Example {example_id} must contain one clean and {expected_faults} fault rows for {cohort}"
            )
        reference = clean[0]
        if reference.get("fault_spec") is not None or reference.get("observed_governance_state") != reference.get("gold_governance_state"):
            raise ValueError(f"Clean row is not identical to its gold state: {example_id}")
        for row in group:
            for field in ("task", "memory", "gold_governance_state", "gold_admission_decision", "utility_target"):
                if row.get(field) != reference.get(field):
                    raise ValueError(f"Paired invariant failed for {example_id}: {field}")
            if row.get("condition_type") == "fault" and not row.get("fault_spec"):
                raise ValueError(f"Fault row lacks fault_spec: {row.get('run_id')}")
            if cohort == "rq1b_restrictive" and row.get("condition_type") == "fault":
                direction = (row.get("fault_spec") or {}).get("direction")
                if direction != "restrictive":
                    raise ValueError(
                        f"RQ1-B fault row must have restrictive direction: {row.get('run_id')}"
                    )


def load_manifest(path: str | Path, *, require_full: bool = True) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Governance fault manifest must contain JSON objects")
    validate_manifest_rows(rows, require_full=require_full)
    return rows


def select_base_rows(rows: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    groups = _condition_groups(rows)
    ids = sorted(groups)
    selected_ids = ids if limit is None else ids[:limit]
    if limit is not None and limit < 0:
        raise ValueError("--limit must be non-negative")
    selected: list[dict[str, Any]] = []
    for example_id in selected_ids:
        selected.extend(sorted(groups[example_id], key=lambda row: (row.get("condition_type") != "clean", row.get("run_id", ""))))
    return selected


def construct_context(row: dict[str, Any], admission: AdmissionResult) -> tuple[str, list[str], list[str]]:
    selection = select_memory(row, admission)
    if selection.selected_memory:
        admitted = json.dumps(list(selection.selected_memory), ensure_ascii=False, sort_keys=True)
    else:
        admitted = "No protected memory was admitted by the baseline enforcement policy."
    selected, rejected = list(selection.selected_memory_ids), list(selection.rejected_memory_ids)
    prompt = (
        "Task/query:\n" + str(row.get("task", ""))
        + "\n\nAdmitted memory/context:\n" + admitted
        + "\n\nAnswer the task using only the task and admitted context."
        + "\n" + NEUTRAL_CONTEXT_INSTRUCTION
        + "\nReturn valid JSON with one string field named answer."
    )
    return prompt, selected, rejected


def _error_record(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ProviderTransportError):
        details = dict(exc.details)
        details.setdefault("type", "ProviderTransportError")
        details.setdefault("message", str(exc))
        return details
    return {"type": type(exc).__name__, "message": str(exc)}


def _response_answer(response: ProviderResponse) -> Any:
    parsed = response.parsed
    if isinstance(parsed, dict) and "answer" in parsed:
        return parsed["answer"]
    return parsed


def _result_base(
    row: dict[str, Any], admission: AdmissionResult, prompt: str, selected: list[str], rejected: list[str],
    selection_reason: str, model: str, provider: str, temperature: float,
    reasoning_effort: str, effective_temperature: float | None, effective_reasoning_effort: str | None,
) -> dict[str, Any]:
    fault = row.get("fault_spec") or {}
    return {
        "run_id": row.get("run_id"),
        "example_id": row.get("example_id"),
        "source_dataset": row.get("source_dataset"),
        "dimension": row.get("dimension"),
        "condition_type": row.get("condition_type"),
        "fault_spec": row.get("fault_spec"),
        "fault_direction": fault.get("direction"),
        "gold_governance_state": row.get("gold_governance_state"),
        "observed_governance_state": row.get("observed_governance_state"),
        "gold_admission_decision": row.get("gold_admission_decision"),
        "actual_admission": admission.actual_admission,
        "enforcement_reason": admission.enforcement_reason,
        "fallback_used": admission.fallback_used,
        "selected_memory_ids": selected,
        "rejected_memory_ids": rejected,
        "selection_reason": selection_reason,
        "constructed_context": prompt,
        "model": model,
        "provider": provider,
        "temperature": temperature,
        "requested_temperature": temperature,
        "effective_temperature": effective_temperature,
        "requested_reasoning_effort": reasoning_effort,
        "effective_reasoning_effort": effective_reasoning_effort,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "response_reused": False,
        "reused_from_run_id": None,
        "model_answer": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "latency_ms": None,
        "request_id": None,
        "status": None,
        "error": None,
    }


def run_manifest(
    *,
    manifest: str | Path = DEFAULT_MANIFEST,
    output: str | Path,
    provider: str = "deepseek",
    model: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    temperature: float = 0,
    reasoning_effort: str = "none",
    provider_factory: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    if provider not in {"deepseek", "openai", "anthropic"}:
        raise ValueError(f"Unsupported provider: {provider}")
    if model is None:
        model = (
            DEFAULT_OPENAI_MODEL if provider == "openai"
            else DEFAULT_ANTHROPIC_MODEL if provider == "anthropic"
            else DEFAULT_DEEPSEEK_MODEL
        )
    effective_temperature = (
        temperature if provider == "deepseek" or reasoning_effort == "none"
        else None
    )
    if provider == "anthropic":
        effective_temperature = temperature if anthropic_supports_temperature(model) else None
    requested_reasoning_effort = reasoning_effort if provider == "openai" else None
    effective_reasoning_effort = requested_reasoning_effort
    rows = load_manifest(manifest, require_full=limit is None)
    selected = select_base_rows(rows, limit)
    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {output_path}")
    if dry_run:
        client_provider = None
    elif provider_factory:
        try:
            factory_kwargs = {"temperature": temperature}
            if provider == "openai":
                factory_kwargs["reasoning_effort"] = reasoning_effort
            client_provider = provider_factory(model, **factory_kwargs)
        except TypeError:
            # Backward-compatible injection point for existing test doubles.
            client_provider = provider_factory(model)
    else:
        if provider == "deepseek":
            client_provider = DeepSeekDownstreamProvider(model=model, temperature=temperature)
        elif provider == "openai":
            client_provider = OpenAIDownstreamProvider(
                model=model, temperature=temperature, reasoning_effort=reasoning_effort
            )
        else:
            client_provider = AnthropicMessagesProvider(model=model, temperature=temperature)
    results: list[dict[str, Any]] = []
    cached_responses: dict[tuple[Any, ...], tuple[ProviderResponse, str]] = {}
    cached_example_id: str | None = None
    for row in selected:
        if row.get("example_id") != cached_example_id:
            cached_responses = {}
            cached_example_id = row.get("example_id")
        admission = decide_admission(row)
        prompt, selected_ids, rejected_ids = construct_context(row, admission)
        result = _result_base(
            row, admission, prompt, selected_ids, rejected_ids, select_memory(row, admission).selection_reason,
            model, provider, temperature, requested_reasoning_effort, effective_temperature, effective_reasoning_effort,
        )
        if dry_run:
            result["status"] = "dry_run"
        else:
            cache_key = (
                prompt, provider, model, requested_reasoning_effort, effective_reasoning_effort,
                temperature, effective_temperature,
            )
            reused = cached_responses.get(cache_key)
            try:
                if reused is not None:
                    response, reused_from = reused
                    result["response_reused"] = True
                    result["reused_from_run_id"] = reused_from
                else:
                    response = client_provider.request_downstream(prompt)
                    cached_responses[cache_key] = (response, str(row.get("run_id")))
                telemetry = response.telemetry
                result.update({
                    "model_answer": _response_answer(response),
                    "input_tokens": telemetry.input_tokens,
                    "output_tokens": telemetry.output_tokens,
                    "total_tokens": telemetry.total_tokens,
                    "latency_ms": telemetry.latency_ms,
                    "request_id": telemetry.request_id,
                    "status": "success",
                })
            except Exception as exc:
                details = _error_record(exc)
                result.update({
                    "input_tokens": details.get("input_tokens"),
                    "output_tokens": details.get("output_tokens"),
                    "total_tokens": details.get("total_tokens"),
                    "latency_ms": details.get("api_latency_ms"),
                    "request_id": details.get("request_id"),
                    "status": "provider_error",
                    "error": details,
                })
        results.append(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=str(output_path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--provider", choices=("deepseek", "openai", "anthropic"), default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--limit", type=int, help="Number of base examples, not condition rows")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Validate and write a zero-provider-call dry-run")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    results = run_manifest(
        manifest=args.manifest, output=args.output, provider=args.provider, model=args.model,
        temperature=args.temperature, reasoning_effort=args.reasoning_effort, limit=args.limit,
        dry_run=args.dry_run or args.preflight, overwrite=args.overwrite,
    )
    print(json.dumps({
        "manifest": args.manifest,
        "selected_conditions": len(results),
        "selected_base_examples": len({row["example_id"] for row in results}),
        "provider": args.provider,
        "model": results[0]["model"] if results else (
            args.model or DEFAULT_OPENAI_MODEL if args.provider == "openai"
            else args.model or DEFAULT_ANTHROPIC_MODEL if args.provider == "anthropic"
            else args.model or DEFAULT_DEEPSEEK_MODEL
        ),
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort if args.provider == "openai" else None,
        "provider_calls": 0 if args.dry_run or args.preflight else sum(not row["response_reused"] for row in results),
        "output": args.output,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
