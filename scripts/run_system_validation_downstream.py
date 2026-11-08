#!/usr/bin/env python3
"""Run the fixed downstream answer stage for frozen system-validation runs.

This script consumes completed formal retrieval outputs only.  It never
re-enters a memory system and performs no scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.providers.anthropic_provider import (  # noqa: E402
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicMessagesProvider,
)
from eval.providers.deepseek_provider import (  # noqa: E402
    DEFAULT_DEEPSEEK_MODEL,
    DeepSeekDownstreamProvider,
)
from eval.providers.openai_provider import DEFAULT_OPENAI_MODEL, OpenAIDownstreamProvider  # noqa: E402
from eval.reconstruction_provider import ProviderResponse, ProviderTransportError  # noqa: E402


MANIFEST_SHA256 = "95abb36509873cbfec811f477291e2887def4b9a6ec2b04e766d27102ee09588"
FORMAL_STATUS = "FROZEN / FORMAL"
PROMPT_TEMPLATE = """Task/query:
{query}

Model-visible memory/context:
{context}

Answer the task using only the task and model-visible context. If the context does not support the requested information, state that it cannot be determined from the available context. Do not fabricate missing facts.
Return valid JSON with one string field named answer."""
_PROMPT_METADATA_TERMS = (
    "condition",
    "fault_type",
    "memory_system",
    "expected_governance_action",
    "governance gold",
    "frozen / formal",
)


def build_downstream_prompt(query: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(query=query, context=context)


def validate_prompt_no_metadata_leakage(prompt: str, query: str, context: str) -> None:
    """Reject metadata in the fixed prompt portion, while allowing it in inputs."""
    if prompt != build_downstream_prompt(query, context):
        raise ValueError("downstream prompt does not match the fixed prompt template")
    prefix, remainder = PROMPT_TEMPLATE.split("{query}", 1)
    middle, suffix = remainder.split("{context}", 1)
    fixed_text = (prefix + middle + suffix).lower()
    leaked = [term for term in _PROMPT_METADATA_TERMS if term in fixed_text]
    if leaked:
        raise ValueError(f"downstream prompt template leaks experiment metadata: {leaked}")


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _condition_metadata(record: dict[str, Any]) -> tuple[Any, Any]:
    """Return the original experimental condition and fault type.

    Formal system rows may use ``condition=fault`` while retaining the
    distinguishing experimental label in ``fault_type``.
    """
    raw_condition = record.get("condition")
    raw_fault_type = record.get("fault_type")
    if raw_condition == "clean":
        return "clean", None
    condition = raw_fault_type if raw_condition == "fault" and raw_fault_type else raw_condition
    fault_type = raw_fault_type if condition != "clean" and raw_fault_type else condition
    return condition, fault_type


def _cache_key(record: dict[str, Any], model: str, prompt_hash: str) -> str:
    # Formal system rows use condition="fault" for both fault siblings; the
    # fault type is therefore part of the row identity as well.
    condition, fault_type = _condition_metadata(record)
    identity = [
        record["memory_system"],
        record["base_example_id"],
        condition,
        fault_type,
        model,
        prompt_hash,
    ]
    encoded = json.dumps(identity, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _answer(response: ProviderResponse) -> str:
    parsed = response.parsed
    if hasattr(parsed, "model_dump"):
        parsed = parsed.model_dump()
    elif hasattr(parsed, "dict"):
        parsed = parsed.dict()
    if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str):
        raise ValueError("downstream provider returned no string answer")
    return parsed["answer"]


def _error(exc: Exception) -> dict[str, Any]:
    details = dict(getattr(exc, "details", {}) or {})
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "provider": getattr(exc, "provider", None),
        "model": getattr(exc, "model", None),
        "request_id": getattr(exc, "request_id", None) or details.get("request_id"),
        "details": details,
    }


def validate_formal_system_result(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("system result must be a JSON object")
    summary = payload.get("summary")
    records = payload.get("records")
    if not isinstance(summary, dict) or not isinstance(records, list):
        raise ValueError("system result must contain summary and records")
    expected = {
        "status": FORMAL_STATUS,
        "cohort_version": "v1",
        "manifest_sha256": MANIFEST_SHA256,
        "selected_base_examples": 60,
        "condition_rows": 180,
        "native_retrieval_error_n": 0,
        "native_retrieval_empty_n": 0,
        "invariant_failure_n": 0,
        "requested_retrieval_k": 1,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"formal system result failed validation: {key}={summary.get(key)!r}, expected {value!r}")
    if len(records) != 180:
        raise ValueError(f"formal system result must contain 180 records, got {len(records)}")
    required = (
        "base_example_id", "source_dataset", "governance_dimension", "condition",
        "fault_type", "memory_system", "manifest_sha256", "requested_retrieval_k",
        "query", "final_model_visible_context",
    )
    system = summary.get("memory_system")
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        if not isinstance(record, dict) or any(field not in record for field in required):
            raise ValueError("formal system result contains a malformed record")
        if record["experiment_status"] != FORMAL_STATUS or record["cohort_version"] != "v1":
            raise ValueError("formal record provenance is invalid")
        if record["manifest_sha256"] != MANIFEST_SHA256 or record["requested_retrieval_k"] != 1:
            raise ValueError("formal record manifest or retrieval provenance is invalid")
        if system is not None and record["memory_system"] != system:
            raise ValueError("formal record memory-system provenance is inconsistent")
        key = (
            str(record["base_example_id"]),
            str(record["condition"]),
            str(record["fault_type"]),
        )
        if key in seen:
            raise ValueError(f"duplicate formal condition record: {key}")
        seen.add(key)
    if len({str(record["base_example_id"]) for record in records}) != 60:
        raise ValueError("formal system result must contain 60 unique base examples")
    return records


def _provider(provider: str, model: str | None) -> Any:
    if provider == "openai":
        return OpenAIDownstreamProvider(model=model or DEFAULT_OPENAI_MODEL, temperature=0, reasoning_effort="none")
    if provider == "deepseek":
        return DeepSeekDownstreamProvider(model=model or DEFAULT_DEEPSEEK_MODEL, temperature=0)
    if provider == "anthropic":
        return AnthropicMessagesProvider(model=model or DEFAULT_ANTHROPIC_MODEL, temperature=0)
    raise ValueError("--provider must be one of: openai, deepseek, anthropic")


def _resolved_model(provider: str, model: str | None) -> str:
    resolved = model or {
        "openai": DEFAULT_OPENAI_MODEL,
        "deepseek": DEFAULT_DEEPSEEK_MODEL,
        "anthropic": DEFAULT_ANTHROPIC_MODEL,
    }.get(provider)
    if resolved is None:
        raise ValueError("--provider must be one of: openai, deepseek, anthropic")
    return resolved


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed downstream cache JSON on line {line_number}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("cache_key"), str):
                raise ValueError(f"Malformed downstream cache record on line {line_number}")
            key = record["cache_key"]
            if key in cached:
                raise ValueError(f"Duplicate downstream cache key: {key}")
            cached[key] = record
    return cached


def _write_cache(path: Path, ordered: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_downstream(
    *,
    system_result: str | Path,
    provider: str,
    model: str | None,
    results_dir: str | Path,
    provider_factory: Callable[[str, str], Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    # Read and validate everything before constructing a provider, which is
    # the point at which a real SDK client could otherwise be initialized.
    payload = json.loads(Path(system_result).read_text(encoding="utf-8"))
    records = validate_formal_system_result(payload)
    resolved_model = _resolved_model(provider, model)
    output = Path(results_dir)
    output_jsonl = output / "system_validation_downstream.jsonl"
    output_summary = output / "system_validation_downstream_summary.json"
    if dry_run:
        dry_rows: list[dict[str, Any]] = []
        prompt_hashes: set[str] = set()
        template_hash = prompt_sha256(PROMPT_TEMPLATE)
        for source in records:
            condition, fault_type = _condition_metadata(source)
            query = str(source["query"])
            context = str(source["final_model_visible_context"])
            prompt = build_downstream_prompt(query, context)
            validate_prompt_no_metadata_leakage(prompt, query, context)
            phash = prompt_sha256(prompt)
            prompt_hashes.add(phash)
            dry_rows.append({
                "base_example_id": source["base_example_id"],
                "condition": condition,
                "fault_type": fault_type,
                "source_dataset": source["source_dataset"],
                "governance_dimension": source["governance_dimension"],
                "memory_system": source["memory_system"],
                "downstream_provider": provider,
                "downstream_model": resolved_model,
                "manifest_sha256": MANIFEST_SHA256,
                "requested_retrieval_k": 1,
                "query": query,
                "final_model_visible_context": context,
                "prompt_sha256": phash,
                "prompt_text": prompt,
                "cache_key": _cache_key(source, resolved_model, phash),
            })
        dry_result = {
            "summary": {
                "status": "FROZEN / FORMAL / DRY-RUN",
                "memory_system": payload["summary"].get("memory_system"),
                "downstream_provider": provider,
                "downstream_model": resolved_model,
                "condition_rows": len(dry_rows),
                "unique_base_examples": len({row["base_example_id"] for row in dry_rows}),
                "manifest_sha256": MANIFEST_SHA256,
                "prompt_template_sha256": template_hash,
                "unique_prompt_sha256_count": len(prompt_hashes),
                "api_calls_made": 0,
                "validation_errors": 0,
                "scoring_performed": False,
            },
            "records": dry_rows,
        }
        _write_json(output / "system_validation_downstream_dry_run.json", dry_result)
        _write_json(output / "system_validation_downstream_dry_run_summary.json", dry_result["summary"])
        return dry_result

    cached = _read_cache(output_jsonl)
    expected_keys: set[str] = set()
    prepared: list[tuple[dict[str, Any], str, str]] = []
    for source in records:
        query = str(source["query"])
        context = str(source["final_model_visible_context"])
        prompt = build_downstream_prompt(query, context)
        validate_prompt_no_metadata_leakage(prompt, query, context)
        phash = prompt_sha256(prompt)
        target = dict(source)
        target["condition"], target["fault_type"] = _condition_metadata(source)
        target["cache_key"] = _cache_key(source, resolved_model, phash)
        expected_keys.add(target["cache_key"])
        prepared.append((target, prompt, phash))
    extra = set(cached) - expected_keys
    if extra:
        raise ValueError("downstream cache does not match this formal input (unexpected cache keys)")
    for target, _, phash in prepared:
        prior = cached.get(target["cache_key"])
        if prior is not None:
            if (
                prior.get("memory_system") != target["memory_system"]
                or prior.get("base_example_id") != target["base_example_id"]
                or prior.get("condition") != target["condition"]
                or prior.get("fault_type") != target["fault_type"]
                or prior.get("downstream_provider") != provider
                or prior.get("downstream_model") != resolved_model
                or prior.get("prompt_sha256") != phash
                or prior.get("manifest_sha256") != MANIFEST_SHA256
            ):
                raise ValueError("downstream cache configuration does not match this run")

    client = provider_factory(provider, resolved_model) if provider_factory else _provider(provider, resolved_model)
    rows: list[dict[str, Any]] = []
    reused_n = 0
    for target, prompt, phash in prepared:
        key = target["cache_key"]
        prior = cached.get(key)
        if prior is not None:
            if prior.get("prompt_sha256") != phash:
                raise ValueError(f"downstream cache prompt mismatch for {target['base_example_id']}:{target['condition']}")
            rows.append(prior)
            reused_n += 1
            continue
        result = dict(target)
        result.update({
            "experiment_status": FORMAL_STATUS,
            "cohort_version": "v1",
            "manifest_sha256": MANIFEST_SHA256,
            "requested_retrieval_k": 1,
            "downstream_provider": provider,
            "downstream_model": resolved_model,
            "prompt_sha256": phash,
            "downstream_response": None,
            "downstream_error": None,
            "downstream_latency": None,
            "downstream_request_id": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
        })
        started = time.perf_counter()
        try:
            response = client.request_downstream(prompt)
            result["downstream_response"] = _answer(response)
            telemetry = response.telemetry
            result.update({
                "downstream_latency": telemetry.latency_ms,
                "downstream_request_id": telemetry.request_id,
                "input_tokens": telemetry.input_tokens,
                "output_tokens": telemetry.output_tokens,
                "total_tokens": telemetry.total_tokens,
            })
        except Exception as exc:
            result["downstream_error"] = _error(exc)
            # Provider telemetry is unavailable on a failed normalized call;
            # use only measured transport latency when the provider exposed it.
            details = getattr(exc, "details", {}) or {}
            result["downstream_latency"] = details.get("api_latency_ms")
            result["downstream_request_id"] = getattr(exc, "request_id", None) or details.get("request_id")
        rows.append(result)
        # Preserve already cached later rows if a run is interrupted between
        # requests; this keeps resume from causing duplicate calls.
        merged = dict(cached)
        merged.update({row["cache_key"]: row for row in rows})
        _write_cache(output_jsonl, [merged[target["cache_key"]] for target, _, _ in prepared if target["cache_key"] in merged])
    rows.sort(key=lambda row: (str(row["base_example_id"]), str(row["condition"])))
    _write_cache(output_jsonl, rows)
    errors = sum(row.get("downstream_error") is not None for row in rows)
    summary = {
        "status": FORMAL_STATUS,
        "cohort_version": "v1",
        "seed": 42,
        "manifest_sha256": MANIFEST_SHA256,
        "memory_system": payload["summary"].get("memory_system"),
        "downstream_provider": provider,
        "downstream_model": resolved_model,
        "requested_retrieval_k": 1,
        "selected_base_examples": 60,
        "condition_rows": 180,
        "inference_success_n": 180 - errors,
        "inference_error_n": errors,
        "response_reused_n": reused_n,
        "prompt_template_sha256": prompt_sha256(PROMPT_TEMPLATE),
        "scoring_performed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary, "records": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-result", required=True)
    parser.add_argument("--provider", required=True, choices=("openai", "deepseek", "anthropic"))
    parser.add_argument("--model")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate and materialize prompts without making provider calls")
    args = parser.parse_args(argv)
    result = run_downstream(
        system_result=args.system_result,
        provider=args.provider,
        model=args.model,
        results_dir=args.results_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
