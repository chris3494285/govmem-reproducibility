from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.openai_reconstruction import (  # noqa: E402
    DEFAULT_MODEL,
    PROMPT_SCHEMA_VERSION,
    OpenAIGovernanceStateReconstructor,
    ReconstructionError,
)
from eval.reconstruction import GovernanceReconstructionInput  # noqa: E402
from eval.reconstruction_applicability import (  # noqa: E402
    applicable_governance_dimensions,
    dataset_native_task_type,
)
from eval.oracle_support import structured_oracle_support  # noqa: E402
from eval.providers.anthropic_provider import (  # noqa: E402
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicMessagesProvider,
)
from eval.providers.deepseek_provider import (  # noqa: E402
    DEFAULT_DEEPSEEK_MODEL,
    DeepSeekChatCompletionsProvider,
)
from scripts.run_reconstruction_smoke import (  # noqa: E402
    ADAPTERS,
    filter_examples_by_query_type,
)


RESERVED_RESULT_FIELDS = frozenset({
    "dataset", "query_id", "query_type", "task_type", "applicable_governance_dimensions",
    "evaluator_only", "skipped", "skipped_reason", "reconstructed_states",
    "reconstruction_status", "validation_status", "condition",
    "governance_applied", "governance_policy_version",
    "provider", "model", "prompt_schema_version", "cache_hit", "usage", "latency_ms",
    "api_latency_ms", "request_size_diagnostics", "reconstruction_policy", "streaming_diagnostics",
    "chunk_count", "interaction_count", "all_interactions_processed", "per_chunk",
    "total_input_tokens", "total_output_tokens", "total_api_latency_ms", "cache_hits", "telemetry", "error",
})


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def _state_record(state) -> dict[str, Any]:
    provenance = json.loads(state.provenance) if state.provenance else {}
    return {
        "interaction_id": state.interaction_id,
        "authorized_scopes": list(state.authorized_scopes),
        "is_deleted": state.is_deleted,
        "valid_from": state.valid_from.isoformat() if state.valid_from else None,
        "valid_until": state.valid_until.isoformat() if state.valid_until else None,
        "entity_key": state.entity_key,
        "confidence": state.confidence,
        "missing_fields": list(state.missing_fields),
        "metadata": _jsonable(dict(state.metadata)),
        "provenance": provenance,
    }


def _evaluator_only(example) -> dict[str, Any]:
    support = structured_oracle_support(example)
    return {
        "model_visible_interaction_ids": [interaction.interaction_id for interaction in example.interactions],
        "oracle_available": bool(support.get("oracle_available", False)),
        "oracle_fields_supported": _jsonable(support.get("oracle_fields_supported", ())),
        "oracle_governance": _jsonable([
            {
                "interaction_id": state.interaction_id,
                "authorized_scopes": list(state.authorized_scopes),
                "is_deleted": state.is_deleted,
                "valid_from": state.valid_from,
                "valid_until": state.valid_until,
                "entity_key": state.entity_key,
                "metadata": dict(state.metadata),
            }
            for state in example.oracle_governance
        ]),
        "oracle_support": _jsonable(support),
        "adapted_metadata": _jsonable(dict(example.metadata)),
        "evaluator_metadata": _jsonable(dict(example.evaluator_metadata)),
    }


def _telemetry_record(client: Any) -> dict[str, Any]:
    result = getattr(client, "last_result", {}) or {}
    telemetry = dict(result.get("telemetry", {}) or {})
    if not telemetry:
        telemetry = dict(getattr(client, "last_telemetry", {}) or {})
    usage = telemetry.get("usage")
    if usage is None and telemetry.get("usage_available"):
        usage = {
            "input_tokens": telemetry.get("input_tokens"),
            "output_tokens": telemetry.get("output_tokens"),
            "total_tokens": telemetry.get("total_tokens"),
        }
    latency_ms = telemetry.get("latency_ms", telemetry.get("api_latency_ms"))
    streaming = telemetry.get("streaming_diagnostics") or {}
    return {
        "provider": telemetry.get("provider"),
        "cache_hit": telemetry.get("cache_hit"),
        "latency_ms": latency_ms,
        "api_latency_ms": telemetry.get("api_latency_ms", latency_ms),
        "api_request_executed": telemetry.get("api_request_executed"),
        "original_api_latency_ms": telemetry.get("original_api_latency_ms"),
        "usage_available": bool(telemetry.get("usage_available", False)),
        "usage": usage,
        "input_tokens": telemetry.get("input_tokens"),
        "output_tokens": telemetry.get("output_tokens"),
        "total_tokens": telemetry.get("total_tokens"),
        "request_size_diagnostics": telemetry.get("request_size_diagnostics"),
        "reconstruction_policy": telemetry.get("reconstruction_policy"),
        "streaming_diagnostics": telemetry.get("streaming_diagnostics"),
        "chunk_count": streaming.get("chunk_count"),
        "interaction_count": streaming.get("interaction_count"),
        "all_interactions_processed": streaming.get("all_interactions_processed"),
        "per_chunk": streaming.get("per_chunk"),
        "total_input_tokens": streaming.get("total_input_tokens"),
        "total_output_tokens": streaming.get("total_output_tokens"),
        "total_api_latency_ms": streaming.get("total_api_latency_ms"),
    }


def _base_record(example, dimensions: tuple[str, ...]) -> dict[str, Any]:
    return {
        "dataset": example.query.metadata.get("dataset") or example.metadata.get("dataset"),
        "query_id": example.query.query_id,
        "query_type": example.query.metadata.get("query_type"),
        "task_type": dataset_native_task_type(example),
        "applicable_governance_dimensions": list(dimensions),
        "evaluator_only": _evaluator_only(example),
    }


def _read_completed_query_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    last_good_offset = 0
    with path.open("rb") as handle:
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                last_good_offset = handle.tell()
                break
            try:
                record = json.loads(line.decode("utf-8"))
                query_id = record.get("query_id")
                if query_id is None:
                    break
                if str(query_id) in completed:
                    raise ValueError(f"Duplicate completed query_id in output: {query_id}")
                completed.add(str(query_id))
                last_good_offset = handle.tell()
            except (UnicodeDecodeError, json.JSONDecodeError):
                # A process interrupted during the final append can leave a
                # partial line. Preserve completed records and remove only the
                # incomplete trailing bytes before resuming.
                with path.open("r+b") as repair:
                    repair.truncate(line_start)
                break
    return completed


def _write_record(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def run_batch(
    *,
    dataset: str,
    input_path: str | Path,
    output_path: str | Path,
    model: str | None = None,
    provider: str = "openai",
    cache_dir: str | Path | None = None,
    limit: int | None = None,
    query_type: str | None = None,
    query_ids: Sequence[str] | None = None,
    run_metadata: dict[str, Any] | None = None,
    condition: str | None = None,
    governance_applied: bool | None = None,
    governance_policy_version: str | None = None,
    overwrite: bool = False,
    reconstructor: Any | None = None,
    max_input_tokens: int | None = None,
    streaming_chunk_tokens: int | None = None,
) -> dict[str, int | str]:
    """Run a resumable reconstruction batch and append one record per example."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    examples = list(ADAPTERS[dataset]().load(input_path))
    total_loaded = len(examples)
    if query_ids is not None:
        requested_ids = [str(query_id) for query_id in query_ids]
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("query_ids contains duplicates")
        by_id = {example.query.query_id: example for example in examples}
        missing_ids = [query_id for query_id in requested_ids if query_id not in by_id]
        if missing_ids:
            raise ValueError(
                f"Manifest query_ids are missing from adapted {dataset} input: {missing_ids[:5]}"
            )
        examples = [by_id[query_id] for query_id in requested_ids]
    if query_type is not None:
        examples = filter_examples_by_query_type(examples, query_type)
        if not examples:
            raise ValueError(f"No {dataset} examples matched --query-type {query_type!r}")
    if limit is not None:
        examples = examples[:limit]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        output.write_text("", encoding="utf-8")
    completed = _read_completed_query_ids(output)
    if provider not in {"openai", "anthropic", "deepseek"}:
        raise ValueError(f"Unknown provider {provider!r}; expected 'openai', 'anthropic', or 'deepseek'")
    default_model = {
        "openai": DEFAULT_MODEL,
        "anthropic": DEFAULT_ANTHROPIC_MODEL,
        "deepseek": DEFAULT_DEEPSEEK_MODEL,
    }[provider]
    resolved_model = model or default_model
    if reconstructor is not None:
        client = reconstructor
    elif provider == "anthropic":
        transport = AnthropicMessagesProvider(model=resolved_model)
        client = OpenAIGovernanceStateReconstructor(
            model=resolved_model, provider=transport, cache_dir=cache_dir,
            max_input_tokens=max_input_tokens, streaming_chunk_tokens=streaming_chunk_tokens,
        )
    elif provider == "deepseek":
        transport = DeepSeekChatCompletionsProvider(model=resolved_model)
        client = OpenAIGovernanceStateReconstructor(
            model=resolved_model, provider=transport, cache_dir=cache_dir,
            max_input_tokens=max_input_tokens, streaming_chunk_tokens=streaming_chunk_tokens,
        )
    else:
        client = OpenAIGovernanceStateReconstructor(
            model=model, cache_dir=cache_dir,
            max_input_tokens=max_input_tokens, streaming_chunk_tokens=streaming_chunk_tokens,
        )

    summary = {
        "total_loaded": total_loaded,
        "selected": len(examples),
        "applicable": 0,
        "non_applicable": 0,
        "skipped": 0,
        "already_completed": 0,
        "executed": 0,
        "valid": 0,
        "invalid_or_error": 0,
        "cache_hits": 0,
        "resumed": 0,
        "output": str(output),
    }
    mode = "a" if output.exists() else "w"
    dimensions_by_query_id = {
        example.query.query_id: applicable_governance_dimensions(example)
        for example in examples
    }
    summary["applicable"] = sum(bool(dimensions) for dimensions in dimensions_by_query_id.values())
    summary["non_applicable"] = len(examples) - summary["applicable"]
    summary["skipped"] = summary["non_applicable"]
    with output.open(mode, encoding="utf-8") as handle:
        for example in examples:
            query_id = example.query.query_id
            if query_id in completed:
                summary["resumed"] += 1
                summary["already_completed"] += 1
                continue
            dimensions = dimensions_by_query_id[query_id]
            record = _base_record(example, dimensions)
            if condition is not None:
                record["condition"] = condition
            if governance_applied is not None:
                record["governance_applied"] = governance_applied
                # Keep the auditable policy field explicit even when no
                # intervention was applied in reconstruction-only RQ1.
                record["governance_policy_version"] = governance_policy_version
            if run_metadata:
                collisions = RESERVED_RESULT_FIELDS.intersection(run_metadata)
                if collisions:
                    raise ValueError(
                        f"run_metadata cannot overwrite reserved result fields: {sorted(collisions)}"
                    )
                record["run_metadata"] = _jsonable(run_metadata)
            if not dimensions:
                record.update({
                    "skipped": True,
                    "skipped_reason": "no_applicable_governance_dimensions",
                    "reconstructed_states": [],
                    "reconstruction_status": "skipped",
                    "validation_status": "not_applicable",
                    "provider": None,
                    "model": resolved_model,
                    "prompt_schema_version": PROMPT_SCHEMA_VERSION,
                    "cache_hit": False,
                    "usage": None,
                    "latency_ms": None,
                    "api_latency_ms": None,
                    "api_request_executed": False,
                    "original_api_latency_ms": None,
                    "request_size_diagnostics": None,
                    "reconstruction_policy": None,
                    "streaming_diagnostics": None,
                    "chunk_count": None,
                    "interaction_count": None,
                    "all_interactions_processed": False,
                    "per_chunk": None,
                    "total_input_tokens": None,
                    "total_output_tokens": None,
                    "total_api_latency_ms": None,
                    "cache_hits": [],
                    "telemetry": {
                        "provider": None,
                        "cache_hit": False,
                        "latency_ms": None,
                        "api_latency_ms": None,
                        "usage_available": False,
                        "usage": None,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                    },
                })
                _write_record(handle, record)
                completed.add(query_id)
                continue

            try:
                summary["executed"] += 1
                states = tuple(client.reconstruct(GovernanceReconstructionInput.from_example(example)))
                state_records = [_state_record(state) for state in states]
                cache_hits = [bool(state.get("provenance", {}).get("cache_hit")) for state in state_records]
                telemetry = _telemetry_record(client)
                record.update({
                    "skipped": False,
                    "skipped_reason": None,
                    "reconstructed_states": state_records,
                    "reconstruction_status": "success",
                    "validation_status": "valid",
                    "provider": telemetry["provider"] or "openai",
                    "model": model or getattr(client, "model", DEFAULT_MODEL),
                    "prompt_schema_version": PROMPT_SCHEMA_VERSION,
                    "cache_hit": bool(telemetry.get("cache_hit")) or any(cache_hits),
                    "usage": telemetry["usage"],
                    "latency_ms": telemetry["latency_ms"],
                    "api_latency_ms": telemetry["api_latency_ms"],
                    "api_request_executed": telemetry.get("api_request_executed"),
                    "original_api_latency_ms": telemetry.get("original_api_latency_ms"),
                    "request_size_diagnostics": telemetry["request_size_diagnostics"],
                    "reconstruction_policy": telemetry["reconstruction_policy"],
                    "streaming_diagnostics": telemetry["streaming_diagnostics"],
                    "chunk_count": telemetry["chunk_count"],
                    "interaction_count": telemetry["interaction_count"],
                    "all_interactions_processed": telemetry["all_interactions_processed"],
                    "per_chunk": telemetry["per_chunk"],
                    "total_input_tokens": telemetry["total_input_tokens"],
                    "total_output_tokens": telemetry["total_output_tokens"],
                    "total_api_latency_ms": telemetry["total_api_latency_ms"],
                    "cache_hits": cache_hits,
                    "telemetry": telemetry,
                })
                summary["valid"] += 1
                summary["cache_hits"] += int(bool(telemetry.get("cache_hit")))
            except Exception as exc:
                summary["invalid_or_error"] += 1
                if isinstance(exc, ReconstructionError):
                    error_message = str(exc)
                    validation_status = exc.status
                else:
                    error_message = type(exc).__name__
                    validation_status = "error"
                telemetry = _telemetry_record(client)
                record.update({
                    "skipped": False,
                    "skipped_reason": None,
                    "reconstructed_states": [],
                    "reconstruction_status": "error",
                    "validation_status": validation_status,
                    "provider": telemetry["provider"] or "openai",
                    "model": model or getattr(client, "model", DEFAULT_MODEL),
                    "prompt_schema_version": PROMPT_SCHEMA_VERSION,
                    "cache_hit": telemetry.get("cache_hit", False),
                    "usage": telemetry["usage"],
                    "latency_ms": telemetry["latency_ms"],
                    "api_latency_ms": telemetry["api_latency_ms"],
                    "api_request_executed": telemetry.get("api_request_executed"),
                    "original_api_latency_ms": telemetry.get("original_api_latency_ms"),
                    "request_size_diagnostics": telemetry["request_size_diagnostics"],
                    "reconstruction_policy": telemetry["reconstruction_policy"],
                    "streaming_diagnostics": telemetry["streaming_diagnostics"],
                    "chunk_count": telemetry["chunk_count"],
                    "interaction_count": telemetry["interaction_count"],
                    "all_interactions_processed": telemetry["all_interactions_processed"],
                    "per_chunk": telemetry["per_chunk"],
                    "total_input_tokens": telemetry["total_input_tokens"],
                    "total_output_tokens": telemetry["total_output_tokens"],
                    "total_api_latency_ms": telemetry["total_api_latency_ms"],
                    "cache_hits": [],
                    "telemetry": telemetry,
                    "error": {
                        "type": type(exc).__name__,
                        "message": error_message,
                        **(getattr(exc, "details", {}) if isinstance(exc, ReconstructionError) else {}),
                    },
                })
            _write_record(handle, record)
            completed.add(query_id)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable governance reconstruction batch.")
    parser.add_argument("--dataset", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=("openai", "anthropic", "deepseek"), default="openai")
    parser.add_argument("--model", default=None, help=f"Exact provider model (default OpenAI: {DEFAULT_MODEL})")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-type", default=None, help="Dataset-native normalized task type filter")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output JSONL")
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--streaming-chunk-tokens", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_batch(
            dataset=args.dataset,
            input_path=args.input,
            output_path=args.output,
            model=args.model,
            provider=args.provider,
            cache_dir=args.cache_dir,
            limit=args.limit,
            query_type=args.query_type,
            overwrite=args.overwrite,
            max_input_tokens=args.max_input_tokens,
            streaming_chunk_tokens=args.streaming_chunk_tokens,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(
        "Batch summary: "
        f"total loaded {summary['total_loaded']}, selected {summary['selected']}, "
        f"applicable {summary['applicable']}, non-applicable {summary['non_applicable']}, "
        f"skipped {summary['skipped']}, already completed {summary['already_completed']}, "
        f"executed {summary['executed']}, "
        f"valid {summary['valid']}, invalid/error {summary['invalid_or_error']}, "
        f"cache hits {summary['cache_hits']}, output {summary['output']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
