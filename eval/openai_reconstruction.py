"""OpenAI-backed governance-state reconstruction for small smoke tests.

The serializer in this module is intentionally allowlisted.  In particular,
adapter metadata is not sent to the model because it may contain evaluator
annotations in some benchmark formats.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from .interfaces import RECONSTRUCTABLE_GOVERNANCE_FIELDS, ReconstructedGovernanceState
from .reconstruction import GovernanceReconstructionInput
from .reconstruction_provider import ProviderTransportError, ReconstructionProvider


PROMPT_SCHEMA_VERSION = "governance-reconstruction-v6"
DEFAULT_MODEL = "gpt-4o-mini"
SINGLE_CALL_POLICY = "single_call_v1"
STREAMING_POLICY = "full_or_chronological_stream_v1"
DEFAULT_MAX_INPUT_TOKENS = 12000
DEFAULT_STREAMING_CHUNK_TOKENS = 8000
DEFAULT_OUTPUT_RESERVE_TOKENS = 2000

# This is the only vocabulary allowed in ``missing_fields``.  ``interaction_id``
# identifies evidence and is required output, but is not itself a reconstructable
# governance fact.
CANONICAL_RECONSTRUCTION_FIELDS = RECONSTRUCTABLE_GOVERNANCE_FIELDS


class ReconstructionError(RuntimeError):
    """A safe, user-facing reconstruction failure without secret material."""

    def __init__(self, message: str, *, status: str = "invalid", details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.details = dict(details or {})


class _StateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interaction_id: str
    authorized_scopes: list[str] | None = None
    is_deleted: StrictBool | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    entity_key: str | None = None
    contextually_valid: StrictBool | None = None
    # Pydantic safely normalizes JSON integers such as 0 or 1 to float while
    # still rejecting non-numeric values and enforcing the [0, 1] bound.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    # Some model outputs omit this optional field or return null.  Normalize
    # both cases to [] after the wire parse.
    missing_fields: list[str] | None = None


class _ResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Compactness is enforced by the prompt and deterministic ID filtering,
    # not by a wire-level maxItems constraint that can make otherwise valid
    # structured output fail before post-processing.
    states: list[_StateOutput] = Field(default_factory=list)


def _datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def model_visible_payload(input_data: GovernanceReconstructionInput) -> dict[str, Any]:
    """Return only the allowlisted fields permitted at the reconstruction boundary."""
    return {
        "raw_interactions": [
            {
                "interaction_id": item.interaction_id,
                "user_id": item.user_id,
                "content": item.content,
                "timestamp": _datetime(item.timestamp),
                "role": item.role,
                "session_id": item.session_id,
            }
            for item in input_data.raw_interactions
        ],
        "query": {
            "query_id": input_data.query.query_id,
            "user_id": input_data.query.user_id,
            "text": input_data.query.text,
            "current_time": _datetime(input_data.query.current_time),
            "allowed_scopes": list(input_data.query.allowed_scopes),
            "requester_role": input_data.query.requester_role,
        },
        "runtime_context": {
            "user_id": input_data.runtime_context.get("user_id"),
            "requester_role": input_data.runtime_context.get("requester_role"),
            "current_time": _datetime(input_data.query.current_time),
            "allowed_scopes": list(input_data.runtime_context.get("allowed_scopes", ())),
        },
    }


def build_reconstruction_prompt(input_data: GovernanceReconstructionInput) -> str:
    payload = model_visible_payload(input_data)
    instructions = (
        "You reconstruct governance state from the supplied model-visible data. "
        "Infer only what is supported by the interactions and runtime context. "
        "Mark unsupported or missing fields in missing_fields rather than guessing. "
        "Distinguish uncertainty from a confident governance fact: use confidence "
        "between 0 and 1 only for your overall confidence. "
        "This is a current checkpoint reconstruction: return only compact states "
        "for interactions that contain governance evidence relevant to the current "
        "query/checkpoint. Do not return one state per historical interaction. "
        "Usually return zero to a few states, not the full history. "
        "Use only these shared-schema "
        "fields: interaction_id, authorized_scopes, is_deleted, valid_from, valid_until, "
        "entity_key, contextually_valid, confidence, missing_fields. "
        "missing_fields may contain only authorized_scopes, is_deleted, valid_from, "
        "valid_until, entity_key, or contextually_valid. A concrete value must never "
        "also appear in missing_fields. Unknown deletion is null plus is_deleted in "
        "missing_fields; never encode unknown deletion as false. For lifecycle or "
        "deletion checkpoints, output only states directly relevant to the memory, "
        "entity, or fact referenced by the current query; do not output unrelated "
        "ordinary facts merely because they occur later or nearby. When an interaction "
        "explicitly states that an earlier memory must be deleted or unavailable, "
        "first identify the memory/fact referenced by the current query, then identify "
        "its original supporting memory-bearing interaction. Inspect later visible "
        "interactions for deletion or revocation of that same memory/fact only. Attach "
        "the resulting lifecycle state to the original memory-bearing interaction_id, not to the "
        "later deletion instruction. A deletion instruction about another entity or "
        "fact must not affect this state. Use entity_key to preserve the supported "
        "identity of the queried memory when available; when same-fact deletion is "
        "explicitly supported, emit is_deleted=true. If deletion is not supported "
        "for that same memory, leave is_deleted unknown or omit the state."
    )
    return (
        f"Prompt/schema version: {PROMPT_SCHEMA_VERSION}\n"
        f"Instructions:\n{instructions}\n\n"
        "Model-visible reconstruction input (JSON):\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    )


def _cache_payload(model: str, prompt: str, input_data: GovernanceReconstructionInput, provider: str = "openai") -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "input": model_visible_payload(input_data),
        "prompt": prompt,
    }
    # Preserve the frozen OpenAI cache identity; namespace new providers so
    # their results cannot collide with legacy OpenAI entries.
    if provider != "openai":
        payload["provider"] = provider
    return payload


def _validation_summary(exc: Exception) -> str:
    """Summarize validation failures without exposing parsed values/prompts."""
    if isinstance(exc, ValidationError):
        parts: list[str] = []
        for error in exc.errors()[:3]:
            path = ".".join(str(part) for part in error.get("loc", ())) or "response"
            reason = str(error.get("msg", "validation failed"))
            error_type = str(error.get("type", "validation_error"))
            parts.append(f"{path}: {reason} [{error_type}]")
        return "; ".join(parts) or "response: validation failed [validation_error]"
    return f"response: {type(exc).__name__}"


def _provider_exception_details(exc: Exception) -> dict[str, Any]:
    """Extract safe scalar provider diagnostics from an OpenAI/API exception."""
    if isinstance(exc, ProviderTransportError):
        return dict(exc.details)
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        provider_body = body["error"]
    elif isinstance(body, dict):
        provider_body = body
    else:
        provider_body = {}
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    request_id = getattr(exc, "request_id", None)
    if request_id is None and headers is not None:
        request_id = headers.get("x-request-id")
    provider_message = provider_body.get("message") or getattr(exc, "message", None) or str(exc)
    details = {
        "provider_exception": type(exc).__name__,
        "http_status": getattr(exc, "status_code", None) or getattr(response, "status_code", None),
        "provider_code": provider_body.get("code") or getattr(exc, "code", None),
        "provider_type": provider_body.get("type") or getattr(exc, "type", None),
        "provider_message": str(provider_message) if provider_message is not None else None,
        "request_id": str(request_id) if request_id is not None else None,
    }
    return details


def _provider_failure_status(details: dict[str, Any]) -> str:
    category = details.get("error_category")
    return category if category in {"transport_error", "structural_invalid", "semantic_invalid"} else "transport_error"


def _provider_failure_prefix(provider: str) -> str:
    return "OpenAI" if provider == "openai" else provider.capitalize()


def _request_size_diagnostics(input_data: GovernanceReconstructionInput, prompt: str) -> dict[str, Any]:
    return {
        "interaction_count": len(input_data.raw_interactions),
        "serialized_prompt_input_character_count": len(prompt),
        "serialized_prompt_input_byte_count": len(prompt.encode("utf-8")),
        "estimated_input_tokens": None,
        "estimated_input_tokens_available": False,
        "budget_unit": None,
    }


@lru_cache(maxsize=8)
def _token_counter(model: str):
    """Return an exact model tokenizer when its local encoding is available."""
    try:
        import tiktoken
        encoder = tiktoken.encoding_for_model(model)
        return encoder.encode, "exact_model_tokenizer"
    except Exception:
        return None, "utf8_bytes_conservative_fallback"


def _state_payload(state: ReconstructedGovernanceState) -> dict[str, Any]:
    metadata = dict(state.metadata)
    return {
        "interaction_id": state.interaction_id,
        "authorized_scopes": list(state.authorized_scopes),
        "is_deleted": state.is_deleted,
        "valid_from": _datetime(state.valid_from),
        "valid_until": _datetime(state.valid_until),
        "entity_key": state.entity_key,
        "confidence": state.confidence,
        "missing_fields": list(state.missing_fields),
        "contextually_valid": metadata.get("contextually_valid"),
    }


def _normalize_response(parsed: _ResponseSchema) -> _ResponseSchema:
    """Apply only lossless normalization, then enforce v3 semantics."""
    normalized_states: list[_StateOutput] = []
    for index, item in enumerate(parsed.states):
        missing = list(dict.fromkeys(item.missing_fields or ()))
        invalid = set(missing) - CANONICAL_RECONSTRUCTION_FIELDS
        if invalid:
            raise ReconstructionError(
                f"Structured reconstruction validation failed: states.{index}.missing_fields: "
                f"non-canonical field name(s) {sorted(invalid)} [canonical_field]",
                status="semantic_invalid",
            )
        # A concrete fact is stronger than a duplicated missing marker.
        # Empty authorized_scopes is deliberately not normalized: [] with a
        # missing marker means unknown, while [] without one means known-empty.
        for field in CANONICAL_RECONSTRUCTION_FIELDS:
            value = getattr(item, field)
            if value is not None and field in missing and not (
                field == "authorized_scopes" and value == []
            ):
                missing.remove(field)
        for field in CANONICAL_RECONSTRUCTION_FIELDS:
            if field in missing and getattr(item, field) is not None and not (
                field == "authorized_scopes" and getattr(item, field) == []
            ):
                raise ReconstructionError(
                    f"Structured reconstruction validation failed: states.{index}.{field}: "
                    "concrete value cannot also be missing [inconsistent_state]",
                    status="semantic_invalid",
                )
        # An omitted/null value for a boolean or scope field means unknown;
        # recording it as missing is the only safe v3 representation.
        for field in ("authorized_scopes", "is_deleted", "contextually_valid"):
            if getattr(item, field) is None and field not in missing:
                missing.append(field)
        if item.valid_from is not None and item.valid_until is not None and item.valid_from > item.valid_until:
            raise ReconstructionError(
                f"Structured reconstruction validation failed: states.{index}: "
                "valid_from must not be later than valid_until [invalid_temporal_interval]",
                status="semantic_invalid",
            )
        normalized_states.append(item.model_copy(update={"missing_fields": missing}))
    return _ResponseSchema(states=normalized_states)


def _state_from_output(
    item: _StateOutput,
    *,
    model: str,
    cache_hit: bool,
    telemetry: dict[str, Any] | None = None,
    provider: str = "openai",
) -> ReconstructedGovernanceState:
    # ``missing_fields`` is authoritative.  In particular, a null temporal
    # bound is a valid representation of an unbounded interval, not itself a
    # claim that temporal information was unsupported.
    missing = set(item.missing_fields or ())
    confidence = item.confidence
    provenance_data: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "cache_hit": cache_hit,
        "validation_status": "valid",
    }
    if telemetry is not None:
        provenance_data["telemetry"] = telemetry
    if confidence is not None:
        provenance_data["confidence"] = confidence
    zero_length_interval = (
        item.valid_from is not None
        and item.valid_until is not None
        and item.valid_from == item.valid_until
    )
    return ReconstructedGovernanceState(
        interaction_id=item.interaction_id,
        authorized_scopes=tuple(item.authorized_scopes or ()),
        is_deleted=item.is_deleted,
        valid_from=item.valid_from,
        valid_until=item.valid_until,
        entity_key=item.entity_key,
        confidence=confidence,
        provenance=json.dumps(provenance_data, sort_keys=True, separators=(",", ":")),
        missing_fields=tuple(sorted(missing)),
        metadata={
            "provider": provider,
            "model": model,
            "prompt_schema_version": PROMPT_SCHEMA_VERSION,
            "cache_hit": cache_hit,
            "validation_status": "valid",
            "zero_length_interval": zero_length_interval,
            **({"telemetry": telemetry} if telemetry is not None else {}),
            **({"contextually_valid": item.contextually_valid} if item.contextually_valid is not None else {}),
        },
    )


def _reconstructed_states(
    parsed: _ResponseSchema,
    input_data: GovernanceReconstructionInput,
    *,
    model: str,
    cache_hit: bool,
    telemetry: dict[str, Any] | None = None,
    valid_ids: set[str] | None = None,
    provider: str = "openai",
) -> tuple[ReconstructedGovernanceState, ...]:
    """Keep unique grounded states; deterministically filter unknown IDs."""
    valid_ids = valid_ids or {interaction.interaction_id for interaction in input_data.raw_interactions}
    seen: set[str] = set()
    states: list[ReconstructedGovernanceState] = []
    for item in parsed.states:
        if item.interaction_id not in valid_ids or item.interaction_id in seen:
            continue
        seen.add(item.interaction_id)
        states.append(_state_from_output(item, model=model, cache_hit=cache_hit, telemetry=telemetry, provider=provider))
    return tuple(states)


class OpenAIGovernanceStateReconstructor:
    """Reconstruct compact checkpoint-relevant states in one Responses API call."""

    def __init__(
        self,
        model: str | None = None,
        cache_dir: str | Path | None = None,
        client: Any | None = None,
        provider: ReconstructionProvider | None = None,
        max_input_tokens: int | None = None,
        streaming_chunk_tokens: int | None = None,
        output_reserve_tokens: int | None = None,
    ) -> None:
        self.model = model or os.environ.get("OPENAI_RECONSTRUCTION_MODEL", DEFAULT_MODEL)
        configured_cache = cache_dir or os.environ.get("OPENAI_RECONSTRUCTION_CACHE_DIR", ".cache/reconstruction")
        self.cache_dir = Path(configured_cache)
        self._client = client
        self.provider = provider
        self.provider_name = provider.provider_name if provider is not None else "openai"
        if provider is not None and model is None:
            self.model = provider.model
        self.max_input_tokens = (
            max_input_tokens if max_input_tokens is not None
            else int(os.environ.get("OPENAI_RECONSTRUCTION_MAX_INPUT_TOKENS", DEFAULT_MAX_INPUT_TOKENS))
        )
        self.streaming_chunk_tokens = (
            streaming_chunk_tokens if streaming_chunk_tokens is not None
            else int(os.environ.get("OPENAI_RECONSTRUCTION_STREAMING_CHUNK_TOKENS", DEFAULT_STREAMING_CHUNK_TOKENS))
        )
        self.output_reserve_tokens = (
            output_reserve_tokens if output_reserve_tokens is not None
            else int(os.environ.get("OPENAI_RECONSTRUCTION_OUTPUT_RESERVE_TOKENS", DEFAULT_OUTPUT_RESERVE_TOKENS))
        )
        if self.max_input_tokens <= self.output_reserve_tokens:
            raise ValueError("max_input_tokens must exceed output_reserve_tokens")
        if self.streaming_chunk_tokens < 1:
            raise ValueError("streaming_chunk_tokens must be positive")
        self._count_tokens, self.token_counting = _token_counter(self.model)
        self.budget_unit = "model_tokens" if self._count_tokens else "utf8_bytes_conservative_units"
        self.last_telemetry: dict[str, Any] = {
            "provider": self.provider_name,
            "model": self.model,
            "cache_hit": False,
            "api_latency_ms": None,
            "latency_ms": None,
            "usage_available": False,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "request_size_diagnostics": None,
            "reconstruction_policy": SINGLE_CALL_POLICY,
            "streaming_diagnostics": None,
        }
        # Reconstruction-level publication point used by batch execution.  It
        # avoids requiring callers to recover request telemetry from states.
        self.last_result: dict[str, Any] = {
            "states": (),
            "status": "not_run",
            "telemetry": self.last_telemetry,
        }

    @staticmethod
    def _usage_telemetry(usage: Any) -> dict[str, Any]:
        def value(name: str) -> int | None:
            if usage is None:
                return None
            if isinstance(usage, dict):
                raw = usage.get(name)
                if raw is None:
                    raw = usage.get({
                        "input_tokens": "prompt_tokens",
                        "output_tokens": "completion_tokens",
                    }.get(name, name))
            else:
                raw = getattr(usage, name, None)
                if raw is None:
                    raw = getattr(usage, {
                        "input_tokens": "prompt_tokens",
                        "output_tokens": "completion_tokens",
                    }.get(name, name), None)
            return int(raw) if raw is not None else None

        input_tokens = value("input_tokens")
        output_tokens = value("output_tokens")
        total_tokens = value("total_tokens")
        values = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        return {
            "usage_available": any(item is not None for item in (input_tokens, output_tokens, total_tokens)),
            **values,
            "usage": values if any(item is not None for item in values.values()) else None,
        }

    def _publish_result(
        self,
        states: tuple[ReconstructedGovernanceState, ...] = (),
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        result: dict[str, Any] = {
            "states": states,
            "provider": self.provider_name,
            "model": self.model,
            "cache_hit": bool(self.last_telemetry.get("cache_hit")),
            "telemetry": dict(self.last_telemetry),
            "status": status,
        }
        if error is not None:
            result["error"] = error
        self.last_result = result

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ReconstructionError("The openai package is required for OpenAI reconstruction") from exc
            if not os.environ.get("OPENAI_API_KEY"):
                raise ReconstructionError("OPENAI_API_KEY is required for OpenAI reconstruction")
            self._client = OpenAI()
        return self._client

    def _request_structured(self, prompt: str) -> Any:
        if self.provider is not None:
            result = self.provider.request_structured_reconstruction(
                prompt=prompt,
                schema=_ResponseSchema,
                request_metadata={"model": self.model, "provider": self.provider_name},
            )
            usage = SimpleNamespace(
                input_tokens=result.telemetry.input_tokens,
                output_tokens=result.telemetry.output_tokens,
                total_tokens=result.telemetry.total_tokens,
            )
            return SimpleNamespace(
                output_parsed=result.parsed,
                usage=usage,
                provider_telemetry=result.telemetry.as_dict(),
            )
        return self.client.responses.parse(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            text_format=_ResponseSchema,
        )

    def _measure_prompt(self, prompt: str) -> int:
        return len(self._count_tokens(prompt)) if self._count_tokens else len(prompt.encode("utf-8"))

    def _stream_prompt(
        self,
        input_data: GovernanceReconstructionInput,
        interactions: tuple,
        previous_states: tuple[ReconstructedGovernanceState, ...],
        chunk_index: int,
    ) -> str:
        chunk_input = GovernanceReconstructionInput(
            raw_interactions=interactions,
            query=input_data.query,
            runtime_context=input_data.runtime_context,
        )
        prompt = build_reconstruction_prompt(chunk_input)
        carry = {
            "reconstruction_policy": STREAMING_POLICY,
            "chunk_index": chunk_index,
            "previous_model_generated_state": [_state_payload(state) for state in previous_states],
        }
        return prompt + "\n\nIncremental reconstruction instructions (model-generated state only):\n" + json.dumps(
            carry, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\nUpdate, revise, or remove prior state when this chronological chunk provides newer evidence."

    def _stream_call(
        self,
        prompt: str,
        input_data: GovernanceReconstructionInput,
        cache_path: Path,
    ) -> tuple[_ResponseSchema, dict[str, Any]]:
        request_size = {
            "interaction_count": len(input_data.raw_interactions),
            "serialized_prompt_input_character_count": len(prompt),
            "serialized_prompt_input_byte_count": len(prompt.encode("utf-8")),
            "estimated_input_tokens": len(self._count_tokens(prompt)) if self._count_tokens else None,
            "estimated_input_tokens_available": self._count_tokens is not None,
            "token_counting": self.token_counting,
            "budget_unit": self.budget_unit,
        }
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_response = cached.get("response", cached)
                cached_telemetry = dict(cached.get("telemetry", {})) if isinstance(cached, dict) else {}
                parsed = _normalize_response(_ResponseSchema.model_validate(cached_response))
            except (OSError, json.JSONDecodeError, ValidationError, ReconstructionError) as exc:
                detail = _validation_summary(exc) if isinstance(exc, ValidationError) else str(exc)
                raise ReconstructionError(f"Invalid streaming reconstruction cache entry: {cache_path.name}: {detail}", status="structural_invalid") from exc
            telemetry = {
                "provider": self.provider_name, "model": self.model, "cache_hit": True,
                "api_request_executed": False,
                "api_latency_ms": cached_telemetry.get("api_latency_ms"), "latency_ms": None,
                "original_api_latency_ms": cached_telemetry.get(
                    "original_api_latency_ms", cached_telemetry.get("api_latency_ms")
                ),
                "request_size_diagnostics": request_size,
                **self._usage_telemetry(cached_telemetry if cached_telemetry.get("usage_available") else None),
            }
            return parsed, telemetry
        started = time.perf_counter()
        try:
            response = self._request_structured(prompt)
        except Exception as exc:
            latency = round((time.perf_counter() - started) * 1000, 3)
            details = {} if isinstance(exc, ValidationError) else _provider_exception_details(exc)
            name = details.get("provider_exception", type(exc).__name__)
            message = details.get("provider_message", _validation_summary(exc))
            raise ReconstructionError(
                f"{_provider_failure_prefix(self.provider_name)} reconstruction request failed: {name}: {message}",
                status="structural_invalid" if isinstance(exc, ValidationError) else _provider_failure_status(details),
                details={**details, "api_latency_ms": latency, "request_size_diagnostics": request_size},
            ) from exc
        latency = round((time.perf_counter() - started) * 1000, 3)
        provider_metadata = getattr(response, "provider_telemetry", {}) or {}
        telemetry = {
            "provider": self.provider_name, "model": self.model, "cache_hit": False,
            "api_request_executed": True,
            "api_latency_ms": latency, "latency_ms": latency,
            "original_api_latency_ms": latency,
            "request_size_diagnostics": request_size,
            "provider_request_id": provider_metadata.get("request_id"),
            **self._usage_telemetry(getattr(response, "usage", None)),
        }
        try:
            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                raise ReconstructionError("Structured reconstruction returned no structured output", status="structural_invalid")
            if not isinstance(parsed, _ResponseSchema):
                parsed = _ResponseSchema.model_validate(parsed)
            parsed = _normalize_response(parsed)
        except (ValidationError, ReconstructionError) as exc:
            detail = _validation_summary(exc) if isinstance(exc, ValidationError) else str(exc)
            raise ReconstructionError(
                f"Structured reconstruction returned malformed streaming output: {detail}",
                status=exc.status if isinstance(exc, ReconstructionError) else "structural_invalid",
                details={
                    "api_request_executed": bool(telemetry.get("api_request_executed")),
                    "api_latency_ms": telemetry.get("api_latency_ms"),
                    "original_api_latency_ms": telemetry.get("original_api_latency_ms"),
                    "usage": telemetry.get("usage"),
                    "input_tokens": telemetry.get("input_tokens"),
                    "output_tokens": telemetry.get("output_tokens"),
                    "total_tokens": telemetry.get("total_tokens"),
                    "request_size_diagnostics": telemetry.get("request_size_diagnostics"),
                    "provider_request_id": telemetry.get("provider_request_id"),
                },
            ) from exc
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "response": json.loads(parsed.model_dump_json()),
            "telemetry": telemetry,
            "reconstruction_policy": STREAMING_POLICY,
        }, sort_keys=True), encoding="utf-8")
        return parsed, telemetry

    def _reconstruct_streaming(self, input_data: GovernanceReconstructionInput) -> tuple[ReconstructedGovernanceState, ...]:
        interactions = input_data.raw_interactions
        previous: tuple[ReconstructedGovernanceState, ...] = ()
        processed: set[str] = set()
        chunk_records: list[dict[str, Any]] = []
        position = 0
        chunk_index = 0
        budget = min(self.streaming_chunk_tokens, self.max_input_tokens - self.output_reserve_tokens)
        while position < len(interactions):
            current: list = []
            while position + len(current) < len(interactions):
                candidate = tuple(current + [interactions[position + len(current)]])
                prompt = self._stream_prompt(input_data, candidate, previous, chunk_index)
                if current and self._measure_prompt(prompt) > budget:
                    break
                if not current and self._measure_prompt(prompt) > budget:
                    request_size = {
                        "interaction_count": 1,
                        "serialized_prompt_input_character_count": len(prompt),
                        "serialized_prompt_input_byte_count": len(prompt.encode("utf-8")),
                        "estimated_input_tokens": self._measure_prompt(prompt),
                        "estimated_input_tokens_available": self._count_tokens is not None,
                        "token_counting": self.token_counting,
                        "budget_unit": self.budget_unit,
                    }
                    error = ReconstructionError(
                        "A single interaction exceeds the configured streaming input budget",
                        status="configuration_error",
                        details={
                            "chunk_index": chunk_index,
                            "chunk_interaction_count": 1,
                            "reconstruction_policy": STREAMING_POLICY,
                            "request_size_diagnostics": request_size,
                        },
                    )
                    self.last_telemetry = {
                        "provider": self.provider_name, "model": self.model, "cache_hit": False,
                        "api_request_executed": False,
                        "reconstruction_policy": STREAMING_POLICY,
                        "streaming_diagnostics": {
                            "chunk_count": chunk_index + 1, "interaction_count": len(interactions),
                            "chunk_budget": budget, "max_input_budget": self.max_input_tokens,
                            "output_reserve": self.output_reserve_tokens, "budget_unit": self.budget_unit,
                            "all_interactions_processed": False,
                            "per_chunk": [*chunk_records, {
                                "chunk_index": chunk_index, "interaction_count": 1,
                                "interaction_ids": [interactions[position].interaction_id],
                                "cache_hit": False, "api_request_executed": False,
                                "api_latency_ms": None, "request_size_diagnostics": request_size,
                            }],
                        },
                        "request_size_diagnostics": request_size,
                        "api_latency_ms": None, "latency_ms": None,
                        "original_api_latency_ms": None,
                        **self._usage_telemetry(None),
                    }
                    self._publish_result(status=error.status, error=str(error))
                    raise error
                current.append(interactions[position + len(current)])
            current_tuple = tuple(current)
            prompt = self._stream_prompt(input_data, current_tuple, previous, chunk_index)
            cache_identity = {
                "model": self.model, "policy": STREAMING_POLICY, "prompt_schema_version": PROMPT_SCHEMA_VERSION,
                "prompt": prompt,
            }
            if self.provider_name != "openai":
                cache_identity["provider"] = self.provider_name
            cache_key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
            try:
                parsed, telemetry = self._stream_call(prompt, GovernanceReconstructionInput(
                    raw_interactions=current_tuple, query=input_data.query, runtime_context=input_data.runtime_context
                ), self.cache_dir / "streaming" / f"{cache_key}.json")
            except ReconstructionError as exc:
                exc.details.setdefault("chunk_index", chunk_index)
                exc.details.setdefault("chunk_interaction_count", len(current_tuple))
                exc.details.setdefault("reconstruction_policy", STREAMING_POLICY)
                failed_chunk = {
                    "chunk_index": chunk_index,
                    "interaction_count": len(current_tuple),
                    "interaction_ids": [item.interaction_id for item in current_tuple],
                    "cache_hit": bool(exc.details.get("cache_hit", False)),
                    "api_request_executed": bool(exc.details.get("api_request_executed", False)),
                    "api_latency_ms": exc.details.get("api_latency_ms"),
                    "original_api_latency_ms": exc.details.get("original_api_latency_ms"),
                    "usage": exc.details.get("usage"),
                    "request_size_diagnostics": exc.details.get("request_size_diagnostics"),
                    "error": exc.status,
                }
                observed_chunks = [*chunk_records, failed_chunk]

                def observed_total(field: str) -> int | None:
                    values = [
                        (item.get("usage") or {}).get(field)
                        for item in observed_chunks
                        if isinstance((item.get("usage") or {}).get(field), int)
                    ]
                    return sum(values) if values else None

                cumulative_latency = round(sum(
                    item.get("api_latency_ms") or 0
                    for item in observed_chunks
                    if item.get("api_request_executed")
                ), 3)
                cumulative_original_latency = round(sum(
                    item.get("original_api_latency_ms") or 0 for item in observed_chunks
                ), 3)
                cumulative_usage = {
                    "input_tokens": observed_total("input_tokens"),
                    "output_tokens": observed_total("output_tokens"),
                    "total_tokens": observed_total("total_tokens"),
                }
                cumulative_usage = cumulative_usage if any(value is not None for value in cumulative_usage.values()) else None
                exc.details.update({
                    "completed_chunk_count": len(chunk_records),
                    "failed_chunk_index": chunk_index,
                    "cumulative_api_latency_ms": cumulative_latency,
                    "cumulative_usage": cumulative_usage,
                })
                self.last_telemetry = {
                    "provider": self.provider_name, "model": self.model, "cache_hit": False,
                    "reconstruction_policy": STREAMING_POLICY, "streaming_diagnostics": {
                        "chunk_count": chunk_index + 1, "interaction_count": len(interactions),
                        "chunk_budget": budget, "max_input_budget": self.max_input_tokens,
                        "output_reserve": self.output_reserve_tokens, "budget_unit": self.budget_unit,
                        "all_interactions_processed": False,
                        "completed_chunk_count": len(chunk_records),
                        "failed_chunk_index": chunk_index,
                        "per_chunk": observed_chunks,
                        "total_input_tokens": cumulative_usage["input_tokens"] if cumulative_usage else None,
                        "total_output_tokens": cumulative_usage["output_tokens"] if cumulative_usage else None,
                        "total_tokens": cumulative_usage["total_tokens"] if cumulative_usage else None,
                        "total_api_latency_ms": cumulative_latency,
                        "total_original_api_latency_ms": cumulative_original_latency,
                    },
                    "request_size_diagnostics": exc.details.get("request_size_diagnostics"),
                    "api_latency_ms": cumulative_latency, "latency_ms": cumulative_latency,
                    "original_api_latency_ms": cumulative_original_latency,
                    "usage_available": cumulative_usage is not None,
                    "input_tokens": cumulative_usage["input_tokens"] if cumulative_usage else None,
                    "output_tokens": cumulative_usage["output_tokens"] if cumulative_usage else None,
                    "total_tokens": cumulative_usage["total_tokens"] if cumulative_usage else None,
                    "usage": cumulative_usage,
                }
                self._publish_result(status=exc.status, error=str(exc))
                raise
            valid_ids = processed | {item.interaction_id for item in current_tuple}
            previous = _reconstructed_states(parsed, input_data, model=self.model, cache_hit=telemetry["cache_hit"], telemetry=telemetry, valid_ids=valid_ids, provider=self.provider_name)
            processed.update(item.interaction_id for item in current_tuple)
            chunk_records.append({
                "chunk_index": chunk_index, "interaction_count": len(current_tuple),
                "interaction_ids": [item.interaction_id for item in current_tuple],
                "cache_hit": telemetry["cache_hit"],
                "api_request_executed": telemetry.get("api_request_executed", not telemetry["cache_hit"]),
                "api_latency_ms": telemetry.get("api_latency_ms"),
                "original_api_latency_ms": telemetry.get("original_api_latency_ms"),
                "usage": telemetry.get("usage"), "request_size_diagnostics": telemetry.get("request_size_diagnostics"),
            })
            position += len(current_tuple)
            chunk_index += 1
        total_input = [(item.get("usage") or {}).get("input_tokens") for item in chunk_records]
        total_output = [(item.get("usage") or {}).get("output_tokens") for item in chunk_records]
        total_tokens = [(item.get("usage") or {}).get("total_tokens") for item in chunk_records]

        def complete(values: list[Any]) -> bool:
            return len(values) == len(chunk_records) and all(isinstance(value, int) for value in values)

        input_complete = complete(total_input)
        output_complete = complete(total_output)
        total_complete = complete(total_tokens)
        self.last_telemetry = {
            "provider": self.provider_name, "model": self.model, "cache_hit": all(item["cache_hit"] for item in chunk_records),
            "reconstruction_policy": STREAMING_POLICY,
            "streaming_diagnostics": {
                "chunk_count": len(chunk_records), "interaction_count": len(interactions),
                "chunk_budget": budget, "max_input_budget": self.max_input_tokens,
                "output_reserve": self.output_reserve_tokens, "budget_unit": self.budget_unit,
                "all_interactions_processed": processed == {item.interaction_id for item in interactions},
                "per_chunk": chunk_records,
                "total_input_tokens": sum(total_input) if input_complete else None,
                "total_output_tokens": sum(total_output) if output_complete else None,
                "total_tokens": sum(total_tokens) if total_complete else None,
                # Only fresh provider requests count toward API latency.  A
                # cached chunk may retain its original request latency for
                # provenance, but cache lookup time is never included here.
                "total_api_latency_ms": round(sum(
                    item.get("api_latency_ms") or 0 for item in chunk_records if item.get("api_request_executed")
                ), 3),
                "total_original_api_latency_ms": round(sum(
                    item.get("original_api_latency_ms") or 0 for item in chunk_records
                ), 3),
            },
            "request_size_diagnostics": chunk_records[-1]["request_size_diagnostics"] if chunk_records else None,
            "api_latency_ms": round(sum(
                item.get("api_latency_ms") or 0 for item in chunk_records if item.get("api_request_executed")
            ), 3),
            "latency_ms": round(sum(
                item.get("api_latency_ms") or 0 for item in chunk_records if item.get("api_request_executed")
            ), 3),
            "usage_available": input_complete or output_complete or total_complete,
            "input_tokens": sum(total_input) if input_complete else None,
            "output_tokens": sum(total_output) if output_complete else None,
            "total_tokens": sum(total_tokens) if total_complete else None,
            "usage": {
                "input_tokens": sum(total_input) if input_complete else None,
                "output_tokens": sum(total_output) if output_complete else None,
                "total_tokens": sum(total_tokens) if total_complete else None,
            } if input_complete or output_complete or total_complete else None,
        }
        self._publish_result(previous, status="success")
        return previous

    def reconstruct(self, input_data: GovernanceReconstructionInput) -> tuple[ReconstructedGovernanceState, ...]:
        prompt = build_reconstruction_prompt(input_data)
        request_size_diagnostics = _request_size_diagnostics(input_data, prompt)
        request_size_diagnostics["token_counting"] = self.token_counting
        request_size_diagnostics["budget_unit"] = self.budget_unit
        budget = self.max_input_tokens - self.output_reserve_tokens
        if self._measure_prompt(prompt) > budget:
            return self._reconstruct_streaming(input_data)
        cache_key = hashlib.sha256(json.dumps(_cache_payload(self.model, prompt, input_data, self.provider_name), sort_keys=True, default=str).encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_response = cached.get("response", cached)
                cached_telemetry = dict(cached.get("telemetry", {})) if isinstance(cached, dict) else {}
                parsed = _normalize_response(_ResponseSchema.model_validate(cached_response))
                self.last_telemetry = {
                    "provider": self.provider_name,
                    "model": self.model,
                    "cache_hit": True,
                    "api_request_executed": False,
                    # This is the original provider-request latency, not the
                    # cache lookup duration.  Current cache replay latency is
                    # intentionally represented by latency_ms=None.
                    "api_latency_ms": cached_telemetry.get("api_latency_ms"),
                    "original_api_latency_ms": cached_telemetry.get(
                        "original_api_latency_ms", cached_telemetry.get("api_latency_ms")
                    ),
                    "latency_ms": None,
                    "request_size_diagnostics": request_size_diagnostics,
                    "reconstruction_policy": SINGLE_CALL_POLICY,
                    "streaming_diagnostics": {
                        "chunk_count": 1, "interaction_count": len(input_data.raw_interactions),
                        "all_interactions_processed": True,
                        "total_input_tokens": cached_telemetry.get("input_tokens"),
                        "total_output_tokens": cached_telemetry.get("output_tokens"),
                        "total_api_latency_ms": cached_telemetry.get("api_latency_ms"),
                        "per_chunk": [{"chunk_index": 0, "interaction_count": len(input_data.raw_interactions),
                                       "cache_hit": True, "api_request_executed": False,
                                       "api_latency_ms": cached_telemetry.get("api_latency_ms"),
                                       "original_api_latency_ms": cached_telemetry.get(
                                           "original_api_latency_ms", cached_telemetry.get("api_latency_ms")
                                       ),
                                       "usage": cached_telemetry.get("usage"),
                                       "request_size_diagnostics": request_size_diagnostics}],
                    },
                    **self._usage_telemetry(cached_telemetry if cached_telemetry.get("usage_available") else None),
                }
                states = _reconstructed_states(parsed, input_data, model=self.model, cache_hit=True, telemetry=self.last_telemetry, provider=self.provider_name)
                self._publish_result(states, status="success")
                return states
            except (OSError, json.JSONDecodeError) as exc:
                raise ReconstructionError(f"Invalid reconstruction cache entry: {cache_path.name}") from exc
            except (ValidationError, ReconstructionError) as exc:
                if isinstance(exc, ValidationError):
                    detail = _validation_summary(exc)
                else:
                    detail = str(exc)
                raise ReconstructionError(f"Invalid reconstruction cache entry: {cache_path.name}: {detail}") from exc

        api_started = time.perf_counter()
        try:
            response = self._request_structured(prompt)
        except Exception as exc:
            latency_ms = round((time.perf_counter() - api_started) * 1000, 3)
            self.last_telemetry = {
                "provider": self.provider_name,
                "model": self.model,
                "cache_hit": False,
                "api_request_executed": True,
                "api_latency_ms": latency_ms,
                "latency_ms": latency_ms,
                "original_api_latency_ms": latency_ms,
                "request_size_diagnostics": request_size_diagnostics,
                "reconstruction_policy": SINGLE_CALL_POLICY,
                "streaming_diagnostics": {
                    "chunk_count": 1, "interaction_count": len(input_data.raw_interactions),
                    "all_interactions_processed": True,
                "total_input_tokens": None, "total_output_tokens": None, "total_api_latency_ms": latency_ms,
                    "per_chunk": [{"chunk_index": 0, "interaction_count": len(input_data.raw_interactions),
                                   "cache_hit": False, "api_latency_ms": latency_ms,
                                   "request_size_diagnostics": request_size_diagnostics}],
                },
                **self._usage_telemetry(None),
            }
            details = {} if isinstance(exc, ValidationError) else _provider_exception_details(exc)
            provider_name = details.get("provider_exception", type(exc).__name__)
            provider_message = details.get("provider_message", _validation_summary(exc))
            error = ReconstructionError(
                f"{_provider_failure_prefix(self.provider_name)} reconstruction request failed: "
                f"{provider_name}: {provider_message}",
                status="structural_invalid" if isinstance(exc, ValidationError) else _provider_failure_status(details),
                details=details,
            )
            self._publish_result(status=error.status, error=str(error))
            raise error from exc
        latency_ms = round((time.perf_counter() - api_started) * 1000, 3)
        provider_metadata = getattr(response, "provider_telemetry", {}) or {}
        self.last_telemetry = {
            "provider": self.provider_name,
            "model": self.model,
            "cache_hit": False,
            "api_request_executed": True,
            "api_latency_ms": latency_ms,
            "latency_ms": latency_ms,
            "original_api_latency_ms": latency_ms,
            "request_size_diagnostics": request_size_diagnostics,
            "provider_request_id": provider_metadata.get("request_id"),
            "reconstruction_policy": SINGLE_CALL_POLICY,
            "streaming_diagnostics": {
                "chunk_count": 1, "interaction_count": len(input_data.raw_interactions),
                "all_interactions_processed": True,
                "total_input_tokens": self._usage_telemetry(getattr(response, "usage", None)).get("input_tokens"),
                "total_output_tokens": self._usage_telemetry(getattr(response, "usage", None)).get("output_tokens"),
                "total_api_latency_ms": latency_ms,
                "per_chunk": [{"chunk_index": 0, "interaction_count": len(input_data.raw_interactions),
                               "cache_hit": False, "api_latency_ms": latency_ms,
                               "usage": self._usage_telemetry(getattr(response, "usage", None)).get("usage"),
                               "request_size_diagnostics": request_size_diagnostics}],
            },
            **self._usage_telemetry(getattr(response, "usage", None)),
        }
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            error = ReconstructionError("Structured reconstruction returned no structured output", status="structural_invalid")
            self._publish_result(status=error.status, error=str(error))
            raise error
        if not isinstance(parsed, _ResponseSchema):
            try:
                parsed = _ResponseSchema.model_validate(parsed)
            except ValidationError as exc:
                error = ReconstructionError(
                    f"Structured reconstruction returned malformed structured output: {_validation_summary(exc)}",
                    status="structural_invalid",
                )
                self._publish_result(status=error.status, error=str(error))
                raise error from exc
        try:
            parsed = _normalize_response(parsed)
        except (ValidationError, ReconstructionError) as exc:
            detail = _validation_summary(exc) if isinstance(exc, ValidationError) else str(exc)
            error = ReconstructionError(
                f"Structured reconstruction returned semantically invalid structured output: {detail}",
                status=exc.status if isinstance(exc, ReconstructionError) else "semantic_invalid",
            )
            self._publish_result(status=error.status, error=str(error))
            raise error from exc
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({
                "response": json.loads(parsed.model_dump_json()),
                "telemetry": self.last_telemetry,
            }, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            error = ReconstructionError("Unable to write reconstruction cache")
            self._publish_result(status=error.status, error=str(error))
            raise error from exc
        states = _reconstructed_states(parsed, input_data, model=self.model, cache_hit=False, telemetry=self.last_telemetry, provider=self.provider_name)
        self._publish_result(states, status="success")
        return states


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_MAX_INPUT_TOKENS",
    "DEFAULT_STREAMING_CHUNK_TOKENS",
    "DEFAULT_OUTPUT_RESERVE_TOKENS",
    "CANONICAL_RECONSTRUCTION_FIELDS",
    "PROMPT_SCHEMA_VERSION",
    "SINGLE_CALL_POLICY",
    "STREAMING_POLICY",
    "OpenAIGovernanceStateReconstructor",
    "ReconstructionError",
    "build_reconstruction_prompt",
    "model_visible_payload",
]
