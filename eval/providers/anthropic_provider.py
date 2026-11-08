"""Native Anthropic Messages transport for governance reconstruction.

This module owns only Anthropic SDK transport and response/usage extraction.
Prompt construction, chunking, carry-forward, canonical parsing, and semantic
validation remain in the shared reconstruction implementation.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from pydantic import BaseModel

from ..reconstruction_provider import (
    ProviderResponse,
    ProviderTelemetry,
    ProviderTransportError,
)


DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
_ANTHROPIC_MODELS_WITHOUT_TEMPERATURE = frozenset({"claude-sonnet-5"})
_UNSUPPORTED_SCHEMA_CONSTRAINTS = frozenset({
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
})


def anthropic_supports_temperature(model: str) -> bool:
    """Return whether the selected Anthropic model accepts ``temperature``."""

    return model not in _ANTHROPIC_MODELS_WITHOUT_TEMPERATURE


def anthropic_transport_schema(schema: Any) -> Any:
    """Return an Anthropic-compatible copy of a JSON Schema.

    Anthropic's structured-output schema dialect does not accept several JSON
    Schema validation constraints and requires object schemas to be strict.
    This transformation only changes the transport copy; it does not alter
    the canonical schema used for post-response Pydantic/semantic validation.
    """
    if isinstance(schema, dict):
        transformed = {
            key: anthropic_transport_schema(value)
            for key, value in schema.items()
            if key not in _UNSUPPORTED_SCHEMA_CONSTRAINTS
        }
        if transformed.get("type") == "object":
            # Preserve an already stricter schema (false or a schema-valued
            # additionalProperties).  Replace only an absent or permissive
            # true value with Anthropic's required strict setting.
            if "additionalProperties" not in transformed or transformed["additionalProperties"] is True:
                transformed["additionalProperties"] = False
        return transformed
    if isinstance(schema, list):
        return [anthropic_transport_schema(value) for value in schema]
    return schema


def _redact(value: Any) -> str:
    text = str(value)
    # Keep provider diagnostics useful without retaining credential-shaped
    # values from exception text.
    text = re.sub(r"sk-ant-[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return text


def _error_details(exc: Exception) -> dict[str, Any]:
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    if not isinstance(body, dict):
        body = {}
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    request_id = (
        getattr(exc, "request_id", None)
        or getattr(response, "request_id", None)
        or (headers.get("request-id") if hasattr(headers, "get") else None)
    )
    message = body.get("message") or getattr(exc, "message", None) or str(exc)
    return {
        "provider_exception": type(exc).__name__,
        "http_status": getattr(exc, "status_code", None) or getattr(response, "status_code", None),
        "provider_code": body.get("code") or getattr(exc, "code", None),
        "provider_type": body.get("type") or getattr(exc, "type", None),
        "provider_message": _redact(message),
        "request_id": _redact(request_id) if request_id is not None else None,
    }


def _usage(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None

    def value(name: str) -> int | None:
        raw = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        return int(raw) if raw is not None else None

    input_tokens = value("input_tokens")
    output_tokens = value("output_tokens")
    total_tokens = value("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


_RAW_DIAGNOSTIC_CHARS = 256


def _block_type(block: Any) -> str | None:
    value = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
    return str(value) if value is not None else None


def _block_text(block: Any) -> str | None:
    value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
    return value if isinstance(value, str) else None


def _extract_text(response: Any) -> str | None:
    """Return joined text blocks, or ``None`` when no text is present.

    Never raises: callers decide how to fail so that transport diagnostics are
    preserved for both missing-text and malformed-JSON outcomes.
    """
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        return None
    parts = [text for text in (_block_text(block) for block in content) if text is not None]
    return "".join(parts) if parts else None


def _bounded_raw_diagnostic(text: str) -> dict[str, Any]:
    """Safe, bounded shape diagnostic for a raw structured response.

    The value is redacted and length-bounded so it cannot retain credentials
    and cannot grow without limit.  Brace/quote counts plus the exact tail make
    truncation visible without repairing or re-interpreting the payload.
    """
    safe = _redact(text)
    return {
        "raw_text_preview": safe[:_RAW_DIAGNOSTIC_CHARS],
        "raw_text_tail": safe[-_RAW_DIAGNOSTIC_CHARS:],
        "raw_text_preview_truncated": len(safe) > _RAW_DIAGNOSTIC_CHARS,
        "raw_text_open_braces": safe.count("{"),
        "raw_text_close_braces": safe.count("}"),
        "raw_text_quote_count": safe.count('"'),
        "raw_text_last_char": safe[-1] if safe else "",
    }


def _response_diagnostics(
    response: Any, *, raw_text: str | None = None, json_parse_ok: bool | None = None
) -> dict[str, Any]:
    """Collect safe transport metadata without interpreting the payload."""
    content = getattr(response, "content", None)
    content_list_present = isinstance(content, list)
    blocks = content if content_list_present else []
    block_types = [_block_type(block) for block in blocks]
    text_values = [_block_text(block) for block in blocks]
    text_blocks = [value for value in text_values if value is not None]
    resolved = raw_text if raw_text is not None else ("".join(text_blocks) if text_blocks else None)
    input_tokens, output_tokens, total_tokens = _usage(response)
    request_id = getattr(response, "_request_id", None) or getattr(response, "request_id", None)
    stop_reason = getattr(response, "stop_reason", None)
    non_text_types = [value for value in block_types if value is not None and value != "text"]
    diagnostics: dict[str, Any] = {
        "stop_reason": str(stop_reason) if stop_reason is not None else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "request_id": _redact(request_id) if request_id is not None else None,
        "content_list_present": content_list_present,
        "content_block_types": block_types,
        "content_block_count": len(blocks),
        "text_block_count": len(text_blocks),
        "non_text_block_types": non_text_types,
        "has_non_text_content": bool(non_text_types),
        "has_raw_text": bool(resolved),
        "raw_text_char_count": len(resolved) if resolved is not None else 0,
        "json_parse_ok": json_parse_ok,
    }
    if resolved:
        diagnostics.update(_bounded_raw_diagnostic(resolved))
    return diagnostics


def _text_content(response: Any) -> str:
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        raise ValueError("Anthropic response did not contain a content list")
    text_parts = []
    for block in content:
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if isinstance(text, str):
            text_parts.append(text)
    if not text_parts:
        raise ValueError("Anthropic response did not contain structured JSON text")
    return "".join(text_parts)


class AnthropicMessagesProvider:
    """Anthropic native Messages API adapter implementing the shared protocol."""

    provider_name = "anthropic"

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> None:
        self.client = client
        self.model = model or os.environ.get("ANTHROPIC_RECONSTRUCTION_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.max_tokens = max_tokens
        # None preserves the historical reconstruction request shape.  The
        # downstream method supplies the RQ1 default of zero explicitly.
        self.temperature = temperature
        self.requested_temperature = temperature
        self.effective_temperature = (
            temperature if temperature is not None and anthropic_supports_temperature(self.model) else None
        )

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderTransportError(
                "ANTHROPIC_API_KEY is required for Anthropic reconstruction",
                provider=self.provider_name,
                model=self.model,
                details={"provider_exception": "ConfigurationError", "provider_message": "ANTHROPIC_API_KEY is required"},
            )
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderTransportError(
                "The anthropic package is required for Anthropic reconstruction",
                provider=self.provider_name,
                model=self.model,
                details={"provider_exception": "ImportError", "provider_message": "anthropic package is required"},
            ) from exc
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self.client

    def _request_structured(self, *, prompt: str, schema: Any, temperature: float | None = None, max_tokens: int | None = None) -> ProviderResponse:
        canonical_schema = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema
        schema_json = anthropic_transport_schema(canonical_schema)
        started = time.perf_counter()
        response: Any = None
        raw_text: str | None = None
        try:
            # Anthropic native structured output: the schema is supplied through
            # the Messages API output_config, while the shared prompt is kept
            # byte-for-byte provider-neutral.
            request: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "output_config": {"format": {"type": "json_schema", "schema": schema_json}},
            }
            if temperature is not None:
                request["temperature"] = temperature
            response = self._client().messages.create(**request)
            # Capture safe transport metadata before extracting text or parsing
            # so a missing-text response does not discard stop_reason/usage.
            raw_text = _extract_text(response)
            stop_reason = getattr(response, "stop_reason", None)
            if str(stop_reason).lower() in {"max_tokens", "max_output_tokens"}:
                diagnostics = _response_diagnostics(response, raw_text=raw_text, json_parse_ok=False)
                raise ProviderTransportError(
                    "Anthropic response reached the max token limit",
                    provider=self.provider_name,
                    model=self.model,
                    details={
                        "provider_exception": "TokenLimitTruncation",
                        "error_category": "token_limit_truncation",
                        "token_limit_truncation": True,
                        "response_status": getattr(response, "status", None),
                        "incomplete_details": {"reason": "max_tokens"},
                        "raw_output_text": raw_text,
                        **diagnostics,
                    },
                )
            if raw_text is None:
                diagnostics = _response_diagnostics(response, json_parse_ok=False)
                raise ProviderTransportError(
                    "Anthropic response did not contain structured JSON text",
                    provider=self.provider_name,
                    model=self.model,
                    details={
                        "provider_exception": "ValueError",
                        "error_category": "structural_invalid",
                        "provider_message": "Anthropic structured output did not contain text",
                        **diagnostics,
                    },
                )
            parsed = json.loads(raw_text)
        except ProviderTransportError:
            raise
        except (json.JSONDecodeError, ValueError) as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            raise ProviderTransportError(
                "Anthropic returned malformed structured JSON",
                provider=self.provider_name,
                model=self.model,
                details={
                    "provider_exception": type(exc).__name__,
                    "error_category": "structural_invalid",
                    "provider_message": "Anthropic structured output was not valid JSON",
                    "api_latency_ms": elapsed,
                    **_response_diagnostics(response, raw_text=raw_text, json_parse_ok=False),
                },
            ) from exc
        except Exception as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            details = _error_details(exc)
            details["api_latency_ms"] = elapsed
            raise ProviderTransportError(
                _redact(details["provider_message"]),
                provider=self.provider_name,
                model=self.model,
                details=details,
            ) from exc

        elapsed = round((time.perf_counter() - started) * 1000, 3)
        input_tokens, output_tokens, total_tokens = _usage(response)
        request_id = getattr(response, "_request_id", None) or getattr(response, "request_id", None)
        stop_reason = getattr(response, "stop_reason", None)
        telemetry = ProviderTelemetry(
            self.provider_name,
            self.model,
            input_tokens,
            output_tokens,
            total_tokens,
            elapsed,
            str(request_id) if request_id is not None else None,
            str(stop_reason) if stop_reason is not None else None,
        )
        return ProviderResponse(parsed=parsed, telemetry=telemetry, raw=response)

    def request_structured_reconstruction(
        self, *, prompt: str, schema: Any, request_metadata: dict[str, Any]
    ) -> ProviderResponse:
        return self._request_structured(prompt=prompt, schema=schema, temperature=self.effective_temperature)

    def request_downstream(self, prompt: str, *, max_tokens: int | None = None) -> ProviderResponse:
        """Request the provider-neutral RQ1 answer envelope."""

        class DownstreamAnswer(BaseModel):
            answer: str

        requested_temperature = 0 if self.temperature is None else self.temperature
        self.requested_temperature = requested_temperature
        self.effective_temperature = (
            requested_temperature if anthropic_supports_temperature(self.model) else None
        )
        response = self._request_structured(
            prompt=prompt,
            schema=DownstreamAnswer,
            temperature=self.effective_temperature,
            max_tokens=max_tokens,
        )
        if (
            not isinstance(response.parsed, dict)
            or set(response.parsed) != {"answer"}
            or not isinstance(response.parsed["answer"], str)
        ):
            raise ProviderTransportError(
                "Anthropic downstream response did not match the answer schema",
                provider=self.provider_name,
                model=self.model,
                details={"provider_exception": "ValueError", "error_category": "structural_invalid"},
            )
        return response


__all__ = [
    "AnthropicMessagesProvider",
    "DEFAULT_ANTHROPIC_MODEL",
    "anthropic_supports_temperature",
    "anthropic_transport_schema",
]
