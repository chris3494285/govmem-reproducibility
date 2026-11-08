"""Provider-neutral transport boundary for governance reconstruction.

Providers perform exactly one request.  Prompt construction, chunking,
carry-forward, normalization, and semantic validation remain in the existing
reconstruction engine for compatibility with the frozen RQ1 path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Protocol


@dataclass(frozen=True)
class ProviderTelemetry:
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    request_id: str | None = None
    stop_reason: str | None = None
    cached_input_tokens: int | None = None
    thoughts_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        usage = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        return {
            "provider": self.provider,
            "model": self.model,
            "usage": usage if any(value is not None for value in usage.values()) else None,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "request_id": self.request_id,
            "stop_reason": self.stop_reason,
            "cached_input_tokens": self.cached_input_tokens,
            "thoughts_tokens": self.thoughts_tokens,
        }


@dataclass(frozen=True)
class ProviderResponse:
    """One provider response before shared semantic normalization."""

    parsed: Any
    telemetry: ProviderTelemetry
    raw: Any = field(default=None, repr=False)

    @property
    def output_parsed(self) -> Any:
        """Compatibility shape used by the existing OpenAI engine."""
        return self.parsed

    @property
    def usage(self) -> Any:
        return getattr(self.raw, "usage", None)


class ProviderTransportError(RuntimeError):
    """Provider failure normalized at the transport boundary."""

    def __init__(self, message: str, *, provider: str, model: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.details = dict(details or {})
        self.provider_exception = self.details.get("provider_exception", type(self).__name__)
        self.status_code = self.details.get("http_status")
        self.request_id = self.details.get("request_id")


class ReconstructionProvider(Protocol):
    provider_name: str
    model: str

    def request_structured_reconstruction(
        self, *, prompt: str, schema: Any, request_metadata: dict[str, Any]
    ) -> ProviderResponse:
        """Perform exactly one transport request."""


def _usage_values(usage: Any) -> tuple[int | None, int | None, int | None]:
    def get(name: str, alias: str) -> int | None:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if value is None:
            value = usage.get(alias) if isinstance(usage, dict) else getattr(usage, alias, None)
        return int(value) if value is not None else None

    if usage is None:
        return None, None, None
    return get("input_tokens", "prompt_tokens"), get("output_tokens", "completion_tokens"), get("total_tokens", "total_tokens")


class OpenAIResponsesProvider:
    """OpenAI Responses transport adapter; no reconstruction semantics live here."""

    provider_name = "openai"

    def __init__(self, *, client: Any, model: str):
        self.client = client
        self.model = model

    def request_structured_reconstruction(self, *, prompt: str, schema: Any, request_metadata: dict[str, Any]) -> ProviderResponse:
        started = time.perf_counter()
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[{"role": "user", "content": prompt}],
                text_format=schema,
            )
        except Exception as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            raise ProviderTransportError(
                str(exc), provider=self.provider_name, model=self.model,
                details={"provider_exception": type(exc).__name__, "api_latency_ms": elapsed},
            ) from exc
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        input_tokens, output_tokens, total_tokens = _usage_values(getattr(response, "usage", None))
        request_id = getattr(response, "_request_id", None) or getattr(response, "request_id", None)
        return ProviderResponse(
            parsed=getattr(response, "output_parsed", None), raw=response,
            telemetry=ProviderTelemetry(self.provider_name, self.model, input_tokens, output_tokens, total_tokens, elapsed, request_id),
        )


class RecordingFakeProvider:
    """Deterministic test provider that records the exact transport requests."""

    def __init__(self, provider_name: str, model: str, responses: list[Any]):
        self.provider_name = provider_name
        self.model = model
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def request_structured_reconstruction(self, *, prompt: str, schema: Any, request_metadata: dict[str, Any]) -> ProviderResponse:
        self.requests.append({"prompt": prompt, "schema": schema, "request_metadata": dict(request_metadata)})
        if not self.responses:
            raise ProviderTransportError("fake provider exhausted", provider=self.provider_name, model=self.model)
        parsed = self.responses.pop(0)
        return ProviderResponse(
            parsed=parsed,
            telemetry=ProviderTelemetry(self.provider_name, self.model, latency_ms=0.0),
            raw=SimpleNamespace(usage=None),
        )


__all__ = [
    "OpenAIResponsesProvider",
    "ProviderResponse",
    "ProviderTelemetry",
    "ProviderTransportError",
    "ReconstructionProvider",
    "RecordingFakeProvider",
]
