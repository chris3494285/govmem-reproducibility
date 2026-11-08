"""OpenAI Responses transport for the governance-fault downstream runner."""

from __future__ import annotations

import time
import re
import threading
import json
from typing import Any

from pydantic import BaseModel

from eval.reconstruction_provider import (
    ProviderResponse,
    ProviderTelemetry,
    ProviderTransportError,
    _usage_values,
)


DEFAULT_OPENAI_MODEL = "gpt-5.6-sol"


class DownstreamAnswer(BaseModel):
    """The complete structured output contract for downstream inference."""

    answer: str


class OpenAIDownstreamProvider:
    """One-request OpenAI Responses adapter for RQ1 downstream answers."""

    provider_name = "openai"

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        temperature: float = 0,
        reasoning_effort: str = "none",
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self._external_client = client
        self._owned_http_client: Any | None = None
        self._client_owned = client is None
        self._thread_local = threading.local()
        self._owned_clients: list[Any] = []
        self._owned_http_clients: list[Any] = []
        self._clients_lock = threading.Lock()
        self._closed = False
        self.downstream_transport = "legacy_httpx" if self._client_owned else "external_client"
        self.model = model or DEFAULT_OPENAI_MODEL
        self.requested_temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        # Responses reasoning requests do not use temperature.  This is an
        # explicit effective value, rather than a silent temperature=0 claim.
        self.effective_temperature = None if reasoning_effort != "none" else temperature
        self.effective_reasoning_effort = reasoning_effort

    def _client(self) -> Any:
        if self._external_client is not None:
            return self._external_client
        client = getattr(self._thread_local, "client", None)
        if client is not None:
            return client
        if self._closed:
            raise RuntimeError("OpenAI provider is closed")
        if self._external_client is None:
            import httpx
            from openai import OpenAI

            http_client = httpx.Client()
            client = OpenAI(http_client=http_client)
            self._thread_local.client = client
            with self._clients_lock:
                self._owned_clients.append(client)
                self._owned_http_clients.append(http_client)
                if self._owned_http_client is None:
                    self._owned_http_client = http_client
            return client
        return self._external_client

    def close(self) -> None:
        """Close only the HTTP client owned by this provider."""
        if self._closed:
            return
        self._closed = True
        if not self._client_owned:
            return
        with self._clients_lock:
            clients = list(self._owned_clients)
            http_clients = list(self._owned_http_clients)
            self._owned_clients.clear()
            self._owned_http_clients.clear()
            self._owned_http_client = None
        for owned in http_clients:
            close = getattr(owned, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _deepest_cause(exc: BaseException) -> BaseException | None:
        cause = exc
        seen: set[int] = set()
        deepest: BaseException | None = None
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            deepest = cause
            cause = cause.__cause__ or cause.__context__
        return deepest if deepest is not exc else None

    @staticmethod
    def _safe_message(exc: BaseException) -> str:
        return re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", str(exc))

    @staticmethod
    def _output_text(response: Any) -> str | None:
        if isinstance(response, dict):
            text = response.get("output_text")
            if isinstance(text, str):
                return text
            output = response.get("output")
        else:
            text = getattr(response, "output_text", None)
            if isinstance(text, str):
                return text
            output = getattr(response, "output", None)
        parts: list[str] = []
        if isinstance(output, list):
            for item in output:
                content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                if not isinstance(content, list):
                    continue
                for block in content:
                    value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                    if isinstance(value, str):
                        parts.append(value)
        return "".join(parts) or None

    @staticmethod
    def _usage_details(response: Any) -> dict[str, int | None]:
        usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
        input_tokens, output_tokens, total_tokens = _usage_values(usage)
        return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}

    @staticmethod
    def _field(response: Any, name: str, default: Any = None) -> Any:
        return response.get(name, default) if isinstance(response, dict) else getattr(response, name, default)

    def request_payload(self, prompt: str, *, max_tokens: int | None = None) -> dict[str, Any]:
        """Build the exact Responses request used by sync and Batch modes."""
        request: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": prompt}],
            "text": self._structured_text_config(),
            "reasoning": {"effort": self.reasoning_effort},
        }
        if self.effective_temperature is not None:
            request["temperature"] = self.requested_temperature
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if effective_max_tokens is not None:
            request["max_output_tokens"] = effective_max_tokens
        return request

    def parse_raw_response(self, response: Any) -> ProviderResponse:
        """Parse a raw Responses object/dict after retaining its transport state."""
        status = self._field(response, "status")
        incomplete_details = self._field(response, "incomplete_details")
        if hasattr(incomplete_details, "model_dump"):
            incomplete_details = incomplete_details.model_dump()
        raw_output_text = self._output_text(response)
        reason = incomplete_details.get("reason") if isinstance(incomplete_details, dict) else None
        if reason in {"max_output_tokens", "max_tokens"}:
            raise ProviderTransportError(
                "OpenAI response ended at the output-token limit",
                provider=self.provider_name,
                model=self.model,
                details={
                    "provider_exception": "TokenLimitTruncation",
                    "error_category": "token_limit_truncation",
                    "token_limit_truncation": True,
                    "response_status": status,
                    "incomplete_details": incomplete_details,
                    "raw_output_text": raw_output_text,
                    **self._usage_details(response),
                },
            )
        if status == "incomplete":
            raise ProviderTransportError(
                "OpenAI response was incomplete",
                provider=self.provider_name,
                model=self.model,
                details={"provider_exception": "IncompleteResponse", "response_status": status, "incomplete_details": incomplete_details, "raw_output_text": raw_output_text, **self._usage_details(response)},
            )
        if not raw_output_text:
            raise ValueError("OpenAI downstream response did not contain output text")
        parsed = DownstreamAnswer.model_validate(json.loads(raw_output_text)).model_dump()
        return ProviderResponse(parsed=parsed, telemetry=ProviderTelemetry(self.provider_name, self.model, *self._usage_tuple(response), request_id=str(self._field(response, "id") or self._field(response, "request_id")) if (self._field(response, "id") or self._field(response, "request_id")) is not None else None, cached_input_tokens=self._cached_input_tokens(response)), raw=response)

    @staticmethod
    def _usage_tuple(response: Any) -> tuple[int | None, int | None, int | None,]:
        return _usage_values(response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None))

    @staticmethod
    def _cached_input_tokens(response: Any) -> int | None:
        usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
        details = usage.get("input_tokens_details") if isinstance(usage, dict) else getattr(usage, "input_tokens_details", None)
        value = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
        return int(value) if value is not None else None

    @staticmethod
    def _structured_text_config() -> dict[str, Any]:
        schema = DownstreamAnswer.model_json_schema()
        # OpenAI strict structured outputs require every object schema to
        # explicitly forbid undeclared properties.  Pydantic's default model
        # schema does not emit this keyword unless configured to do so.
        def close_objects(value: Any) -> Any:
            if isinstance(value, dict):
                result = {key: close_objects(item) for key, item in value.items()}
                if result.get("type") == "object" or "properties" in result:
                    result.setdefault("additionalProperties", False)
                return result
            if isinstance(value, list):
                return [close_objects(item) for item in value]
            return value

        schema = close_objects(schema)
        return {"format": {"type": "json_schema", "name": "DownstreamAnswer", "strict": True, "schema": schema}}

    def request_downstream(self, prompt: str, *, max_tokens: int | None = None) -> ProviderResponse:
        started = time.perf_counter()
        request = self.request_payload(prompt, max_tokens=max_tokens)
        try:
            responses = self._client().responses
            if callable(getattr(responses, "create", None)):
                response = responses.create(**request)
                parsed_response = self.parse_raw_response(response)
                parsed = parsed_response.parsed
            else:
                # Compatibility for older injected test doubles.  Production
                # uses responses.create above so raw response state survives
                # semantic validation failures.
                response = responses.parse(text_format=DownstreamAnswer, **{key: value for key, value in request.items() if key != "text"})
                parsed = getattr(response, "output_parsed", None)
                if hasattr(parsed, "model_dump"):
                    parsed = parsed.model_dump()
                elif hasattr(parsed, "dict"):
                    parsed = parsed.dict()
                if not isinstance(parsed, dict) or set(parsed) != {"answer"} or not isinstance(parsed["answer"], str):
                    raise ValueError("OpenAI downstream response did not match the answer schema")
        except Exception as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            request_id = getattr(exc, "request_id", None)
            if isinstance(exc, ProviderTransportError):
                raise
            details = {
                "provider_exception": type(exc).__name__,
                "http_status": getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None),
                "api_latency_ms": elapsed,
                "request_id": str(request_id) if request_id is not None else None,
            }
            cause = self._deepest_cause(exc)
            if cause is not None:
                details["provider_exception_cause"] = type(cause).__name__
                details["provider_exception_cause_message"] = self._safe_message(cause)
            category = "structural_invalid" if isinstance(exc, ValueError) else None
            if category:
                details["error_category"] = category
            raise ProviderTransportError(
                self._safe_message(exc), provider=self.provider_name, model=self.model, details=details
            ) from exc

        elapsed = round((time.perf_counter() - started) * 1000, 3)
        input_tokens, output_tokens, total_tokens = _usage_values(getattr(response, "usage", None))
        request_id = getattr(response, "_request_id", None) or getattr(response, "request_id", None)
        request_id = request_id or getattr(response, "id", None)
        telemetry = ProviderTelemetry(
            self.provider_name,
            self.model,
            input_tokens,
            output_tokens,
            total_tokens,
            elapsed,
            str(request_id) if request_id is not None else None,
            cached_input_tokens=self._cached_input_tokens(response),
        )
        return ProviderResponse(parsed=parsed, telemetry=telemetry, raw=response)


__all__ = ["DEFAULT_OPENAI_MODEL", "DownstreamAnswer", "OpenAIDownstreamProvider"]
