"""DeepSeek OpenAI-compatible Chat Completions transport.

Only API transport and JSON/usage extraction live here.  Reconstruction
prompts, chunking, carry-over, canonical parsing, and semantic validation are
owned by the shared reconstruction implementation.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

from ..reconstruction_provider import ProviderResponse, ProviderTelemetry, ProviderTransportError


DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def deepseek_transport_prompt(prompt: str, schema: Any) -> str:
    """Add only the JSON-mode transport contract to the shared prompt."""
    canonical_schema = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema
    return (
        prompt
        + "\n\nTransport output requirement: Return the reconstruction as valid JSON "
        "matching the canonical output contract below. Do not add commentary.\n"
        + json.dumps(canonical_schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _redact(value: Any) -> str:
    text = str(value)
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
        or (headers.get("x-request-id") if hasattr(headers, "get") else None)
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

    input_tokens = value("prompt_tokens")
    output_tokens = value("completion_tokens")
    total_tokens = value("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def _message_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise ValueError("DeepSeek response did not contain a choice")
    message = getattr(choices[0], "message", None)
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if not isinstance(content, str) or not content:
        raise ValueError("DeepSeek response did not contain JSON content")
    return content


class DeepSeekChatCompletionsProvider:
    """DeepSeek provider using the official OpenAI-compatible API surface."""

    provider_name = "deepseek"

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        base_url: str = DEEPSEEK_BASE_URL,
    ) -> None:
        self.client = client
        self.model = model or os.environ.get("DEEPSEEK_RECONSTRUCTION_MODEL", DEFAULT_DEEPSEEK_MODEL)
        self.base_url = base_url

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ProviderTransportError(
                "DEEPSEEK_API_KEY is required for DeepSeek reconstruction",
                provider=self.provider_name,
                model=self.model,
                details={"provider_exception": "ConfigurationError", "provider_message": "DEEPSEEK_API_KEY is required"},
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderTransportError(
                "The openai package is required for DeepSeek reconstruction",
                provider=self.provider_name,
                model=self.model,
                details={"provider_exception": "ImportError", "provider_message": "openai package is required"},
            ) from exc
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self.client

    def request_structured_reconstruction(
        self, *, prompt: str, schema: Any, request_metadata: dict[str, Any]
    ) -> ProviderResponse:
        request_prompt = deepseek_transport_prompt(prompt, schema)
        started = time.perf_counter()
        try:
            response = self._client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": request_prompt}],
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            parsed = json.loads(_message_content(response))
        except (json.JSONDecodeError, ValueError) as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            raise ProviderTransportError(
                "DeepSeek returned malformed JSON",
                provider=self.provider_name,
                model=self.model,
                details={
                    "provider_exception": type(exc).__name__,
                    "error_category": "structural_invalid",
                    "provider_message": "DeepSeek JSON output was not valid JSON",
                    "api_latency_ms": elapsed,
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
        request_id = getattr(response, "_request_id", None) or getattr(response, "request_id", None) or getattr(response, "id", None)
        stop_reason = getattr(response.choices[0], "finish_reason", None) if getattr(response, "choices", None) else None
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


class DeepSeekDownstreamProvider:
    """Ordinary DeepSeek JSON-mode inference for the governance-fault runner.

    This transport intentionally does not add the reconstruction prompt or
    canonical reconstruction schema.  It only asks for a JSON envelope with
    an ``answer`` field so the downstream response is auditable.
    """

    provider_name = "deepseek"

    def __init__(self, *, client: Any | None = None, model: str | None = None, base_url: str = DEEPSEEK_BASE_URL, temperature: float = 0, max_tokens: int | None = None) -> None:
        self.client = client
        self._external_client = client
        self._thread_local = threading.local()
        self._clients: list[Any] = []
        self._clients_lock = threading.Lock()
        self._closed = False
        self.model = model or os.environ.get("DEEPSEEK_DOWNSTREAM_MODEL", DEFAULT_DEEPSEEK_MODEL)
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _transport(self) -> DeepSeekChatCompletionsProvider:
        return DeepSeekChatCompletionsProvider(client=self._client(), model=self.model, base_url=self.base_url)

    def _client(self) -> Any:
        """Return one client per worker thread, or the injected client."""
        if self._external_client is not None:
            return self._external_client
        client = getattr(self._thread_local, "client", None)
        if client is not None:
            return client
        if self._closed:
            raise RuntimeError("DeepSeek provider is closed")
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ProviderTransportError(
                "DEEPSEEK_API_KEY is required for DeepSeek downstream inference",
                provider=self.provider_name,
                model=self.model,
                details={"provider_exception": "ConfigurationError", "provider_message": "DEEPSEEK_API_KEY is required"},
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderTransportError(
                "The openai package is required for DeepSeek downstream inference",
                provider=self.provider_name,
                model=self.model,
                details={"provider_exception": "ImportError", "provider_message": "openai package is required"},
            ) from exc
        client = OpenAI(api_key=api_key, base_url=self.base_url)
        self._thread_local.client = client
        with self._clients_lock:
            self._clients.append(client)
        return client

    def request_downstream(self, prompt: str, *, max_tokens: int | None = None) -> ProviderResponse:
        transport = self._transport()
        started = time.perf_counter()
        try:
            request = dict(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                temperature=self.temperature,
            )
            effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
            if effective_max_tokens is not None:
                request["max_tokens"] = effective_max_tokens
            response = transport._client().chat.completions.create(**request)
            parsed = json.loads(_message_content(response))
        except (json.JSONDecodeError, ValueError) as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            raise ProviderTransportError(
                "DeepSeek downstream response was malformed JSON",
                provider=self.provider_name,
                model=self.model,
                details={"provider_exception": type(exc).__name__, "error_category": "structural_invalid", "api_latency_ms": elapsed},
            ) from exc
        except Exception as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            details = _error_details(exc)
            details["api_latency_ms"] = elapsed
            raise ProviderTransportError(str(details.get("provider_message") or exc), provider=self.provider_name, model=self.model, details=details) from exc
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        input_tokens, output_tokens, total_tokens = _usage(response)
        request_id = getattr(response, "_request_id", None) or getattr(response, "request_id", None) or getattr(response, "id", None)
        stop_reason = getattr(response.choices[0], "finish_reason", None) if getattr(response, "choices", None) else None
        return ProviderResponse(
            parsed=parsed,
            telemetry=ProviderTelemetry(self.provider_name, self.model, input_tokens, output_tokens, total_tokens, elapsed, str(request_id) if request_id is not None else None, str(stop_reason) if stop_reason is not None else None),
            raw=response,
        )

    def close(self) -> None:
        """Close provider-owned worker clients; never close an injected client."""
        if self._closed:
            return
        self._closed = True
        if self._external_client is not None:
            return
        with self._clients_lock:
            clients = []
            seen_ids: set[int] = set()
            for client in self._clients:
                if id(client) not in seen_ids:
                    clients.append(client)
                    seen_ids.add(id(client))
            self._clients.clear()
        for client in clients:
            close = getattr(client, "close", None)
            if callable(close):
                close()


__all__ = [
    "DEFAULT_DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
    "DeepSeekChatCompletionsProvider",
    "DeepSeekDownstreamProvider",
    "deepseek_transport_prompt",
]
