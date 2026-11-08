"""Alibaba Cloud Model Studio OpenAI-compatible Qwen transport."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from ..reconstruction_provider import ProviderResponse, ProviderTelemetry, ProviderTransportError

DEFAULT_QWEN_MODEL = "qwen3.7-plus-2026-05-26"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_API_KEY_ENV = "DASHSCOPE_API_KEY"


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _message_content(response: Any) -> str:
    choices = _field(response, "choices") or []
    if not choices:
        raise ValueError("Qwen response did not contain choices")
    message = _field(choices[0], "message")
    content = _field(message, "content")
    if not isinstance(content, str) or not content:
        raise ValueError("Qwen response did not contain message content")
    return content


def _usage(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = _field(response, "usage")
    if usage is None:
        return None, None, None
    input_tokens = _field(usage, "prompt_tokens")
    output_tokens = _field(usage, "completion_tokens")
    total_tokens = _field(usage, "total_tokens")
    return tuple(int(value) if value is not None else None for value in (input_tokens, output_tokens, total_tokens))


def _thoughts_tokens(response: Any) -> int | None:
    usage = _field(response, "usage")
    details = _field(usage, "completion_tokens_details")
    value = _field(details, "reasoning_tokens") or _field(details, "thinking_tokens")
    return int(value) if value is not None else None


def _request_id(response: Any) -> str | None:
    value = _field(response, "id") or _field(response, "request_id")
    return str(value) if value is not None else None


def _finish_reason(response: Any) -> str | None:
    choices = _field(response, "choices") or []
    value = _field(choices[0], "finish_reason") if choices else None
    return str(value) if value is not None else None


def _error_details(exc: Exception) -> dict[str, Any]:
    response = _field(exc, "response")
    status = _field(exc, "status_code") or _field(response, "status_code")
    if status is None:
        status = _field(exc, "http_status")
    return {"provider_exception": type(exc).__name__, "http_status": status, "provider_message": str(exc)}


class QwenDownstreamProvider:
    """OpenAI-compatible Chat Completions adapter for final-main Qwen runs."""

    provider_name = "qwen"

    def __init__(self, *, client: Any | None = None, model: str = DEFAULT_QWEN_MODEL, base_url: str | None = None, temperature: float = 0, max_tokens: int | None = None) -> None:
        self.client = client
        self._external_client = client
        self._local = threading.local()
        self._owned_clients: list[Any] = []
        self._lock = threading.Lock()
        self._closed = False
        self.model = model
        self.base_url = base_url or os.environ.get("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _client(self) -> Any:
        if self._external_client is not None:
            return self._external_client
        if self._closed:
            raise RuntimeError("Qwen provider is closed")
        client = getattr(self._local, "client", None)
        if client is not None:
            return client
        api_key = os.environ.get(QWEN_API_KEY_ENV)
        if not api_key:
            raise ProviderTransportError("DASHSCOPE_API_KEY is required for Qwen final-main inference", provider=self.provider_name, model=self.model, details={"provider_exception": "ConfigurationError", "retryable": False})
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=self.base_url)
        except ImportError as exc:  # pragma: no cover
            raise ProviderTransportError("The openai package is required for Qwen inference", provider=self.provider_name, model=self.model, details={"provider_exception": "ImportError", "retryable": False}) from exc
        self._local.client = client
        with self._lock:
            self._owned_clients.append(client)
        return client

    def request_payload(self, prompt: str, *, max_tokens: int | None = None) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
        }
        effective_max = self.max_tokens if max_tokens is None else max_tokens
        if effective_max is not None:
            request["max_tokens"] = effective_max
        return request

    def request_downstream(self, prompt: str, *, max_tokens: int | None = None) -> ProviderResponse:
        started = time.perf_counter()
        try:
            response = self._client().chat.completions.create(**self.request_payload(prompt, max_tokens=max_tokens))
            finish_reason = _finish_reason(response)
            if finish_reason and finish_reason.lower() in {"length", "max_tokens", "max_output_tokens"}:
                choices = _field(response, "choices") or []
                message = _field(choices[0], "message") if choices else None
                content = _field(message, "content") or ""
                input_tokens, output_tokens, total_tokens = _usage(response)
                raise ProviderTransportError("Qwen response reached max output tokens", provider=self.provider_name, model=self.model, details={"provider_exception": "TokenLimitTruncation", "error_category": "token_limit_truncation", "token_limit_truncation": True, "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens, "finish_reason": finish_reason, "raw_output_text": content, "retryable": False})
            content = _message_content(response)
            parsed = json.loads(content)
            if not isinstance(parsed, dict) or set(parsed) != {"answer"} or not isinstance(parsed["answer"], str):
                raise ValueError("Qwen output did not match the answer schema")
        except ProviderTransportError:
            raise
        except (json.JSONDecodeError, ValueError) as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            raise ProviderTransportError("Qwen downstream response was malformed JSON", provider=self.provider_name, model=self.model, details={"provider_exception": type(exc).__name__, "error_category": "structural_invalid", "api_latency_ms": elapsed}) from exc
        except Exception as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            details = _error_details(exc)
            details["api_latency_ms"] = elapsed
            raise ProviderTransportError(str(exc), provider=self.provider_name, model=self.model, details=details) from exc
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        input_tokens, output_tokens, total_tokens = _usage(response)
        return ProviderResponse(parsed=parsed, telemetry=ProviderTelemetry(self.provider_name, self.model, input_tokens, output_tokens, total_tokens, elapsed, _request_id(response), _finish_reason(response), thoughts_tokens=_thoughts_tokens(response)), raw=response)

    def close(self) -> None:
        if self._closed or self._external_client is not None:
            self._closed = True
            return
        self._closed = True
        with self._lock:
            clients, self._owned_clients = self._owned_clients, []
        for client in clients:
            close = getattr(client, "close", None)
            if callable(close):
                close()


__all__ = ["DEFAULT_QWEN_MODEL", "DEFAULT_QWEN_BASE_URL", "QWEN_API_KEY_ENV", "QwenDownstreamProvider"]
