"""Google Gemini transport for the provider-neutral final-main contract."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

from pydantic import BaseModel

from ..reconstruction_provider import ProviderResponse, ProviderTelemetry, ProviderTransportError

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_THINKING_LEVEL = "LOW"


def gemini_answer_schema() -> dict[str, Any]:
    """Exact logical schema sent through Gemini's response_schema transport."""
    return {
        "type": "OBJECT",
        "properties": {"answer": {"type": "STRING"}},
        "required": ["answer"],
    }


def gemini_request_config(*, max_output_tokens: int, temperature: float = 0) -> dict[str, Any]:
    """Build the transport config; no tools, grounding, or search are enabled."""
    return {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "response_mime_type": "application/json",
        "response_schema": gemini_answer_schema(),
        "thinking_config": {"thinking_level": GEMINI_THINKING_LEVEL},
    }


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _usage(response: Any) -> tuple[int | None, int | None, int | None, int | None]:
    usage = _field(response, "usage_metadata") or _field(response, "usage")
    if usage is None:
        return None, None, None, None
    input_tokens = _field(usage, "prompt_token_count")
    if input_tokens is None:
        input_tokens = _field(usage, "input_tokens")
    output_tokens = _field(usage, "candidates_token_count")
    if output_tokens is None:
        output_tokens = _field(usage, "output_tokens")
    total_tokens = _field(usage, "total_token_count")
    if total_tokens is None:
        total_tokens = _field(usage, "total_tokens")
    thoughts_tokens = _field(usage, "thoughts_token_count")
    if thoughts_tokens is None:
        thoughts_tokens = _field(usage, "thoughts_tokens")
    return tuple(int(value) if value is not None else None for value in (input_tokens, output_tokens, total_tokens, thoughts_tokens))


def _text(response: Any) -> str:
    text = _field(response, "text")
    if isinstance(text, str) and text:
        return text
    candidates = _field(response, "candidates") or []
    if candidates:
        content = _field(candidates[0], "content")
        parts = _field(content, "parts") or []
        text = "".join(str(_field(part, "text")) for part in parts if _field(part, "text") is not None)
        if text:
            return text
    raise ValueError("Gemini response did not contain output text")


def _finish_reason(response: Any) -> str | None:
    candidates = _field(response, "candidates") or []
    reason = _field(candidates[0], "finish_reason") if candidates else _field(response, "finish_reason")
    return str(reason) if reason is not None else None


def _response_status(response: Any) -> str | None:
    status = _field(response, "status")
    return str(status) if status is not None else None


def _incomplete_details(response: Any) -> Any:
    return _field(response, "incomplete_details") or _field(response, "incompleteDetails")


def _error_details(exc: Exception) -> dict[str, Any]:
    status = _field(exc, "status_code") or _field(_field(exc, "response"), "status_code")
    message = re.sub(r"AIza[A-Za-z0-9_-]+", "[REDACTED]", str(exc))
    return {"provider_exception": type(exc).__name__, "http_status": status, "provider_message": message}


class GeminiGenerativeProvider:
    provider_name = "gemini"

    def __init__(self, *, client: Any | None = None, model: str = DEFAULT_GEMINI_MODEL, temperature: float = 0, max_tokens: int = 256) -> None:
        self.client = client
        self._external_client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._local = threading.local()
        self._owned_clients: list[Any] = []
        self._lock = threading.Lock()
        self._closed = False

    def _client(self) -> Any:
        if self._external_client is not None:
            return self._external_client
        if self._closed:
            raise RuntimeError("Gemini provider is closed")
        client = getattr(self._local, "client", None)
        if client is not None:
            return client
        api_key = os.environ.get(GEMINI_API_KEY_ENV)
        if not api_key:
            raise ProviderTransportError("GEMINI_API_KEY is required for Gemini final-main inference", provider=self.provider_name, model=self.model, details={"provider_exception": "ConfigurationError", "retryable": False})
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
        except ImportError as exc:  # pragma: no cover
            raise ProviderTransportError("google-genai package is required for Gemini inference", provider=self.provider_name, model=self.model, details={"provider_exception": "ImportError", "retryable": False}) from exc
        self._local.client = client
        with self._lock:
            self._owned_clients.append(client)
        return client

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._external_client is not None:
            return
        with self._lock:
            clients, self._owned_clients = self._owned_clients, []
        for client in clients:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def request_downstream(self, prompt: str, *, max_tokens: int | None = None) -> ProviderResponse:
        budget = self.max_tokens if max_tokens is None else max_tokens
        started = time.perf_counter()
        config = gemini_request_config(max_output_tokens=budget, temperature=self.temperature)
        try:
            response = self._client().models.generate_content(model=self.model, contents=prompt, config=config)
            text = _text(response)
            input_tokens, output_tokens, total_tokens, thoughts_tokens = _usage(response)
            finish_reason = _finish_reason(response)
            if finish_reason and finish_reason.upper().endswith("MAX_TOKENS"):
                raise ProviderTransportError(
                    "Gemini response reached max output tokens",
                    provider=self.provider_name,
                    model=self.model,
                    details={"provider_exception": "TokenLimitTruncation", "error_category": "token_limit_truncation", "token_limit_truncation": True, "raw_output_text": text, "output_tokens": output_tokens, "input_tokens": input_tokens, "total_tokens": total_tokens, "finish_reason": finish_reason, "response_status": _response_status(response), "incomplete_details": _incomplete_details(response), "retryable": False},
                )
            parsed = json.loads(text)
            if not isinstance(parsed, dict) or set(parsed) != {"answer"} or not isinstance(parsed["answer"], str):
                raise ValueError("Gemini output did not match the answer schema")
        except ProviderTransportError:
            raise
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderTransportError("Gemini returned malformed structured JSON", provider=self.provider_name, model=self.model, details={"provider_exception": type(exc).__name__, "error_category": "structural_invalid", "retryable": False}) from exc
        except Exception as exc:
            details = _error_details(exc)
            raise ProviderTransportError(details["provider_message"], provider=self.provider_name, model=self.model, details=details) from exc
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        request_id = _field(response, "response_id") or _field(response, "request_id")
        return ProviderResponse(parsed=parsed, raw=response, telemetry=ProviderTelemetry(self.provider_name, self.model, input_tokens, output_tokens, total_tokens, elapsed, str(request_id) if request_id else None, finish_reason, thoughts_tokens=thoughts_tokens))


__all__ = ["DEFAULT_GEMINI_MODEL", "GEMINI_THINKING_LEVEL", "GeminiGenerativeProvider", "gemini_answer_schema", "gemini_request_config"]
