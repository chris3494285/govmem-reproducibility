"""xAI Grok OpenAI-compatible Chat Completions transport."""

from __future__ import annotations

import os
from typing import Any

from .qwen_provider import QwenDownstreamProvider
from ..reconstruction_provider import ProviderResponse, ProviderTransportError

DEFAULT_GROK_MODEL = "grok-4.3"
DEFAULT_GROK_BASE_URL = "https://api.x.ai/v1"
XAI_API_KEY_ENV = "XAI_API_KEY"


class GrokDownstreamProvider(QwenDownstreamProvider):
    """Reuse OpenAI-compatible parsing/telemetry with xAI configuration."""

    provider_name = "xai"

    def __init__(self, *, client: Any | None = None, model: str = DEFAULT_GROK_MODEL, base_url: str | None = None, reasoning_effort: str = "none", temperature: float | None = None, max_tokens: int | None = None) -> None:
        if reasoning_effort != "none":
            raise ValueError("Grok final-main requires reasoning_effort='none'")
        super().__init__(client=client, model=model, base_url=base_url or os.environ.get("XAI_BASE_URL", DEFAULT_GROK_BASE_URL), temperature=temperature if temperature is not None else 0, max_tokens=max_tokens)
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def request_payload(self, prompt: str, *, max_tokens: int | None = None) -> dict[str, Any]:
        request = super().request_payload(prompt, max_tokens=max_tokens)
        request["reasoning_effort"] = self.reasoning_effort
        if self.temperature is None:
            request.pop("temperature", None)
        return request

    def request_downstream(self, prompt: str, *, max_tokens: int | None = None) -> ProviderResponse:
        try:
            return super().request_downstream(prompt, max_tokens=max_tokens)
        except ProviderTransportError as exc:
            category = exc.details.get("error_category")
            if category == "token_limit_truncation":
                raise ProviderTransportError("Grok response reached max output tokens", provider=self.provider_name, model=self.model, details=exc.details) from exc
            if category == "structural_invalid":
                raise ProviderTransportError("Grok downstream response was malformed JSON", provider=self.provider_name, model=self.model, details=exc.details) from exc
            raise ProviderTransportError(str(exc), provider=self.provider_name, model=self.model, details=exc.details) from exc

    def _client(self) -> Any:
        if self._external_client is not None:
            return self._external_client
        if self._closed:
            raise RuntimeError("Grok provider is closed")
        client = getattr(self._local, "client", None)
        if client is not None:
            return client
        api_key = os.environ.get(XAI_API_KEY_ENV)
        if not api_key:
            raise ProviderTransportError("XAI_API_KEY is required for Grok final-main inference", provider=self.provider_name, model=self.model, details={"provider_exception": "ConfigurationError", "retryable": False})
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=self.base_url)
        except ImportError as exc:  # pragma: no cover
            raise ProviderTransportError("The openai package is required for Grok inference", provider=self.provider_name, model=self.model, details={"provider_exception": "ImportError", "retryable": False}) from exc
        self._local.client = client
        with self._lock:
            self._owned_clients.append(client)
        return client


__all__ = ["DEFAULT_GROK_MODEL", "DEFAULT_GROK_BASE_URL", "XAI_API_KEY_ENV", "GrokDownstreamProvider"]
