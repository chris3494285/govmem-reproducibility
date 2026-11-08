"""Moonshot Kimi OpenAI-compatible Chat Completions transport."""

from __future__ import annotations

import os
import threading
from typing import Any

from .qwen_provider import QwenDownstreamProvider
from ..reconstruction_provider import ProviderResponse, ProviderTransportError

DEFAULT_KIMI_MODEL = "kimi-k2.6"
DEFAULT_KIMI_BASE_URL = "https://api.moonshot.ai/v1"
KIMI_API_KEY_ENV = "KIMI_API_KEY"
KIMI_THINKING_PROVIDER_DEFAULT = "provider_default"
KIMI_THINKING_DISABLED = "disabled"


def kimi_temperature_for_thinking(thinking: str) -> float:
    """Return Kimi's model-required temperature for an explicit mode."""
    if thinking == KIMI_THINKING_PROVIDER_DEFAULT:
        return 1.0
    if thinking == KIMI_THINKING_DISABLED:
        return 0.6
    raise ValueError(f"Unsupported Kimi thinking mode: {thinking}")


class KimiDownstreamProvider(QwenDownstreamProvider):
    """Reuse the OpenAI-compatible JSON/usage logic with Kimi credentials."""

    provider_name = "kimi"

    def __init__(self, *, client: Any | None = None, model: str = DEFAULT_KIMI_MODEL, base_url: str | None = None, temperature: float = 1.0, top_p: float = 0.95, max_tokens: int | None = None, thinking: str = KIMI_THINKING_PROVIDER_DEFAULT) -> None:
        resolved_temperature = kimi_temperature_for_thinking(thinking)
        # Kimi K2.6 requires 0.6 when thinking is disabled; retain the
        # established 1.0 provider-default behavior.  The mode, rather than
        # any condition metadata, determines this request setting.
        super().__init__(client=client, model=model, base_url=base_url or os.environ.get("KIMI_BASE_URL", DEFAULT_KIMI_BASE_URL), temperature=resolved_temperature, max_tokens=max_tokens)
        self.top_p = top_p
        self.thinking = thinking

    def request_payload(self, prompt: str, *, max_tokens: int | None = None) -> dict[str, Any]:
        request = super().request_payload(prompt, max_tokens=max_tokens)
        request["top_p"] = self.top_p
        if self.thinking == KIMI_THINKING_DISABLED:
            # Moonshot documents this Kimi-specific extension for the
            # OpenAI-compatible API.  It is carried in extra_body by the
            # OpenAI SDK and is deliberately absent in provider-default mode.
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        return request

    def request_downstream(self, prompt: str, *, max_tokens: int | None = None) -> ProviderResponse:
        try:
            return super().request_downstream(prompt, max_tokens=max_tokens)
        except ProviderTransportError as exc:
            if exc.details.get("error_category") == "token_limit_truncation":
                raise ProviderTransportError("Kimi response reached max output tokens", provider=self.provider_name, model=self.model, details=exc.details) from exc
            if exc.details.get("error_category") == "structural_invalid":
                raise ProviderTransportError("Kimi downstream response was malformed JSON", provider=self.provider_name, model=self.model, details=exc.details) from exc
            raise

    def _client(self) -> Any:
        if self._external_client is not None:
            return self._external_client
        if self._closed:
            raise RuntimeError("Kimi provider is closed")
        client = getattr(self._local, "client", None)
        if client is not None:
            return client
        api_key = os.environ.get(KIMI_API_KEY_ENV)
        if not api_key:
            raise ProviderTransportError("KIMI_API_KEY is required for Kimi final-main inference", provider=self.provider_name, model=self.model, details={"provider_exception": "ConfigurationError", "retryable": False})
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=self.base_url)
        except ImportError as exc:  # pragma: no cover
            raise ProviderTransportError("The openai package is required for Kimi inference", provider=self.provider_name, model=self.model, details={"provider_exception": "ImportError", "retryable": False}) from exc
        self._local.client = client
        with self._lock:
            self._owned_clients.append(client)
        return client


__all__ = ["DEFAULT_KIMI_MODEL", "DEFAULT_KIMI_BASE_URL", "KIMI_API_KEY_ENV", "KIMI_THINKING_PROVIDER_DEFAULT", "KIMI_THINKING_DISABLED", "kimi_temperature_for_thinking", "KimiDownstreamProvider"]
