"""Thin Mem0 adapter; the Mem0 package remains an optional dependency."""

from __future__ import annotations

import hashlib
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping

from ._optional import call_backend, metadata_dict, require_dependency
from .base import MemoryRetrievalResult, MemorySystemAdapter, normalize_retrieval


class Mem0Adapter(MemorySystemAdapter):
    memory_system = "mem0"
    adapter_version = "mem0-oss-s1-v1"

    def __init__(self, backend: Any | None = None, *, config: Mapping[str, Any] | None = None):
        if backend is None:
            module = require_dependency("mem0", "mem0ai", "Mem0")
            memory_cls = getattr(module, "Memory", None)
            if memory_cls is None:
                raise RuntimeError("Installed mem0ai package does not expose Memory")
            backend = memory_cls.from_config(dict(config)) if config else memory_cls()
        self.backend = backend
        self.config = dict(config or {})
        self._legacy_http_client: Any | None = None
        self.mem0_embedding_transport: str | None = None
        self._configure_openai_embedding_transport()

    def _configure_openai_embedding_transport(self) -> None:
        """Use the compatibility HTTP transport only for Mem0's OpenAI embedder."""
        embedder = getattr(self.backend, "embedding_model", None)
        if embedder is None:
            return
        embedder_type = type(embedder)
        if embedder_type.__module__ != "mem0.embeddings.openai" or embedder_type.__name__ != "OpenAIEmbedding":
            return
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Mem0 OpenAI embedding transport compatibility requires httpx and openai"
            ) from exc
        embedder_config = getattr(embedder, "config", None)
        api_key = getattr(embedder_config, "api_key", None) or os.environ.get("OPENAI_API_KEY")
        base_url = (
            getattr(embedder_config, "openai_base_url", None)
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self._legacy_http_client = httpx.Client()
        embedder.client = OpenAI(api_key=api_key, base_url=base_url, http_client=self._legacy_http_client)
        self.mem0_embedding_transport = "legacy_httpx"

    @staticmethod
    def user_id_for_base(base_example_id: str) -> str:
        """Return a stable, non-overlapping Mem0 OSS user namespace."""
        digest = hashlib.sha256(str(base_example_id).encode("utf-8")).hexdigest()
        return f"govmem-s1-{digest}"

    @staticmethod
    def package_version() -> str | None:
        try:
            return version("mem0ai")
        except PackageNotFoundError:
            return None

    def reset(self) -> None:
        call_backend(self.backend, ("reset", "clear", "delete_all"))

    def add_memory(
        self,
        content: str,
        *,
        user_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        infer: bool = False,
        **kwargs: Any,
    ) -> Any:
        if infer is not False:
            raise ValueError("Mem0 system-validation insertions require infer=False")
        # Keep this call explicit: the generic fallback helper must never
        # retry without infer=False or move user_id into another location.
        return self.backend.add(
            content,
            user_id=user_id or "govmem-default",
            metadata=metadata_dict(metadata),
            infer=False,
            **kwargs,
        )

    def update_memory(self, memory_id: str, content: str | None = None, *, metadata: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        payload = content if content is not None else metadata_dict(metadata)
        return call_backend(self.backend, ("update",), memory_id, payload, **kwargs)

    def delete_memory(self, memory_id: str, **kwargs: Any) -> Any:
        return call_backend(self.backend, ("delete",), memory_id, **kwargs)

    def retrieve(self, query: str, *, user_id: str | None = None, limit: int | None = None, **kwargs: Any) -> MemoryRetrievalResult:
        # Current Mem0 OSS search requires entity identity inside filters;
        # user_id must not be passed as a top-level search keyword.
        if "filters" in kwargs:
            raise ValueError("Mem0Adapter owns search filters; pass user_id instead of filters")
        filters = {"user_id": user_id or "govmem-default"}
        if limit is None:
            payload = self.backend.search(query, filters=filters, **kwargs)
        else:
            try:
                payload = self.backend.search(query, filters=filters, top_k=limit, **kwargs)
            except TypeError:
                # Older OSS revisions may not expose a native top_k. Preserve
                # their returned order and apply only a final breadth cap.
                payload = self.backend.search(query, filters=filters, **kwargs)
        result = normalize_retrieval(payload, memory_system=self.memory_system, query=query)
        if limit is not None and len(result.memories) > limit:
            return MemoryRetrievalResult(
                memory_system=result.memory_system,
                query=result.query,
                memories=result.memories[:limit],
                raw_response=result.raw_response,
                error=result.error,
            )
        return result

    def snapshot(self) -> Any:
        return call_backend(self.backend, ("get_all", "snapshot", "dump"))

    def close(self) -> None:
        if self._legacy_http_client is not None:
            self._legacy_http_client.close()
            self._legacy_http_client = None
