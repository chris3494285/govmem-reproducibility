"""LangMem/LangGraph InMemoryStore adapter for exact-content smoke validation."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping, cast

from ._optional import metadata_dict, require_dependency
from .base import MemoryRetrievalResult, MemorySystemAdapter, normalize_retrieval


DEFAULT_INDEX_CONFIG = {"dims": 1536, "embed": "openai:text-embedding-3-small"}
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def build_langmem_embeddings(*, http_client: Any | None = None, model: str = DEFAULT_EMBEDDING_MODEL) -> tuple[Any, Any]:
    """Build the explicit legacy-httpx LangChain embedding transport.

    The returned tuple is ``(embeddings, owned_http_client)``.  The caller
    owns and closes the second value when it was created here.
    """
    try:
        import httpx
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "LangMem embedding support requires langchain-openai and httpx. "
            "Install the LangMem OSS stack plus langchain-openai."
        ) from exc
    owned_client = http_client is None
    client = http_client or httpx.Client()
    embeddings = OpenAIEmbeddings(model=model, http_client=cast(Any, client))
    return embeddings, client if owned_client else None


class LangMemAdapter(MemorySystemAdapter):
    """Use LangGraph's real BaseStore path without LangMem LLM managers."""

    memory_system = "langmem"
    adapter_version = "langmem-langgraph-inmemory-s1-v1"

    def __init__(self, backend: Any | None = None, *, config: Mapping[str, Any] | None = None):
        self.config = dict(config or {})
        self.index_config = dict(self.config.get("index", DEFAULT_INDEX_CONFIG))
        self._inserted: dict[tuple[str, ...], dict[str, Any]] = {}
        self._inmemory_store_cls = None
        self._real_backend = backend is None
        self._http_client: Any | None = None
        self.embedding: Any | None = self.config.get("embeddings")
        self.embedding_config: dict[str, Any] = {
            "model": self.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            "dims": int(self.config.get("dims", DEFAULT_INDEX_CONFIG["dims"])),
            "transport": "explicit_httpx.Client",
            "embed_object": self.embedding is not None,
        }
        if backend is None:
            try:
                from langgraph.store.memory import InMemoryStore
            except ImportError as exc:
                raise RuntimeError(
                    "LangMem smoke requires langgraph. Install the OSS stack with pip install -U langmem langgraph."
                ) from exc
            require_dependency("langmem", "langmem", "LangMem")
            self._inmemory_store_cls = InMemoryStore
            self.embedding, self._http_client = build_langmem_embeddings(model=self.embedding_config["model"])
            self.embedding_config["embed_object"] = True
            self.index_config = {
                "dims": self.embedding_config["dims"],
                "embed": self.embedding,
            }
            backend = InMemoryStore(index=self.index_config)
        elif self.embedding is not None:
            self.index_config = {
                "dims": self.embedding_config["dims"],
                "embed": self.embedding,
            }
        self.backend = backend
        self._langgraph_store = callable(getattr(backend, "put", None)) and callable(getattr(backend, "search", None))

    @staticmethod
    def namespace_for_base(base_example_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(str(base_example_id).encode("utf-8")).hexdigest()
        return ("govmem_system_validation", digest)

    @staticmethod
    def package_versions() -> tuple[str | None, str | None]:
        def get(name: str) -> str | None:
            try:
                return version(name)
            except PackageNotFoundError:
                return None
        return get("langmem"), get("langgraph")

    @staticmethod
    def _namespace(namespace: tuple[str, ...] | list[str] | str | None) -> tuple[str, ...]:
        if namespace is None:
            return ("govmem_system_validation", "default")
        if isinstance(namespace, str):
            return (namespace,)
        return tuple(str(part) for part in namespace)

    def reset(self) -> None:
        if self._real_backend and self._inmemory_store_cls is not None:
            self.backend = self._inmemory_store_cls(index=self.index_config)
        else:
            for name in ("reset", "clear", "delete_all"):
                method = getattr(self.backend, name, None)
                if callable(method):
                    method()
                    break
        self._inserted = {}

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def add_memory(self, content: str, *, namespace: tuple[str, ...] | list[str] | str | None = None, memory_id: str | None = None, metadata: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        ns = self._namespace(namespace)
        if not self._langgraph_store:
            return self.backend.add(content, metadata=metadata_dict(metadata), **kwargs)
        key = memory_id or hashlib.sha256(content.encode("utf-8")).hexdigest()
        value = {"content": content, "metadata": metadata_dict(metadata)}
        result = self.backend.put(ns, key, deepcopy(value), **kwargs)
        self._inserted.setdefault(ns, {})[key] = deepcopy(value)
        return result

    def update_memory(self, memory_id: str, content: str | None = None, *, namespace: tuple[str, ...] | list[str] | str | None = None, metadata: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        ns = self._namespace(namespace)
        if not self._langgraph_store:
            return self.backend.update(memory_id, content, metadata=metadata_dict(metadata), **kwargs)
        previous = self._inserted.get(ns, {}).get(memory_id, {})
        value = {
            "content": content if content is not None else previous.get("content", ""),
            "metadata": metadata_dict(metadata) if metadata is not None else dict(previous.get("metadata", {})),
        }
        result = self.backend.put(ns, memory_id, deepcopy(value), **kwargs)
        self._inserted.setdefault(ns, {})[memory_id] = deepcopy(value)
        return result

    def delete_memory(self, memory_id: str, *, namespace: tuple[str, ...] | list[str] | str | None = None, **kwargs: Any) -> Any:
        ns = self._namespace(namespace)
        if not self._langgraph_store:
            return self.backend.delete(memory_id, **kwargs)
        result = self.backend.delete(ns, memory_id, **kwargs)
        self._inserted.get(ns, {}).pop(memory_id, None)
        return result

    @staticmethod
    def _item_payload(item: Any) -> dict[str, Any]:
        value = getattr(item, "value", None)
        key = getattr(item, "key", None)
        namespace = getattr(item, "namespace", None)
        score = getattr(item, "score", None)
        if isinstance(item, Mapping):
            value = item.get("value", value)
            key = item.get("key", item.get("id", key))
            namespace = item.get("namespace", namespace)
            score = item.get("score", score)
        if isinstance(value, Mapping):
            content = value.get("content", value.get("text", value))
            metadata = value.get("metadata", {})
        else:
            content, metadata = value, {}
        return {
            "id": str(key) if key is not None else None,
            "content": content if isinstance(content, str) else str(content),
            "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
            "namespace": list(namespace) if isinstance(namespace, tuple) else namespace,
            "score": score,
        }

    def retrieve(self, query: str, *, namespace: tuple[str, ...] | list[str] | str | None = None, limit: int = 10, **kwargs: Any) -> MemoryRetrievalResult:
        ns = self._namespace(namespace)
        if not self._langgraph_store:
            payload = self.backend.search(query, **kwargs)
            return normalize_retrieval(payload, memory_system=self.memory_system, query=query)
        items = self.backend.search(ns, query=query, limit=limit, **kwargs)
        return normalize_retrieval([self._item_payload(item) for item in items], memory_system=self.memory_system, query=query)

    def snapshot(self) -> Any:
        if not self._langgraph_store:
            return self.backend.get_all()
        return {
            "backend": "InMemoryStore" if self._real_backend else type(self.backend).__name__,
            "index": deepcopy(self.embedding_config),
            "namespaces": {"/".join(namespace): deepcopy(values) for namespace, values in self._inserted.items()},
        }
