"""Optional A-Mem adapter for exact-content, control-plane smoke runs."""

from __future__ import annotations

import hashlib
import inspect
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping

from ._optional import call_backend, metadata_dict
from .base import MemoryRetrievalResult, MemorySystemAdapter, normalize_retrieval


class AMemAdapter(MemorySystemAdapter):
    """Use official A-Mem storage/index structures without agentic rewriting.

    The official ``add_note`` path performs note processing/evolution. S1
    instead constructs a fully specified ``MemoryNote`` and uses the public
    ``search_agentic`` surface when it is demonstrably non-LLM/mutating.
    """

    memory_system = "amem"
    adapter_version = "a-mem-native-raw-s1-v1"

    def __init__(self, backend: Any | None = None, *, config: Mapping[str, Any] | None = None):
        self.config = dict(config or {})
        self._injected = backend is not None
        self._factory: Any | None = None
        self._active_namespace: tuple[str, ...] | None = None
        self._notes: list[Any] = []
        self._note_by_id: dict[str, Any] = {}
        self._raw_insertion_used = False
        self._agentic_evolution_disabled = True
        self._generated_metadata_disabled = True
        self._llm_constructor_initialized = True
        self.retrieval_mode = str(self.config.get("retrieval_mode", "public_search_agentic"))
        self.llm_backend_constructor = str(self.config.get("llm_backend", "openai"))
        self.llm_model_constructor = str(self.config.get("llm_model", "gpt-4o-mini"))
        self.retrieval_k = int(self.config.get("retrieval_k", self.config.get("k", 5)))
        self.backend = backend if backend is not None else self._build_real_backend()
        self._validate_native_backend()

    @staticmethod
    def namespace_for_base(base_example_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(str(base_example_id).encode("utf-8")).hexdigest()
        return ("govmem_system_validation", digest)

    @staticmethod
    def package_version() -> str | None:
        for name in ("a-mem", "agentic-memory"):
            try:
                return version(name)
            except PackageNotFoundError:
                continue
        return None

    @staticmethod
    def _normalize_namespace(namespace: tuple[str, ...] | list[str] | str | None) -> tuple[str, ...]:
        if namespace is None:
            return ("govmem_system_validation", "default")
        if isinstance(namespace, str):
            return (namespace,)
        return tuple(str(part) for part in namespace)

    def _build_real_backend(self) -> Any:
        try:
            module = __import__("agentic_memory.memory_system", fromlist=["AgenticMemorySystem"])
            factory = getattr(module, "AgenticMemorySystem")
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "A-Mem smoke requires the official A-Mem package exposing "
                "agentic_memory.memory_system.AgenticMemorySystem. Install the "
                "official package (for example: pip install a-mem), or inject "
                "a backend for tests; unrelated forks are not supported."
            ) from exc
        self._factory = factory
        constructor_only = {"retrieval_mode", "retrieval_k", "k", "allow_unverified_search_agentic", "allow_agentic_query_llm"}
        kwargs = {key: value for key, value in self.config.items() if key not in constructor_only}
        kwargs.setdefault("model_name", "all-MiniLM-L6-v2")
        # Current official A-Mem validates this constructor value even though
        # S1 never invokes its LLM controller.  These are compatibility-only
        # parameters; raw insertion and native retrieval make no LLM calls.
        kwargs.setdefault("llm_backend", "openai")
        kwargs.setdefault("llm_model", "gpt-4o-mini")
        self.llm_backend_constructor = str(kwargs["llm_backend"])
        self.llm_model_constructor = str(kwargs["llm_model"])
        return factory(**kwargs)

    def _validate_native_backend(self) -> None:
        if hasattr(self.backend, "memories") and hasattr(self.backend, "retriever"):
            retriever = self.backend.retriever
            required = ("add_document", "delete_document", "search")
            if not all(callable(getattr(retriever, name, None)) for name in required):
                raise RuntimeError("A-Mem backend does not expose the official ChromaRetriever add_document/delete_document/search API")
            if not callable(getattr(self.backend, "search_agentic", None)) and self.retrieval_mode != "low_level_ablation":
                raise RuntimeError(
                    "This A-Mem revision has no search_agentic method. Set "
                    "retrieval_mode=low_level_ablation explicitly to run the "
                    "lower-level native retriever ablation; it is not full A-Mem."
                )
            return
        if not any(callable(getattr(self.backend, name, None)) for name in ("add", "add_memory", "store")):
            raise RuntimeError("A-Mem backend must be AgenticMemorySystem-like or an injected generic test backend")

    @property
    def raw_insertion_used(self) -> bool:
        return self._raw_insertion_used

    @property
    def native_insertion_path(self) -> str:
        return "MemoryNote+ChromaRetriever.add_document" if hasattr(self.backend, "memories") else "injected_backend"

    @property
    def native_retrieval_surface(self) -> str:
        if not self._is_native():
            return "injected_backend"
        if self.retrieval_mode == "low_level_ablation":
            return "retriever.search (low_level_ablation)"
        return "AgenticMemorySystem.search_agentic"

    def _is_native(self) -> bool:
        return hasattr(self.backend, "memories") and hasattr(self.backend, "retriever")

    def _require_namespace(self, namespace: tuple[str, ...] | list[str] | str | None) -> tuple[str, ...]:
        ns = self._normalize_namespace(namespace)
        if self._active_namespace is None:
            self._active_namespace = ns
        elif ns != self._active_namespace:
            raise ValueError("A-Mem smoke namespaces are isolated; reset before another base example")
        return ns

    def reset(self) -> None:
        self._active_namespace = None
        self._notes = []
        self._note_by_id = {}
        self._raw_insertion_used = False
        if self._injected or self._factory is None:
            method = next((getattr(self.backend, name, None) for name in ("reset", "clear", "delete_all") if callable(getattr(self.backend, name, None))), None)
            if method is not None:
                method()
            return
        self.backend = self._build_real_backend()
        self._validate_native_backend()

    @staticmethod
    def _note_payload(note: Any, *, rank: int, score: float | None = None) -> dict[str, Any]:
        def value(name: str, default: Any = None) -> Any:
            return getattr(note, name, default)

        return {
            "id": str(value("id", "")),
            "content": str(value("content", "")),
            "metadata": deepcopy(value("govmem_metadata", {})),
            "keywords": deepcopy(value("keywords", [])),
            "context": value("context", None),
            "tags": deepcopy(value("tags", [])),
            "category": value("category", None),
            "links": deepcopy(value("links", [])),
            "rank": rank,
            "score": score,
        }

    def _make_raw_note(self, content: str, memory_id: str, metadata: Mapping[str, Any]) -> Any:
        try:
            from agentic_memory.memory_system import MemoryNote
        except ImportError as exc:
            raise RuntimeError("Installed A-Mem has no MemoryNote raw-insertion path; refusing LLM-rewritten insertion") from exc
        note = MemoryNote(
            content=content,
            id=memory_id,
            keywords=[],
            links={},
            retrieval_count=0,
            context="",
            evolution_history=[],
            category="raw",
            tags=[],
            timestamp=metadata.get("timestamp"),
        )
        note.id = memory_id
        note.govmem_metadata = deepcopy(metadata)
        return note

    @staticmethod
    def _official_metadata(note: Any) -> dict[str, Any]:
        return {
            "id": note.id,
            "content": note.content,
            "keywords": deepcopy(getattr(note, "keywords", [])),
            "links": deepcopy(getattr(note, "links", [])),
            "retrieval_count": getattr(note, "retrieval_count", 0),
            "timestamp": getattr(note, "timestamp", None),
            "last_accessed": getattr(note, "last_accessed", None),
            "context": getattr(note, "context", ""),
            "evolution_history": deepcopy(getattr(note, "evolution_history", [])),
            "category": getattr(note, "category", "raw"),
            "tags": deepcopy(getattr(note, "tags", [])),
        }

    def _rebuild_native_index(self) -> None:
        retriever = self.backend.retriever
        try:
            model_name = getattr(retriever, "model_name", self.config.get("model_name", "all-MiniLM-L6-v2"))
            for note in self._notes:
                self.backend.retriever.delete_document(note.id)
            for note in self._notes:
                self.backend.retriever.add_document(note.content, self._official_metadata(note), note.id)
        except Exception as exc:
            raise RuntimeError("A-Mem native retriever could not be rebuilt after mutation") from exc

    def add_memory(self, content: str, *, namespace: tuple[str, ...] | list[str] | str | None = None, memory_id: str | None = None, metadata: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        self._require_namespace(namespace)
        if not self._is_native():
            return call_backend(self.backend, ("add", "add_memory", "store"), content, metadata=metadata_dict(metadata), **kwargs)
        metadata_map = metadata_dict(metadata)
        note_id = memory_id or hashlib.sha256(content.encode("utf-8")).hexdigest()
        if note_id in self._note_by_id:
            raise ValueError(f"A-Mem memory id already exists: {note_id}")
        note = self._make_raw_note(content, note_id, metadata_map)
        self.backend.memories[note_id] = note
        self._notes.append(note)
        self._note_by_id[note_id] = note
        self.backend.retriever.add_document(note.content, self._official_metadata(note), note.id)
        self._raw_insertion_used = True
        return note_id

    def update_memory(self, memory_id: str, content: str | None = None, *, namespace: tuple[str, ...] | list[str] | str | None = None, metadata: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        self._require_namespace(namespace)
        if not self._is_native():
            return call_backend(self.backend, ("update", "update_memory"), memory_id, content, metadata=metadata_dict(metadata), **kwargs)
        note = self._note_by_id.get(str(memory_id))
        if note is None:
            raise KeyError(memory_id)
        if content is not None:
            note.content = content
        if metadata is not None:
            note.govmem_metadata = deepcopy(metadata)
        self.backend.retriever.delete_document(note.id)
        self.backend.retriever.add_document(note.content, self._official_metadata(note), note.id)
        return memory_id

    def delete_memory(self, memory_id: str, *, namespace: tuple[str, ...] | list[str] | str | None = None, **kwargs: Any) -> Any:
        self._require_namespace(namespace)
        if not self._is_native():
            return call_backend(self.backend, ("delete", "delete_memory", "remove"), memory_id, **kwargs)
        note = self._note_by_id.pop(str(memory_id), None)
        if note is None:
            return None
        self.backend.memories.pop(str(memory_id), None)
        self.backend.retriever.delete_document(str(memory_id))
        self._notes = [candidate for candidate in self._notes if candidate is not note]
        return memory_id

    def _stored_state(self) -> list[tuple[str, str]]:
        return [(str(note.id), str(note.content)) for note in self._notes]

    def _assert_public_search_safe(self, method: Any) -> None:
        """Reject a public search implementation that visibly invokes an LLM.

        A-Mem revisions differ.  Static inspection is deliberately conservative:
        if the implementation cannot be inspected, users must explicitly opt
        into that version rather than silently adding a model call.
        """
        try:
            source = inspect.getsource(method)
        except (OSError, TypeError) as exc:
            if not self.config.get("allow_unverified_search_agentic", False):
                raise RuntimeError(
                    "A-Mem search_agentic could not be inspected for LLM query "
                    "rewriting. Set allow_unverified_search_agentic=true only "
                    "after auditing the installed official revision."
                ) from exc
            return
        lower = source.lower()
        llm_markers = ("llm_controller", "openai", "chat.completions", "generate_response", "invoke_llm")
        if any(marker in lower for marker in llm_markers) and not self.config.get("allow_agentic_query_llm", False):
            raise RuntimeError(
                "A-Mem search_agentic appears to invoke an LLM for query "
                "processing in this installed revision. Refusing to introduce "
                "that call; use retrieval_mode=low_level_ablation explicitly "
                "and label results as an ablation."
            )

    def _public_payloads(self, result: Any) -> list[Any]:
        if isinstance(result, Mapping):
            for key in ("results", "memories", "data", "items"):
                if isinstance(result.get(key), (list, tuple)):
                    return list(result[key])
            return [result]
        if isinstance(result, (list, tuple)):
            return list(result)
        return [result] if result is not None else []

    def _retrieve_public(self, query: str, k: int) -> MemoryRetrievalResult:
        method = getattr(self.backend, "search_agentic", None)
        if not callable(method):
            raise RuntimeError("A-Mem search_agentic is unavailable; full A-Mem retrieval cannot be claimed")
        self._assert_public_search_safe(method)
        before = self._stored_state()
        native_result = method(query, k=k)
        after = self._stored_state()
        if before != after:
            raise RuntimeError("A-Mem search_agentic mutated stored notes; refusing causal smoke output")
        payloads = []
        for rank, item in enumerate(self._public_payloads(native_result), start=1):
            if hasattr(item, "content") and not isinstance(item, Mapping):
                payloads.append(self._note_payload(item, rank=rank, score=getattr(item, "score", None)))
            else:
                payloads.append(item)
        return normalize_retrieval({"results": payloads, "native": deepcopy(native_result)}, memory_system=self.memory_system, query=query)

    def _retrieve_low_level(self, query: str, k: int) -> MemoryRetrievalResult:
        native_indices = self.backend.retriever.search(query, k=k)
        indices = list(native_indices) if native_indices is not None else []
        payloads: list[dict[str, Any]] = []
        for rank, native_index in enumerate(indices, start=1):
            if isinstance(native_index, Mapping):
                index = native_index.get("index", native_index.get("document_id"))
                score = native_index.get("score")
                note_id = native_index.get("id")
            else:
                index, score, note_id = native_index, None, None
            try:
                note = self._notes[int(index)] if note_id is None else self._note_by_id.get(str(note_id))
            except (TypeError, ValueError, IndexError):
                note = None
            if note is None:
                continue
            item = self._note_payload(note, rank=rank, score=score)
            item["native_index"] = int(index) if isinstance(index, (int, float)) else index
            payloads.append(item)
        raw = {"native_indices": deepcopy(indices), "notes": deepcopy(payloads), "query": query, "k": k}
        return normalize_retrieval({"results": payloads, "native": raw}, memory_system=self.memory_system, query=query)

    def retrieve(self, query: str, *, namespace: tuple[str, ...] | list[str] | str | None = None, limit: int | None = None, **kwargs: Any) -> MemoryRetrievalResult:
        self._require_namespace(namespace)
        if not self._is_native():
            payload = call_backend(self.backend, ("search", "retrieve", "query"), query, **kwargs)
            return normalize_retrieval(payload, memory_system=self.memory_system, query=query)
        k = int(limit if limit is not None else self.retrieval_k)
        if self.retrieval_mode != "low_level_ablation":
            return self._retrieve_public(query, k)
        return self._retrieve_low_level(query, k)

    def snapshot(self) -> Any:
        if not self._is_native():
            return call_backend(self.backend, ("snapshot", "dump", "get_all"))
        return {
            "backend": "AgenticMemorySystem",
            "namespace": list(self._active_namespace) if self._active_namespace else None,
            "raw_insertion_used": self._raw_insertion_used,
            "agentic_evolution_disabled": self._agentic_evolution_disabled,
            "notes": [self._note_payload(note, rank=index) for index, note in enumerate(self._notes, start=1)],
            "config": deepcopy(self.config),
            "retrieval_mode": self.retrieval_mode,
            "native_retrieval_surface": self.native_retrieval_surface,
            "llm_backend_constructor": self.llm_backend_constructor,
            "llm_model_constructor": self.llm_model_constructor,
            "amem_backend_variant": "official_chroma" if self._is_native() else "injected",
            "native_index_backend": "ChromaDB" if self._is_native() else None,
            "generated_metadata_disabled": self._generated_metadata_disabled,
            "llm_constructor_initialized": self._llm_constructor_initialized,
        }

    def package_metadata(self) -> dict[str, Any]:
        return {
            "amem_package_version": self.package_version(),
            "native_insertion_path": self.native_insertion_path,
            "raw_insertion_used": self.raw_insertion_used,
            "agentic_evolution_disabled": self._agentic_evolution_disabled,
            "retrieval_k": self.retrieval_k,
            "backend_configuration": deepcopy(self.config),
            "retrieval_mode": self.retrieval_mode,
            "native_retrieval_surface": self.native_retrieval_surface,
            "llm_backend_constructor": self.llm_backend_constructor,
            "llm_model_constructor": self.llm_model_constructor,
            "amem_backend_variant": "official_chroma" if self._is_native() else "injected",
            "native_index_backend": "ChromaDB" if self._is_native() else None,
            "generated_metadata_disabled": self._generated_metadata_disabled,
            "llm_constructor_initialized": self._llm_constructor_initialized,
        }

    def close(self) -> None:
        closer = getattr(self.backend, "close", None)
        if callable(closer):
            closer()
