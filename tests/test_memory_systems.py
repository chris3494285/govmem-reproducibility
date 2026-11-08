from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from govmem.memory_systems import (
    AMemAdapter,
    LangMemAdapter,
    Mem0Adapter,
    MemorySystemDependencyError,
    normalize_retrieval,
)
from govmem.memory_systems.langmem import DEFAULT_INDEX_CONFIG, LangMemAdapter, build_langmem_embeddings
from govmem.memory_systems.cohort import build_system_validation_smoke
from govmem.memory_systems.harness import SystemValidationHarness, validate_paired_invariants


class FakeBackend:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.append(("reset",))

    def add(self, content, **kwargs):
        self.calls.append(("add", content, kwargs))
        return {"id": "fake-1"}

    def update(self, memory_id, payload, **kwargs):
        self.calls.append(("update", memory_id, payload, kwargs))

    def delete(self, memory_id, **kwargs):
        self.calls.append(("delete", memory_id, kwargs))

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return [
            {"id": "b", "memory": "second", "score": 0.8, "metadata": {"source": "fake"}, "rank": 2},
            {"id": "a", "memory": "first", "score": 0.9, "metadata": {"source": "fake"}, "rank": 1},
        ]

    def get_all(self):
        return {"snapshot": True}


class FakeAMemRetriever:
    def __init__(self):
        self.documents = []
        self.search_calls = []
        self.evolution_calls = 0

    def add_document(self, document, metadata, doc_id):
        self.documents.append((document, metadata, doc_id))

    def delete_document(self, doc_id):
        self.documents = [item for item in self.documents if item[2] != doc_id]

    def search(self, query, k=5):
        self.search_calls.append((query, k))
        return [{"id": item[2], "content": item[0], "score": 1.0} for item in self.documents[:k]]


class FakeAMemNative:
    def __init__(self):
        self.memories = {}
        self.retriever = FakeAMemRetriever()
        self.llm_calls = 0

    def reset(self):
        self.memories.clear()
        self.retriever = FakeAMemRetriever()

    def add_note(self, *args, **kwargs):
        raise AssertionError("A-Mem add_note must not be used for raw smoke insertion")

    def process_memory(self, *args, **kwargs):
        raise AssertionError("A-Mem process_memory must not be used for raw smoke insertion")

    def search_agentic(self, query, k=5):
        return list(self.memories.values())[:k]


def _row(base: str, condition: str, *, operator: str | None = None):
    return {
        "example_id": base,
        "source_dataset": "gatemem",
        "dimension": "authorization",
        "condition_type": condition,
        "fault_spec": None if condition == "clean" else {"operator": operator, "direction": "ambiguous"},
        "task": "What is remembered?",
        "memory": [{"content": "fixed underlying content", "memory_id": "m1"}],
        "observed_governance_state": {"authorization": {"decision": "allow"}},
    }


def test_normalized_retrieval_is_deterministic_and_preserves_raw_metadata():
    result = normalize_retrieval(
        [{"id": "b", "content": "B", "rank": 2}, {"id": "a", "content": "A", "rank": 1, "metadata": {"x": 1}}],
        memory_system="fake",
        query="q",
    )
    assert result.retrieved_memory_ids == ("a", "b")
    assert result.normalized_retrieved_contents == ("A", "B")
    assert result.memories[0].metadata == {"x": 1}
    assert result.memories[0].raw_payload["id"] == "a"


def test_mem0_optional_retrieval_limit_is_native_and_preserves_order():
    backend = FakeBackend()
    result = Mem0Adapter(backend).retrieve("query", limit=1)
    search_call = next(call for call in backend.calls if call[0] == "search")
    assert search_call[2]["top_k"] == 1
    assert "limit" not in search_call[2]
    assert result.retrieved_memory_ids == ("a",)


def test_mem0_openai_embedder_gets_owned_legacy_httpx_client(monkeypatch):
    import sys
    import types

    clients = []

    class HttpClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class OpenAI:
        def __init__(self, **kwargs):
            clients.append(kwargs)

    class EmbedderConfig:
        api_key = "configured-key"
        openai_base_url = "https://configured.example/v1"

    class OpenAIEmbedding:
        def __init__(self):
            self.config = EmbedderConfig()
            self.client = None

    OpenAIEmbedding.__module__ = "mem0.embeddings.openai"
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=HttpClient))
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))
    backend = types.SimpleNamespace(embedding_model=OpenAIEmbedding())
    adapter = Mem0Adapter(backend)
    assert adapter.mem0_embedding_transport == "legacy_httpx"
    assert isinstance(adapter._legacy_http_client, HttpClient)
    assert clients == [{
        "api_key": "configured-key",
        "base_url": "https://configured.example/v1",
        "http_client": adapter._legacy_http_client,
    }]
    adapter.close()
    assert adapter._legacy_http_client is None


def test_mem0_non_openai_embedder_is_not_modified(monkeypatch):
    class OtherEmbedding:
        def __init__(self):
            self.client = "original"

    backend = type("Backend", (), {"embedding_model": OtherEmbedding()})()
    adapter = Mem0Adapter(backend)
    assert backend.embedding_model.client == "original"
    assert adapter.mem0_embedding_transport is None


@pytest.mark.parametrize("adapter_cls", [Mem0Adapter, LangMemAdapter, AMemAdapter])
def test_each_adapter_exposes_common_contract_with_injected_fake(adapter_cls):
    adapter = adapter_cls(FakeBackend())
    adapter.reset()
    adapter.add_memory("x")
    result = adapter.retrieve("query")
    adapter.update_memory("fake-1", "y")
    adapter.delete_memory("fake-1")
    assert result.memory_system == adapter.memory_system
    assert result.retrieved_memory_ids == ("a", "b")
    add_call = next(call for call in adapter.backend.calls if call[0] == "add")
    if adapter_cls is Mem0Adapter:
        assert add_call[2]["infer"] is False
        assert add_call[2]["user_id"] == "govmem-default"
    search_call = next(call for call in adapter.backend.calls if call[0] == "search")
    if adapter_cls is Mem0Adapter:
        assert search_call[2]["filters"] == {"user_id": "govmem-default"}
        assert "user_id" not in search_call[2]
    assert adapter.snapshot() == {"snapshot": True}


def test_optional_dependency_failure_is_actionable(monkeypatch):
    import govmem.memory_systems._optional as optional

    def fail(*args, **kwargs):
        raise ImportError("missing")

    monkeypatch.setattr(optional, "__import__", fail, raising=False)
    # The helper uses the built-in import function, so exercise its stable
    # failure contract with a guaranteed-missing module instead.
    with pytest.raises(MemorySystemDependencyError, match="pip install missing-system-package"):
        optional.require_dependency("module_that_does_not_exist_for_govmem_tests", "missing-system-package", "FakeSystem")


def test_smoke_manifest_is_deterministic_and_non_frozen_without_mutating_source():
    source = Path("data/governance_fault_rq1_pilot_frozen_v2.jsonl")
    before = source.read_bytes()
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    first = build_system_validation_smoke(rows, source_manifest=str(source))
    second = build_system_validation_smoke(rows, source_manifest=str(source))
    assert first == second
    assert first["status"] == "NON-FROZEN / SMOKE"
    assert first["selected_base_examples"] == 9
    assert first["selected_condition_rows"] == 27
    assert source.read_bytes() == before


def test_pair_content_and_query_invariance_and_exact_context_hash():
    clean = _row("base-1", "clean")
    fault = _row("base-1", "fault", operator="missing")
    backend = FakeBackend()
    harness = SystemValidationHarness(Mem0Adapter(backend), system_config={"k": 1})
    clean_result, fault_result = harness.run_pair(clean, fault)
    assert clean_result.query == fault_result.query == clean["task"]
    assert clean_result.normalized_retrieved_contents == fault_result.normalized_retrieved_contents
    assert clean_result.context_sha256 == hashlib.sha256(clean_result.model_visible_context.encode()).hexdigest()
    assert clean_result.latency_ms is None
    assert clean_result.memory_system == "mem0"


def test_pair_invariance_rejects_changed_underlying_content_or_query():
    clean = _row("base-1", "clean")
    fault = _row("base-1", "fault", operator="missing")
    fault["task"] = "changed query"
    with pytest.raises(ValueError, match="invariant field 'task'"):
        validate_paired_invariants(clean, fault)


def test_context_is_system_specific_and_never_pooled():
    row = _row("base-1", "clean")
    mem0 = SystemValidationHarness(Mem0Adapter(FakeBackend())).run_condition(row)
    langmem = SystemValidationHarness(LangMemAdapter(FakeBackend())).run_condition(row)
    assert {mem0.memory_system, langmem.memory_system} == {"mem0", "langmem"}
    assert mem0.to_dict()["memory_system"] != langmem.to_dict()["memory_system"]


def test_amem_native_path_preserves_exact_content_and_uses_native_search(monkeypatch):
    import sys
    import types

    class MemoryNote:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    layer = types.ModuleType("agentic_memory.memory_system")
    layer.MemoryNote = MemoryNote
    monkeypatch.setitem(sys.modules, "agentic_memory.memory_system", layer)

    backend = FakeAMemNative()
    adapter = AMemAdapter(backend, config={"retrieval_k": 1})
    namespace = AMemAdapter.namespace_for_base("base-a")
    exact = "Frozen transition: current=Free; previous=On_Business_Trip"
    memory_id = adapter.add_memory(exact, namespace=namespace, metadata={"source": "frozen"}, memory_id="m-a")
    result = adapter.retrieve("invariant query", namespace=namespace)

    assert memory_id == "m-a"
    assert backend.memories["m-a"].content == exact
    assert backend.memories["m-a"].id == "m-a"
    assert backend.memories["m-a"].keywords == []
    assert backend.memories["m-a"].links == {}
    assert backend.memories["m-a"].retrieval_count == 0
    assert backend.memories["m-a"].context == ""
    assert backend.memories["m-a"].evolution_history == []
    assert backend.memories["m-a"].category == "raw"
    assert backend.memories["m-a"].tags == []
    assert not hasattr(backend.memories["m-a"], "llm_controller")
    assert backend.retriever.documents[0][0] == exact
    assert backend.retriever.documents[0][2] == "m-a"
    assert not backend.retriever.search_calls
    assert adapter.native_retrieval_surface == "AgenticMemorySystem.search_agentic"
    assert result.normalized_retrieved_contents == (exact,)
    assert result.memories[0].raw_payload["id"] == "m-a"
    assert adapter.raw_insertion_used is True
    assert adapter.native_insertion_path == "MemoryNote+ChromaRetriever.add_document"
    assert adapter.package_metadata()["agentic_evolution_disabled"] is True
    assert adapter.package_metadata()["llm_backend_constructor"] == "openai"
    assert adapter.package_metadata()["amem_backend_variant"] == "official_chroma"
    assert adapter.package_metadata()["native_index_backend"] == "ChromaDB"
    assert adapter.package_metadata()["generated_metadata_disabled"] is True
    assert adapter.package_metadata()["llm_constructor_initialized"] is True
    assert backend.llm_calls == 0


def test_amem_real_constructor_uses_current_compatible_llm_parameters(monkeypatch):
    import builtins
    import sys
    import types
    import govmem.memory_systems.amem as amem_module

    captured = {}

    class FakeSystem(FakeAMemNative):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    module = types.ModuleType("agentic_memory.memory_system")
    module.AgenticMemorySystem = FakeSystem
    original_import = builtins.__import__

    def fake_import(name, fromlist=(), *args, **kwargs):
        if name == "agentic_memory.memory_system":
            return module
        return original_import(name, fromlist, *args, **kwargs)

    monkeypatch.setattr(amem_module, "__import__", fake_import, raising=False)
    adapter = AMemAdapter()
    assert captured["model_name"] == "all-MiniLM-L6-v2"
    assert captured["llm_backend"] == "openai"
    assert captured["llm_model"] == "gpt-4o-mini"
    assert adapter.package_metadata()["agentic_evolution_disabled"] is True
    assert not hasattr(adapter.backend, "add_note") or adapter.backend.llm_calls == 0


def test_amem_namespaces_are_isolated_and_reset_between_bases(monkeypatch):
    import sys
    import types

    class MemoryNote:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    layer = types.ModuleType("agentic_memory.memory_system")
    layer.MemoryNote = MemoryNote
    monkeypatch.setitem(sys.modules, "agentic_memory.memory_system", layer)

    backend = FakeAMemNative()
    adapter = AMemAdapter(backend)
    namespace_a = AMemAdapter.namespace_for_base("base-a")
    namespace_b = AMemAdapter.namespace_for_base("base-b")
    adapter.add_memory("only-a", namespace=namespace_a, memory_id="a")
    with pytest.raises(ValueError, match="reset"):
        adapter.add_memory("wrong namespace", namespace=namespace_b, memory_id="b")
    adapter.reset()
    adapter.add_memory("only-b", namespace=namespace_b, memory_id="b")
    assert adapter.retrieve("q", namespace=namespace_b).normalized_retrieved_contents == ("only-b",)
    assert "only-a" not in [note.content for note in backend.memories.values()]


def test_amem_low_level_retriever_is_explicitly_labeled_ablation(monkeypatch):
    import sys
    import types

    class MemoryNote:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    layer = types.ModuleType("agentic_memory.memory_system")
    layer.MemoryNote = MemoryNote
    monkeypatch.setitem(sys.modules, "agentic_memory.memory_system", layer)
    adapter = AMemAdapter(FakeAMemNative(), config={"retrieval_mode": "low_level_ablation"})
    assert "low_level_ablation" in adapter.native_retrieval_surface


class FakeLangGraphStore:
    def __init__(self):
        self.values = {}
        self.calls = []

    def put(self, namespace, key, value, **kwargs):
        self.calls.append(("put", namespace, key, value, kwargs))
        self.values[(tuple(namespace), key)] = value

    def search(self, namespace, *, query, limit=10, **kwargs):
        self.calls.append(("search", namespace, query, limit, kwargs))
        return [
            {"namespace": tuple(namespace), "key": key, "value": value, "score": 0.5}
            for (stored_namespace, key), value in self.values.items()
            if stored_namespace == tuple(namespace)
        ][:limit]

    def delete(self, namespace, key, **kwargs):
        self.values.pop((tuple(namespace), key), None)

    def reset(self):
        self.values.clear()


def test_langmem_direct_store_path_preserves_exact_content_and_namespace():
    store = FakeLangGraphStore()
    adapter = LangMemAdapter(store)
    namespace_a = LangMemAdapter.namespace_for_base("base-a")
    namespace_b = LangMemAdapter.namespace_for_base("base-b")
    assert namespace_a != namespace_b
    exact = "Raw frozen transition: current=Free; previous=On_Business_Trip"
    adapter.add_memory(exact, namespace=namespace_a, memory_id="m-a", metadata={"source": "frozen"})
    adapter.add_memory("other base", namespace=namespace_b, memory_id="m-b")
    result = adapter.retrieve("query", namespace=namespace_a)
    assert result.normalized_retrieved_contents == (exact,)
    assert result.memories[0].raw_payload["namespace"] == list(namespace_a)
    assert result.memories[0].metadata == {"source": "frozen"}
    assert store.calls[0][0] == "put"
    assert store.calls[0][3]["content"] == exact
    assert DEFAULT_INDEX_CONFIG["embed"] == "openai:text-embedding-3-small"
    assert not any(call[0] == "llm" for call in store.calls)


def test_langmem_embedding_builder_injects_explicit_http_client(monkeypatch):
    import sys
    import types

    class FakeHttpClient:
        pass

    class FakeEmbeddings:
        def __init__(self, *, model, http_client):
            self.model = model
            self.http_client = http_client

    fake_httpx = types.SimpleNamespace(Client=FakeHttpClient)
    fake_langchain_openai = types.SimpleNamespace(OpenAIEmbeddings=FakeEmbeddings)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_langchain_openai)
    client = FakeHttpClient()
    embeddings, owned = build_langmem_embeddings(http_client=client)
    assert embeddings.model == "text-embedding-3-small"
    assert embeddings.http_client is client
    assert owned is None


def test_langmem_adapter_uses_injected_embedding_object_not_string_shortcut():
    embedding = object()
    adapter = LangMemAdapter(FakeLangGraphStore(), config={"embeddings": embedding})
    assert adapter.index_config["embed"] is embedding
    assert adapter.embedding_config["embed_object"] is True
    assert adapter.embedding_config["model"] == "text-embedding-3-small"
