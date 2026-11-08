from __future__ import annotations

import json
from pathlib import Path

from govmem.memory_systems.mem0 import Mem0Adapter
from govmem.memory_systems.langmem import LangMemAdapter
from govmem.memory_systems.amem import AMemAdapter
from govmem.memory_systems.base import normalize_retrieval
from govmem.memory_systems.harness import project_retrieved_context
from govmem.governance.enforcement import decide_admission
from scripts.run_system_validation_smoke import run_smoke


class NamespaceBackend:
    def __init__(self, *, empty: bool = False):
        self.by_user = {}
        self.calls = []
        self.empty = empty

    def reset(self):
        self.calls.append(("reset",))
        self.by_user.clear()

    def add(self, messages, *, user_id, metadata, infer):
        self.calls.append(("add", messages, {"user_id": user_id, "metadata": metadata, "infer": infer}))
        self.by_user.setdefault(user_id, []).append({"id": f"{user_id}-memory", "memory": messages, "metadata": metadata, "rank": 1})

    def search(self, query, *, filters, top_k=None):
        self.calls.append(("search", query, {"filters": filters, "top_k": top_k}))
        if self.empty:
            return []
        results = list(self.by_user.get(filters["user_id"], []))
        return results if top_k is None else results[:top_k]

    def update(self, *args, **kwargs):
        return None

    def delete(self, *args, **kwargs):
        return None

    def get_all(self):
        return self.by_user


class LangGraphSmokeStore:
    def __init__(self):
        self.values = {}
        self.calls = []

    def reset(self):
        self.values.clear()
        self.calls.append(("reset",))

    def put(self, namespace, key, value, **kwargs):
        self.values[(tuple(namespace), key)] = value
        self.calls.append(("put", tuple(namespace), key, value))

    def search(self, namespace, *, query, limit=10, **kwargs):
        self.calls.append(("search", tuple(namespace), query, limit, kwargs))
        return [
            {"namespace": tuple(ns), "key": key, "value": value, "score": 0.7}
            for (ns, key), value in self.values.items()
            if ns == tuple(namespace)
        ][:limit]

    def delete(self, namespace, key, **kwargs):
        self.values.pop((tuple(namespace), key), None)

    def get_all(self):
        return self.values


class AMemSmokeRetriever:
    def __init__(self):
        self.documents = []
        self.search_calls = []

    def add_document(self, document, metadata, doc_id):
        self.documents.append((document, metadata, doc_id))

    def delete_document(self, doc_id):
        self.documents = [item for item in self.documents if item[2] != doc_id]

    def search(self, query, k=5):
        self.search_calls.append((query, k))
        return [{"id": item[2], "content": item[0], "score": 1.0} for item in self.documents[:k]]


class AMemSmokeBackend:
    def __init__(self):
        self.memories = {}
        self.retriever = AMemSmokeRetriever()
        self.search_agentic_calls = []

    def reset(self):
        self.memories.clear()
        self.retriever = AMemSmokeRetriever()

    def add_note(self, *args, **kwargs):
        raise AssertionError("A-Mem add_note must not be called")

    def process_memory(self, *args, **kwargs):
        raise AssertionError("A-Mem process_memory must not be called")

    def search_agentic(self, query, k=5):
        self.search_agentic_calls.append((query, k))
        return list(self.memories.values())[:k]


def _smoke_source() -> Path:
    return Path("data/system_validation_smoke_non_frozen.json")


def test_smoke_runner_reuses_one_native_retrieval_per_base_and_preserves_context(tmp_path):
    backend = NamespaceBackend()
    result = run_smoke(
        manifest=_smoke_source(),
        results_dir=tmp_path,
        limit_base=1,
        adapter_factory=lambda config: Mem0Adapter(backend),
    )
    records = result["records"]
    assert len(records) == 3
    assert sum(call[0] == "search" for call in backend.calls) == 1
    assert records[0]["native_retrieval_reused"] is False
    assert all(record["native_retrieval_reused"] for record in records[1:])
    assert all(record["native_retrieval_reused_from"] == records[0]["run_id"] for record in records[1:])
    assert len({record["query"] for record in records}) == 1
    assert len({record["underlying_memory_sha256"] for record in records}) == 1
    assert len({record["user_id"] for record in records}) == 1
    assert records[0]["native_retrieval_status"] == "native_retrieval_success"
    assert records[0]["native_raw_payload"]
    assert records[0]["native_retrieved_contents"] == records[0]["underlying_memory_content"]
    assert records[0]["context_sha256"] == __import__("hashlib").sha256(records[0]["final_model_visible_context"].encode()).hexdigest()


def test_smoke_runner_applies_retrieval_budget_and_summary_telemetry(tmp_path):
    backend = NamespaceBackend()
    result = run_smoke(
        manifest=_smoke_source(),
        results_dir=tmp_path,
        limit_base=1,
        retrieval_k=1,
        adapter_factory=lambda config: Mem0Adapter(backend),
    )
    assert all(record["requested_retrieval_k"] == 1 for record in result["records"])
    assert result["summary"]["requested_retrieval_k"] == 1
    assert result["summary"]["max_native_retrieved_memory_count"] == 1
    assert all(len(record["native_retrieved_memory_ids"]) <= 1 for record in result["records"])
    search_call = next(call for call in backend.calls if call[0] == "search")
    assert search_call[2]["top_k"] == 1


def test_smoke_runner_namespaces_different_base_examples_and_prevents_leakage(tmp_path):
    backend = NamespaceBackend()
    result = run_smoke(
        manifest=_smoke_source(),
        results_dir=tmp_path,
        limit_base=2,
        adapter_factory=lambda config: Mem0Adapter(backend),
    )
    groups = {}
    for record in result["records"]:
        groups.setdefault(record["base_example_id"], []).append(record)
    assert len(groups) == 2
    assert len({records[0]["user_id"] for records in groups.values()}) == 2
    assert all(record["native_retrieved_memory_ids"] for record in result["records"])
    for base_id, records in groups.items():
        user_id = records[0]["user_id"]
        assert all(memory_id.startswith(user_id) for record in records for memory_id in record["native_retrieved_memory_ids"])


def test_empty_native_retrieval_is_not_governance_denial(tmp_path):
    backend = NamespaceBackend(empty=True)
    result = run_smoke(
        manifest=_smoke_source(),
        results_dir=tmp_path,
        limit_base=1,
        adapter_factory=lambda config: Mem0Adapter(backend),
    )
    assert result["summary"]["native_retrieval_empty_n"] == 3
    assert result["summary"]["native_retrieval_error_n"] == 0
    assert all(record["native_retrieval_status"] == "native_retrieval_empty" for record in result["records"])
    assert all(record["governance_action"].startswith("governance_") for record in result["records"])
    assert all(record["final_model_visible_context"] == "[]" for record in result["records"])


def test_run_smoke_rejects_non_positive_retrieval_budget(tmp_path):
    for value in (0, -1):
        try:
            run_smoke(manifest=_smoke_source(), results_dir=tmp_path, retrieval_k=value, adapter_factory=lambda config: Mem0Adapter(NamespaceBackend()))
        except ValueError as exc:
            assert "positive" in str(exc)
        else:
            raise AssertionError("non-positive retrieval budget was accepted")


def test_smoke_outputs_are_json_and_csv_and_no_frozen_manifest_changes(tmp_path):
    source = _smoke_source()
    before = source.read_bytes()
    run_smoke(manifest=source, results_dir=tmp_path, limit_base=1, adapter_factory=lambda config: Mem0Adapter(NamespaceBackend()))
    assert source.read_bytes() == before
    assert (tmp_path / "system_validation_smoke.json").exists()
    assert (tmp_path / "system_validation_smoke.csv").exists()
    summary = json.loads((tmp_path / "system_validation_smoke_summary.json").read_text())
    assert summary["invariant_failure_n"] == 0


def _memconflict_row(condition: str, operator: str | None, temporal: dict | None, *, base: str = "mc-1"):
    current = {"Current_State": "Free"}
    previous = {"Current_State": "On_Business_Trip"}
    return {
        "base_example_id": base,
        "example_id": base,
        "run_id": f"{base}::{condition}",
        "source_dataset": "memconflict",
        "dimension": "temporal",
        "condition_type": "clean" if condition == "clean" else "fault",
        "fault_spec": None if condition == "clean" else {"operator": operator, "direction": "unsafe" if operator == "stale" else "ambiguous"},
        "manifest_version": "governance-fault-rq1-v2-corrected",
        "task": "What is the work status?",
        "memory": [{"attribute": "Work_Status", "value": {"transition": "current/previous"}}],
        "source_metadata": {
            "attribute": "Work_Status",
            "current_value": current,
            "previous_value": previous,
            "current_session": {"session_id": 7},
            "previous_session": {"session_id": 6},
        },
        "observed_governance_state": {
            "temporal": temporal,
            "temporal_assertions": [],
        },
        "gold_admission_decision": "allow",
    }


def test_memconflict_temporal_projection_changes_only_post_retrieval_context():
    native = normalize_retrieval(
        [{"id": "transition-1", "memory": json.dumps({"transition": "raw-current-and-previous"}), "metadata": {"native": True}}],
        memory_system="mem0",
        query="What is the work status?",
    )
    clean = _memconflict_row("clean", None, {"status": "valid"})
    missing = _memconflict_row("missing", "missing", None)
    stale = _memconflict_row("stale", "stale", {"status": "expired"})
    projected_clean, clean_meta = project_retrieved_context(clean, native, decide_admission(clean))
    projected_missing, missing_meta = project_retrieved_context(missing, native, decide_admission(missing))
    projected_stale, stale_meta = project_retrieved_context(stale, native, decide_admission(stale))

    assert native.normalized_retrieved_contents == native.normalized_retrieved_contents
    assert projected_clean.memories[0].raw_payload == native.memories[0].raw_payload
    assert projected_stale.memories[0].raw_payload == native.memories[0].raw_payload
    assert clean_meta["operative_memory_role"] == "current"
    assert stale_meta["operative_memory_role"] == "previous"
    assert missing_meta["operative_memory_role"] == "current"
    clean_context = json.dumps([memory.content for memory in projected_clean.memories], sort_keys=True)
    missing_context = json.dumps([memory.content for memory in projected_missing.memories], sort_keys=True)
    stale_context = json.dumps([memory.content for memory in projected_stale.memories], sort_keys=True)
    assert "Free" in clean_context
    assert "On_Business_Trip" in stale_context
    assert "Free" not in stale_context
    assert clean_context == missing_context
    assert clean_context != stale_context
    assert projected_clean.normalized_retrieved_contents != projected_stale.normalized_retrieved_contents


def test_non_memconflict_projection_preserves_native_context():
    native = normalize_retrieval([{"id": "g1", "memory": "authorization content"}], memory_system="mem0", query="q")
    row = {
        "example_id": "gate-1",
        "source_dataset": "gatemem",
        "dimension": "authorization",
        "condition_type": "clean",
        "task": "q",
        "memory": [{"content": "authorization content"}],
        "observed_governance_state": {"authorization": {"decision": "allow"}},
    }
    projected, metadata = project_retrieved_context(row, native, decide_admission(row))
    assert projected == native
    assert metadata["projection_reason"] == "no_temporal_projection"


def test_langmem_runner_reuses_native_retrieval_and_records_namespace(tmp_path):
    store = LangGraphSmokeStore()
    result = run_smoke(
        manifest=_smoke_source(),
        results_dir=tmp_path,
        limit_base=1,
        memory_system="langmem",
        adapter_factory=lambda config: LangMemAdapter(store),
    )
    records = result["records"]
    assert len(records) == 3
    assert sum(call[0] == "search" for call in store.calls) == 1
    assert all(record["memory_system"] == "langmem" for record in records)
    assert len({tuple(record["namespace"]) for record in records}) == 1
    assert all(record["store_backend"] == "InMemoryStore" for record in records)
    assert all(record["native_retrieval_reused"] for record in records[1:])
    assert all(record["native_retrieval_status"] == "native_retrieval_success" for record in records)


def test_langmem_runner_receives_retrieval_budget(tmp_path):
    store = LangGraphSmokeStore()
    run_smoke(
        manifest=_smoke_source(),
        results_dir=tmp_path,
        limit_base=1,
        retrieval_k=1,
        memory_system="langmem",
        adapter_factory=lambda config: LangMemAdapter(store),
    )
    searches = [call for call in store.calls if call[0] == "search"]
    assert searches and searches[0][3] == 1


def test_langmem_runner_memconflict_projection_preserves_native_result(tmp_path):
    manifest = json.loads(_smoke_source().read_text())
    selected = [
        row for row in manifest["records"]
        if row["source_dataset"] == "memconflict"
    ]
    base = selected[0]["base_example_id"]
    manifest["records"] = [row for row in selected if row["base_example_id"] == base]
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = run_smoke(
        manifest=path,
        results_dir=tmp_path / "out",
        memory_system="langmem",
        adapter_factory=lambda config: LangMemAdapter(LangGraphSmokeStore()),
    )
    records = {(record["condition"], record["fault_type"]): record for record in result["records"]}
    clean = records[("clean", "clean")]
    missing = records[("fault", "missing")]
    stale = records[("fault", "stale")]
    assert clean["native_retrieved_contents"] == missing["native_retrieved_contents"] == stale["native_retrieved_contents"]
    assert clean["operative_memory_role"] == "current"
    assert missing["operative_memory_role"] == "current"
    assert stale["operative_memory_role"] == "previous"
    assert clean["operative_memory_value"] == missing["operative_memory_value"]
    assert stale["operative_memory_value"] != clean["operative_memory_value"]
    stale_leaf = next(str(value) for value in stale["operative_memory_value"].values() if isinstance(value, str))
    clean_leaf = next(str(value) for value in clean["operative_memory_value"].values() if isinstance(value, str))
    assert stale_leaf in stale["final_model_visible_context"]
    assert clean_leaf not in stale["final_model_visible_context"]
    assert clean["final_model_visible_context"] == missing["final_model_visible_context"]
    assert clean["final_model_visible_context"] != stale["final_model_visible_context"]


def test_amem_runner_reuses_native_retrieval_and_stops_before_answer_model(monkeypatch, tmp_path):
    import sys
    import types

    class MemoryNote:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    layer = types.ModuleType("agentic_memory.memory_system")
    layer.MemoryNote = MemoryNote
    monkeypatch.setitem(sys.modules, "agentic_memory.memory_system", layer)

    backend = AMemSmokeBackend()
    result = run_smoke(
        manifest=_smoke_source(),
        results_dir=tmp_path,
        limit_base=1,
        memory_system="amem",
        retrieval_k=1,
        adapter_factory=lambda config: AMemAdapter(backend),
    )
    records = result["records"]
    assert len(records) == 3
    assert not backend.retriever.search_calls
    assert backend.search_agentic_calls and backend.search_agentic_calls[0][1] == 1
    assert all(record["memory_system"] == "amem" for record in records)
    assert all(record["raw_insertion_used"] is True for record in records)
    assert all(record["agentic_evolution_disabled"] is True for record in records)
    assert all(record["llm_backend_constructor"] == "openai" for record in records)
    assert all(record["amem_backend_variant"] == "official_chroma" for record in records)
    assert all(record["native_index_backend"] == "ChromaDB" for record in records)
    assert all(record["generated_metadata_disabled"] is True for record in records)
    assert all(record["llm_constructor_initialized"] is True for record in records)
    assert all(record["native_retrieval_surface"] == "AgenticMemorySystem.search_agentic" for record in records)
    assert len({tuple(record["namespace"]) for record in records}) == 1
    assert all(record["native_retrieval_status"] == "native_retrieval_success" for record in records)
    assert all(record["governance_applied_after_native_retrieval"] is True for record in records)
    assert all(record["requested_retrieval_k"] == 1 for record in records)
