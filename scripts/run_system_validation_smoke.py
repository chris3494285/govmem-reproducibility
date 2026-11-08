#!/usr/bin/env python3
"""Run the NON-FROZEN smoke cohort through real Mem0, LangMem, or A-Mem storage, stopping at context."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from govmem.governance.enforcement import decide_admission  # noqa: E402
from govmem.memory_systems.base import MemoryRetrievalResult, MemorySystemAdapter  # noqa: E402
from govmem.memory_systems.harness import _content, _model_context, project_retrieved_context, validate_paired_invariants  # noqa: E402
from govmem.memory_systems.mem0 import Mem0Adapter  # noqa: E402
from govmem.memory_systems.langmem import LangMemAdapter  # noqa: E402
from govmem.memory_systems.amem import AMemAdapter  # noqa: E402


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("System-validation manifest must be an object with a top-level records list")
    if payload.get("status") != "NON-FROZEN / SMOKE":
        raise ValueError("This runner accepts only the NON-FROZEN / SMOKE manifest")
    records = payload["records"]
    if not all(isinstance(row, dict) for row in records):
        raise ValueError("System-validation manifest records must be JSON objects")
    return records


def _base_id(row: dict[str, Any]) -> str:
    return str(row.get("base_example_id") or row.get("example_id"))


def _user_id(base_example_id: str) -> str:
    return Mem0Adapter.user_id_for_base(base_example_id)


def _namespace(base_example_id: str) -> tuple[str, str]:
    return LangMemAdapter.namespace_for_base(base_example_id)


def _governance_action(row: dict[str, Any]) -> str:
    result = decide_admission(row)
    return "governance_admit" if result.actual_admission == "allow" else "governance_deny"


def _underlying_contents(row: dict[str, Any]) -> list[str]:
    return [_content(item) for item in row.get("memory") or []]


def _underlying_hash(contents: list[str]) -> str:
    payload = json.dumps(contents, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _native_status(result: MemoryRetrievalResult | None, error: Exception | None) -> str:
    if error is not None:
        return "native_retrieval_error"
    return "native_retrieval_success" if result and result.memories else "native_retrieval_empty"


def _condition_record(
    row: dict[str, Any],
    *,
    adapter: MemorySystemAdapter,
    user_id: str,
    namespace: tuple[str, ...] | None,
    underlying_contents: list[str],
    native_result: MemoryRetrievalResult | None,
    native_error: Exception | None,
    reused: bool,
    reused_from: str | None,
    system_config: dict[str, Any],
    mem0_package_version: str | None,
    langmem_package_version: str | None,
    langgraph_package_version: str | None,
    amem_package_version: str | None,
    native_insertion_path: str | None,
    raw_insertion_used: bool | None,
    agentic_evolution_disabled: bool | None,
    retrieval_k: int | None,
    backend_configuration: dict[str, Any] | None,
    native_retrieval_surface: str | None,
    retrieval_mode: str | None,
    llm_backend_constructor: str | None,
    requested_retrieval_k: int | None,
    mem0_embedding_transport: str | None,
    amem_backend_variant: str | None,
    native_index_backend: str | None,
    generated_metadata_disabled: bool | None,
    llm_constructor_initialized: bool | None,
    store_backend: str,
    embedding_config: dict[str, Any] | None,
) -> dict[str, Any]:
    query = str(row.get("task", ""))
    native_memories = native_result.memories if native_result is not None else ()
    native_contents = [memory.content for memory in native_memories]
    native_ids = [memory.memory_id for memory in native_memories]
    native_scores = [memory.score for memory in native_memories]
    governance_action = _governance_action(row)
    # This is intentionally after native retrieval.  Governance changes only
    # whether the captured native result becomes model-visible, with the
    # deterministic temporal projection applied to that captured result.
    admission = decide_admission(row)
    projected_result, projection = project_retrieved_context(row, native_result or MemoryRetrievalResult(
        memory_system=adapter.memory_system, query=query, memories=(), raw_response=None
    ), admission)
    visible = projected_result if governance_action == "governance_admit" else MemoryRetrievalResult(
        memory_system=adapter.memory_system,
        query=query,
        memories=(),
        raw_response=native_result.raw_response if native_result else None,
    )
    context = _model_context(visible)
    fault = row.get("fault_spec") or {}
    error = None if native_error is None else {"type": type(native_error).__name__, "message": str(native_error)}
    return {
        "memory_system": adapter.memory_system,
        "base_example_id": _base_id(row),
        "source_dataset": row.get("source_dataset"),
        "governance_dimension": row.get("dimension"),
        "condition": row.get("condition_type"),
        "fault_type": "clean" if row.get("condition_type") == "clean" else fault.get("operator"),
        "fault_direction": "clean" if row.get("condition_type") == "clean" else fault.get("direction"),
        "run_id": row.get("run_id"),
        "user_id": user_id,
        "namespace": list(namespace) if namespace is not None else None,
        "query": query,
        "underlying_memory_sha256": _underlying_hash(underlying_contents),
        "underlying_memory_content": underlying_contents,
        "native_retrieval_status": _native_status(native_result, native_error),
        "native_retrieved_memory_count": len(native_memories),
        "native_retrieved_memory_ids": native_ids,
        "native_retrieved_contents": native_contents,
        "native_retrieval_scores": native_scores,
        "native_raw_payload": _json_safe(native_result.raw_response if native_result else None),
        "native_retrieval_reused": reused,
        "native_retrieval_reused_from": reused_from,
        "observed_governance_state": row.get("observed_governance_state"),
        "governance_action": governance_action,
        "governance_applied_after_native_retrieval": True,
        "operative_memory_role": projection["operative_memory_role"],
        "operative_memory_value": projection["operative_memory_value"],
        "projection_reason": projection["projection_reason"],
        "final_model_visible_context": context,
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "adapter_version": adapter.adapter_version,
        "mem0_package_version": mem0_package_version,
        "langmem_package_version": langmem_package_version,
        "langgraph_package_version": langgraph_package_version,
        "amem_package_version": amem_package_version,
        "native_insertion_path": native_insertion_path,
        "raw_insertion_used": raw_insertion_used,
        "agentic_evolution_disabled": agentic_evolution_disabled,
        "retrieval_k": retrieval_k,
        "backend_configuration": backend_configuration,
        "native_retrieval_surface": native_retrieval_surface,
        "retrieval_mode": retrieval_mode,
        "llm_backend_constructor": llm_backend_constructor,
        "amem_backend_variant": amem_backend_variant,
        "native_index_backend": native_index_backend,
        "generated_metadata_disabled": generated_metadata_disabled,
        "llm_constructor_initialized": llm_constructor_initialized,
        "requested_retrieval_k": requested_retrieval_k,
        "mem0_embedding_transport": mem0_embedding_transport,
        "store_backend": store_backend,
        "embedding_config": embedding_config,
        "system_config": system_config,
        "error": error,
    }


def _native_reuse_invariant_failures(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return base-level mismatches in captured native sibling retrieval."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["base_example_id"])].append(record)
    failures: list[dict[str, Any]] = []
    native_fields = ("native_retrieved_memory_ids", "native_retrieved_contents", "native_raw_payload")
    for base_id, group in groups.items():
        clean = next((record for record in group if record["condition"] == "clean"), None)
        if clean is None:
            failures.append({"base_example_id": base_id, "fields": ["clean_condition_missing"]})
            continue
        mismatched = [
            field for field in native_fields
            if any(record.get(field) != clean.get(field) for record in group)
        ]
        if mismatched:
            failures.append({"base_example_id": base_id, "fields": mismatched})
    return failures


def _validate_result_invariants(records: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["base_example_id"])].append(record)
    for base_id, group in groups.items():
        clean = next((record for record in group if record["condition"] == "clean"), None)
        if clean is None:
            raise ValueError(f"Base {base_id} lacks a clean sibling")
        for record in group:
            if record["base_example_id"] != base_id:
                raise ValueError(f"Base identity changed within group {base_id}")
            for field in ("query", "underlying_memory_sha256", "underlying_memory_content", "user_id", "namespace", "requested_retrieval_k"):
                if record[field] != clean[field]:
                    raise ValueError(f"Paired invariant failed for {base_id}: {field}")
            if record["governance_applied_after_native_retrieval"] is not True:
                raise ValueError(f"Governance stage ordering was not recorded for {base_id}")


def run_smoke(
    *,
    manifest: str | Path,
    results_dir: str | Path,
    limit_base: int | None = None,
    adapter_factory: Callable[[dict[str, Any]], MemorySystemAdapter] | None = None,
    system_config: dict[str, Any] | None = None,
    memory_system: str = "mem0",
    retrieval_k: int | None = None,
    rows_override: list[dict[str, Any]] | None = None,
    output_prefix: str = "system_validation_smoke",
    output_status: str = "NON-FROZEN / SMOKE",
    strict_retrieval: bool = False,
    result_metadata: dict[str, Any] | None = None,
    write_csv: bool = True,
) -> dict[str, Any]:
    if memory_system not in {"mem0", "langmem", "amem"}:
        raise ValueError("Phase S1 supports only --memory-system mem0, langmem, or amem")
    if retrieval_k is not None and retrieval_k <= 0:
        raise ValueError("retrieval_k must be a positive integer")
    rows = list(rows_override) if rows_override is not None else _load_manifest(manifest)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_base_id(row)].append(row)
    base_ids = sorted(groups)
    if limit_base is not None:
        if limit_base < 0:
            raise ValueError("--limit-base must be non-negative")
        base_ids = base_ids[:limit_base]
    if not base_ids:
        raise ValueError("No base examples selected")
    config = dict(system_config or {})
    if memory_system == "langmem":
        factory = adapter_factory or (lambda cfg: LangMemAdapter(config=cfg))
    elif memory_system == "amem":
        factory = adapter_factory or (lambda cfg: AMemAdapter(config=cfg))
    else:
        factory = adapter_factory or (lambda cfg: Mem0Adapter(config=cfg))
    adapter = factory(config)
    if adapter.memory_system != memory_system:
        raise ValueError(f"Requested {memory_system} but adapter is {adapter.memory_system}")
    mem0_package_version = Mem0Adapter.package_version()
    langmem_package_version, langgraph_package_version = LangMemAdapter.package_versions()
    output_records: list[dict[str, Any]] = []
    for base_id in base_ids:
        group = sorted(groups[base_id], key=lambda row: (row.get("condition_type") != "clean", row.get("run_id", "")))
        if len(group) < 2:
            raise ValueError(f"Base {base_id} does not contain clean/fault siblings")
        clean = next((row for row in group if row.get("condition_type") == "clean"), None)
        if clean is None:
            raise ValueError(f"Base {base_id} lacks clean sibling")
        for row in group:
            validate_paired_invariants(clean, row)
        user_id = _user_id(base_id)
        namespace = (_namespace(base_id) if memory_system == "langmem" else AMemAdapter.namespace_for_base(base_id) if memory_system == "amem" else None)
        contents = _underlying_contents(clean)
        query = str(clean.get("task", ""))
        try:
            adapter.reset()
            for item_index, (item, content) in enumerate(zip(clean.get("memory") or [], contents)):
                if memory_system in {"langmem", "amem"}:
                    adapter.add_memory(content, namespace=namespace, memory_id=f"memory-{item_index}", metadata={"govmem_source": item})
                else:
                    adapter.add_memory(content, user_id=user_id, metadata={"govmem_source": item}, infer=False)
            retrieve_kwargs = {"limit": retrieval_k} if retrieval_k is not None else {}
            native_result: MemoryRetrievalResult | None = (
                adapter.retrieve(query, namespace=namespace, **retrieve_kwargs) if memory_system in {"langmem", "amem"}
                else adapter.retrieve(query, user_id=user_id, **retrieve_kwargs)
            )
            native_error: Exception | None = None
        except Exception as exc:
            native_result = None
            native_error = exc
        clean_run_id = str(clean.get("run_id"))
        for index, row in enumerate(group):
            output_records.append(_condition_record(
                row,
                adapter=adapter,
                user_id=user_id,
                namespace=namespace,
                underlying_contents=contents,
                native_result=native_result,
                native_error=native_error,
                reused=index > 0,
                reused_from=clean_run_id if index > 0 else None,
                system_config=config,
                mem0_package_version=mem0_package_version,
                langmem_package_version=langmem_package_version,
                langgraph_package_version=langgraph_package_version,
                store_backend="InMemoryStore" if memory_system == "langmem" else "AgenticMemorySystem" if memory_system == "amem" else "Mem0 OSS",
                embedding_config=getattr(adapter, "embedding_config", getattr(adapter, "index_config", None)),
                amem_package_version=(getattr(adapter, "package_metadata", lambda: {})().get("amem_package_version") if memory_system == "amem" else None),
                native_insertion_path=(getattr(adapter, "package_metadata", lambda: {})().get("native_insertion_path") if memory_system == "amem" else None),
                raw_insertion_used=(getattr(adapter, "package_metadata", lambda: {})().get("raw_insertion_used") if memory_system == "amem" else None),
                agentic_evolution_disabled=(getattr(adapter, "package_metadata", lambda: {})().get("agentic_evolution_disabled") if memory_system == "amem" else None),
                retrieval_k=(getattr(adapter, "package_metadata", lambda: {})().get("retrieval_k") if memory_system == "amem" else None),
                backend_configuration=(getattr(adapter, "package_metadata", lambda: {})().get("backend_configuration") if memory_system == "amem" else None),
                native_retrieval_surface=(getattr(adapter, "package_metadata", lambda: {})().get("native_retrieval_surface") if memory_system == "amem" else None),
                retrieval_mode=(getattr(adapter, "package_metadata", lambda: {})().get("retrieval_mode") if memory_system == "amem" else None),
                llm_backend_constructor=(getattr(adapter, "package_metadata", lambda: {})().get("llm_backend_constructor") if memory_system == "amem" else None),
                amem_backend_variant=(getattr(adapter, "package_metadata", lambda: {})().get("amem_backend_variant") if memory_system == "amem" else None),
                native_index_backend=(getattr(adapter, "package_metadata", lambda: {})().get("native_index_backend") if memory_system == "amem" else None),
                generated_metadata_disabled=(getattr(adapter, "package_metadata", lambda: {})().get("generated_metadata_disabled") if memory_system == "amem" else None),
                llm_constructor_initialized=(getattr(adapter, "package_metadata", lambda: {})().get("llm_constructor_initialized") if memory_system == "amem" else None),
                requested_retrieval_k=retrieval_k,
                mem0_embedding_transport=(getattr(adapter, "mem0_embedding_transport", None) if memory_system == "mem0" else None),
            ))
    try:
        _validate_result_invariants(output_records)
        native_reuse_failures = _native_reuse_invariant_failures(output_records)
        if strict_retrieval and native_reuse_failures:
            details = "; ".join(
                f"{failure['base_example_id']}: {', '.join(failure['fields'])}"
                for failure in native_reuse_failures
            )
            raise RuntimeError(f"Formal sibling native-retrieval invariant failed: {details}")
        if retrieval_k is not None:
            clean_counts = [record["native_retrieved_memory_count"] for record in output_records if record["condition"] == "clean"]
            exceeded = [count for count in clean_counts if count > retrieval_k]
            if exceeded:
                raise ValueError(f"Native retrieval exceeded requested retrieval_k={retrieval_k}: counts={exceeded}")
        statuses = Counter(record["native_retrieval_status"] for record in output_records)
        if strict_retrieval and (statuses.get("native_retrieval_error", 0) or statuses.get("native_retrieval_empty", 0)):
            raise RuntimeError(
                "Formal system validation requires successful non-empty native retrieval; "
                f"errors={statuses.get('native_retrieval_error', 0)}, "
                f"empty={statuses.get('native_retrieval_empty', 0)}"
            )
        dimensions = Counter(f"{record['source_dataset']}:{record['governance_dimension']}" for record in output_records)
        summary = {
            "memory_system": memory_system,
            "manifest": str(manifest),
            "selected_base_examples": len(base_ids),
            "condition_rows": len(output_records),
            "native_retrieval_success_n": statuses.get("native_retrieval_success", 0),
            "native_retrieval_empty_n": statuses.get("native_retrieval_empty", 0),
            "native_retrieval_error_n": statuses.get("native_retrieval_error", 0),
            "requested_retrieval_k": retrieval_k,
            "max_native_retrieved_memory_count": max((record["native_retrieved_memory_count"] for record in output_records), default=0),
            "invariant_failure_n": len(native_reuse_failures),
            "per_dimension_counts": dict(sorted(dimensions.items())),
            "status": output_status,
        }
        if result_metadata:
            for record in output_records:
                record.update(result_metadata)
            summary.update(result_metadata)
        result = {"summary": summary, "records": output_records}
        output = Path(results_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{output_prefix}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if write_csv:
            fields = sorted({field for record in output_records for field in record})
            with (output / f"{output_prefix}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for record in output_records:
                    writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in record.items()})
        (output / f"{output_prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-system", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--limit-base", type=int, default=None)
    parser.add_argument("--retrieval-k", type=int, default=None, help="Optional positive native retrieval breadth budget")
    parser.add_argument("--mem0-config", help="JSON file passed to Mem0 OSS Memory.from_config")
    parser.add_argument("--langmem-config", help="JSON file containing LangGraph InMemoryStore index configuration")
    parser.add_argument("--amem-config", help="JSON file containing official A-Mem constructor configuration")
    args = parser.parse_args(argv)
    if args.memory_system not in {"mem0", "langmem", "amem"}:
        parser.error("Phase S1 supports only --memory-system mem0, langmem, or amem")
    if args.retrieval_k is not None and args.retrieval_k <= 0:
        parser.error("--retrieval-k must be a positive integer")
    config_path = {"mem0": args.mem0_config, "langmem": args.langmem_config, "amem": args.amem_config}[args.memory_system]
    config = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else {}
    result = run_smoke(manifest=args.manifest, results_dir=args.results_dir, limit_base=args.limit_base, system_config=config, memory_system=args.memory_system, retrieval_k=args.retrieval_k)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
