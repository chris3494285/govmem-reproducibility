"""Control-plane harness for future external memory-system validation runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

from govmem.governance.enforcement import decide_admission, select_memory

from .base import MemoryRetrieval, MemoryRetrievalResult, MemorySystemAdapter


@dataclass(frozen=True)
class SystemValidationTelemetry:
    memory_system: str
    base_example_id: str
    governance_dimension: str
    condition: str
    fault_type: str
    query: str
    retrieved_memory_count: int
    retrieved_memory_ids: tuple[str, ...]
    normalized_retrieved_contents: tuple[str, ...]
    retrieval_scores: tuple[float | None, ...]
    model_visible_context: str
    context_sha256: str
    operative_memory_role: str | None
    operative_memory_value: Any
    projection_reason: str
    adapter_version: str
    system_config: dict[str, Any]
    error: dict[str, Any] | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_paired_invariants(clean: dict[str, Any], fault: dict[str, Any]) -> None:
    for field in ("example_id", "source_dataset", "dimension", "task", "memory"):
        if field == "example_id":
            continue
        if clean.get(field) != fault.get(field):
            raise ValueError(f"Clean/fault pair changed invariant field {field!r}")
    if clean.get("example_id") != fault.get("example_id"):
        raise ValueError("Clean/fault rows do not share base example identity")


def _content(item: dict[str, Any]) -> str:
    for key in ("content", "text", "item", "value"):
        if key in item:
            value = item[key]
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return json.dumps(item, ensure_ascii=False, sort_keys=True)


def _model_context(result: MemoryRetrievalResult) -> str:
    return json.dumps([memory.content for memory in result.memories], ensure_ascii=False, sort_keys=True)


def project_retrieved_context(
    row: dict[str, Any],
    native_result: MemoryRetrievalResult,
    admission: Any,
) -> tuple[MemoryRetrievalResult, dict[str, Any]]:
    """Apply GovMem's deterministic post-retrieval operative-state projection.

    Native retrieval is never edited.  For corrected MemConflict temporal
    rows, the existing ``select_memory`` implementation supplies the frozen
    attribute binding and chooses current versus previous after retrieval.
    Other datasets/dimensions retain their native retrieved context exactly.
    """
    if not native_result.memories:
        return native_result, {"operative_memory_role": None, "operative_memory_value": None, "projection_reason": "native_retrieval_empty"}
    if row.get("source_dataset") != "memconflict" or row.get("dimension") != "temporal":
        return native_result, {"operative_memory_role": None, "operative_memory_value": None, "projection_reason": "no_temporal_projection"}

    selection = select_memory(row, admission)
    if not selection.selected_memory:
        return MemoryRetrievalResult(
            memory_system=native_result.memory_system,
            query=native_result.query,
            memories=(),
            raw_response=native_result.raw_response,
        ), {
            "operative_memory_role": None,
            "operative_memory_value": None,
            "projection_reason": selection.selection_reason,
        }

    source = native_result.memories[0]
    projected: list[MemoryRetrieval] = []
    for index, selected in enumerate(selection.selected_memory, start=1):
        role = selected.get("role")
        value = selected.get("value")
        projected.append(
            MemoryRetrieval(
                content=json.dumps(selected, ensure_ascii=False, sort_keys=True),
                memory_id=source.memory_id,
                rank=index,
                score=source.score,
                metadata={**source.metadata, "govmem_projection": selected},
                raw_payload=source.raw_payload,
            )
        )
    operator = (row.get("fault_spec") or {}).get("operator")
    role = "previous" if operator == "stale" else "current"
    value = projected[0].metadata["govmem_projection"].get("value")
    return MemoryRetrievalResult(
        memory_system=native_result.memory_system,
        query=native_result.query,
        memories=tuple(projected),
        raw_response=native_result.raw_response,
    ), {
        "operative_memory_role": role,
        "operative_memory_value": value,
        "projection_reason": selection.selection_reason,
    }


class SystemValidationHarness:
    """Runs only deterministic adapter/control-plane plumbing.

    No downstream model call is made.  The default admission decision is the
    existing GovMem deterministic enforcement decision; future policy variants
    can be injected without changing the adapter or frozen fault semantics.
    """

    def __init__(self, adapter: MemorySystemAdapter, *, system_config: dict[str, Any] | None = None, admission_fn: Callable[[dict[str, Any]], Any] = decide_admission):
        self.adapter = adapter
        self.system_config = dict(system_config or {})
        self.admission_fn = admission_fn

    def run_condition(self, row: dict[str, Any]) -> SystemValidationTelemetry:
        query = str(row.get("task", ""))
        condition = str(row.get("condition_type", ""))
        fault = row.get("fault_spec") or {}
        fault_type = "clean" if condition == "clean" else str(fault.get("operator"))
        try:
            self.adapter.reset()
            for item in row.get("memory") or []:
                self.adapter.add_memory(_content(item), metadata={"govmem_source": item})
            retrieval = self.adapter.retrieve(query)
            admission = self.admission_fn(row)
            projected, projection = project_retrieved_context(row, retrieval, admission)
            visible = projected if admission.actual_admission == "allow" else MemoryRetrievalResult(
                memory_system=retrieval.memory_system, query=query, memories=(), raw_response=retrieval.raw_response
            )
            context = _model_context(visible)
            return SystemValidationTelemetry(
                memory_system=self.adapter.memory_system,
                base_example_id=str(row.get("example_id")),
                governance_dimension=str(row.get("dimension")),
                condition=condition,
                fault_type=fault_type,
                query=query,
                retrieved_memory_count=len(retrieval.memories),
                retrieved_memory_ids=retrieval.retrieved_memory_ids,
                normalized_retrieved_contents=retrieval.normalized_retrieved_contents,
                retrieval_scores=tuple(memory.score for memory in retrieval.memories),
                model_visible_context=context,
                context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
                operative_memory_role=projection["operative_memory_role"],
                operative_memory_value=projection["operative_memory_value"],
                projection_reason=projection["projection_reason"],
                adapter_version=str(self.adapter.adapter_version),
                system_config=dict(self.system_config),
            )
        except Exception as exc:
            return SystemValidationTelemetry(
                memory_system=self.adapter.memory_system,
                base_example_id=str(row.get("example_id")),
                governance_dimension=str(row.get("dimension")),
                condition=condition,
                fault_type=fault_type,
                query=query,
                retrieved_memory_count=0,
                retrieved_memory_ids=(),
                normalized_retrieved_contents=(),
                retrieval_scores=(),
                model_visible_context="",
                context_sha256=hashlib.sha256(b"").hexdigest(),
                operative_memory_role=None,
                operative_memory_value=None,
                projection_reason="error",
                adapter_version=str(self.adapter.adapter_version),
                system_config=dict(self.system_config),
                error={"type": type(exc).__name__, "message": str(exc)},
            )

    def run_pair(self, clean: dict[str, Any], fault: dict[str, Any]) -> tuple[SystemValidationTelemetry, SystemValidationTelemetry]:
        validate_paired_invariants(clean, fault)
        return self.run_condition(clean), self.run_condition(fault)
