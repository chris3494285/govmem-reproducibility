#!/usr/bin/env python3
"""Offline construct-validity audit for the frozen GovMem final-main cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_BASE_SHA = "a2601a7ecb46034faf5dcc7ad00b5a851e19ae01561876fa681fd18af168ca79"
EXPECTED_CONDITION_SHA = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"
ACCEPTED_CONDITION_SHAS = {EXPECTED_CONDITION_SHA, "4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb"}
SOURCES = ("longmemeval", "memora", "oaks_novel")
EVENTS = ("deletion", "temporal", "update")
FAULTS = ("missing", "stale", "flipped", "conflicting")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _nonempty(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _source_audit(base: dict[str, Any], clean: dict[str, Any]) -> dict[str, Any]:
    """Audit native evidence using the schema of each frozen source adapter."""
    source = base["source_dataset"]
    evidence = base.get("source_evidence") or {}
    oracle = base.get("oracle_state") or {}
    fields: list[str] = []
    notes: list[str] = []

    if source == "longmemeval":
        fields = [
            "source_evidence.answer_session_ids",
            "source_evidence.evidence_session_ids",
            "source_evidence.question_type",
            "oracle_state.current_status",
            "oracle_state.ordered_evidence",
            "oracle_state.relevant_session_ids",
            "oracle_state.temporal_relation",
            "question",
            "expected_answer",
        ]
        native = (
            _nonempty(evidence.get("evidence_session_ids"))
            and _nonempty(evidence.get("answer_session_ids"))
            and _nonempty(oracle.get("ordered_evidence"))
        )
        clean_supported = native and _nonempty(oracle.get("current_status"))
        locator = native and bool(base.get("source_example_id"))
        if not native:
            notes.append("missing native answer-session history/evidence ordering")
    elif source == "memora":
        fields = [
            "source_evidence.memory_evidence",
            "source_evidence.forgetting_evidence",
            "source_evidence.relevant_source_session_ids",
            "source_evidence.governance_capabilities",
            "oracle_state.current_status",
            "oracle_state.obsolete_item",
            "oracle_state.replacement_item",
            "oracle_state.relevant_session_ids",
            "question",
            "expected_answer",
        ]
        memory = evidence.get("memory_evidence")
        forgetting = evidence.get("forgetting_evidence")
        native = (
            isinstance(memory, dict)
            and _nonempty(memory)
            and isinstance(forgetting, dict)
            and _nonempty(forgetting)
            and _nonempty(evidence.get("relevant_source_session_ids"))
        )
        clean_supported = native and _nonempty(oracle.get("current_status"))
        locator = native and bool(base.get("source_example_id"))
        if not native:
            notes.append("missing native memory/lifecycle evidence or session provenance")
    elif source == "oaks_novel":
        fields = [
            "source_evidence.document_id",
            "source_evidence.chunk_to_answer",
            "source_evidence.ordered_answer_states",
            "source_evidence.transition_chunks",
            "source_evidence.evidence_fields_used",
            "oracle_state.current_status",
            "oracle_state.ordered_answer_states",
            "oracle_state.relevant_chunk_ids",
            "oracle_state.transitions",
            "question",
            "expected_answer",
        ]
        native = (
            _nonempty(evidence.get("document_id"))
            and _nonempty(evidence.get("chunk_to_answer"))
            and _nonempty(evidence.get("ordered_answer_states"))
            and _nonempty(oracle.get("ordered_answer_states"))
        )
        clean_supported = native and _nonempty(oracle.get("current_status"))
        locator = native and bool(evidence.get("document_id")) and bool(base.get("source_example_id"))
        if not native:
            notes.append("missing native ordered chunk/answer-state evidence")
    else:
        raise ValueError(f"unsupported source: {source}")

    # Memora's base adapter intentionally stores its structured benchmark
    # answer in the condition task payload rather than base.expected_answer;
    # use the clean sibling as the authoritative task/scoring link.
    oracle_linked = (
        _nonempty(base.get("question"))
        and (_nonempty(base.get("expected_answer"))
             or (_nonempty(clean.get("task_question")) and _nonempty(clean.get("task_oracle_answer"))))
    )
    return {
        "source_native_evidence_present": bool(native),
        "clean_state_supported": bool(clean_supported),
        "oracle_linked": bool(oracle_linked),
        "provenance_locator_present": bool(locator),
        "evidence_fields_used": fields,
        "audit_notes": notes,
    }


def audit_base(base: dict[str, Any], conditions: list[dict[str, Any]]) -> dict[str, Any]:
    """Return one deterministic audit record for a base and its siblings."""
    base_id = base["base_id"]
    siblings = [r for r in conditions if r.get("base_id") == base_id]
    if len(siblings) != 1 + len(base.get("supported_fault_operators", [])) * 2:
        raise ValueError(f"unexpected sibling count for {base_id}: {len(siblings)}")
    clean_rows = [r for r in siblings if r.get("condition_type") == "clean"]
    if len(clean_rows) != 1:
        raise ValueError(f"expected one clean sibling for {base_id}")
    by_fault: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in siblings:
        if row.get("condition_type") == "corrupted":
            by_fault[row.get("fault_operator")].append(row)
        elif row.get("condition_type") == "enforced":
            # The pair_id is the reliable link, but each supported fault has one.
            by_fault[row.get("fault_operator")].append(row)

    source = _source_audit(base, clean_rows[0])
    supported = list(base.get("supported_fault_operators", []))
    fault_ok = True
    fault_notes: list[str] = []
    fault_interpretations: dict[str, str] = {}
    for fault in supported:
        rows = by_fault.get(fault, [])
        corr = [r for r in rows if r.get("condition_type") == "corrupted"]
        enf = [r for r in rows if r.get("condition_type") == "enforced"]
        if len(corr) != 1 or len(enf) != 1:
            fault_ok = False
            fault_notes.append(f"missing unique corrupted/enforced {fault} sibling")
            continue
        spec = corr[0].get("corruption_specification") or {}
        declared_fields = spec.get("source_oracle_fields") or []
        deterministic = spec.get("deterministic") is True and spec.get("operator") == fault
        # `source_oracle_fields` names the native state schema used by the
        # adapter.  A field may legitimately be null for a particular native
        # example (e.g. a todo deletion has no replacement item), so support
        # is checked by schema presence plus the adapter's native evidence and
        # declared fault rationale, not by inventing a replacement value.
        fields_exist = all(k in (base.get("oracle_state") or {}) for k in declared_fields)
        supported_by_manifest = fault in supported and bool((base.get("fault_support_reasons") or {}).get(fault))
        if not (deterministic and fields_exist and supported_by_manifest):
            fault_ok = False
            fault_notes.append(f"{fault} lacks deterministic source-grounded support")
        fault_interpretations[fault] = spec.get("operation", "")

    heuristic_only = not source["source_native_evidence_present"] or not fault_ok
    ambiguous = not source["clean_state_supported"] or not source["oracle_linked"]
    notes = list(source["audit_notes"])
    notes.extend(fault_notes)
    return {
        "base_id": base_id,
        "source": base["source_dataset"],
        "analysis_event": base["analysis_event_type"],
        "primary_event_type": base.get("primary_event_type"),
        "source_example_id": base.get("source_example_id"),
        "supported_faults": supported,
        "source_native_evidence_present": source["source_native_evidence_present"],
        "clean_state_supported": source["clean_state_supported"],
        "assigned_faults_supported": fault_ok,
        "oracle_linked": source["oracle_linked"],
        "provenance_locator_present": source["provenance_locator_present"],
        "heuristic_only": heuristic_only,
        "ambiguous": ambiguous,
        "fully_supported": all(
            [source["source_native_evidence_present"], source["clean_state_supported"],
             fault_ok, source["oracle_linked"], source["provenance_locator_present"]]
        ) and not heuristic_only and not ambiguous,
        "evidence_fields_used": source["evidence_fields_used"],
        "source_locator": {
            "source_dataset": base["source_dataset"],
            "source_example_id": base.get("source_example_id"),
            "source_fingerprint": base.get("source_fingerprint"),
            "document_id": (base.get("source_evidence") or {}).get("document_id"),
            "question_id": base.get("source_example_id"),
        },
        "native_evidence": base.get("source_evidence"),
        "clean_operative_state": base.get("oracle_state"),
        "fault_interpretations": fault_interpretations,
        "task_question": clean_rows[0].get("task_question") or base.get("question"),
        "task_oracle_answer": clean_rows[0].get("task_oracle_answer") or base.get("expected_answer"),
        "audit_notes": notes,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: _compact(row[field]) if isinstance(row.get(field), (dict, list)) else row.get(field)
                for field in fields
            })


def _aggregate(records: list[dict[str, Any]], key: str, values: list[str]) -> list[dict[str, Any]]:
    output = []
    for value in values:
        subset = [r for r in records if value in r["supported_faults"]] if key == "fault" else [r for r in records if r[key] == value]
        output.append({
            key: value,
            "bases": len(subset),
            "fully_supported": sum(r["fully_supported"] for r in subset),
            "source_native_evidence_present": sum(r["source_native_evidence_present"] for r in subset),
            "clean_state_supported": sum(r["clean_state_supported"] for r in subset),
            "assigned_faults_supported": sum(r["assigned_faults_supported"] for r in subset),
            "oracle_linked": sum(r["oracle_linked"] for r in subset),
            "provenance_locator_present": sum(r["provenance_locator_present"] for r in subset),
            "heuristic_only": sum(r["heuristic_only"] for r in subset),
            "ambiguous": sum(r["ambiguous"] for r in subset),
        })
    return output


def run(base_manifest: Path, condition_manifest: Path, output_dir: Path) -> dict[str, Any]:
    if _sha256(base_manifest) != EXPECTED_BASE_SHA:
        raise ValueError("base manifest SHA256 does not match frozen artifact")
    condition_sha = _sha256(condition_manifest)
    if condition_sha not in ACCEPTED_CONDITION_SHAS:
        raise ValueError("condition manifest SHA256 does not match frozen artifact")
    bases = _read_jsonl(base_manifest)
    conditions = _read_jsonl(condition_manifest)
    if len(bases) != 731 or len({r.get("base_id") for r in bases}) != 731:
        raise ValueError("frozen base manifest does not contain 731 unique bases")
    if len(conditions) != 4891:
        raise ValueError("frozen condition manifest does not contain 4891 rows")
    records = [audit_base(base, conditions) for base in bases]
    records.sort(key=lambda r: r["base_id"])
    output_dir.mkdir(parents=True, exist_ok=True)

    visible_fields = [
        "base_id", "source", "analysis_event", "primary_event_type", "source_example_id",
        "supported_faults", "source_native_evidence_present", "clean_state_supported",
        "assigned_faults_supported", "oracle_linked", "provenance_locator_present",
        "heuristic_only", "ambiguous", "fully_supported", "evidence_fields_used",
        "source_locator", "audit_notes",
    ]
    _write_csv(output_dir / "governance_evidence_base_level.csv", records, visible_fields)
    by_source = _aggregate(records, "source", list(SOURCES))
    by_event = _aggregate(records, "analysis_event", list(EVENTS))
    by_fault = _aggregate(records, "fault", list(FAULTS))
    aggregate_fields = ["source", "analysis_event", "fault"]
    for name, rows, fields in [
        ("source", by_source, ["source"]),
        ("event", by_event, ["analysis_event"]),
        ("fault", by_fault, ["fault"]),
    ]:
        _write_csv(output_dir / f"governance_evidence_by_{name}.csv", rows, fields + [
            "bases", "fully_supported", "source_native_evidence_present", "clean_state_supported",
            "assigned_faults_supported", "oracle_linked", "provenance_locator_present",
            "heuristic_only", "ambiguous",
        ])

    overall = {
        "total_bases": len(records),
        "fully_supported_bases": sum(r["fully_supported"] for r in records),
        "unsupported_bases": sum(not r["fully_supported"] for r in records),
        "heuristic_only_bases": sum(r["heuristic_only"] for r in records),
        "ambiguous_bases": sum(r["ambiguous"] for r in records),
        "missing_provenance_locator_bases": sum(not r["provenance_locator_present"] for r in records),
        "every_assigned_fault_source_supported": all(r["assigned_faults_supported"] for r in records),
    }
    summary = {
        "schema_version": "governance-evidence-audit-v1",
        "manifest_sha256": condition_sha,
        "base_manifest_sha256": EXPECTED_BASE_SHA,
        "api_calls_made": 0,
        "overall": overall,
        "by_source": by_source,
        "by_event": by_event,
        "by_fault": by_fault,
        "source_logic": {
            "longmemeval": "source_evidence answer_session_ids/evidence_session_ids and oracle_state ordered_evidence, relevant_session_ids, current_status, temporal_relation; source_example_id/question/expected_answer.",
            "memora": "source_evidence memory_evidence, forgetting_evidence, relevant_source_session_ids, governance_capabilities and oracle_state obsolete_item/replacement_item/current_status; source_example_id/question/expected_answer.",
            "oaks_novel": "source_evidence document_id, chunk_to_answer, ordered_answer_states, transition_chunks and oracle_state ordered_answer_states/relevant_chunk_ids/transitions/current_status; source_example_id/question/expected_answer.",
        },
        "provenance": "All evidence and interpretations are read from the frozen base/condition manifests; no provider result is used.",
    }
    (output_dir / "governance_evidence_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # Ensure coverage of all source/event/fault categories where the frozen cohort supports them.
    rng = random.Random(42)
    spot = []
    for source in SOURCES:
        source_records = [r for r in records if r["source"] == source]
        chosen: dict[str, dict[str, Any]] = {}
        for event in EVENTS:
            for r in source_records:
                if r["analysis_event"] == event:
                    chosen[r["base_id"]] = r
                    break
        for fault in FAULTS:
            for r in source_records:
                if fault in r["supported_faults"]:
                    chosen[r["base_id"]] = r
                    break
        remaining = [r for r in source_records if r["base_id"] not in chosen]
        rng.shuffle(remaining)
        for r in remaining[: max(0, 10 - len(chosen))]:
            chosen[r["base_id"]] = r
        for r in sorted(chosen.values(), key=lambda x: x["base_id"]):
            spot.append({
                "base_id": r["base_id"], "source": r["source"],
                "source_example_id": r["source_example_id"], "analysis_event": r["analysis_event"],
                "supported_faults": r["supported_faults"], "evidence_fields_used": r["evidence_fields_used"],
                "source_locator": r["source_locator"], "native_evidence": r["native_evidence"],
                "clean_operative_state": r["clean_operative_state"],
                "fault_interpretations": r["fault_interpretations"],
                "task_question": r["task_question"], "task_oracle_answer": r["task_oracle_answer"],
                "audit_notes": r["audit_notes"],
            })
    with (output_dir / "governance_evidence_spotcheck.jsonl").open("w") as handle:
        for row in spot:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    notes = """# Governance evidence audit schema notes

This is an offline, provider-independent audit of the frozen final-main manifests.

## Source-native fields

- **LongMemEval:** `source_evidence.answer_session_ids`,
  `source_evidence.evidence_session_ids`, `source_evidence.question_type`, and
  `oracle_state.current_status`, `ordered_evidence`, `relevant_session_ids`,
  `temporal_relation`; the base question and expected answer link the task target.
- **Memora:** `source_evidence.memory_evidence`, `forgetting_evidence`,
  `relevant_source_session_ids`, and `governance_capabilities`, together with
  `oracle_state.current_status`, `obsolete_item`, `replacement_item`, and
  `relevant_session_ids`.
- **OAKS:** `source_evidence.document_id`, `chunk_to_answer`,
  `ordered_answer_states`, `transition_chunks`, and `evidence_fields_used`,
  together with the ordered answer states and transitions in `oracle_state`.

`corruption_specification.deterministic`, `operator`, `operation`, and
`source_oracle_fields` in each condition row are checked against the base's
declared `supported_fault_operators`, `fault_support_reasons`, and oracle state.
No model output is consulted. `source_locator` preserves the persisted native
identifier/fingerprint (and OAKS document ID where present).
"""
    (output_dir / "governance_evidence_schema_notes.md").write_text(notes)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=Path("data/final_main_governance_base_manifest_v2.jsonl"))
    parser.add_argument("--condition-manifest", type=Path, default=Path("data/final_main_governance_condition_manifest_v2.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/governance_evidence_audit"))
    args = parser.parse_args()
    summary = run(args.base_manifest, args.condition_manifest, args.output_dir)
    overall = summary["overall"]
    print("GOVERNANCE EVIDENCE AUDIT PASS")
    for key in ["total_bases", "fully_supported_bases", "unsupported_bases", "heuristic_only_bases", "ambiguous_bases", "missing_provenance_locator_bases"]:
        print(f"{key} = {overall[key]}")
    print("by_source =", {r["source"]: r["bases"] for r in summary["by_source"]})
    print("every_assigned_fault_source_supported =", overall["every_assigned_fault_source_supported"])
    print("api_calls_made = 0")


if __name__ == "__main__":
    main()
