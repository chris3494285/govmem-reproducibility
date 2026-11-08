#!/usr/bin/env python3
"""Build a native-evidence v3 repair without changing the frozen v2 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
V3 = ROOT / "data/final_main_governance_condition_manifest_v3.jsonl"
FREEZE = ROOT / "data/final_main_governance_condition_manifest_v3.freeze.json"
OUT_DIR = ROOT / "results/answer_leakage_repair"
AUDIT = ROOT / "results/answer_leakage_audit.json"
SCHEMA = "main-governance-condition-v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def native_long_items(raw: dict[str, Any], evidence_ids: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    """Return only official native user messages, never the oracle answer field."""
    wanted = {str(value) for value in evidence_ids}
    dates = raw.get("haystack_dates", [])
    session_ids = raw.get("haystack_session_ids", [])
    candidates: list[tuple[str, str, int, str, bool]] = []
    for session_index, session in enumerate(raw.get("haystack_sessions", [])):
        session_id = str(session_ids[session_index])
        if session_id not in wanted:
            continue
        date = str(dates[session_index]) if session_index < len(dates) else ""
        for message_index, message in enumerate(session):
            if message.get("role", "unknown") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content:
                candidates.append((date, f"{raw['question_id']}::{session_id}::{message_index}", message_index, content, bool(message.get("has_answer"))))
    answer_candidates = [item for item in candidates if item[4]]
    selected = answer_candidates if answer_candidates else candidates
    selected.sort(key=lambda item: (item[0], item[1]))
    if not selected:
        raise ValueError(f"No native user-message evidence for {raw['question_id']}")
    items = [item[3] for item in selected]
    locators = [
        {
            "source_file": "longmemeval-cleaned/longmemeval_oracle.json",
            "question_id": raw["question_id"],
            "session_id": item[1].split("::", 2)[1],
            "message_index": item[2],
            "role": "user",
            "content_sha256": hashlib.sha256(item[3].encode("utf-8")).hexdigest(),
            "has_answer_marker": item[4],
        }
        for item in selected
    ]
    return items, locators


def condition_items(operator: str | None, condition_type: str, current: list[str]) -> list[str]:
    if condition_type in {"clean", "enforced"}:
        return current
    if operator == "missing":
        return current[1:] if len(current) > 1 else []
    if operator == "stale":
        return current[:-1]
    raise ValueError(f"Unexpected LongMemEval affected operator: {operator!r}")


def _context(row: dict[str, Any]) -> str:
    value = row.get("model_visible_context_text")
    return value if isinstance(value, str) else ""


def prompt_hash(row: dict[str, Any]) -> str:
    # Keep this in sync with the zero-API runner's deterministic prompt renderer.
    try:
        from scripts.run_final_main_smoke_deepseek import render_prompt
    except ModuleNotFoundError:  # direct script execution
        from run_final_main_smoke_deepseek import render_prompt

    return hashlib.sha256(render_prompt(row).encode("utf-8")).hexdigest()


def build(source: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"Official LongMemEval source is missing: {source}")
    old_lines = V2.read_text(encoding="utf-8").splitlines(keepends=True)
    old_rows = [json.loads(line) for line in old_lines if line.strip()]
    old_line_by_id = {row["condition_id"]: line for row, line in zip(old_rows, old_lines) if line.strip()}
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    affected = set(audit["affected_base_ids"])
    raw_by_id = {str(row["question_id"]): row for row in json.loads(source.read_text(encoding="utf-8"))}
    missing = sorted(base_id.split("::", 1)[1] for base_id in affected if base_id.split("::", 1)[1] not in raw_by_id)
    if missing:
        raise ValueError(f"Missing official native source rows: {missing}")

    provenance: dict[str, Any] = {}
    native_by_base: dict[str, list[str]] = {}
    for base_id in sorted(affected):
        source_id = base_id.split("::", 1)[1]
        old = next(row for row in old_rows if row["base_id"] == base_id)
        evidence_ids = [str(value) for value in old.get("source_evidence", {}).get("evidence_session_ids", [])]
        items, locators = native_long_items(raw_by_id[source_id], evidence_ids)
        native_by_base[base_id] = items
        provenance[base_id] = {
            "source_dataset": "longmemeval",
            "source_example_id": source_id,
            "source_file": "longmemeval-cleaned/longmemeval_oracle.json",
            "source_file_sha256": sha256(source),
            "evidence_session_ids": evidence_ids,
            "native_message_locators": locators,
            "construction": "official native user-message content selected by evidence_session_ids; expected_answer is not read for context construction",
            "oracle_answer_substitution": False,
        }

    new_rows: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    for old in old_rows:
        row = deepcopy(old)
        if row["base_id"] in affected:
            current = native_by_base[row["base_id"]]
            items = condition_items(row.get("fault_operator"), row["condition_type"], current)
            row["model_visible_memory_items"] = items
            row["model_visible_context_text"] = "\n".join(items)
            row["model_visible_context_spec"] = {
                "content_source": "official_longmemeval_native_evidence",
                "item_count": len(items),
                "projection": "admitted_evidence",
                "oracle_answer_substitution": False,
            }
            row["source_provenance"] = provenance[row["base_id"]]
            row["condition_schema_version"] = SCHEMA
        new_rows.append(row)

    V3.parent.mkdir(parents=True, exist_ok=True)
    with V3.open("w", encoding="utf-8") as handle:
        for row in new_rows:
            if row["base_id"] not in affected:
                handle.write(old_line_by_id[row["condition_id"]])
            else:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    # The comparison is deliberately limited to every sibling of every affected base.
    old_by_id = {row["condition_id"]: row for row in old_rows}
    new_by_id = {row["condition_id"]: row for row in new_rows}
    for base_id in sorted(affected):
        for condition_id, old in old_by_id.items():
            if old["base_id"] != base_id:
                continue
            new = new_by_id[condition_id]
            old_context = _context(old)
            new_context = _context(new)
            old_prompt = prompt_hash(old)
            new_prompt = prompt_hash(new)
            if old_context != new_context or old_prompt != new_prompt:
                changed.append({
                    "condition_id": condition_id,
                    "base_id": base_id,
                    "source_dataset": new["source_dataset"],
                    "condition_type": new["condition_type"],
                    "fault_operator": new.get("fault_operator"),
                    "event_type": new.get("primary_event_type"),
                    "old_context_sha256": hashlib.sha256(old_context.encode("utf-8")).hexdigest(),
                    "new_context_sha256": hashlib.sha256(new_context.encode("utf-8")).hexdigest(),
                    "old_prompt_sha256": old_prompt,
                    "new_prompt_sha256": new_prompt,
                    "source_provenance": provenance[base_id],
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "changed_conditions.jsonl").open("w", encoding="utf-8") as handle:
        for item in changed:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    (OUT_DIR / "changed_condition_ids.txt").write_text("".join(item["condition_id"] + "\n" for item in changed), encoding="utf-8")
    (OUT_DIR / "source_evidence_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    provider_dir = OUT_DIR / "provider_rerun_manifests"
    provider_dir.mkdir(parents=True, exist_ok=True)
    changed_ids = {item["condition_id"] for item in changed}
    for provider in ("openai", "anthropic", "deepseek", "gemini", "kimi", "grok"):
        with (provider_dir / f"{provider}.jsonl").open("w", encoding="utf-8") as handle:
            for row in new_rows:
                if row["condition_id"] in changed_ids:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    condition_counts = Counter(item["condition_type"] for item in changed)
    fault_counts = Counter(str(item["fault_operator"]) for item in changed)
    event_counts = Counter(item["event_type"] for item in changed)
    source_counts = Counter(item["source_dataset"] for item in changed)
    summary = {
        "status": "NATIVE_EVIDENCE_REPAIR_PENDING_PROVIDER_RERUN",
        "submission_ready": False,
        "leakage_gate_status": "PASS_FOR_V3_MANIFEST",
        "provider_rerun_required": True,
        "v2_manifest": str(V2.relative_to(ROOT)),
        "v2_manifest_sha256": sha256(V2),
        "v3_manifest": str(V3.relative_to(ROOT)),
        "v3_manifest_sha256": sha256(V3),
        "official_source_locator": "longmemeval-cleaned/longmemeval_oracle.json",
        "official_source_sha256": sha256(source),
        "corrected_base_count": len(affected),
        "changed_condition_row_count": len(changed),
        "changed_prompt_count": sum(item["old_prompt_sha256"] != item["new_prompt_sha256"] for item in changed),
        "changed_counts_by_condition_type": dict(sorted(condition_counts.items())),
        "changed_counts_by_fault_operator": dict(sorted(fault_counts.items())),
        "changed_counts_by_event_type": dict(sorted(event_counts.items())),
        "changed_counts_by_source": dict(sorted(source_counts.items())),
        "providers": ["openai", "anthropic", "deepseek", "gemini", "kimi", "grok"],
        "required_reruns_per_provider": len(changed),
        "total_required_requests_across_six_providers": len(changed) * 6,
        "provider_rerun_manifest_dir": str(provider_dir.relative_to(ROOT)),
        "affected_paper_results": [
            "clean/corrupted/enforced accuracy",
            "corruption effect",
            "recovery effect",
            "LongMemEval source analysis",
            "missing/stale fault analysis",
            "event-level macro results",
            "six-provider aggregate",
            "control-plane propagation diagnostic",
            "matched-fault robustness",
            "all affected tables and figures",
        ],
        "api_calls_made": False,
    }
    (OUT_DIR / "repair_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze = {
        "manifest_filename": V3.name,
        "manifest_sha256": summary["v3_manifest_sha256"],
        "supersedes_manifest": V2.name,
        "superseded_manifest_sha256": summary["v2_manifest_sha256"],
        "official_source_locator": summary["official_source_locator"],
        "official_source_sha256": summary["official_source_sha256"],
        "schema_version": SCHEMA,
        "statement": "Only official native LongMemEval user-message content was used for the 53 corrected bases; no provider APIs were called and expected_answer was not used to construct context.",
        "rerun_status": "pending",
    }
    FREEZE.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Official local LongMemEval oracle JSON")
    args = parser.parse_args()
    build(args.source)


if __name__ == "__main__":
    main()
