#!/usr/bin/env python3
"""Analyze deterministic RQ1 fault propagation from governance state to answers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Support both ``python scripts/analyze_rq3_error_propagation.py`` and
# ``python -m scripts.analyze_rq3_error_propagation`` from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_governance_fault_rq1 import load_manifest


PROVIDERS = ("deepseek", "openai", "anthropic")
RQ1A_MANIFEST = "data/governance_fault_rq1_pilot_frozen.jsonl"
RQ1B_MANIFEST = "data/governance_fault_rq1_restrictive_frozen.jsonl"
RQ1A_INFERENCE = "rq1a.jsonl"
RQ1A_SCORED = "rq1a_scored.jsonl"
RQ1B_INFERENCE = "rq1b_restrictive.jsonl"
RQ1B_SCORED = "rq1b_restrictive_scored.jsonl"


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Referenced RQ1 output is missing: {path}")
    result = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("run_id") is None:
            raise ValueError(f"Invalid RQ1 output row: {path}:{line_number}")
        run_id = str(row["run_id"])
        if run_id in result:
            raise ValueError(f"Duplicate RQ1 run_id in {path}: {run_id}")
        result[run_id] = row
    return result


def _fault_key(manifest_row: dict[str, Any]) -> tuple[str, str] | None:
    fault = manifest_row.get("fault_spec") or {}
    if manifest_row.get("condition_type") == "clean":
        return None
    if not fault.get("operator") or not fault.get("direction"):
        raise ValueError(f"Fault manifest row lacks operator/direction: {manifest_row.get('run_id')}")
    return str(fault["operator"]), str(fault["direction"])


def _text_candidates(value: Any) -> list[str]:
    if value is None:
        return []
    candidates = [value] if isinstance(value, str) else [json.dumps(value, ensure_ascii=False, sort_keys=True)]
    if isinstance(value, dict):
        for child in value.values():
            candidates.extend(_text_candidates(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            candidates.extend(_text_candidates(child))
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _contains_metadata_value(context: Any, value: Any) -> bool:
    if not isinstance(context, str):
        return False
    return any(candidate in context for candidate in _text_candidates(value))


def _target(metadata: dict[str, Any], dataset: str) -> Any:
    if dataset == "memora":
        return metadata.get("item")
    if dataset == "memconflict":
        return metadata.get("current_value")
    return None


def _context_mechanisms(manifest: dict[str, Any], clean: dict[str, Any], fault: dict[str, Any]) -> list[str]:
    metadata = manifest.get("source_metadata") or {}
    dataset = manifest.get("source_dataset")
    direction = (manifest.get("fault_spec") or {}).get("direction")
    clean_context, fault_context = clean.get("constructed_context"), fault.get("constructed_context")
    labels: list[str] = []
    if not isinstance(clean_context, str) or not isinstance(fault_context, str):
        return labels
    if direction == "unsafe":
        if dataset == "gatemem" and any(
            _contains_metadata_value(fault_context, target) and not _contains_metadata_value(clean_context, target)
            for target in (metadata.get("leak_targets") or [])
        ):
            labels.append("forbidden_content_exposed")
        if dataset == "memora" and _contains_metadata_value(fault_context, metadata.get("item")) and not _contains_metadata_value(clean_context, metadata.get("item")):
            labels.append("deleted_content_exposed")
        if dataset == "memconflict" and (manifest.get("fault_spec") or {}).get("operator") == "stale":
            previous = metadata.get("previous_value")
            if _contains_metadata_value(fault_context, previous) and not _contains_metadata_value(clean_context, previous):
                labels.append("stale_predecessor_promoted_or_exposed")
    if direction == "restrictive":
        target = _target(metadata, str(dataset))
        if _contains_metadata_value(clean_context, target) and not _contains_metadata_value(fault_context, target):
            labels.append("required_target_removed")
    return labels


def _admission_type(clean: dict[str, Any], fault: dict[str, Any], changed: bool | None) -> str:
    if changed is None:
        return "unscorable"
    clean_ids, fault_ids = set(clean.get("selected_memory_ids") or []), set(fault.get("selected_memory_ids") or [])
    added, removed = fault_ids - clean_ids, clean_ids - fault_ids
    if added and removed:
        return "operative_state_swapped"
    if added:
        return "unsafe_added"
    if removed:
        return "required_removed"
    return "no_change"


def _bool_change(clean: dict[str, Any], fault: dict[str, Any], field: str) -> bool | None:
    left, right = clean.get(field), fault.get(field)
    if isinstance(left, bool) and isinstance(right, bool):
        return left != right
    return None


def _pair_record(provider: str, track: str, manifest: dict[str, Any], clean_inference: dict[str, Any], fault_inference: dict[str, Any], clean_scored: dict[str, Any], fault_scored: dict[str, Any]) -> dict[str, Any]:
    clean_admission, fault_admission = clean_inference.get("actual_admission"), fault_inference.get("actual_admission")
    clean_ids, fault_ids = clean_inference.get("selected_memory_ids"), fault_inference.get("selected_memory_ids")
    decision_known = isinstance(clean_admission, str) and isinstance(fault_admission, str)
    set_known = isinstance(clean_ids, list) and isinstance(fault_ids, list)
    admission_changed = (
        (clean_admission != fault_admission if decision_known else False)
        or (set(clean_ids) != set(fault_ids) if set_known else False)
    ) if decision_known or set_known else None
    clean_context, fault_context = clean_inference.get("constructed_context"), fault_inference.get("constructed_context")
    context_changed = clean_context != fault_context if isinstance(clean_context, str) and isinstance(fault_context, str) else None
    direction = (manifest.get("fault_spec") or {}).get("direction")
    downstream_failure: bool | None
    clean_target_ok: bool | None = None
    fault_target_ok: bool | None = None
    if track == "rq1a":
        clean_gvr = clean_scored.get("governance_violation")
        fault_gvr = fault_scored.get("governance_violation")
        if isinstance(clean_gvr, bool) and isinstance(fault_gvr, bool):
            downstream_failure = clean_gvr is False and fault_gvr is True
        else:
            downstream_failure = None
    elif direction == "restrictive":
        clean_target_ok = clean_scored.get("mentions_required_target") if isinstance(clean_scored.get("mentions_required_target"), bool) else None
        fault_target_ok = fault_scored.get("mentions_required_target") if isinstance(fault_scored.get("mentions_required_target"), bool) else None
        if clean_target_ok is not True:
            downstream_failure = None
        else:
            value = fault_scored.get("downstream_false_denial")
            downstream_failure = value if isinstance(value, bool) else None
    else:
        downstream_failure = None
    mechanisms = _context_mechanisms(manifest, clean_inference, fault_inference)
    if direction == "unsafe":
        full = (
            True if admission_changed is True and context_changed is True and downstream_failure is True and mechanisms
            else False if downstream_failure is not None else None
        )
    else:
        full = (
            True if admission_changed is True and context_changed is True and clean_target_ok is True and fault_target_ok is False and downstream_failure is True
            else False if downstream_failure is not None else None
        )
    if downstream_failure is None:
        stop = "clean_failed" if direction == "restrictive" and clean_target_ok is not True else "unscorable_downstream"
    elif full:
        stop = "full_downstream_failure"
    elif admission_changed is False:
        stop = "fault_no_admission_change"
    elif admission_changed is True and context_changed is False:
        stop = "admission_change_no_context_change"
    elif context_changed is True and downstream_failure is False:
        stop = "context_change_no_downstream_failure"
    else:
        stop = "unscorable_admission"
    return {
        "provider": provider,
        "track": track,
        "source_dataset": manifest.get("source_dataset"),
        "governance_dimension": manifest.get("dimension"),
        "fault_type": (manifest.get("fault_spec") or {}).get("operator"),
        "fault_direction": direction,
        "base_example_id": manifest.get("example_id"),
        "admission_changed": admission_changed,
        "admission_change_type": _admission_type(clean_inference, fault_inference, admission_changed),
        "context_changed": context_changed,
        "context_mechanisms": mechanisms,
        "governance_violation_changed": _bool_change(clean_scored, fault_scored, "governance_violation"),
        "task_utility_changed": _bool_change(clean_scored, fault_scored, "task_utility"),
        "target_utility_changed": _bool_change(clean_scored, fault_scored, "target_utility"),
        "downstream_false_denial": fault_scored.get("downstream_false_denial") if isinstance(fault_scored.get("downstream_false_denial"), bool) else None,
        "downstream_failure": downstream_failure,
        "full_propagation": full,
        "propagation_stop": stop,
        "clean_target_ok": clean_target_ok,
        "fault_target_ok": fault_target_ok,
    }


def _pair_all(provider: str, track: str, manifest_rows: list[dict[str, Any]], inference: dict[str, dict[str, Any]], scored: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_by_run = {str(row["run_id"]): row for row in manifest_rows}
    if set(inference) != set(scored) or set(inference) != set(manifest_by_run):
        raise ValueError(f"{provider} {track} inference/scored/manifest cohorts do not align")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_id, manifest in manifest_by_run.items():
        groups[str(manifest["example_id"])].append(manifest)
    records = []
    for base, rows in sorted(groups.items()):
        clean = next((row for row in rows if row.get("condition_type") == "clean"), None)
        if clean is None:
            raise ValueError(f"Missing clean sibling for {base} in {track}")
        for fault in (row for row in rows if row.get("condition_type") == "fault"):
            clean_id, fault_id = str(clean["run_id"]), str(fault["run_id"])
            records.append(_pair_record(provider, track, fault, inference[clean_id], inference[fault_id], scored[clean_id], scored[fault_id]))
    return records


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[field] for field in ("provider", "source_dataset", "governance_dimension", "fault_type", "fault_direction"))].append(record)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        admission = [row["admission_changed"] for row in values if isinstance(row["admission_changed"], bool)]
        context = [row["context_changed"] for row in values if isinstance(row["context_changed"], bool)]
        failures = [row["downstream_failure"] for row in values if isinstance(row["downstream_failure"], bool)]
        full = [row["full_propagation"] for row in values if isinstance(row["full_propagation"], bool)]
        context_given_admission = [row["context_changed"] for row in values if row["admission_changed"] is True and isinstance(row["context_changed"], bool)]
        downstream_given_context = [row["downstream_failure"] for row in values if row["context_changed"] is True and isinstance(row["downstream_failure"], bool)]
        full_given_admission = [row["full_propagation"] for row in values if row["admission_changed"] is True and isinstance(row["full_propagation"], bool)]
        output.append({
            "provider": key[0], "source_dataset": key[1], "governance_dimension": key[2], "fault_type": key[3], "fault_direction": key[4],
            "paired_n": len(values),
            "admission_changed_n": sum(admission), "admission_scored_n": len(admission), "admission_change_rate": _rate(admission),
            "context_changed_n": sum(context), "context_scored_n": len(context), "context_change_rate": _rate(context),
            "downstream_failure_n": sum(failures), "downstream_scorable_n": len(failures), "downstream_failure_rate": _rate(failures),
            "full_propagation_n": sum(full), "full_scorable_n": len(full), "full_propagation_rate": _rate(full),
            "p_context_changed_given_admission_changed": _rate(context_given_admission), "context_given_admission_n": len(context_given_admission),
            "p_downstream_failure_given_context_changed": _rate(downstream_given_context), "downstream_given_context_n": len(downstream_given_context),
            "p_full_propagation_given_admission_changed": _rate(full_given_admission), "full_given_admission_n": len(full_given_admission),
            "mechanism_breakdown": {label: sum(row["propagation_stop"] == label for row in values) for label in (
                "fault_no_admission_change", "admission_change_no_context_change", "context_change_no_downstream_failure", "full_downstream_failure", "unscorable_downstream", "clean_failed", "unscorable_admission"
            )},
        })
    return output


def _cross_provider_stability(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        groups[tuple(row[field] for field in ("source_dataset", "governance_dimension", "fault_type", "fault_direction"))].append(row)
    result = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        provider_rates = [{"provider": row["provider"], "admission_change_rate": row["admission_change_rate"], "context_change_rate": row["context_change_rate"], "downstream_failure_rate": row["downstream_failure_rate"], "full_propagation_rate": row["full_propagation_rate"], "downstream_scorable_n": row["downstream_scorable_n"]} for row in rows]
        full_direction = [row["full_propagation_rate"] for row in rows if row["full_scorable_n"] > 0 and row["full_propagation_rate"] is not None]
        positive = sum(value > 0 for value in full_direction)
        zero = sum(value == 0 for value in full_direction)
        result.append({
            "source_dataset": key[0], "governance_dimension": key[1], "fault_type": key[2], "fault_direction": key[3],
            "providers": provider_rates,
            "providers_full_propagation_positive_n": positive,
            "providers_full_propagation_zero_n": zero,
            "providers_full_propagation_scorable_n": len(full_direction),
            "full_propagation_observed_in_all_scorable_providers": bool(full_direction) and positive == len(full_direction),
        })
    return result


def analyze(*, input_root: str | Path = "results/rq1", rq1a_manifest: str | Path = RQ1A_MANIFEST, rq1b_manifest: str | Path = RQ1B_MANIFEST) -> dict[str, Any]:
    root = Path(input_root)
    manifests = {"rq1a": load_manifest(rq1a_manifest, require_full=False), "rq1b": load_manifest(rq1b_manifest, require_full=False)}
    records = []
    source_files = {}
    for provider in PROVIDERS:
        provider_root = root / provider
        paths = {
            "rq1a_inference": provider_root / RQ1A_INFERENCE,
            "rq1a_scored": provider_root / RQ1A_SCORED,
            "rq1b_inference": provider_root / RQ1B_INFERENCE,
            "rq1b_scored": provider_root / RQ1B_SCORED,
        }
        source_files[provider] = {key: str(value) for key, value in paths.items()}
        for track in ("rq1a", "rq1b"):
            records.extend(_pair_all(provider, track, manifests[track], _load_jsonl(paths[f"{track}_inference"]), _load_jsonl(paths[f"{track}_scored"])))
    aggregates = _aggregate(records)
    return {"analysis_version": "rq3-error-propagation-v1", "providers": list(PROVIDERS), "source_files": source_files, "paired_records": records, "by_provider_fault": aggregates, "cross_provider_stability": _cross_provider_stability(aggregates)}


PAPER_COLUMNS = ["provider", "source_dataset", "governance_dimension", "fault_type", "fault_direction", "paired_n", "admission_change_rate", "context_change_rate", "downstream_failure_rate", "full_propagation_rate", "downstream_scorable_n"]


def write_csv(rows: list[dict[str, Any]], path: str | Path, columns: list[str] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = columns or sorted({field for row in rows for field in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="results/rq1")
    parser.add_argument("--output-dir", default="results/rq3")
    parser.add_argument("--rq1a-manifest", default=RQ1A_MANIFEST)
    parser.add_argument("--rq1b-manifest", default=RQ1B_MANIFEST)
    args = parser.parse_args(argv)
    summary = analyze(input_root=args.input_root, rq1a_manifest=args.rq1a_manifest, rq1b_manifest=args.rq1b_manifest)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "rq3_error_propagation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(summary["by_provider_fault"], output / "rq3_error_propagation.csv")
    write_csv(summary["paired_records"], output / "rq3_propagation_stages.csv")
    write_csv(summary["by_provider_fault"], output / "rq3_error_propagation.csv", PAPER_COLUMNS)
    print(json.dumps({"providers": summary["providers"], "paired_records": len(summary["paired_records"]), "groups": len(summary["by_provider_fault"]), "output_dir": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
