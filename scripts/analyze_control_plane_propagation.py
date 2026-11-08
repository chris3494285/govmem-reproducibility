#!/usr/bin/env python3
"""Offline diagnostic of governance-to-context-to-behavior propagation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
OUTPUT = ROOT / "results/control_plane_propagation"
EXPECTED_SHA = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"
ACCEPTED_SHAS = {EXPECTED_SHA, "4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb"}
EVENTS = ("deletion", "temporal", "update")
FAULTS = ("missing", "stale", "flipped", "conflicting")
SOURCES = ("longmemeval", "memora", "oaks_novel")
PROVIDER_RESULTS = {
    "deepseek": ROOT / "results/final_main_deepseek_v4_pro/scored_results.jsonl",
    "openai": ROOT / "results/final_main_openai_gpt56_sol_final_v2/scored/full_results_scored.jsonl",
    "anthropic": ROOT / "results/final_main_anthropic_sonnet5_final/scored/full_results_scored.jsonl",
    "gemini": ROOT / "results/final_main_gemini_3_flash_final/scored/full_results_scored.jsonl",
    "kimi": ROOT / "results/final_main_kimi_k2_6_nothinking_final/scored/full_results_scored.jsonl",
    "grok": ROOT / "results/final_main_grok_43_final/scored/full_results_scored.jsonl",
}


def _artifact_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def canonical_admission(row: dict[str, Any]) -> str:
    """Canonical A: ordered admitted memory-item projection persisted in v2."""
    items = row.get("model_visible_memory_items")
    if isinstance(items, list):
        return _canonical([_normalize_text(str(item)) for item in items])
    return _canonical(row.get("model_visible_context_spec") or {})


def canonical_context(row: dict[str, Any]) -> str:
    """Canonical C: semantic admitted items, preserving item membership/order."""
    items = row.get("model_visible_memory_items")
    if isinstance(items, list):
        return _canonical([_normalize_text(str(item)) for item in items])
    return _normalize_text(str(row.get("model_visible_context_text") or ""))


def _score(row: dict[str, Any]) -> int:
    if row.get("correctness") is True or row.get("primary_correctness") is True:
        return 1
    # Frozen primary failure policy: final failures remain in the denominator
    # and contribute zero.  Existing scored rows expose primary_score=0.
    if row.get("correctness") is False or row.get("primary_score") == 0 or row.get("status") == "error":
        return 0
    raise ValueError(f"Result row has no deterministic final score: {row.get('condition_id')}")


def _index_results(path: Path, provider: str) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = row.get("condition_id")
        if not cid or cid in indexed:
            raise ValueError(f"{provider} result has missing or duplicate condition_id: {cid}")
        indexed[cid] = row
    return indexed


def _rate(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    if not records:
        return {"n": 0, "denominator": 0, "rate": None}
    n = sum(bool(record[field]) for record in records)
    return {"n": n, "denominator": len(records), "rate": n / len(records)}


def _group_rates(records: list[dict[str, Any]], field: str, key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {name: _rate(groups[name], field) for name in sorted(groups)}


def analyze(*, manifest: Path = MANIFEST, result_paths: dict[str, Path] | None = None, output_dir: Path = OUTPUT) -> dict[str, Any]:
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest not in ACCEPTED_SHAS:
        raise ValueError(f"Frozen condition manifest SHA256 mismatch: {digest}")
    conditions = _read_jsonl(manifest)
    if len(conditions) != 4891 or len({row.get("condition_id") for row in conditions}) != 4891:
        raise ValueError("Frozen condition manifest must contain 4891 unique rows")
    condition_by_id = {row["condition_id"]: row for row in conditions}
    results = {provider: _index_results(path, provider) for provider, path in (result_paths or PROVIDER_RESULTS).items()}
    expected_ids = set(condition_by_id)
    for provider, rows in results.items():
        if set(rows) != expected_ids:
            raise ValueError(f"{provider} results do not exactly match frozen condition IDs")

    groups: dict[tuple[str, str | None], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in conditions:
        groups[(row["base_id"], row.get("fault_operator"))][row["condition_type"]] = row
    pairs = [group for (base_id, fault), group in groups.items() if fault is not None]
    if len(pairs) != 2080 or any(set(pair) != {"corrupted", "enforced"} for pair in pairs):
        raise ValueError("Expected exactly 2080 complete corrupted/enforced pairs")
    clean_by_base = {base: group["clean"] for (base, fault), group in groups.items() if fault is None}
    if len(clean_by_base) != 731:
        raise ValueError("Expected one clean row for each of 731 bases")

    instances: list[dict[str, Any]] = []
    for provider, provider_rows in results.items():
        for group in sorted(pairs, key=lambda item: item["corrupted"]["pair_id"]):
            corr, enf = group["corrupted"], group["enforced"]
            clean = clean_by_base[corr["base_id"]]
            a_clean, a_corr, a_enf = canonical_admission(clean), canonical_admission(corr), canonical_admission(enf)
            c_clean, c_corr, c_enf = canonical_context(clean), canonical_context(corr), canonical_context(enf)
            sr = {kind: _score(provider_rows[row["condition_id"]]) for kind, row in (("clean", clean), ("corrupted", corr), ("enforced", enf))}
            instances.append({"provider": provider, "base_id": corr["base_id"], "pair_id": corr["pair_id"], "source_dataset": corr["source_dataset"], "analysis_event_type": corr["analysis_event_type"], "fault_operator": corr["fault_operator"], "a_clean": a_clean, "a_corrupted": a_corr, "a_enforced": a_enf, "a_corr_changed": a_corr != a_clean, "a_enf_restored": a_enf == a_clean, "c_clean": c_clean, "c_corrupted": c_corr, "c_enforced": c_enf, "c_corr_changed": c_corr != c_clean, "c_enf_restored": c_enf == c_clean, "s_clean": sr["clean"], "s_corrupted": sr["corrupted"], "s_enforced": sr["enforced"], "corruption_delta": sr["corrupted"] - sr["clean"], "recovery_delta": sr["enforced"] - sr["corrupted"]})

    def by(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            out[str(record[key])].append(record)
        return out

    metric_fields = {"admission_change": "a_corr_changed", "admission_restoration": "a_enf_restored", "context_change": "c_corr_changed", "context_restoration": "c_enf_restored"}
    def mechanism_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
        return {name: _rate(records, field) for name, field in metric_fields.items()}

    provider_summary: dict[str, Any] = {}
    for provider, records in by(instances, "provider").items():
        changed = [record for record in records if record["c_corr_changed"]]
        unchanged = [record for record in records if not record["c_corr_changed"]]
        clean_correct = [record for record in records if record["s_clean"] == 1]
        induced = [record for record in clean_correct if record["s_corrupted"] == 0 and record["c_corr_changed"]]
        diagnostic = [record for record in records if record["a_corr_changed"] and record["c_corr_changed"] and record["s_clean"] == 1 and record["s_corrupted"] == 0 and record["a_enf_restored"] and record["c_enf_restored"] and record["s_enforced"] == 1]
        provider_summary[provider] = {"pair_count": len(records), "mechanism": mechanism_summary(records), "corruption_effect_by_context_change": {"changed": {"n": len(changed), "mean_delta": mean(record["corruption_delta"] for record in changed) if changed else None}, "unchanged": {"n": len(unchanged), "mean_delta": mean(record["corruption_delta"] for record in unchanged) if unchanged else None}}, "clean_correct_failure": {"clean_correct_n": len(clean_correct), "context_changed_n": len([r for r in clean_correct if r["c_corr_changed"]]), "corrupted_wrong_given_context_changed": {"n": sum(r["s_corrupted"] == 0 for r in clean_correct if r["c_corr_changed"]), "denominator": len([r for r in clean_correct if r["c_corr_changed"]]), "probability": sum(r["s_corrupted"] == 0 for r in clean_correct if r["c_corr_changed"]) / len([r for r in clean_correct if r["c_corr_changed"]]) if any(r["c_corr_changed"] for r in clean_correct) else None}, "enforced_correct_given_induced_failure": {"n": sum(r["s_enforced"] == 1 for r in induced), "denominator": len(induced), "probability": sum(r["s_enforced"] == 1 for r in induced) / len(induced) if induced else None}}, "propagation_recovery_diagnostic": {"n": len(diagnostic), "denominator": len(records), "rate": len(diagnostic) / len(records)}}

    grouped_tables = {}
    for label, key in (("event", "analysis_event_type"), ("fault", "fault_operator"), ("source", "source_dataset")):
        table = []
        for name, records in sorted(by(instances, key).items()):
            row = {key: name, "pair_count": len(records)}
            row.update(mechanism_summary(records))
            row["mean_corruption_delta"] = mean(record["corruption_delta"] for record in records)
            row["mean_recovery_delta"] = mean(record["recovery_delta"] for record in records)
            table.append(row)
        grouped_tables[label] = table
    summary = {"status": "OFFLINE CONTROL-PLANE PROPAGATION DIAGNOSTIC", "manifest_sha256": digest, "base_count": 731, "fault_pair_count": 2080, "provider_count": len(results), "provider_rows": len(instances), "providers": sorted(results), "representations": {"A": "reconstructed from persisted model_visible_memory_items (ordered admitted-item projection); no separate admission decision field is persisted", "C": "persisted model_visible_memory_items canonicalized by Unicode-preserving whitespace collapse per item; text fallback normalizes whitespace", "Y": "existing scored correctness; final provider/parse failures are scored 0"}, "overall": mechanism_summary(instances), "by_provider": provider_summary, "by_event": grouped_tables["event"], "by_fault": grouped_tables["fault"], "by_source": grouped_tables["source"], "provider_result_paths": {provider: _artifact_label(path) for provider, path in (result_paths or PROVIDER_RESULTS).items()}, "full_diagnostic": {provider: provider_summary[provider]["propagation_recovery_diagnostic"] for provider in sorted(provider_summary)}}
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(instances[0])
    with (output_dir / "mechanism_instance_level.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(instances)
    (output_dir / "mechanism_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, rows in grouped_tables.items():
        fields = list(rows[0]) if rows else []
        with (output_dir / f"mechanism_by_{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    provider_rows = []
    for provider, metrics in sorted(provider_summary.items()):
        provider_rows.append({"provider": provider, "pair_count": metrics["pair_count"], **metrics["mechanism"]})
    provider_fields = list(provider_rows[0])
    with (output_dir / "mechanism_by_provider.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=provider_fields); writer.writeheader(); writer.writerows(provider_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--provider-result", action="append", nargs=2, metavar=("PROVIDER", "PATH"), help="Provider and complete sanitized result path; repeat for all providers")
    args = parser.parse_args()
    result_paths = dict(args.provider_result) if args.provider_result else None
    summary = analyze(manifest=args.manifest, result_paths={key: Path(value) for key, value in result_paths.items()} if result_paths else None, output_dir=args.output_dir)
    print(json.dumps({"bases": summary["base_count"], "fault_pairs": summary["fault_pair_count"], "provider_rows": summary["provider_rows"], "providers": summary["providers"], "overall": summary["overall"], "api_calls_made": 0, "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
