#!/usr/bin/env python3
"""Aggregate paper-ready deterministic RQ1 summaries across providers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_PROVIDERS = ("deepseek", "openai", "anthropic")
RQ1A_FILENAME = "rq1a_summary.json"
RQ1B_FILENAME = "rq1b_restrictive_summary.json"
CSV_COLUMNS = [
    "track", "row_type", "provider", "source_dataset", "governance_dimension",
    "fault_type", "fault_direction", "paired_n", "paired_governance_violation_n",
    "mean_delta_governance_violation", "paired_task_utility_n", "mean_delta_task_utility",
    "downstream_false_denial_rate", "scored_downstream_false_denial_n",
    "fault_only_control_plane_false_rejection_rate",
    "clean_governance_violation_rate", "clean_scored_governance_violation_n",
    "clean_task_utility_rate", "clean_target_utility_rate", "clean_scored_task_utility_n",
    "clean_scored_target_utility_n", "scorable", "unscorable_reason",
]
FINAL_MAIN_ROOT = Path("results")
FINAL_MAIN_V1 = FINAL_MAIN_ROOT / "final_main_cross_provider_summary_v1"
FINAL_MAIN_V2 = FINAL_MAIN_ROOT / "final_main_cross_provider_summary_v2"
FINAL_MAIN_V3 = FINAL_MAIN_ROOT / "final_main_cross_provider_summary_v3"
FINAL_MAIN_KIMI_ANALYSIS = FINAL_MAIN_ROOT / "final_main_gemini_3_flash_vs_kimi_k2_6_analysis" / "analysis_summary.json"
FINAL_MAIN_ANALYSES = {
    "openai": FINAL_MAIN_ROOT / "final_main_deepseek_v4_pro_vs_openai_gpt56_sol_analysis" / "analysis_summary.json",
    "anthropic": FINAL_MAIN_ROOT / "final_main_deepseek_v4_pro_vs_anthropic_sonnet5_analysis" / "analysis_summary.json",
    "gemini": FINAL_MAIN_ROOT / "final_main_deepseek_v4_pro_vs_gemini_3_flash_analysis" / "analysis_summary.json",
}
FINAL_MAIN_KIMI_RAW = Path("results/final_main_kimi_k2_6_nothinking_final/full_results_merged.jsonl")
FINAL_MAIN_KIMI_SCORED = Path("results/final_main_kimi_k2_6_nothinking_final/scored/full_results_scored.jsonl")
FINAL_MAIN_GROK_ANALYSIS = FINAL_MAIN_ROOT / "final_main_kimi_vs_grok_43_analysis" / "analysis_summary.json"
FINAL_MAIN_GROK_RAW = Path("results/final_main_grok_43_full/full_results.jsonl")
FINAL_MAIN_GROK_SCORED = Path("results/final_main_grok_43_final/scored/full_results_scored.jsonl")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Referenced RQ1 summary file is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"RQ1 summary must be a JSON object: {path}")
    return value


def _parse_group_key(key: str) -> tuple[str, str, str, str]:
    parts = key.split("|", 3)
    if len(parts) != 4:
        raise ValueError(f"Malformed RQ1 summary group key: {key!r}")
    return tuple(parts)  # type: ignore[return-value]


def _scorable(record: dict[str, Any], denominator_fields: tuple[str, ...]) -> tuple[bool, str | None]:
    scored = [record.get(field) for field in denominator_fields]
    if any(isinstance(value, int) and value > 0 for value in scored):
        return True, None
    return False, "No deterministic metric values were scored for this condition"


def _clean_baseline(provider: str, groups: dict[str, Any], track: str) -> list[dict[str, Any]]:
    rows = []
    for key, metrics in sorted(groups.items()):
        dataset, dimension, fault_type, fault_direction = _parse_group_key(key)
        if fault_type != "clean" or fault_direction != "clean":
            continue
        if not isinstance(metrics, dict):
            raise ValueError(f"Malformed group metrics for {provider}: {key}")
        scorable, reason = _scorable(
            metrics,
            ("scored_governance_violation_n", "scored_task_utility_n", "scored_target_utility_n"),
        )
        rows.append({
            "track": track,
            "row_type": "clean_baseline",
            "provider": provider,
            "source_dataset": dataset,
            "governance_dimension": dimension,
            "fault_type": None,
            "fault_direction": None,
            "clean_governance_violation_rate": metrics.get("governance_violation_rate"),
            "clean_scored_governance_violation_n": metrics.get("scored_governance_violation_n"),
            "clean_task_utility_rate": metrics.get("task_utility_rate"),
            "clean_target_utility_rate": metrics.get("target_utility_rate"),
            "clean_scored_task_utility_n": metrics.get("scored_task_utility_n"),
            "clean_scored_target_utility_n": metrics.get("scored_target_utility_n"),
            "scorable": scorable,
            "unscorable_reason": reason,
        })
    return rows


def _rq1a_rows(provider: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    paired = summary.get("paired")
    groups = summary.get("groups")
    if not isinstance(paired, dict) or not isinstance(groups, dict):
        raise ValueError(f"RQ1-A summary for {provider} lacks paired/groups sections")
    rows: list[dict[str, Any]] = []
    for key, metrics in sorted(paired.items()):
        dataset, dimension, fault_type, fault_direction = _parse_group_key(key)
        if fault_type == "clean" or fault_direction == "clean":
            continue
        if not isinstance(metrics, dict):
            raise ValueError(f"Malformed paired metrics for {provider}: {key}")
        scorable, reason = _scorable(
            metrics, ("paired_governance_violation_n", "paired_task_utility_n")
        )
        rows.append({
            "track": "rq1a",
            "row_type": "paired_fault",
            "provider": provider,
            "source_dataset": dataset,
            "governance_dimension": dimension,
            "fault_type": fault_type,
            "fault_direction": fault_direction,
            "paired_n": metrics.get("paired_n"),
            "paired_governance_violation_n": metrics.get("paired_governance_violation_n"),
            "mean_delta_governance_violation": metrics.get("mean_delta_governance_violation"),
            "paired_task_utility_n": metrics.get("paired_task_utility_n"),
            "mean_delta_task_utility": metrics.get("mean_delta_task_utility"),
            "scorable": scorable,
            "unscorable_reason": reason,
        })
    rows.extend(_clean_baseline(provider, groups, "rq1a"))
    return rows


def _rq1b_rows(provider: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    paired = summary.get("paired")
    groups = summary.get("groups")
    if not isinstance(paired, dict) or not isinstance(groups, dict):
        raise ValueError(f"RQ1-B summary for {provider} lacks paired/groups sections")
    rows: list[dict[str, Any]] = []
    for key, metrics in sorted(paired.items()):
        dataset, dimension, fault_type, fault_direction = _parse_group_key(key)
        if fault_type == "clean" or fault_direction == "clean":
            continue
        group = groups.get(key)
        if not isinstance(metrics, dict) or not isinstance(group, dict):
            raise ValueError(f"Missing or malformed RQ1-B fault group for {provider}: {key}")
        scorable_values = (
            metrics.get("paired_task_utility_n"),
            group.get("scored_downstream_false_denial_n"),
            group.get("scored_control_plane_false_rejection_n"),
        )
        scorable = any(isinstance(value, int) and value > 0 for value in scorable_values)
        reason = None if scorable else "No deterministic metric values were scored for this condition"
        rows.append({
            "track": "rq1b",
            "row_type": "paired_fault",
            "provider": provider,
            "source_dataset": dataset,
            "governance_dimension": dimension,
            "fault_type": fault_type,
            "fault_direction": fault_direction,
            "paired_n": metrics.get("paired_n"),
            "paired_task_utility_n": metrics.get("paired_task_utility_n"),
            "mean_delta_task_utility": metrics.get("mean_delta_task_utility"),
            "downstream_false_denial_rate": group.get("downstream_false_denial_rate"),
            "scored_downstream_false_denial_n": group.get("scored_downstream_false_denial_n"),
            "fault_only_control_plane_false_rejection_rate": group.get("control_plane_false_rejection_rate"),
            "scorable": scorable,
            "unscorable_reason": reason,
        })
    rows.extend(_clean_baseline(provider, groups, "rq1b"))
    return rows


def aggregate_summaries(provider_paths: dict[str, tuple[str | Path, str | Path]]) -> dict[str, Any]:
    missing = [provider for provider in EXPECTED_PROVIDERS if provider not in provider_paths]
    if missing:
        raise ValueError(f"Missing expected RQ1 providers: {', '.join(missing)}")
    extra = sorted(set(provider_paths) - set(EXPECTED_PROVIDERS))
    if extra:
        raise ValueError(f"Unexpected RQ1 providers: {', '.join(extra)}")

    rows: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, str]] = {}
    for provider in EXPECTED_PROVIDERS:
        rq1a_path, rq1b_path = provider_paths[provider]
        rq1a_path, rq1b_path = Path(rq1a_path), Path(rq1b_path)
        rq1a, rq1b = _read_json(rq1a_path), _read_json(rq1b_path)
        source_files[provider] = {"rq1a": str(rq1a_path), "rq1b": str(rq1b_path)}
        rows.extend(_rq1a_rows(provider, rq1a))
        rows.extend(_rq1b_rows(provider, rq1b))
    return {
        "summary_version": "rq1-cross-provider-v1",
        "providers": list(EXPECTED_PROVIDERS),
        "source_files": source_files,
        "rows": rows,
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in CSV_COLUMNS} for row in rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _final_main_model_row(label: str, provider: str, analysis: dict[str, Any], *, run_key: str = "run_b") -> dict[str, Any]:
    run = analysis[run_key]
    per_event, per_fault, per_source = run["per_event"], run["per_fault"], run["per_source"]
    row: dict[str, Any] = {
        "model": label,
        "provider": provider,
        "clean_accuracy": run["condition_accuracy"]["clean"]["accuracy"],
        "corrupted_accuracy": run["condition_accuracy"]["corrupted"]["accuracy"],
        "enforced_accuracy": run["condition_accuracy"]["enforced"]["accuracy"],
        "event_macro_corruption": run["event_macro"]["corruption"]["mean"],
        "event_macro_corruption_ci_low": run["event_macro"]["corruption"]["ci_low"],
        "event_macro_corruption_ci_high": run["event_macro"]["corruption"]["ci_high"],
        "event_macro_recovery": run["event_macro"]["recovery"]["mean"],
        "event_macro_recovery_ci_low": run["event_macro"]["recovery"]["ci_low"],
        "event_macro_recovery_ci_high": run["event_macro"]["recovery"]["ci_high"],
        "micro_corruption": run["micro"]["corruption"]["mean"],
        "micro_recovery": run["micro"]["recovery"]["mean"],
    }
    for event in ("deletion", "temporal", "update"):
        row[f"{event}_corruption"] = per_event[event]["corruption"]["mean_delta"]
        row[f"{event}_recovery"] = per_event[event]["recovery"]["mean_delta"]
    for fault in ("conflicting", "flipped", "missing", "stale"):
        row[f"{fault}_corruption"] = per_fault[fault]["corruption"]["mean_delta"]
        row[f"{fault}_recovery"] = per_fault[fault]["recovery"]["mean_delta"]
    for source, key in (("longmemeval", "longmemeval"), ("memora", "memora"), ("oaks_novel", "oaks")):
        row[f"{key}_corruption"] = per_source[source]["corruption"]["mean_delta"]
        row[f"{key}_recovery"] = per_source[source]["recovery"]["mean_delta"]
    row["balanced_macro_corruption"] = run["balanced_robustness"]["event_macro"]["corruption"]["mean"]
    row["balanced_macro_recovery"] = run["balanced_robustness"]["event_macro"]["recovery"]["mean"]
    row["oaks_cluster_corruption"] = run["oaks_cluster_robustness"]["corruption"]["mean"]
    row["oaks_cluster_recovery"] = run["oaks_cluster_robustness"]["recovery"]["mean"]
    return row


def _final_main_summary() -> dict[str, Any]:
    previous = json.loads((FINAL_MAIN_V1 / "cross_provider_summary.json").read_text(encoding="utf-8"))
    analyses = {name: _read_json(path) for name, path in FINAL_MAIN_ANALYSES.items()}
    kimi_analysis = _read_json(FINAL_MAIN_KIMI_ANALYSIS)
    # Keep the four existing model rows exactly as frozen in v1.
    models = list(previous["models"])
    kimi_row = _final_main_model_row("Kimi K2.6 no-thinking", "Kimi", kimi_analysis)
    models.append(kimi_row)

    corruption = [row["event_macro_corruption"] for row in models]
    recovery = [row["event_macro_recovery"] for row in models]
    provenance = {
        "openai": {"analysis_summary": str(FINAL_MAIN_ANALYSES["openai"]), "analysis_summary_sha256": _sha256(FINAL_MAIN_ANALYSES["openai"])},
        "anthropic": {"analysis_summary": str(FINAL_MAIN_ANALYSES["anthropic"]), "analysis_summary_sha256": _sha256(FINAL_MAIN_ANALYSES["anthropic"])},
        "gemini": {"analysis_summary": str(FINAL_MAIN_ANALYSES["gemini"]), "analysis_summary_sha256": _sha256(FINAL_MAIN_ANALYSES["gemini"])},
        "kimi": {
            "analysis_summary": str(FINAL_MAIN_KIMI_ANALYSIS),
            "analysis_summary_sha256": _sha256(FINAL_MAIN_KIMI_ANALYSIS),
            "raw_results": str(FINAL_MAIN_KIMI_RAW),
            "raw_results_sha256": _sha256(FINAL_MAIN_KIMI_RAW),
            "scored_results": str(FINAL_MAIN_KIMI_SCORED),
            "scored_results_sha256": _sha256(FINAL_MAIN_KIMI_SCORED),
            "model": "kimi-k2.6",
            "thinking": "disabled",
            "temperature": 0.6,
            "top_p": 0.95,
        },
    }
    metadata = {
        "status": "FROZEN / FIVE-PROVIDER CROSS-PROVIDER SUMMARY",
        "terminology": "cross-provider robustness and cross-provider agreement; providers are not replication runs",
        "manifest_sha256": previous["metadata"]["manifest_sha256"],
        "base_manifest_sha256": previous["metadata"]["base_manifest_sha256"],
        "analysis_spec": previous["metadata"]["analysis_spec"],
        "failure_spec": previous["metadata"]["failure_spec"],
        "bootstrap_replicates": previous["metadata"]["bootstrap_replicates"],
        "models": [row["model"] for row in models],
        "artifact_provenance": provenance,
        "descriptive_aggregates": {
            "event_macro_corruption": {"mean": sum(corruption) / len(corruption), "min": min(corruption), "max": max(corruption)},
            "event_macro_recovery": {"mean": sum(recovery) / len(recovery), "min": min(recovery), "max": max(recovery)},
        },
    }
    return {"metadata": metadata, "models": models}


def _write_final_main_outputs(summary: dict[str, Any], output_dir: Path = FINAL_MAIN_V2) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cross_provider_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Preserve the v1 table schema and append the fifth formal provider.
    with (FINAL_MAIN_V1 / "cross_provider_summary.csv").open(encoding="utf-8", newline="") as handle:
        fieldnames = next(csv.reader(handle))
    with (output_dir / "cross_provider_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in summary["models"])
    def pct(value: float) -> str:
        return f"{100 * value:.2f}"
    def ci(low: float, high: float) -> str:
        return f"[{100 * low:.2f}, {100 * high:.2f}]"
    lines = ["# Six-provider final-main cross-provider summary", "", "| Model | Clean | Corrupted | Enforced | Corruption macro | 95% CI | Recovery macro | 95% CI |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["models"]:
        lines.append(f"| {row['model']} | {pct(row['clean_accuracy'])}% | {pct(row['corrupted_accuracy'])}% | {pct(row['enforced_accuracy'])}% | {pct(row['event_macro_corruption'])} pp | {ci(row['event_macro_corruption_ci_low'], row['event_macro_corruption_ci_high'])} | +{pct(row['event_macro_recovery'])} pp | {ci(row['event_macro_recovery_ci_low'], row['event_macro_recovery_ci_high'])} |")
    lines += ["", "## Descriptive six-provider aggregates", "", f"- Event-macro corruption mean/range: {pct(summary['metadata']['descriptive_aggregates']['event_macro_corruption']['mean'])} pp; {pct(summary['metadata']['descriptive_aggregates']['event_macro_corruption']['min'])} to {pct(summary['metadata']['descriptive_aggregates']['event_macro_corruption']['max'])} pp", f"- Event-macro recovery mean/range: +{pct(summary['metadata']['descriptive_aggregates']['event_macro_recovery']['mean'])} pp; +{pct(summary['metadata']['descriptive_aggregates']['event_macro_recovery']['min'])} to +{pct(summary['metadata']['descriptive_aggregates']['event_macro_recovery']['max'])} pp", "", "## Event breakdown", "", "| Model | Deletion corruption/recovery | Temporal corruption/recovery | Update corruption/recovery |", "|---|---:|---:|---:|"]
    for row in summary["models"]:
        lines.append(f"| {row['model']} | {pct(row['deletion_corruption'])} / +{pct(row['deletion_recovery'])} | {pct(row['temporal_corruption'])} / +{pct(row['temporal_recovery'])} | {pct(row['update_corruption'])} / +{pct(row['update_recovery'])} |")
    lines += ["", "## Fault breakdown", "", "| Model | Conflicting | Flipped | Missing | Stale |", "|---|---:|---:|---:|---:|"]
    for row in summary["models"]:
        lines.append(f"| {row['model']} | {pct(row['conflicting_corruption'])} | {pct(row['flipped_corruption'])} | {pct(row['missing_corruption'])} | {pct(row['stale_corruption'])} |")
    lines += ["", "## Source breakdown", "", "| Model | LongMemEval | Memora | OAKS |", "|---|---:|---:|---:|"]
    for row in summary["models"]:
        lines.append(f"| {row['model']} | {pct(row['longmemeval_corruption'])} / +{pct(row['longmemeval_recovery'])} | {pct(row['memora_corruption'])} / +{pct(row['memora_recovery'])} | {pct(row['oaks_corruption'])} / +{pct(row['oaks_recovery'])} |")
    (output_dir / "cross_provider_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"providers": summary["metadata"]["models"], "output_dir": str(output_dir)}, sort_keys=True))


def _final_main_summary_v3() -> dict[str, Any]:
    """Extend the frozen v2 descriptive table with the formal Grok run."""
    summary = _final_main_summary()
    grok_analysis = _read_json(FINAL_MAIN_GROK_ANALYSIS)
    summary["models"].append(_final_main_model_row("Grok 4.3 no-reasoning", "xAI", grok_analysis))
    previous_provenance = dict(summary["metadata"].get("artifact_provenance", {}))
    previous_provenance["grok"] = {
        "analysis_summary": str(FINAL_MAIN_GROK_ANALYSIS),
        "analysis_summary_sha256": _sha256(FINAL_MAIN_GROK_ANALYSIS),
        "raw_results": str(FINAL_MAIN_GROK_RAW),
        "raw_results_sha256": _sha256(FINAL_MAIN_GROK_RAW),
        "scored_results": str(FINAL_MAIN_GROK_SCORED),
        "scored_results_sha256": _sha256(FINAL_MAIN_GROK_SCORED),
        "provider": "xai",
        "model": "grok-4.3",
        "reasoning_effort": "none",
        "temperature": None,
        "top_p": None,
    }
    corruption = [row["event_macro_corruption"] for row in summary["models"]]
    recovery = [row["event_macro_recovery"] for row in summary["models"]]
    summary["metadata"] = {
        **summary["metadata"],
        "status": "FROZEN / SIX-PROVIDER CROSS-PROVIDER SUMMARY",
        "models": [row["model"] for row in summary["models"]],
        "artifact_provenance": previous_provenance,
        "descriptive_aggregates": {
            "event_macro_corruption": {"mean": sum(corruption) / len(corruption), "min": min(corruption), "max": max(corruption)},
            "event_macro_recovery": {"mean": sum(recovery) / len(recovery), "min": min(recovery), "max": max(recovery)},
        },
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    if argv is None and len(sys.argv) == 1:
        _write_final_main_outputs(_final_main_summary_v3(), FINAL_MAIN_V3)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="results/rq1")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args(argv)
    root = Path(args.input_root)
    paths = {
        provider: (root / provider / RQ1A_FILENAME, root / provider / RQ1B_FILENAME)
        for provider in EXPECTED_PROVIDERS
    }
    summary = aggregate_summaries(paths)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(summary["rows"], args.output_csv)
    print(json.dumps({"providers": summary["providers"], "rows": len(summary["rows"]), "output_json": args.output_json, "output_csv": args.output_csv}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
