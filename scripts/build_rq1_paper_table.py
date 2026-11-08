"""Build the deterministic, paired RQ1 paper table from frozen artifacts.

This module intentionally imports no provider implementation.  It reads the
already-scored RQ1-A JSONL files and their summaries only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MANIFEST = Path("data/governance_fault_rq1_pilot_frozen_v2.jsonl")
OUT_DIR = Path("results/rq1_paper")
PROVIDERS = {
    "openai": ("gpt-5.6-sol", Path("results/rq1/openai")),
    "anthropic": ("claude-sonnet-5", Path("results/rq1/anthropic")),
    "deepseek": ("deepseek-v4-pro", Path("results/rq1/deepseek")),
}
REQUIRED_ROWS = (
    ("authorization", "flipped"),
    ("authorization", "conflicting"),
    ("deletion", "flipped"),
    ("deletion", "stale"),
    ("temporal", "missing"),
    ("temporal", "stale"),
)
DATASET_FOR_DIMENSION = {
    "authorization": "gatemem",
    "deletion": "memora",
    "temporal": "memconflict",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required RQ1 artifact is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object in {path}:{line_number}")
        rows.append(value)
    return rows


def _load_manifest(path: Path = MANIFEST) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen v2 manifest is missing: {path}")
    raw = path.read_bytes()
    rows = _read_jsonl(path)
    if not rows or any(row.get("manifest_version") != "governance-fault-rq1-v2-corrected" for row in rows):
        raise ValueError(f"Manifest is not the corrected frozen v2 cohort: {path}")
    by_run = {row.get("run_id"): row for row in rows}
    if len(by_run) != len(rows) or None in by_run:
        raise ValueError("Frozen v2 manifest has duplicate or missing run_id values")
    return rows, hashlib.sha256(raw).hexdigest()


def _operator(row: dict[str, Any]) -> str | None:
    spec = row.get("fault_spec")
    return spec.get("operator") if isinstance(spec, dict) else None


def _metric_value(row: dict[str, Any], metric: str) -> int | None:
    value = row.get(metric)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"Scoring field {metric!r} is not boolean/null in {row.get('run_id')}")
    return int(value)


def _summary_key(dataset: str, dimension: str, fault: str, direction: str) -> str:
    return f"{dataset}|{dimension}|{fault}|{direction}"


def _assert_close(left: float | None, right: float | None, label: str) -> None:
    if left is None or right is None:
        if left != right:
            raise ValueError(f"RQ1 summary disagreement for {label}: {left!r} != {right!r}")
    elif not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"RQ1 summary disagreement for {label}: {left!r} != {right!r}")


def _pair_effects(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    index: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for row in rows:
        provider = row.get("provider")
        key = (row.get("example_id"), row.get("condition_type"), _operator(row))
        if key in index:
            raise ValueError(f"Duplicate scored RQ1 row: {key}")
        index[key] = row

    result: list[dict[str, Any]] = []
    for dimension, fault in REQUIRED_ROWS:
        dataset = DATASET_FOR_DIMENSION[dimension]
        fault_rows = [
            row for row in rows
            if row.get("source_dataset") == dataset
            and row.get("dimension") == dimension
            and row.get("condition_type") == "fault"
            and _operator(row) == fault
        ]
        if not fault_rows:
            # An unavailable comparison is represented explicitly, never invented.
            result.append({"governance_dimension": dimension, "fault_type": fault,
                           "metric": "governance_violation", "available": False})
            continue
        paired: list[tuple[int, int]] = []
        unscored = 0
        for fault_row in fault_rows:
            clean = index.get((fault_row.get("example_id"), "clean", None))
            if clean is None:
                raise ValueError(f"Missing clean sibling for {fault_row.get('example_id')}")
            clean_value = _metric_value(clean, "governance_violation")
            fault_value = _metric_value(fault_row, "governance_violation")
            if clean_value is None or fault_value is None:
                unscored += 1
            else:
                paired.append((clean_value, fault_value))
        deltas = [fault_value - clean_value for clean_value, fault_value in paired]
        direction = fault_rows[0].get("fault_direction")
        group = summary.get("paired", {}).get(_summary_key(dataset, dimension, fault, direction))
        if group is None:
            raise ValueError(f"Final RQ1 summary lacks {dataset}/{dimension}/{fault}/{direction}")
        expected_n = group.get("paired_n")
        expected_scored = group.get("paired_governance_violation_n")
        if expected_n != len(fault_rows) or expected_scored != len(paired):
            raise ValueError(
                f"RQ1 denominator disagreement for {dataset}/{fault}: "
                f"reconstructed {len(fault_rows)}/{len(paired)}, summary {expected_n}/{expected_scored}"
            )
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        _assert_close(mean_delta, group.get("mean_delta_governance_violation"),
                      f"{dataset}/{dimension}/{fault}/delta")
        clean_rate = sum(clean for clean, _ in paired) / len(paired) if paired else None
        fault_rate = sum(fault_value for _, fault_value in paired) / len(paired) if paired else None
        result.append({
            "governance_dimension": dimension,
            "fault_type": fault,
            "metric": "governance_violation",
            "fault_direction": direction,
            "available": bool(paired),
            "clean_rate": clean_rate,
            "fault_rate": fault_rate,
            "delta": mean_delta,
            "total_base_n": len(fault_rows),
            "paired_n": len(paired),
            "paired_unscored_n": unscored,
            "paired_coverage": len(paired) / len(fault_rows),
            "ci_low": None,
            "ci_high": None,
        })
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10f}".rstrip("0").rstrip(".")
    return str(value)


def _signed(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:+.4f}"


def build(manifest: Path = MANIFEST, output_dir: Path = OUT_DIR) -> dict[str, Any]:
    manifest_rows, manifest_sha = _load_manifest(manifest)
    manifest_runs = {row["run_id"] for row in manifest_rows}
    provider_results: dict[str, list[dict[str, Any]]] = {}
    provenance_sources: dict[str, Any] = {}
    for provider, (model, directory) in PROVIDERS.items():
        scored_path = directory / "rq1a_scored.jsonl"
        summary_path = directory / "rq1a_summary.json"
        rows = _read_jsonl(scored_path)
        if len(rows) != 300:
            raise ValueError(f"Expected 300 corrected v2 RQ1-A rows for {provider}, got {len(rows)}")
        if any(row.get("provider") != provider or row.get("model") != model for row in rows):
            raise ValueError(f"Provider/model mismatch in {scored_path}")
        if any(row.get("run_id") not in manifest_runs for row in rows):
            raise ValueError(f"{scored_path} contains rows outside the frozen v2 manifest")
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
        if not isinstance(summary, dict):
            raise FileNotFoundError(f"Required RQ1 summary is missing: {summary_path}")
        provider_results[provider] = _pair_effects(rows, summary)
        provenance_sources[provider] = {
            "scored_file": str(scored_path), "summary_file": str(summary_path),
            "scored_rows_read": len(rows), "model": model,
        }

    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for provider, entries in provider_results.items():
        for entry in entries:
            key = (entry["governance_dimension"], entry["fault_type"], entry["metric"])
            by_key.setdefault(key, {})[provider] = entry

    fields = ["governance_dimension", "fault_type", "metric"]
    for provider in PROVIDERS:
        fields += [f"{provider}_{suffix}" for suffix in
                   ("clean_rate", "fault_rate", "delta", "paired_n", "total_base_n",
                    "paired_unscored_n", "paired_coverage", "ci_low", "ci_high")]
    table: list[dict[str, Any]] = []
    for key in [(d, f, "governance_violation") for d, f in REQUIRED_ROWS]:
        record = {"governance_dimension": key[0], "fault_type": key[1], "metric": key[2]}
        for provider in PROVIDERS:
            entry = by_key.get(key, {}).get(provider, {"available": False})
            for suffix in ("clean_rate", "fault_rate", "delta", "paired_n", "total_base_n",
                           "paired_unscored_n", "paired_coverage", "ci_low", "ci_high"):
                record[f"{provider}_{suffix}"] = entry.get(suffix)
        table.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rq1_permissive_effects.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: _fmt(row.get(field)) for field in fields} for row in table)

    latex_lines = [
        r"\begin{table*}[t]", r"\centering", r"\caption{Paired clean-versus-fault governance-violation effects on the frozen RQ1 cohort.}",
        r"\begin{tabular}{llrrr}", r"\toprule", r"Dimension & Fault & OpenAI $\Delta$GVR & Anthropic $\Delta$GVR & DeepSeek $\Delta$GVR \\", r"\midrule",
    ]
    labels = {"authorization": "Authorization", "deletion": "Deletion", "temporal": "Temporal"}
    for row in table:
        cells = []
        for provider in PROVIDERS:
            value = _signed(row.get(f"{provider}_delta"))
            n = row.get(f"{provider}_paired_n")
            cells.append(f"{value} (n={n})" if n is not None else "NA")
        latex_lines.append(f"{labels[row['governance_dimension']]} & {row['fault_type']} & " + " & ".join(cells) + r" \\")
    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\begin{minipage}{0.94\linewidth}", "\\footnotesize Positive values indicate more governance violations. Effects are paired within base example, not independent samples; NA denotes unavailable scoring.\\", r"\end{minipage}", r"\end{table*}", ""]
    (output_dir / "rq1_permissive_effects.tex").write_text("\n".join(latex_lines), encoding="utf-8")

    markdown = ["# RQ1 paper-table replacement report", "", "## Exact paired effects", "", "| Dimension | Fault | " + " | ".join(f"{p} ΔGVR" for p in PROVIDERS) + " |", "|---|---|---:|---:|---:|"]
    for row in table:
        markdown.append("| " + row["governance_dimension"] + " | " + row["fault_type"] + " | " + " | ".join(_signed(row.get(f"{p}_delta")) for p in PROVIDERS) + " |")
    markdown += ["", "## Suggested RQ1 Results replacement", "", "Across the corrected frozen v2 cohort, permissive governance-state faults were evaluated using paired clean-versus-fault comparisons within base example. The table reports the exact mean change in governance-violation rate for each provider; the entries are not independent-sample contrasts. Effects and denominators are reported separately by fault type, and unavailable comparisons remain NA.", "", "## Audit notes", "", "- Values are reconstructed from the final provider-specific `rq1a_scored.jsonl` artifacts and checked against each provider's final paired summary.", "- `paired_n` is the number of candidate fault bases with a clean sibling; `paired_unscored_n` is retained explicitly, and coverage is not silently replaced by a zero.", "- No confidence intervals were present in the source RQ1 summaries, so CI columns are unavailable rather than fabricated."]
    (output_dir / "rq1_paper_replacement.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    provenance = {
        "manifest_path": str(manifest), "manifest_sha256": manifest_sha,
        "provider_model_mapping": {p: {"provider": p, "model": m} for p, (m, _) in PROVIDERS.items()},
        "source_artifacts": provenance_sources,
        "scoring_field": "governance_violation",
        "paired_effect_definition": "mean(fault - clean) over base_example_id with both boolean scores",
        "deterministic_config": {"source_scope": "frozen v2 RQ1-A scored artifacts", "confidence_intervals": "not available from source summaries"},
    }
    (output_dir / "rq1_paper_table_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"rows": table, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paired frozen-v2 RQ1 paper table without inference")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    build(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
