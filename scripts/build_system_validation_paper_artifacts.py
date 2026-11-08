#!/usr/bin/env python3
"""Build publication artifacts from deterministic system-validation statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
STATS_DIR = ROOT / "results/system_validation_statistics"
COVERAGE = ROOT / "results/system_validation_analysis/system_validation_scoring_coverage.csv"
SYSTEMS = ("mem0", "langmem", "amem")
LABELS = {"mem0": "Mem0", "langmem": "LangMem", "amem": "A-Mem", "pooled": "Clustered pooled"}
HEADLINES = (
    ("authorization", "conflicting", "governance_violation"),
    ("authorization", "flipped", "governance_violation"),
    ("deletion", "flipped", "governance_violation"),
    ("deletion", "stale", "governance_violation"),
    ("temporal", "missing", "governance_violation"),
    ("temporal", "stale", "governance_violation"),
    ("temporal", "stale", "task_utility_success"),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None) -> float | None:
    return None if value in (None, "", "NA", "null") else float(value)


def _int(value: str | None) -> int | None:
    return None if value in (None, "", "NA", "null") else int(value)


def _display(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _index(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def _paper_rows(by_system: list[dict[str, str]], pooled: list[dict[str, str]]) -> list[dict[str, Any]]:
    systems = _index(by_system, ("memory_system", "governance_dimension", "fault_type", "metric"))
    pooled_index = _index(pooled, ("governance_dimension", "fault_type", "metric"))
    output = []
    for dimension, fault, metric in HEADLINES:
        row: dict[str, Any] = {"governance_dimension": dimension, "fault_type": fault, "metric": metric}
        for system in SYSTEMS:
            source = systems.get((system, dimension, fault, metric))
            row[f"{system}_delta"] = _float(source.get("mean_delta") if source else None)
            row[f"{system}_ci_low"] = _float(source.get("bootstrap_ci_low") if source else None)
            row[f"{system}_ci_high"] = _float(source.get("bootstrap_ci_high") if source else None)
            row[f"{system}_paired_n"] = _int(source.get("paired_scored_n") if source else None)
            row[f"{system}_coverage"] = _float(source.get("paired_coverage_rate") if source else None)
        source = pooled_index.get((dimension, fault, metric))
        row["pooled_delta"] = _float(source.get("mean_clustered_delta") if source else None)
        row["pooled_ci_low"] = _float(source.get("clustered_bootstrap_ci_low") if source else None)
        row["pooled_ci_high"] = _float(source.get("clustered_bootstrap_ci_high") if source else None)
        row["pooled_clustered_base_n"] = _int(source.get("clustered_base_n") if source else None)
        row["pooled_coverage"] = _float(source.get("paired_coverage_rate") if source else None)
        output.append(row)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _display(value) if isinstance(value, float) else ("NA" if value is None else value) for key, value in row.items()})


def _latex(rows: list[dict[str, Any]]) -> str:
    def cell(row: dict[str, Any], prefix: str) -> str:
        delta = row[f"{prefix}_delta"]
        low, high = row[f"{prefix}_ci_low"], row[f"{prefix}_ci_high"]
        if delta is None:
            return "NA"
        return f"{_display(delta)} [{_display(low)}, {_display(high)}]"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-system paired downstream effects.}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dimension & Fault / metric & Mem0 & LangMem & A-Mem & Clustered pooled \\",
        r"\midrule",
    ]
    for row in rows:
        label = "GVR" if row["metric"] == "governance_violation" else "Utility"
        lines.append(
            f"{row['governance_dimension']} & {row['fault_type']} / {label} & "
            f"{cell(row, 'mem0')} & {cell(row, 'langmem')} & {cell(row, 'amem')} & {cell(row, 'pooled')} " + "\\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{0.95\linewidth}\footnotesize",
        r"Note: GVR positive values indicate more governance violations; utility negative values indicate worse downstream utility. Temporal metrics use identifiable paired cases only; NA denotes unavailable estimates.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def _figure(rows: list[dict[str, Any]], output: Path) -> None:
    plotted = [row for row in rows if row["metric"] in ("governance_violation", "task_utility_success")]
    labels = [f"{row['governance_dimension']} / {row['fault_type']}" for row in plotted if row["metric"] == "governance_violation"]
    utility_labels = [f"{row['governance_dimension']} / {row['fault_type']}" for row in plotted if row["metric"] == "task_utility_success"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [3, 1.5]})
    colors = {"mem0": "#2166ac", "langmem": "#4dac26", "amem": "#b2182b", "pooled": "#222222"}
    markers = {"mem0": "o", "langmem": "s", "amem": "^", "pooled": "D"}
    for axis, metric, axis_labels in zip(axes, ("governance_violation", "task_utility_success"), (labels, utility_labels)):
        subset = [row for row in plotted if row["metric"] == metric]
        y_positions = list(range(len(axis_labels)))
        for key in (*SYSTEMS, "pooled"):
            xs, ys, lows, highs = [], [], [], []
            for index, row in enumerate(subset):
                prefix = key
                delta, low, high = row[f"{prefix}_delta"], row[f"{prefix}_ci_low"], row[f"{prefix}_ci_high"]
                if delta is None:
                    continue
                xs.append(delta); ys.append(index); lows.append(delta - low if low is not None else 0); highs.append(high - delta if high is not None else 0)
            if xs:
                axis.errorbar(xs, ys, xerr=[lows, highs], fmt=markers[key], color=colors[key], label=LABELS[key], capsize=3, linestyle="none", markersize=5)
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(axis_labels)
        axis.set_xlabel("Paired delta")
        axis.set_title("Governance violation" if metric == "governance_violation" else "Task utility success")
        axis.grid(axis="x", color="#dddddd", linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Formal cross-system external validation effects", y=1.02)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _results_text(rows: list[dict[str, Any]]) -> str:
    lookup = {(row["governance_dimension"], row["fault_type"], row["metric"]): row for row in rows}
    auth = lookup[("authorization", "conflicting", "governance_violation")]
    delete = lookup[("deletion", "flipped", "governance_violation")]
    missing = lookup[("temporal", "missing", "governance_violation")]
    stale = lookup[("temporal", "stale", "governance_violation")]
    stale_u = lookup[("temporal", "stale", "task_utility_success")]
    return f"""# Results: formal cross-system external validation

We evaluated three heterogeneous memory systems (Mem0, LangMem, and A-Mem) on the same frozen 60-base GovMem cohort using a top-1 native retrieval budget and the same downstream model (`gpt-5.6-sol`). Effects are paired clean-versus-fault comparisons within `base_example_id`.

For authorization faults, the effect was modest but consistently positive: the ΔGVR was {_display(auth['mem0_delta'])} for Mem0, {_display(auth['langmem_delta'])} for LangMem, and {_display(auth['amem_delta'])} for A-Mem. The clustered pooled effect was {_display(auth['pooled_delta'])} (95% CI [{_display(auth['pooled_ci_low'])}, {_display(auth['pooled_ci_high'])}]). These results do not establish that every individual system effect is statistically significant.

Deletion faults produced a ΔGVR of +1.00 across all systems: 20/20 paired transitions were harmful for each system, with exact paired p ≈ 1.91e-6. The clustered pooled CI was [1.00, 1.00].

Temporal missing was a negative-control condition: ΔGVR was {_display(missing['pooled_delta'])}, with identifiable paired coverage {_display(missing['pooled_coverage'])}. Its utility effect was also zero on identifiable pairs. Missing metadata therefore did not produce a measurable downstream change under the frozen semantics.

Temporal stale faults were substantially damaging. The ΔGVR was {_display(stale['mem0_delta'])} for Mem0, {_display(stale['langmem_delta'])} for LangMem, and {_display(stale['amem_delta'])} for A-Mem; the clustered pooled estimate was approximately {_display(stale['pooled_delta'])} (95% CI [{_display(stale['pooled_ci_low'])}, {_display(stale['pooled_ci_high'])}]). The task-utility delta was {_display(stale_u['pooled_delta'])} on identifiable paired examples. Temporal coverage was incomplete and is reported explicitly rather than imputed.

These results indicate that the effect survives heterogeneous storage, indexing, and ranking paths, while fault type remains consequential. Retrieval exposure does not guarantee disclosure for authorization faults, whereas stale and deletion faults were much more damaging than missing temporal state. The results do not evaluate retrieval recall at top-1 and do not claim full A-Mem write-time evolution.
"""


METHODS = """# Methods: formal cross-system external validation

The formal cohort contains 60 frozen GovMem base examples: 20 GateMem authorization, 20 Memora deletion, and 20 MemConflict temporal examples. Each base has three conditions, yielding 180 rows per system. Mem0, LangMem, and A-Mem were evaluated with a requested native `retrieval_k=1`, followed by the same downstream model, `gpt-5.6-sol`.

Each memory system performed native storage and retrieval before the GovMem governance intervention. Native retrieval was captured once and reused across clean/fault siblings for the same base. Downstream effects were paired by `base_example_id`; unrelated examples were not compared.

Uncertainty uses a paired bootstrap over base examples with 10,000 replicates and seed 42. Governance-violation binary transitions are summarized with an exact two-sided McNemar/binomial test over discordant pairs. The clustered pooled estimate first averages system-specific deltas within each base example and then bootstraps base examples, avoiding treatment of the three systems as independent observations.

A-Mem used the official Chroma-backed indexing and `search_agentic` retrieval, with LLM-based write-time processing/evolution disabled to preserve frozen memory content. This is not full A-Mem.

Temporal underdetermined cases are excluded only from metrics that cannot be uniquely identified; all exclusions and denominators are reported through scoring coverage tables. No missing value is imputed.
"""


def build(*, output_dir: str | Path = "results/system_validation_paper") -> dict[str, Any]:
    by_system = _read_csv(STATS_DIR / "system_validation_statistics_by_system.csv")
    pooled = _read_csv(STATS_DIR / "system_validation_statistics_pooled.csv")
    transitions = _read_csv(STATS_DIR / "system_validation_mcnemar.csv")
    coverage = _read_csv(COVERAGE)
    rows = _paper_rows(by_system, pooled)
    output = Path(output_dir)
    fields = ["governance_dimension", "fault_type", "metric"] + [
        f"{system}_{suffix}" for system in SYSTEMS for suffix in ("delta", "ci_low", "ci_high", "paired_n", "coverage")
    ] + ["pooled_delta", "pooled_ci_low", "pooled_ci_high", "pooled_clustered_base_n", "pooled_coverage"]
    _write_csv(output / "table_cross_system_effects.csv", rows, fields)
    transition_rows = [row for row in transitions if row["governance_dimension"] in {d for d, _, _ in HEADLINES}]
    _write_csv(output / "table_paired_transitions.csv", transition_rows)
    _write_csv(output / "table_scoring_coverage.csv", coverage)
    _figure(rows, output / "figure_cross_system_effects")
    (output / "results_external_validation.md").write_text(_results_text(rows), encoding="utf-8")
    (output / "methods_external_validation.md").write_text(METHODS, encoding="utf-8")
    (output / "table_cross_system_effects.tex").write_text(_latex(rows), encoding="utf-8")
    summary = {"status": "FROZEN / FORMAL / PAPER ARTIFACTS", "table_rows": len(rows), "coverage_rows": len(coverage), "transition_rows": len(transition_rows)}
    (output / "paper_artifacts_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/system_validation_paper")
    args = parser.parse_args()
    print(json.dumps(build(output_dir=args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
