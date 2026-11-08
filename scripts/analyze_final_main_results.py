#!/usr/bin/env python3
"""Deterministic analysis of the frozen final-main DeepSeek runs.

This module is analysis-only: it reads frozen scored JSONL and policy/manifest
files, never initializes a provider, and never changes an input artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONDITION_SHA = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"
ACCEPTED_CONDITION_SHAS = {EXPECTED_CONDITION_SHA, "4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb"}
EXPECTED_BASE_SHA = "a2601a7ecb46034faf5dcc7ad00b5a851e19ae01561876fa681fd18af168ca79"
EVENTS = ("deletion", "temporal", "update")
FAULTS = ("missing", "stale", "flipped", "conflicting")
SOURCES = ("longmemeval", "memora", "oaks_novel")
BOOTSTRAP_REPLICATES = 10_000
SEED = 42


def _bootstrap_ci(values: list[float], *, replicates: int, seed: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        samples = rng.choice(np.asarray(values, dtype=float), size=(replicates, len(values)), replace=True).mean(axis=1)
        return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))
    except ImportError:  # pragma: no cover - numpy is available in the analysis environment
        rng = random.Random(seed)
        samples = [mean(rng.choices(values, k=len(values))) for _ in range(replicates)]
        return _percentile(samples, 0.025), _percentile(samples, 0.975)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _validate_frozen_inputs(
    run_a: list[dict[str, Any]],
    run_b: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    analysis_spec: dict[str, Any],
    failure_spec: dict[str, Any],
) -> None:
    condition_sha = sha256_file(Path(_validate_frozen_inputs.condition_path))
    if condition_sha not in ACCEPTED_CONDITION_SHAS:
        raise ValueError("Frozen condition manifest SHA256 mismatch")
    if sha256_file(Path(_validate_frozen_inputs.base_path)) != EXPECTED_BASE_SHA:
        raise ValueError("Frozen base manifest SHA256 mismatch")
    if len(condition_rows) != 4891:
        raise ValueError(f"Expected 4891 condition rows, found {len(condition_rows)}")
    if len(base_rows) != 731:
        raise ValueError(f"Expected 731 base rows, found {len(base_rows)}")
    condition_by_id = {row.get("condition_id"): row for row in condition_rows}
    if len(condition_by_id) != len(condition_rows) or None in condition_by_id:
        raise ValueError("Frozen condition manifest has duplicate or missing condition IDs")
    expected_ids = set(condition_by_id)
    for label, rows in (("run-a", run_a), ("run-b", run_b)):
        if len(rows) != 4891:
            raise ValueError(f"{label} must contain 4891 rows, found {len(rows)}")
        ids = [row.get("condition_id") for row in rows]
        if len(set(ids)) != len(ids) or None in ids:
            raise ValueError(f"{label} has duplicate or missing condition IDs")
        if set(ids) != expected_ids:
            raise ValueError(f"{label} condition IDs do not exactly match the frozen manifest")
        hashes = {row.get("manifest_sha256") for row in rows if row.get("manifest_sha256") is not None}
        if hashes and not hashes.issubset(ACCEPTED_CONDITION_SHAS):
            raise ValueError(f"{label} contains a mismatched manifest hash")
    if analysis_spec.get("status") != "FROZEN" or analysis_spec.get("deterministic_seed") != SEED:
        raise ValueError("Analysis spec is not the expected frozen seed-42 policy")
    if failure_spec.get("status") != "FROZEN_PRE_SCORING":
        raise ValueError("Failure-handling spec is not frozen pre-scoring")
    if analysis_spec.get("balanced_event_robustness", {}).get("counts") != {e: 169 for e in EVENTS}:
        raise ValueError("Balanced robustness policy does not specify 169 examples per event")


def _load_and_validate(run_a_path: Path, run_b_path: Path, condition_path: Path, base_path: Path, analysis_path: Path, failure_path: Path):
    # Keep paths on the function for the hash validator without adding global
    # mutable state to the analysis calculations.
    _validate_frozen_inputs.condition_path = condition_path
    _validate_frozen_inputs.base_path = base_path
    run_a, run_b = read_jsonl(run_a_path), read_jsonl(run_b_path)
    condition_rows, base_rows = read_jsonl(condition_path), read_jsonl(base_path)
    analysis_spec = json.loads(analysis_path.read_text(encoding="utf-8"))
    failure_spec = json.loads(failure_path.read_text(encoding="utf-8"))
    _validate_frozen_inputs(run_a, run_b, condition_rows, base_rows, analysis_spec, failure_spec)
    return run_a, run_b, condition_rows, base_rows, analysis_spec, failure_spec


def _authoritative_rows(result_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = {row["condition_id"]: row for row in manifest_rows}
    output = []
    for result in result_rows:
        metadata = manifest[result["condition_id"]]
        row = dict(result)
        # Frozen manifest values are authoritative for experimental grouping.
        for key in ("base_id", "source_dataset", "analysis_event_type", "primary_event_type", "condition_type", "fault_operator"):
            if key in metadata:
                row[key] = metadata[key]
        if "document_id" not in row:
            row["document_id"] = metadata.get("document_id") or (metadata.get("task_source_metadata") or {}).get("document_id")
        output.append(row)
    return output


def _score(row: dict[str, Any], *, complete_case: bool = False) -> int | None:
    final_failure = row.get("status") != "success" or row.get("parse_success") is not True
    if final_failure:
        return None if complete_case else 0
    # ``primary_score=0`` is present on some historical unscored rows for
    # compatibility.  Only final provider/parse failures are converted to
    # zero by the frozen failure policy; successful but unscored rows remain
    # missing and are excluded from metric-specific numerators.
    if row.get("scoring_status") == "unscored":
        return None
    if isinstance(row.get("primary_score"), bool):
        return int(row["primary_score"])
    if isinstance(row.get("primary_score"), int) and row["primary_score"] in (0, 1):
        return row["primary_score"]
    if isinstance(row.get("correctness"), bool):
        return int(row["correctness"])
    return None


def _pair_effects(rows: list[dict[str, Any]], *, complete_case: bool = False) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["base_id"]].append(row)
    effects = []
    for base_id, group in sorted(groups.items()):
        clean = [r for r in group if r["condition_type"] == "clean"]
        if len(clean) != 1:
            raise ValueError(f"Base {base_id} lacks exactly one clean row")
        clean_row = clean[0]
        clean_score = _score(clean_row, complete_case=complete_case)
        # One paired effect is defined per supported base×fault pair.  The
        # corrupted row is the pair's fault observation; its enforced sibling
        # is looked up below and must not create a second effect.
        for fault_row in sorted((r for r in group if r["condition_type"] == "corrupted"), key=lambda r: r["condition_id"]):
            fault_score = _score(fault_row, complete_case=complete_case)
            effects.append({
                "base_id": base_id,
                "source_dataset": fault_row["source_dataset"],
                "analysis_event_type": fault_row["analysis_event_type"],
                "fault_operator": fault_row.get("fault_operator"),
                "clean": clean_score,
                "corrupted": fault_score,
                "enforced": _score(next((r for r in group if r.get("pair_id") == fault_row.get("pair_id") and r["condition_type"] == "enforced"), {}), complete_case=complete_case) if fault_row.get("pair_id") else None,
                "document_id": fault_row.get("document_id"),
            })
    for effect in effects:
        effect["corruption_delta"] = effect["corrupted"] - effect["clean"] if effect["clean"] is not None and effect["corrupted"] is not None else None
        effect["recovery_delta"] = effect["enforced"] - effect["corrupted"] if effect["enforced"] is not None and effect["corrupted"] is not None else None
    return effects


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return mean(values) if values else None


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * p
    low, high = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def bootstrap_base_effects(effects: list[dict[str, Any]], field: str, *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = SEED) -> tuple[float | None, float | None]:
    by_base: dict[str, list[float]] = defaultdict(list)
    for row in effects:
        if row.get(field) is not None:
            by_base[row["base_id"]].append(row[field])
    base_values = [mean(values) for values in by_base.values()]
    if not base_values:
        return None, None
    return _bootstrap_ci(base_values, replicates=replicates, seed=seed)


def _effect_record(effects: list[dict[str, Any]], field: str, *, replicates: int = BOOTSTRAP_REPLICATES, with_ci: bool = True) -> dict[str, Any]:
    values = [row[field] for row in effects if row.get(field) is not None]
    bases = {row["base_id"] for row in effects if row.get(field) is not None}
    low, high = bootstrap_base_effects(effects, field, replicates=replicates) if with_ci else (None, None)
    return {"total_pair_n": len(effects), "paired_scored_n": len(values), "paired_unscored_n": len(effects) - len(values), "paired_coverage_rate": len(values) / len(effects) if effects else None, "base_n": len(bases), "mean_delta": _mean(values), "ci_low": low, "ci_high": high}


def _condition_accuracy(rows: list[dict[str, Any]], *, complete_case: bool = False) -> dict[str, Any]:
    result = {}
    for condition in ("clean", "corrupted", "enforced"):
        selected = [row for row in rows if row["condition_type"] == condition]
        scores = [_score(row, complete_case=complete_case) for row in selected]
        valid = [score for score in scores if score is not None]
        result[condition] = {"total_n": len(selected), "scored_n": len(valid), "unscored_n": len(selected) - len(valid), "coverage_rate": len(valid) / len(selected) if selected else None, "accuracy": _mean(valid)}
    return result


def _group_effects(effects: list[dict[str, Any]], key: str, *, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in effects:
        groups[str(row.get(key))].append(row)
    return {name: {"corruption": _effect_record(group, "corruption_delta", with_ci=False), "recovery": _effect_record(group, "recovery_delta", with_ci=False)} for name, group in sorted(groups.items())}


def _event_effects(effects: list[dict[str, Any]], *, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    result = {}
    for event in EVENTS:
        group = [row for row in effects if row["analysis_event_type"] == event]
        result[event] = {"corruption": _effect_record(group, "corruption_delta", replicates=replicates), "recovery": _effect_record(group, "recovery_delta", replicates=replicates)}
    return result


def _event_macro(effects: list[dict[str, Any]], field: str, *, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    event_values = {}
    for event in EVENTS:
        event_effects = [row for row in effects if row["analysis_event_type"] == event and row.get(field) is not None]
        by_base: dict[str, list[float]] = defaultdict(list)
        for row in event_effects:
            by_base[row["base_id"]].append(row[field])
        event_values[event] = {base: mean(values) for base, values in by_base.items()}
    available = [values for values in event_values.values() if values]
    observed = mean(mean(values.values()) for values in available) if len(available) == len(EVENTS) else None
    samples = []
    event_arrays = [list(values.values()) for values in event_values.values()]
    if len(event_arrays) == len(EVENTS) and all(event_arrays):
        try:
            import numpy as np
            rng = np.random.default_rng(SEED)
            sample_means = [rng.choice(np.asarray(values, dtype=float), size=(replicates, len(values)), replace=True).mean(axis=1) for values in event_arrays]
            samples = np.mean(np.vstack(sample_means), axis=0).tolist()
        except ImportError:  # pragma: no cover
            rng = random.Random(SEED)
            for _ in range(replicates):
                samples.append(mean(mean(rng.choices(values, k=len(values))) for values in event_arrays))
    return {"mean": observed, "ci_low": _percentile(samples, 0.025), "ci_high": _percentile(samples, 0.975), "event_count": len(available), "event_values": {event: _mean(values.values()) for event, values in event_values.items()}}


def _micro_effect(effects: list[dict[str, Any]], field: str, *, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    values = [row[field] for row in effects if row.get(field) is not None]
    low, high = bootstrap_base_effects(effects, field, replicates=replicates)
    return {"mean": _mean(values), "ci_low": low, "ci_high": high, "pair_n": len(values), "total_pair_n": len(effects), "coverage_rate": len(values) / len(effects) if effects else None}


def _balanced_base_ids(base_rows: list[dict[str, Any]]) -> set[str]:
    selected = set()
    for event in EVENTS:
        group = [row for row in base_rows if row.get("analysis_event_type") == event]
        if event == "temporal":
            chosen = group
        else:
            chosen = sorted(group, key=lambda row: hashlib.sha256(f"{SEED}|{event}|{row['base_id']}".encode()).hexdigest())[:169]
        if len(chosen) != 169:
            raise ValueError(f"Balanced cohort cannot select 169 {event} bases")
        selected.update(row["base_id"] for row in chosen)
    return selected


def _oaks_cluster_robustness(effects: list[dict[str, Any]], field: str, *, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    oaks = [row for row in effects if row["source_dataset"] == "oaks_novel" and row.get(field) is not None]
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in oaks:
        clusters[str(row.get("document_id"))].append(row[field])
    cluster_values = {key: mean(values) for key, values in clusters.items() if key != "None"}
    samples = []
    if cluster_values:
        low, high = _bootstrap_ci(list(cluster_values.values()), replicates=replicates, seed=SEED)
        samples = []
    lodo = [mean(value for key, value in cluster_values.items() if key != omitted) for omitted in cluster_values if len(cluster_values) > 1]
    return {"cluster_n": len(cluster_values), "base_n": len({row["base_id"] for row in oaks}), "mean": _mean(oaks[i][field] for i in range(len(oaks))), "ci_low": low if cluster_values else None, "ci_high": high if cluster_values else None, "leave_one_document_out": {"n": len(lodo), "min": min(lodo) if lodo else None, "max": max(lodo) if lodo else None, "values": lodo}}


def analyze_run(rows: list[dict[str, Any]], base_rows: list[dict[str, Any]], *, complete_case: bool = False, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    effects = _pair_effects(rows, complete_case=complete_case)
    event = _event_effects(effects, replicates=replicates)
    result = {"row_count": len(rows), "base_count": len({row["base_id"] for row in rows}), "pair_count": len(effects), "condition_accuracy": _condition_accuracy(rows, complete_case=complete_case), "per_event": event, "per_fault": _group_effects(effects, "fault_operator", replicates=replicates), "per_source": _group_effects(effects, "source_dataset", replicates=replicates), "event_macro": {"corruption": _event_macro(effects, "corruption_delta", replicates=replicates), "recovery": _event_macro(effects, "recovery_delta", replicates=replicates)}, "micro": {"corruption": _micro_effect(effects, "corruption_delta", replicates=replicates), "recovery": _micro_effect(effects, "recovery_delta", replicates=replicates)}}
    balanced_ids = _balanced_base_ids(base_rows)
    balanced_effects = [row for row in effects if row["base_id"] in balanced_ids]
    result["balanced_robustness"] = {"base_counts": {event: sum(row.get("analysis_event_type") == event and row.get("base_id") in balanced_ids for row in base_rows) for event in EVENTS}, "base_count": len(balanced_ids), "event_macro": {"corruption": _event_macro(balanced_effects, "corruption_delta", replicates=replicates), "recovery": _event_macro(balanced_effects, "recovery_delta", replicates=replicates)}}
    result["oaks_cluster_robustness"] = {"corruption": _oaks_cluster_robustness(effects, "corruption_delta", replicates=replicates), "recovery": _oaks_cluster_robustness(effects, "recovery_delta", replicates=replicates)}
    return result


def _replication(a: list[dict[str, Any]], b: list[dict[str, Any]], base_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def accuracy(rows, condition):
        vals = [_score(r) for r in rows if r["condition_type"] == condition]
        vals = [v for v in vals if v is not None]
        return mean(vals) if vals else None
    ar = {r["condition_id"]: r for r in a}; br = {r["condition_id"]: r for r in b}
    agreement = [(_score(ar[cid]), _score(br[cid])) for cid in ar]
    agreement = [(x, y) for x, y in agreement if x is not None and y is not None]
    effects_a = _pair_effects(a); effects_b = _pair_effects(b)
    def effect_mean(effects, field): return _mean(row[field] for row in effects if row.get(field) is not None)
    return {"run_a_clean_accuracy": accuracy(a, "clean"), "run_b_clean_accuracy": accuracy(b, "clean"), "clean_accuracy_difference": accuracy(b, "clean") - accuracy(a, "clean"), "run_a_corrupted_accuracy": accuracy(a, "corrupted"), "run_b_corrupted_accuracy": accuracy(b, "corrupted"), "corrupted_accuracy_difference": accuracy(b, "corrupted") - accuracy(a, "corrupted"), "run_a_enforced_accuracy": accuracy(a, "enforced"), "run_b_enforced_accuracy": accuracy(b, "enforced"), "enforced_accuracy_difference": accuracy(b, "enforced") - accuracy(a, "enforced"), "run_a_event_macro_corruption": _event_macro(effects_a, "corruption_delta")["mean"], "run_b_event_macro_corruption": _event_macro(effects_b, "corruption_delta")["mean"], "event_macro_corruption_difference": _event_macro(effects_b, "corruption_delta")["mean"] - _event_macro(effects_a, "corruption_delta")["mean"], "run_a_event_macro_recovery": _event_macro(effects_a, "recovery_delta")["mean"], "run_b_event_macro_recovery": _event_macro(effects_b, "recovery_delta")["mean"], "event_macro_recovery_difference": _event_macro(effects_b, "recovery_delta")["mean"] - _event_macro(effects_a, "recovery_delta")["mean"], "run_a_micro_corruption": effect_mean(effects_a, "corruption_delta"), "run_b_micro_corruption": effect_mean(effects_b, "corruption_delta"), "micro_corruption_difference": effect_mean(effects_b, "corruption_delta") - effect_mean(effects_a, "corruption_delta"), "run_a_micro_recovery": effect_mean(effects_a, "recovery_delta"), "run_b_micro_recovery": effect_mean(effects_b, "recovery_delta"), "micro_recovery_difference": effect_mean(effects_b, "recovery_delta") - effect_mean(effects_a, "recovery_delta"), "condition_level_n": len(agreement), "exact_correctness_agreement_n": sum(x == y for x, y in agreement), "exact_correctness_agreement_rate": sum(x == y for x, y in agreement) / len(agreement) if agreement else None, "disagreement_n": sum(x != y for x, y in agreement)}


def _write_table(path: Path, analyses: list[tuple[str, dict[str, Any]]]) -> None:
    rows = []
    for run_label, analysis in analyses:
        for scope, groups in (("event", analysis["per_event"]), ("fault", analysis["per_fault"]), ("source", analysis["per_source"])):
            for name, values in groups.items():
                for effect_name in ("corruption", "recovery"):
                    row = {"run": run_label, "scope": scope, "group": name, "effect": effect_name, **values[effect_name]}
                    rows.append(row)
        for aggregate, values in (("event_macro", analysis["event_macro"]), ("micro", analysis["micro"])):
            for effect_name, value in values.items():
                rows.append({"run": run_label, "scope": aggregate, "group": "all", "effect": effect_name, **value})
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def analyze(run_a_path: Path, run_b_path: Path, condition_path: Path, base_path: Path, analysis_path: Path, failure_path: Path, output_dir: Path) -> dict[str, Any]:
    run_a, run_b, condition, base, analysis_spec, failure_spec = _load_and_validate(run_a_path, run_b_path, condition_path, base_path, analysis_path, failure_path)
    a = _authoritative_rows(run_a, condition); b = _authoritative_rows(run_b, condition)
    primary_a = analyze_run(a, base); primary_b = analyze_run(b, base)
    cc_a = analyze_run(a, base, complete_case=True); cc_b = analyze_run(b, base, complete_case=True)
    summary = {"status": "FROZEN / FORMAL / ANALYZED", "manifest_sha256": sha256_file(condition_path), "base_manifest_sha256": sha256_file(base_path), "seed": SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "failure_policy": "final provider/parse failure scored as 0; all attempted rows retained", "run_a": primary_a, "run_b": primary_b, "complete_case_sensitivity": {"run_a": cc_a, "run_b": cc_b}, "run_to_run_replication": _replication(a, b, base), "analysis_spec": analysis_spec.get("schema_version"), "failure_spec": failure_spec.get("schema_version")}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_table(output_dir / "analysis_table.csv", [("run_a", primary_a), ("run_b", primary_b)])
    return summary


def analyze_single(result_path: Path, condition_path: Path, base_path: Path, analysis_path: Path, failure_path: Path, output_dir: Path) -> dict[str, Any]:
    """Analyze one complete provider result without fabricating replication."""
    condition_rows = read_jsonl(condition_path)
    base_rows = read_jsonl(base_path)
    analysis_spec = json.loads(analysis_path.read_text(encoding="utf-8"))
    failure_spec = json.loads(failure_path.read_text(encoding="utf-8"))
    rows = read_jsonl(result_path)
    # Reuse the frozen structural validator while deliberately omitting any
    # run-to-run comparison from the resulting artifact.
    _validate_frozen_inputs.condition_path = condition_path
    _validate_frozen_inputs.base_path = base_path
    _validate_frozen_inputs(rows, rows, condition_rows, base_rows, analysis_spec, failure_spec)
    authoritative = _authoritative_rows(rows, condition_rows)
    primary = analyze_run(authoritative, base_rows)
    complete_case = analyze_run(authoritative, base_rows, complete_case=True)
    summary = {
        "status": "FROZEN / FORMAL / SINGLE PROVIDER ANALYZED",
        "manifest_sha256": sha256_file(condition_path),
        "base_manifest_sha256": sha256_file(base_path),
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "failure_policy": "final provider/parse failure scored as 0; all attempted rows retained",
        "result_path": str(result_path),
        "primary": primary,
        "complete_case_sensitivity": complete_case,
        "replication": None,
        "analysis_spec": analysis_spec.get("schema_version"),
        "failure_spec": failure_spec.get("schema_version"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_table(output_dir / "analysis_table.csv", [("primary", primary)])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path)
    parser.add_argument("--run-b", type=Path)
    parser.add_argument("--primary", type=Path, help="One complete provider result; emits primary metrics only")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--analysis-spec", type=Path, required=True)
    parser.add_argument("--failure-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.primary:
        if args.run_a or args.run_b:
            parser.error("--primary cannot be combined with --run-a/--run-b")
        result = analyze_single(args.primary, args.manifest, args.base_manifest, args.analysis_spec, args.failure_spec, args.output_dir)
        print(json.dumps({"status": result["status"], "primary_rows": result["primary"]["row_count"], "manifest_sha256": result["manifest_sha256"]}, sort_keys=True))
    else:
        if not args.run_a or not args.run_b:
            parser.error("--run-a and --run-b are required unless --primary is supplied")
        result = analyze(args.run_a, args.run_b, args.manifest, args.base_manifest, args.analysis_spec, args.failure_spec, args.output_dir)
        print(json.dumps({"status": result["status"], "run_a_rows": result["run_a"]["row_count"], "run_b_rows": result["run_b"]["row_count"], "manifest_sha256": result["manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
