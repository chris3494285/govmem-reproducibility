"""Summarize completed deterministic smoke responses without model calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if isinstance(r.get("correctness"), bool)]
    return {"n": len(rows), "scored_n": len(scored), "accuracy": (sum(r["correctness"] for r in scored) / len(scored) if scored else None)}


def _pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = {r["base_id"]: r for r in rows if r.get("condition_type") == "clean"}
    faults = defaultdict(dict)
    for r in rows:
        if r.get("condition_type") in {"corrupted", "enforced"}:
            faults[r["pair_id"]][r["condition_type"]] = r
    out = []
    for pair_id, pair in faults.items():
        base_id = pair_id.rsplit("::", 1)[0]
        if base_id not in clean or "corrupted" not in pair or "enforced" not in pair:
            continue
        c, f, e = clean[base_id], pair["corrupted"], pair["enforced"]
        out.append({"pair_id": pair_id, "base_id": base_id, "fault_operator": f.get("fault_operator"), "analysis_event_type": f.get("analysis_event_type"), "source_dataset": f.get("source_dataset"), "clean": c.get("correctness"), "corrupted": f.get("correctness"), "enforced": e.get("correctness")})
    return out


def _effects(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [p for p in pairs if isinstance(p["clean"], bool) and isinstance(p["corrupted"], bool) and isinstance(p["enforced"], bool)]
    def mean(values):
        return sum(values) / len(values) if values else None
    recovery = [p["enforced"] - p["corrupted"] for p in eligible]
    harm = [p["corrupted"] - p["clean"] for p in eligible]
    meaningful = [p for p in eligible if p["clean"] - p["corrupted"] != 0]
    fractions = [(p["enforced"] - p["corrupted"]) / (p["clean"] - p["corrupted"]) for p in meaningful]
    return {"n": len(pairs), "scored_n": len(eligible), "corruption_delta": {"mean": mean(harm)}, "recovery_delta": {"mean": mean(recovery)}, "recovery_fraction": {"mean": mean(fractions), "scored_n": len(fractions)}}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No smoke results found")
    pairs = _pairs(rows)
    grouped = defaultdict(list)
    for p in pairs:
        grouped[(p["fault_operator"], p["analysis_event_type"], p["source_dataset"])].append(p)
    condition_groups = defaultdict(list)
    for row in rows:
        condition_groups[(row.get("source_dataset"), row.get("analysis_event_type"), row.get("fault_operator"), row.get("condition_type"))].append(row)
    causal = Counter((p["clean"], p["corrupted"], p["enforced"]) for p in pairs)
    sort_key = lambda item: tuple("" if value is None else str(value) for value in item[0])
    return {"condition_accuracy": {kind: _accuracy([r for r in rows if r.get("condition_type") == kind]) for kind in ("clean", "corrupted", "enforced")}, "condition_accuracy_by_source_event_fault": {"|".join(str(x) for x in key): _accuracy(value) for key, value in sorted(condition_groups.items(), key=sort_key)}, "pairwise_overall": _effects(pairs), "pairwise_by_fault_event_source": {"|".join(str(x) for x in key): _effects(value) for key, value in sorted(grouped.items(), key=sort_key)}, "causal_sanity_checks": {"clean_correct_corrupted_wrong_enforced_correct": causal[(True, False, True)], "clean_correct_corrupted_wrong_enforced_wrong": causal[(True, False, False)], "clean_correct_corrupted_correct_enforced_correct": causal[(True, True, True)], "clean_wrong_baseline": sum(p["clean"] is False for p in pairs), "enforcement_hurts": sum(p["enforced"] is False and p["corrupted"] is True for p in pairs), "parse_or_error_failures": sum(r.get("parse_success") is not True for r in rows)}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/final_main_smoke_deepseek_v4_flash/smoke_results.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/final_main_smoke_deepseek_v4_flash/metrics_summary.json"))
    args = parser.parse_args(argv)
    result = summarize(_read(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
