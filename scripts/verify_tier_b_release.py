#!/usr/bin/env python3
"""Verify that Tier-B pseudonymous scores reproduce the frozen aggregates.

Offline only. It reverses the published pseudonymisation rule with the private
original->pseudonym mapping, re-runs the frozen six-provider analyses on the
reconstructed rows, and compares every recomputed statistic to the frozen
published summaries. Provider API calls: 0.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_control_plane_propagation import analyze as analyze_propagation
from scripts.analyze_final_main_results import analyze_single
from scripts.analyze_matched_fault_support import analyze as analyze_matched

PROVIDERS = ("openai", "anthropic", "deepseek", "gemini", "kimi", "grok")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def reverse_map(tier_b_dir: Path, mapping_path: Path, out_dir: Path) -> dict[str, Path]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    paths: dict[str, Path] = {}
    for provider in PROVIDERS:
        rows = _load_jsonl(tier_b_dir / f"{provider}.jsonl")
        for row in rows:
            row["base_id"] = mapping["base_id"][row["base_id"]]
            row["condition_id"] = mapping["condition_id"][row["condition_id"]]
            if row.get("pair_id") is not None:
                row["pair_id"] = mapping["pair_id"][row["pair_id"]]
        path = out_dir / f"{provider}.jsonl"
        _write_jsonl(path, rows)
        paths[provider] = path
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier-b-dir", type=Path, default=ROOT / "results/final_main_tier_b")
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/final_main_governance_condition_manifest_v3.jsonl")
    parser.add_argument("--base-manifest", type=Path, default=ROOT / "data/final_main_governance_base_manifest_v2.jsonl")
    parser.add_argument("--analysis-spec", type=Path, default=ROOT / "configs/main_analysis_spec_v1.json")
    parser.add_argument("--failure-spec", type=Path, default=ROOT / "configs/main_failure_handling_spec_v1.json")
    parser.add_argument("--frozen-analysis", type=Path, default=ROOT / "results/answer_leakage_repair/final_v3_analysis")
    args = parser.parse_args()

    results: dict[str, object] = {"checks": {}, "failures": []}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        reconstructed = reverse_map(args.tier_b_dir, args.mapping, tmp_path / "reconstructed")
        for provider in PROVIDERS:
            got = analyze_single(
                reconstructed[provider], args.manifest, args.base_manifest,
                args.analysis_spec, args.failure_spec, tmp_path / f"single_{provider}",
            )
            frozen = json.loads((args.frozen_analysis / "providers" / provider / "analysis_summary.json").read_text())
            equal = got.get("primary") == frozen.get("primary")
            results["checks"][f"provider_primary::{provider}"] = equal
            if not equal:
                results["failures"].append(f"provider primary mismatch: {provider}")

        propagation = analyze_propagation(
            manifest=args.manifest, result_paths=reconstructed, output_dir=tmp_path / "propagation"
        )
        frozen_prop = json.loads((args.frozen_analysis / "control_plane_propagation/mechanism_summary.json").read_text())
        prop_equal = propagation.get("overall") == frozen_prop.get("overall")
        results["checks"]["control_plane_overall"] = prop_equal
        if not prop_equal:
            results["failures"].append("control-plane overall mismatch")

        matched = analyze_matched(
            manifest=args.manifest, provider_paths=reconstructed, output_dir=tmp_path / "matched"
        )
        frozen_matched = json.loads((args.frozen_analysis / "matched_fault_support/matched_fault_summary.json").read_text())
        for key in ("matched_instance_count", "overlap_counts", "comparisons", "support_counts"):
            equal = matched.get(key) == frozen_matched.get(key)
            results["checks"][f"matched_fault::{key}"] = equal
            if not equal:
                results["failures"].append(f"matched-fault {key} mismatch")

    results["status"] = "PASS" if not results["failures"] else "FAIL"
    results["api_calls_made"] = 0
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
