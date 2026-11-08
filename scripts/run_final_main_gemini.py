"""Offline-safe/full-run entry point for Gemini on the frozen final cohort."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from eval.final_main_token_budget import budget_class, load_budget_config, load_gemini_budget_config, load_gemini_flash_budget_config, load_gemini_flash_repair_budget_config, load_gemini_flash_repair_v2_budget_config, resolve_max_output_tokens
from eval.providers.gemini_provider import DEFAULT_GEMINI_MODEL, GeminiGenerativeProvider
from eval.final_main_rerun_identity import assert_selective_identity, atomic_write_json

try:
    from scripts.run_final_main_smoke_deepseek import _estimate_tokens, render_prompt, run as run_shared
except ModuleNotFoundError:  # pragma: no cover
    from run_final_main_smoke_deepseek import _estimate_tokens, render_prompt, run as run_shared

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
DEFAULT_RESULTS = ROOT / "results/final_main_gemini_3_flash"
EXPECTED_MANIFEST_SHA = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"
EXPECTED_V3_MANIFEST_SHA = "4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb"
ACCEPTED_MANIFEST_SHAS = {EXPECTED_MANIFEST_SHA, EXPECTED_V3_MANIFEST_SHA}
EXPECTED_COUNTS = {"clean": 731, "corrupted": 2080, "enforced": 2080}
DEFAULT_INPUT_PRICE = 0.50
DEFAULT_OUTPUT_PRICE = 3.0


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ids_file(path: Path) -> set[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError("--condition-ids-file contains no non-empty IDs")
    if len(values) != len(set(values)):
        raise ValueError("--condition-ids-file contains duplicate condition IDs")
    return set(values)


def _cost(rows: list[dict[str, Any]], config: dict[str, Any] | None, input_price: float, output_price: float) -> dict[str, Any]:
    input_tokens = sum(_estimate_tokens(render_prompt(row)) for row in rows)
    max_output = sum(resolve_max_output_tokens(row, config) for row in rows)
    return {"estimated_input_tokens": input_tokens, "estimated_max_output_tokens": max_output, "input_price_per_million": input_price, "output_price_per_million": output_price, "estimated_synchronous_upper_bound_cost": input_tokens / 1_000_000 * input_price + max_output / 1_000_000 * output_price, "pricing_basis": "token-budget upper-bound estimate; provider usage telemetry is authoritative"}


def _augment(summary: dict[str, Any], rows: list[dict[str, Any]], config: dict[str, Any] | None, input_price: float, output_price: float) -> None:
    classes = Counter(budget_class(row) for row in rows)
    budgets = {name: resolve_max_output_tokens(next(row for row in rows if budget_class(row) == name), config) for name in sorted(classes)}
    summary.update({"provider": "gemini", "source_counts": dict(sorted(Counter(row.get("source_dataset") for row in rows).items())), "analysis_event_type_counts": dict(sorted(Counter(row.get("analysis_event_type") for row in rows).items())), "condition_counts": dict(sorted(Counter(row.get("condition_type") for row in rows).items())), "fault_counts": dict(sorted(Counter(row.get("fault_operator") for row in rows if row.get("fault_operator")).items())), "resolved_token_budget_class_counts": dict(sorted(classes.items())), "resolved_max_output_tokens": budgets, "cost_estimate": _cost(rows, config, input_price, output_price), "model_parameters": {"temperature": 0, "thinking": "low", "thinking_level": "LOW", "tools": None}})


def run(*, manifest: Path = DEFAULT_MANIFEST, results_dir: Path = DEFAULT_RESULTS, model: str = DEFAULT_GEMINI_MODEL, workers: int = 1, dry_run: bool = False, limit: int | None = None, force: bool = False, resume: bool = False, token_budget_config: Path | None = None, condition_ids: set[str] | None = None, input_price_per_million: float = DEFAULT_INPUT_PRICE, output_price_per_million: float = DEFAULT_OUTPUT_PRICE, max_estimated_cost: float | None = None, provider_factory: Any = None) -> dict[str, Any]:
    rows = _read(manifest)
    available = {row["condition_id"] for row in rows}
    if condition_ids is not None:
        missing = sorted(condition_ids - available)
        if missing:
            raise ValueError(f"Requested condition_id not found in frozen manifest: {', '.join(missing)}")
    config = None
    if token_budget_config:
        raw_config = json.loads(token_budget_config.read_text(encoding="utf-8"))
        if raw_config.get("schema_version") == "final-main-token-budget-gemini-flash-repair-v2":
            if condition_ids is None:
                raise ValueError("Gemini Flash repair-v2 budgets require --condition-ids-file selective repair")
            config = load_gemini_flash_repair_v2_budget_config(token_budget_config)
        elif raw_config.get("schema_version") == "final-main-token-budget-gemini-flash-repair-v1":
            if condition_ids is None:
                raise ValueError("Gemini Flash repair budgets require --condition-ids-file selective repair")
            config = load_gemini_flash_repair_budget_config(token_budget_config)
        elif raw_config.get("schema_version") == "final-main-token-budget-gemini-flash-v1":
            config = load_gemini_flash_budget_config(token_budget_config)
        elif raw_config.get("schema_version") == "final-main-token-budget-gemini-v1":
            config = load_gemini_budget_config(token_budget_config)
        else:
            config = load_budget_config(token_budget_config)
    gemini_parameters = {"thinking": "low", "thinking_level": "LOW", "tools": None}
    selected = [row for row in rows if condition_ids is None or row["condition_id"] in condition_ids]
    if limit is not None:
        selected = selected[:limit]
    if condition_ids is not None:
        assert_selective_identity(provider="gemini", model=model, model_parameters={"temperature": 0, "thinking": "low", "thinking_level": "LOW", "tools": None}, manifest=manifest, selected_rows=selected, expected_ids=condition_ids)
    preflight = run_shared(manifest=manifest, results_dir=results_dir, model=model, dry_run=True, limit=limit, expected_sha256=ACCEPTED_MANIFEST_SHAS, expected_counts=EXPECTED_COUNTS, expected_base_count=731, output_filename="full_results.jsonl", strict_scoring=True, workers=workers, provider_name="gemini", selected_condition_ids=condition_ids, budget_config=config, model_parameters=gemini_parameters)
    estimate = _cost(selected, config, input_price_per_million, output_price_per_million)
    if max_estimated_cost is not None and estimate["estimated_synchronous_upper_bound_cost"] > max_estimated_cost:
        raise ValueError(f"Estimated upper-bound cost exceeds --max-estimated-cost: ${estimate['estimated_synchronous_upper_bound_cost']:.6f} > ${max_estimated_cost:.6f}")
    if dry_run:
        summary = dict(preflight["summary"])
        _augment(summary, selected, config, input_price_per_million, output_price_per_million)
        summary.update({"status": "FROZEN FULL / DRY-RUN", "provider": "gemini", "model": model, "workers": workers, "api_calls_made": 0, "requested_rows": len(selected), "validation_errors": len(summary.get("scoring_preflight", {}).get("unresolved", []))})
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "full_dry_run_requests.jsonl").write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in preflight["rows"]), encoding="utf-8")
        (results_dir / "full_dry_run_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return {"rows": preflight["rows"], "summary": summary}
    result = run_shared(manifest=manifest, results_dir=results_dir, model=model, dry_run=False, limit=limit, force=force, resume=resume, provider_factory=provider_factory or _provider_factory, expected_sha256=ACCEPTED_MANIFEST_SHAS, expected_counts=EXPECTED_COUNTS, expected_base_count=731, output_filename="full_results.jsonl", strict_scoring=True, workers=workers, provider_name="gemini", selected_condition_ids=condition_ids, budget_config=config, model_parameters=gemini_parameters)
    _augment(result["summary"], selected, config, input_price_per_million, output_price_per_million)
    result["summary"].update({"provider": "gemini", "model": model})
    atomic_write_json(results_dir / "run_summary.json", result["summary"])
    return result


def _provider_factory(model: str, *, temperature: int = 0, max_tokens: int | None = None) -> GeminiGenerativeProvider:
    return GeminiGenerativeProvider(model=model, temperature=temperature, max_tokens=max_tokens or 256)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--token-budget-config", type=Path)
    parser.add_argument("--condition-ids-file", type=Path)
    parser.add_argument("--input-price-per-million", type=float, default=DEFAULT_INPUT_PRICE)
    parser.add_argument("--output-price-per-million", type=float, default=DEFAULT_OUTPUT_PRICE)
    parser.add_argument("--max-estimated-cost", type=float)
    args = parser.parse_args(argv)
    selected = _ids_file(args.condition_ids_file) if args.condition_ids_file else None
    result = run(manifest=args.manifest, results_dir=args.results_dir, model=args.model, workers=args.workers, dry_run=args.dry_run, limit=args.limit, force=args.force, resume=args.resume, token_budget_config=args.token_budget_config, condition_ids=selected, input_price_per_million=args.input_price_per_million, output_price_per_million=args.output_price_per_million, max_estimated_cost=args.max_estimated_cost)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
