#!/usr/bin/env python3
"""Run frozen GovMem final-main conditions through xAI Grok."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from eval.final_main_token_budget import budget_class, load_grok_v1_budget_config, resolve_max_output_tokens
from eval.providers.grok_provider import DEFAULT_GROK_BASE_URL, DEFAULT_GROK_MODEL, GrokDownstreamProvider
from eval.final_main_rerun_identity import assert_selective_identity, atomic_write_json

try:
    from scripts.run_final_main_smoke_deepseek import _estimate_tokens, render_prompt, run as run_shared
except ModuleNotFoundError:  # pragma: no cover
    from run_final_main_smoke_deepseek import _estimate_tokens, render_prompt, run as run_shared

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
DEFAULT_RESULTS = ROOT / "results/final_main_grok"
DEFAULT_BUDGET_CONFIG = ROOT / "configs/final_main_token_budget_grok_v1.json"
EXPECTED_MANIFEST_SHA = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"
EXPECTED_V3_MANIFEST_SHA = "4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb"
ACCEPTED_MANIFEST_SHAS = {EXPECTED_MANIFEST_SHA, EXPECTED_V3_MANIFEST_SHA}
EXPECTED_COUNTS = {"clean": 731, "corrupted": 2080, "enforced": 2080}


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ids_file(path: Path) -> set[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError("--condition-ids-file contains no non-empty IDs")
    if len(values) != len(set(values)):
        raise ValueError("--condition-ids-file contains duplicate condition IDs")
    return set(values)


def _cost(rows: list[dict[str, Any]], config: dict[str, Any], input_price: float, output_price: float) -> dict[str, Any]:
    inputs = sum(_estimate_tokens(render_prompt(row)) for row in rows)
    outputs = sum(resolve_max_output_tokens(row, config) for row in rows)
    return {"estimated_input_tokens": inputs, "estimated_max_output_tokens": outputs, "input_price_per_million": input_price, "output_price_per_million": output_price, "estimated_synchronous_upper_bound_cost": inputs / 1_000_000 * input_price + outputs / 1_000_000 * output_price, "pricing_basis": "token-budget upper-bound estimate; provider usage telemetry is authoritative"}


def _augment(summary: dict[str, Any], rows: list[dict[str, Any]], config: dict[str, Any], input_price: float, output_price: float, model: str, base_url: str, parameters: dict[str, Any]) -> None:
    classes = Counter(budget_class(row) for row in rows)
    budgets = {name: resolve_max_output_tokens(next(row for row in rows if budget_class(row) == name), config) for name in sorted(classes)}
    summary.update({"provider": "xai", "model": model, "base_url": base_url, "source_counts": dict(sorted(Counter(row["source_dataset"] for row in rows).items())), "analysis_event_type_counts": dict(sorted(Counter(row["analysis_event_type"] for row in rows).items())), "resolved_token_budget_class_counts": dict(sorted(classes.items())), "resolved_max_output_tokens": budgets, "cost_estimate": _cost(rows, config, input_price, output_price), "model_parameters": parameters})


def run(*, manifest: Path = DEFAULT_MANIFEST, results_dir: Path = DEFAULT_RESULTS, model: str = DEFAULT_GROK_MODEL, base_url: str | None = None, token_budget_config: Path = DEFAULT_BUDGET_CONFIG, workers: int = 1, limit: int | None = None, condition_ids: set[str] | None = None, dry_run: bool = False, force: bool = False, resume: bool = False, reasoning_effort: str = "none", input_price_per_million: float = 0, output_price_per_million: float = 0, provider_factory: Any = None) -> dict[str, Any]:
    if reasoning_effort != "none":
        raise ValueError("Grok final-main requires --reasoning-effort none")
    rows = _read(manifest)
    available = {row["condition_id"] for row in rows}
    if condition_ids is not None:
        missing = sorted(condition_ids - available)
        if missing:
            raise ValueError(f"Requested condition_id not found in frozen manifest: {', '.join(missing)}")
    config = load_grok_v1_budget_config(token_budget_config)
    resolved_base_url = base_url or os.environ.get("XAI_BASE_URL", DEFAULT_GROK_BASE_URL)
    # xAI's optional sampling controls are omitted rather than guessed; the
    # omitted temperature/top_p are represented as None in audit metadata.
    parameters = {"reasoning_effort": "none", "temperature": None, "top_p": None, "tools": None, "base_url": resolved_base_url}
    selected = [row for row in rows if condition_ids is None or row["condition_id"] in condition_ids]
    if limit is not None:
        selected = selected[:limit]
    if condition_ids is not None:
        assert_selective_identity(provider="xai", model=model, model_parameters={"base_url": resolved_base_url, "reasoning_effort": "none", "temperature": None, "thinking": "disabled", "top_p": None, "tools": None}, manifest=manifest, selected_rows=selected, expected_ids=condition_ids)
    factory = provider_factory or (lambda selected_model, **kwargs: GrokDownstreamProvider(model=selected_model, base_url=resolved_base_url, reasoning_effort="none", temperature=None, max_tokens=kwargs.get("max_tokens")))
    result = run_shared(manifest=manifest, results_dir=results_dir, model=model, dry_run=dry_run, limit=limit, force=force, resume=resume, provider_factory=factory, expected_sha256=ACCEPTED_MANIFEST_SHAS, expected_counts=EXPECTED_COUNTS, expected_base_count=731, output_filename="full_results.jsonl", strict_scoring=True, workers=workers, provider_name="xai", selected_condition_ids=condition_ids, budget_config=config, model_parameters=parameters)
    _augment(result["summary"], selected, config, input_price_per_million, output_price_per_million, model, resolved_base_url, parameters)
    result["summary"].update({"requested_rows": len(selected), "api_calls_made": 0 if dry_run else result["summary"].get("total_provider_attempts", 0), "status": "GROK / DRY-RUN" if dry_run else result["summary"].get("status"), "reasoning_effort": "none"})
    if not dry_run:
        atomic_write_json(results_dir / "run_summary.json", result["summary"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default=DEFAULT_GROK_MODEL)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token-budget-config", type=Path, default=DEFAULT_BUDGET_CONFIG)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--condition-ids-file", type=Path)
    parser.add_argument("--reasoning-effort", choices=["none"], default="none")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input-price-per-million", type=float, default=0)
    parser.add_argument("--output-price-per-million", type=float, default=0)
    args = parser.parse_args(argv)
    ids = _ids_file(args.condition_ids_file) if args.condition_ids_file else None
    result = run(manifest=args.manifest, results_dir=args.results_dir, model=args.model, base_url=args.base_url, token_budget_config=args.token_budget_config, workers=args.workers, limit=args.limit, condition_ids=ids, reasoning_effort=args.reasoning_effort, dry_run=args.dry_run, input_price_per_million=args.input_price_per_million, output_price_per_million=args.output_price_per_million)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
