"""Run the frozen GovMem final cohort through Moonshot Kimi."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from eval.final_main_token_budget import budget_class, load_kimi_budget_config, load_kimi_v2_budget_config, load_kimi_v3_budget_config, load_kimi_v4_budget_config, load_kimi_repair_v1_budget_config, resolve_max_output_tokens
from eval.providers.kimi_provider import DEFAULT_KIMI_BASE_URL, DEFAULT_KIMI_MODEL, KIMI_THINKING_DISABLED, KIMI_THINKING_PROVIDER_DEFAULT, KimiDownstreamProvider, kimi_temperature_for_thinking
from eval.final_main_rerun_identity import assert_selective_identity, atomic_write_json

try:
    from scripts.run_final_main_smoke_deepseek import _estimate_tokens, render_prompt, run as run_shared
except ModuleNotFoundError:  # pragma: no cover
    from run_final_main_smoke_deepseek import _estimate_tokens, render_prompt, run as run_shared

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
DEFAULT_RESULTS = ROOT / "results/final_main_kimi"
DEFAULT_BUDGET_CONFIG = ROOT / "configs/final_main_token_budget_kimi_v1.json"
EXPECTED_MANIFEST_SHA = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"
EXPECTED_V3_MANIFEST_SHA = "4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb"
ACCEPTED_MANIFEST_SHAS = {EXPECTED_MANIFEST_SHA, EXPECTED_V3_MANIFEST_SHA}
EXPECTED_COUNTS = {"clean": 731, "corrupted": 2080, "enforced": 2080}


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ids_file(path: Path) -> set[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError("--condition-ids-file contains no non-empty IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("--condition-ids-file contains duplicate condition IDs")
    return set(ids)


def _cost(rows: list[dict[str, Any]], config: dict[str, Any], input_price: float, output_price: float) -> dict[str, Any]:
    inputs = sum(_estimate_tokens(render_prompt(row)) for row in rows)
    outputs = sum(resolve_max_output_tokens(row, config) for row in rows)
    return {"estimated_input_tokens": inputs, "estimated_max_output_tokens": outputs, "input_price_per_million": input_price, "output_price_per_million": output_price, "estimated_synchronous_upper_bound_cost": inputs / 1_000_000 * input_price + outputs / 1_000_000 * output_price, "pricing_basis": "token-budget upper-bound estimate; provider usage telemetry is authoritative"}


def _augment(summary: dict[str, Any], rows: list[dict[str, Any]], config: dict[str, Any], input_price: float, output_price: float, model: str, base_url: str, thinking: str) -> None:
    temperature = kimi_temperature_for_thinking(thinking)
    classes = Counter(budget_class(row) for row in rows)
    budgets = {name: resolve_max_output_tokens(next(row for row in rows if budget_class(row) == name), config) for name in sorted(classes)}
    summary.update({"provider": "kimi", "model": model, "base_url": base_url, "source_counts": dict(sorted(Counter(row["source_dataset"] for row in rows).items())), "analysis_event_type_counts": dict(sorted(Counter(row["analysis_event_type"] for row in rows).items())), "resolved_token_budget_class_counts": dict(sorted(classes.items())), "resolved_max_output_tokens": budgets, "cost_estimate": _cost(rows, config, input_price, output_price), "model_parameters": {"temperature": temperature, "top_p": 0.95, "tools": None, "thinking": thinking}})


def run(*, manifest: Path = DEFAULT_MANIFEST, results_dir: Path = DEFAULT_RESULTS, model: str = DEFAULT_KIMI_MODEL, base_url: str | None = None, token_budget_config: Path = DEFAULT_BUDGET_CONFIG, workers: int = 1, limit: int | None = None, condition_ids: set[str] | None = None, dry_run: bool = False, force: bool = False, resume: bool = False, input_price_per_million: float = 0, output_price_per_million: float = 0, thinking: str = KIMI_THINKING_PROVIDER_DEFAULT, provider_factory: Any = None) -> dict[str, Any]:
    if thinking not in {KIMI_THINKING_PROVIDER_DEFAULT, KIMI_THINKING_DISABLED}:
        raise ValueError(f"Unsupported Kimi thinking mode: {thinking}")
    rows = _read(manifest)
    available = {row["condition_id"] for row in rows}
    if condition_ids is not None:
        missing = sorted(condition_ids - available)
        if missing:
            raise ValueError(f"Requested condition_id not found in frozen manifest: {', '.join(missing)}")
    raw_config = json.loads(token_budget_config.read_text(encoding="utf-8"))
    schema = raw_config.get("schema_version")
    if schema == "final-main-token-budget-kimi-v1":
        config = load_kimi_budget_config(token_budget_config)
    elif schema == "final-main-token-budget-kimi-v2":
        config = load_kimi_v2_budget_config(token_budget_config)
    elif schema == "final-main-token-budget-kimi-v3":
        config = load_kimi_v3_budget_config(token_budget_config)
    elif schema == "final-main-token-budget-kimi-v4":
        config = load_kimi_v4_budget_config(token_budget_config)
    elif schema == "final-main-token-budget-kimi-repair-v1":
        config = load_kimi_repair_v1_budget_config(token_budget_config)
    else:
        raise ValueError(f"Unsupported Kimi token-budget schema: {schema}")
    resolved_base_url = base_url or os.environ.get("KIMI_BASE_URL", DEFAULT_KIMI_BASE_URL)
    temperature = kimi_temperature_for_thinking(thinking)
    parameters = {"temperature": temperature, "top_p": 0.95, "thinking": thinking, "tools": None, "base_url": resolved_base_url}
    selected = [row for row in rows if condition_ids is None or row["condition_id"] in condition_ids]
    if limit is not None:
        selected = selected[:limit]
    if condition_ids is not None:
        assert_selective_identity(provider="kimi", model=model, model_parameters={"base_url": resolved_base_url, "temperature": temperature, "thinking": thinking, "top_p": 0.95, "tools": None}, manifest=manifest, selected_rows=selected, expected_ids=condition_ids)
    factory = provider_factory or (lambda selected_model, **kwargs: KimiDownstreamProvider(model=selected_model, base_url=resolved_base_url, temperature=temperature, top_p=0.95, thinking=thinking, max_tokens=kwargs.get("max_tokens")))
    result = run_shared(manifest=manifest, results_dir=results_dir, model=model, dry_run=dry_run, limit=limit, force=force, resume=resume, provider_factory=factory, expected_sha256=ACCEPTED_MANIFEST_SHAS, expected_counts=EXPECTED_COUNTS, expected_base_count=731, output_filename="full_results.jsonl", strict_scoring=True, workers=workers, provider_name="kimi", selected_condition_ids=condition_ids, budget_config=config, model_parameters=parameters)
    _augment(result["summary"], selected, config, input_price_per_million, output_price_per_million, model, resolved_base_url, thinking)
    result["summary"].update({"requested_rows": len(selected), "api_calls_made": 0 if dry_run else result["summary"].get("total_provider_attempts", 0), "status": "KIMI / DRY-RUN" if dry_run else result["summary"].get("status")})
    if not dry_run:
        atomic_write_json(results_dir / "run_summary.json", result["summary"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default=DEFAULT_KIMI_MODEL)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token-budget-config", type=Path, default=DEFAULT_BUDGET_CONFIG)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--condition-ids-file", type=Path)
    parser.add_argument("--thinking", choices=[KIMI_THINKING_PROVIDER_DEFAULT, KIMI_THINKING_DISABLED], default=KIMI_THINKING_PROVIDER_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input-price-per-million", type=float, default=0)
    parser.add_argument("--output-price-per-million", type=float, default=0)
    args = parser.parse_args(argv)
    ids = _ids_file(args.condition_ids_file) if args.condition_ids_file else None
    result = run(manifest=args.manifest, results_dir=args.results_dir, model=args.model, base_url=args.base_url, token_budget_config=args.token_budget_config, workers=args.workers, limit=args.limit, condition_ids=ids, dry_run=args.dry_run, input_price_per_million=args.input_price_per_million, output_price_per_million=args.output_price_per_million, thinking=args.thinking)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
