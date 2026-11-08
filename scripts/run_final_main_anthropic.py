"""Run the frozen final-main GovMem cohort through Anthropic Messages.

This entry point delegates prompt construction, scoring preflight, retries,
cache identity, and bounded concurrency to the existing provider-neutral
final-main runner.  Anthropic-specific behavior is limited to transport.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from eval.final_main_token_budget import load_budget_config, load_repair_budget_config, resolve_max_output_tokens
from eval.providers.anthropic_provider import AnthropicMessagesProvider
from eval.final_main_rerun_identity import assert_selective_identity, atomic_write_json

try:
    from scripts.run_final_main_smoke_deepseek import _estimate_tokens, render_prompt, run as run_shared
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from run_final_main_smoke_deepseek import _estimate_tokens, render_prompt, run as run_shared


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
DEFAULT_RESULTS = ROOT / "results/final_main_anthropic_sonnet5"
EXPECTED_MANIFEST_SHA = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"
EXPECTED_V3_MANIFEST_SHA = "4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb"
ACCEPTED_MANIFEST_SHAS = {EXPECTED_MANIFEST_SHA, EXPECTED_V3_MANIFEST_SHA}
EXPECTED_COUNTS = {"clean": 731, "corrupted": 2080, "enforced": 2080}
EXPECTED_BASE_COUNT = 731
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_INPUT_PRICE = 2.0
DEFAULT_OUTPUT_PRICE = 10.0


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _provider_factory(model: str, *, temperature: int = 0, max_tokens: int | None = None) -> AnthropicMessagesProvider:
    return AnthropicMessagesProvider(model=model, temperature=temperature, max_tokens=max_tokens or 256)


def _cost_estimate(rows: list[dict[str, Any]], budget_config: dict[str, Any] | None, input_price: float, output_price: float) -> dict[str, Any]:
    input_tokens = sum(_estimate_tokens(render_prompt(row)) for row in rows)
    output_tokens = sum(resolve_max_output_tokens(row, budget_config) for row in rows)
    return {
        "input_price_per_million": input_price,
        "output_price_per_million": output_price,
        "estimated_input_tokens": input_tokens,
        "estimated_max_output_tokens": output_tokens,
        "estimated_synchronous_upper_bound_cost": (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price,
        "pricing_basis": "token-budget upper-bound estimate; provider usage telemetry is authoritative",
    }


def _augment_summary(summary: dict[str, Any], rows: list[dict[str, Any]], budget_config: dict[str, Any] | None, input_price: float, output_price: float) -> None:
    budget_classes = Counter()
    budget_values: dict[str, int] = {}
    for row in rows:
        source = row.get("source_dataset")
        if source == "longmemeval":
            budget_class = "longmemeval"
        elif source == "oaks_novel":
            budget_class = "oaks_novel"
        else:
            # The shared resolver is authoritative; this value is only a
            # human-readable summary key and never selects rows.
            from eval.final_main_token_budget import budget_class as resolve_class
            budget_class = resolve_class(row)
        budget_classes[budget_class] += 1
        budget_values[budget_class] = resolve_max_output_tokens(row, budget_config)
    summary.update({
        "provider": "anthropic",
        "model": summary.get("model") or DEFAULT_MODEL,
        "reasoning_effort": "none",
        "model_parameters": {"temperature": 0, "thinking": "disabled", "tools": None},
        "source_counts": dict(sorted(Counter(row.get("source_dataset") for row in rows).items())),
        "analysis_event_type_counts": dict(sorted(Counter(row.get("analysis_event_type") for row in rows).items())),
        "condition_counts": dict(sorted(Counter(row.get("condition_type") for row in rows).items())),
        "resolved_token_budget_class_counts": dict(sorted(budget_classes.items())),
        "resolved_max_output_tokens": dict(sorted(budget_values.items())),
        "cost_estimate": _cost_estimate(rows, budget_config, input_price, output_price),
    })


def run(
    *,
    manifest: Path = DEFAULT_MANIFEST,
    results_dir: Path = DEFAULT_RESULTS,
    model: str = DEFAULT_MODEL,
    workers: int = 1,
    dry_run: bool = False,
    limit: int | None = None,
    force: bool = False,
    resume: bool = False,
    token_budget_config: Path | None = None,
    input_price_per_million: float = DEFAULT_INPUT_PRICE,
    output_price_per_million: float = DEFAULT_OUTPUT_PRICE,
    max_estimated_cost: float | None = None,
    override_cost_cap: bool = False,
    provider_factory: Any = None,
    selected_condition_ids: set[str] | None = None,
) -> dict[str, Any]:
    rows = _read(manifest)
    if selected_condition_ids is not None and limit is not None:
        raise ValueError("condition-id selection cannot be combined with --limit")
    available_ids = {row["condition_id"] for row in rows}
    if selected_condition_ids is not None:
        missing = sorted(set(selected_condition_ids) - available_ids)
        if missing:
            raise ValueError(f"Requested condition_id not found in frozen manifest: {', '.join(missing)}")
    budget_config = None
    if token_budget_config:
        raw_config = json.loads(token_budget_config.read_text(encoding="utf-8"))
        is_repair_config = raw_config.get("schema_version") == "final-main-token-budget-anthropic-repair-v1"
        if is_repair_config:
            if selected_condition_ids is None:
                raise ValueError("Anthropic repair budgets require --condition-ids-file selective repair")
            budget_config = load_repair_budget_config(token_budget_config)
        else:
            budget_config = load_budget_config(token_budget_config)
    selected_rows = [row for row in rows if selected_condition_ids is None or row["condition_id"] in selected_condition_ids]
    selected_rows = selected_rows if limit is None else selected_rows[:limit]
    if selected_condition_ids is not None:
        assert_selective_identity(provider="anthropic", model=model, model_parameters={"temperature": 0, "thinking": "disabled", "tools": None}, manifest=manifest, selected_rows=selected_rows, expected_ids=selected_condition_ids)
    factory = provider_factory or _provider_factory

    preflight = run_shared(
        manifest=manifest, results_dir=results_dir, model=model, dry_run=True, limit=limit,
        force=False, resume=False, provider_factory=None, expected_sha256=ACCEPTED_MANIFEST_SHAS,
        expected_counts=EXPECTED_COUNTS, expected_base_count=EXPECTED_BASE_COUNT,
        output_filename="full_results.jsonl", strict_scoring=True, workers=workers,
        provider_name="anthropic", budget_config=budget_config, selected_condition_ids=selected_condition_ids,
    )
    estimate = _cost_estimate(selected_rows, budget_config, input_price_per_million, output_price_per_million)
    if max_estimated_cost is not None and estimate["estimated_synchronous_upper_bound_cost"] > max_estimated_cost and not override_cost_cap:
        raise ValueError(
            f"Estimated upper-bound cost ${estimate['estimated_synchronous_upper_bound_cost']:.6f} exceeds --max-estimated-cost ${max_estimated_cost:.6f}; use --override-cost-cap explicitly"
        )
    if dry_run:
        summary = dict(preflight["summary"])
        _augment_summary(summary, selected_rows, budget_config, input_price_per_million, output_price_per_million)
        summary.update({"status": "FROZEN FULL / DRY-RUN", "provider": "anthropic", "model": model, "workers": workers, "api_calls_made": 0, "estimated_input_tokens": estimate["estimated_input_tokens"], "estimated_max_output_tokens": estimate["estimated_max_output_tokens"], "estimated_synchronous_upper_bound_cost": estimate["estimated_synchronous_upper_bound_cost"], "validation_errors": len(summary.get("scoring_preflight", {}).get("unresolved", []))})
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "full_dry_run_requests.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in preflight["rows"]), encoding="utf-8")
        (results_dir / "full_dry_run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"rows": preflight["rows"], "summary": summary}

    result = run_shared(
        manifest=manifest, results_dir=results_dir, model=model, dry_run=False, limit=limit,
        force=force, resume=resume, provider_factory=factory, expected_sha256=ACCEPTED_MANIFEST_SHAS,
        expected_counts=EXPECTED_COUNTS, expected_base_count=EXPECTED_BASE_COUNT,
        output_filename="full_results.jsonl", strict_scoring=True, workers=workers,
        provider_name="anthropic", budget_config=budget_config, selected_condition_ids=selected_condition_ids,
    )
    _augment_summary(result["summary"], selected_rows, budget_config, input_price_per_million, output_price_per_million)
    result["summary"].update({"model": model, "provider": "anthropic"})
    atomic_write_json(results_dir / "run_summary.json", result["summary"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--token-budget-config", type=Path)
    parser.add_argument("--input-price-per-million", type=float, default=DEFAULT_INPUT_PRICE)
    parser.add_argument("--output-price-per-million", type=float, default=DEFAULT_OUTPUT_PRICE)
    parser.add_argument("--max-estimated-cost", type=float)
    parser.add_argument("--override-cost-cap", action="store_true")
    parser.add_argument("--condition-ids-file", type=Path, help="one frozen condition_id per non-empty line")
    args = parser.parse_args(argv)
    selected_ids = None
    if args.condition_ids_file is not None:
        try:
            values = [line.strip() for line in args.condition_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError as exc:
            parser.error(f"cannot read --condition-ids-file: {exc}")
        if len(values) != len(set(values)):
            parser.error("--condition-ids-file contains duplicate condition IDs")
        if not values:
            parser.error("--condition-ids-file contains no non-empty condition IDs")
        selected_ids = set(values)
    try:
        result = run(
            manifest=args.manifest, results_dir=args.results_dir, model=args.model, workers=args.workers,
            dry_run=args.dry_run, limit=args.limit, force=args.force, resume=args.resume,
            token_budget_config=args.token_budget_config,
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
            max_estimated_cost=args.max_estimated_cost, override_cost_cap=args.override_cost_cap,
            selected_condition_ids=selected_ids,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
