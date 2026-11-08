"""Run the frozen full GovMem condition cohort through DeepSeek.

Execution, prompt rendering, scoring preflight, retry policy, and cache
semantics are shared with the validated smoke runner.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from eval.final_main_token_budget import load_budget_config, memora_task_subtype, project_proposal_condition_ids, resolve_max_output_tokens
from eval.final_main_rerun_identity import assert_selective_identity, atomic_write_json
try:
    from scripts.run_final_main_smoke_deepseek import run as run_shared
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from run_final_main_smoke_deepseek import run as run_shared

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
DEFAULT_RESULTS = ROOT / "results/final_main_deepseek_v4_flash"
EXPECTED_MANIFEST_SHA = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"
EXPECTED_V3_MANIFEST_SHA = "4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb"
ACCEPTED_MANIFEST_SHAS = {EXPECTED_MANIFEST_SHA, EXPECTED_V3_MANIFEST_SHA}
EXPECTED_COUNTS = {"clean": 731, "corrupted": 2080, "enforced": 2080}
EXPECTED_BASE_COUNT = 731


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summary_fields(rows: list[dict[str, Any]], budget_config: dict[str, Any] | None = None) -> dict[str, Any]:
    def grouped(fields: tuple[str, ...], include_none: bool = True) -> dict[str, int]:
        result = Counter()
        for row in rows:
            values = tuple(row.get(field) for field in fields)
            if not include_none and any(value is None for value in values):
                continue
            result["|".join("" if value is None else str(value) for value in values)] += 1
        return dict(sorted(result.items()))

    budget_classes = Counter()
    budget_values = {}
    for row in rows:
        source = row.get("source_dataset")
        if source == "longmemeval":
            name = "longmemeval"
        elif source == "oaks_novel":
            name = "oaks_novel"
        else:
            subtype = memora_task_subtype(row)
            name = {"content_project_proposal": "memora_project_proposal", "content_meeting_notes": "memora_meeting_notes", "content_email_writeup": "memora_email_writeup", "preference_recommendation": "memora_preference", "activity_todos": "memora_todos"}.get(subtype, "memora_unresolved")
        budget_classes[name] += 1
        budget_values[name] = resolve_max_output_tokens(row, budget_config)
    return {
        "total_rows": len(rows),
        "condition_counts": dict(Counter(row.get("condition_type") for row in rows)),
        "source_counts": dict(Counter(row.get("source_dataset") for row in rows)),
        "analysis_event_type_counts": dict(Counter(row.get("analysis_event_type") for row in rows)),
        "fault_operator_counts": dict(Counter(row.get("fault_operator") for row in rows if row.get("fault_operator"))),
        "source_event_counts": grouped(("source_dataset", "analysis_event_type")),
        "source_fault_counts": grouped(("source_dataset", "fault_operator"), include_none=False),
        "resolved_token_budget_class_counts": dict(sorted(budget_classes.items())),
        "resolved_max_output_tokens": dict(sorted(budget_values.items())),
    }


def run(*, manifest: Path = DEFAULT_MANIFEST, results_dir: Path = DEFAULT_RESULTS, model: str = "deepseek-v4-flash", dry_run: bool = False, limit: int | None = None, force: bool = False, resume: bool = False, provider_factory: Any = None, workers: int = 1, proposal_only: bool = False, budget_config_path: Path | None = None, condition_ids: set[str] | None = None) -> dict[str, Any]:
    rows = _read(manifest)
    if proposal_only and limit is not None:
        raise ValueError("--proposal-only cannot be combined with --limit")
    budget_config = load_budget_config(budget_config_path) if budget_config_path else None
    selected_ids = project_proposal_condition_ids(rows) if proposal_only else condition_ids
    selected_rows = [row for row in rows if selected_ids is None or row["condition_id"] in selected_ids]
    selected_rows = selected_rows if limit is None else selected_rows[:limit]
    if selected_ids is not None:
        assert_selective_identity(provider="deepseek", model=model, model_parameters={"temperature": 0, "thinking": "disabled", "tools": None}, manifest=manifest, selected_rows=selected_rows, expected_ids=selected_ids)
    result = run_shared(
        manifest=manifest,
        results_dir=results_dir,
        model=model,
        dry_run=dry_run,
        limit=limit,
        force=force,
        resume=resume,
        provider_factory=provider_factory,
        expected_sha256=ACCEPTED_MANIFEST_SHAS,
        expected_counts=EXPECTED_COUNTS,
        expected_base_count=EXPECTED_BASE_COUNT,
        output_filename="full_results.jsonl",
        strict_scoring=True,
        workers=workers,
        selected_condition_ids=selected_ids,
        budget_config=budget_config,
    )
    result["summary"].update(_summary_fields(selected_rows, budget_config))
    results_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        result["summary"]["status"] = "FROZEN FULL / DRY-RUN"
        result["summary"]["validation_errors"] = len(result["summary"]["scoring_preflight"]["unresolved"])
        (results_dir / "full_dry_run_requests.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in result["rows"]), encoding="utf-8")
        (results_dir / "full_dry_run_summary.json").write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        atomic_write_json(results_dir / "run_summary.json", result["summary"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--proposal-only", action="store_true", help="select all authoritative frozen Memora content_project_proposal conditions")
    parser.add_argument("--token-budget-config", type=Path, help="versioned provider-neutral token-budget configuration")
    parser.add_argument("--condition-ids-file", type=Path, help="run only frozen condition IDs listed one per non-empty line")
    args = parser.parse_args(argv)
    if args.proposal_only and args.limit is not None:
        parser.error("--proposal-only cannot be combined with --limit")
    condition_ids = None
    if args.condition_ids_file is not None:
        values = [line.strip() for line in args.condition_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not values:
            parser.error("--condition-ids-file contains no non-empty IDs")
        if len(values) != len(set(values)):
            parser.error("--condition-ids-file contains duplicate condition IDs")
        condition_ids = set(values)
    result = run(manifest=args.manifest, results_dir=args.results_dir, model=args.model, dry_run=args.dry_run, limit=args.limit, force=args.force, resume=args.resume, workers=args.workers, proposal_only=args.proposal_only, budget_config_path=args.token_budget_config, condition_ids=condition_ids)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
