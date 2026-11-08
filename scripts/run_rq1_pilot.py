#!/usr/bin/env python3
"""Run the frozen RQ1 pilot manifest through the existing batch runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_reconstruction_batch import run_batch  # noqa: E402
from scripts.build_rq1_pilot_manifest import DEFAULT_SEED, MANIFEST_VERSION, validate_manifest  # noqa: E402
from scripts.build_rq1_pilot_preflight import (  # noqa: E402
    FROZEN_PARENT_SHA256,
    PREFLIGHT_VERSION,
    validate_preflight,
)
from scripts.build_rq1_final_manifest import (  # noqa: E402
    FROZEN_FINAL_SHA256,
    MANIFEST_VERSION as FINAL_MANIFEST_VERSION,
    validate_final_manifest,
)
from scripts.build_rq1_gpt56_smoke import (  # noqa: E402
    SMOKE_MANIFEST_VERSION,
    validate_smoke_manifest,
)
from scripts.run_reconstruction_batch import _read_completed_query_ids  # noqa: E402
from scripts.run_reconstruction_smoke import ADAPTERS  # noqa: E402
from eval.reconstruction_applicability import applicable_governance_dimensions  # noqa: E402
from eval.openai_reconstruction import DEFAULT_MODEL  # noqa: E402
from eval.providers.anthropic_provider import DEFAULT_ANTHROPIC_MODEL  # noqa: E402
from eval.providers.deepseek_provider import DEFAULT_DEEPSEEK_MODEL  # noqa: E402


CONDITIONS = ("reconstruction",)
DISABLED_CONDITIONS = ("baseline", "govmem")
CONDITION_EXECUTION_PATH = "frozen_rq1_structural_grounded_reconstruction_only"


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_manifest(path: str | Path, *, require_frozen_hash: bool = True) -> tuple[dict[str, Any], str]:
    source = Path(path)
    payload = source.read_bytes()
    manifest = json.loads(payload.decode("utf-8"))
    digest = hashlib.sha256(payload).hexdigest()
    if manifest.get("manifest_version") == FINAL_MANIFEST_VERSION:
        validate_final_manifest(manifest)
        if require_frozen_hash and digest != FROZEN_FINAL_SHA256:
            raise ValueError("Final manifest does not match the frozen RQ1 final manifest SHA256")
        return manifest, digest
    if manifest.get("manifest_version") == SMOKE_MANIFEST_VERSION:
        validate_smoke_manifest(manifest)
        return manifest, digest
    validate_manifest(manifest)
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("Unsupported pilot manifest version")
    if require_frozen_hash and digest != FROZEN_PARENT_SHA256:
        raise ValueError("Pilot manifest does not match the frozen RQ1 manifest SHA256")
    return manifest, digest


def _load_preflight(path: str | Path, parent: dict[str, Any], parent_sha256: str) -> tuple[dict[str, Any], str]:
    source = Path(path)
    payload = source.read_bytes()
    selection = json.loads(payload.decode("utf-8"))
    validate_preflight(selection, parent)
    if selection.get("parent_manifest_sha256") != parent_sha256:
        raise ValueError("Preflight parent manifest SHA256 does not match the supplied frozen manifest")
    if selection.get("preflight_version") != PREFLIGHT_VERSION:
        raise ValueError("Unsupported preflight manifest version")
    return selection, hashlib.sha256(payload).hexdigest()


def _selected_records(manifest: dict[str, Any], dataset: str, limit: int | None) -> list[dict[str, Any]]:
    rows = [row for row in manifest["records"] if dataset == "all" or row["dataset"] == dataset]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        rows = rows[:limit]
    return rows


def run_pilot(
    *,
    manifest_path: str | Path,
    memora_input: str | Path | None,
    memconflict_input: str | Path | None,
    results_dir: str | Path,
    dataset: str = "all",
    model: str | None = None,
    provider: str = "openai",
    cache_dir: str | Path | None = None,
    condition: str = "reconstruction",
    limit: int | None = None,
    overwrite: bool = False,
    max_input_tokens: int | None = None,
    streaming_chunk_tokens: int | None = None,
    preflight: bool = False,
    preflight_manifest_path: str | Path | None = None,
    require_frozen_manifest: bool = False,
    dry_run: bool = False,
    batch_runner: Callable[..., dict[str, Any]] = run_batch,
) -> dict[str, Any]:
    if condition in DISABLED_CONDITIONS:
        raise ValueError(
            f"RQ1 is structural grounded reconstruction evaluation only; condition {condition!r} is disabled "
            "until a real downstream GovMem intervention is implemented"
        )
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected {CONDITIONS[0]!r}")
    if provider not in {"openai", "anthropic", "deepseek"}:
        raise ValueError(f"Unknown provider {provider!r}; expected 'openai', 'anthropic', or 'deepseek'")
    if preflight and limit is not None:
        raise ValueError("--preflight and --limit are mutually exclusive")
    if preflight and not preflight_manifest_path:
        raise ValueError("--preflight requires --preflight-manifest")
    if not preflight and preflight_manifest_path:
        raise ValueError("--preflight-manifest requires --preflight")
    manifest, manifest_sha256 = _load_manifest(manifest_path, require_frozen_hash=require_frozen_manifest)
    preflight_sha256 = None
    if manifest.get("manifest_version") == SMOKE_MANIFEST_VERSION:
        selected = list(manifest["examples"])
        if dataset != "all":
            selected = [row for row in selected if row["dataset"] == dataset]
    elif preflight:
        selection, preflight_sha256 = _load_preflight(preflight_manifest_path, manifest, manifest_sha256)
        selected = list(selection["records"])
        if dataset != "all":
            selected = [row for row in selected if row["dataset"] == dataset]
    else:
        selected = _selected_records(manifest, dataset, limit)
    if not selected:
        raise ValueError(f"No manifest records selected for dataset={dataset!r}")
    source_by_dataset = {"memora": memora_input, "memconflict": memconflict_input}
    for name in {row["dataset"] for row in selected}:
        if not source_by_dataset.get(name):
            raise ValueError(f"An input path is required for selected dataset {name}")

    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    manifest_scope = "final" if manifest.get("manifest_version") == FINAL_MANIFEST_VERSION else ("preflight" if preflight else "full")
    metadata_base = {
        "pilot_manifest_version": manifest["manifest_version"],
        "pilot_manifest_sha256": manifest_sha256,
        "pilot_seed": manifest.get("seed", DEFAULT_SEED),
        "run_condition": condition,
        "condition_execution_path": CONDITION_EXECUTION_PATH,
        "pilot_scope": manifest_scope,
        "pilot_selected_n": len(selected),
        "pilot_preflight_manifest_sha256": preflight_sha256,
        "pilot_git_commit": _git_sha(),
        "pilot_run_timestamp": timestamp,
    }
    summaries = []
    for name in ("memora", "memconflict"):
        rows = [row for row in selected if row["dataset"] == name]
        if not rows:
            continue
        output = results / f"{name}_{condition}.jsonl"
        run_metadata = {
            **metadata_base,
            "pilot_dataset_source": str(source_by_dataset[name]),
        }
        if dry_run:
            completed = _read_completed_query_ids(output)
            adapted = list(ADAPTERS[name]().load(source_by_dataset[name]))
            selected_ids = {str(row["example_id"]) for row in rows}
            selected_examples = [example for example in adapted if str(example.query.query_id) in selected_ids]
            applicable = sum(bool(applicable_governance_dimensions(example)) for example in selected_examples)
            summaries.append({
                "dataset": name, "output": str(output), "selected": len(rows),
                "applicable": applicable, "executable": applicable,
                "already_completed": len(selected_ids.intersection(completed)),
                "remaining": len(selected_ids - completed), "executed": 0,
                "provider_calls": 0,
            })
            continue
        configured_cache = Path(cache_dir) if cache_dir is not None else Path(".cache/reconstruction")
        summary = batch_runner(
            dataset=name,
            input_path=source_by_dataset[name],
            output_path=output,
            model=model,
            provider=provider,
            cache_dir=configured_cache,
            query_ids=[row["example_id"] for row in rows],
            overwrite=overwrite,
            condition=condition,
            governance_applied=False,
            run_metadata=run_metadata,
            max_input_tokens=max_input_tokens,
            streaming_chunk_tokens=streaming_chunk_tokens,
        )
        summaries.append({"dataset": name, "output": str(output), **summary})
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "condition": condition,
        "selected": len(selected),
        "summaries": summaries,
        "dry_run": dry_run,
        "provider_calls": 0 if dry_run else None,
        "model": model or {
            "openai": DEFAULT_MODEL,
            "anthropic": DEFAULT_ANTHROPIC_MODEL,
            "deepseek": DEFAULT_DEEPSEEK_MODEL,
        }[provider],
        "provider": provider,
        "cache_dir": str(cache_dir) if cache_dir is not None else ".cache/reconstruction",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--memora-input")
    parser.add_argument("--memconflict-input")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--dataset", choices=("all", "memora", "memconflict"), default="all")
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", choices=("openai", "anthropic", "deepseek"), default="openai")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-manifest", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Deterministic prefix of the selected manifest subset")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Plan final selection/resume locally without constructing a provider")
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--streaming-chunk-tokens", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_pilot(
            manifest_path=args.manifest,
            memora_input=args.memora_input,
            memconflict_input=args.memconflict_input,
            results_dir=args.results_dir,
            dataset=args.dataset,
            model=args.model,
            provider=args.provider,
            cache_dir=args.cache_dir,
            condition=args.condition,
            limit=args.limit,
            overwrite=args.overwrite,
            max_input_tokens=args.max_input_tokens,
            streaming_chunk_tokens=args.streaming_chunk_tokens,
            preflight=args.preflight,
            preflight_manifest_path=args.preflight_manifest,
            require_frozen_manifest=True,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
