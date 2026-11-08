#!/usr/bin/env python3
"""Run one frozen GovMem system-validation cohort with strict formal checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from govmem.memory_systems.base import MemorySystemAdapter  # noqa: E402
from scripts.run_system_validation_smoke import run_smoke  # noqa: E402
from scripts.validate_system_validation_frozen_manifest import read_rows, validate_manifest  # noqa: E402


DEFAULT_MANIFEST = "data/system_validation_frozen_v1.jsonl"
DEFAULT_METADATA = "data/system_validation_frozen_v1_metadata.json"


def run_formal(
    *,
    memory_system: str,
    manifest: str | Path = DEFAULT_MANIFEST,
    metadata: str | Path = DEFAULT_METADATA,
    results_dir: str | Path,
    retrieval_k: int = 1,
    adapter_factory: Callable[[dict[str, Any]], MemorySystemAdapter] | None = None,
    system_config: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if memory_system not in {"mem0", "langmem", "amem"}:
        raise ValueError("Formal validation supports only mem0, langmem, or amem")
    if retrieval_k <= 0:
        raise ValueError("--retrieval-k must be a positive integer")
    output = Path(results_dir)
    formal_json = output / "system_validation_formal.json"
    formal_summary = output / "system_validation_formal_summary.json"
    if not force and (formal_json.exists() or formal_summary.exists()):
        raise FileExistsError(f"Formal result exists; pass --force to rerun: {output}")

    # This validation is deliberately completed before adapter construction.
    frozen_metadata = json.loads(Path(metadata).read_text(encoding="utf-8"))
    validate_manifest(manifest, metadata)
    if frozen_metadata.get("status") != "FROZEN" or frozen_metadata.get("version") != "v1" or frozen_metadata.get("seed") != 42:
        raise ValueError("Formal cohort metadata is not the required frozen v1/seed 42 contract")
    rows = read_rows(manifest)
    result_metadata = {
        "experiment_status": "FROZEN / FORMAL",
        "cohort_version": "v1",
        "seed": 42,
        "manifest_sha256": frozen_metadata["manifest_sha256"],
    }
    result = run_smoke(
        manifest=manifest,
        results_dir=output,
        adapter_factory=adapter_factory,
        system_config=system_config,
        memory_system=memory_system,
        retrieval_k=retrieval_k,
        rows_override=rows,
        output_prefix="system_validation_formal",
        output_status="FROZEN / FORMAL",
        strict_retrieval=True,
        result_metadata=result_metadata,
        write_csv=False,
    )
    summary = result["summary"]
    summary["cohort_version"] = "v1"
    summary["seed"] = 42
    summary["manifest_sha256"] = frozen_metadata["manifest_sha256"]
    summary["experiment_status"] = "FROZEN / FORMAL"
    formal_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    formal_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-system", required=True, choices=("mem0", "langmem", "amem"))
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--retrieval-k", type=int, default=1)
    parser.add_argument("--mem0-config")
    parser.add_argument("--langmem-config")
    parser.add_argument("--amem-config")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.retrieval_k <= 0:
        parser.error("--retrieval-k must be a positive integer")
    config_path = {"mem0": args.mem0_config, "langmem": args.langmem_config, "amem": args.amem_config}[args.memory_system]
    config = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else {}
    result = run_formal(
        memory_system=args.memory_system,
        manifest=args.manifest,
        metadata=args.metadata,
        results_dir=args.results_dir,
        retrieval_k=args.retrieval_k,
        system_config=config,
        force=args.force,
    )
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
