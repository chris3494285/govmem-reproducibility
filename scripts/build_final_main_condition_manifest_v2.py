"""Reproducible production entry point for the frozen condition manifest v2.

The implementation remains in the previously validated governance-prefixed
builder.  This entry point provides the stable production filename requested
for the final main-condition artifact without introducing a second condition
construction implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # Works both as ``python scripts/...`` and as an imported module.
    from scripts.build_final_main_governance_condition_manifest_v2 import (
        FREEZE as DEFAULT_FREEZE,
        OUTPUT as DEFAULT_OUTPUT,
        SUMMARY as DEFAULT_SUMMARY,
        build as _build,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI use
    from build_final_main_governance_condition_manifest_v2 import (
        FREEZE as DEFAULT_FREEZE,
        OUTPUT as DEFAULT_OUTPUT,
        SUMMARY as DEFAULT_SUMMARY,
        build as _build,
    )


def build(
    *,
    output: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    freeze_path: Path = DEFAULT_FREEZE,
    force: bool = False,
):
    """Build v2 using the single validated implementation."""
    return _build(
        output=output,
        summary_path=summary_path,
        freeze_path=freeze_path,
        force=force,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the frozen GovMem main condition manifest v2"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    build(
        output=args.output,
        summary_path=args.summary,
        freeze_path=args.freeze,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
