"""Write failed condition IDs from a final-main result artifact, offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def failed_condition_ids(results: Path) -> list[str]:
    """Return sorted IDs for every non-success result, rejecting bad identity."""
    seen: set[str] = set()
    failed: list[str] = []
    for line in results.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        condition_id = row.get("condition_id")
        if not condition_id or condition_id in seen:
            raise ValueError("Result artifact has a missing or duplicate condition_id")
        seen.add(condition_id)
        if row.get("status") != "success":
            failed.append(condition_id)
    return sorted(failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ids = failed_condition_ids(args.results)
    args.output.write_text("".join(item + "\n" for item in ids), encoding="utf-8")
    print(json.dumps({"failed_rows": len(ids), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
