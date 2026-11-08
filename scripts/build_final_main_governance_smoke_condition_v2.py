"""Filter condition-v2 rows for the already-frozen smoke base IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE_BASE = ROOT / "data/final_main_governance_smoke_base_manifest_v1.jsonl"
CONDITION_V2 = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
OUTPUT = ROOT / "data/final_main_governance_smoke_condition_manifest_v2.jsonl"
EXPECTED_V2_SHA = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output: Path = OUTPUT, force: bool = False) -> str:
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite smoke condition v2: {output}")
    if not CONDITION_V2.is_file() or not SMOKE_BASE.is_file():
        raise FileNotFoundError("Smoke base or condition-v2 manifest is missing")
    if _sha(CONDITION_V2) != EXPECTED_V2_SHA:
        raise ValueError("Condition-v2 manifest hash mismatch")
    base_ids = {json.loads(line)["base_id"] for line in SMOKE_BASE.read_text().splitlines() if line.strip()}
    if len(base_ids) != 24:
        raise ValueError(f"Expected 24 frozen smoke base IDs, found {len(base_ids)}")
    rows = [json.loads(line) for line in CONDITION_V2.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = sorted((row for row in rows if row["base_id"] in base_ids), key=lambda row: row["condition_id"])
    if len(selected) != 160:
        raise ValueError(f"Expected 160 smoke condition-v2 rows, found {len(selected)}")
    if sum(row["condition_type"] == "clean" for row in selected) != 24 or sum(row["condition_type"] == "corrupted" for row in selected) != 68 or sum(row["condition_type"] == "enforced" for row in selected) != 68:
        raise ValueError("Smoke condition-v2 composition mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    digest = _sha(output)
    print(f"Smoke condition v2: bases=24 clean=24 corrupted=68 enforced=68 total=160")
    print(f"Smoke condition v2 SHA256: {digest}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the task-complete smoke condition-v2 subset")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.output, args.force)


if __name__ == "__main__":
    main()
