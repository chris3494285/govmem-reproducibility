"""Select deterministic, metadata-only Gemini engineering canary IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.final_main_token_budget import budget_class, resolve_max_output_tokens

try:
    from scripts.run_final_main_smoke_deepseek import render_prompt
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from run_final_main_smoke_deepseek import render_prompt


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
OUTPUT = ROOT / "results/final_main_gemini_stratified_canary_ids.txt"
CLASSES = (
    "longmemeval",
    "memora_email_writeup",
    "memora_meeting_notes",
    "memora_preference",
    "memora_project_proposal",
    "memora_todos",
    "oaks_novel",
)


def select(rows: list[dict]) -> list[dict]:
    """Select one stable representative per budget class plus long-output risks."""
    selected: dict[str, dict] = {}
    for name in CLASSES:
        candidates = sorted(
            (row for row in rows if budget_class(row) == name),
            key=lambda row: row["condition_id"],
        )
        if not candidates:
            raise ValueError(f"No rows for budget class {name}")
        representative = candidates[0]
        selected[representative["condition_id"]] = {
            "row": representative,
            "reason": "representative",
        }
        if name in {
            "memora_email_writeup",
            "memora_meeting_notes",
            "memora_project_proposal",
        }:
            longest = max(
                candidates,
                key=lambda row: (len(render_prompt(row)), row["condition_id"]),
            )
            selected.setdefault(
                longest["condition_id"],
                {"row": longest, "reason": "longest-risk"},
            )
            if selected[longest["condition_id"]]["reason"] == "representative":
                selected[longest["condition_id"]]["reason"] = (
                    "representative; longest-risk"
                )
    return [selected[key] for key in sorted(selected)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chosen = select(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(item["row"]["condition_id"] + "\n" for item in chosen),
        encoding="utf-8",
    )
    for item in chosen:
        row = item["row"]
        print(
            json.dumps(
                {
                    "condition_id": row["condition_id"],
                    "budget_class": budget_class(row),
                    "max_output_tokens": resolve_max_output_tokens(row),
                    "prompt_characters": len(render_prompt(row)),
                    "reason": item["reason"],
                },
                sort_keys=True,
            )
        )
    print(json.dumps({"selected_count": len(chosen), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
