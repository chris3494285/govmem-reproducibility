#!/usr/bin/env python3
"""Create public, condition-level score artifacts without provider payloads.

The transformation is deterministic and retains the score-bearing fields used
by the offline analyses. It excludes response bodies, request IDs, prompts,
endpoint URLs, and provider account metadata. The mapping records SHA-256
digests of each private source and public output; it does not contain secrets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/final_main_sanitized"
DEFAULT_MAPPING = DEFAULT_OUTPUT / "sha256_mapping.json"
SOURCES = {
    "openai": ROOT / "results/final_main_openai_gpt56_sol_final_v2/scored/full_results_scored.jsonl",
    "anthropic": ROOT / "results/final_main_anthropic_sonnet5_final/scored/full_results_scored.jsonl",
    "deepseek": ROOT / "results/final_main_deepseek_v4_pro/scored_results.jsonl",
    "gemini": ROOT / "results/final_main_gemini_3_flash_final/scored/full_results_scored.jsonl",
    "kimi": ROOT / "results/final_main_kimi_k2_6_nothinking_final/scored/full_results_scored.jsonl",
    "grok": ROOT / "results/final_main_grok_43_final/scored/full_results_scored.jsonl",
}

KEEP = (
    "condition_id", "base_id", "provider", "model", "condition_type",
    "fault_operator", "pair_id", "source_dataset", "analysis_event_type",
    "primary_event_type", "status", "parse_success", "correctness",
    "primary_correctness", "primary_score", "scoring_status", "scoring_version",
    "scoring_method", "manifest_sha256", "prompt_sha256",
    "model_visible_context_sha256", "model_parameters", "resolved_max_output_tokens",
    "attempt_count", "retry_count", "recovered_by_retry", "input_tokens",
    "output_tokens", "total_tokens", "latency_ms", "error_type", "truncated_output",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _canonical(row: dict[str, Any]) -> dict[str, Any]:
    # Keep only deterministic score and provenance fields. Do not infer or
    # normalize scores; values are copied byte-for-value at the JSON value level.
    return {key: row[key] for key in KEEP if key in row}


def _label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sanitize(*, output_dir: Path = DEFAULT_OUTPUT, mapping_path: Path = DEFAULT_MAPPING, input_dir: Path | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, Any] = {
        "schema_version": "final-main-public-score-v1",
        "api_calls_made": 0,
        "providers": {},
        "excluded_fields": [
            "provider response object", "provider output text", "request identifier", "downstream response",
            "parsed_answer", "oracle_answer", "task_oracle_answer", "task_question",
            "normalized_model_answer", "normalized_oracle_answer", "attempt_errors",
            "private endpoint and account metadata",
        ],
    }
    resolved_input_dir = input_dir.resolve() if input_dir is not None else None
    sources = {provider: (resolved_input_dir / f"merged_{provider}.jsonl" if resolved_input_dir is not None else source) for provider, source in SOURCES.items()}
    for provider, source in sources.items():
        rows = _read(source)
        public = [_canonical(row) for row in rows]
        destination = output_dir / f"{provider}.jsonl"
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in public),
            encoding="utf-8",
        )
        mapping["providers"][provider] = {
            "private_source": str(source.relative_to(ROOT)),
            "private_source_sha256": _sha256(source),
            "public_output": _label(destination),
            "public_output_sha256": _sha256(destination),
            "row_count": len(public),
            "fields": list(KEEP),
        }
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--input-dir", type=Path, help="offline merged-result directory containing merged_<provider>.jsonl files")
    args = parser.parse_args()
    result = sanitize(output_dir=args.output_dir, mapping_path=args.mapping, input_dir=args.input_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
