"""Score completed final-main results without invoking a provider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.final_main_result_join import join_results_to_manifest, load_condition_manifest
from eval.system_validation_scoring import score_task_answer_detailed


SUPPORTED_SCORING_METHODS = {
    "normalized_exact_or_native_alternative",
    "normalized_native_option_match",
    "required_native_fact_presence",
    "memora_preference_constraints_v1",
    "memora_activity_todos_v1",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def score_results(
    results_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    verify_manifest_hash: bool = False,
) -> list[dict[str, Any]]:
    # The CLI is strict for production frozen artifacts.  The opt-out keeps
    # this helper usable by small synthetic unit-test fixtures while still
    # applying the same authoritative in-memory join and ID validation.
    manifest, manifest_sha = load_condition_manifest(
        manifest_path, verify_hash=verify_manifest_hash
    )
    results = _read_jsonl(results_path)
    seen: set[str] = set()
    scored: list[dict[str, Any]] = []
    joined = join_results_to_manifest(results, manifest)
    for item in joined:
        raw = item["result"]
        manifest_row = item["manifest"]
        condition_id = raw.get("condition_id")
        if condition_id in seen:
            raise ValueError(f"Duplicate result condition_id: {condition_id}")
        seen.add(condition_id)
        if raw.get("manifest_sha256") not in (None, manifest_sha):
            raise ValueError(f"Manifest SHA256 mismatch for {condition_id}")
        answer = raw.get("parsed_answer")
        if answer is None:
            answer = raw.get("downstream_response")
        output = dict(raw)
        output.update({
            "source_dataset": manifest_row["source_dataset"],
            "base_id": manifest_row.get("base_id"),
            "analysis_event_type": manifest_row.get("analysis_event_type"),
            "primary_event_type": manifest_row.get("primary_event_type"),
            "condition_type": manifest_row.get("condition_type"),
            "fault_operator": manifest_row.get("fault_operator"),
            "task_answer_type": manifest_row.get("task_answer_type"),
            "manifest_sha256": manifest_sha,
        })
        # error_type/error_message can describe an earlier retry attempt.  A
        # recovered row is eligible when its final state is successful and a
        # final answer is available; retry telemetry remains untouched.
        final_failure = (
            raw.get("status") == "error"
            or raw.get("parse_success") is False
            or not isinstance(answer, str)
            or not answer.strip()
        )
        if final_failure:
            detail = {"reason": "provider_or_parse_error"}
            output.update({"correctness": None, "primary_correctness": False, "primary_score": 0, "scoring_status": "unscored", "scoring_method": None, "scoring_version": "final-main-source-aware-v2", "normalized_model_answer": None, "normalized_oracle_answer": None, "scoring_details": detail, "deterministic_score": None, "native_validation_score": None, "adjudication_required": False})
        else:
            details = score_task_answer_detailed(manifest_row, answer)
            if details.get("scoring_method") in SUPPORTED_SCORING_METHODS and details.get("correctness") is None and details.get("scoring_status") != "adjudication_required":
                raise ValueError(
                    f"Successful supported row remained unscored: {condition_id} "
                    f"({details.get('scoring_method')})"
                )
            output.update(details)
            output["primary_correctness"] = details["correctness"]
            output["primary_score"] = int(details["correctness"]) if isinstance(details["correctness"], bool) else None
        scored.append(output)
    scored.sort(key=lambda row: row["condition_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scored), encoding="utf-8")
    return scored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = score_results(
        args.results, args.manifest, args.output, verify_manifest_hash=True
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output), "api_calls_made": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
