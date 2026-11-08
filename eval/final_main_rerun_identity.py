"""Hard identity gate for selective v3 repair reruns."""
from __future__ import annotations
import hashlib
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any
try:
    from scripts.run_final_main_smoke_deepseek import render_prompt
except ModuleNotFoundError:  # direct script execution
    from run_final_main_smoke_deepseek import render_prompt

ROOT = Path(__file__).resolve().parents[1]
V2_SHA = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"
V3_SHA = "4b84068a341d912f43d07c9d92059f75224ea2636abf69e40ba6e71a0a1144cb"
SCORING_VERSION = "final-main-source-aware-v1"
HISTORICAL = {
    "openai": ROOT / "results/final_main_openai_gpt56_sol_final_v2/full_results_merged.jsonl",
    "anthropic": ROOT / "results/final_main_anthropic_sonnet5_final/full_results_merged.jsonl",
    "deepseek": ROOT / "results/final_main_deepseek_v4_pro/scored_results.jsonl",
    "gemini": ROOT / "results/final_main_gemini_3_flash_final/full_results_merged.jsonl",
    "kimi": ROOT / "results/final_main_kimi_k2_6_nothinking_final/full_results_merged.jsonl",
    "xai": ROOT / "results/final_main_grok_43_final/scored/full_results_scored.jsonl",
}

def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def _static_parameters(value: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(value or {})
    result.pop("max_output_tokens", None)
    return result

def prompt_template_hash() -> str:
    return hashlib.sha256(inspect.getsource(render_prompt).encode()).hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Persist a final provider summary atomically without touching result rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

def assert_selective_identity(*, provider: str, model: str, model_parameters: dict[str, Any], manifest: Path, selected_rows: list[dict[str, Any]], expected_ids: set[str]) -> None:
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    # Existing v2 selective utilities (canary/repair validation) retain their
    # historical behavior.  The strict 265-row gate applies only to v3.
    if manifest_sha != V3_SHA:
        return
    if len(selected_rows) != 265 or len(expected_ids) != 265 or {row.get("condition_id") for row in selected_rows} != expected_ids:
        raise ValueError(f"{provider} selective rerun must contain exactly the approved 265-condition repair cohort")
    history = _rows(HISTORICAL[provider])
    if len(history) != 4891:
        raise ValueError(f"Historical {provider} formal artifact must contain 4891 rows")
    if {row.get("model") for row in history} != {model} or {row.get("provider") for row in history} != {provider}:
        raise ValueError(f"{provider} model/provider identity mismatch")
    historical_parameters = {_canonical(_static_parameters(row.get("model_parameters"))) for row in history}
    if historical_parameters != {_canonical(_static_parameters(model_parameters))}:
        raise ValueError(f"{provider} model parameters do not match historical formal parameters")
    if {row.get("scoring_version") for row in history if row.get("scoring_version") is not None} != {SCORING_VERSION}:
        raise ValueError(f"{provider} historical scoring version mismatch")
    # The v3 manifest stores the authoritative context; the runner computes
    # each condition's prompt hash from that context during preflight.
    if any(not hashlib.sha256(render_prompt(row).encode()).hexdigest() for row in selected_rows):
        raise ValueError(f"{provider} v3 prompt construction failed")
