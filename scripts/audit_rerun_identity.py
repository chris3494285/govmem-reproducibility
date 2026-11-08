#!/usr/bin/env python3
"""Audit historical formal model identity against proposed v3 reruns."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from eval.final_main_rerun_identity import HISTORICAL, SCORING_VERSION, V2_SHA, V3_SHA, _canonical, _rows, _static_parameters, prompt_template_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/answer_leakage_repair/rerun_identity_audit.json"
PROPOSED = {"openai": ("openai", "gpt-5.6-sol"), "anthropic": ("anthropic", "claude-sonnet-5"), "deepseek": ("deepseek", "deepseek-v4-pro"), "gemini": ("gemini", "gemini-3-flash-preview"), "kimi": ("kimi", "kimi-k2.6"), "grok": ("xai", "grok-4.3")}
PARAMETERS = {"openai": {"temperature": 0, "thinking": "disabled", "tools": None}, "anthropic": {"temperature": 0, "thinking": "disabled", "tools": None}, "deepseek": {"temperature": 0, "thinking": "disabled", "tools": None}, "gemini": {"temperature": 0, "thinking": "low", "thinking_level": "LOW", "tools": None}, "kimi": {"base_url": "https://api.moonshot.cn/v1", "temperature": 0.6, "thinking": "disabled", "top_p": 0.95, "tools": None}, "grok": {"base_url": "https://api.x.ai/v1", "reasoning_effort": "none", "temperature": None, "thinking": "disabled", "top_p": None, "tools": None}}

def main() -> int:
    report = {"schema_version": "final-main-rerun-identity-audit-v1", "historical_manifest_sha256": V2_SHA, "v3_manifest_sha256": V3_SHA, "prompt_template_hash": prompt_template_hash(), "system_instruction_hash": None, "scoring_version": SCORING_VERSION, "providers": {}, "all_identities_pass": True}
    for name, (provider, proposed_model) in PROPOSED.items():
        rows = _rows(HISTORICAL[provider])
        models = sorted({str(row.get("model")) for row in rows}); providers = sorted({str(row.get("provider")) for row in rows})
        hist_params = sorted({_canonical(_static_parameters(row.get("model_parameters"))) for row in rows})
        proposed_params = _canonical(_static_parameters(PARAMETERS[name]))
        match = models == [proposed_model] and providers == [provider] and hist_params == [proposed_params] and {row.get("scoring_version") for row in rows if row.get("scoring_version") is not None} == {SCORING_VERSION}
        report["providers"][name] = {"provider": provider, "historical_model_id": models, "proposed_model_id": proposed_model, "model_match_exact": models == [proposed_model], "historical_provider_values": providers, "provider_match_exact": providers == [provider], "api_endpoint": PARAMETERS[name].get("base_url", {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com/v1", "deepseek": "https://api.deepseek.com", "gemini": "https://generativelanguage.googleapis.com"}.get(name)), "prompt_template_hash": report["prompt_template_hash"], "system_instruction_hash": None, "historical_model_parameters": hist_params, "proposed_model_parameters": PARAMETERS[name], "model_parameters_match": hist_params == [proposed_params], "token_budget_policy": "provider-specific versioned configuration; LongMemEval repair ceiling is recorded in rerun_cost_estimate.json", "scoring_version": sorted({str(row.get("scoring_version")) for row in rows}), "retry_policy": "shared runner, up to 3 attempts, retryable failures only", "original_manifest_sha256": sorted({str(row.get("manifest_sha256")) for row in rows}), "v3_manifest_sha256": V3_SHA, "selective_result_merge_valid": match}
        report["all_identities_pass"] &= match
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_identities_pass": report["all_identities_pass"], "providers": {k: v["selective_result_merge_valid"] for k, v in report["providers"].items()}}, sort_keys=True))
    return 0 if report["all_identities_pass"] else 1
if __name__ == "__main__": raise SystemExit(main())
