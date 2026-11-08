import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_formal_provider_identities_pass():
    report = json.loads((ROOT / "results/answer_leakage_repair/rerun_identity_audit.json").read_text())
    assert report["all_identities_pass"] is True
    assert report["providers"]["deepseek"]["historical_model_id"] == ["deepseek-v4-pro"]
    assert all(item["selective_result_merge_valid"] for item in report["providers"].values())


def test_rerun_cost_report_uses_deepseek_pro():
    report = json.loads((ROOT / "results/answer_leakage_repair/rerun_cost_estimate.json").read_text())
    assert report["per_provider"]["deepseek"]["model"] == "deepseek-v4-pro"
    assert "deepseek-v4-pro" in report["per_provider"]["deepseek"]["live_command"]
