from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.final_main_token_budget import KIMI_BUDGETS, KIMI_REPAIR_V1_BUDGETS, KIMI_V2_BUDGETS, KIMI_V3_BUDGETS, KIMI_V4_BUDGETS, budget_class, load_kimi_budget_config, load_kimi_repair_v1_budget_config, load_kimi_v2_budget_config, load_kimi_v3_budget_config, load_kimi_v4_budget_config
from eval.providers.kimi_provider import DEFAULT_KIMI_BASE_URL, DEFAULT_KIMI_MODEL, KimiDownstreamProvider
from scripts.run_final_main_kimi import run

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/final_main_governance_condition_manifest_v2.jsonl"
CONFIG = ROOT / "configs/final_main_token_budget_kimi_v1.json"
SHA = "58cf32242254b64ad7f5e3e56818bcc5c006625321f5ed2a86024cca4d7492f8"


def test_kimi_full_dry_run_is_offline(tmp_path):
    calls = []
    result = run(manifest=MANIFEST, results_dir=tmp_path, model=DEFAULT_KIMI_MODEL, dry_run=True, provider_factory=lambda *a, **k: calls.append(1))
    summary = result["summary"]
    assert summary["requested_rows"] == 4891
    assert summary["api_calls_made"] == 0
    assert summary["manifest_sha256"] == SHA
    assert summary["scoring_preflight"]["total_supported"] == 4891
    assert summary["resolved_token_budget_class_counts"] == {"longmemeval": 1111, "memora_email_writeup": 238, "memora_meeting_notes": 245, "memora_preference": 1400, "memora_project_proposal": 581, "memora_todos": 196, "oaks_novel": 1120}
    assert calls == []


def test_kimi_budget_config_is_strict():
    assert load_kimi_budget_config(CONFIG)["budgets"] == KIMI_BUDGETS
    assert DEFAULT_KIMI_BASE_URL == "https://api.moonshot.ai/v1"


def test_kimi_v2_budget_config_has_exact_canary_repair_ceilings():
    assert load_kimi_v2_budget_config(ROOT / "configs/final_main_token_budget_kimi_v2.json")["budgets"] == KIMI_V2_BUDGETS


def test_kimi_request_uses_json_contract_and_no_tools():
    provider = KimiDownstreamProvider(client=object(), model=DEFAULT_KIMI_MODEL)
    payload = provider.request_payload("question", max_tokens=1536)
    assert payload["model"] == DEFAULT_KIMI_MODEL
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95
    assert payload["max_tokens"] == 1536
    assert payload["response_format"] == {"type": "json_object"}
    assert "tools" not in payload
    assert "functions" not in payload
    assert "extra_body" not in payload


def test_kimi_disabled_thinking_uses_documented_extra_body_field():
    provider = KimiDownstreamProvider(client=object(), model=DEFAULT_KIMI_MODEL, thinking="disabled")
    payload = provider.request_payload("question", max_tokens=1536)
    assert payload["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "tools" not in payload
    assert "functions" not in payload


def test_kimi_thinking_modes_resolve_provider_temperature():
    assert KimiDownstreamProvider(client=object(), thinking="provider_default").temperature == 1.0
    assert KimiDownstreamProvider(client=object(), thinking="disabled").temperature == 0.6
    assert KimiDownstreamProvider(client=object(), thinking="disabled").request_payload("q")["temperature"] == 0.6


def test_kimi_model_parameters_are_deterministic(tmp_path):
    from scripts.run_final_main_kimi import run
    result = run(manifest=MANIFEST, results_dir=tmp_path, dry_run=True)
    assert result["summary"]["model_parameters"] == {"temperature": 1.0, "top_p": 0.95, "tools": None, "thinking": "provider_default"}


def test_kimi_disabled_thinking_is_a_uniform_dry_run_setting(tmp_path):
    result = run(manifest=MANIFEST, results_dir=tmp_path, dry_run=True, thinking="disabled")
    assert result["summary"]["api_calls_made"] == 0
    assert result["summary"]["model_parameters"]["thinking"] == "disabled"
    assert result["summary"]["model_parameters"]["temperature"] == 0.6
    assert all(row["model_parameters"]["thinking"] == "disabled" for row in result["rows"])
    assert all(row["model_parameters"]["temperature"] == 0.6 for row in result["rows"])


def test_kimi_provider_default_temperature_remains_one(tmp_path):
    result = run(manifest=MANIFEST, results_dir=tmp_path, dry_run=True)
    assert result["summary"]["model_parameters"]["thinking"] == "provider_default"
    assert result["summary"]["model_parameters"]["temperature"] == 1.0
    assert {row["model_parameters"]["temperature"] for row in result["rows"]} == {1.0}


def test_kimi_mocked_response_parses_usage_and_finish_reason():
    response = SimpleNamespace(id="kimi-1", choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"ok"}'), finish_reason="stop")], usage=SimpleNamespace(prompt_tokens=17, completion_tokens=64, total_tokens=81, completion_tokens_details=SimpleNamespace(reasoning_tokens=63)))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    result = KimiDownstreamProvider(client=client).request_downstream("p", max_tokens=256)
    assert result.parsed == {"answer": "ok"}
    assert result.telemetry.input_tokens == 17
    assert result.telemetry.output_tokens == 64
    assert result.telemetry.total_tokens == 81
    assert result.telemetry.thoughts_tokens == 63
    assert result.telemetry.provider == "kimi"


def test_kimi_malformed_and_truncated_responses_are_classified():
    malformed = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"), finish_reason="stop")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: malformed)))
    with pytest.raises(Exception) as exc_info:
        KimiDownstreamProvider(client=client).request_downstream("p")
    assert getattr(exc_info.value, "details", {}).get("error_category") == "structural_invalid"

    truncated = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="length")], usage=SimpleNamespace(prompt_tokens=17, completion_tokens=64, total_tokens=81, completion_tokens_details=SimpleNamespace(reasoning_tokens=63)))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: truncated)))
    with pytest.raises(Exception) as exc_info:
        KimiDownstreamProvider(client=client).request_downstream("p")
    assert getattr(exc_info.value, "details", {}).get("error_category") == "token_limit_truncation"
    assert "Qwen" not in str(exc_info.value)


def _frozen_condition_id(*, source_dataset, condition_type, fault_operator, budget_class_name):
    """Select a condition id from the local frozen manifest instead of embedding one.

    The manifest is a licensed local input; the concrete condition ids are
    source-derived benchmark identifiers, so the public test source never
    hard-codes them. This test is skipped by the public release scope when the
    manifest is absent.
    """
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("source_dataset") != source_dataset:
            continue
        if row.get("condition_type") != condition_type:
            continue
        if fault_operator is not None and row.get("fault_operator") != fault_operator:
            continue
        if budget_class_name is not None and budget_class(row) != budget_class_name:
            continue
        return row["condition_id"]
    raise AssertionError(f"no {source_dataset}/{condition_type}/{fault_operator} condition in the frozen manifest")


def test_kimi_v2_targeted_dry_run_is_offline_and_resolves_budget(tmp_path):
    cid = _frozen_condition_id(
        source_dataset="memora",
        condition_type="corrupted",
        fault_operator="missing",
        budget_class_name="memora_meeting_notes",
    )
    result = run(manifest=MANIFEST, results_dir=tmp_path, model=DEFAULT_KIMI_MODEL, token_budget_config=ROOT / "configs/final_main_token_budget_kimi_v2.json", condition_ids={cid}, dry_run=True)
    assert result["summary"]["requested_rows"] == 1
    assert result["summary"]["api_calls_made"] == 0
    assert result["rows"][0]["model_parameters"]["max_output_tokens"] == 2048


def test_kimi_v3_budget_config_is_strict_and_v2_remains_unchanged(tmp_path):
    assert load_kimi_v3_budget_config(ROOT / "configs/final_main_token_budget_kimi_v3.json")["budgets"] == KIMI_V3_BUDGETS
    assert load_kimi_v2_budget_config(ROOT / "configs/final_main_token_budget_kimi_v2.json")["budgets"]["oaks_novel"] == 2048
    altered = json.loads((ROOT / "configs/final_main_token_budget_kimi_v3.json").read_text())
    altered["budgets"]["oaks_novel"] = 2048
    path = tmp_path / "bad-kimi-v3.json"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="approved measurement ceilings"):
        load_kimi_v3_budget_config(path)


def test_kimi_runner_rejects_unknown_budget_schema(tmp_path):
    config = json.loads((ROOT / "configs/final_main_token_budget_kimi_v3.json").read_text())
    config["schema_version"] = "unknown"
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unsupported Kimi token-budget schema"):
        run(manifest=MANIFEST, results_dir=tmp_path / "results", token_budget_config=path, dry_run=True)


def test_kimi_v4_budget_config_is_strict(tmp_path):
    path = ROOT / "configs/final_main_token_budget_kimi_v4.json"
    assert load_kimi_v4_budget_config(path)["budgets"] == KIMI_V4_BUDGETS
    altered = json.loads(path.read_text())
    altered["budgets"]["oaks_novel"] = 4096
    bad = tmp_path / "bad-kimi-v4.json"
    bad.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="approved measurement ceilings"):
        load_kimi_v4_budget_config(bad)


def test_kimi_repair_v1_budget_config_is_strict_and_formal_v4_unchanged(tmp_path):
    path = ROOT / "configs/final_main_token_budget_kimi_repair_v1.json"
    assert load_kimi_repair_v1_budget_config(path)["budgets"] == KIMI_REPAIR_V1_BUDGETS
    assert load_kimi_v4_budget_config(ROOT / "configs/final_main_token_budget_kimi_v4.json")["budgets"] == KIMI_V4_BUDGETS
    altered = json.loads(path.read_text())
    altered["budgets"]["memora_preference"] = 3072
    bad = tmp_path / "bad-kimi-repair.json"
    bad.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="approved measurement ceilings"):
        load_kimi_repair_v1_budget_config(bad)


def test_kimi_malformed_error_is_provider_specific():
    malformed = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"), finish_reason="stop")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: malformed)))
    with pytest.raises(Exception) as exc_info:
        KimiDownstreamProvider(client=client).request_downstream("p")
    assert "Qwen" not in str(exc_info.value)
    assert "Kimi" in str(exc_info.value)


def test_kimi_repair_dry_run_is_offline(tmp_path):
    result = run(manifest=MANIFEST, results_dir=tmp_path, token_budget_config=ROOT / "configs/final_main_token_budget_kimi_repair_v1.json", dry_run=True, thinking="disabled")
    assert result["summary"]["api_calls_made"] == 0
    assert result["summary"]["model_parameters"]["temperature"] == 0.6
