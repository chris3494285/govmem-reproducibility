from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_rq1_cross_provider import _final_main_summary, _final_main_summary_v3


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "results/final_main_cross_provider_summary_v1/cross_provider_summary.json"


def test_final_main_summary_has_five_formal_providers_and_frozen_kimi_values():
    summary = _final_main_summary()
    assert len(summary["models"]) == 5
    assert summary["metadata"]["models"] == [row["model"] for row in summary["models"]]
    kimi = next(row for row in summary["models"] if row["model"] == "Kimi K2.6 no-thinking")
    assert kimi["provider"] == "Kimi"
    assert kimi["clean_accuracy"] == 0.4692202462380301
    assert kimi["event_macro_corruption"] == pytest.approx(-0.336421, abs=1e-6)
    assert kimi["event_macro_recovery"] == pytest.approx(0.344081, abs=1e-6)


def test_prior_four_model_rows_are_unchanged():
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    current = _final_main_summary()["models"][:4]
    assert current == v1["models"]


def test_provider_effect_directions_and_stale_fault_order():
    rows = _final_main_summary()["models"]
    for row in rows:
        assert row["event_macro_corruption"] < 0
        assert row["event_macro_recovery"] > 0
        fault_values = {fault: row[f"{fault}_corruption"] for fault in ("conflicting", "flipped", "missing", "stale")}
        assert fault_values["stale"] == min(fault_values.values())


def test_final_main_summary_generation_is_deterministic():
    assert _final_main_summary() == _final_main_summary()


def test_final_main_v3_has_six_providers_and_grok_values():
    summary = _final_main_summary_v3()
    assert len(summary["models"]) == 6
    grok = next(row for row in summary["models"] if row["model"] == "Grok 4.3 no-reasoning")
    assert grok["provider"] == "xAI"
    assert grok["clean_accuracy"] == pytest.approx(0.305062, abs=1e-6)
    assert grok["event_macro_corruption"] == pytest.approx(-0.213344, abs=1e-6)
    assert grok["event_macro_recovery"] == pytest.approx(0.214243, abs=1e-6)
    assert summary["metadata"]["models"][-1] == "Grok 4.3 no-reasoning"


def test_final_main_v3_preserves_prior_five_and_is_deterministic():
    v2 = json.loads((ROOT / "results/final_main_cross_provider_summary_v2/cross_provider_summary.json").read_text(encoding="utf-8"))
    v3 = _final_main_summary_v3()
    assert v3["models"][:5] == v2["models"]
    assert _final_main_summary_v3() == _final_main_summary_v3()
    assert v3["metadata"]["descriptive_aggregates"]["event_macro_corruption"]["min"] < 0
    assert v3["metadata"]["descriptive_aggregates"]["event_macro_recovery"]["min"] > 0
