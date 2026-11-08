from __future__ import annotations

from eval.reconstruction_provider import ProviderResponse, ProviderTelemetry
from scripts.run_governance_fault_rq1 import load_manifest, run_manifest


class RecordingDownstream:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.prompts: list[str] = []

    def request_downstream(self, prompt: str) -> ProviderResponse:
        self.prompts.append(prompt)
        return ProviderResponse(
            {"answer": "ok"},
            ProviderTelemetry(self.provider, self.model, 10, 4, 14, 1.0, "request", "end_turn"),
        )


def test_anthropic_preflight_selects_four_conditions_without_provider_calls(tmp_path):
    calls = []
    results = run_manifest(
        manifest="data/governance_fault_rq1_restrictive_frozen.jsonl",
        output=tmp_path / "preflight.jsonl",
        provider="anthropic",
        model="claude-sonnet-5",
        temperature=0,
        limit=2,
        dry_run=True,
        overwrite=True,
        provider_factory=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert len(results) == 4
    assert len({row["example_id"] for row in results}) == 2
    assert calls == []
    assert all(row["requested_temperature"] == 0 for row in results)
    assert all(row["effective_temperature"] is None for row in results)
    assert all(row["requested_reasoning_effort"] is None for row in results)
    assert all(row["effective_reasoning_effort"] is None for row in results)


def test_anthropic_runner_selection_prompt_equality_and_telemetry(tmp_path):
    providers = {
        name: RecordingDownstream(name, "test-model")
        for name in ("anthropic", "deepseek", "openai")
    }
    for name, provider in providers.items():
        run_manifest(
            manifest="data/governance_fault_rq1_restrictive_frozen.jsonl",
            output=tmp_path / f"{name}.jsonl",
            provider=name,
            model="test-model",
            temperature=0,
            limit=1,
            provider_factory=lambda model, _provider=provider, **kwargs: _provider,
            overwrite=True,
        )
    assert providers["anthropic"].prompts == providers["deepseek"].prompts
    assert providers["anthropic"].prompts == providers["openai"].prompts


def test_anthropic_reuse_is_within_base_and_not_across_examples(tmp_path):
    rows = load_manifest("data/governance_fault_rq1_pilot_frozen.jsonl")
    provider = RecordingDownstream("anthropic", "claude-sonnet-5")
    results = run_manifest(
        manifest="data/governance_fault_rq1_pilot_frozen.jsonl",
        output=tmp_path / "reuse.jsonl",
        provider="anthropic",
        model="claude-sonnet-5",
        temperature=0,
        limit=2,
        provider_factory=lambda model, _provider=provider, **kwargs: _provider,
        overwrite=True,
    )
    expected_calls = sum(
        len({row["prompt_sha256"] for row in results if row["example_id"] == example_id})
        for example_id in sorted({row["example_id"] for row in rows})[:2]
    )
    assert len(provider.prompts) == expected_calls
    assert any(row["response_reused"] for row in results)
    for row in results:
        if row["response_reused"]:
            source = next(item for item in results if item["run_id"] == row["reused_from_run_id"])
            assert source["example_id"] == row["example_id"]
    assert all(row["requested_reasoning_effort"] is None for row in results)
